import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

from dataset_config import DEFAULT_DATASET_ROOT, PROJECT_ROOT, SPLITS, load_dataset_metadata
from duplicate_utils import IMAGE_EXTENSIONS


DEFAULT_UNLABELED_ROOT = (
    PROJECT_ROOT.parent / "resurse project 2" / "unlabbeled images" / "images"
)
DEFAULT_TEACHER_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "pothole-yolo11n-40-epoch-20261115"
    / "weights"
    / "best.pt"
)
DEFAULT_OVERLAP_REPORT = PROJECT_ROOT / "reports" / "unlabeled_overlap_candidates.csv"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "pseudo_labels" / "yolo11n-40epoch-conf060"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conservative YOLO pseudo-labels from unlabeled images."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_UNLABELED_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_TEACHER_WEIGHTS)
    parser.add_argument("--overlap-report", type=Path, default=DEFAULT_OVERLAP_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--exclude-splits",
        nargs="+",
        choices=SPLITS,
        default=["valid", "test"],
        help="Labeled splits whose perceptual-hash candidates must be excluded.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.60,
        help="Minimum detection confidence; selected using validation precision/recall.",
    )
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--review-images", type=int, default=36)
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def ensure_within(path: Path, root: Path, description: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{description} is outside the expected folder: {path}") from error


def read_overlap_exclusions(
    report_path: Path,
    images_root: Path,
    dataset_root: Path,
    excluded_splits: set[str],
) -> dict[str, set[str]]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Overlap report not found: {report_path}")

    excluded: dict[str, set[str]] = {}
    with report_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            split = row.get("labeled_split", "")
            if split not in SPLITS:
                raise ValueError(f"Unexpected labeled split in overlap report: {split!r}")
            if split not in excluded_splits:
                continue
            image = Path(row["unlabeled_file"]).expanduser().resolve()
            labeled_image = Path(row["labeled_file"]).expanduser().resolve()
            ensure_within(image, images_root, "Candidate unlabeled image")
            ensure_within(
                labeled_image,
                dataset_root / split / "images",
                "Candidate labeled image",
            )
            if not image.is_file() or not labeled_image.is_file():
                raise FileNotFoundError(
                    f"Overlap report contains a missing image: {image} or {labeled_image}"
                )
            excluded.setdefault(path_key(image), set()).add(split)
    return excluded


def class_names(model: YOLO) -> list[str]:
    names: Any = model.names
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise TypeError(f"Unsupported model class-name representation: {type(names).__name__}")


def create_review_sheets(samples: list[tuple[float, str, Image.Image]], root: Path) -> list[str]:
    sheet_names: list[str] = []
    columns = 4
    cell_width = 250
    cell_height = 220
    page_size = 24
    for page_index, start in enumerate(range(0, len(samples), page_size), start=1):
        page = samples[start:start + page_size]
        rows = math.ceil(len(page) / columns)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (minimum_confidence, filename, image) in enumerate(page):
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height
            draw.text(
                (x + 5, y + 4),
                f"{filename} | min conf {minimum_confidence:.3f}",
                fill="black",
            )
            sheet.paste(image, (x + 5, y + 24))
        name = f"pseudo_label_review_{page_index:02d}.jpg"
        sheet.save(root / name, quality=90)
        sheet_names.append(name)
    return sheet_names


def main() -> None:
    args = parse_args()
    if not 0 <= args.conf < 1:
        raise SystemExit("--conf must be at least 0 and less than 1.")
    if not 0 < args.iou <= 1:
        raise SystemExit("--iou must be greater than 0 and at most 1.")
    if args.imgsz <= 0 or args.batch <= 0 or args.review_images < 0:
        raise SystemExit("--imgsz and --batch must be positive; --review-images cannot be negative.")

    images_root = args.images.expanduser().resolve()
    dataset_root, expected_names = load_dataset_metadata(args.dataset_root)
    weights = args.weights.expanduser().resolve()
    overlap_report = args.overlap_report.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not images_root.is_dir():
        raise NotADirectoryError(f"Unlabeled image folder not found: {images_root}")
    if not weights.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {weights}")
    if output_root.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output_root}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu or repair the CUDA environment.")

    input_images = image_files(images_root)
    if not input_images:
        raise FileNotFoundError(f"No supported images found under: {images_root}")
    for image_path in input_images:
        with Image.open(image_path) as image:
            image.verify()

    excluded_splits = set(args.exclude_splits)
    exclusions = read_overlap_exclusions(
        overlap_report,
        images_root,
        dataset_root,
        excluded_splits,
    )
    input_by_key = {path_key(path): path for path in input_images}
    unknown_exclusions = set(exclusions) - set(input_by_key)
    if unknown_exclusions:
        raise ValueError(
            f"Overlap report references {len(unknown_exclusions)} images not found in the input folder."
        )

    eligible_images = [
        path for path in input_images if path_key(path) not in exclusions
    ]
    if not eligible_images:
        raise ValueError("No eligible images remain after applying overlap exclusions.")

    model = YOLO(str(weights))
    actual_names = class_names(model)
    if actual_names != expected_names:
        raise ValueError(
            f"Teacher class names {actual_names} do not match dataset names {expected_names}."
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any]
    manifest_rows: list[dict[str, str]] = []
    box_confidences: list[float] = []
    pseudo_box_count = 0
    pseudo_image_count = 0
    no_detection_count = 0
    processed: set[str] = set()
    review_samples: list[tuple[float, str, Image.Image]] = []

    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}-",
    ) as temporary_directory:
        staging_root = Path(temporary_directory)
        staging_images = staging_root / "images"
        staging_labels = staging_root / "labels"
        staging_images.mkdir()
        staging_labels.mkdir()

        for image_path in input_images:
            matches = sorted(exclusions.get(path_key(image_path), set()))
            if matches:
                manifest_rows.append(
                    {
                        "source_image": str(image_path),
                        "status": "excluded_" + "_and_".join(matches) + "_candidate",
                        "matched_splits": ";".join(matches),
                        "pseudo_image": "",
                        "pseudo_label": "",
                        "detection_count": "0",
                        "minimum_confidence": "",
                        "mean_confidence": "",
                        "maximum_confidence": "",
                    }
                )

        prediction_stream = model.predict(
            source=[str(path) for path in eligible_images],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            quantize=16 if args.device != "cpu" else None,
            stream=True,
            verbose=False,
            save=False,
        )
        for result in prediction_stream:
            source_path = Path(result.path).resolve()
            source_key = path_key(source_path)
            if source_key not in input_by_key or source_key in exclusions:
                raise RuntimeError(f"Prediction returned an unexpected input image: {result.path}")
            if source_key in processed:
                raise RuntimeError(f"Prediction returned the same input more than once: {result.path}")
            processed.add(source_key)

            boxes = result.boxes
            normalized_boxes = boxes.xywhn.cpu().tolist() if boxes is not None else []
            confidences = boxes.conf.cpu().tolist() if boxes is not None else []
            class_ids = boxes.cls.cpu().tolist() if boxes is not None else []
            label_lines: list[str] = []
            for normalized_box, confidence, class_id_value in zip(
                normalized_boxes,
                confidences,
                class_ids,
                strict=True,
            ):
                class_id = int(class_id_value)
                if class_id < 0 or class_id >= len(expected_names):
                    raise ValueError(f"Teacher returned invalid class id {class_id}.")
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError(f"Teacher returned invalid confidence {confidence}.")
                if len(normalized_box) != 4 or any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in normalized_box
                ):
                    raise ValueError(f"Teacher returned invalid normalized box: {normalized_box}")
                center_x, center_y, width, height = normalized_box
                if width <= 0 or height <= 0:
                    raise ValueError(f"Teacher returned a zero-area box: {normalized_box}")
                label_lines.append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                )
                box_confidences.append(float(confidence))

            if not label_lines:
                no_detection_count += 1
                manifest_rows.append(
                    {
                        "source_image": str(source_path),
                        "status": "no_detection_at_threshold",
                        "matched_splits": "",
                        "pseudo_image": "",
                        "pseudo_label": "",
                        "detection_count": "0",
                        "minimum_confidence": "",
                        "mean_confidence": "",
                        "maximum_confidence": "",
                    }
                )
                continue

            relative_path = source_path.relative_to(images_root)
            pseudo_image = output_root / "images" / relative_path
            pseudo_label = output_root / "labels" / relative_path.with_suffix(".txt")
            staged_image = staging_images / relative_path
            staged_label = staging_labels / relative_path.with_suffix(".txt")
            staged_image.parent.mkdir(parents=True, exist_ok=True)
            staged_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, staged_image)
            staged_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            current_confidences = [float(value) for value in confidences]
            minimum_confidence = min(current_confidences)
            mean_confidence = sum(current_confidences) / len(current_confidences)
            maximum_confidence = max(current_confidences)
            pseudo_image_count += 1
            pseudo_box_count += len(label_lines)
            manifest_rows.append(
                {
                    "source_image": str(source_path),
                    "status": "pseudo_labeled",
                    "matched_splits": "",
                    "pseudo_image": str(pseudo_image),
                    "pseudo_label": str(pseudo_label),
                    "detection_count": str(len(label_lines)),
                    "minimum_confidence": f"{minimum_confidence:.6f}",
                    "mean_confidence": f"{mean_confidence:.6f}",
                    "maximum_confidence": f"{maximum_confidence:.6f}",
                }
            )

            if args.review_images:
                annotated = result.plot()
                preview = Image.fromarray(annotated[:, :, ::-1].copy())
                preview.thumbnail((240, 180))
                review_samples.append((minimum_confidence, source_path.name, preview.copy()))
                review_samples.sort(key=lambda item: (item[0], item[1].casefold()))
                del review_samples[args.review_images:]

        if processed != {path_key(path) for path in eligible_images}:
            missing = len({path_key(path) for path in eligible_images} - processed)
            raise RuntimeError(f"Prediction did not return results for {missing} eligible image(s).")

        manifest_path = staging_root / "pseudo_label_manifest.csv"
        manifest_columns = (
            "source_image",
            "status",
            "matched_splits",
            "pseudo_image",
            "pseudo_label",
            "detection_count",
            "minimum_confidence",
            "mean_confidence",
            "maximum_confidence",
        )
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=manifest_columns)
            writer.writeheader()
            writer.writerows(
                sorted(manifest_rows, key=lambda row: row["source_image"].casefold())
            )

        review_sheets = create_review_sheets(review_samples, staging_root)
        summary = {
            "source_images_root": str(images_root),
            "labeled_dataset_root": str(dataset_root),
            "teacher_weights": str(weights),
            "class_count": len(expected_names),
            "overlap_report": str(overlap_report),
            "excluded_labeled_splits": sorted(excluded_splits),
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "image_size": args.imgsz,
            "device": args.device,
            "source_image_count": len(input_images),
            "excluded_candidate_images": len(exclusions),
            "excluded_candidate_images_by_split": {
                split: sum(split in splits for splits in exclusions.values())
                for split in sorted(excluded_splits)
            },
            "eligible_image_count": len(eligible_images),
            "pseudo_labeled_image_count": pseudo_image_count,
            "images_without_detections_at_threshold": no_detection_count,
            "pseudo_box_count": pseudo_box_count,
            "mean_pseudo_box_confidence": (
                sum(box_confidences) / len(box_confidences) if box_confidences else None
            ),
            "review_sheets": review_sheets,
            "manifest": str(output_root / "pseudo_label_manifest.csv"),
            "notes": [
                "Only images with at least one detection at the confidence threshold are copied into the pseudo-labeled training subset.",
                "Images with no detections are omitted rather than treated as confirmed negatives.",
                "Validation/test perceptual-hash candidates are excluded; source images are never moved or modified.",
            ],
        }
        (staging_root / "pseudo_label_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_root.rename(output_root)

    print(f"Source images: {len(input_images)}")
    print(f"Excluded validation/test candidates: {len(exclusions)}")
    print(f"Eligible for teacher inference: {len(eligible_images)}")
    print(f"Pseudo-labeled images: {pseudo_image_count}")
    print(f"Images with no detections at conf {args.conf:g}: {no_detection_count}")
    print(f"Pseudo boxes: {pseudo_box_count}")
    print(f"Pseudo-label dataset: {output_root}")
    print(f"Manifest: {output_root / 'pseudo_label_manifest.csv'}")
    print(f"Summary: {output_root / 'pseudo_label_summary.json'}")
    if review_sheets:
        print(f"Review sheet(s): {', '.join(str(output_root / name) for name in review_sheets)}")


if __name__ == "__main__":
    main()
