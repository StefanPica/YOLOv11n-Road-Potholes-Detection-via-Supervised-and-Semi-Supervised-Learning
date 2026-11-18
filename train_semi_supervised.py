import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from dataset_config import DEFAULT_DATASET_ROOT, PROJECT_ROOT, validate_clean_split_audit
from prepare_pseudo_labels import DEFAULT_OUTPUT_ROOT, DEFAULT_TEACHER_WEIGHTS


DEFAULT_OVERLAP_REPORT = PROJECT_ROOT / "reports" / "unlabeled_overlap_candidates.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the supervised teacher with reviewed pseudo-labels."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--teacher-weights", type=Path, default=DEFAULT_TEACHER_WEIGHTS)
    parser.add_argument("--overlap-report", type=Path, default=DEFAULT_OVERLAP_REPORT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", default="pothole-yolo11n-semi-supervised-r1")
    return parser.parse_args()


def path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def validate_pseudo_data(
    pseudo_root: Path,
    dataset_root: Path,
    overlap_report: Path,
) -> tuple[int, int]:
    summary_path = pseudo_root / "pseudo_label_summary.json"
    manifest_path = pseudo_root / "pseudo_label_manifest.csv"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expected pseudo-label summary and manifest under {pseudo_root}."
        )

    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    excluded_splits = set(summary.get("excluded_labeled_splits", []))
    if not {"valid", "test"}.issubset(excluded_splits):
        raise ValueError("Pseudo-label preparation must exclude both validation and test candidates.")
    if Path(summary.get("labeled_dataset_root", "")).resolve() != dataset_root:
        raise ValueError("Pseudo-label summary was created for a different labeled dataset.")
    if Path(summary.get("overlap_report", "")).resolve() != overlap_report:
        raise ValueError("Pseudo-label summary used a different overlap report.")

    pseudo_images = pseudo_root / "images"
    pseudo_labels = pseudo_root / "labels"
    if not pseudo_images.is_dir() or not pseudo_labels.is_dir():
        raise FileNotFoundError(f"Expected pseudo image and label folders under {pseudo_root}.")

    included: dict[str, str] = {}
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["status"] != "pseudo_labeled":
                continue
            image_path = Path(row["pseudo_image"]).resolve()
            label_path = Path(row["pseudo_label"]).resolve()
            try:
                image_path.relative_to(pseudo_images.resolve())
                label_path.relative_to(pseudo_labels.resolve())
            except ValueError as error:
                raise ValueError("Pseudo manifest references files outside its dataset folder.") from error
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing pseudo image or label: {image_path}, {label_path}")
            included[path_key(Path(row["source_image"]))] = str(image_path)

    if not included:
        raise ValueError("Pseudo-label manifest contains no usable pseudo-labeled images.")

    excluded_candidates: set[str] = set()
    with overlap_report.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["labeled_split"] in {"valid", "test"}:
                excluded_candidates.add(path_key(Path(row["unlabeled_file"])))
    leaked = set(included) & excluded_candidates
    if leaked:
        raise ValueError(
            f"{len(leaked)} validation/test overlap candidate(s) are present in pseudo training data."
        )

    image_paths = sorted(
        path for path in pseudo_images.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )
    if {path_key(path) for path in image_paths} != {
        path_key(Path(value)) for value in included.values()
    }:
        raise ValueError("Pseudo image files and manifest entries do not match.")

    class_count = int(summary.get("class_count", 1))
    box_count = 0
    for image_path in image_paths:
        label_path = (pseudo_labels / image_path.relative_to(pseudo_images)).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing pseudo label for {image_path}: {label_path}")
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Pseudo label file is empty: {label_path}")
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {line}")
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
            if class_id < 0 or class_id >= class_count:
                raise ValueError(f"Invalid class id at {label_path}:{line_number}: {class_id}")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
                raise ValueError(f"Invalid normalized coordinates at {label_path}:{line_number}")
            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise ValueError(f"Zero-area box at {label_path}:{line_number}")
            box_count += 1
    return len(image_paths), box_count


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise SystemExit("Epochs, image size, and batch must be positive; workers cannot be negative.")
    if args.lr0 <= 0 or args.patience < 0:
        raise SystemExit("--lr0 must be positive and --patience cannot be negative.")

    dataset_root = args.dataset_root.expanduser().resolve()
    pseudo_root = args.pseudo_root.expanduser().resolve()
    teacher_weights = args.teacher_weights.expanduser().resolve()
    overlap_report = args.overlap_report.expanduser().resolve()
    if not teacher_weights.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_weights}")
    if not overlap_report.is_file():
        raise FileNotFoundError(f"Overlap report not found: {overlap_report}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu or repair the CUDA environment.")

    validate_clean_split_audit(dataset_root)
    pseudo_image_count, pseudo_box_count = validate_pseudo_data(
        pseudo_root,
        dataset_root,
        overlap_report,
    )
    metadata_path = dataset_root / "data.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    class_names = metadata["names"]
    if isinstance(class_names, dict):
        class_names = [name for _, name in sorted(class_names.items(), key=lambda item: int(item[0]))]
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid class names in {metadata_path}")

    with tempfile.TemporaryDirectory(prefix="pothole-semi-yolo-") as temp_dir:
        combined_yaml = Path(temp_dir) / "dataset.yaml"
        data = {
            "path": str(dataset_root),
            "train": [
                str(dataset_root / "train" / "images"),
                str(pseudo_root / "images"),
            ],
            "val": "valid/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names,
        }
        combined_yaml.write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )

        model = YOLO(str(teacher_weights))
        model.train(
            data=str(combined_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=42,
            deterministic=True,
            pretrained=True,
            patience=args.patience,
            cache=False,
            plots=True,
            lr0=args.lr0,
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name=args.name,
            exist_ok=False,
        )

    save_dir = Path(model.trainer.save_dir)
    print(f"Supervised training images: {len(list((dataset_root / 'train' / 'images').glob('*')))}")
    print(f"Pseudo-labeled images: {pseudo_image_count}")
    print(f"Pseudo boxes: {pseudo_box_count}")
    print(f"Semi-supervised checkpoint: {save_dir / 'weights' / 'best.pt'}")
    print(f"Training and validation plots: {save_dir}")
    print("The held-out test split was not evaluated or used for training.")


if __name__ == "__main__":
    main()
