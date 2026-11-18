import argparse
import csv
import io
import json
import math
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO

from dataset_config import DEFAULT_DATASET_ROOT, PROJECT_ROOT, load_dataset_metadata, validate_clean_split_audit
from duplicate_utils import IMAGE_EXTENSIONS
from evaluate import operating_point_metrics


DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "pothole-yolo11n-semi-r1-conf060-40ep-20261116"
    / "weights"
    / "best.pt"
)

CORRUPTIONS: dict[str, dict[str, int | float]] = {
    "low_light": {"mild": 0.8, "moderate": 0.6, "strong": 0.4},
    "blur": {"mild": 0.5, "moderate": 1.0, "strong": 2.0},
    "jpeg": {"mild": 70, "moderate": 40, "strong": 15},
    "gaussian_noise": {"mild": 5, "moderate": 15, "strong": 30},
}
DEFAULT_COVERAGES = (0.10, 0.25, 0.50, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate common image corruptions, using validation by default."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--conf", type=float, default=0.368368)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        default="valid",
        help="Use test only for one frozen family/severity/coverage condition.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(CORRUPTIONS),
        default=list(CORRUPTIONS),
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        choices=("mild", "moderate", "strong"),
        default=["mild", "moderate", "strong"],
    )
    parser.add_argument(
        "--coverages",
        nargs="+",
        type=float,
        default=list(DEFAULT_COVERAGES),
        help="Fractions of validation images to corrupt, from 0 (exclusive) to 1.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def select_affected_indices(
    image_count: int,
    coverages: tuple[float, ...],
    seed: int,
) -> dict[float, set[int]]:
    order = list(range(image_count))
    random.Random(seed).shuffle(order)
    selected: dict[float, set[int]] = {}
    for coverage in sorted(coverages):
        count = min(image_count, math.ceil(image_count * coverage))
        selected[coverage] = set(order[:count])
    return selected


def validate_test_scope(
    split: str,
    families: list[str],
    severities: list[str],
    coverages: list[float],
) -> None:
    if split == "test" and (
        len(families) != 1 or len(severities) != 1 or len(coverages) != 1
    ):
        raise SystemExit(
            "Test evaluation requires exactly one family, severity, and coverage; "
            "freeze that condition using validation results first."
        )


def apply_corruption(
    image: Image.Image,
    corruption: str,
    parameter: int | float,
    seed: int,
) -> Image.Image:
    rgb = image.convert("RGB")
    if corruption == "low_light":
        return ImageEnhance.Brightness(rgb).enhance(float(parameter))
    if corruption == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=float(parameter)))
    if corruption == "jpeg":
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=int(parameter))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if corruption == "gaussian_noise":
        pixels = np.asarray(rgb, dtype=np.float32)
        noise = np.random.default_rng(seed).normal(0, float(parameter), pixels.shape)
        return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
    raise ValueError(f"Unknown corruption type: {corruption}")


def prepare_split_copy(
    source_images: list[Path],
    source_labels: Path,
    destination_root: Path,
    affected_indices: set[int],
    corruption: str | None,
    severity: str | None,
    parameter: int | float | None,
    seed: int,
) -> Path:
    images_dir = destination_root / "valid" / "images"
    labels_dir = destination_root / "valid" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    family_index = list(CORRUPTIONS).index(corruption) if corruption else 0
    severity_index = (
        list(CORRUPTIONS[corruption]).index(severity)
        if corruption and severity is not None
        else 0
    )
    for image_index, source_image in enumerate(source_images):
        source_label = source_labels / f"{source_image.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(f"Missing split label: {source_label}")
        shutil.copy2(source_label, labels_dir / source_label.name)
        destination_image = images_dir / source_image.name
        if image_index not in affected_indices or corruption is None:
            shutil.copy2(source_image, destination_image)
            continue
        if parameter is None:
            raise ValueError(f"Corruption {corruption} is missing its severity parameter.")
        with Image.open(source_image) as opened:
            transformed = apply_corruption(
                opened,
                corruption,
                parameter,
                seed + image_index * 1_000_003 + family_index * 1_009 + severity_index,
            )
            transformed.save(images_dir / f"{source_image.stem}.png", format="PNG")
    return destination_root


def write_dataset_yaml(destination_root: Path, names: list[str]) -> Path:
    dataset_yaml = destination_root / "data.yaml"
    metadata = {
        "path": str(destination_root.resolve()),
        "train": "valid/images",
        "val": "valid/images",
        "test": "valid/images",
        "nc": len(names),
        "names": names,
    }
    dataset_yaml.write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    return dataset_yaml


def evaluate_condition(
    model: YOLO,
    dataset_yaml: Path,
    output_root: Path,
    name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    shared = {
        "data": str(dataset_yaml),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "save_json": False,
        "project": str(output_root / "ultralytics"),
        "exist_ok": False,
    }
    map_metrics = model.val(conf=0.001, plots=False, name=f"{name}-map", **shared)
    # Ultralytics only fills its operating-point confusion matrix when plots are enabled.
    point_metrics = model.val(conf=args.conf, plots=True, name=f"{name}-operating", **shared)
    operating_point, _ = operating_point_metrics(point_metrics)
    return {
        "map50": float(map_metrics.box.map50),
        "map50_95": float(map_metrics.box.map),
        "confidence": args.conf,
        "precision": operating_point["precision"],
        "recall": operating_point["recall"],
        "f1": operating_point["f1"],
        "true_positives": operating_point["true_positives"],
        "false_positives": operating_point["false_positives"],
        "false_negatives": operating_point["false_negatives"],
    }


def save_results(output_root: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    summary_path = output_root / "robustness_summary.json"
    summary_path.write_text(
        json.dumps({"settings": metadata, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output_root / "robustness_summary.csv"
    columns = [
        "condition",
        "corruption",
        "severity",
        "parameter",
        "coverage",
        "affected_images",
        "total_images",
        "confidence",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
        "map50",
        "map50_95",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.conf < 1:
        raise SystemExit("--conf must be at least 0 and less than 1.")
    if args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise SystemExit("Image size and batch must be positive; workers cannot be negative.")
    if not args.coverages or any(not 0 < value <= 1 for value in args.coverages):
        raise SystemExit("Every coverage must be greater than 0 and at most 1.")
    if len(set(args.coverages)) != len(args.coverages):
        raise SystemExit("Coverage values must not contain duplicates.")
    if not args.families or len(set(args.families)) != len(args.families):
        raise SystemExit("Choose one or more distinct corruption families.")
    if not args.severities or len(set(args.severities)) != len(args.severities):
        raise SystemExit("Choose one or more distinct severity levels.")
    validate_test_scope(args.split, args.families, args.severities, args.coverages)

    dataset_root, class_names = load_dataset_metadata(args.dataset_root)
    validate_clean_split_audit(dataset_root)
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu or repair the CUDA environment.")

    image_dir = dataset_root / args.split / "images"
    label_dir = dataset_root / args.split / "labels"
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No validation images found: {image_dir}")

    coverages = tuple(sorted(args.coverages))
    selected = select_affected_indices(len(images), coverages, args.seed)
    output_root = args.output
    if output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_root = (
            PROJECT_ROOT
            / "runs"
            / "detect"
            / f"pothole-yolo11n-semi-r1-corruption-robustness-val-{timestamp}"
        )
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    output_root.mkdir(parents=True)

    settings = {
        "weights": str(weights),
        "dataset": str(dataset_root),
        "split": args.split,
        "device": args.device,
        "image_size": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": 0.7,
        "matching_iou_threshold": 0.5,
        "total_images": len(images),
        "families": args.families,
        "severities": args.severities,
        "coverage_counts": {
            f"{coverage:g}": len(selected[coverage]) for coverage in coverages
        },
        "corruptions": CORRUPTIONS,
        "notes": [
            f"Only the selected {args.split} split is read by this run.",
            "Each corruption family and severity is evaluated independently.",
            "Coverage subsets are deterministic and nested for comparisons.",
            "Corrupted pixels are saved losslessly as PNG after applying the selected transform.",
            "Clean images and labels in the source dataset are never modified.",
        ],
    }
    if args.split == "test":
        settings["notes"].append(
            "This is a single frozen test condition; interpret results as exploratory "
            "because the test split was previously evaluated."
        )
    model = YOLO(str(weights))
    rows: list[dict[str, Any]] = []
    conditions: list[tuple[str | None, str | None, int | float | None, float, set[int]]] = [
        (None, None, None, 0.0, set())
    ]
    for corruption in args.families:
        severity_values = CORRUPTIONS[corruption]
        for severity, parameter in severity_values.items():
            if severity not in args.severities:
                continue
            for coverage in coverages:
                conditions.append(
                    (corruption, severity, parameter, coverage, selected[coverage])
                )

    for corruption, severity, parameter, coverage, affected in conditions:
        condition = (
            "clean"
            if corruption is None
            else f"{corruption}-{severity}-{int(round(coverage * 100)):03d}"
        )
        print(f"Evaluating {condition} ({len(affected)}/{len(images)} images)")
        with tempfile.TemporaryDirectory(prefix="robustness-", dir=output_root) as temp_dir:
            temp_root = Path(temp_dir)
            prepare_split_copy(
                images,
                label_dir,
                temp_root,
                affected,
                corruption,
                severity,
                parameter,
                args.seed,
            )
            dataset_yaml = write_dataset_yaml(temp_root, class_names)
            metrics = evaluate_condition(model, dataset_yaml, output_root, condition, args)

        row = {
            "condition": condition,
            "corruption": corruption or "none",
            "severity": severity or "none",
            "parameter": parameter if parameter is not None else "",
            "coverage": coverage,
            "affected_images": len(affected),
            "total_images": len(images),
            **metrics,
        }
        rows.append(row)
        save_results(output_root, rows, settings)
        print(
            f"  P/R/F1={metrics['precision']:.3f}/{metrics['recall']:.3f}/"
            f"{metrics['f1']:.3f} | TP/FP/FN={metrics['true_positives']}/"
            f"{metrics['false_positives']}/{metrics['false_negatives']} | "
            f"mAP50/95={metrics['map50']:.3f}/{metrics['map50_95']:.3f}"
        )

    print(f"Robustness summary: {output_root / 'robustness_summary.csv'}")
    print(f"Detailed settings and metrics: {output_root / 'robustness_summary.json'}")


if __name__ == "__main__":
    main()
