import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from dataset_config import DEFAULT_DATASET_ROOT, PROJECT_ROOT, validate_clean_split_audit
from duplicate_utils import IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review validation-set false positives and missed detections."
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained best.pt.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--conf", type=float, default=0.442)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching boxes.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="GPU index, or 'cpu' if needed.")
    parser.add_argument("--samples", type=int, default=20, help="Maximum annotated error images to save.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def read_labels(label_path: Path, width: int, height: int) -> list[dict[str, object]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())
        boxes.append(
            {
                "class_id": int(class_id),
                "box": [
                    (x_center - box_width / 2) * width,
                    (y_center - box_height / 2) * height,
                    (x_center + box_width / 2) * width,
                    (y_center + box_height / 2) * height,
                ],
            }
        )
    return boxes


def annotate_errors(
    image_path: Path,
    destination: Path,
    ground_truth: list[dict[str, object]],
    predictions: list[dict[str, object]],
    true_positive_indices: set[int],
    confidence: float,
    iou_threshold: float,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for target in ground_truth:
        box = target["box"]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1] - 14)), "GT", fill="red")
    for index, prediction in enumerate(predictions):
        box = prediction["box"]
        color = "lime" if index in true_positive_indices else "yellow"
        tag = "TP" if index in true_positive_indices else "FP"
        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0], min(image.height - 14, box[3] + 2)),
            f"{tag} {prediction['confidence']:.2f}",
            fill=color,
        )
    draw.rectangle((0, 0, image.width, 22), fill="black")
    draw.text(
        (4, 4),
        f"red=GT green=TP yellow=FP | conf>{confidence:g}, IoU>{iou_threshold:g}",
        fill="white",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=90)


def main() -> None:
    args = parse_args()
    if not 0 <= args.conf < 1:
        raise SystemExit("--conf must be at least 0 and less than 1.")
    if not 0 < args.iou <= 1:
        raise SystemExit("--iou must be greater than 0 and at most 1.")
    if args.samples < 0:
        raise SystemExit("--samples cannot be negative.")

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu or install CUDA-enabled PyTorch.")
    validate_clean_split_audit(args.dataset_root)

    dataset_root = args.dataset_root.expanduser().resolve()
    image_dir = dataset_root / "valid" / "images"
    label_dir = dataset_root / "valid" / "labels"
    images = sorted(
        image
        for image in image_dir.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No validation images found: {image_dir}")

    output = args.output
    if output is None:
        output = PROJECT_ROOT / "runs" / "detect" / f"{weights.parent.parent.name}-val-error-review"
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists and will not be overwritten: {output}")

    model = YOLO(str(weights))
    rows: list[dict[str, object]] = []
    sample_data: list[
        tuple[Path, list[dict[str, object]], list[dict[str, object]], set[int], int]
    ] = []
    totals = {"images": 0, "ground_truth": 0, "predictions": 0, "true_positives": 0,
              "false_positives": 0, "false_negatives": 0}
    results = model.predict(
        source=[str(image) for image in images],
        stream=True,
        conf=args.conf,
        iou=0.7,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        quantize=16 if args.device != "cpu" else None,
        verbose=False,
    )

    for result in results:
        image_path = Path(result.path)
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing validation label: {label_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        ground_truth = read_labels(label_path, width, height)

        predictions = []
        if result.boxes is not None:
            for box in result.boxes:
                coordinates = box.xyxy[0].cpu().tolist()
                predictions.append(
                    {
                        "class_id": int(box.cls[0].item()),
                        "confidence": float(box.conf[0].item()),
                        "box": [float(value) for value in coordinates],
                    }
                )
        predictions.sort(key=lambda prediction: prediction["confidence"], reverse=True)

        matched_ground_truth: set[int] = set()
        true_positive_indices: set[int] = set()
        false_positive_boxes: list[dict[str, object]] = []
        for prediction_index, prediction in enumerate(predictions):
            best_index = -1
            best_iou = 0.0
            for target_index, target in enumerate(ground_truth):
                if target_index in matched_ground_truth:
                    continue
                overlap = box_iou(prediction["box"], target["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = target_index
            if (
                best_index >= 0
                and best_iou > args.iou
                and best_index not in matched_ground_truth
                and prediction["class_id"] == ground_truth[best_index]["class_id"]
            ):
                matched_ground_truth.add(best_index)
                true_positive_indices.add(prediction_index)
            else:
                false_positive_boxes.append(prediction)

        missed_boxes = [
            target for index, target in enumerate(ground_truth)
            if index not in matched_ground_truth
        ]
        false_positive_count = len(false_positive_boxes)
        false_negative_count = len(missed_boxes)
        totals["images"] += 1
        totals["ground_truth"] += len(ground_truth)
        totals["predictions"] += len(predictions)
        totals["true_positives"] += len(true_positive_indices)
        totals["false_positives"] += false_positive_count
        totals["false_negatives"] += false_negative_count

        if false_positive_count or false_negative_count:
            rows.append(
                {
                    "image": str(image_path),
                    "ground_truth": len(ground_truth),
                    "predictions": len(predictions),
                    "true_positives": len(true_positive_indices),
                    "false_positives": false_positive_count,
                    "false_negatives": false_negative_count,
                    "false_positive_boxes_xyxy_confidence": json.dumps(
                        [
                            {"box": [round(value, 1) for value in prediction["box"]],
                             "confidence": round(prediction["confidence"], 4)}
                            for prediction in false_positive_boxes
                        ]
                    ),
                    "missed_boxes_xyxy": json.dumps(
                        [
                            [round(value, 1) for value in target["box"]]
                            for target in missed_boxes
                        ]
                    ),
                }
            )
            sample_data.append(
                (
                    image_path,
                    ground_truth,
                    predictions,
                    true_positive_indices,
                    false_positive_count + false_negative_count,
                )
            )

    output.mkdir(parents=True)
    csv_path = output / "validation_errors.csv"
    columns = [
        "image", "ground_truth", "predictions", "true_positives", "false_positives",
        "false_negatives", "false_positive_boxes_xyxy_confidence", "missed_boxes_xyxy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    sample_data.sort(key=lambda item: (-item[4], str(item[0]).casefold()))
    for image_path, ground_truth, predictions, true_positive_indices, _ in sample_data[:args.samples]:
        annotate_errors(
            image_path,
            output / "samples" / f"{image_path.stem}__errors.jpg",
            ground_truth,
            predictions,
            true_positive_indices,
            args.conf,
            args.iou,
        )

    summary = {
        "weights": str(weights),
        "dataset": str(dataset_root),
        "split": "valid",
        "device": args.device,
        "inference_precision": "fp16" if args.device != "cpu" else "fp32",
        "confidence_threshold": args.conf,
        "nms_iou_threshold": 0.7,
        "matching_iou_threshold": args.iou,
        **totals,
        "precision": (
            totals["true_positives"] / (totals["true_positives"] + totals["false_positives"])
            if totals["true_positives"] + totals["false_positives"]
            else 0.0
        ),
        "recall": (
            totals["true_positives"] / (totals["true_positives"] + totals["false_negatives"])
            if totals["true_positives"] + totals["false_negatives"]
            else 0.0
        ),
        "images_with_errors": len(rows),
        "annotated_samples": min(args.samples, len(sample_data)),
        "csv": str(csv_path),
        "metric_note": (
            "Standalone prediction pass for qualitative review; use evaluate.py for reported metrics. "
            "CPU/GPU inference can differ for predictions near the confidence threshold."
        ),
    }
    summary["f1"] = (
        2 * summary["precision"] * summary["recall"] / (summary["precision"] + summary["recall"])
        if summary["precision"] + summary["recall"]
        else 0.0
    )
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Validation images: {totals['images']}")
    print(f"TP / FP / FN: {totals['true_positives']} / {totals['false_positives']} / "
          f"{totals['false_negatives']}")
    print(f"Precision / recall / F1: {summary['precision']:.4f} / "
          f"{summary['recall']:.4f} / {summary['f1']:.4f}")
    print(f"Images with errors: {len(rows)}")
    print(f"Error report: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
