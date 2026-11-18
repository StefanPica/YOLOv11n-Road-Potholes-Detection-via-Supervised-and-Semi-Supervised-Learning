import argparse
import json
from pathlib import Path
import tempfile

import torch
from ultralytics import YOLO

from dataset_config import (
    DEFAULT_DATASET_ROOT,
    PROJECT_ROOT,
    validate_clean_split_audit,
    write_ultralytics_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector on a held-out split.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained best.pt.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for the operating-point confusion matrix (default: 0.25).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="GPU index, or 'cpu' if needed.")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def operating_point_metrics(metrics: object) -> tuple[dict[str, float | int], list[list[float]]]:
    confusion_matrix = getattr(metrics, "confusion_matrix", None)
    if confusion_matrix is None:
        raise RuntimeError("Ultralytics did not return a detection confusion matrix.")

    matrix = confusion_matrix.matrix
    true_positives = int(matrix[:-1, :-1].trace())
    false_positives = int(matrix[:-1, :].sum() - true_positives)
    false_negatives = int(matrix[:, :-1].sum() - true_positives)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        matrix.tolist(),
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.conf < 1:
        raise SystemExit("--conf must be at least 0 and less than 1.")
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch. Install the CUDA-enabled PyTorch build "
            "using the commands in README.md, or pass --device cpu."
        )
    validate_clean_split_audit(args.dataset_root)

    name = args.name or f"{weights.parent.parent.name}-{args.split}"
    with tempfile.TemporaryDirectory(prefix="pothole-yolo-") as temp_dir:
        data_yaml = write_ultralytics_yaml(
            args.dataset_root,
            Path(temp_dir) / "dataset.yaml",
        )
        model = YOLO(str(weights))
        map_name = f"{name}-map"
        operating_name = f"{name}-conf-{args.conf:g}"
        map_metrics = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            plots=True,
            save_json=True,
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name=map_name,
            exist_ok=False,
        )
        operating_metrics = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            plots=True,
            save_json=False,
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name=operating_name,
            exist_ok=False,
        )

    operating_point, confusion_matrix = operating_point_metrics(operating_metrics)
    summary = {
        "weights": str(weights),
        "dataset": str(args.dataset_root.expanduser().resolve()),
        "split": args.split,
        "image_size": args.imgsz,
        "map_evaluation": {
            "confidence_floor": 0.001,
            "map50": float(map_metrics.box.map50),
            "map50_95": float(map_metrics.box.map),
            "output_dir": str(map_metrics.save_dir),
        },
        "operating_point": {
            "confidence_threshold": args.conf,
            "iou_threshold": 0.5,
            **operating_point,
            "confusion_matrix": {
                "labels": ["pothole", "background"],
                "orientation": "rows=predicted, columns=true",
                "counts": confusion_matrix,
            },
            "output_dir": str(operating_metrics.save_dir),
        },
    }
    summary_path = Path(operating_metrics.save_dir) / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nThreshold-swept detection metrics")
    print(f"mAP@0.50:         {summary['map_evaluation']['map50']:.4f}")
    print(f"mAP@0.50:0.95:    {summary['map_evaluation']['map50_95']:.4f}")
    print(f"mAP curves/output: {map_metrics.save_dir}")
    print(f"\nOperating point (confidence > {args.conf:g}, IoU >= 0.50)")
    print(f"Precision:        {operating_point['precision']:.4f}")
    print(f"Recall:           {operating_point['recall']:.4f}")
    print(f"F1:               {operating_point['f1']:.4f}")
    print(f"TP / FP / FN:     {operating_point['true_positives']} / "
          f"{operating_point['false_positives']} / {operating_point['false_negatives']}")
    print(f"Operating-point outputs: {operating_metrics.save_dir}")
    print(f"Metrics summary: {summary_path}")
    print("Use validation results to choose --conf; reserve the test split for final evaluation.")


if __name__ == "__main__":
    main()
