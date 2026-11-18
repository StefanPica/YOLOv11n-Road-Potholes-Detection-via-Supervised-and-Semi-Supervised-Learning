import argparse
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO

from dataset_config import (
    DEFAULT_DATASET_ROOT,
    PROJECT_ROOT,
    validate_clean_split_audit,
    write_ultralytics_yaml,
)


def check_cuda() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch. Install the CUDA-enabled PyTorch build "
            "using the commands in README.md, then run this check again."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the supervised pothole YOLO baseline.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4, help="Keep small for a 4 GB GPU.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0", help="GPU index, or 'cpu' if needed.")
    parser.add_argument("--name", default="pothole-yolo11n-baseline")
    parser.add_argument("--check-cuda", action="store_true", help="Check CUDA and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_cuda:
        check_cuda()
        return
    if args.device != "cpu":
        check_cuda()
    validate_clean_split_audit(args.dataset_root)

    with tempfile.TemporaryDirectory(prefix="pothole-yolo-") as temp_dir:
        data_yaml = write_ultralytics_yaml(
            args.dataset_root,
            Path(temp_dir) / "dataset.yaml",
        )
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=42,
            deterministic=True,
            pretrained=True,
            patience=20,
            cache=False,
            plots=True,
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name=args.name,
            exist_ok=False,
        )

    save_dir = Path(model.trainer.save_dir)
    print(f"\nTraining complete. Best checkpoint: {save_dir / 'weights' / 'best.pt'}")
    print(f"Training and validation plots: {save_dir}")


if __name__ == "__main__":
    main()
