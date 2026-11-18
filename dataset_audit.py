import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from dataset_config import PROJECT_ROOT, SOURCE_DATASET_ROOT, SPLITS, load_dataset_metadata
from duplicate_utils import IMAGE_EXTENSIONS, find_duplicate_pairs, phashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check YOLO labels and flag exact/near-duplicate images."
    )
    parser.add_argument("--dataset-root", type=Path, default=SOURCE_DATASET_ROOT)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "source_dataset_audit.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, class_names = load_dataset_metadata(args.dataset_root)
    errors: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    counts: dict[str, int] = {}

    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[split] = len(images)
        if not images:
            errors.append({"issue": "empty_split", "split": split, "file": str(image_dir)})

        image_stems = {image.stem for image in images}
        label_stems = {label.stem for label in label_dir.glob("*.txt")}
        for stem in sorted(image_stems - label_stems):
            errors.append(
                {"issue": "missing_label", "split": split, "file": str(image_dir / stem)}
            )
        for stem in sorted(label_stems - image_stems):
            errors.append(
                {"issue": "orphan_label", "split": split, "file": str(label_dir / f"{stem}.txt")}
            )

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                hashes = phashes(image_path)
            except (OSError, UnidentifiedImageError) as error:
                errors.append(
                    {"issue": "unreadable_image", "split": split, "file": str(image_path),
                     "detail": str(error)}
                )
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                for line_number, line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    fields = line.split()
                    try:
                        if len(fields) != 5:
                            raise ValueError("expected class x_center y_center width height")
                        class_id = int(fields[0])
                        coordinates = [float(value) for value in fields[1:]]
                        if not 0 <= class_id < len(class_names):
                            raise ValueError(f"class id {class_id} is out of range")
                        x_center, y_center, box_width, box_height = coordinates
                        if not all(0 <= value <= 1 for value in coordinates):
                            raise ValueError("coordinates must be normalized to [0, 1]")
                        if box_width <= 0 or box_height <= 0:
                            raise ValueError("box width and height must be positive")
                        if not (0 <= x_center - box_width / 2 and x_center + box_width / 2 <= 1):
                            raise ValueError("box extends outside the image horizontally")
                        if not (0 <= y_center - box_height / 2 and y_center + box_height / 2 <= 1):
                            raise ValueError("box extends outside the image vertically")
                    except ValueError as error:
                        errors.append(
                            {
                                "issue": "invalid_label",
                                "split": split,
                                "file": str(label_path),
                                "detail": f"line {line_number}: {error}",
                            }
                        )

            records.append(
                {
                    "split": split,
                    "path": image_path,
                    "sha256": digest,
                    "phashes": hashes,
                    "dimensions": f"{width}x{height}",
                }
            )

    candidates = find_duplicate_pairs(records, args.phash_threshold)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "issue", "split", "file", "match_split", "match_file",
        "phash_distance", "dimensions", "detail", "dataset_root",
    )
    with args.report.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({"issue": "audit_metadata", "dataset_root": str(root)})
        writer.writerows(errors)
        writer.writerows(candidates)

    print(f"Dataset: {root}")
    print(f"Classes: {', '.join(class_names)}")
    print("Images: " + ", ".join(f"{split}={counts[split]}" for split in SPLITS))
    print(f"Structural/label issues: {len(errors)}")
    cross_split = sum(row["split"] != row["match_split"] for row in candidates)
    same_split = len(candidates) - cross_split
    print(f"Near/exact duplicate candidates: {len(candidates)} total")
    print(f"  Cross-split: {cross_split}; within-split: {same_split}")
    print(f"Report: {args.report.resolve()}")
    if candidates:
        print("Candidates are grouped by the clean-split preparation script; this audit does not move or delete files.")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
