import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import TypedDict

from PIL import Image, UnidentifiedImageError

from dataset_config import DEFAULT_DATASET_ROOT, PROJECT_ROOT, SPLITS, load_dataset_metadata
from duplicate_utils import IMAGE_EXTENSIONS, hamming_distance, phashes


class ImageRecord(TypedDict):
    path: Path
    sha256: str
    phashes: tuple[int, ...]
    dimensions: str
    size_bytes: int


class LabeledImageRecord(ImageRecord):
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit unlabeled images for readability and overlap with labeled splits."
    )
    parser.add_argument("--images", type=Path, required=True, help="Folder containing unlabeled images.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def fingerprint(path: Path) -> ImageRecord:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": path.resolve(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "phashes": phashes(path),
        "dimensions": f"{width}x{height}",
        "size_bytes": path.stat().st_size,
    }


def image_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.phash_threshold <= 64:
        raise SystemExit("--phash-threshold must be between 0 and 64.")

    unlabeled_root = args.images.expanduser().resolve()
    if not unlabeled_root.is_dir():
        raise NotADirectoryError(f"Unlabeled image folder not found: {unlabeled_root}")
    dataset_root, _ = load_dataset_metadata(args.dataset_root)
    unlabeled_paths = image_files(unlabeled_root)
    if not unlabeled_paths:
        raise FileNotFoundError(f"No supported images found under: {unlabeled_root}")

    unreadable: list[dict[str, str]] = []
    unlabeled_records: list[ImageRecord] = []
    for path in unlabeled_paths:
        try:
            unlabeled_records.append(fingerprint(path))
        except (OSError, UnidentifiedImageError) as error:
            unreadable.append({"split": "unlabeled", "file": str(path), "error": str(error)})

    labeled_records: list[LabeledImageRecord] = []
    labeled_counts: dict[str, int] = {}
    for split in SPLITS:
        paths = image_files(dataset_root / split / "images")
        labeled_counts[split] = len(paths)
        for path in paths:
            try:
                record = fingerprint(path)
                labeled_record = {**record, "split": split}
                labeled_records.append(labeled_record)
            except (OSError, UnidentifiedImageError) as error:
                unreadable.append({"split": split, "file": str(path), "error": str(error)})

    candidates: list[dict[str, str]] = []
    labeled_by_sha: dict[str, list[LabeledImageRecord]] = {}
    for record in labeled_records:
        labeled_by_sha.setdefault(str(record["sha256"]), []).append(record)

    for unlabeled in unlabeled_records:
        exact_matches = labeled_by_sha.get(str(unlabeled["sha256"]), [])
        for labeled in exact_matches:
            candidates.append(
                {
                    "kind": "exact",
                    "unlabeled_file": str(unlabeled["path"]),
                    "labeled_split": str(labeled["split"]),
                    "labeled_file": str(labeled["path"]),
                    "phash_distance": "0",
                    "unlabeled_dimensions": str(unlabeled["dimensions"]),
                    "labeled_dimensions": str(labeled["dimensions"]),
                }
            )

        for labeled in labeled_records:
            if labeled["sha256"] == unlabeled["sha256"]:
                continue
            distance = hamming_distance(
                unlabeled["phashes"],
                labeled["phashes"],
            )
            if distance <= args.phash_threshold:
                candidates.append(
                    {
                        "kind": "near",
                        "unlabeled_file": str(unlabeled["path"]),
                        "labeled_split": str(labeled["split"]),
                        "labeled_file": str(labeled["path"]),
                        "phash_distance": str(distance),
                        "unlabeled_dimensions": str(unlabeled["dimensions"]),
                        "labeled_dimensions": str(labeled["dimensions"]),
                    }
                )

    unlabeled_internal_candidates = 0
    for index, first in enumerate(unlabeled_records):
        for second in unlabeled_records[index + 1:]:
            if first["sha256"] == second["sha256"]:
                unlabeled_internal_candidates += 1
            elif hamming_distance(
                first["phashes"],
                second["phashes"],
            ) <= args.phash_threshold:
                unlabeled_internal_candidates += 1

    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = report_dir / "unlabeled_overlap_candidates.csv"
    columns = (
        "kind",
        "unlabeled_file",
        "labeled_split",
        "labeled_file",
        "phash_distance",
        "unlabeled_dimensions",
        "labeled_dimensions",
    )
    with candidate_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(candidates)

    unreadable_path = report_dir / "unlabeled_audit_unreadable.csv"
    with unreadable_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("split", "file", "error"))
        writer.writeheader()
        writer.writerows(unreadable)

    exact_count = sum(candidate["kind"] == "exact" for candidate in candidates)
    near_count = sum(candidate["kind"] == "near" for candidate in candidates)
    summary = {
        "unlabeled_root": str(unlabeled_root),
        "dataset_root": str(dataset_root),
        "phash_threshold": args.phash_threshold,
        "unlabeled_images_found": len(unlabeled_paths),
        "unlabeled_images_readable": len(unlabeled_records),
        "labeled_images_by_split": labeled_counts,
        "labeled_images_readable": len(labeled_records),
        "unreadable_images": len(unreadable),
        "exact_unlabeled_to_labeled_pairs": exact_count,
        "near_unlabeled_to_labeled_pairs": near_count,
        "unlabeled_internal_duplicate_candidates": unlabeled_internal_candidates,
        "overlap_candidates_csv": str(candidate_path),
        "unreadable_images_csv": str(unreadable_path),
        "notes": [
            "Perceptual-hash matches are candidates and require visual review.",
            "No images were moved, modified, labeled, or used for training.",
        ],
    }
    summary_path = report_dir / "unlabeled_dataset_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Unlabeled images found/readable: {len(unlabeled_paths)}/{len(unlabeled_records)}")
    print("Labeled images: " + ", ".join(f"{split}={labeled_counts[split]}" for split in SPLITS))
    print(f"Unreadable images: {len(unreadable)}")
    print(f"Exact overlaps with labeled data: {exact_count}")
    print(f"Near-duplicate candidates with labeled data (pHash <= {args.phash_threshold}): {near_count}")
    print(f"Unlabeled internal duplicate candidates: {unlabeled_internal_candidates}")
    print(f"Summary: {summary_path}")
    print(f"Overlap candidates: {candidate_path}")


if __name__ == "__main__":
    main()
