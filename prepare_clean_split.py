import argparse
import csv
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from dataset_config import (
    DEFAULT_DATASET_ROOT,
    PROJECT_ROOT,
    SOURCE_DATASET_ROOT,
    SPLITS,
    load_dataset_metadata,
)
from duplicate_utils import IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a separate grouped train/validation/test copy from an audit report."
    )
    parser.add_argument("--dataset-root", type=Path, default=SOURCE_DATASET_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "source_dataset_audit.csv",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_root(parents: dict[Path, Path], path: Path) -> Path:
    parents.setdefault(path, path)
    if parents[path] != path:
        parents[path] = find_root(parents, parents[path])
    return parents[path]


def join_groups(parents: dict[Path, Path], first: Path, second: Path) -> None:
    first_root = find_root(parents, first)
    second_root = find_root(parents, second)
    if first_root != second_root:
        parents[second_root] = first_root


def main() -> None:
    args = parse_args()
    root, class_names = load_dataset_metadata(args.dataset_root)
    report = args.report.expanduser().resolve()
    if not report.is_file():
        raise FileNotFoundError(
            f"Audit report not found: {report}. Run dataset_audit.py first."
        )
    report_root = None
    report_errors: list[str] = []
    with report.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            issue = row.get("issue", "")
            if issue == "audit_metadata":
                report_root = Path(row["dataset_root"]).resolve()
            elif issue not in {
                "near_duplicate_candidate",
                "near_duplicate_within_split_candidate",
                "exact_duplicate_across_splits",
                "exact_duplicate_within_split",
            }:
                report_errors.append(issue or "unknown")
    if report_root != root:
        raise ValueError(f"Audit report does not match the source dataset: {root}")
    if report_errors:
        raise ValueError(
            f"Fix {len(report_errors)} structural/label issue(s) in the source audit before splitting."
        )

    output_root = args.output_root.expanduser().resolve()
    if root == output_root or root in output_root.parents or output_root in root.parents:
        raise ValueError("The output and source dataset must be separate, non-nested folders.")
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_root}"
        )

    fractions = {
        "train": args.train_fraction,
        "valid": args.val_fraction,
        "test": args.test_fraction,
    }
    if any(value <= 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-8:
        raise ValueError("Train, validation, and test fractions must be positive and sum to 1.")

    records: dict[Path, dict[str, Any]] = {}
    parents: dict[Path, Path] = {}
    for split in SPLITS:
        image_dir = root / split / "images"
        for image in sorted(image_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = root / split / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing YOLO label for {image}: {label}")
            resolved_image = image.resolve()
            records[resolved_image] = {
                "image": image,
                "label": label,
                "source_split": split,
            }
            parents[resolved_image] = resolved_image

    candidate_pairs = 0
    with report.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            issue = row.get("issue", "")
            if not issue.startswith(("near_duplicate", "exact_duplicate")):
                continue
            first = Path(row["file"]).expanduser().resolve()
            second = Path(row["match_file"]).expanduser().resolve()
            if first not in records or second not in records:
                raise ValueError(
                    f"Audit report does not match the source dataset: {first}, {second}"
                )
            join_groups(parents, first, second)
            candidate_pairs += 1

    groups: dict[Path, list[Path]] = {}
    for path in records:
        groups.setdefault(find_root(parents, path), []).append(path)
    ordered_groups = sorted(groups.values(), key=lambda items: min(str(path) for path in items))
    randomizer = random.Random(args.seed)
    randomizer.shuffle(ordered_groups)
    ordered_groups.sort(key=len, reverse=True)

    targets = {split: len(records) * fraction for split, fraction in fractions.items()}
    assigned_counts = {split: 0 for split in SPLITS}
    assigned_groups: list[tuple[str, list[Path]]] = []
    for group in ordered_groups:
        size = len(group)

        def assignment_cost(candidate: str) -> float:
            return sum(
                ((assigned_counts[split] + (size if split == candidate else 0) - targets[split])
                 / targets[split]) ** 2
                for split in SPLITS
            )

        destination_split = min(SPLITS, key=assignment_cost)
        assigned_counts[destination_split] += size
        assigned_groups.append((destination_split, group))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}-",
    ) as temporary:
        temporary_root = Path(temporary)
        for split in SPLITS:
            (temporary_root / split / "images").mkdir(parents=True)
            (temporary_root / split / "labels").mkdir(parents=True)

        group_rows: list[dict[str, str]] = []
        duplicate_group_count = 0
        group_number = 0
        for destination_split, group in assigned_groups:
            group_number += 1
            group_id = f"group-{group_number:05d}"
            if len(group) > 1:
                duplicate_group_count += 1
            for source_path in group:
                record = records[source_path]
                filename = f"{record['source_split']}__{record['image'].name}"
                image_destination = temporary_root / destination_split / "images" / filename
                label_destination = temporary_root / destination_split / "labels" / f"{Path(filename).stem}.txt"
                shutil.copy2(record["image"], image_destination)
                shutil.copy2(record["label"], label_destination)
                group_rows.append(
                    {
                        "group_id": group_id,
                        "duplicate_group": str(len(group) > 1).lower(),
                        "source_split": record["source_split"],
                        "new_split": destination_split,
                        "source_image": str(record["image"]),
                        "new_image": str(output_root / destination_split / "images" / filename),
                    }
                )

        data_yaml = {
            "path": str(output_root),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names,
        }
        with (temporary_root / "data.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(data_yaml, file, sort_keys=False)
        with (temporary_root / "groups.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=group_rows[0].keys())
            writer.writeheader()
            writer.writerows(group_rows)

        temporary_root.rename(output_root)

    print(f"Source images: {len(records)}")
    print(f"Duplicate candidate pairs grouped: {candidate_pairs}")
    print(f"Duplicate groups: {duplicate_group_count}")
    print("New split counts: " + ", ".join(f"{split}={assigned_counts[split]}" for split in SPLITS))
    print(f"Clean split copy: {output_root}")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
