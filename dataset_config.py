import csv
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DATASET_ROOT = PROJECT_ROOT.parent / "resurse project 2" / "dataset1"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "leakage_clean_split"
SPLITS = ("train", "valid", "test")


def load_dataset_metadata(dataset_root: Path) -> tuple[Path, list[str]]:
    root = dataset_root.expanduser().resolve()
    metadata_file = root / "data.yaml"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {metadata_file}")

    with metadata_file.open("r", encoding="utf-8") as file:
        metadata: dict[str, Any] = yaml.safe_load(file) or {}

    names = metadata.get("names")
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Expected a non-empty class-name list in {metadata_file}")
    if metadata.get("nc", len(names)) != len(names):
        raise ValueError(f"Class count 'nc' does not match 'names' in {metadata_file}")

    for split in SPLITS:
        for folder in (root / split / "images", root / split / "labels"):
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected dataset folder: {folder}")

    return root, names


def write_ultralytics_yaml(dataset_root: Path, destination: Path) -> Path:
    root, names = load_dataset_metadata(dataset_root)
    data = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return destination


def validate_clean_split_audit(dataset_root: Path) -> None:
    root, _ = load_dataset_metadata(dataset_root)
    report = PROJECT_ROOT / "reports" / "clean_split_audit.csv"
    if not report.is_file():
        raise FileNotFoundError(
            f"Clean-split audit is missing: {report}. Run dataset_audit.py on the clean split first."
        )

    metadata_root = None
    cross_split_candidates: list[dict[str, str]] = []
    structural_issues: list[dict[str, str]] = []
    allowed_issues = {
        "audit_metadata",
        "exact_duplicate_within_split",
        "near_duplicate_within_split_candidate",
    }
    with report.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            issue = row.get("issue", "")
            if issue == "audit_metadata":
                metadata_root = Path(row["dataset_root"]).resolve()
            elif issue in {"near_duplicate_candidate", "exact_duplicate_across_splits"}:
                cross_split_candidates.append(row)
            elif issue not in allowed_issues:
                structural_issues.append(row)

    if metadata_root != root:
        raise ValueError(
            f"Audit report {report} does not match the dataset being used: {root}"
        )
    if structural_issues:
        raise ValueError(
            f"Clean-split audit reported {len(structural_issues)} structural/label issue(s): {report}"
        )
    if cross_split_candidates:
        raise ValueError(
            f"Clean-split audit still has {len(cross_split_candidates)} cross-split duplicate candidate(s): "
            f"{report}. Regroup the candidates before training or evaluation."
        )
