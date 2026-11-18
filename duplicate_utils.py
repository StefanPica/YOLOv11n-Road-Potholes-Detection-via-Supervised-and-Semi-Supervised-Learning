from pathlib import Path
from typing import Any

import imagehash
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ROTATIONS = (0, 90, 180, 270)


def phashes(path: Path) -> tuple[int, ...]:
    with Image.open(path) as image:
        return tuple(
            int(
                "".join(
                    "1" if bit else "0"
                    for bit in imagehash.phash(image.rotate(angle, expand=True)).hash.flatten()
                ),
                2,
            )
            for angle in ROTATIONS
        )


def hamming_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    distance = 64
    for left in first:
        for right in second:
            distance = min(distance, (left ^ right).bit_count())
            if distance == 0:
                return 0
    return distance


def find_duplicate_pairs(
    records: list[dict[str, Any]],
    threshold: int,
) -> list[dict[str, str]]:
    if not 0 <= threshold <= 64:
        raise ValueError("Perceptual-hash threshold must be between 0 and 64.")

    candidates: list[dict[str, str]] = []
    ordered = sorted(records, key=lambda record: str(record["path"]).casefold())
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            exact = first["sha256"] == second["sha256"]
            if exact:
                distance = 0
            else:
                distance = hamming_distance(first["phashes"], second["phashes"])
                if distance > threshold:
                    continue

            same_split = first["split"] == second["split"]
            if exact:
                issue = "exact_duplicate_within_split" if same_split else "exact_duplicate_across_splits"
            else:
                issue = "near_duplicate_within_split_candidate" if same_split else "near_duplicate_candidate"
            candidates.append(
                {
                    "issue": issue,
                    "split": str(first["split"]),
                    "file": str(first["path"]),
                    "match_split": str(second["split"]),
                    "match_file": str(second["path"]),
                    "phash_distance": str(distance),
                    "dimensions": str(first.get("dimensions", "")),
                }
            )
    return candidates
