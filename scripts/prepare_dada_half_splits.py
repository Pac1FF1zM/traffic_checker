"""Create deterministic, source-disjoint DADA-2000 train/validation/test splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, items: list[str]) -> None:
    path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")


def digest(items: list[str]) -> str:
    return hashlib.sha256("".join(f"{item}\n" for item in items).encode()).hexdigest()


def stable_order(item: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()


def parse_clip(clip: str) -> tuple[int, int]:
    parts = clip.replace("\\", "/").split("/")
    if len(parts) != 2:
        raise ValueError(f"invalid DADA clip id: {clip!r}")
    return int(parts[0]), int(parts[1])


def metadata_strata(items: list[str], annotation_path: Path) -> dict[str, tuple[int, int, int]]:
    required = {
        "type",
        "video",
        "whether an accident occurred (1/0)",
        "light(day,night)1-2",
    }
    lookup: dict[tuple[int, int], tuple[int, int, int]] = {}
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"annotation is missing columns: {sorted(missing)}")
        for row in reader:
            key = (int(row["type"]), int(row["video"]))
            if key in lookup:
                raise ValueError(f"duplicate annotation row for {key}")
            lookup[key] = (
                key[0],
                int(row["whether an accident occurred (1/0)"]),
                int(row["light(day,night)1-2"]),
            )
    result = {}
    for item in items:
        key = parse_clip(item)
        if key not in lookup:
            raise ValueError(f"missing annotation for {item}")
        result[item] = lookup[key]
    return result


def assert_unique(name: str, items: list[str]) -> None:
    duplicates = [item for item, count in Counter(items).items() if count > 1]
    if duplicates:
        raise ValueError(f"{name} contains duplicate clips: {duplicates[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5)")

    root = args.data_root.expanduser().resolve()
    split_dir = root / "DADA2K_my_split"
    annotation_path = root / "annotation" / "full_anno.csv"
    official_train = split_dir / "official_training.txt"
    official_test = split_dir / "official_validation.txt"
    current_train = split_dir / "training.txt"
    current_validation = split_dir / "validation.txt"
    for path in (current_train, current_validation, annotation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not official_train.exists():
        shutil.copy2(current_train, official_train)
    if not official_test.exists():
        shutil.copy2(current_validation, official_test)

    source_train = read_lines(official_train)
    source_test = read_lines(official_test)
    assert_unique("official training", source_train)
    assert_unique("official test", source_test)
    overlap = sorted(set(source_train).intersection(source_test))
    if overlap:
        raise ValueError(f"official train/test leakage: {overlap[:10]}")

    strata = metadata_strata(source_train + source_test, annotation_path)
    grouped: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for item in source_train:
        grouped[strata[item]].append(item)

    train: list[str] = []
    validation: list[str] = []
    half_pool: list[str] = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda item: stable_order(item, args.seed))
        selected_count = max(1, round(len(items) * args.fraction))
        if len(items) >= 2:
            selected_count = max(2, selected_count)
        selected = items[: min(len(items), selected_count)]
        half_pool.extend(selected)
        if len(selected) == 1:
            train.extend(selected)
            continue
        validation_count = max(1, round(len(selected) * args.validation_fraction))
        validation_count = min(validation_count, len(selected) - 1)
        validation.extend(selected[:validation_count])
        train.extend(selected[validation_count:])

    train.sort()
    validation.sort()
    half_pool.sort()
    test = sorted(source_test)
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("generated splits are not source-disjoint")

    write_lines(split_dir / "half_training.txt", train)
    write_lines(split_dir / "validation.txt", validation)
    write_lines(split_dir / "test.txt", test)
    write_lines(split_dir / "half_pool.txt", half_pool)

    summary = {
        "seed": args.seed,
        "requested_training_fraction": args.fraction,
        "validation_fraction_within_selected_pool": args.validation_fraction,
        "policy": (
            "official training only -> deterministic stratified half pool -> train/validation; "
            "official validation held out unchanged as test"
        ),
        "stratification": ["accident category", "accident present", "day/night"],
        "counts": {
            "official_train": len(source_train),
            "selected_half_pool": len(half_pool),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "sha256": {
            "official_train": digest(source_train),
            "selected_half_pool": digest(half_pool),
            "train": digest(train),
            "validation": digest(validation),
            "test": digest(test),
        },
        "pairwise_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
    }
    manifest = split_dir / "split_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved split manifest: {manifest}")


if __name__ == "__main__":
    main()
