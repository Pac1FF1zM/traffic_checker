"""Validate the laboratory GPU, checkpoint, dataset layout, and split isolation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WIUT_FILE_IDS = {
    "1kr9joda2wotw4gwkvprkdqfadnjnc1ns",
    "1hp8dyeqtyhswfm6qao9fpsrhlpmfrin_",
    "10cherecwzo3u-vk1cnnghax6eggy5mwj",
    "1aj-qsazvyjtlkhirvkkebq1d3gwnobrd",
}
REQUIRED_MODULES = (
    "cv2",
    "decord",
    "einops",
    "matplotlib",
    "natsort",
    "numpy",
    "pandas",
    "psutil",
    "scipy",
    "seaborn",
    "sklearn",
    "tensorboardX",
    "timm",
    "torchmetrics",
    "torchvision",
)


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_splits(train_path: Path, val_path: Path) -> dict[str, Any]:
    train = read_split(train_path)
    val = read_split(val_path)
    train_counts = Counter(train)
    val_counts = Counter(val)
    overlap = sorted(set(train).intersection(val))
    combined = [item.lower() for item in train + val]
    forbidden = sorted(
        item
        for item in combined
        if "wiut" in item or any(file_id in item for file_id in WIUT_FILE_IDS)
    )
    return {
        "train_count": len(train),
        "validation_count": len(val),
        "train_duplicates": sorted(item for item, count in train_counts.items() if count > 1),
        "validation_duplicates": sorted(item for item, count in val_counts.items() if count > 1),
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(val_path),
        "overlap": overlap,
        "forbidden_wiut_references": forbidden,
        "train_items": train,
        "validation_items": val,
    }


def inspect_zip(path: Path) -> int:
    with zipfile.ZipFile(path, "r") as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"corrupt member {bad!r} in {path}")
    return len(names)


def validate_dataset(kind: str, root: Path, full_check: bool) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if kind == "DoTA":
        train_path = root / "dataset" / "train_split.txt"
        val_path = root / "dataset" / "val_split.txt"
        annotations = root / "dataset" / "annotations"
    else:
        train_path = root / "DADA2K_my_split" / "training.txt"
        val_path = root / "DADA2K_my_split" / "validation.txt"
        annotations = root / "annotation" / "full_anno.csv"

    for required in (train_path, val_path, annotations, root / "frames"):
        if not required.exists():
            errors.append(f"missing dataset path: {required}")
    if errors:
        return {}, errors

    audit = audit_splits(train_path, val_path)
    if not audit["train_items"] or not audit["validation_items"]:
        errors.append("training and validation splits must both be non-empty")
    if audit["overlap"]:
        errors.append(
            f"train/validation leakage: {len(audit['overlap'])} source videos overlap"
        )
    if audit["train_duplicates"] or audit["validation_duplicates"]:
        errors.append("duplicate source videos found inside a split")
    if audit["forbidden_wiut_references"]:
        errors.append("WIUT sample references found in external training splits")

    if full_check:
        items = audit["train_items"] + audit["validation_items"]
    else:
        items = audit["train_items"][:6] + audit["validation_items"][:6]
    checked: list[dict[str, Any]] = []
    for clip in items:
        frame_zip = root / "frames" / clip / "images.zip"
        if not frame_zip.is_file():
            errors.append(f"missing frame archive: {frame_zip}")
            continue
        try:
            member_count = inspect_zip(frame_zip)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        if member_count < 16:
            errors.append(f"too few frames ({member_count}) in {frame_zip}")
        checked.append({"clip": clip, "archive": str(frame_zip), "members": member_count})

        if kind == "DoTA":
            annotation = root / "dataset" / "annotations" / f"{clip}.json"
            if not annotation.is_file():
                errors.append(f"missing annotation: {annotation}")

    audit.pop("train_items", None)
    audit.pop("validation_items", None)
    return {"kind": kind, "root": str(root), "splits": audit, "archives_checked": checked}, errors


def inspect_environment(skip_cuda: bool, skip_version_check: bool) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        errors.append(f"missing Python modules: {', '.join(missing)}")

    import torch

    if not skip_version_check and not torch.__version__.startswith("2.5.1"):
        errors.append(f"expected torch 2.5.1, found {torch.__version__}")
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "required_modules_missing": missing,
    }
    if skip_cuda:
        info["cuda_check_skipped"] = True
        return info, errors
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable; the T4 training profile requires an NVIDIA GPU")
        return info, errors

    props = torch.cuda.get_device_properties(0)
    memory_gib = props.total_memory / 1024**3
    info["gpu"] = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "memory_gib": memory_gib,
        "count": torch.cuda.device_count(),
    }
    if memory_gib < 14.0:
        errors.append(f"GPU has only {memory_gib:.2f} GiB; this profile expects at least 14 GiB")
    return info, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-kind", choices=("DoTA", "DADA2K"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-check", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true", help="CI only")
    parser.add_argument("--skip-version-check", action="store_true", help="CI only")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    errors: list[str] = []

    wiut_root = (ROOT / "data" / "wiut_blind").resolve()
    if data_root == wiut_root or wiut_root in data_root.parents:
        errors.append("refusing to train on data/wiut_blind")
    if not checkpoint.is_file():
        errors.append(f"checkpoint not found: {checkpoint}")

    environment, environment_errors = inspect_environment(
        args.skip_cuda, args.skip_version_check
    )
    dataset, dataset_errors = validate_dataset(args.dataset_kind, data_root, args.full_check)
    errors.extend(environment_errors)
    errors.extend(dataset_errors)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not errors else "failed",
        "leakage_policy": "external dataset only; source-video-disjoint train/validation; no WIUT samples",
        "environment": environment,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "sha256": file_sha256(checkpoint) if checkpoint.is_file() else None,
        },
        "dataset": dataset,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
