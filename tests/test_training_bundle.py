from __future__ import annotations

import tempfile
import zipfile
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lab.export_videomae_b_checkpoint import extract_state_dict
from scripts.lab.preflight_training import audit_splits, validate_dataset


def write_zip(path: Path, frames: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(frames):
            archive.writestr(f"frame_{index:04d}.jpg", b"test")


def test_split_audit_detects_source_leakage() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        train = root / "train.txt"
        val = root / "val.txt"
        train.write_text("1/001\n1/002\n", encoding="utf-8")
        val.write_text("1/002\n2/001\n", encoding="utf-8")
        report = audit_splits(train, val)
        assert report["overlap"] == ["1/002"]


def test_dada_layout_passes_and_checks_archives() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        split_root = root / "DADA2K_my_split"
        split_root.mkdir(parents=True)
        (split_root / "training.txt").write_text("1/001\n", encoding="utf-8")
        (split_root / "validation.txt").write_text("2/001\n", encoding="utf-8")
        annotation = root / "annotation"
        annotation.mkdir()
        (annotation / "full_anno.csv").write_text("clip,label\n", encoding="utf-8")
        write_zip(root / "frames" / "1" / "001" / "images.zip")
        write_zip(root / "frames" / "2" / "001" / "images.zip")

        report, errors = validate_dataset("DADA2K", root, full_check=True)
        assert not errors
        assert report["splits"]["train_count"] == 1
        assert len(report["archives_checked"]) == 2


def test_exporter_unwraps_and_removes_ddp_prefix() -> None:
    tensor = torch.ones(2)
    state = extract_state_dict({"model": {"module.head.bias": tensor}, "epoch": 3})
    assert list(state) == ["head.bias"]
    assert torch.equal(state["head.bias"], tensor)


if __name__ == "__main__":
    test_split_audit_detects_source_leakage()
    test_dada_layout_passes_and_checks_archives()
    test_exporter_unwraps_and_removes_ddp_prefix()
    print("training bundle tests passed")
