from __future__ import annotations

import tempfile
import zipfile
import sys
import json
import subprocess
import csv
import struct
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lab.export_videomae_b_checkpoint import extract_state_dict
from scripts.lab.preflight_training import audit_splits, read_split, validate_dataset
from scripts.download_dada_selected import clip_from_archive_name, parse_zip64_extra
from scripts.download_dada_missing_hf import parse_hf_member


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


def test_split_audit_detects_test_leakage() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        train = root / "train.txt"
        val = root / "val.txt"
        test = root / "test.txt"
        train.write_text("1/001\n1/002\n", encoding="utf-8")
        val.write_text("2/001\n", encoding="utf-8")
        test.write_text("1/002\n3/001\n", encoding="utf-8")
        report = audit_splits(train, val, test)
        assert report["train_validation_overlap"] == []
        assert report["train_test_overlap"] == ["1/002"]
        assert report["validation_test_overlap"] == []


def test_dada_layout_passes_and_checks_archives() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        split_root = root / "DADA2K_my_split"
        split_root.mkdir(parents=True)
        (split_root / "training.txt").write_text("1/001\n", encoding="utf-8")
        (split_root / "validation.txt").write_text("2/001\n", encoding="utf-8")
        (split_root / "test.txt").write_text("3/001\n", encoding="utf-8")
        annotation = root / "annotation"
        annotation.mkdir()
        (annotation / "full_anno.csv").write_text("clip,label\n", encoding="utf-8")
        write_zip(root / "frames" / "1" / "001" / "images.zip")
        write_zip(root / "frames" / "2" / "001" / "images.zip")
        write_zip(root / "frames" / "3" / "001" / "images.zip")

        report, errors = validate_dataset(
            "DADA2K", root, full_check=True, test_split="test.txt"
        )
        assert not errors
        assert report["splits"]["train_count"] == 1
        assert report["splits"]["test_count"] == 1
        assert len(report["archives_checked"]) == 3


def test_split_generator_is_disjoint_and_deterministic() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        split_root = root / "DADA2K_my_split"
        annotation_root = root / "annotation"
        split_root.mkdir(parents=True)
        annotation_root.mkdir()
        train = [f"{cat}/{idx:03d}" for cat in (1, 2) for idx in range(1, 9)]
        test = [f"{cat}/{idx:03d}" for cat in (1, 2) for idx in range(20, 23)]
        (split_root / "training.txt").write_text(
            "".join(f"{item}\n" for item in train), encoding="utf-8"
        )
        (split_root / "validation.txt").write_text(
            "".join(f"{item}\n" for item in test), encoding="utf-8"
        )
        rows = []
        for item in train + test:
            cat, video = item.split("/")
            rows.append(
                {
                    "type": int(cat),
                    "video": int(video),
                    "whether an accident occurred (1/0)": int(video) % 2,
                    "light(day,night)1-2": 1 + int(video) % 2,
                }
            )
        with (annotation_root / "full_anno.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dada_half_splits.py"
        command = [sys.executable, str(script), "--data-root", str(root), "--seed", "42"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        first = json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))
        subprocess.run(command, check=True, capture_output=True, text=True)
        second = json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))
        generated_train = set(read_split(split_root / "half_training.txt"))
        generated_val = set(read_split(split_root / "validation.txt"))
        generated_test = set(read_split(split_root / "test.txt"))
        assert first == second
        assert not generated_train & generated_val
        assert not generated_train & generated_test
        assert not generated_val & generated_test
        assert generated_test.issubset(set(test))
        assert generated_test
        assert first["counts"]["official_validation"] == len(test)
        assert first["counts"]["selected_dataset_clips"] == len(
            set(read_split(split_root / "selected_dataset_clips.txt"))
        )


def test_exporter_unwraps_and_removes_ddp_prefix() -> None:
    tensor = torch.ones(2)
    state = extract_state_dict({"model": {"module.head.bias": tensor}, "epoch": 3})
    assert list(state) == ["head.bias"]
    assert torch.equal(state["head.bias"], tensor)


def test_selective_dada_downloader_parses_archive_metadata() -> None:
    assert clip_from_archive_name("DADA2000/8/002/images/0134.png") == (
        "8/002",
        "0134.png",
    )
    assert clip_from_archive_name("DADA2000/8/002/fixation/0134.png") is None
    extra = struct.pack("<HHQ", 0x0001, 8, 5_000_000_000)
    assert parse_zip64_extra(extra, 10, 9, 0xFFFFFFFF, 2) == (
        10,
        9,
        5_000_000_000,
        2,
    )
    assert parse_hf_member(
        "Origin/DADA2000/DADA2000/8/002/images/0134.png"
    ) == ("8/002", "0134.png")
    assert parse_hf_member(
        "Origin/DADA2000/DADA2000/8/002/fixation/0134.png"
    ) == ("8/002", None)


if __name__ == "__main__":
    test_split_audit_detects_source_leakage()
    test_split_audit_detects_test_leakage()
    test_dada_layout_passes_and_checks_archives()
    test_split_generator_is_disjoint_and_deterministic()
    test_exporter_unwraps_and_removes_ddp_prefix()
    test_selective_dada_downloader_parses_archive_metadata()
    print("training bundle tests passed")
