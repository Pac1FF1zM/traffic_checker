"""Export three source-disjoint DADA held-out clips as browser-friendly MP4 files."""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ACCIDENT_COLUMN = "whether an accident occurred (1/0)"
LIGHT_COLUMN = "light(day,night)1-2"


@dataclass(frozen=True)
class Candidate:
    clip: str
    category: int
    video: int
    accident: int
    light: int
    archive: Path
    frames: int


def natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def read_test_ids(path: Path) -> dict[tuple[int, int], str]:
    clips = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines()]
    result: dict[tuple[int, int], str] = {}
    for clip in filter(None, clips):
        category, video = clip.split("/")
        result[(int(category), int(video))] = clip
    return result


def frame_names(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as bundle:
        return sorted(
            (
                name
                for name in bundle.namelist()
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            ),
            key=natural_key,
        )


def candidates(data_root: Path) -> list[Candidate]:
    split = data_root / "DADA2K_my_split" / "test.txt"
    annotation = data_root / "annotation" / "full_anno.csv"
    if not split.is_file():
        raise FileNotFoundError(split)
    if not annotation.is_file():
        raise FileNotFoundError(annotation)

    test_ids = read_test_ids(split)
    found: list[Candidate] = []
    with annotation.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["type"]), int(row["video"]))
            clip = test_ids.get(key)
            if clip is None:
                continue
            archive = data_root / "frames" / Path(clip) / "images.zip"
            if not archive.is_file():
                continue
            count = len(frame_names(archive))
            if count:
                found.append(
                    Candidate(
                        clip=clip,
                        category=key[0],
                        video=key[1],
                        accident=int(row[ACCIDENT_COLUMN]),
                        light=int(row[LIGHT_COLUMN]),
                        archive=archive,
                        frames=count,
                    )
                )
    return found


def choose_examples(items: list[Candidate]) -> list[tuple[str, Candidate]]:
    targets = [
        ("safe_control", lambda item: item.accident == 0),
        ("accident_day", lambda item: item.accident == 1 and item.light == 1),
        ("accident_night", lambda item: item.accident == 1 and item.light == 2),
    ]
    selected: list[tuple[str, Candidate]] = []
    used_categories: set[int] = set()
    for label, predicate in targets:
        matches = sorted((item for item in items if predicate(item)), key=lambda item: (-item.frames, item.clip))
        if not matches:
            raise RuntimeError(f"No held-out test clip is available for {label}")
        choice = next((item for item in matches if item.category not in used_categories), matches[0])
        selected.append((label, choice))
        used_categories.add(choice.category)
    return selected


def decode(bundle: zipfile.ZipFile, name: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(bundle.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {name}")
    return image


def export(candidate: Candidate, destination: Path, fps: float) -> int:
    names = frame_names(candidate.archive)
    with zipfile.ZipFile(candidate.archive) as bundle:
        first = decode(bundle, names[0])
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not create {destination}")
        try:
            for name in names:
                frame = decode(bundle, name)
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = choose_examples(candidates(data_root))

    manifest = {
        "source_split": str(data_root / "DADA2K_my_split" / "test.txt"),
        "selection": "held-out test only; one safe, one accident/day, one accident/night",
        "fps": args.fps,
        "clips": [],
    }
    for index, (label, candidate) in enumerate(selected, start=1):
        filename = f"{index:02d}_{label}_{candidate.category}-{candidate.video:03d}.mp4"
        destination = output_dir / filename
        written = export(candidate, destination, args.fps)
        record = {
            "label": label,
            "clip": candidate.clip,
            "accident_present": bool(candidate.accident),
            "lighting": "day" if candidate.light == 1 else "night",
            "frames": written,
            "duration_seconds": round(written / args.fps, 2),
            "file": str(destination),
        }
        manifest["clips"].append(record)
        print(json.dumps(record, ensure_ascii=False))

    manifest_path = output_dir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved 3 held-out demo clips to: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
