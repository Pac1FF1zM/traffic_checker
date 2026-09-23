from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "weights" / "yolo11n.pt"
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        print(f"Downloading {URL} -> {TARGET}")
        urllib.request.urlretrieve(URL, TARGET)
    print(f"weights: {TARGET} ({TARGET.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"sha256: {sha256(TARGET)}")


if __name__ == "__main__":
    main()
