from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "weights" / "yolo11n.pt"
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
EXPECTED_BYTES = 5_613_764
EXPECTED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    valid = (
        TARGET.is_file()
        and TARGET.stat().st_size == EXPECTED_BYTES
        and sha256(TARGET) == EXPECTED_SHA256
    )
    if not valid:
        partial = TARGET.with_suffix(TARGET.suffix + ".partial")
        print(f"Downloading {URL} -> {TARGET}")
        urllib.request.urlretrieve(URL, partial)
        actual_sha256 = sha256(partial)
        if partial.stat().st_size != EXPECTED_BYTES or actual_sha256 != EXPECTED_SHA256:
            partial.unlink(missing_ok=True)
            raise RuntimeError("downloaded YOLO11n checkpoint failed size/SHA-256 validation")
        partial.replace(TARGET)
    print(f"weights: {TARGET} ({TARGET.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"sha256: {sha256(TARGET)}")


if __name__ == "__main__":
    main()
