"""Download the recommended traffic-adapted Simple-TAD VideoMAE-B checkpoint."""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "weights" / "simpletad_ft-dota_dapt-vm1-b_auroc.pth"
PARTIAL = TARGET.with_suffix(TARGET.suffix + ".partial")
URL = (
    "https://huggingface.co/tue-mps/simple-tad/resolve/main/models/Finetune_DoTA/"
    "simpletad_ft-dota_dapt-vm1-b_auroc.pth"
)
EXPECTED_BYTES = 172_489_826


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.is_file() and TARGET.stat().st_size == EXPECTED_BYTES:
        print(f"Already downloaded: {TARGET}")
        return

    print(f"Downloading {EXPECTED_BYTES / 1024**2:.1f} MiB from Hugging Face...")
    request = urllib.request.Request(URL, headers={"User-Agent": "traffic-checker/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, PARTIAL.open("wb") as output:
        downloaded = 0
        next_report = 10
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            downloaded += len(block)
            percent = int(downloaded * 100 / EXPECTED_BYTES)
            if percent >= next_report:
                print(f"  {min(percent, 100)}%")
                next_report += 10

    actual = PARTIAL.stat().st_size
    if actual != EXPECTED_BYTES:
        raise RuntimeError(f"incomplete download: expected {EXPECTED_BYTES} bytes, got {actual}")
    shutil.move(str(PARTIAL), str(TARGET))
    print(f"Saved: {TARGET}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
