"""Download the recommended traffic-adapted Simple-TAD VideoMAE-B checkpoint."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "weights" / "simpletad_ft-dota_dapt-vm1-b_auroc.pth"
PARTIAL = TARGET.with_suffix(TARGET.suffix + ".partial")
REPO_ID = "tue-mps/simple-tad"
FILENAME = "models/Finetune_DoTA/simpletad_ft-dota_dapt-vm1-b_auroc.pth"
EXPECTED_BYTES = 172_489_826


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.is_file() and TARGET.stat().st_size == EXPECTED_BYTES:
        print(f"Already downloaded: {TARGET}")
        return

    print(f"Downloading {EXPECTED_BYTES / 1024**2:.1f} MiB from Hugging Face...")
    cached = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    shutil.copy2(cached, PARTIAL)

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
