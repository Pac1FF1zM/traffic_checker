"""Download the traffic-adapted Simple-TAD checkpoints used in comparisons."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "tue-mps/simple-tad"
CHECKPOINTS = (
    (
        "models/Finetune_DoTA/simpletad_ft-dota_dapt-vm1-s_auroc.pth",
        ROOT / "weights" / "simpletad_ft-dota_dapt-vm1-s_auroc.pth",
        43_793_762,
    ),
    (
        "models/Finetune_DoTA/simpletad_ft-dota_dapt-vm1-b_auroc.pth",
        ROOT / "weights" / "simpletad_ft-dota_dapt-vm1-b_auroc.pth",
        172_489_826,
    ),
)


def main() -> None:
    for filename, target, expected_bytes in CHECKPOINTS:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == expected_bytes:
            print(f"Already downloaded: {target}")
            continue

        partial = target.with_suffix(target.suffix + ".partial")
        print(f"Downloading {expected_bytes / 1024**2:.1f} MiB from Hugging Face...")
        cached = Path(hf_hub_download(repo_id=REPO_ID, filename=filename))
        shutil.copy2(cached, partial)

        actual = partial.stat().st_size
        if actual != expected_bytes:
            raise RuntimeError(
                f"incomplete download: expected {expected_bytes} bytes, got {actual}"
            )
        shutil.move(str(partial), str(target))
        print(f"Saved: {target}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
