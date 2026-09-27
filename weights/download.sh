#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/download_weights.py
git submodule update --init --recursive

TARGET="weights/simpletad_ft-dota_dapt-vm1-s_auroc.pth"
if [[ -f "$TARGET" ]]; then
  echo "Using packaged temporal checkpoint: $TARGET"
else
  echo "Final Team404 checkpoint is absent; installing the public Simple-TAD DoTA fallback."
  python - <<'PY'
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

root = Path.cwd()
target = root / "weights" / "simpletad_ft-dota_dapt-vm1-s_auroc.pth"
cached = Path(hf_hub_download(
    repo_id="tue-mps/simple-tad",
    filename="models/Finetune_DoTA/simpletad_ft-dota_dapt-vm1-s_auroc.pth",
))
target.parent.mkdir(parents=True, exist_ok=True)
partial = target.with_suffix(target.suffix + ".partial")
shutil.copy2(cached, partial)
if partial.stat().st_size != 43_793_762:
    raise SystemExit(f"unexpected checkpoint size: {partial.stat().st_size}")
partial.replace(target)
print(f"Saved fallback checkpoint: {target}")
PY
fi

python - <<'PY'
from pathlib import Path
required = [
    Path("weights/yolo11n.pt"),
    Path("weights/simpletad_ft-dota_dapt-vm1-s_auroc.pth"),
    Path("third_party/simple_tad/run_inference_simple.py"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("missing required offline assets: " + ", ".join(missing))
print("Offline assets are ready.")
PY
