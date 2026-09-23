#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_PYTHON="${LAB_PYTHON:-python3.10}"
LAB_VENV="${LAB_VENV:-${REPO_ROOT}/.venv-lab}"

cd "${REPO_ROOT}"
"${LAB_PYTHON}" -m venv "${LAB_VENV}"
source "${LAB_VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-training.txt

git submodule update --init --recursive
python scripts/download_temporal_weights.py

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check the NVIDIA driver and assigned GPU.")
props = torch.cuda.get_device_properties(0)
print(f"gpu={props.name} memory={props.total_memory / 1024**3:.2f} GiB capability={props.major}.{props.minor}")
PY

echo "Environment ready: ${LAB_VENV}"
echo "Next: edit configs/training/videomae_b_t4_dada.env and run scripts/lab/train_videomae_b_t4.sh"
