#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:?Usage: eval_videomae_b.sh CONFIG CHECKPOINT [OUTPUT_DIR]}"
CHECKPOINT_PATH="${2:?Usage: eval_videomae_b.sh CONFIG CHECKPOINT [OUTPUT_DIR]}"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

set -a
source "${CONFIG_PATH}"
set +a

absolute_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then printf '%s\n' "${value}"; else printf '%s\n' "${REPO_ROOT}/${value}"; fi
}

DATA_ROOT="$(absolute_path "${DATA_ROOT}")"
CHECKPOINT_PATH="$(absolute_path "${CHECKPOINT_PATH}")"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
EVAL_OUTPUT="${3:-${OUTPUT_ROOT}/eval_${RUN_ID}}"
EVAL_OUTPUT="$(absolute_path "${EVAL_OUTPUT}")"
mkdir -p "${EVAL_OUTPUT}"

export PYTHONPATH="${REPO_ROOT}/training_compat:${REPO_ROOT}/third_party/simple_tad${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}/third_party/simple_tad"
torchrun --standalone --nproc_per_node=1 run_frame_finetuning.py \
  --eval \
  --dist_eval \
  --model vit_base_patch16_224 \
  --data_set "${DATASET_KIND}" \
  --data_path "${DATA_ROOT}" \
  --finetune "${CHECKPOINT_PATH}" \
  --output_dir "${EVAL_OUTPUT}" \
  --log_dir "${EVAL_OUTPUT}" \
  --loss "${LOSS}" \
  --nb_classes 2 \
  --tubelet_size 2 \
  --batch_size "${BATCH_SIZE}" \
  --num_frames "${NUM_FRAMES}" \
  --view_fps "${VIEW_FPS}" \
  --input_size "${INPUT_SIZE}" \
  --short_side_size "${INPUT_SIZE}" \
  --sampling_rate_val "${VAL_SAMPLING_RATE}" \
  --num_workers "${NUM_WORKERS}" \
  --test_num_segment 1 \
  --test_num_crop 1 \
  --no_auto_resume \
  --no_flash_attn \
  --pin_mem \
  --seed "${SEED}"

echo "Evaluation saved to ${EVAL_OUTPUT}"
