#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/training/videomae_b_t4_dada.env}"

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

required=(
  DATASET_KIND DATA_ROOT OUTPUT_ROOT INIT_CHECKPOINT BATCH_SIZE UPDATE_FREQ
  NUM_WORKERS NUM_SAMPLES_PER_EPOCH FROZEN_EPOCHS FULL_EPOCHS FREEZE_BLOCKS
  WARMUP_EPOCHS_FROZEN WARMUP_EPOCHS_FULL BASE_LR MIN_LR WEIGHT_DECAY
  LAYER_DECAY DROP_PATH CLIP_GRAD LOSS SEED NUM_FRAMES VIEW_FPS INPUT_SIZE
  TRAIN_SAMPLING_RATE VAL_SAMPLING_RATE
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing ${name} in ${CONFIG_PATH}" >&2
    exit 2
  fi
done

absolute_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${REPO_ROOT}/${value}"
  fi
}

DATA_ROOT="$(absolute_path "${DATA_ROOT}")"
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT}")"
INIT_CHECKPOINT="$(absolute_path "${INIT_CHECKPOINT}")"
mkdir -p "${OUTPUT_ROOT}"

export PYTHONPATH="${REPO_ROOT}/training_compat:${REPO_ROOT}/third_party/simple_tad${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python "${REPO_ROOT}/scripts/lab/preflight_training.py" \
  --dataset-kind "${DATASET_KIND}" \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${INIT_CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/preflight.json" \
  ${PREFLIGHT_FULL_CHECK:+--full-check}

run_stage() {
  local stage_output="$1"
  local init_checkpoint="$2"
  local epochs="$3"
  local warmup_epochs="$4"
  local freeze_spec="$5"
  local freeze_args=()
  local resume_args=(--no_auto_resume)
  if [[ -n "${freeze_spec}" ]]; then
    freeze_args=(--freeze_layers "${freeze_spec}")
  fi
  if [[ -f "${stage_output}/checkpoint-last.pth" ]]; then
    resume_args=(--no_auto_resume --resume "${stage_output}/checkpoint-last.pth")
  fi

  mkdir -p "${stage_output}"
  cd "${REPO_ROOT}/third_party/simple_tad"
  torchrun --standalone --nproc_per_node=1 run_frame_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set "${DATASET_KIND}" \
    --data_path "${DATA_ROOT}" \
    --finetune "${init_checkpoint}" \
    --output_dir "${stage_output}" \
    --log_dir "${stage_output}" \
    --loss "${LOSS}" \
    --nb_classes 2 \
    --tubelet_size 2 \
    --batch_size "${BATCH_SIZE}" \
    --update_freq "${UPDATE_FREQ}" \
    --num_sample 1 \
    --num_frames "${NUM_FRAMES}" \
    --view_fps "${VIEW_FPS}" \
    --input_size "${INPUT_SIZE}" \
    --short_side_size "${INPUT_SIZE}" \
    --sampling_rate "${TRAIN_SAMPLING_RATE}" \
    --sampling_rate_val "${VAL_SAMPLING_RATE}" \
    --nb_samples_per_epoch "${NUM_SAMPLES_PER_EPOCH}" \
    --num_workers "${NUM_WORKERS}" \
    --epochs "${epochs}" \
    --warmup_epochs "${warmup_epochs}" \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --lr "${BASE_LR}" \
    --min_lr "${MIN_LR}" \
    --warmup_lr 1e-6 \
    --weight_decay "${WEIGHT_DECAY}" \
    --layer_decay "${LAYER_DECAY}" \
    --drop_path "${DROP_PATH}" \
    --clip_grad "${CLIP_GRAD}" \
    --aa rand-m6-n3-mstd0.5-inc1 \
    --reprob 0.1 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --use_checkpoint \
    --no_flash_attn \
    --pin_mem \
    --dist_eval \
    --save_ckpt \
    "${resume_args[@]}" \
    --seed "${SEED}" \
    "${freeze_args[@]}"
}

stage1_dir="${OUTPUT_ROOT}/stage1_frozen"
stage2_dir="${OUTPUT_ROOT}/stage2_full"
stage2_init="${INIT_CHECKPOINT}"

if (( FROZEN_EPOCHS > 0 )); then
  run_stage \
    "${stage1_dir}" \
    "${INIT_CHECKPOINT}" \
    "${FROZEN_EPOCHS}" \
    "${WARMUP_EPOCHS_FROZEN}" \
    "first N blocks;${FREEZE_BLOCKS}"
  if [[ -f "${stage1_dir}/checkpoint-bestauroc.pth" ]]; then
    stage2_init="${stage1_dir}/checkpoint-bestauroc.pth"
  else
    stage2_init="${stage1_dir}/checkpoint-last.pth"
  fi
fi

run_stage \
  "${stage2_dir}" \
  "${stage2_init}" \
  "${FULL_EPOCHS}" \
  "${WARMUP_EPOCHS_FULL}" \
  ""

echo "Training complete. Preferred checkpoint: ${stage2_dir}/checkpoint-bestauroc.pth"
echo "Validate it with: scripts/lab/eval_videomae_b.sh ${CONFIG_PATH} ${stage2_dir}/checkpoint-bestauroc.pth"
