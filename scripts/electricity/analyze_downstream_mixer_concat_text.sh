#!/usr/bin/env bash
set -euo pipefail

cd /data/pjh_workspace/laya_ts

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_TAG="electricity_laya_mixer_concat_text"
REPRESENTATION_TYPE="${REPRESENTATION_TYPE:-auto}"
DATA="${DATA:-ETTm1}"
ROOT_PATH="${ROOT_PATH:-../Dataset/Time-Series-Library_dataset}"
DATA_PATH="${DATA_PATH:-}"
SEQ_LEN="${SEQ_LEN:-512}"
LABEL_LEN="${LABEL_LEN:-0}"
PRED_LEN="${PRED_LEN:-96}"
DATA_LOWER="$(printf '%s' "${DATA}" | tr '[:upper:]' '[:lower:]')"
DEFAULT_FORECASTING_CHECKPOINT_DIR="${DEFAULT_FORECASTING_CHECKPOINT_DIR:-./checkpoints/forecasting_${DATA_LOWER}_${PRED_LEN}}"
FORECASTING_CHECKPOINT="${FORECASTING_CHECKPOINT:-${DEFAULT_FORECASTING_CHECKPOINT_DIR}/${MODEL_TAG}_best.pt}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
ABLATION="${ABLATION:-zero}"
OUT_DIR="${OUT_DIR:-./analysis/linear_probe_ablation/mixer_concat_text_ETTm1_96}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

EXTRA_ARGS=()
if [ -n "${DATA_PATH}" ]; then
  EXTRA_ARGS+=(--data_path "${DATA_PATH}")
fi

python -u "./analyze_downstream.py" \
  --forecasting_checkpoint "${FORECASTING_CHECKPOINT}" \
  --representation_type "${REPRESENTATION_TYPE}" \
  --data "${DATA}" \
  --root_path "${ROOT_PATH}" \
  --seq_len "${SEQ_LEN}" \
  --label_len "${LABEL_LEN}" \
  --pred_len "${PRED_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --train_epochs "${TRAIN_EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --ablation "${ABLATION}" \
  --output_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
