#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRETRAIN_DATA="weather"
PRETRAIN_DATA_SLUG="weather"
ARCH="laya"
LAYA_MODE="${LAYA_MODE:-independent}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-none}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-none}"

DEFAULT_CHECKPOINT_DIR="./checkpoints/${PRETRAIN_DATA_SLUG}_${ARCH}_ci_none"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT_DIR}/laya_ts_${PRETRAIN_DATA}_s_best.pt}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "⚠️ Pretrained checkpoint not found: $CHECKPOINT"
  echo "Expected default checkpoint under: $DEFAULT_CHECKPOINT_DIR"
  echo "Please run laya_ts/scripts/weather/pretrain_weather_laya_ci.sh first,"
  echo "or override CHECKPOINT=/path/to/laya_ts_${PRETRAIN_DATA}_s_best.pt"
  exit 1
fi

DATA="weather"
DATA_PATH="${DATA_PATH:-../Dataset/Time-Series-Library_dataset/weather/weather.csv}"
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
SEQ_LEN="${SEQ_LEN:-512}"
SAVE_ATTENTION_MAPS="${SAVE_ATTENTION_MAPS:-0}"
NUM_ATTENTION_MAP_SAMPLES="${NUM_ATTENTION_MAP_SAMPLES:-3}"
LOG_DIR="${LOG_DIR:-./runs/forecasting_${PRETRAIN_DATA_SLUG}_to_${PRETRAIN_DATA_SLUG}_${ARCH}_ci_none}"

EXTRA_ARGS=(
  --metadata_fusion_mode "${METADATA_FUSION_MODE}"
)

if [ "${SAVE_ATTENTION_MAPS}" = "1" ]; then
  EXTRA_ARGS+=(--save_attention_maps)
  EXTRA_ARGS+=(--num_attention_map_samples "${NUM_ATTENTION_MAP_SAMPLES}")
fi

mkdir -p "${LOG_DIR}"

for PRED_LEN in "${PRED_LENGTHS[@]}"; do
  python -u "./run_forecasting.py" \
    --data_path "${DATA_PATH}" \
    --dataset_type "${DATA}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --checkpoint "${CHECKPOINT}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --epochs "${NUM_EPOCHS}" \
    --num_workers 0 \
    --channel_mixer_type "${LAYA_MODE}" \
    --channel_metadata_mode "${CHANNEL_METADATA_MODE}" \
    --log_dir "${LOG_DIR}/pred_${PRED_LEN}" \
    "${EXTRA_ARGS[@]}"
done
