#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARCH="laya"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-random_linear_probe}"
LAYA_MODE="${LAYA_MODE:-independent}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-none}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-none}"
CHANNEL_MIXER_RELATION_MODE="${CHANNEL_MIXER_RELATION_MODE:-none}"
CHANNEL_MIXER_RELATION_SCALE_INIT="${CHANNEL_MIXER_RELATION_SCALE_INIT:-1.0}"
PATCHIFIER_MODE="${PATCHIFIER_MODE:-single}"
MULTISCALE_PATCH_SIZES="${MULTISCALE_PATCH_SIZES:-4,8,16,32}"
MULTISCALE_BASE_PATCH="${MULTISCALE_BASE_PATCH:-16}"
MULTISCALE_GATE_TEMPERATURE="${MULTISCALE_GATE_TEMPERATURE:-1.0}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
TEXT_METADATA_CACHE_DIR="${TEXT_METADATA_CACHE_DIR:-./metadata_cache}"
STATS_METADATA_DIM="${STATS_METADATA_DIM:-384}"
LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW:-1}"

TARGET_DATASETS=("ETTm1" "ETTm2" "weather")
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
SEQ_LEN="${SEQ_LEN:-512}"

export LAYA_TS_LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW}"

for DATA in "${TARGET_DATASETS[@]}"; do
  if [ "$DATA" = "ETTm1" ] || [ "$DATA" = "ETTm2" ]; then
    DATA_DIR="ETT-small"
  else
    DATA_DIR="$DATA"
  fi
  DATA_PATH="../Dataset/Time-Series-Library_dataset/${DATA_DIR}/${DATA}.csv"
  LOG_DIR="./runs/forecasting_random_to_${DATA}_${ARCH}_${EXPERIMENT_NAME}"
  mkdir -p "$LOG_DIR"

  for PRED_LEN in "${PRED_LENGTHS[@]}"; do
    python -u "./run_random_forecasting.py" \
      --data_path "$DATA_PATH" \
      --dataset_type "$DATA" \
      --seq_len "$SEQ_LEN" \
      --pred_len "$PRED_LEN" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --epochs "$NUM_EPOCHS" \
      --num_workers 0 \
      --channel_mixer_type "$LAYA_MODE" \
      --channel_metadata_mode "$CHANNEL_METADATA_MODE" \
      --metadata_fusion_mode "$METADATA_FUSION_MODE" \
      --channel_mixer_relation_mode "$CHANNEL_MIXER_RELATION_MODE" \
      --channel_mixer_relation_scale_init "$CHANNEL_MIXER_RELATION_SCALE_INIT" \
      --patchifier_mode "$PATCHIFIER_MODE" \
      --multiscale_patch_sizes "$MULTISCALE_PATCH_SIZES" \
      --multiscale_base_patch "$MULTISCALE_BASE_PATCH" \
      --multiscale_gate_temperature "$MULTISCALE_GATE_TEMPERATURE" \
      --text_encoder_name_or_path "$TEXT_ENCODER_NAME" \
      --text_metadata_cache_dir "$TEXT_METADATA_CACHE_DIR" \
      --stats_metadata_dim "$STATS_METADATA_DIM" \
      --log_dir "$LOG_DIR/pred_${PRED_LEN}"
  done
done
