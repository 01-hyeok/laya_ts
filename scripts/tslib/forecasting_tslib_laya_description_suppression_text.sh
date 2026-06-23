#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAIN_DATA="tslib"
ARCH="laya"
LAYA_MODE="${LAYA_MODE:-mixer}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-text}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-attention_suppress_gate}"
CHANNEL_MIXER_RELATION_MODE="${CHANNEL_MIXER_RELATION_MODE:-none}"

DESCRIPTION_RELATION_METRIC="${DESCRIPTION_RELATION_METRIC:-cosine}"
DESCRIPTION_RELATION_LAMBDA_INIT="${DESCRIPTION_RELATION_LAMBDA_INIT:-1.0}"
DESCRIPTION_RELATION_GAMMA_INIT="${DESCRIPTION_RELATION_GAMMA_INIT:-1.0}"

DEFAULT_CHECKPOINT_DIR="./checkpoints/${PRETRAIN_DATA}_${ARCH}_attention_suppress_gate_text"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT_DIR}/laya_ts_${PRETRAIN_DATA}_s_best.pt}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "⚠️ Pretrained checkpoint not found: $CHECKPOINT"
  echo "Expected default checkpoint under: $DEFAULT_CHECKPOINT_DIR"
  echo "Please run laya_ts/scripts/pretrain_tslib_laya_description_suppression_text.sh first,"
  echo "or override CHECKPOINT=/path/to/laya_ts_${PRETRAIN_DATA}_s_best.pt"
  exit 1
fi

TARGET_DATASETS=("ETTm1" "ETTm2" "weather")
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
SEQ_LEN="${SEQ_LEN:-512}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
TEXT_METADATA_CACHE_DIR="${TEXT_METADATA_CACHE_DIR:-./metadata_cache}"
LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW:-1}"
SAVE_ATTENTION_MAPS="${SAVE_ATTENTION_MAPS:-1}"
NUM_ATTENTION_MAP_SAMPLES="${NUM_ATTENTION_MAP_SAMPLES:-3}"


export LAYA_TS_LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW}"

EXTRA_ARGS=(
  --metadata_fusion_mode "${METADATA_FUSION_MODE}"
  --channel_mixer_relation_mode "${CHANNEL_MIXER_RELATION_MODE}"
  --description_relation_metric "${DESCRIPTION_RELATION_METRIC}"
  --description_relation_lambda_init "${DESCRIPTION_RELATION_LAMBDA_INIT}"
  --description_relation_gamma_init "${DESCRIPTION_RELATION_GAMMA_INIT}"
)

if [ "${SAVE_ATTENTION_MAPS}" = "1" ]; then
  EXTRA_ARGS+=(--save_attention_maps)
  EXTRA_ARGS+=(--num_attention_map_samples "${NUM_ATTENTION_MAP_SAMPLES}")
fi

for DATA in "${TARGET_DATASETS[@]}"; do
  if [ "$DATA" = "ETTm1" ] || [ "$DATA" = "ETTm2" ]; then
    DATA_DIR="ETT-small"
  else
    DATA_DIR="$DATA"
  fi
  DATA_PATH="../Dataset/Time-Series-Library_dataset/${DATA_DIR}/${DATA}.csv"
  LOG_DIR="./runs/forecasting_${PRETRAIN_DATA}_to_${DATA}_${ARCH}_attention_suppress_gate_text"
  mkdir -p "$LOG_DIR"

  for PRED_LEN in "${PRED_LENGTHS[@]}"; do
    python -u "./run_forecasting.py" \
      --data_path "$DATA_PATH" \
      --dataset_type "$DATA" \
      --seq_len "$SEQ_LEN" \
      --pred_len "$PRED_LEN" \
      --checkpoint "$CHECKPOINT" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --epochs "$NUM_EPOCHS" \
      --num_workers 0 \
      --channel_mixer_type "$LAYA_MODE" \
      --channel_metadata_mode "$CHANNEL_METADATA_MODE" \
      --text_encoder_name_or_path "$TEXT_ENCODER_NAME" \
      --text_metadata_cache_dir "$TEXT_METADATA_CACHE_DIR" \
      --log_dir "$LOG_DIR/pred_${PRED_LEN}" \
      "${EXTRA_ARGS[@]}"
  done
done
