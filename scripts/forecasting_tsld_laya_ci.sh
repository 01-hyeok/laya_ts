#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
PRETRAIN_DATA="tsld"
ARCH="laya"
LAYA_MODE="independent"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-none}"
DEFAULT_CHECKPOINT_DIR="./checkpoints/${PRETRAIN_DATA}_${ARCH}_${LAYA_MODE}_${CHANNEL_METADATA_MODE}"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT_DIR}/laya_ts_${PRETRAIN_DATA}_s_best.pt}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "⚠️ Pretrained checkpoint not found: $CHECKPOINT"
  echo "Expected default checkpoint under: $DEFAULT_CHECKPOINT_DIR"
  echo "Please provide a tsld CI checkpoint, or override CHECKPOINT=/path/to/laya_ts_${PRETRAIN_DATA}_s_best.pt"
  exit 1
fi

TARGET_DATASETS=("ETTm1" "ETTm2" "weather")
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
SEQ_LEN="${SEQ_LEN:-512}"

for DATA in "${TARGET_DATASETS[@]}"; do
  if [ "$DATA" = "ETTm1" ] || [ "$DATA" = "ETTm2" ]; then
    DATA_DIR="ETT-small"
  else
    DATA_DIR="$DATA"
  fi
  DATA_PATH="../Dataset/Time-Series-Library_dataset/${DATA_DIR}/${DATA}.csv"
  LOG_DIR="./runs/forecasting_${PRETRAIN_DATA}_to_${DATA}_${ARCH}_${LAYA_MODE}_${CHANNEL_METADATA_MODE}"
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
      --log_dir "$LOG_DIR/pred_${PRED_LEN}"
  done
done
