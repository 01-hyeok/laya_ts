#!/usr/bin/env bash
set -euo pipefail

source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PRETRAIN_DATA="electricity"
ARCH="laya"
LAYA_MODE="${LAYA_MODE:-mixer}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-stats}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-concat_kv}"
CHANNEL_MIXER_RELATION_MODE="${CHANNEL_MIXER_RELATION_MODE:-none}"
STATS_METADATA_DIM="${STATS_METADATA_DIM:-384}"

DEFAULT_CHECKPOINT_DIR="./checkpoints/${PRETRAIN_DATA}_${ARCH}_mixer_concat_stats"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT_DIR}/laya_ts_${PRETRAIN_DATA}_s_best.pt}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "⚠️ Pretrained checkpoint not found: $CHECKPOINT"
  echo "Expected default checkpoint under: $DEFAULT_CHECKPOINT_DIR"
  echo "Please run laya_ts/scripts/pretrain_electricity_laya_mixer_concat_stats.sh first,"
  echo "or override CHECKPOINT=/path/to/laya_ts_${PRETRAIN_DATA}_s_best.pt"
  exit 1
fi

TARGET_DATASETS=("ETTm1" "ETTm2" "weather")
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
SEQ_LEN="${SEQ_LEN:-512}"
SAVE_ATTENTION_MAPS="${SAVE_ATTENTION_MAPS:-1}"
NUM_ATTENTION_MAP_SAMPLES="${NUM_ATTENTION_MAP_SAMPLES:-3}"



EXTRA_ARGS=(
  --metadata_fusion_mode "${METADATA_FUSION_MODE}"
  --channel_mixer_relation_mode "${CHANNEL_MIXER_RELATION_MODE}"
  --stats_metadata_dim "${STATS_METADATA_DIM}"
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
  LOG_DIR="./runs/forecasting_${PRETRAIN_DATA}_to_${DATA}_${ARCH}_mixer_concat_stats"
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
      --log_dir "$LOG_DIR/pred_${PRED_LEN}" \
      "${EXTRA_ARGS[@]}"
  done
done

