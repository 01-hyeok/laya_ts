#!/usr/bin/env bash
set -euo pipefail

source /data/pjh_workspace/ts-env/bin/activate
cd /data/pjh_workspace/laya_ts

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAIN_DATA="electricity"
ARCH="laya"
DEFAULT_CHECKPOINT_DIR="${DEFAULT_CHECKPOINT_DIR:-./checkpoints/${PRETRAIN_DATA}_${ARCH}_mixer_concat_stats}"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT_DIR}/laya_ts_${PRETRAIN_DATA}_s_best.pt}"
DATA="${DATA:-Electricity}"
ROOT_PATH="${ROOT_PATH:-../Dataset/long_term_forecast/electricity}"
DATA_PATH="${DATA_PATH:-electricity.csv}"
SPLIT="${SPLIT:-val}"
NUM_BATCHES="${NUM_BATCHES:-32}"
TOP_K="${TOP_K:-5}"
SEQ_LEN="${SEQ_LEN:-512}"
OUT_DIR="${OUT_DIR:-./analysis/mixer_structure/electricity_mixer_concat_stats}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "⚠️ Pretrained checkpoint not found: $CHECKPOINT"
  echo "Expected default checkpoint under: $DEFAULT_CHECKPOINT_DIR"
  echo "Please run laya_ts/scripts/electricity/pretrain_electricity_laya_mixer_concat_stats.sh first,"
  echo "or override CHECKPOINT=/path/to/laya_ts_${PRETRAIN_DATA}_s_best.pt"
  exit 1
fi

echo "🔎 Mixer structure analysis"
echo "   - checkpoint: ${CHECKPOINT}"
echo "   - data: ${DATA}"
echo "   - split: ${SPLIT}"
echo "   - num_batches: ${NUM_BATCHES}"
echo "   - output_dir: ${OUT_DIR}"

python -u "./analyze_mixer_structure.py" \
  --checkpoint "${CHECKPOINT}" \
  --data "${DATA}" \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --seq_len "${SEQ_LEN}" \
  --split "${SPLIT}" \
  --num_batches "${NUM_BATCHES}" \
  --top_k "${TOP_K}" \
  --output_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}"
