#!/usr/bin/env bash
set -euo pipefail

source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/NHNHOME/pjh_data/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

PRETRAIN_DATA="ETTm1"
PRETRAIN_DATA_SLUG="ettm1"
ARCH="laya"
LAYA_MODE="${LAYA_MODE:-ci_adapter}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-text}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-none}"

RELATION_NUM_HEADS="${RELATION_NUM_HEADS:-4}"
RELATION_DROPOUT="${RELATION_DROPOUT:-0.1}"
RELATION_SCALE_INIT="${RELATION_SCALE_INIT:-1e-3}"
METADATA_SCALE_INIT="${METADATA_SCALE_INIT:-1e-3}"
METADATA_DROPOUT="${METADATA_DROPOUT:-0.0}"
USE_METADATA_BIAS="${USE_METADATA_BIAS:-1}"
USE_METADATA_GATE="${USE_METADATA_GATE:-1}"
TEXT_METADATA_CACHE_DIR="${TEXT_METADATA_CACHE_DIR:-/NHNHOME/pjh_data/laya_ts_metadata_cache}"

DEFAULT_CHECKPOINT_DIR="./checkpoints/${PRETRAIN_DATA_SLUG}_${ARCH}_ci_adapter"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT_DIR}/laya_ts_${PRETRAIN_DATA}_s_best.pt}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "⚠️ Pretrained checkpoint not found: $CHECKPOINT"
  echo "Expected default checkpoint under: $DEFAULT_CHECKPOINT_DIR"
  echo "Please run laya_ts/scripts/ettm1/pretrain_ettm1_laya_ci_adapter.sh first,"
  echo "or override CHECKPOINT=/path/to/laya_ts_${PRETRAIN_DATA}_s_best.pt"
  exit 1
fi

DATA="ETTm1"
DATA_PATH="${DATA_PATH:-../Dataset/Time-Series-Library_dataset/ETT-small/ETTm1.csv}"
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
SEQ_LEN="${SEQ_LEN:-512}"
SAVE_ATTENTION_MAPS="${SAVE_ATTENTION_MAPS:-0}"
NUM_ATTENTION_MAP_SAMPLES="${NUM_ATTENTION_MAP_SAMPLES:-3}"
LOG_DIR="${LOG_DIR:-./runs/forecasting_${PRETRAIN_DATA_SLUG}_to_${PRETRAIN_DATA_SLUG}_${ARCH}_ci_adapter}"

mkdir -p "${TEXT_METADATA_CACHE_DIR}" "${LOG_DIR}"

EXTRA_ARGS=(
  --metadata_fusion_mode "${METADATA_FUSION_MODE}"
  --use_relation_adapter
  --relation_num_heads "${RELATION_NUM_HEADS}"
  --relation_dropout "${RELATION_DROPOUT}"
  --relation_scale_init "${RELATION_SCALE_INIT}"
  --metadata_scale_init "${METADATA_SCALE_INIT}"
  --metadata_dropout "${METADATA_DROPOUT}"
  --relation_adapter_position post_encoder
)

if [ "${USE_METADATA_BIAS}" = "1" ]; then
  EXTRA_ARGS+=(--use_metadata_bias)
else
  EXTRA_ARGS+=(--no-use_metadata_bias)
fi

if [ "${USE_METADATA_GATE}" = "1" ]; then
  EXTRA_ARGS+=(--use_metadata_gate)
else
  EXTRA_ARGS+=(--no-use_metadata_gate)
fi

if [ "${SAVE_ATTENTION_MAPS}" = "1" ]; then
  EXTRA_ARGS+=(--save_attention_maps)
  EXTRA_ARGS+=(--num_attention_map_samples "${NUM_ATTENTION_MAP_SAMPLES}")
fi

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
    --text_metadata_cache_dir "${TEXT_METADATA_CACHE_DIR}" \
    --log_dir "${LOG_DIR}/pred_${PRED_LEN}" \
    "${EXTRA_ARGS[@]}"
done
