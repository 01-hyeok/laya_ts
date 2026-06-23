#!/usr/bin/env bash
set -euo pipefail

source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/NHNHOME/pjh_data/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

DATA="weather"
DATA_SLUG="weather"
DATA_PATH="${DATA_PATH:-../Dataset/Time-Series-Library_dataset/weather/weather.csv}"

ARCH="laya"
VARIANT="${VARIANT:-s}"
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

SEQ_LEN="${SEQ_LEN:-512}"
D_MODEL="${D_MODEL:-256}"
PATCH_SIZE="${PATCH_SIZE:-16}"
STRIDE="${STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-100}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-3}"
PROJ_DIM="${PROJ_DIM:-128}"
PREDICTOR_DEPTH="${PREDICTOR_DEPTH:-2}"
PREDICTOR_HEADS="${PREDICTOR_HEADS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
MIN_LR="${MIN_LR:-1e-6}"

SAVE_ATTENTION_MAPS="${SAVE_ATTENTION_MAPS:-0}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/${DATA_SLUG}_${ARCH}_ci_adapter}"
LOG_DIR="${LOG_DIR:-./runs/pretrain_${DATA_SLUG}_${ARCH}_ci_adapter}"

mkdir -p "${TEXT_METADATA_CACHE_DIR}"

echo "🚀 Laya-TS Pretraining: ${DATA} (${ARCH} + ci_adapter)"
echo "📊 data_path: ${DATA_PATH}"
echo "📝 metadata: mode=${CHANNEL_METADATA_MODE}, fusion=${METADATA_FUSION_MODE}"
echo "📝 relation_adapter: heads=${RELATION_NUM_HEADS}, dropout=${RELATION_DROPOUT}, metadata_dropout=${METADATA_DROPOUT}"
echo "📝 metadata_cache: ${TEXT_METADATA_CACHE_DIR}"
echo "📝 hf_cache: ${HF_HOME}"
echo "📝 log_dir: ${LOG_DIR}"

EXTRA_ARGS=(
  --attention_map_tag ci_adapter
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
fi

python -u "./train_pretrain.py" \
  --dataset_type "${DATA}" \
  --data_path "${DATA_PATH}" \
  --variant "${VARIANT}" \
  --seq_len "${SEQ_LEN}" \
  --d_model "${D_MODEL}" \
  --patch_size "${PATCH_SIZE}" \
  --stride "${STRIDE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --n_heads "${N_HEADS}" \
  --n_layers "${N_LAYERS}" \
  --proj_dim "${PROJ_DIM}" \
  --predictor_depth "${PREDICTOR_DEPTH}" \
  --predictor_heads "${PREDICTOR_HEADS}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --min_lr "${MIN_LR}" \
  --channel_metadata_mode "${CHANNEL_METADATA_MODE}" \
  --text_metadata_cache_dir "${TEXT_METADATA_CACHE_DIR}" \
  --channel_mixer_type "${LAYA_MODE}" \
  --save_dir "${SAVE_DIR}" \
  --log_dir "${LOG_DIR}" \
  "${EXTRA_ARGS[@]}"
