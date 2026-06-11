#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA="lotsa"
ARCH="laya"
VARIANT="${VARIANT:-s}"
LAYA_MODE="${LAYA_MODE:-mixer}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-text_stats_avg}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-concat_kv}"
CHANNEL_MIXER_RELATION_MODE="${CHANNEL_MIXER_RELATION_MODE:-none}"
STATS_METADATA_DIM="${STATS_METADATA_DIM:-384}"
LOTSA_SPLIT_MODE="${LOTSA_SPLIT_MODE:-official}"
LOTSA_SAMPLING_MODE="${LOTSA_SAMPLING_MODE:-official}"
LOTSA_PREPROCESSING_MODE="${LOTSA_PREPROCESSING_MODE:-official}"
LOTSA_SAMPLE_TIME_SERIES="${LOTSA_SAMPLE_TIME_SERIES:-proportional}"
LOTSA_SUBSET_SAMPLING="${LOTSA_SUBSET_SAMPLING:-official}"
LOTSA_MIN_PATCHES="${LOTSA_MIN_PATCHES:-2}"
LOTSA_MAX_CHANNEL="${LOTSA_MAX_CHANNEL:-}"
LOTSA_WINDOWS_PER_SERIES="${LOTSA_WINDOWS_PER_SERIES:-32}"

LOTSA_SUBSETS="${LOTSA_SUBSETS:-beijing_air_quality,HZMETRO,residential_pv_power,residential_load_power,china_air_quality}"
SEQ_LEN="${SEQ_LEN:-512}"
STRIDE="${STRIDE:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-5000}"
N_HEADS="${N_HEADS:-6}"
N_LAYERS="${N_LAYERS:-12}"
D_MODEL="${D_MODEL:-384}"
PROJ_DIM="${PROJ_DIM:-128}"
PREDICTOR_DEPTH="${PREDICTOR_DEPTH:-4}"
PREDICTOR_HEADS="${PREDICTOR_HEADS:-4}"
PATCH_SIZE="${PATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
MIN_LR="${MIN_LR:-1e-6}"

TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
TEXT_METADATA_CACHE_DIR="${TEXT_METADATA_CACHE_DIR:-./metadata_cache}"
LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW:-1}"
DEBUG_LOTSA="${DEBUG_LOTSA:-1}"
SAVE_ATTENTION_MAPS="${SAVE_ATTENTION_MAPS:-1}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/${DATA}_${ARCH}_mixer_concat_text_stats_avg}"
LOG_DIR="${LOG_DIR:-./runs/pretrain_${DATA}_${ARCH}_mixer_concat_text_stats_avg}"

export LAYA_TS_LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW}"
export LAYA_TS_DEBUG_LOTSA="${DEBUG_LOTSA}"

echo "🚀 Laya-TS Pretraining: ${DATA} (${ARCH} + mixer_concat_text_stats_avg)"
echo "📊 subsets: ${LOTSA_SUBSETS:-ALL}"
echo "🧩 split_mode: ${LOTSA_SPLIT_MODE}"
echo "🧩 sampling_mode: ${LOTSA_SAMPLING_MODE}"
echo "🧩 preprocessing_mode: ${LOTSA_PREPROCESSING_MODE}"
echo "🧩 sample_time_series: ${LOTSA_SAMPLE_TIME_SERIES}"
echo "🧩 subset_sampling: ${LOTSA_SUBSET_SAMPLING}"
echo "🧩 min_patches: ${LOTSA_MIN_PATCHES}, max_channel: ${LOTSA_MAX_CHANNEL:-none}"
echo "🧩 windows_per_series: ${LOTSA_WINDOWS_PER_SERIES}"
echo "🧩 debug_lotsa: ${DEBUG_LOTSA}"
echo "📝 metadata: mode=${CHANNEL_METADATA_MODE}, fusion=${METADATA_FUSION_MODE}, relation=${CHANNEL_MIXER_RELATION_MODE}, stats_dim=${STATS_METADATA_DIM}"
echo "📝 log_dir: ${LOG_DIR}"

EXTRA_ARGS=(
  --attention_map_tag mixer_concat_text_stats_avg
  --metadata_fusion_mode "${METADATA_FUSION_MODE}"
  --channel_mixer_relation_mode "${CHANNEL_MIXER_RELATION_MODE}"
)

if [ "${SAVE_ATTENTION_MAPS}" = "1" ]; then
  EXTRA_ARGS+=(--save_attention_maps)
fi
if [ -n "${LOTSA_MAX_CHANNEL}" ]; then
  EXTRA_ARGS+=(--lotsa_max_channel "${LOTSA_MAX_CHANNEL}")
fi

python -u "./train_pretrain.py" \
  --dataset_type "${DATA}" \
  --data_path "${LOTSA_SUBSETS}" \
  --variant "${VARIANT}" \
  --seq_len "${SEQ_LEN}" \
  --d_model "${D_MODEL}" \
  --patch_size "${PATCH_SIZE}" \
  --stride "${STRIDE}" \
  --batch_size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --val_interval "${VAL_INTERVAL}" \
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
  --stats_metadata_dim "${STATS_METADATA_DIM}" \
  --lotsa_split_mode "${LOTSA_SPLIT_MODE}" \
  --lotsa_sampling_mode "${LOTSA_SAMPLING_MODE}" \
  --lotsa_preprocessing_mode "${LOTSA_PREPROCESSING_MODE}" \
  --lotsa_sample_time_series "${LOTSA_SAMPLE_TIME_SERIES}" \
  --lotsa_subset_sampling "${LOTSA_SUBSET_SAMPLING}" \
  --lotsa_min_patches "${LOTSA_MIN_PATCHES}" \
  --lotsa_windows_per_series "${LOTSA_WINDOWS_PER_SERIES}" \
  --channel_mixer_type "${LAYA_MODE}" \
  --text_encoder_name_or_path "${TEXT_ENCODER_NAME}" \
  --text_metadata_cache_dir "${TEXT_METADATA_CACHE_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --log_dir "${LOG_DIR}" \
  "${EXTRA_ARGS[@]}"
