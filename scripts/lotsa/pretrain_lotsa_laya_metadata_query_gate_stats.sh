#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA="lotsa"
ARCH="laya"
VARIANT="${VARIANT:-s}"
LAYA_MODE="${LAYA_MODE:-mixer}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-stats}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-none}"
CHANNEL_MIXER_RELATION_MODE="${CHANNEL_MIXER_RELATION_MODE:-metadata_query_gate}"
CHANNEL_MIXER_RELATION_SCALE_INIT="${CHANNEL_MIXER_RELATION_SCALE_INIT:-1.0}"
STATS_METADATA_DIM="${STATS_METADATA_DIM:-384}"
LOTSA_SPLIT_MODE="${LOTSA_SPLIT_MODE:-temporal_70_10_20}"
LOTSA_SAMPLING_MODE="${LOTSA_SAMPLING_MODE:-sliding_window}"
LOTSA_PREPROCESSING_MODE="${LOTSA_PREPROCESSING_MODE:-standardize}"
LOTSA_SAMPLE_TIME_SERIES="${LOTSA_SAMPLE_TIME_SERIES:-proportional}"
LOTSA_MIN_PATCHES="${LOTSA_MIN_PATCHES:-2}"
LOTSA_MAX_DIM="${LOTSA_MAX_DIM:-128}"

LOTSA_SUBSETS="${LOTSA_SUBSETS:-beijing_air_quality,HZMETRO,residential_pv_power,residential_load_power,china_air_quality}"
SEQ_LEN="${SEQ_LEN:-512}"
STRIDE="${STRIDE:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-5000}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-3}"
D_MODEL="${D_MODEL:-128}"
PROJ_DIM="${PROJ_DIM:-256}"
PATCH_SIZE="${PATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
MIN_LR="${MIN_LR:-1e-6}"
DEBUG_LOTSA="${DEBUG_LOTSA:-1}"
SAVE_ATTENTION_MAPS="${SAVE_ATTENTION_MAPS:-1}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/${DATA}_${ARCH}_metadata_query_gate_stats}"
LOG_DIR="${LOG_DIR:-./runs/pretrain_${DATA}_${ARCH}_metadata_query_gate_stats}"

export LAYA_TS_DEBUG_LOTSA="${DEBUG_LOTSA}"


echo "🚀 Laya-TS Pretraining: ${DATA} (${ARCH} + metadata_query_gate_stats)"
echo "📊 subsets: ${LOTSA_SUBSETS:-ALL}"
echo "🧩 split_mode: ${LOTSA_SPLIT_MODE}"
echo "🧩 sampling_mode: ${LOTSA_SAMPLING_MODE}"
echo "🧩 preprocessing_mode: ${LOTSA_PREPROCESSING_MODE}"
echo "🧩 sample_time_series: ${LOTSA_SAMPLE_TIME_SERIES}"
echo "🧩 min_patches: ${LOTSA_MIN_PATCHES}, max_dim: ${LOTSA_MAX_DIM}"
echo "🧩 debug_lotsa: ${DEBUG_LOTSA}"
echo "📝 metadata: mode=${CHANNEL_METADATA_MODE}, fusion=${METADATA_FUSION_MODE}, relation=${CHANNEL_MIXER_RELATION_MODE}, stats_dim=${STATS_METADATA_DIM}"
echo "🔗 query-bias scale init: ${CHANNEL_MIXER_RELATION_SCALE_INIT}"
echo "📝 log_dir: ${LOG_DIR}"

EXTRA_ARGS=(
  --attention_map_tag metadata_query_gate_stats
  --metadata_fusion_mode "${METADATA_FUSION_MODE}"
  --channel_mixer_relation_mode "${CHANNEL_MIXER_RELATION_MODE}"
  --channel_mixer_relation_scale_init "${CHANNEL_MIXER_RELATION_SCALE_INIT}"
)

if [ "${SAVE_ATTENTION_MAPS}" = "1" ]; then
  EXTRA_ARGS+=(--save_attention_maps)
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
  --lotsa_min_patches "${LOTSA_MIN_PATCHES}" \
  --lotsa_max_dim "${LOTSA_MAX_DIM}" \
  --channel_mixer_type "${LAYA_MODE}" \
  --save_dir "${SAVE_DIR}" \
  --log_dir "${LOG_DIR}" \
  "${EXTRA_ARGS[@]}"
