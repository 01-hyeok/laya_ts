#!/usr/bin/env bash
set -euo pipefail

source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATA="electricity"
DATA_PATH="${DATA_PATH:-../Dataset/long_term_forecast/electricity/electricity.csv}"

ARCH="laya"
VARIANT="${VARIANT:-s}"
LAYA_MODE="${LAYA_MODE:-mixer}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-text_stats_avg}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-attention_suppress_gate}"
CHANNEL_MIXER_RELATION_MODE="${CHANNEL_MIXER_RELATION_MODE:-none}"
STATS_METADATA_DIM="${STATS_METADATA_DIM:-384}"

DESCRIPTION_RELATION_METRIC="${DESCRIPTION_RELATION_METRIC:-cosine}"
DESCRIPTION_RELATION_LAMBDA_INIT="${DESCRIPTION_RELATION_LAMBDA_INIT:-1.0}"
DESCRIPTION_RELATION_GAMMA_INIT="${DESCRIPTION_RELATION_GAMMA_INIT:-1.0}"

SEQ_LEN="${SEQ_LEN:-512}"
D_MODEL="${D_MODEL:-128}"
PATCH_SIZE="${PATCH_SIZE:-16}"
STRIDE="${STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-100}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-3}"
PROJ_DIM="${PROJ_DIM:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
MIN_LR="${MIN_LR:-1e-6}"

TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
TEXT_METADATA_CACHE_DIR="${TEXT_METADATA_CACHE_DIR:-./metadata_cache}"
LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW:-1}"
SAVE_ATTENTION_MAPS="${SAVE_ATTENTION_MAPS:-1}"

SAVE_DIR="${SAVE_DIR:-./checkpoints/${DATA}_${ARCH}_attention_suppress_gate_text_stats_avg}"
LOG_DIR="${LOG_DIR:-./runs/pretrain_${DATA}_${ARCH}_attention_suppress_gate_text_stats_avg}"


export LAYA_TS_LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW}"

echo "🚀 Laya-TS Pretraining: ${DATA} (${ARCH} + attention_suppress_gate_text_stats_avg)"
echo "📊 data_path: ${DATA_PATH}"
echo "📝 metadata: mode=${CHANNEL_METADATA_MODE}, fusion=${METADATA_FUSION_MODE}, relation=${CHANNEL_MIXER_RELATION_MODE}, stats_dim=${STATS_METADATA_DIM}"
echo "🔗 suppression relation: metric=${DESCRIPTION_RELATION_METRIC}, lambda_init=${DESCRIPTION_RELATION_LAMBDA_INIT}, gamma_init=${DESCRIPTION_RELATION_GAMMA_INIT}"
echo "📝 log_dir: ${LOG_DIR}"

EXTRA_ARGS=(
  --attention_map_tag attention_suppress_gate_text_stats_avg
  --metadata_fusion_mode "${METADATA_FUSION_MODE}"
  --channel_mixer_relation_mode "${CHANNEL_MIXER_RELATION_MODE}"
  --description_relation_metric "${DESCRIPTION_RELATION_METRIC}"
  --description_relation_lambda_init "${DESCRIPTION_RELATION_LAMBDA_INIT}"
  --description_relation_gamma_init "${DESCRIPTION_RELATION_GAMMA_INIT}"
)

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
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --min_lr "${MIN_LR}" \
  --channel_metadata_mode "${CHANNEL_METADATA_MODE}" \
  --stats_metadata_dim "${STATS_METADATA_DIM}" \
  --channel_mixer_type "${LAYA_MODE}" \
  --text_encoder_name_or_path "${TEXT_ENCODER_NAME}" \
  --text_metadata_cache_dir "${TEXT_METADATA_CACHE_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --log_dir "${LOG_DIR}" \
  "${EXTRA_ARGS[@]}"
