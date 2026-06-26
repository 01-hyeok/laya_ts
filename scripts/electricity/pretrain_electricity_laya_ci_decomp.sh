#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA="electricity"
DATA_PATH="${DATA_PATH:-../Dataset/long_term_forecast/electricity/electricity.csv}"

ARCH="laya"
VARIANT="${VARIANT:-s}"
MODEL_ID="${MODEL_ID:-laya_ci_decomp}"
LAYA_MODE="${LAYA_MODE:-independent}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-none}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-none}"
PATCHIFIER_MODE="${PATCHIFIER_MODE:-trend_seasonal}"
TREND_SEASONAL_KERNEL="${TREND_SEASONAL_KERNEL:-25}"
TREND_SEASONAL_GATE_TEMPERATURE="${TREND_SEASONAL_GATE_TEMPERATURE:-1.0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-laya_ci_decomp}"

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
SAVE_DIR="${SAVE_DIR:-./checkpoints/${DATA}_${ARCH}_${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-./runs/pretrain_${DATA}_${EXPERIMENT_NAME}}"

echo "Laya-TS Pretraining: ${DATA} (${ARCH} + ${EXPERIMENT_NAME})"
echo "data_path: ${DATA_PATH}"
echo "metadata: mode=${CHANNEL_METADATA_MODE}, fusion=${METADATA_FUSION_MODE}"
echo "decomp: mode=${PATCHIFIER_MODE}, kernel=${TREND_SEASONAL_KERNEL}, temp=${TREND_SEASONAL_GATE_TEMPERATURE}"
echo "log_dir: ${LOG_DIR}"

EXTRA_ARGS=(
  --attention_map_tag "${EXPERIMENT_NAME}"
  --model_id "${MODEL_ID}"
  --metadata_fusion_mode "${METADATA_FUSION_MODE}"
  --patchifier_mode "${PATCHIFIER_MODE}"
  --trend_seasonal_kernel "${TREND_SEASONAL_KERNEL}"
  --trend_seasonal_gate_temperature "${TREND_SEASONAL_GATE_TEMPERATURE}"
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
  --predictor_depth "${PREDICTOR_DEPTH}" \
  --predictor_heads "${PREDICTOR_HEADS}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --min_lr "${MIN_LR}" \
  --channel_metadata_mode "${CHANNEL_METADATA_MODE}" \
  --channel_mixer_type "${LAYA_MODE}" \
  --save_dir "${SAVE_DIR}" \
  --log_dir "${LOG_DIR}" \
  "${EXTRA_ARGS[@]}"
