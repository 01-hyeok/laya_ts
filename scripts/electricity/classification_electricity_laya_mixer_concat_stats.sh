#!/usr/bin/env bash
set -euo pipefail

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

UEA_DATASETS=("EthanolConcentration" "FaceDetection" "Handwriting" "Heartbeat" "JapaneseVowels" "PEMS-SF" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "UWaveGestureLibrary")
UCR_DATASETS=("ECG200" "ECG5000" "FordA" "FordB")

BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
SEQ_LEN="${SEQ_LEN:-0}"
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

run_group() {
  local root_prefix="$1"
  shift
  local datasets=("$@")
  for dataset_name in "${datasets[@]}"; do
    local data_root="${root_prefix}/${dataset_name}"
    local log_dir="./runs/classification_${PRETRAIN_DATA}_to_${dataset_name}_${ARCH}_mixer_concat_stats"
    mkdir -p "$log_dir"
    python -u "./run_classification.py" \
      --data_root "$data_root" \
      --seq_len "$SEQ_LEN" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --epochs "$NUM_EPOCHS" \
      --num_workers 0 \
      --checkpoint "$CHECKPOINT" \
      --channel_mixer_type "$LAYA_MODE" \
      --channel_metadata_mode "$CHANNEL_METADATA_MODE" \
      --log_dir "$log_dir" \
      "${EXTRA_ARGS[@]}"
  done
}

run_group "../Dataset/Time-Series-Library_dataset/UEA" "${UEA_DATASETS[@]}"
run_group "../Dataset/Time-Series-Library_dataset/UCR" "${UCR_DATASETS[@]}"
