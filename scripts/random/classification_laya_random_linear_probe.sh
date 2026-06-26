#!/usr/bin/env bash
set -euo pipefail

source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARCH="laya"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-random_linear_probe}"
LAYA_MODE="${LAYA_MODE:-independent}"
CHANNEL_METADATA_MODE="${CHANNEL_METADATA_MODE:-none}"
METADATA_FUSION_MODE="${METADATA_FUSION_MODE:-none}"
CHANNEL_MIXER_RELATION_MODE="${CHANNEL_MIXER_RELATION_MODE:-none}"
CHANNEL_MIXER_RELATION_SCALE_INIT="${CHANNEL_MIXER_RELATION_SCALE_INIT:-1.0}"
PATCHIFIER_MODE="${PATCHIFIER_MODE:-single}"
MULTISCALE_PATCH_SIZES="${MULTISCALE_PATCH_SIZES:-4,8,16,32}"
MULTISCALE_BASE_PATCH="${MULTISCALE_BASE_PATCH:-16}"
MULTISCALE_GATE_TEMPERATURE="${MULTISCALE_GATE_TEMPERATURE:-1.0}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
TEXT_METADATA_CACHE_DIR="${TEXT_METADATA_CACHE_DIR:-./metadata_cache}"
STATS_METADATA_DIM="${STATS_METADATA_DIM:-384}"
LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW:-1}"

UEA_DATASETS=("EthanolConcentration" "FaceDetection" "Handwriting" "Heartbeat" "JapaneseVowels" "PEMS-SF" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "UWaveGestureLibrary")
UCR_DATASETS=("ECG200" "ECG5000" "FordA" "FordB")

BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
SEQ_LEN="${SEQ_LEN:-0}"

export LAYA_TS_LOG_TEXT_METADATA_PREVIEW="${LOG_TEXT_METADATA_PREVIEW}"

run_group() {
  local root_prefix="$1"
  shift
  local datasets=("$@")
  for dataset_name in "${datasets[@]}"; do
    local data_root="${root_prefix}/${dataset_name}"
    local log_dir="./runs/classification_random_to_${dataset_name}_${ARCH}_${EXPERIMENT_NAME}"
    mkdir -p "$log_dir"
    python -u "./run_random_classification.py" \
      --data_root "$data_root" \
      --seq_len "$SEQ_LEN" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --epochs "$NUM_EPOCHS" \
      --num_workers 0 \
      --channel_mixer_type "$LAYA_MODE" \
      --channel_metadata_mode "$CHANNEL_METADATA_MODE" \
      --metadata_fusion_mode "$METADATA_FUSION_MODE" \
      --channel_mixer_relation_mode "$CHANNEL_MIXER_RELATION_MODE" \
      --channel_mixer_relation_scale_init "$CHANNEL_MIXER_RELATION_SCALE_INIT" \
      --patchifier_mode "$PATCHIFIER_MODE" \
      --multiscale_patch_sizes "$MULTISCALE_PATCH_SIZES" \
      --multiscale_base_patch "$MULTISCALE_BASE_PATCH" \
      --multiscale_gate_temperature "$MULTISCALE_GATE_TEMPERATURE" \
      --text_encoder_name_or_path "$TEXT_ENCODER_NAME" \
      --text_metadata_cache_dir "$TEXT_METADATA_CACHE_DIR" \
      --stats_metadata_dim "$STATS_METADATA_DIM" \
      --log_dir "$log_dir"
  done
}

run_group "../Dataset/Time-Series-Library_dataset/UEA" "${UEA_DATASETS[@]}"
run_group "../Dataset/Time-Series-Library_dataset/UCR" "${UCR_DATASETS[@]}"
