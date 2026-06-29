#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/electricity"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

scripts=(
  "classification_electricity_laya_metadata_query_gate_text"
  "classification_electricity_laya_metadata_query_gate_stats"
  "classification_electricity_laya_metadata_query_gate_text_stats_joint"
  "classification_electricity_laya_metadata_query_gate_text_stats_avg"
  "classification_electricity_laya_mixer_text"
  "classification_electricity_laya_mixer_stats"
  "classification_electricity_laya_mixer_text_stats_avg"
  "classification_electricity_laya_mixer_text_stats_joint"
  "classification_electricity_laya_mixer_concat_text"
  "classification_electricity_laya_mixer_concat_stats"
  "classification_electricity_laya_mixer_concat_text_stats_avg"
  "classification_electricity_laya_mixer_concat_text_stats_joint"
  "classification_electricity_laya_ci"
)

for name in "${scripts[@]}"; do
  script="./scripts/electricity/${name}.sh"
  log="./logs/electricity/${name}.log"

  if [[ ! -x "${script}" ]]; then
    echo "Missing or non-executable script: ${script}" >&2
    exit 1
  fi

  nohup "${script}" > "${log}" 2>&1 &
  echo "Started ${name} pid=$! log=${log}"
done
