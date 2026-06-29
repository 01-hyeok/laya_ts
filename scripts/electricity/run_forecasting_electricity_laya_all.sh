#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/electricity"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

scripts=(
  "forecasting_electricity_laya_metadata_query_gate_text"
  "forecasting_electricity_laya_metadata_query_gate_stats"
  "forecasting_electricity_laya_metadata_query_gate_text_stats_joint"
  "forecasting_electricity_laya_metadata_query_gate_text_stats_avg"
  "forecasting_electricity_laya_mixer_text"
  "forecasting_electricity_laya_mixer_stats"
  "forecasting_electricity_laya_mixer_text_stats_avg"
  "forecasting_electricity_laya_mixer_text_stats_joint"
  "forecasting_electricity_laya_mixer_concat_text"
  "forecasting_electricity_laya_mixer_concat_stats"
  "forecasting_electricity_laya_mixer_concat_text_stats_avg"
  "forecasting_electricity_laya_mixer_concat_text_stats_joint"
  "forecasting_electricity_laya_ci"
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
