#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-sc26_gpu_benchmark_$(date -u +%Y%m%d-%H%M%S)}"
BUILD_DIR="${BUILD_DIR:-build-h100}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/experiments/configs/sc26_scaling.json}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/experiments/results}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/experiments/results/checkpoints/${RUN_ID}.json}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/experiments/results/logs/${RUN_ID}.log}"
UNIT_NAME="${UNIT_NAME:-hybridcvdv-gpu-benchmark-${RUN_ID}}"
RUNNER_SCRIPT="${ROOT_DIR}/experiments/scripts/run_gpu_benchmark_matrix.sh"
GPU_INDEX="${GPU_INDEX:-0}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"
BASELINE_PYTHON="${BASELINE_PYTHON:-}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CASE_FILTER="${CASE_FILTER:-}"
BACKEND="${BACKEND:-}"
HYBRIDCVDV_SCALING_WARMUP_RUNS="${HYBRIDCVDV_SCALING_WARMUP_RUNS:-}"
HYBRIDCVDV_SCALING_MEASURED_RUNS="${HYBRIDCVDV_SCALING_MEASURED_RUNS:-}"

setup_user_systemd_env() {
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "${XDG_RUNTIME_DIR}/bus" ]]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
  fi
}

setup_user_systemd_env

if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "ERROR: systemd --user bus is unavailable for $(id -un)." >&2
  echo "       Ensure the account has an active user systemd session (for example via a normal SSH login)." >&2
  exit 1
fi

mkdir -p "$(dirname "${CHECKPOINT_PATH}")" "$(dirname "${LOG_PATH}")" "${RESULTS_DIR}"

runner_cmd="$(printf '%q ' "${RUNNER_SCRIPT}" "$@")"
log_path_quoted="$(printf '%q' "${LOG_PATH}")"
shell_cmd="${runner_cmd}>> ${log_path_quoted} 2>&1"

systemctl --user stop "${UNIT_NAME}" >/dev/null 2>&1 || true
systemctl --user reset-failed "${UNIT_NAME}" >/dev/null 2>&1 || true

systemd-run \
  --user \
  --unit "${UNIT_NAME}" \
  --description "HybridCVDV SC26 GPU benchmark matrix (${RUN_ID})" \
  --same-dir \
  --setenv=RUN_ID="${RUN_ID}" \
  --setenv=BUILD_DIR="${BUILD_DIR}" \
  --setenv=CONFIG_PATH="${CONFIG_PATH}" \
  --setenv=RESULTS_DIR="${RESULTS_DIR}" \
  --setenv=CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  --setenv=GPU_INDEX="${GPU_INDEX}" \
  --setenv=TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS}" \
  --setenv=BASELINE_PYTHON="${BASELINE_PYTHON}" \
  --setenv=BUILD_JOBS="${BUILD_JOBS}" \
  --setenv=SKIP_BUILD="${SKIP_BUILD}" \
  --setenv=CASE_FILTER="${CASE_FILTER}" \
  --setenv=BACKEND="${BACKEND}" \
  --setenv=HYBRIDCVDV_SCALING_WARMUP_RUNS="${HYBRIDCVDV_SCALING_WARMUP_RUNS}" \
  --setenv=HYBRIDCVDV_SCALING_MEASURED_RUNS="${HYBRIDCVDV_SCALING_MEASURED_RUNS}" \
  /bin/bash -lc "${shell_cmd}"

cat <<EOF
Launched benchmark matrix via systemd user service.
unit: ${UNIT_NAME}.service
run_id: ${RUN_ID}
checkpoint: ${CHECKPOINT_PATH}
log: ${LOG_PATH}

Inspect:
  systemctl --user status ${UNIT_NAME}
  tail -f ${LOG_PATH}

Resume with:
  RUN_ID=${RUN_ID} CHECKPOINT_PATH=${CHECKPOINT_PATH} bash ${ROOT_DIR}/experiments/scripts/resume_gpu_benchmark_matrix_systemd.sh
EOF
