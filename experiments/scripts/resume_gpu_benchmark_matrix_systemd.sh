#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  if [[ -n "${RUN_ID}" ]]; then
    CHECKPOINT_PATH="${ROOT_DIR}/experiments/results/checkpoints/${RUN_ID}.json"
  else
    echo "ERROR: resume requires RUN_ID or CHECKPOINT_PATH." >&2
    exit 1
  fi
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(python3 - "${CHECKPOINT_PATH}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("run_id", ""))
PY
)"
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: checkpoint does not contain a run_id: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

BUILD_DIR="${BUILD_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
RESULTS_DIR="${RESULTS_DIR:-}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/experiments/results/logs/${RUN_ID}-resume-$(date -u +%Y%m%d-%H%M%S).log}"
UNIT_NAME="${UNIT_NAME:-hybridcvdv-gpu-benchmark-${RUN_ID}-resume}"
RUNNER_SCRIPT="${ROOT_DIR}/experiments/scripts/resume_gpu_benchmark_matrix.sh"
GPU_INDEX="${GPU_INDEX:-0}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"
BASELINE_PYTHON="${BASELINE_PYTHON:-}"
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

mkdir -p "$(dirname "${LOG_PATH}")" "$(dirname "${CHECKPOINT_PATH}")"

runner_cmd="$(printf '%q ' "${RUNNER_SCRIPT}" "$@")"
log_path_quoted="$(printf '%q' "${LOG_PATH}")"
shell_cmd="${runner_cmd}>> ${log_path_quoted} 2>&1"

systemctl --user stop "${UNIT_NAME}" >/dev/null 2>&1 || true
systemctl --user reset-failed "${UNIT_NAME}" >/dev/null 2>&1 || true

systemd-run \
  --user \
  --unit "${UNIT_NAME}" \
  --description "HybridCVDV SC26 GPU benchmark resume (${RUN_ID})" \
  --same-dir \
  --setenv=RUN_ID="${RUN_ID}" \
  --setenv=BUILD_DIR="${BUILD_DIR}" \
  --setenv=CONFIG_PATH="${CONFIG_PATH}" \
  --setenv=RESULTS_DIR="${RESULTS_DIR}" \
  --setenv=CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  --setenv=GPU_INDEX="${GPU_INDEX}" \
  --setenv=TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS}" \
  --setenv=BASELINE_PYTHON="${BASELINE_PYTHON}" \
  --setenv=CASE_FILTER="${CASE_FILTER}" \
  --setenv=BACKEND="${BACKEND}" \
  --setenv=HYBRIDCVDV_SCALING_WARMUP_RUNS="${HYBRIDCVDV_SCALING_WARMUP_RUNS}" \
  --setenv=HYBRIDCVDV_SCALING_MEASURED_RUNS="${HYBRIDCVDV_SCALING_MEASURED_RUNS}" \
  /bin/bash -lc "${shell_cmd}"

cat <<EOF
Launched benchmark resume via systemd user service.
unit: ${UNIT_NAME}.service
run_id: ${RUN_ID}
checkpoint: ${CHECKPOINT_PATH}
log: ${LOG_PATH}

Inspect:
  systemctl --user status ${UNIT_NAME}
  tail -f ${LOG_PATH}
EOF
