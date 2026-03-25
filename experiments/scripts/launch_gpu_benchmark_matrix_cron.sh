#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  launch_gpu_benchmark_matrix_cron.sh install
  launch_gpu_benchmark_matrix_cron.sh tick --env-file <path>
  launch_gpu_benchmark_matrix_cron.sh status --env-file <path>
  launch_gpu_benchmark_matrix_cron.sh uninstall --env-file <path>

install mode writes a launcher env file and installs a user-cron entry that:
  1. waits for the target GPU to have no active compute apps,
  2. launches a smoke preflight worker,
  3. starts or resumes the checkpointed SC26 matrix once preflight passes.

Key env vars for install mode:
  RUN_ID
  BUILD_DIR
  CONFIG_PATH
  RESULTS_DIR
  CHECKPOINT_PATH
  GPU_INDEX
  TELEMETRY_INTERVAL_MS
  SKIP_BUILD
  CASE_FILTER
  BACKEND
  BASELINE_PYTHON
  HYBRIDCVDV_SCALING_WARMUP_RUNS
  HYBRIDCVDV_SCALING_MEASURED_RUNS
  PRECHECK_CASE_FILTER              (default: sc26_vqe_nq3_nm7_c16)
  PRECHECK_WARMUP_RUNS              (default: 0)
  PRECHECK_MEASURED_RUNS            (default: 1)
  PRECHECK_DEBUG                    (default: 1)
  PREFLIGHT_MAX_ATTEMPTS            (default: 0, unlimited retries)
  WAIT_FOR_IDLE_GPU                 (default: 1)
  CRON_SCHEDULE                     (default: "* * * * *")
  CRON_SECONDARY_DELAY_SECONDS      (default: empty, install a second tick after this delay)
  ENV_FILE                          (default: <results>/cron/<run-id>/launcher.env)
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_crontab() {
  command -v crontab >/dev/null 2>&1 || die "crontab is not available in PATH"
}

write_env_file() {
  local env_file="$1"
  mkdir -p "$(dirname "${env_file}")"
  cat > "${env_file}" <<EOF
ROOT_DIR=$(printf '%q' "${ROOT_DIR}")
RUN_ID=$(printf '%q' "${RUN_ID}")
BUILD_DIR=$(printf '%q' "${BUILD_DIR}")
CONFIG_PATH=$(printf '%q' "${CONFIG_PATH}")
RESULTS_DIR=$(printf '%q' "${RESULTS_DIR}")
CHECKPOINT_PATH=$(printf '%q' "${CHECKPOINT_PATH}")
BINARY_PATH=$(printf '%q' "${BINARY_PATH}")
STATUS_JSON=$(printf '%q' "${STATUS_JSON}")
GPU_INDEX=$(printf '%q' "${GPU_INDEX}")
TELEMETRY_INTERVAL_MS=$(printf '%q' "${TELEMETRY_INTERVAL_MS}")
BASELINE_PYTHON=$(printf '%q' "${BASELINE_PYTHON}")
BUILD_JOBS=$(printf '%q' "${BUILD_JOBS}")
SKIP_BUILD=$(printf '%q' "${SKIP_BUILD}")
CASE_FILTER=$(printf '%q' "${CASE_FILTER}")
BACKEND=$(printf '%q' "${BACKEND}")
HYBRIDCVDV_SCALING_WARMUP_RUNS=$(printf '%q' "${HYBRIDCVDV_SCALING_WARMUP_RUNS}")
HYBRIDCVDV_SCALING_MEASURED_RUNS=$(printf '%q' "${HYBRIDCVDV_SCALING_MEASURED_RUNS}")
PRECHECK_CASE_FILTER=$(printf '%q' "${PRECHECK_CASE_FILTER}")
PRECHECK_WARMUP_RUNS=$(printf '%q' "${PRECHECK_WARMUP_RUNS}")
PRECHECK_MEASURED_RUNS=$(printf '%q' "${PRECHECK_MEASURED_RUNS}")
PRECHECK_DEBUG=$(printf '%q' "${PRECHECK_DEBUG}")
PREFLIGHT_MAX_ATTEMPTS=$(printf '%q' "${PREFLIGHT_MAX_ATTEMPTS}")
WAIT_FOR_IDLE_GPU=$(printf '%q' "${WAIT_FOR_IDLE_GPU}")
CRON_SCHEDULE=$(printf '%q' "${CRON_SCHEDULE}")
CRON_SECONDARY_DELAY_SECONDS=$(printf '%q' "${CRON_SECONDARY_DELAY_SECONDS}")
STATE_DIR=$(printf '%q' "${STATE_DIR}")
LAUNCHER_LOG=$(printf '%q' "${LAUNCHER_LOG}")
MATRIX_LOG=$(printf '%q' "${MATRIX_LOG}")
SMOKE_DIR=$(printf '%q' "${SMOKE_DIR}")
SMOKE_JSON=$(printf '%q' "${SMOKE_JSON}")
SMOKE_LOG=$(printf '%q' "${SMOKE_LOG}")
MANIFEST_PATH=$(printf '%q' "${MANIFEST_PATH}")
CRON_MARKER=$(printf '%q' "${CRON_MARKER}")
EOF
}

install_crontab_entry() {
  local env_file="$1"
  local tmp
  tmp="$(mktemp)"
  local escaped_script
  local escaped_env
  local escaped_delay=""
  escaped_script="$(printf '%q' "${ROOT_DIR}/experiments/scripts/launch_gpu_benchmark_matrix_cron.sh")"
  escaped_env="$(printf '%q' "${env_file}")"
  if [[ -n "${CRON_SECONDARY_DELAY_SECONDS}" ]]; then
    escaped_delay="$(printf '%q' "${CRON_SECONDARY_DELAY_SECONDS}")"
  fi
  {
    crontab -l 2>/dev/null | sed "/# BEGIN ${CRON_MARKER}\$/,/# END ${CRON_MARKER}\$/d" || true
    echo "# BEGIN ${CRON_MARKER}"
    echo "${CRON_SCHEDULE} /bin/bash ${escaped_script} tick --env-file ${escaped_env}"
    if [[ -n "${escaped_delay}" ]]; then
      echo "${CRON_SCHEDULE} /bin/sleep ${escaped_delay} && /bin/bash ${escaped_script} tick --env-file ${escaped_env}"
    fi
    echo "# END ${CRON_MARKER}"
  } > "${tmp}"
  crontab "${tmp}"
  rm -f "${tmp}"
}

uninstall_crontab_entry() {
  local tmp
  tmp="$(mktemp)"
  {
    crontab -l 2>/dev/null | sed "/# BEGIN ${CRON_MARKER}\$/,/# END ${CRON_MARKER}\$/d" || true
  } > "${tmp}"
  crontab "${tmp}"
  rm -f "${tmp}"
}

preflight_ok() {
  python3 - "${SMOKE_JSON}" "$1" <<'PY'
import json
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
if rc != 0 or not json_path.exists():
    raise SystemExit(1)
payload = json.loads(json_path.read_text(encoding="utf-8"))
results = payload.get("results") or []
if not results or results[0].get("status") != "ok":
    raise SystemExit(1)
PY
}

manifest_completed() {
  python3 - "${MANIFEST_PATH}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "completed" else 1)
PY
}

load_state_paths() {
  LOCK_FILE="${STATE_DIR}/launcher.lock"
  DONE_FILE="${STATE_DIR}/matrix.done"
  PREFLIGHT_OK_FILE="${STATE_DIR}/preflight.ok"
  PREFLIGHT_HOLD_FILE="${STATE_DIR}/preflight.hold"
  PREFLIGHT_ATTEMPTS_FILE="${STATE_DIR}/preflight_attempts.txt"
  PREFLIGHT_PID_FILE="${STATE_DIR}/preflight.pid"
  PREFLIGHT_RC_FILE="${STATE_DIR}/preflight.rc"
  MATRIX_PID_FILE="${STATE_DIR}/matrix.pid"
  MATRIX_RC_FILE="${STATE_DIR}/matrix.rc"
}

launcher_log() {
  echo "[launcher] $(date -u +%FT%TZ) $*" >> "${LAUNCHER_LOG}"
}

signal_exit_code() {
  case "${1:-}" in
    HUP) echo 129 ;;
    INT) echo 130 ;;
    QUIT) echo 131 ;;
    TERM) echo 143 ;;
    *) echo 1 ;;
  esac
}

worker_signal_trap() {
  WORKER_SIGNAL="${1:-}"
  exit "$(signal_exit_code "${WORKER_SIGNAL}")"
}

worker_exit_trap() {
  local worker_name="$1"
  local pid_file="$2"
  local rc_file="$3"
  local rc=$?

  rm -f "${pid_file}"
  if [[ ! -f "${rc_file}" ]]; then
    printf '%s\n' "${rc}" > "${rc_file}"
    if [[ -n "${WORKER_SIGNAL:-}" ]]; then
      launcher_log "${worker_name} worker interrupted by ${WORKER_SIGNAL} (rc=${rc})"
    else
      launcher_log "${worker_name} worker exited unexpectedly rc=${rc}"
    fi
  fi
}

read_pid_file() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  tr -d '[:space:]' < "${pid_file}" 2>/dev/null || true
}

pid_is_running() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 1
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

clear_stale_pid_file() {
  local pid_file="$1"
  local pid
  pid="$(read_pid_file "${pid_file}")"
  if [[ -n "${pid}" ]] && ! pid_is_running "${pid}"; then
    rm -f "${pid_file}"
  fi
}

start_detached_worker() {
  local mode="$1"
  local env_file="$2"
  local pid_file="$3"
  local launcher="${ROOT_DIR}/experiments/scripts/launch_gpu_benchmark_matrix_cron.sh"
  local i

  rm -f "${pid_file}"
  if command -v setsid >/dev/null 2>&1; then
    setsid /bin/bash "${launcher}" "${mode}" --env-file "${env_file}" >/dev/null 2>&1 < /dev/null &
  else
    nohup /bin/bash "${launcher}" "${mode}" --env-file "${env_file}" >/dev/null 2>&1 < /dev/null &
  fi

  for ((i = 0; i < 20; i++)); do
    if [[ -s "${pid_file}" ]]; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

tick_mode() {
  local env_file=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --env-file)
        [[ "$#" -ge 2 ]] || die "--env-file requires a path"
        env_file="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown tick argument: $1"
        ;;
    esac
  done
  [[ -n "${env_file}" ]] || die "tick requires --env-file"
  [[ -f "${env_file}" ]] || die "launcher env file not found: ${env_file}"

  # shellcheck disable=SC1090
  source "${env_file}"

  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
  load_state_paths

  mkdir -p "${STATE_DIR}" "${SMOKE_DIR}" "$(dirname "${LAUNCHER_LOG}")" \
            "$(dirname "${MATRIX_LOG}")" "$(dirname "${CHECKPOINT_PATH}")"

  exec 9> "${LOCK_FILE}"
  if ! flock -n 9; then
    exit 0
  fi

  log() {
    launcher_log "$@"
  }

  write_status() {
    local state="$1"
    local note="$2"
    local apps_raw="${3:-}"
    STATUS_JSON_PATH="${STATUS_JSON}" \
    STATUS_STATE="${state}" \
    STATUS_NOTE="${note}" \
    STATUS_APPS_RAW="${apps_raw}" \
    STATUS_RUN_ID="${RUN_ID}" \
    STATUS_CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    STATUS_MANIFEST_PATH="${MANIFEST_PATH}" \
    STATUS_SMOKE_JSON="${SMOKE_JSON}" \
    STATUS_MATRIX_LOG="${MATRIX_LOG}" \
    STATUS_LAUNCHER_LOG="${LAUNCHER_LOG}" \
    STATUS_PREFLIGHT_OK_FILE="${PREFLIGHT_OK_FILE}" \
    STATUS_PREFLIGHT_HOLD_FILE="${PREFLIGHT_HOLD_FILE}" \
    STATUS_DONE_FILE="${DONE_FILE}" \
    STATUS_PREFLIGHT_ATTEMPTS_FILE="${PREFLIGHT_ATTEMPTS_FILE}" \
    python3 - <<'PY'
import json
import os
import pathlib
import time

def read_attempts(path_str: str) -> int:
    if not path_str:
        return 0
    path = pathlib.Path(path_str)
    if not path.exists():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        return 0

payload = {
    "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_id": os.environ.get("STATUS_RUN_ID", ""),
    "state": os.environ.get("STATUS_STATE", ""),
    "note": os.environ.get("STATUS_NOTE", ""),
    "compute_apps_raw": os.environ.get("STATUS_APPS_RAW", ""),
    "checkpoint_path": os.environ.get("STATUS_CHECKPOINT_PATH", ""),
    "manifest_path": os.environ.get("STATUS_MANIFEST_PATH", ""),
    "smoke_json": os.environ.get("STATUS_SMOKE_JSON", ""),
    "matrix_log": os.environ.get("STATUS_MATRIX_LOG", ""),
    "launcher_log": os.environ.get("STATUS_LAUNCHER_LOG", ""),
    "preflight_ok": pathlib.Path(os.environ.get("STATUS_PREFLIGHT_OK_FILE", "")).exists(),
    "preflight_hold": pathlib.Path(os.environ.get("STATUS_PREFLIGHT_HOLD_FILE", "")).exists(),
    "done": pathlib.Path(os.environ.get("STATUS_DONE_FILE", "")).exists(),
    "preflight_attempts": read_attempts(os.environ.get("STATUS_PREFLIGHT_ATTEMPTS_FILE", "")),
}
path = pathlib.Path(os.environ["STATUS_JSON_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  }

  local previous_preflight_pid=""
  local previous_matrix_pid=""
  previous_preflight_pid="$(read_pid_file "${PREFLIGHT_PID_FILE}")"
  previous_matrix_pid="$(read_pid_file "${MATRIX_PID_FILE}")"
  clear_stale_pid_file "${PREFLIGHT_PID_FILE}"
  clear_stale_pid_file "${MATRIX_PID_FILE}"
  if [[ -n "${previous_preflight_pid}" && ! -f "${PREFLIGHT_PID_FILE}" && ! -f "${PREFLIGHT_RC_FILE}" ]]; then
    log "preflight worker pid ${previous_preflight_pid} vanished before writing rc"
  fi
  if [[ -n "${previous_matrix_pid}" && ! -f "${MATRIX_PID_FILE}" && ! -f "${MATRIX_RC_FILE}" ]]; then
    log "matrix worker pid ${previous_matrix_pid} vanished before writing rc"
  fi

  if [[ -f "${DONE_FILE}" ]]; then
    write_status "completed" "done flag already present"
    exit 0
  fi

  if [[ -f "${MANIFEST_PATH}" ]] && manifest_completed; then
    touch "${DONE_FILE}"
    log "manifest already completed; marking done"
    write_status "completed" "manifest already marked completed"
    exit 0
  fi

  if [[ -f "${PREFLIGHT_HOLD_FILE}" ]]; then
    log "preflight hold file present; waiting for manual intervention"
    write_status "preflight_hold" "preflight hold file present"
    exit 0
  fi

  local running_pid=""
  running_pid="$(read_pid_file "${PREFLIGHT_PID_FILE}")"
  if [[ -n "${running_pid}" ]] && pid_is_running "${running_pid}"; then
    write_status "preflight_running" "preflight worker pid ${running_pid} is running"
    exit 0
  fi

  running_pid="$(read_pid_file "${MATRIX_PID_FILE}")"
  if [[ -n "${running_pid}" ]] && pid_is_running "${running_pid}"; then
    write_status "matrix_running" "matrix worker pid ${running_pid} is running"
    exit 0
  fi

  if [[ "${WAIT_FOR_IDLE_GPU}" == "1" ]]; then
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | sed '/^\s*$/d' || true)"
    if [[ -n "${apps}" ]]; then
      log "GPU busy; deferring launch"
      printf '%s\n' "${apps}" >> "${LAUNCHER_LOG}"
      write_status "waiting_for_gpu" "active compute apps detected" "${apps}"
      exit 0
    fi
  fi

  if [[ -n "${PRECHECK_CASE_FILTER}" && ! -f "${PREFLIGHT_OK_FILE}" ]]; then
    [[ -x "${BINARY_PATH}" ]] || die "preflight binary not found or not executable: ${BINARY_PATH}"

    local attempts=0
    if [[ -f "${PREFLIGHT_ATTEMPTS_FILE}" ]]; then
      attempts="$(cat "${PREFLIGHT_ATTEMPTS_FILE}" 2>/dev/null || echo 0)"
    fi
    if [[ -f "${PREFLIGHT_RC_FILE}" ]] && [[ "${PREFLIGHT_MAX_ATTEMPTS}" =~ ^[0-9]+$ ]] && \
       (( PREFLIGHT_MAX_ATTEMPTS > 0 )) && (( attempts >= PREFLIGHT_MAX_ATTEMPTS )); then
      touch "${PREFLIGHT_HOLD_FILE}"
      log "preflight reached max attempts (${PREFLIGHT_MAX_ATTEMPTS}); holding"
      write_status "preflight_hold" "preflight failed and reached max attempts (${PREFLIGHT_MAX_ATTEMPTS})"
      exit 0
    fi

    attempts=$((attempts + 1))
    printf '%s\n' "${attempts}" > "${PREFLIGHT_ATTEMPTS_FILE}"

    rm -f "${PREFLIGHT_RC_FILE}"
    log "GPU appears idle; launching preflight attempt ${attempts} (${PRECHECK_CASE_FILTER})"
    write_status "preflight_running" "launching ${PRECHECK_CASE_FILTER} (attempt ${attempts})"
    if ! start_detached_worker "run-preflight" "${env_file}" "${PREFLIGHT_PID_FILE}"; then
      log "failed to launch detached preflight worker"
      write_status "preflight_failed" "failed to launch detached preflight worker"
    fi
    exit 0
  fi

  local launch_note
  if [[ -f "${CHECKPOINT_PATH}" ]]; then
    log "resuming SC26 matrix from checkpoint"
    launch_note="resuming matrix from checkpoint"
  else
    log "starting full SC26 matrix"
    launch_note="starting full matrix"
  fi

  rm -f "${MATRIX_RC_FILE}"
  write_status "matrix_running" "${launch_note}"
  if ! start_detached_worker "run-matrix" "${env_file}" "${MATRIX_PID_FILE}"; then
    log "failed to launch detached matrix worker"
    write_status "matrix_retry_pending" "failed to launch detached matrix worker"
  fi
}

run_preflight_mode() {
  local env_file=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --env-file)
        [[ "$#" -ge 2 ]] || die "--env-file requires a path"
        env_file="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown run-preflight argument: $1"
        ;;
    esac
  done
  [[ -n "${env_file}" ]] || die "run-preflight requires --env-file"
  [[ -f "${env_file}" ]] || die "launcher env file not found: ${env_file}"

  # shellcheck disable=SC1090
  source "${env_file}"
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
  load_state_paths

  mkdir -p "${STATE_DIR}" "${SMOKE_DIR}" "$(dirname "${LAUNCHER_LOG}")"
  WORKER_SIGNAL=""
  echo $$ > "${PREFLIGHT_PID_FILE}"
  trap 'worker_exit_trap "preflight" "${PREFLIGHT_PID_FILE}" "${PREFLIGHT_RC_FILE}"' EXIT
  trap 'worker_signal_trap HUP' HUP
  trap 'worker_signal_trap INT' INT
  trap 'worker_signal_trap QUIT' QUIT
  trap 'worker_signal_trap TERM' TERM
  rm -f "${PREFLIGHT_RC_FILE}"

  export HYBRIDCVDV_SCALING_WARMUP_RUNS="${PRECHECK_WARMUP_RUNS}"
  export HYBRIDCVDV_SCALING_MEASURED_RUNS="${PRECHECK_MEASURED_RUNS}"
  if [[ "${PRECHECK_DEBUG}" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
    export HYBRIDCVDV_FALLBACK_DEBUG=1
  else
    unset CUDA_LAUNCH_BLOCKING HYBRIDCVDV_FALLBACK_DEBUG
  fi

  set +e
  "${BINARY_PATH}" --suite scaling --name-filter "${PRECHECK_CASE_FILTER}" --output "${SMOKE_JSON}" > "${SMOKE_LOG}" 2>&1
  local smoke_rc=$?
  set -e

  printf '%s\n' "${smoke_rc}" > "${PREFLIGHT_RC_FILE}"
  if preflight_ok "${smoke_rc}"; then
    unset CUDA_LAUNCH_BLOCKING HYBRIDCVDV_FALLBACK_DEBUG
    touch "${PREFLIGHT_OK_FILE}"
    rm -f "${PREFLIGHT_HOLD_FILE}"
    launcher_log "preflight passed"
  else
    launcher_log "preflight failed (rc=${smoke_rc}); see ${SMOKE_LOG}"
  fi
}

run_matrix_mode() {
  local env_file=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --env-file)
        [[ "$#" -ge 2 ]] || die "--env-file requires a path"
        env_file="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown run-matrix argument: $1"
        ;;
    esac
  done
  [[ -n "${env_file}" ]] || die "run-matrix requires --env-file"
  [[ -f "${env_file}" ]] || die "launcher env file not found: ${env_file}"

  # shellcheck disable=SC1090
  source "${env_file}"
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
  load_state_paths

  mkdir -p "${STATE_DIR}" "$(dirname "${LAUNCHER_LOG}")" "$(dirname "${MATRIX_LOG}")" \
           "$(dirname "${CHECKPOINT_PATH}")"
  WORKER_SIGNAL=""
  echo $$ > "${MATRIX_PID_FILE}"
  trap 'worker_exit_trap "matrix" "${MATRIX_PID_FILE}" "${MATRIX_RC_FILE}"' EXIT
  trap 'worker_signal_trap HUP' HUP
  trap 'worker_signal_trap INT' INT
  trap 'worker_signal_trap QUIT' QUIT
  trap 'worker_signal_trap TERM' TERM
  rm -f "${MATRIX_RC_FILE}"

  export RUN_ID BUILD_DIR CONFIG_PATH RESULTS_DIR CHECKPOINT_PATH
  export GPU_INDEX TELEMETRY_INTERVAL_MS BASELINE_PYTHON BUILD_JOBS SKIP_BUILD CASE_FILTER BACKEND
  export HYBRIDCVDV_SCALING_WARMUP_RUNS HYBRIDCVDV_SCALING_MEASURED_RUNS

  local runner
  if [[ -f "${CHECKPOINT_PATH}" ]]; then
    runner="${ROOT_DIR}/experiments/scripts/resume_gpu_benchmark_matrix.sh"
  else
    runner="${ROOT_DIR}/experiments/scripts/run_gpu_benchmark_matrix.sh"
  fi

  set +e
  bash "${runner}" >> "${MATRIX_LOG}" 2>&1
  local run_rc=$?
  set -e

  printf '%s\n' "${run_rc}" > "${MATRIX_RC_FILE}"
  if [[ "${run_rc}" -eq 0 ]]; then
    touch "${DONE_FILE}"
    launcher_log "matrix completed successfully"
  else
    launcher_log "matrix runner exited rc=${run_rc}; cron will retry/resume on the next tick"
  fi
}

install_mode() {
  require_crontab

  RUN_ID="${RUN_ID:-sc26_gpu_benchmark_$(date -u +%Y%m%d-%H%M%S)}"
  BUILD_DIR="${BUILD_DIR:-build-h100}"
  CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/experiments/configs/sc26_scaling.json}"
  RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/experiments/results}"
  CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESULTS_DIR}/checkpoints/${RUN_ID}.json}"
  GPU_INDEX="${GPU_INDEX:-0}"
  TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"
  BASELINE_PYTHON="${BASELINE_PYTHON:-}"
  BUILD_JOBS="${BUILD_JOBS:-8}"
  SKIP_BUILD="${SKIP_BUILD:-0}"
  CASE_FILTER="${CASE_FILTER:-}"
  BACKEND="${BACKEND:-}"
  HYBRIDCVDV_SCALING_WARMUP_RUNS="${HYBRIDCVDV_SCALING_WARMUP_RUNS:-}"
  HYBRIDCVDV_SCALING_MEASURED_RUNS="${HYBRIDCVDV_SCALING_MEASURED_RUNS:-}"
  PRECHECK_CASE_FILTER="${PRECHECK_CASE_FILTER:-sc26_vqe_nq3_nm7_c16}"
  PRECHECK_WARMUP_RUNS="${PRECHECK_WARMUP_RUNS:-0}"
  PRECHECK_MEASURED_RUNS="${PRECHECK_MEASURED_RUNS:-1}"
  PRECHECK_DEBUG="${PRECHECK_DEBUG:-1}"
  PREFLIGHT_MAX_ATTEMPTS="${PREFLIGHT_MAX_ATTEMPTS:-0}"
  WAIT_FOR_IDLE_GPU="${WAIT_FOR_IDLE_GPU:-1}"
  CRON_SCHEDULE="${CRON_SCHEDULE:-* * * * *}"
  CRON_SECONDARY_DELAY_SECONDS="${CRON_SECONDARY_DELAY_SECONDS:-}"
  STATE_DIR="${STATE_DIR:-${RESULTS_DIR}/cron/${RUN_ID}}"
  LAUNCHER_LOG="${LAUNCHER_LOG:-${RESULTS_DIR}/logs/${RUN_ID}.launcher-cron.log}"
  MATRIX_LOG="${MATRIX_LOG:-${RESULTS_DIR}/logs/${RUN_ID}.log}"
  SMOKE_DIR="${SMOKE_DIR:-${RESULTS_DIR}/a100-hybrid-smoke}"
  SMOKE_JSON="${SMOKE_JSON:-${SMOKE_DIR}/${RUN_ID}-preflight.json}"
  SMOKE_LOG="${SMOKE_LOG:-${SMOKE_DIR}/${RUN_ID}-preflight.log}"
  STATUS_JSON="${STATUS_JSON:-${STATE_DIR}/status.json}"
  MANIFEST_PATH="${MANIFEST_PATH:-${RESULTS_DIR}/${RUN_ID}/manifest.json}"
  CRON_MARKER="HYBRIDCVDV_GPU_MATRIX_${RUN_ID}"

  [[ -f "${CONFIG_PATH}" ]] || die "config file not found: ${CONFIG_PATH}"

  if [[ "${BUILD_DIR}" = /* ]]; then
    BINARY_PATH="${BUILD_DIR}/hybridcvdv_single_gpu_experiments"
  else
    BINARY_PATH="${ROOT_DIR}/${BUILD_DIR}/hybridcvdv_single_gpu_experiments"
  fi
  if [[ -n "${PRECHECK_CASE_FILTER}" && ! -x "${BINARY_PATH}" ]]; then
    die "preflight requires an existing executable binary: ${BINARY_PATH}"
  fi

  ENV_FILE="${ENV_FILE:-${STATE_DIR}/launcher.env}"

  write_env_file "${ENV_FILE}"
  install_crontab_entry "${ENV_FILE}"

  echo "Installed cron launcher."
  echo "run_id: ${RUN_ID}"
  echo "env_file: ${ENV_FILE}"
  echo "launcher_log: ${LAUNCHER_LOG}"
  echo "matrix_log: ${MATRIX_LOG}"
  echo "checkpoint: ${CHECKPOINT_PATH}"
  echo "status_json: ${STATUS_JSON}"
  echo "crontab marker: ${CRON_MARKER}"
  echo
  echo "Current crontab:"
  crontab -l
  echo
  echo "Running an immediate tick now..."
  tick_mode --env-file "${ENV_FILE}"
}

uninstall_mode() {
  require_crontab

  local env_file=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --env-file)
        [[ "$#" -ge 2 ]] || die "--env-file requires a path"
        env_file="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown uninstall argument: $1"
        ;;
    esac
  done
  [[ -n "${env_file}" ]] || die "uninstall requires --env-file"
  [[ -f "${env_file}" ]] || die "launcher env file not found: ${env_file}"

  # shellcheck disable=SC1090
  source "${env_file}"
  uninstall_crontab_entry
  echo "Removed cron launcher marker ${CRON_MARKER}"
}

status_mode() {
  local env_file=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --env-file)
        [[ "$#" -ge 2 ]] || die "--env-file requires a path"
        env_file="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown status argument: $1"
        ;;
    esac
  done
  [[ -n "${env_file}" ]] || die "status requires --env-file"
  [[ -f "${env_file}" ]] || die "launcher env file not found: ${env_file}"

  # shellcheck disable=SC1090
  source "${env_file}"

  load_state_paths

  local preflight_ok_file="${PREFLIGHT_OK_FILE}"
  local preflight_hold_file="${PREFLIGHT_HOLD_FILE}"
  local done_file="${DONE_FILE}"
  local preflight_pid_file="${PREFLIGHT_PID_FILE}"
  local preflight_rc_file="${PREFLIGHT_RC_FILE}"
  local matrix_pid_file="${MATRIX_PID_FILE}"
  local matrix_rc_file="${MATRIX_RC_FILE}"

  echo "run_id: ${RUN_ID}"
  echo "env_file: ${env_file}"
  echo "crontab marker: ${CRON_MARKER}"
  echo "binary_path: ${BINARY_PATH}"
  echo "checkpoint: ${CHECKPOINT_PATH}"
  echo "launcher_log: ${LAUNCHER_LOG}"
  echo "matrix_log: ${MATRIX_LOG}"
  echo "smoke_json: ${SMOKE_JSON}"
  echo "status_json: ${STATUS_JSON}"
  echo "manifest: ${MANIFEST_PATH}"
  echo
  echo "crontab:"
  crontab -l | sed -n "/# BEGIN ${CRON_MARKER}\$/,/# END ${CRON_MARKER}\$/p" || true
  echo
  echo "state flags:"
  [[ -f "${preflight_ok_file}" ]] && echo "  preflight.ok present" || echo "  preflight.ok missing"
  [[ -f "${preflight_hold_file}" ]] && echo "  preflight.hold present" || echo "  preflight.hold missing"
  [[ -f "${done_file}" ]] && echo "  matrix.done present" || echo "  matrix.done missing"
  if [[ -f "${preflight_pid_file}" ]]; then
    echo "  preflight.pid: $(cat "${preflight_pid_file}")"
  else
    echo "  preflight.pid missing"
  fi
  if [[ -f "${preflight_rc_file}" ]]; then
    echo "  preflight.rc: $(cat "${preflight_rc_file}")"
  else
    echo "  preflight.rc missing"
  fi
  if [[ -f "${matrix_pid_file}" ]]; then
    echo "  matrix.pid: $(cat "${matrix_pid_file}")"
  else
    echo "  matrix.pid missing"
  fi
  if [[ -f "${matrix_rc_file}" ]]; then
    echo "  matrix.rc: $(cat "${matrix_rc_file}")"
  else
    echo "  matrix.rc missing"
  fi
  echo
  echo "compute apps:"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
  echo
  echo "launcher log tail:"
  tail -n 20 "${LAUNCHER_LOG}" 2>/dev/null || true
  echo
  echo "matrix log tail:"
  tail -n 20 "${MATRIX_LOG}" 2>/dev/null || true
  echo
  if [[ -f "${SMOKE_JSON}" ]]; then
    echo "smoke json tail:"
    tail -n 40 "${SMOKE_JSON}"
    echo
  fi
  if [[ -f "${STATUS_JSON}" ]]; then
    echo "status json:"
    cat "${STATUS_JSON}"
    echo
  fi
  if [[ -f "${CHECKPOINT_PATH}" ]]; then
    echo "checkpoint tail:"
    tail -n 40 "${CHECKPOINT_PATH}"
    echo
  fi
  if [[ -f "${MANIFEST_PATH}" ]]; then
    echo "manifest tail:"
    tail -n 40 "${MANIFEST_PATH}"
  fi
}

main() {
  local mode="${1:-install}"
  if [[ "$#" -gt 0 ]]; then
    shift
  fi

  case "${mode}" in
    install)
      install_mode "$@"
      ;;
    tick)
      tick_mode "$@"
      ;;
    run-preflight)
      run_preflight_mode "$@"
      ;;
    run-matrix)
      run_matrix_mode "$@"
      ;;
    uninstall)
      uninstall_mode "$@"
      ;;
    status)
      status_mode "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      die "unknown mode: ${mode}"
      ;;
  esac
}

main "$@"
