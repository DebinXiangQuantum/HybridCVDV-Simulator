#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  if [[ -n "${RUN_ID}" ]]; then
    CHECKPOINT_PATH="${ROOT_DIR}/experiments/results/checkpoints/${RUN_ID}.json"
  else
    echo "ERROR: resume requires CHECKPOINT_PATH or RUN_ID so the checkpoint file can be located" >&2
    exit 1
  fi
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  export RUN_ID
  exec bash "${ROOT_DIR}/experiments/scripts/run_gpu_benchmark_matrix.sh"
fi

readarray -t checkpoint_fields < <(python3 - "${CHECKPOINT_PATH}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("step", ""))
print(payload.get("run_id", ""))
print(payload.get("build_dir", ""))
print(payload.get("config_path", ""))
print(payload.get("results_dir", ""))
PY
)

step="${checkpoint_fields[0]:-}"
checkpoint_run_id="${checkpoint_fields[1]:-}"
checkpoint_build_dir="${checkpoint_fields[2]:-}"
checkpoint_config_path="${checkpoint_fields[3]:-}"
checkpoint_results_dir="${checkpoint_fields[4]:-}"

if [[ -z "${RUN_ID}" && -n "${checkpoint_run_id}" ]]; then
  RUN_ID="${checkpoint_run_id}"
fi
if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: checkpoint does not contain a run_id: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

BUILD_DIR="${BUILD_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
RESULTS_DIR="${RESULTS_DIR:-}"
GPU_INDEX="${GPU_INDEX:-0}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"
BASELINE_PYTHON="${BASELINE_PYTHON:-}"

if [[ -z "${BUILD_DIR}" && -n "${checkpoint_build_dir}" ]]; then
  BUILD_DIR="${checkpoint_build_dir}"
fi
if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="build-h100"
fi

if [[ -z "${CONFIG_PATH}" && -n "${checkpoint_config_path}" ]]; then
  CONFIG_PATH="${checkpoint_config_path}"
fi
if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${ROOT_DIR}/experiments/configs/sc26_scaling.json"
fi

if [[ -z "${RESULTS_DIR}" && -n "${checkpoint_results_dir}" ]]; then
  RESULTS_DIR="${checkpoint_results_dir}"
fi
if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="${ROOT_DIR}/experiments/results"
fi

write_checkpoint() {
  local next_step="$1"
  local next_status="$2"
  python3 - "$CHECKPOINT_PATH" "$RUN_ID" "$next_step" "$next_status" "$BUILD_DIR" "$CONFIG_PATH" "$RESULTS_DIR" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
payload.update(
    {
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": sys.argv[2],
        "step": sys.argv[3],
        "status": sys.argv[4],
        "build_dir": sys.argv[5],
        "config_path": sys.argv[6],
        "results_dir": sys.argv[7],
    }
)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

case "${step}" in
  completed)
    cat "${CHECKPOINT_PATH}"
    ;;
  run_benchmark_matrix)
    cd "${ROOT_DIR}"
    cmd=(
      python3
      experiments/python/run_gpu_benchmark_matrix.py
      --config "${CONFIG_PATH}"
      --build-dir "${BUILD_DIR}"
      --results-dir "${RESULTS_DIR}"
      --run-id "${RUN_ID}"
      --checkpoint-path "${CHECKPOINT_PATH}"
      --gpu-index "${GPU_INDEX}"
      --telemetry-interval-ms "${TELEMETRY_INTERVAL_MS}"
    )
    if [[ -n "${BASELINE_PYTHON}" ]]; then
      cmd+=(--baseline-python "${BASELINE_PYTHON}")
    fi
    if [[ -n "${CASE_FILTER:-}" ]]; then
      cmd+=(--case-filter "${CASE_FILTER}")
    fi
    if [[ -n "${BACKEND:-}" ]]; then
      cmd+=(--backend "${BACKEND}")
    fi
    if [[ "$#" -gt 0 ]]; then
      cmd+=("$@")
    fi
    set +e
    "${cmd[@]}"
    run_rc=$?
    set -e
    if [[ "${run_rc}" -ne 0 ]]; then
      write_checkpoint "run_benchmark_matrix" "error"
      exit "${run_rc}"
    fi
    write_checkpoint "completed" "ok"
    ;;
  *)
    export RUN_ID BUILD_DIR CONFIG_PATH RESULTS_DIR CHECKPOINT_PATH
    exec bash "${ROOT_DIR}/experiments/scripts/run_gpu_benchmark_matrix.sh" "$@"
    ;;
esac
