#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-sc26_gpu_benchmark_$(date -u +%Y%m%d-%H%M%S)}"
BUILD_DIR="${BUILD_DIR:-build-h100}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/experiments/configs/sc26_scaling.json}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/experiments/results}"
GPU_INDEX="${GPU_INDEX:-0}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"
BASELINE_PYTHON="${BASELINE_PYTHON:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/experiments/results/checkpoints/${RUN_ID}.json}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SKIP_BUILD="${SKIP_BUILD:-0}"

write_checkpoint() {
  local step="$1"
  local status="$2"
  python3 - "$CHECKPOINT_PATH" "$RUN_ID" "$step" "$status" "$BUILD_DIR" "$CONFIG_PATH" "$RESULTS_DIR" <<'PY'
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

cd "${ROOT_DIR}"
mkdir -p "${RESULTS_DIR}" "$(dirname "${CHECKPOINT_PATH}")"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
  write_checkpoint "configure_build" "running"
  cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

  write_checkpoint "compile_internal_runner" "running"
  cmake --build "${BUILD_DIR}" --target hybridcvdv_single_gpu_experiments -j "${BUILD_JOBS}"
fi

if [[ "${BUILD_DIR}" = /* ]]; then
  BINARY_PATH="${BUILD_DIR}/hybridcvdv_single_gpu_experiments"
else
  BINARY_PATH="${ROOT_DIR}/${BUILD_DIR}/hybridcvdv_single_gpu_experiments"
fi
if [[ ! -f "${BINARY_PATH}" ]]; then
  echo "ERROR: Built benchmark binary not found: ${BINARY_PATH}" >&2
  write_checkpoint "compile_internal_runner" "error"
  exit 1
fi

write_checkpoint "run_benchmark_matrix" "running"
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
