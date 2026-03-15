#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build-h100}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/experiments/configs/gpu_benchmark_matrix.json}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/experiments/results}"
GPU_INDEX="${GPU_INDEX:-0}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"
BASELINE_PYTHON="${BASELINE_PYTHON:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/experiments/results/gpu_benchmark_checkpoint.json}"
BUILD_JOBS="${BUILD_JOBS:-8}"

write_checkpoint() {
  local step="$1"
  local status="$2"
  python3 - "$CHECKPOINT_PATH" "$step" "$status" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
payload = {
    "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "step": sys.argv[2],
    "status": sys.argv[3],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

cd "${ROOT_DIR}"

write_checkpoint "configure_build" "running"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

write_checkpoint "compile_internal_runner" "running"
cmake --build "${BUILD_DIR}" --target hybridcvdv_single_gpu_experiments -j "${BUILD_JOBS}"

write_checkpoint "run_benchmark_matrix" "running"
cmd=(
  python3
  experiments/python/run_gpu_benchmark_matrix.py
  --config "${CONFIG_PATH}"
  --build-dir "${BUILD_DIR}"
  --results-dir "${RESULTS_DIR}"
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

"${cmd[@]}"

write_checkpoint "completed" "ok"
