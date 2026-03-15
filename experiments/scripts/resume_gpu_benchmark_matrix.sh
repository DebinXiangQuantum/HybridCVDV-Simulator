#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/experiments/results/gpu_benchmark_checkpoint.json}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  exec "${ROOT_DIR}/experiments/scripts/run_gpu_benchmark_matrix.sh"
fi

step="$(python3 - "${CHECKPOINT_PATH}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("step", ""))
PY
)"

case "${step}" in
  completed)
    cat "${CHECKPOINT_PATH}"
    ;;
  run_benchmark_matrix)
    cd "${ROOT_DIR}"
    BASELINE_PYTHON="${BASELINE_PYTHON:-}"
    cmd=(
      python3
      experiments/python/run_gpu_benchmark_matrix.py
      --config "${CONFIG_PATH:-${ROOT_DIR}/experiments/configs/gpu_benchmark_matrix.json}"
      --build-dir "${BUILD_DIR:-build-h100}"
      --results-dir "${RESULTS_DIR:-${ROOT_DIR}/experiments/results}"
      --gpu-index "${GPU_INDEX:-0}"
      --telemetry-interval-ms "${TELEMETRY_INTERVAL_MS:-200}"
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
    python3 - "${CHECKPOINT_PATH}" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
payload = {
    "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "step": "completed",
    "status": "ok",
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    ;;
  *)
    exec "${ROOT_DIR}/experiments/scripts/run_gpu_benchmark_matrix.sh"
    ;;
esac
