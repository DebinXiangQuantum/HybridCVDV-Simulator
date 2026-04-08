#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-sc26_paper_eval_$(date -u +%Y%m%d-%H%M%S)}"
BUILD_DIR="${BUILD_DIR:-build-h100}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/experiments/results}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${RUN_ID}}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_DIR}/summary.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESULTS_ROOT}/checkpoints/${RUN_ID}.json}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SKIP_BUILD="${SKIP_BUILD:-0}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
MEASURED_RUNS="${MEASURED_RUNS:-5}"
GAUSSIAN_SYMBOLIC_MODE_LIMIT="${GAUSSIAN_SYMBOLIC_MODE_LIMIT:-16}"
SYMBOLIC_BRANCH_LIMIT="${SYMBOLIC_BRANCH_LIMIT:-64}"

write_checkpoint() {
  local step="$1"
  local status="$2"
  python3 - "$CHECKPOINT_PATH" "$RUN_ID" "$step" "$status" "$BUILD_DIR" "$RESULTS_DIR" "$SUMMARY_PATH" "$WARMUP_RUNS" "$MEASURED_RUNS" "$GAUSSIAN_SYMBOLIC_MODE_LIMIT" "$SYMBOLIC_BRANCH_LIMIT" <<'PY'
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

summary_path = pathlib.Path(sys.argv[7])
completed_tasks = None
if summary_path.exists():
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tasks = summary.get("tasks", [])
        completed_tasks = sum(1 for task in tasks if isinstance(task, dict) and task.get("status") == "ok")
        payload["summary_status"] = summary.get("status")
    except Exception:
        pass

payload.update(
    {
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": sys.argv[2],
        "step": sys.argv[3],
        "status": sys.argv[4],
        "build_dir": sys.argv[5],
        "results_dir": sys.argv[6],
        "summary_path": sys.argv[7],
        "warmup_runs": int(sys.argv[8]),
        "measured_runs": int(sys.argv[9]),
        "gaussian_symbolic_mode_limit": int(sys.argv[10]),
        "symbolic_branch_limit": int(sys.argv[11]),
    }
)
if completed_tasks is not None:
    payload["completed_task_count"] = completed_tasks

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

cd "${ROOT_DIR}"
mkdir -p "${RESULTS_DIR}" "$(dirname "${CHECKPOINT_PATH}")"

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

write_checkpoint "run_eval" "running"
set +e
python3 experiments/python/run_sc26_paper_eval.py \
  --build-dir "${BUILD_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --summary-path "${SUMMARY_PATH}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --measured-runs "${MEASURED_RUNS}" \
  --gaussian-symbolic-mode-limit "${GAUSSIAN_SYMBOLIC_MODE_LIMIT}" \
  --symbolic-branch-limit "${SYMBOLIC_BRANCH_LIMIT}"
run_rc=$?
set -e

if [[ "${run_rc}" -ne 0 ]]; then
  write_checkpoint "run_eval" "error"
  exit "${run_rc}"
fi

write_checkpoint "completed" "ok"
