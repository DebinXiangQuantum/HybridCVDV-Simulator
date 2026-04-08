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
  exec bash "${ROOT_DIR}/experiments/scripts/run_sc26_paper_eval.sh" "$@"
fi

readarray -t checkpoint_fields < <(python3 - "${CHECKPOINT_PATH}" <<'PY'
import json
import pathlib
import sys

checkpoint_path = pathlib.Path(sys.argv[1])
payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
summary_path = pathlib.Path(payload.get("summary_path", ""))
summary = {}
if summary_path.is_file():
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        summary = {}

print(payload.get("step", ""))
print(payload.get("run_id", ""))
print(payload.get("build_dir", ""))
print(payload.get("results_dir", ""))
print(payload.get("summary_path", ""))
print(payload.get("warmup_runs", summary.get("warmup_runs", "")))
print(payload.get("measured_runs", summary.get("measured_runs", "")))
print(payload.get("gaussian_symbolic_mode_limit", summary.get("gaussian_symbolic_mode_limit", "")))
print(payload.get("symbolic_branch_limit", summary.get("symbolic_branch_limit", "")))
PY
)

step="${checkpoint_fields[0]:-}"
checkpoint_run_id="${checkpoint_fields[1]:-}"
checkpoint_build_dir="${checkpoint_fields[2]:-}"
checkpoint_results_dir="${checkpoint_fields[3]:-}"
checkpoint_summary_path="${checkpoint_fields[4]:-}"
checkpoint_warmup_runs="${checkpoint_fields[5]:-}"
checkpoint_measured_runs="${checkpoint_fields[6]:-}"
checkpoint_gaussian_symbolic_mode_limit="${checkpoint_fields[7]:-}"
checkpoint_symbolic_branch_limit="${checkpoint_fields[8]:-}"

if [[ -z "${RUN_ID}" && -n "${checkpoint_run_id}" ]]; then
  RUN_ID="${checkpoint_run_id}"
fi
if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: checkpoint does not contain a run_id: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

BUILD_DIR="${BUILD_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
WARMUP_RUNS="${WARMUP_RUNS:-}"
MEASURED_RUNS="${MEASURED_RUNS:-}"
GAUSSIAN_SYMBOLIC_MODE_LIMIT="${GAUSSIAN_SYMBOLIC_MODE_LIMIT:-}"
SYMBOLIC_BRANCH_LIMIT="${SYMBOLIC_BRANCH_LIMIT:-}"

if [[ -z "${BUILD_DIR}" && -n "${checkpoint_build_dir}" ]]; then
  BUILD_DIR="${checkpoint_build_dir}"
fi
if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="build-h100"
fi

if [[ -z "${RESULTS_DIR}" && -n "${checkpoint_results_dir}" ]]; then
  RESULTS_DIR="${checkpoint_results_dir}"
fi
if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="${ROOT_DIR}/experiments/results/${RUN_ID}"
fi

if [[ -z "${SUMMARY_PATH}" && -n "${checkpoint_summary_path}" ]]; then
  SUMMARY_PATH="${checkpoint_summary_path}"
fi
if [[ -z "${SUMMARY_PATH}" ]]; then
  SUMMARY_PATH="${RESULTS_DIR}/summary.json"
fi

if [[ -z "${WARMUP_RUNS}" && -n "${checkpoint_warmup_runs}" ]]; then
  WARMUP_RUNS="${checkpoint_warmup_runs}"
fi
if [[ -z "${WARMUP_RUNS}" ]]; then
  WARMUP_RUNS="2"
fi

if [[ -z "${MEASURED_RUNS}" && -n "${checkpoint_measured_runs}" ]]; then
  MEASURED_RUNS="${checkpoint_measured_runs}"
fi
if [[ -z "${MEASURED_RUNS}" ]]; then
  MEASURED_RUNS="5"
fi

if [[ -z "${GAUSSIAN_SYMBOLIC_MODE_LIMIT}" && -n "${checkpoint_gaussian_symbolic_mode_limit}" ]]; then
  GAUSSIAN_SYMBOLIC_MODE_LIMIT="${checkpoint_gaussian_symbolic_mode_limit}"
fi
if [[ -z "${GAUSSIAN_SYMBOLIC_MODE_LIMIT}" ]]; then
  GAUSSIAN_SYMBOLIC_MODE_LIMIT="16"
fi

if [[ -z "${SYMBOLIC_BRANCH_LIMIT}" && -n "${checkpoint_symbolic_branch_limit}" ]]; then
  SYMBOLIC_BRANCH_LIMIT="${checkpoint_symbolic_branch_limit}"
fi
if [[ -z "${SYMBOLIC_BRANCH_LIMIT}" ]]; then
  SYMBOLIC_BRANCH_LIMIT="64"
fi

write_checkpoint() {
  local next_step="$1"
  local next_status="$2"
  python3 - "$CHECKPOINT_PATH" "$RUN_ID" "$next_step" "$next_status" "$BUILD_DIR" "$RESULTS_DIR" "$SUMMARY_PATH" "$WARMUP_RUNS" "$MEASURED_RUNS" "$GAUSSIAN_SYMBOLIC_MODE_LIMIT" "$SYMBOLIC_BRANCH_LIMIT" <<'PY'
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
        "results_dir": sys.argv[6],
        "summary_path": sys.argv[7],
        "warmup_runs": int(sys.argv[8]),
        "measured_runs": int(sys.argv[9]),
        "gaussian_symbolic_mode_limit": int(sys.argv[10]),
        "symbolic_branch_limit": int(sys.argv[11]),
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
  run_eval)
    cd "${ROOT_DIR}"
    cmd=(
      python3
      experiments/python/run_sc26_paper_eval.py
      --build-dir "${BUILD_DIR}"
      --results-dir "${RESULTS_DIR}"
      --summary-path "${SUMMARY_PATH}"
      --warmup-runs "${WARMUP_RUNS}"
      --measured-runs "${MEASURED_RUNS}"
      --gaussian-symbolic-mode-limit "${GAUSSIAN_SYMBOLIC_MODE_LIMIT}"
      --symbolic-branch-limit "${SYMBOLIC_BRANCH_LIMIT}"
    )
    if [[ "$#" -gt 0 ]]; then
      cmd+=("$@")
    fi
    set +e
    "${cmd[@]}"
    run_rc=$?
    set -e
    if [[ "${run_rc}" -ne 0 ]]; then
      write_checkpoint "run_eval" "error"
      exit "${run_rc}"
    fi
    write_checkpoint "completed" "ok"
    ;;
  *)
    export RUN_ID BUILD_DIR RESULTS_DIR SUMMARY_PATH CHECKPOINT_PATH
    export WARMUP_RUNS MEASURED_RUNS GAUSSIAN_SYMBOLIC_MODE_LIMIT SYMBOLIC_BRANCH_LIMIT
    exec bash "${ROOT_DIR}/experiments/scripts/run_sc26_paper_eval.sh" "$@"
    ;;
esac
