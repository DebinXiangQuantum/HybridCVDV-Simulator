#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/experiments/results/distributed_8xH800}"
RUNNER="${REPO_ROOT}/experiments/scripts/run_distributed_scaling.sh"
GPU_COUNTS="1,2,4,6,8"

mkdir -p "${RESULT_ROOT}"
exec 9>"${RESULT_ROOT}/distributed-pipeline.lock"
if ! flock -n 9; then
  echo "Another distributed pipeline is already running" >&2
  exit 1
fi

cd "${REPO_ROOT}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

manifest_complete() {
  python3 - "$1" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text())
raise SystemExit(0 if payload.get("status") == "complete" else 1)
PY
}

validate_manifest() {
  local manifest="$1"
  local policy="$2"
  local expected="${3:-0}"
  python3 - "${manifest}" "${policy}" "${expected}" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
policy = sys.argv[2]
expected = int(sys.argv[3])
if not path.exists():
    raise SystemExit(f"Missing manifest: {path}")
payload = json.loads(path.read_text())
artifacts = payload.get("artifacts", [])
counts = collections.Counter(item.get("status", "missing") for item in artifacts)
print(f"VALIDATE run={payload.get('run_id')} status={payload.get('status')} "
      f"artifacts={len(artifacts)} statuses={dict(counts)}")
if payload.get("status") != "complete":
    raise SystemExit(f"Incomplete run: {path}")
if expected and len(artifacts) != expected:
    raise SystemExit(f"Expected {expected} artifacts, found {len(artifacts)}: {path}")
if not artifacts:
    raise SystemExit(f"No artifacts: {path}")
if policy == "all-ok" and counts != {"ok": len(artifacts)}:
    raise SystemExit(f"Non-ok formal artifacts: {dict(counts)}")
if policy == "no-infra":
    fatal = {
        "configuration_error",
        "missing_telemetry",
        "incorrect_result",
        "crash",
        "crash_cuda",
    }
    found = fatal.intersection(counts)
    if found:
        raise SystemExit(f"Infrastructure/correctness failures: {sorted(found)}")
PY
}

run_step() {
  local run_id="$1"
  shift
  local manifest="${RESULT_ROOT}/${run_id}/manifest.json"
  if manifest_complete "${manifest}"; then
    log "SKIP completed ${run_id}"
    return
  fi
  log "START ${run_id}"
  "${RUNNER}" "$@" \
    --run-id "${run_id}" \
    --result-root "${RESULT_ROOT}" \
    --resume
  log "DONE ${run_id}"
}

log "PIPELINE start"
validate_manifest "${RESULT_ROOT}/phase-b-hybrid-fixed/manifest.json" all-ok 120

while screen -ls 2>/dev/null | rg -q '\.phase-b-atlas'; do
  log "WAIT phase-b-atlas"
  sleep 60
done
run_step phase-b-atlas \
  --systems atlas --gpu-counts "${GPU_COUNTS}" --phase strong \
  --warmup-runs 2 --measured-runs 10 --repetitions 3 \
  --telemetry-interval-ms 1000 --timeout-seconds 1800 --seed 2608
validate_manifest "${RESULT_ROOT}/phase-b-atlas/manifest.json" no-infra 360

run_step phase-c-hybrid \
  --systems hybridcvdv --gpu-counts "${GPU_COUNTS}" --phase capacity \
  --warmup-runs 2 --measured-runs 10 --repetitions 3 \
  --telemetry-interval-ms 1000 --timeout-seconds 1800 --seed 2609
validate_manifest "${RESULT_ROOT}/phase-c-hybrid/manifest.json" no-infra 360

run_step phase-c-atlas \
  --systems atlas --gpu-counts "${GPU_COUNTS}" --phase capacity \
  --warmup-runs 2 --measured-runs 10 --repetitions 3 \
  --telemetry-interval-ms 1000 --timeout-seconds 1800 --seed 2610
validate_manifest "${RESULT_ROOT}/phase-c-atlas/manifest.json" no-infra 360

for system in hybridcvdv atlas bqsim; do
  run_step "phase-d-${system}" \
    --systems "${system}" --gpu-counts "${GPU_COUNTS}" --phase throughput \
    --warmup-runs 0 --measured-runs 1 --repetitions 3 --total-tasks 256 \
    --telemetry-interval-ms 1000 --timeout-seconds 1800 --seed 2611
  validate_manifest "${RESULT_ROOT}/phase-d-${system}/manifest.json" all-ok 120
done

for system in hybridcvdv atlas bqsim; do
  # Feasibility uses a short per-case timeout so host-bound / pathological
  # cases fail fast. Known HybridCVDV host-bound patterns are skipped by the
  # runner as host_bound_skipped and never enter the formal Phase E matrix.
  run_step "phase-e-feasibility-${system}" \
    --systems "${system}" --gpu-counts "${GPU_COUNTS}" --phase full \
    --warmup-runs 0 --measured-runs 1 --repetitions 1 --total-tasks 1 \
    --telemetry-interval-ms 1000 --timeout-seconds 300 --seed 2612
  validate_manifest "${RESULT_ROOT}/phase-e-feasibility-${system}/manifest.json" complete

  run_step "phase-e-${system}" \
    --systems "${system}" --gpu-counts "${GPU_COUNTS}" --phase full \
    --successful-work-manifest \
      "${RESULT_ROOT}/phase-e-feasibility-${system}/manifest.json" \
    --warmup-runs 1 --measured-runs 3 --repetitions 1 --total-tasks 3 \
    --telemetry-interval-ms 1000 --timeout-seconds 1800 --seed 2613
  validate_manifest "${RESULT_ROOT}/phase-e-${system}/manifest.json" all-ok
done

FINAL_ROOT="${RESULT_ROOT}/final"
mkdir -p "${FINAL_ROOT}/hybridcvdv" "${FINAL_ROOT}/atlas" "${FINAL_ROOT}/bqsim"
ln -sfn "${RESULT_ROOT}/phase-b-hybrid-fixed/hybridcvdv/strong" \
  "${FINAL_ROOT}/hybridcvdv/strong"
ln -sfn "${RESULT_ROOT}/phase-b-atlas/atlas/strong" \
  "${FINAL_ROOT}/atlas/strong"
for system in hybridcvdv atlas; do
  ln -sfn "${RESULT_ROOT}/phase-c-${system}/${system}/capacity" \
    "${FINAL_ROOT}/${system}/capacity"
done
for system in hybridcvdv atlas bqsim; do
  ln -sfn "${RESULT_ROOT}/phase-d-${system}/${system}/throughput" \
    "${FINAL_ROOT}/${system}/throughput"
  ln -sfn "${RESULT_ROOT}/phase-e-${system}/${system}/full" \
    "${FINAL_ROOT}/${system}/full"
done

python3 experiments/python/merge_distributed_scaling.py \
  --run-root "${FINAL_ROOT}" --output-dir "${FINAL_ROOT}/merged"
python3 experiments/python/plot_distributed_scaling.py \
  --manifest "${FINAL_ROOT}/merged/manifest.json" \
  --output-dir "${FINAL_ROOT}/merged/plots"
log "PIPELINE complete artifacts=${FINAL_ROOT}"
