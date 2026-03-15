#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/experiments/results/gpu_baseline_envs_checkpoint.json}"
SF_ENV="${SF_ENV:-${ROOT_DIR}/.venv-sf-gpu}"
MM_ENV="${MM_ENV:-${ROOT_DIR}/.venv-mm-gpu}"
SF_PACKAGE="${SF_PACKAGE:-strawberryfields}"
TF_PACKAGE="${TF_PACKAGE:-tensorflow[and-cuda]}"
MM_PACKAGE="${MM_PACKAGE:-mrmustard}"
JAX_PACKAGE="${JAX_PACKAGE:-jax[cuda12]}"

checkpoint_query() {
  local step="$1"
  python3 - "$CHECKPOINT_PATH" "$step" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
step = sys.argv[2]
if not path.exists():
    print("0")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
print("1" if step in payload.get("completed_steps", []) else "0")
PY
}

checkpoint_update() {
  local step="$1"
  local status="$2"
  python3 - "$CHECKPOINT_PATH" "$step" "$status" "$SF_ENV" "$MM_ENV" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
step = sys.argv[2]
status = sys.argv[3]
sf_env = sys.argv[4]
mm_env = sys.argv[5]
if path.exists():
    payload = json.loads(path.read_text(encoding="utf-8"))
else:
    payload = {"completed_steps": []}
payload["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
payload["current_step"] = step
payload["status"] = status
payload["envs"] = {
    "strawberryfields_tf": sf_env,
    "mrmustard_jax": mm_env,
}
if status == "ok" and step not in payload["completed_steps"]:
    payload["completed_steps"].append(step)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_step() {
  local step="$1"
  shift
  if [[ "$(checkpoint_query "${step}")" == "1" ]]; then
    echo "[skip] ${step}"
    return 0
  fi
  echo "[run] ${step}"
  checkpoint_update "${step}" "running"
  "$@"
  checkpoint_update "${step}" "ok"
}

ensure_uv() {
  if python3 -m uv --version >/dev/null 2>&1; then
    python3 -m uv --version
    return 0
  fi
  python3 -m pip install --user uv
  python3 -m uv --version
}

create_env() {
  local env_path="$1"
  if [[ -x "${env_path}/bin/python" ]]; then
    "${env_path}/bin/python" --version
    return 0
  fi
  python3 -m uv venv "${env_path}"
  "${env_path}/bin/python" --version
}

install_sf_stack() {
  python3 -m uv pip install --python "${SF_ENV}/bin/python" --upgrade pip setuptools wheel
  python3 -m uv pip install --python "${SF_ENV}/bin/python" "${SF_PACKAGE}" "${TF_PACKAGE}"
}

verify_sf_stack() {
  "${SF_ENV}/bin/python" - <<'PY'
import json
import strawberryfields as sf
import tensorflow as tf

gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
payload = {
    "strawberryfields_version": getattr(sf, "__version__", "unknown"),
    "tensorflow_version": getattr(tf, "__version__", "unknown"),
    "visible_gpu_devices": gpus,
}
print(json.dumps(payload, indent=2))
if not gpus:
    raise SystemExit("TensorFlow did not detect any GPU devices")
PY
}

install_mm_stack() {
  python3 -m uv pip install --python "${MM_ENV}/bin/python" --upgrade pip setuptools wheel
  python3 -m uv pip install --python "${MM_ENV}/bin/python" "${MM_PACKAGE}" "${JAX_PACKAGE}"
}

verify_mm_stack() {
  "${MM_ENV}/bin/python" - <<'PY'
import json
import jax
import mrmustard
from mrmustard import math as mmath

gpus = [str(device) for device in jax.devices() if getattr(device, "platform", "") == "gpu"]
if hasattr(mmath, "change_backend"):
    mmath.change_backend("jax")
payload = {
    "mrmustard_version": getattr(mrmustard, "__version__", "unknown"),
    "jax_version": getattr(jax, "__version__", "unknown"),
    "visible_gpu_devices": gpus,
    "default_backend": jax.default_backend(),
}
print(json.dumps(payload, indent=2))
if not gpus:
    raise SystemExit("JAX did not detect any GPU devices")
PY
}

cd "${ROOT_DIR}"

run_step "ensure_uv" ensure_uv
run_step "create_sf_env" create_env "${SF_ENV}"
run_step "install_sf_stack" install_sf_stack
run_step "verify_sf_stack" verify_sf_stack
run_step "create_mm_env" create_env "${MM_ENV}"
run_step "install_mm_stack" install_mm_stack
run_step "verify_mm_stack" verify_mm_stack

checkpoint_update "completed" "ok"
