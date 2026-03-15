#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/experiments/results/gpu_baseline_envs_checkpoint.json}"

if [[ -f "${CHECKPOINT_PATH}" ]]; then
  python3 - "${CHECKPOINT_PATH}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "current_step": payload.get("current_step", ""),
            "status": payload.get("status", ""),
            "completed_steps": payload.get("completed_steps", []),
        },
        indent=2,
    )
)
PY
fi

exec "${ROOT_DIR}/experiments/scripts/setup_gpu_baseline_envs.sh"
