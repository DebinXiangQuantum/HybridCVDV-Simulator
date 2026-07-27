#!/usr/bin/env python3
"""Shared schema and helpers for distributed scaling experiments."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import time
from typing import Any, Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "3.0"
RESULT_STATUSES = {
    "ok",
    "oom_single_gpu_pool",
    "oom_single_state_too_large",
    "oom_aggregate",
    "unsupported_gpu_count",
    "unsupported_backend",
    "timeout",
    "host_bound_skipped",
    "crash_cuda",
    "crash_host",
    "incorrect_result",
    "missing_telemetry",
    "configuration_error",
}


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: Iterable[str], cwd: pathlib.Path = REPO_ROOT) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    output = (completed.stdout or completed.stderr).strip()
    return {"ok": completed.returncode == 0, "returncode": completed.returncode, "output": output}


def classify_failure(returncode: int | None, stdout: str, stderr: str, timed_out: bool = False) -> str:
    if timed_out:
        return "timeout"
    text = f"{stdout}\n{stderr}".lower()
    if "single state" in text and ("out of memory" in text or "too large" in text):
        return "oom_single_state_too_large"
    if "aggregate" in text and "out of memory" in text:
        return "oom_aggregate"
    if "out of memory" in text or "cudamalloc" in text:
        return "oom_single_gpu_pool"
    if "unsupported_gpu_count" in text or "power of two" in text:
        return "unsupported_gpu_count"
    if "unsupported" in text:
        return "unsupported_backend"
    if any(token in text for token in ("illegal memory access", "cuda error", "nccl error", "cusv error")):
        return "crash_cuda"
    if returncode in (0, None):
        return "configuration_error"
    return "crash_host"


def empty_result(
    *,
    system: str,
    case: dict[str, Any],
    phase: str,
    gpu_ids: list[int],
    warmup_runs: int,
    measured_runs: int,
    repetition: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": iso_now(),
        "system": system,
        "case_name": case["name"],
        "phase": phase,
        "gpu_count": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "process_repetition": repetition,
        "status": "configuration_error",
        "cutoff": case.get("cutoff"),
        "num_modes": case.get("num_modes"),
        "num_qubits": case.get("num_qubits"),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "timing": {},
        "throughput": {},
        "memory": {"per_gpu": {}, "aggregate": {}},
        "communication": {},
        "correctness": {},
        "telemetry": {"per_gpu": {}, "aggregate": {}},
        "environment": {},
    }


def validate_result(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "system",
        "case_name",
        "phase",
        "gpu_count",
        "gpu_ids",
        "status",
        "timing",
        "throughput",
        "memory",
        "communication",
        "correctness",
        "telemetry",
        "environment",
    )
    for field in required:
        if field not in payload:
            errors.append(f"missing field: {field}")
    if payload.get("status") not in RESULT_STATUSES:
        errors.append(f"invalid status: {payload.get('status')}")
    gpu_ids = payload.get("gpu_ids")
    if isinstance(gpu_ids, list) and payload.get("gpu_count") != len(gpu_ids):
        errors.append("gpu_count does not match gpu_ids")
    elif not isinstance(gpu_ids, list):
        errors.append("gpu_ids must be a list")
    return errors
