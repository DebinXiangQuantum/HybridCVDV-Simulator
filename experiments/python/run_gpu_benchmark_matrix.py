#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "gpu_benchmark_matrix.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BASELINE_PYTHON = REPO_ROOT / "baselines" / "venv" / "bin" / "python"
NVIDIA_SMI = "nvidia-smi"
MAX_LOG_CHARS = 4000


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def truncate_text(text: str, max_chars: int = MAX_LOG_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    keep = max_chars - 64
    return text[:keep] + "\n...[truncated]...\n", True


def choose_baseline_python(explicit_python: str | None) -> str:
    if explicit_python:
        return explicit_python
    if DEFAULT_BASELINE_PYTHON.exists():
        return str(DEFAULT_BASELINE_PYTHON)
    return sys.executable


def resolve_python_path(path_str: str) -> str:
    candidate = pathlib.Path(path_str)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate)


def choose_backend_python(
    backend: str,
    explicit_python: str | None,
    backend_python_map: dict[str, Any] | None,
) -> str:
    if explicit_python:
        return explicit_python

    if backend_python_map:
        mapped = backend_python_map.get(backend)
        if mapped:
            return resolve_python_path(str(mapped))

    return choose_baseline_python(None)


def query_gpu_metadata(gpu_index: int) -> dict[str, Any]:
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "power.limit",
    ]
    completed = subprocess.run(
        [NVIDIA_SMI, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", f"--id={gpu_index}"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }

    values = [value.strip() for value in completed.stdout.strip().split(",")]
    return {
        "available": True,
        "index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "driver_version": values[3],
        "memory_total_mb": float(values[4]),
        "power_limit_w": float(values[5]),
    }


def sample_gpu(gpu_index: int) -> dict[str, float] | None:
    fields = [
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "power.draw",
        "temperature.gpu",
        "clocks.sm",
    ]
    completed = subprocess.run(
        [NVIDIA_SMI, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", f"--id={gpu_index}"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None

    values = [value.strip() for value in completed.stdout.strip().split(",")]
    return {
        "timestamp_unix": time.time(),
        "gpu_util_pct": float(values[0]),
        "memory_util_pct": float(values[1]),
        "memory_used_mb": float(values[2]),
        "power_draw_w": float(values[3]),
        "temperature_c": float(values[4]),
        "sm_clock_mhz": float(values[5]),
    }


def summarize_telemetry(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {"gpu_sample_count": 0.0}

    def avg(key: str) -> float:
        return statistics.fmean(sample[key] for sample in samples)

    def peak(key: str) -> float:
        return max(sample[key] for sample in samples)

    return {
        "gpu_sample_count": float(len(samples)),
        "gpu_avg_util_pct": avg("gpu_util_pct"),
        "gpu_peak_util_pct": peak("gpu_util_pct"),
        "gpu_avg_memory_util_pct": avg("memory_util_pct"),
        "gpu_peak_memory_util_pct": peak("memory_util_pct"),
        "gpu_avg_memory_used_mb": avg("memory_used_mb"),
        "gpu_peak_memory_used_mb": peak("memory_used_mb"),
        "gpu_avg_power_draw_w": avg("power_draw_w"),
        "gpu_peak_power_draw_w": peak("power_draw_w"),
        "gpu_peak_temperature_c": peak("temperature_c"),
        "gpu_peak_sm_clock_mhz": peak("sm_clock_mhz"),
    }


def run_with_telemetry(
    cmd: list[str],
    cwd: pathlib.Path,
    gpu_index: int,
    sample_interval_s: float,
) -> tuple[int, str, str, list[dict[str, float]], float]:
    started_at = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    samples: list[dict[str, float]] = []
    while process.poll() is None:
        sample = sample_gpu(gpu_index)
        if sample is not None:
            samples.append(sample)
        time.sleep(sample_interval_s)

    sample = sample_gpu(gpu_index)
    if sample is not None:
        samples.append(sample)

    stdout, stderr = process.communicate()
    duration_ms = (time.perf_counter() - started_at) * 1000.0
    return process.returncode, stdout, stderr, samples, duration_ms


def update_case_output(
    output_path: pathlib.Path,
    telemetry_summary: dict[str, float],
    telemetry_samples_path: pathlib.Path,
    command: list[str],
    returncode: int,
    duration_ms: float,
) -> None:
    if not output_path.exists():
        return

    payload = read_json(output_path)
    payload["telemetry_summary"] = telemetry_summary
    payload["telemetry_samples_path"] = str(telemetry_samples_path)
    payload["runner"] = {
        "command": command,
        "returncode": returncode,
        "wall_time_ms": duration_ms,
    }
    results = payload.get("results", [])
    if len(results) == 1 and isinstance(results[0], dict):
        metrics = results[0].setdefault("metrics", {})
        for key, value in telemetry_summary.items():
            metrics[key] = value
        metrics["runner_wall_time_ms"] = duration_ms
    write_json(output_path, payload)


def should_run_case(case: dict[str, Any], case_filter: str | None, selected_backend: str | None) -> bool:
    if case_filter and case_filter not in case["name"]:
        return False
    if selected_backend and selected_backend not in case["backends"]:
        return False
    return True


def build_internal_command(
    build_dir: pathlib.Path,
    case: dict[str, Any],
    output_path: pathlib.Path,
    gaussian_symbolic_mode_limit: int,
) -> list[str]:
    binary_name = case.get("internal_binary", "hybridcvdv_single_gpu_experiments")
    binary_path = build_dir / binary_name
    return [
        str(binary_path),
        "--suite",
        case.get("internal_suite", "scaling"),
        "--name-filter",
        case["internal_name_filter"],
        "--gaussian-symbolic-mode-limit",
        str(case.get("gaussian_symbolic_mode_limit", gaussian_symbolic_mode_limit)),
        "--output",
        str(output_path),
    ]


def build_baseline_command(
    baseline_python: str,
    backend: str,
    case: dict[str, Any],
    output_path: pathlib.Path,
) -> list[str]:
    runner = REPO_ROOT / "experiments" / "python" / "run_baseline_backend.py"
    command = [
        baseline_python,
        str(runner),
        "--backend",
        backend,
        "--workload",
        case["workload"],
        "--cutoff",
        str(case["cutoff"]),
        "--num-modes",
        str(case["num_modes"]),
        "--layers",
        str(case.get("layers", 2)),
        "--timesteps",
        str(case.get("timesteps", 5)),
        "--warmup-runs",
        str(case.get("warmup_runs", 2)),
        "--measured-runs",
        str(case.get("measured_runs", 5)),
        "--squeezing-r",
        str(case.get("squeezing_r", 0.5)),
        "--displacement-scale",
        str(case.get("displacement_scale", 1.0)),
        "--j-coupling",
        str(case.get("j_coupling", 1.0)),
        "--omega-r",
        str(case.get("omega_r", 1.0)),
        "--tau",
        str(case.get("tau", 0.1)),
        "--output",
        str(output_path),
    ]
    return command


def run_case_backend(
    case: dict[str, Any],
    backend: str,
    build_dir: pathlib.Path,
    run_dir: pathlib.Path,
    baseline_python: str | None,
    backend_python_map: dict[str, Any] | None,
    gpu_index: int,
    sample_interval_s: float,
    gaussian_symbolic_mode_limit: int,
) -> dict[str, Any]:
    output_name = f"{case['name']}__{backend}.json"
    output_path = run_dir / output_name
    telemetry_samples_path = run_dir / "telemetry" / f"{case['name']}__{backend}.json"

    if backend == "hybridcvdv":
        cmd = build_internal_command(build_dir, case, output_path, gaussian_symbolic_mode_limit)
        if not pathlib.Path(cmd[0]).exists():
            return {
                "case_name": case["name"],
                "backend": backend,
                "status": "missing_binary",
                "command": cmd,
                "output_path": str(output_path),
                "telemetry_path": str(telemetry_samples_path),
                "returncode": None,
                "stdout": "",
                "stderr": f"missing internal binary: {cmd[0]}",
            }
    else:
        backend_python = choose_backend_python(backend, baseline_python, backend_python_map)
        cmd = build_baseline_command(backend_python, backend, case, output_path)

    returncode, stdout, stderr, samples, duration_ms = run_with_telemetry(
        cmd=cmd,
        cwd=REPO_ROOT,
        gpu_index=gpu_index,
        sample_interval_s=sample_interval_s,
    )
    stdout_text, stdout_truncated = truncate_text(stdout)
    stderr_text, stderr_truncated = truncate_text(stderr)
    telemetry_summary = summarize_telemetry(samples)
    write_json(telemetry_samples_path, {"samples": samples, "summary": telemetry_summary})
    update_case_output(
        output_path=output_path,
        telemetry_summary=telemetry_summary,
        telemetry_samples_path=telemetry_samples_path,
        command=cmd,
        returncode=returncode,
        duration_ms=duration_ms,
    )
    case_status = "ok" if returncode == 0 else "error"
    output_status = None
    if output_path.exists():
        try:
            payload = read_json(output_path)
            results = payload.get("results", [])
            if payload.get("status") in {"unsupported", "error"}:
                output_status = payload["status"]
            elif results:
                statuses = {result.get("status", "ok") for result in results if isinstance(result, dict)}
                if "error" in statuses:
                    output_status = "error"
                elif "unsupported" in statuses and statuses == {"unsupported"}:
                    output_status = "unsupported"
        except Exception:
            output_status = None
    if output_status is not None:
        case_status = output_status
    return {
        "case_name": case["name"],
        "backend": backend,
        "status": case_status,
        "command": cmd,
        "output_path": str(output_path),
        "telemetry_path": str(telemetry_samples_path),
        "returncode": returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "wall_time_ms": duration_ms,
        "telemetry_summary": telemetry_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a per-case GPU benchmark matrix with nvidia-smi telemetry.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--baseline-python", default=None)
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--telemetry-interval-ms", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(pathlib.Path(args.config))
    build_dir = pathlib.Path(args.build_dir)
    results_root = pathlib.Path(args.results_dir)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu_index = args.gpu_index
    telemetry_interval_ms = args.telemetry_interval_ms or config.get("telemetry_interval_ms", 200)
    sample_interval_s = telemetry_interval_ms / 1000.0
    baseline_python = args.baseline_python
    backend_python_map = config.get("backend_python_map")
    gaussian_symbolic_mode_limit = config.get("gaussian_symbolic_mode_limit", 16)

    manifest = {
        "schema_version": "2.0",
        "generated_at_utc": iso_now(),
        "run_id": run_id,
        "config_path": str(pathlib.Path(args.config).resolve()),
        "gpu_index": gpu_index,
        "telemetry_interval_ms": telemetry_interval_ms,
        "gaussian_symbolic_mode_limit": gaussian_symbolic_mode_limit,
        "device": query_gpu_metadata(gpu_index),
        "artifacts": [],
    }

    for case in config.get("cases", []):
        if not should_run_case(case, args.case_filter, args.backend):
            continue
        for backend in case["backends"]:
            if args.backend and backend != args.backend:
                continue
            artifact = run_case_backend(
                case=case,
                backend=backend,
                build_dir=build_dir,
                run_dir=run_dir,
                baseline_python=baseline_python,
                backend_python_map=backend_python_map,
                gpu_index=gpu_index,
                sample_interval_s=sample_interval_s,
                gaussian_symbolic_mode_limit=gaussian_symbolic_mode_limit,
            )
            manifest["artifacts"].append(artifact)

    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote GPU benchmark manifest to {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
