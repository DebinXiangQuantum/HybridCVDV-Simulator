#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "sc26_scaling.json"
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
    encoded = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(encoded, encoding="utf-8")
    tmp_path.replace(path)


def append_log_message(text: str, message: str) -> str:
    if not text.strip():
        return message
    return f"{text.rstrip()}\n{message}"


def format_nvidia_smi_error(raw_error: str) -> str:
    message = raw_error.strip() or "nvidia-smi failed"
    if "Insufficient Permissions" not in message:
        return message

    capability_path = pathlib.Path("/dev/nvidia-caps/nvidia-cap1")
    capability_hint = f"{capability_path} missing"
    try:
        capability_stat = capability_path.stat()
        capability_hint = (
            f"{capability_path} mode={oct(capability_stat.st_mode & 0o777)} "
            f"owner={capability_stat.st_uid}:{capability_stat.st_gid}"
        )
    except OSError:
        pass

    return (
        message
        + ". NVML access is blocked for the current user; "
        + capability_hint
        + ". Grant `nvidia-smi`/NVML access (or run with sufficient privileges) before collecting "
        + "power/utilization telemetry."
    )


def artifact_key(case_name: str, backend: str) -> str:
    return f"{case_name}::{backend}"


def artifact_key_from_artifact(artifact: dict[str, Any]) -> str | None:
    case_name = artifact.get("case_name")
    backend = artifact.get("backend")
    if not isinstance(case_name, str) or not isinstance(backend, str):
        return None
    return artifact_key(case_name, backend)


def validate_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("config must contain a non-empty 'cases' list")

    seen_names: set[str] = set()
    seen_internal_filters: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case #{index} is not a JSON object")

        name = case.get("name")
        backends = case.get("backends")
        if not isinstance(name, str) or not name:
            raise ValueError(f"case #{index} is missing a non-empty 'name'")
        if name in seen_names:
            raise ValueError(f"duplicate case name in config: {name}")
        seen_names.add(name)

        if not isinstance(backends, list) or not backends or not all(isinstance(backend, str) and backend for backend in backends):
            raise ValueError(f"case '{name}' is missing a non-empty 'backends' list")

        if "hybridcvdv" in backends:
            internal_name_filter = case.get("internal_name_filter")
            if not isinstance(internal_name_filter, str) or not internal_name_filter:
                raise ValueError(f"case '{name}' is missing 'internal_name_filter' for hybridcvdv")
            if internal_name_filter in seen_internal_filters:
                raise ValueError(f"duplicate internal_name_filter in config: {internal_name_filter}")
            seen_internal_filters.add(internal_name_filter)

    return cases


def build_planned_artifacts(
    cases: list[dict[str, Any]],
    case_filter: str | None,
    selected_backend: str | None,
) -> list[tuple[dict[str, Any], str]]:
    planned: list[tuple[dict[str, Any], str]] = []
    seen_keys: set[str] = set()
    for case in cases:
        if not should_run_case(case, case_filter, selected_backend):
            continue
        for backend in case["backends"]:
            if selected_backend and backend != selected_backend:
                continue
            key = artifact_key(case["name"], backend)
            if key in seen_keys:
                raise ValueError(f"duplicate case/backend selection in config: {key}")
            planned.append((case, backend))
            seen_keys.add(key)
    return planned


def select_artifact_slice(
    planned_artifacts: list[tuple[dict[str, Any], str]],
    artifact_start: int | None,
    artifact_end: int | None,
) -> list[tuple[dict[str, Any], str]]:
    if artifact_start is None and artifact_end is None:
        return planned_artifacts

    total = len(planned_artifacts)
    start = artifact_start or 1
    end = artifact_end or total
    if start <= 0 or end <= 0:
        raise ValueError("artifact index bounds must be positive")
    if start > end:
        raise ValueError("artifact-start must be <= artifact-end")
    if start > total:
        raise ValueError(f"artifact-start {start} exceeds planned artifact count {total}")
    if end > total:
        raise ValueError(f"artifact-end {end} exceeds planned artifact count {total}")
    return planned_artifacts[start - 1 : end]


def make_manifest(
    run_id: str,
    config_path: pathlib.Path,
    gpu_index: int,
    telemetry_interval_ms: int,
    gaussian_symbolic_mode_limit: int,
    use_interaction_picture: bool,
    device: dict[str, Any],
    planned_artifact_count: int,
    case_filter: str | None,
    selected_backend: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "2.1",
        "generated_at_utc": iso_now(),
        "updated_at_utc": iso_now(),
        "run_id": run_id,
        "config_path": str(config_path.resolve()),
        "gpu_index": gpu_index,
        "telemetry_interval_ms": telemetry_interval_ms,
        "gaussian_symbolic_mode_limit": gaussian_symbolic_mode_limit,
        "use_interaction_picture": use_interaction_picture,
        "case_filter": case_filter,
        "selected_backend": selected_backend,
        "planned_artifact_count": planned_artifact_count,
        "completed_artifact_count": 0,
        "status_counts": {},
        "status": "running",
        "device": device,
        "artifacts": [],
    }


def update_manifest_summary(manifest: dict[str, Any], planned_artifact_count: int) -> None:
    artifacts = manifest.setdefault("artifacts", [])
    status_counts: dict[str, int] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        status = str(artifact.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    manifest["updated_at_utc"] = iso_now()
    manifest["planned_artifact_count"] = planned_artifact_count
    manifest["completed_artifact_count"] = len(artifacts)
    manifest["status_counts"] = status_counts
    if len(artifacts) < planned_artifact_count:
        manifest["status"] = "running"
    else:
        manifest["status"] = "ok" if status_counts.get("ok", 0) == planned_artifact_count else "error"


def load_or_initialize_manifest(
    manifest_path: pathlib.Path,
    run_id: str,
    config_path: pathlib.Path,
    gpu_index: int,
    telemetry_interval_ms: int,
    gaussian_symbolic_mode_limit: int,
    use_interaction_picture: bool,
    device: dict[str, Any],
    planned_artifact_count: int,
    case_filter: str | None,
    selected_backend: str | None,
) -> dict[str, Any]:
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        existing_run_id = manifest.get("run_id")
        if existing_run_id not in {None, run_id}:
            raise ValueError(f"existing manifest run_id mismatch: expected {run_id}, found {existing_run_id}")
        if not isinstance(manifest.get("artifacts"), list):
            raise ValueError(f"existing manifest has an invalid 'artifacts' payload: {manifest_path}")
        manifest["run_id"] = run_id
        manifest["config_path"] = str(config_path.resolve())
        manifest["gpu_index"] = gpu_index
        manifest["telemetry_interval_ms"] = telemetry_interval_ms
        manifest["gaussian_symbolic_mode_limit"] = gaussian_symbolic_mode_limit
        manifest["use_interaction_picture"] = use_interaction_picture
        manifest["case_filter"] = case_filter
        manifest["selected_backend"] = selected_backend
        manifest["device"] = device
    else:
        manifest = make_manifest(
            run_id=run_id,
            config_path=config_path,
            gpu_index=gpu_index,
            telemetry_interval_ms=telemetry_interval_ms,
            gaussian_symbolic_mode_limit=gaussian_symbolic_mode_limit,
            use_interaction_picture=use_interaction_picture,
            device=device,
            planned_artifact_count=planned_artifact_count,
            case_filter=case_filter,
            selected_backend=selected_backend,
        )

    update_manifest_summary(manifest, planned_artifact_count)
    write_json(manifest_path, manifest)
    return manifest


def merge_artifact(manifest: dict[str, Any], artifact: dict[str, Any]) -> None:
    key = artifact_key_from_artifact(artifact)
    if key is None:
        raise ValueError(f"artifact is missing a case_name/backend key: {artifact}")

    artifacts = manifest.setdefault("artifacts", [])
    for index, existing in enumerate(artifacts):
        if isinstance(existing, dict) and artifact_key_from_artifact(existing) == key:
            artifacts[index] = artifact
            return
    artifacts.append(artifact)


def write_progress_checkpoint(
    checkpoint_path: pathlib.Path | None,
    run_id: str,
    build_dir: pathlib.Path,
    config_path: pathlib.Path,
    run_dir: pathlib.Path,
    manifest: dict[str, Any],
    planned_artifact_count: int,
    step: str = "run_benchmark_matrix",
    status: str | None = None,
    last_artifact: dict[str, Any] | None = None,
) -> None:
    if checkpoint_path is None:
        return

    completed_artifact_keys: list[str] = []
    for artifact in manifest.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        key = artifact_key_from_artifact(artifact)
        if key is not None:
            completed_artifact_keys.append(key)

    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "updated_at_utc": iso_now(),
        "step": step,
        "status": status or str(manifest.get("status", "running")),
        "run_id": run_id,
        "build_dir": str(build_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "planned_artifact_count": planned_artifact_count,
        "completed_artifact_count": len(completed_artifact_keys),
        "completed_artifact_keys": completed_artifact_keys,
        "status_counts": manifest.get("status_counts", {}),
    }
    if last_artifact is not None:
        payload["last_completed_artifact"] = {
            "case_name": last_artifact.get("case_name"),
            "backend": last_artifact.get("backend"),
            "status": last_artifact.get("status"),
            "output_path": last_artifact.get("output_path"),
            "telemetry_path": last_artifact.get("telemetry_path"),
        }
    write_json(checkpoint_path, payload)


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
            "error": format_nvidia_smi_error(completed.stderr or completed.stdout),
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

def drain_stream(
    stream: Any,
    sink: list[str],
    mirror: Any,
) -> None:
    try:
        while True:
            chunk = stream.readline()
            if chunk == "":
                break
            sink.append(chunk)
            mirror.write(chunk)
            mirror.flush()
    finally:
        stream.close()


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
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("failed to capture benchmark subprocess output")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread = threading.Thread(
        target=drain_stream,
        args=(process.stdout, stdout_chunks, sys.stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain_stream,
        args=(process.stderr, stderr_chunks, sys.stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    samples: list[dict[str, float]] = []
    while process.poll() is None:
        sample = sample_gpu(gpu_index)
        if sample is not None:
            samples.append(sample)
        time.sleep(sample_interval_s)

    sample = sample_gpu(gpu_index)
    if sample is not None:
        samples.append(sample)

    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    duration_ms = (time.perf_counter() - started_at) * 1000.0
    return returncode, "".join(stdout_chunks), "".join(stderr_chunks), samples, duration_ms


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
    use_interaction_picture: bool = False,
) -> list[str]:
    binary_name = case.get("internal_binary", "hybridcvdv_single_gpu_experiments")
    binary_path = build_dir / binary_name
    cmd = [
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
    if use_interaction_picture:
        cmd.append("--use-interaction-picture")
    return cmd


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
    use_interaction_picture: bool = False,
) -> dict[str, Any]:
    output_name = f"{case['name']}__{backend}.json"
    output_path = run_dir / output_name
    telemetry_samples_path = run_dir / "telemetry" / f"{case['name']}__{backend}.json"

    if backend == "hybridcvdv":
        cmd = build_internal_command(build_dir, case, output_path, gaussian_symbolic_mode_limit, use_interaction_picture)
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
    if not output_path.exists():
        output_status = "error"
        stderr_text = append_log_message(stderr_text, f"missing benchmark output artifact: {output_path}")
    else:
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
            else:
                output_status = "error"
        except Exception:
            output_status = "error"
            stderr_text = append_log_message(stderr_text, f"failed to parse benchmark output JSON: {output_path}")
    if telemetry_summary.get("gpu_sample_count", 0.0) <= 0.0:
        case_status = "error"
        stderr_text = append_log_message(stderr_text, "no GPU telemetry samples were collected for this benchmark")
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
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--baseline-python", default=None)
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--artifact-start", type=int, default=None)
    parser.add_argument("--artifact-end", type=int, default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--telemetry-interval-ms", type=int, default=None)
    parser.add_argument("--use-interaction-picture", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config = read_json(config_path)
        cases = validate_config(config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: Invalid benchmark config {config_path}: {exc}", file=sys.stderr)
        return 1

    build_dir = pathlib.Path(args.build_dir)
    results_root = pathlib.Path(args.results_dir)
    checkpoint_path = pathlib.Path(args.checkpoint_path).expanduser() if args.checkpoint_path else None

    gpu_index = args.gpu_index
    telemetry_interval_ms = args.telemetry_interval_ms or config.get("telemetry_interval_ms", 200)
    if telemetry_interval_ms <= 0:
        print("ERROR: telemetry interval must be positive", file=sys.stderr)
        return 1
    sample_interval_s = telemetry_interval_ms / 1000.0
    baseline_python = args.baseline_python
    backend_python_map = config.get("backend_python_map")
    gaussian_symbolic_mode_limit = config.get("gaussian_symbolic_mode_limit", 16)
    use_interaction_picture = args.use_interaction_picture
    planned_artifacts = build_planned_artifacts(cases, args.case_filter, args.backend)
    if not planned_artifacts:
        print("ERROR: No benchmark cases matched the requested filters", file=sys.stderr)
        return 1
    try:
        selected_artifacts = select_artifact_slice(planned_artifacts, args.artifact_start, args.artifact_end)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not selected_artifacts:
        print("ERROR: No benchmark artifacts selected for execution", file=sys.stderr)
        return 1

    checkpoint = None
    run_id = args.run_id
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint = read_json(checkpoint_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: Failed to read checkpoint {checkpoint_path}: {exc}", file=sys.stderr)
            return 1
        checkpoint_run_id = checkpoint.get("run_id")
        if isinstance(checkpoint_run_id, str) and checkpoint_run_id:
            if run_id and run_id != checkpoint_run_id:
                print(
                    f"ERROR: run-id mismatch between CLI ({run_id}) and checkpoint ({checkpoint_run_id})",
                    file=sys.stderr,
                )
                return 1
            run_id = checkpoint_run_id

    if not run_id:
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    device = query_gpu_metadata(gpu_index)
    if not device.get("available", False):
        print(
            f"ERROR: GPU telemetry is unavailable for GPU {gpu_index}: {device.get('error', 'nvidia-smi failed')}",
            file=sys.stderr,
        )
        return 1

    manifest = load_or_initialize_manifest(
        manifest_path=manifest_path,
        run_id=run_id,
        config_path=config_path,
        gpu_index=gpu_index,
        telemetry_interval_ms=telemetry_interval_ms,
        gaussian_symbolic_mode_limit=gaussian_symbolic_mode_limit,
        use_interaction_picture=use_interaction_picture,
        device=device,
        planned_artifact_count=len(planned_artifacts),
        case_filter=args.case_filter,
        selected_backend=args.backend,
    )

    planned_keys = {artifact_key(case["name"], backend) for case, backend in planned_artifacts}
    completed_keys = {
        key
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict)
        for key in [artifact_key_from_artifact(artifact)]
        if key is not None
    }
    unexpected_keys = sorted(completed_keys - planned_keys)
    if unexpected_keys:
        print(
            "ERROR: Existing manifest contains artifacts outside the current config/filter selection: "
            + ", ".join(unexpected_keys[:5]),
            file=sys.stderr,
        )
        return 1

    write_progress_checkpoint(
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        build_dir=build_dir,
        config_path=config_path,
        run_dir=run_dir,
        manifest=manifest,
        planned_artifact_count=len(planned_artifacts),
        status="running",
    )

    for case, backend in selected_artifacts:
        key = artifact_key(case["name"], backend)
        if key in completed_keys:
            print(f"Skipping completed benchmark {case['name']} [{backend}]")
            continue

        current_index = len(completed_keys) + 1
        print(f"[{current_index}/{len(planned_artifacts)}] Running {case['name']} [{backend}]")
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
            use_interaction_picture=use_interaction_picture,
        )
        merge_artifact(manifest, artifact)
        completed_keys.add(key)
        update_manifest_summary(manifest, len(planned_artifacts))
        write_json(manifest_path, manifest)
        write_progress_checkpoint(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            build_dir=build_dir,
            config_path=config_path,
            run_dir=run_dir,
            manifest=manifest,
            planned_artifact_count=len(planned_artifacts),
            status="running",
            last_artifact=artifact,
        )
        wall_time_ms = artifact.get("wall_time_ms")
        wall_time_label = f"{wall_time_ms:.2f} ms" if isinstance(wall_time_ms, (int, float)) else "n/a"
        print(
            f"[{len(completed_keys)}/{len(planned_artifacts)}] Finished {case['name']} [{backend}] "
            f"status={artifact['status']} wall_time={wall_time_label}"
        )

    update_manifest_summary(manifest, len(planned_artifacts))
    write_json(manifest_path, manifest)
    write_progress_checkpoint(
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        build_dir=build_dir,
        config_path=config_path,
        run_dir=run_dir,
        manifest=manifest,
        planned_artifact_count=len(planned_artifacts),
        status=str(manifest.get("status", "running")),
    )

    failed_artifacts = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("status") != "ok"
    ]
    print(f"Wrote GPU benchmark manifest to {manifest_path}")
    if failed_artifacts:
        print(f"Benchmark matrix completed with {len(failed_artifacts)} non-ok artifact(s):", file=sys.stderr)
        for artifact in failed_artifacts[:10]:
            print(
                f"  - {artifact.get('case_name')} [{artifact.get('backend')}]: {artifact.get('status')}",
                file=sys.stderr,
            )
        if len(failed_artifacts) > 10:
            print(f"  ... plus {len(failed_artifacts) - 10} additional non-ok artifact(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
