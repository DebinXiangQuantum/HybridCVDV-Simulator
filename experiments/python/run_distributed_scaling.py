#!/usr/bin/env python3
"""Unified HybridCVDV/ATLAS/BQSim distributed experiment orchestrator."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import pathlib
import random
import signal
import subprocess
import time
from typing import Any

from collect_distributed_environment import collect as collect_environment
from distributed_common import (
    REPO_ROOT,
    atomic_write_json,
    classify_failure,
    empty_result,
    iso_now,
    read_json,
    validate_result,
)
from distributed_telemetry import TelemetryCollector


DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/sc26_distributed_scaling.json"
DEFAULT_QASM = REPO_ROOT / "experiments/generated/sc26_baseline_qasm"
DEFAULT_RESULT_ROOT = REPO_ROOT / "experiments/results/distributed_8xH800"
ALLOWED_PHASES = {"smoke", "strong", "capacity", "throughput", "full"}
GPU_SCALING_MAX_HOST_FRACTION = 0.50
GPU_SCALING_MIN_AVG_UTILIZATION = 10.0
GPU_SCALING_MIN_SIMULATION_MS = 10.0
# Known HybridCVDV cases whose wall time is dominated by host qubit gates.
# They are recorded as host_bound_skipped for coverage scans and never enter
# GPU strong-scaling or Phase E formal reruns.
HOST_BOUND_SKIP_TOKENS = (
    "transfer_DVtoCV_nq16",
)


def csv_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def busy_gpu_processes(gpu_ids: list[int]) -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"unable to check GPU processes: {completed.stderr.strip()}")
    # nvidia-smi cannot directly return the physical index in this query. Any compute
    # process is treated as a conflict because formal runs reserve the complete node.
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def process_result(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: pathlib.Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": stdout[-16000:],
        "stderr": stderr[-16000:],
        "timed_out": timed_out,
        "wall_time_ms": (time.perf_counter() - started) * 1000.0,
    }


def qasm_metadata(qasm_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(qasm_dir / "manifest.json")
    return {artifact["case_name"]: artifact for artifact in manifest["artifacts"]}


def select_cases(cases: list[dict[str, Any]], phase: str, case_filter: str | None) -> list[dict[str, Any]]:
    selected = [case for case in cases if not case_filter or case_filter in case["name"]]
    if phase == "full" or case_filter:
        return selected
    workloads = ("jch", "vqe", "qaoa", "qft", "transfer_CVtoDV", "transfer_DVtoCV")
    if phase == "smoke":
        smoke_names = (
            "sc26_jch_nq3_nm2_c4", "sc26_jch_nq4_nm2_c8",
            "sc26_vqe_nq3_nm2_c4", "sc26_vqe_nq4_nm2_c8",
            "sc26_qaoa_nm1_c4", "sc26_qaoa_nm2_c8",
            "sc26_qft_nq3_c4", "sc26_qft_nq5_c8",
            "sc26_transfer_CVtoDV_nq2_c4", "sc26_transfer_CVtoDV_nq4_c8",
            "sc26_transfer_DVtoCV_nq2_c4", "sc26_transfer_DVtoCV_nq4_c8",
        )
        by_name = {case["name"]: case for case in selected}
        return [by_name[name] for name in smoke_names if name in by_name]
    if phase == "strong":
        preferred = (
            "sc26_jch_nq10_nm4",
            "sc26_jch_nq3_nm6",
            "sc26_vqe_nq8_nm5",
            "sc26_vqe_nq3_nm6",
            "sc26_qft_nq9",
            "sc26_transfer_DVtoCV_nq4",
        )
        return [case for case in selected if any(token in case["name"] for token in preferred)]
    if phase == "capacity":
        return sorted(selected, key=lambda case: (case.get("num_modes", 0), case.get("cutoff", 0)), reverse=True)[:24]
    if phase == "throughput":
        return selected[: max(1, min(8, len(selected)))]
    return selected


def filter_gpu_scaling_eligible_cases(
    cases: list[dict[str, Any]],
    eligibility_manifest: pathlib.Path,
    system: str,
) -> list[dict[str, Any]]:
    manifest = read_json(eligibility_manifest)
    eligible_names: set[str] = set()
    for artifact in manifest.get("artifacts", []):
        if (
            artifact.get("system") != system
            or artifact.get("phase") != "strong"
            or artifact.get("gpu_count") != 1
            or artifact.get("status") != "ok"
        ):
            continue
        artifact_path = pathlib.Path(artifact["path"])
        if not artifact_path.exists():
            continue
        payload = read_json(artifact_path)
        if payload.get("diagnostics", {}).get("gpu_scaling_eligible") is True:
            eligible_names.add(str(artifact["case_name"]))
    return [case for case in cases if case["name"] in eligible_names]


def successful_work_from_manifest(
    manifest_path: pathlib.Path,
) -> set[tuple[str, str, str, int]]:
    manifest = read_json(manifest_path)
    return {
        (
            str(artifact["system"]),
            str(artifact["phase"]),
            str(artifact["case_name"]),
            int(artifact["gpu_count"]),
        )
        for artifact in manifest.get("artifacts", [])
        if artifact.get("status") == "ok"
    }


def is_known_host_bound_case(system: str, case_name: str) -> bool:
    if system != "hybridcvdv":
        return False
    return any(token in case_name for token in HOST_BOUND_SKIP_TOKENS)


def write_host_bound_skip_result(
    *,
    system: str,
    case: dict[str, Any],
    phase: str,
    gpu_ids: list[int],
    repetition: int,
    args: argparse.Namespace,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    result = empty_result(
        system=system,
        case=case,
        phase=phase,
        gpu_ids=gpu_ids,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        repetition=repetition,
    )
    result["status"] = "host_bound_skipped"
    result["diagnostics"] = {
        "scaling_role": "host_bound_control",
        "gpu_scaling_eligible": False,
        "skip_reason": "known_host_bound_workload",
        "skip_tokens": list(HOST_BOUND_SKIP_TOKENS),
    }
    atomic_write_json(output_path, result)
    return result


def select_strong_cases_from_pilot(
    cases: list[dict[str, Any]],
    eligibility_manifest: pathlib.Path,
    system: str,
) -> list[dict[str, Any]]:
    eligible = filter_gpu_scaling_eligible_cases(
        cases, eligibility_manifest, system
    )
    selected_names = {case["name"] for case in eligible}
    if system != "hybridcvdv":
        return cases

    # Keep one short transfer case per cutoff as an explicitly labelled
    # Host-bound control. It is reported separately and excluded from GPU
    # speedup plots, but preserves the plan's transfer-workload coverage.
    by_name = {case["name"]: case for case in cases}
    controls: dict[int, tuple[float, str]] = {}
    manifest = read_json(eligibility_manifest)
    for artifact in manifest.get("artifacts", []):
        name = str(artifact.get("case_name", ""))
        if (
            artifact.get("system") != system
            or artifact.get("phase") != "strong"
            or artifact.get("gpu_count") != 1
            or artifact.get("status") != "ok"
            or "transfer_" not in name
            or name not in by_name
        ):
            continue
        payload = read_json(pathlib.Path(artifact["path"]))
        if payload.get("diagnostics", {}).get("scaling_role") != "host_bound_control":
            continue
        simulation_ms = payload.get("timing", {}).get("simulation_ms")
        if not isinstance(simulation_ms, (int, float)):
            continue
        cutoff = int(by_name[name].get("cutoff", 0))
        current = controls.get(cutoff)
        if current is None or float(simulation_ms) < current[0]:
            controls[cutoff] = (float(simulation_ms), name)
    selected_names.update(name for _, name in controls.values())
    return [case for case in cases if case["name"] in selected_names]


def build_command(
    system: str,
    case: dict[str, Any],
    *,
    gpu_count: int,
    output_path: pathlib.Path,
    qasm_dir: pathlib.Path,
    qasm_meta: dict[str, dict[str, Any]],
    warmup_runs: int,
    measured_runs: int,
    batch_tasks: int,
    build_dir: pathlib.Path,
    atlas_build_dir: pathlib.Path,
    bqsim_build_dir: pathlib.Path,
) -> tuple[list[str], pathlib.Path]:
    if system == "hybridcvdv":
        return (
            [
                str(build_dir / "hybridcvdv_single_gpu_experiments"),
                "--suite",
                "scaling",
                "--name-filter",
                case.get("internal_name_filter", case["name"]),
                "--output",
                str(output_path),
            ],
            REPO_ROOT,
        )
    if system == "atlas":
        metadata = qasm_meta[case["name"]]
        qubits = int(metadata["encoded_qubits"])
        if gpu_count & (gpu_count - 1):
            return (["__unsupported_gpu_count__"], REPO_ROOT)
        global_qubits = int(math.log2(gpu_count))
        if qubits < 2 * global_qubits:
            return (["__unsupported_backend__"], REPO_ROOT)
        local = max(1, qubits - global_qubits)
        return (
            [
                "mpirun",
                "-np",
                "1",
                "--bind-to",
                "none",
                str(atlas_build_dir / "examples/mpi-based/run_generated_qasm"),
                "--qasm-path",
                str(qasm_dir / f"{case['name']}.qasm"),
                "--n",
                str(qubits),
                "--local",
                str(local),
                "--device",
                str(gpu_count),
                "--use-ilp",
                "--output",
                str(output_path),
            ],
            atlas_build_dir / "examples/mpi-based",
        )
    if system == "bqsim":
        return (
            [
                str(bqsim_build_dir / "apps/BQSim"),
                "--ps",
                "--batch_size",
                "256",
                "--num_batch",
                str(max(1, batch_tasks)),
                "--conversion_type",
                "2",
                "--file",
                str(qasm_dir / f"{case['name']}.qasm"),
                "--output",
                str(output_path),
            ],
            REPO_ROOT / "baselines/BQSim-main",
        )
    raise ValueError(f"unknown system: {system}")


def normalize_native_output(
    result: dict[str, Any],
    raw_output: pathlib.Path,
    process: dict[str, Any],
    telemetry: dict[str, Any],
    qasm: dict[str, Any] | None,
) -> None:
    result["runner"] = process
    result["telemetry"] = telemetry
    if process["timed_out"] or process["returncode"] != 0 or not raw_output.exists():
        result["status"] = classify_failure(
            process["returncode"], process["stdout"], process["stderr"], process["timed_out"]
        )
        return
    try:
        payload = read_json(raw_output)
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = "configuration_error"
        result["runner"]["parse_error"] = str(exc)
        return
    metrics: dict[str, Any] = {}
    if isinstance(payload.get("results"), list) and payload["results"]:
        raw_result = payload["results"][0]
        metrics = raw_result.get("metrics", {})
        if raw_result.get("status", "ok") != "ok":
            result["status"] = classify_failure(1, "", raw_result.get("note", ""))
            return
    elif isinstance(payload.get("metrics"), dict):
        metrics = payload["metrics"]
    elif isinstance(payload.get("timing"), dict):
        for field in ("timing", "memory", "communication", "correctness", "throughput"):
            if field in payload:
                result[field] = payload[field]
        result["status"] = payload.get("status", "ok")
        return
    result["timing"] = {
        "total_wall_ms": process["wall_time_ms"],
        "simulation_ms": metrics.get("median_total_ms", metrics.get("simulation_ms")),
        "gpu_compute_ms": metrics.get("median_compute_ms", metrics.get("compute_ms")),
        "planning_ms": metrics.get("median_planning_ms", 0.0),
        "host_orchestration_ms": metrics.get("median_host_orchestration_ms"),
        "correctness_reduction_ms": metrics.get("median_correctness_reduction_ms", 0.0),
        "communication_ms": (
            metrics.get("p2p_time_ms", 0.0) + metrics.get("host_staged_time_ms", 0.0)
        ),
        "h2d_ms": metrics.get("h2d_ms", 0.0),
        "d2h_ms": metrics.get("d2h_ms", 0.0),
    }
    per_gpu: dict[str, Any] = {}
    for gpu in result["gpu_ids"]:
        suffix = f"_gpu_{gpu}"
        per_gpu[str(gpu)] = {
            "active_state_count": metrics.get(f"states{suffix}", 0),
            "state_pool_active_bytes": metrics.get(f"active_bytes{suffix}", 0),
            "state_pool_reserved_bytes": metrics.get(f"reserved_bytes{suffix}", 0),
            "scratch_bytes": metrics.get(f"scratch_bytes{suffix}", 0),
            "gpu_memory_peak_bytes": telemetry["per_gpu"].get(str(gpu), {}).get("summary", {}).get(
                "peak_memory_used", 0
            )
            * 1024
            * 1024,
        }
    result["memory"] = {"per_gpu": per_gpu, "aggregate": {}}
    result["communication"] = {
        "p2p_bytes": metrics.get("p2p_bytes", 0),
        "p2p_time_ms": metrics.get("p2p_time_ms", 0),
        "host_staged_bytes": metrics.get("host_staged_bytes", 0),
        "host_staged_time_ms": metrics.get("host_staged_time_ms", 0),
        "transfer_count": metrics.get("p2p_transfer_count", 0)
        + metrics.get("host_staged_transfer_count", 0),
        "state_migration_count": metrics.get("state_migrations", 0),
    }
    result["correctness"] = {
        "output_norm": metrics.get("output_norm"),
        "checksum": metrics.get("output_checksum", payload.get("correctness", {}).get("checksum")),
    }
    if (
        result["correctness"]["output_norm"] == 0
        and metrics.get("median_gaussian_symbolic_blocks", 0) > 0
    ):
        # Symbolic Gaussian terminals intentionally have no materialized Fock
        # amplitudes. Use a decomposition-invariant structural fingerprint and
        # do not misreport the zero backing buffer as a physical norm.
        result["correctness"] = {
            "output_norm": None,
            "checksum": (
                101.0 * metrics.get("median_active_states", 0)
                + 17.0 * metrics.get("median_hdd_nodes", 0)
                + 7.0 * metrics.get("median_gaussian_symbolic_blocks", 0)
                + 3.0 * metrics.get("median_qubit_only_blocks", 0)
            ),
            "verification_mode": "symbolic_structure",
        }
    simulation_ms = result["timing"].get("simulation_ms")
    if isinstance(simulation_ms, (int, float)) and simulation_ms > 0:
        if result["timing"]["host_orchestration_ms"] is None:
            accounted_ms = (
                float(result["timing"].get("gpu_compute_ms") or 0.0)
                + float(metrics.get("median_transfer_ms", 0.0))
                + float(result["timing"].get("planning_ms") or 0.0)
            )
            result["timing"]["host_orchestration_ms"] = max(
                0.0, float(simulation_ms) - accounted_ms
            )
        host_fraction = float(result["timing"]["host_orchestration_ms"]) / float(
            simulation_ms
        )
    else:
        host_fraction = None
    utilization_values = [
        gpu_data.get("summary", {}).get("avg_utilization_gpu")
        for gpu_data in telemetry.get("per_gpu", {}).values()
    ]
    utilization_values = [
        float(value) for value in utilization_values if isinstance(value, (int, float))
    ]
    avg_gpu_utilization = (
        sum(utilization_values) / len(utilization_values)
        if utilization_values
        else None
    )
    host_bound = (
        host_fraction is not None
        and host_fraction > GPU_SCALING_MAX_HOST_FRACTION
    )
    result["diagnostics"] = {
        "host_fraction": host_fraction,
        "avg_gpu_utilization_pct": avg_gpu_utilization,
        "scaling_role": "host_bound_control" if host_bound else "gpu_scaling_candidate",
        "gpu_scaling_eligible": bool(
            not host_bound
            and isinstance(simulation_ms, (int, float))
            and simulation_ms >= GPU_SCALING_MIN_SIMULATION_MS
            and avg_gpu_utilization is not None
            and avg_gpu_utilization >= GPU_SCALING_MIN_AVG_UTILIZATION
        ),
        "eligibility_thresholds": {
            "max_host_fraction": GPU_SCALING_MAX_HOST_FRACTION,
            "min_avg_gpu_utilization_pct": GPU_SCALING_MIN_AVG_UTILIZATION,
            "min_simulation_ms": GPU_SCALING_MIN_SIMULATION_MS,
        },
    }
    result["throughput"] = {
        "completed_circuit_evaluations": result["measured_runs"],
        "circuit_evaluations_per_sec": 1000.0 / simulation_ms if simulation_ms else 0.0,
        "completed_gate_applications": (qasm or {}).get("gate_count", 0) * result["measured_runs"],
    }
    if telemetry["aggregate"].get("sample_count", 0) <= 0:
        result["status"] = "missing_telemetry"
    # Truncated CV workloads can legitimately lose norm at low cutoffs. Treat only
    # non-finite/degenerate values as locally incorrect; cross-GPU checksum
    # consistency is enforced during merge.
    elif result["correctness"].get("output_norm") is not None and not (
        0.0 < float(result["correctness"]["output_norm"]) <= 1.01
    ):
        result["status"] = "incorrect_result"
    else:
        result["status"] = "ok"


def run_atlas_repetitions(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: pathlib.Path,
    raw_output: pathlib.Path,
    warmup_runs: int,
    measured_runs: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    output_index = command.index("--output") + 1
    total_wall_ms = 0.0
    last_process: dict[str, Any] | None = None
    for index in range(warmup_runs + measured_runs):
        is_warmup = index < warmup_runs
        run_number = index if is_warmup else index - warmup_runs
        run_output = raw_output.with_name(
            f"{'warmup' if is_warmup else 'measured'}-{run_number}.json"
        )
        run_command = list(command)
        run_command[output_index] = str(run_output)
        process = process_result(
            run_command, env=env, cwd=cwd, timeout_seconds=timeout_seconds
        )
        last_process = process
        total_wall_ms += process["wall_time_ms"]
        attempts.append(
            {
                "returncode": process["returncode"],
                "timed_out": process["timed_out"],
                "wall_time_ms": process["wall_time_ms"],
                "output": str(run_output),
            }
        )
        if process["returncode"] != 0 or process["timed_out"]:
            process["attempts"] = attempts
            process["wall_time_ms"] = total_wall_ms
            return process
        if not is_warmup:
            payloads.append(read_json(run_output))

    if not payloads or last_process is None:
        raise RuntimeError("ATLAS produced no measured repetitions")
    aggregate = json.loads(json.dumps(payloads[-1]))
    for section in ("timing", "communication"):
        keys = set().union(*(payload.get(section, {}).keys() for payload in payloads))
        for key in keys:
            values = [
                payload.get(section, {}).get(key)
                for payload in payloads
                if isinstance(payload.get(section, {}).get(key), (int, float))
            ]
            if values:
                aggregate.setdefault(section, {})[key] = sorted(values)[len(values) // 2]
    memory_values = [
        payload.get("memory", {}).get("gpu_memory_peak_bytes")
        for payload in payloads
        if isinstance(payload.get("memory", {}).get("gpu_memory_peak_bytes"), (int, float))
    ]
    if memory_values:
        aggregate.setdefault("memory", {})["gpu_memory_peak_bytes"] = max(memory_values)
    checksums = [
        payload.get("correctness", {}).get("checksum")
        for payload in payloads
        if isinstance(payload.get("correctness", {}).get("checksum"), (int, float))
    ]
    if checksums and max(checksums) - min(checksums) > 1e-8 * max(1.0, abs(checksums[0])):
        aggregate["status"] = "incorrect_result"
    aggregate.setdefault("throughput", {})["completed_circuit_evaluations"] = measured_runs
    if "completed_gate_applications" in aggregate["throughput"]:
        aggregate["throughput"]["completed_gate_applications"] *= measured_runs
    atomic_write_json(raw_output, aggregate)
    last_process["attempts"] = attempts
    last_process["wall_time_ms"] = total_wall_ms
    last_process["command"] = command
    return last_process


def run_native(
    *,
    system: str,
    case: dict[str, Any],
    phase: str,
    gpu_ids: list[int],
    repetition: int,
    args: argparse.Namespace,
    qasm_meta: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
) -> dict[str, Any]:
    result = empty_result(
        system=system,
        case=case,
        phase=phase,
        gpu_ids=gpu_ids,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        repetition=repetition,
    )
    raw_output = output_path.with_name("raw.json")
    command, cwd = build_command(
        system,
        case,
        gpu_count=len(gpu_ids),
        output_path=raw_output,
        qasm_dir=args.qasm_dir,
        qasm_meta=qasm_meta,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        batch_tasks=args.total_tasks,
        build_dir=args.build_dir,
        atlas_build_dir=args.atlas_build_dir,
        bqsim_build_dir=args.bqsim_build_dir,
    )
    if command[0] == "__unsupported_gpu_count__":
        result["status"] = "unsupported_gpu_count"
        atomic_write_json(output_path, result)
        return result
    if command[0] == "__unsupported_backend__":
        result["status"] = "unsupported_backend"
        atomic_write_json(output_path, result)
        return result
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    env["HYBRIDCVDV_SCALING_WARMUP_RUNS"] = str(args.warmup_runs)
    env["HYBRIDCVDV_SCALING_MEASURED_RUNS"] = str(args.measured_runs)
    if system == "atlas" and os.geteuid() == 0:
        env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    collector = TelemetryCollector(gpu_ids, args.telemetry_interval_ms)
    collector.start()
    if system == "atlas":
        process = run_atlas_repetitions(
            command,
            env=env,
            cwd=cwd,
            raw_output=raw_output,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        process = process_result(
            command, env=env, cwd=cwd, timeout_seconds=args.timeout_seconds
        )
    telemetry = collector.stop()
    normalize_native_output(result, raw_output, process, telemetry, qasm_meta.get(case["name"]))
    errors = validate_result(result)
    if errors:
        result["status"] = "configuration_error"
        result["schema_errors"] = errors
    atomic_write_json(output_path, result)
    return result


def run_throughput(
    *,
    system: str,
    case: dict[str, Any],
    phase: str,
    gpu_ids: list[int],
    repetition: int,
    args: argparse.Namespace,
    qasm_meta: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
) -> dict[str, Any]:
    result = empty_result(
        system=system,
        case=case,
        phase=phase,
        gpu_ids=gpu_ids,
        warmup_runs=0,
        measured_runs=args.total_tasks,
        repetition=repetition,
    )
    per_worker = math.ceil(args.total_tasks / len(gpu_ids))
    collector = TelemetryCollector(gpu_ids, args.telemetry_interval_ms)
    collector.start()
    started = time.perf_counter()

    def worker(gpu: int) -> dict[str, Any]:
        raw_output = output_path.parent / f"worker-{gpu}.json"
        command, cwd = build_command(
            system,
            case,
            gpu_count=1,
            output_path=raw_output,
            qasm_dir=args.qasm_dir,
            qasm_meta=qasm_meta,
            warmup_runs=0,
            measured_runs=per_worker,
            batch_tasks=per_worker,
            build_dir=args.build_dir,
            atlas_build_dir=args.atlas_build_dir,
            bqsim_build_dir=args.bqsim_build_dir,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["HYBRIDCVDV_SCALING_WARMUP_RUNS"] = "0"
        env["HYBRIDCVDV_SCALING_MEASURED_RUNS"] = str(per_worker)
        if system == "atlas" and os.geteuid() == 0:
            env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
            env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
        process = process_result(command, env=env, cwd=cwd, timeout_seconds=args.timeout_seconds)
        process["raw_output"] = str(raw_output)
        return process

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        workers = list(executor.map(worker, gpu_ids))
    makespan = time.perf_counter() - started
    result["telemetry"] = collector.stop()
    result["workers"] = workers
    failed = [worker for worker in workers if worker["returncode"] != 0 or worker["timed_out"]]
    completed = min(args.total_tasks, per_worker * (len(workers) - len(failed)))
    result["timing"] = {"total_wall_ms": makespan * 1000.0, "simulation_ms": makespan * 1000.0}
    result["throughput"] = {
        "completed_circuit_evaluations": completed,
        "circuit_evaluations_per_sec": completed / makespan if makespan > 0 else 0.0,
        "completed_input_states": completed * (256 if system == "bqsim" else 1),
    }
    if failed:
        first = failed[0]
        result["status"] = classify_failure(
            first["returncode"], first["stdout"], first["stderr"], first["timed_out"]
        )
    elif result["telemetry"]["aggregate"].get("sample_count", 0) <= 0:
        result["status"] = "missing_telemetry"
    else:
        result["status"] = "ok"
    atomic_write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="hybridcvdv,atlas,bqsim")
    parser.add_argument("--gpu-counts", default="1,2,4,6,8")
    parser.add_argument("--phase", default="smoke,strong,capacity,throughput,full")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qasm-dir", type=pathlib.Path, default=DEFAULT_QASM)
    parser.add_argument("--result-root", type=pathlib.Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--build-dir", type=pathlib.Path, default=REPO_ROOT / "build-H800")
    parser.add_argument("--atlas-build-dir", type=pathlib.Path, default=REPO_ROOT / "baselines/atlas-main/build-H800")
    parser.add_argument("--bqsim-build-dir", type=pathlib.Path, default=REPO_ROOT / "baselines/BQSim-main/build-H800")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--telemetry-interval-ms", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--total-tasks", type=int, default=256)
    parser.add_argument("--case-filter")
    parser.add_argument(
        "--eligibility-manifest",
        type=pathlib.Path,
        help="1-GPU strong pilot manifest used to exclude Host-bound/OOM/timeout cases",
    )
    parser.add_argument(
        "--successful-work-manifest",
        type=pathlib.Path,
        help="Run only system/phase/case/GPU combinations marked ok in this manifest",
    )
    parser.add_argument("--seed", type=int, default=2600)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    systems = csv_values(args.systems)
    gpu_counts = [int(value) for value in csv_values(args.gpu_counts)]
    phases = csv_values(args.phase)
    if any(phase not in ALLOWED_PHASES for phase in phases):
        raise ValueError(f"phase must be one of {sorted(ALLOWED_PHASES)}")
    if any(count <= 0 or count > 8 for count in gpu_counts):
        raise ValueError("gpu counts must be between 1 and 8")
    config = read_json(args.config)
    cases = config["cases"]
    qasm_meta = qasm_metadata(args.qasm_dir)
    successful_work = (
        successful_work_from_manifest(args.successful_work_manifest)
        if args.successful_work_manifest
        else None
    )
    busy = busy_gpu_processes(list(range(max(gpu_counts))))
    if busy and not args.allow_busy_gpus:
        raise RuntimeError("GPU node has active compute processes: " + "; ".join(busy[:8]))
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_root = (args.result_root / run_id).resolve()
    if not args.resume or not (run_root / "metadata/environment.json").exists():
        collect_environment(run_root / "metadata")
    (run_root / "configs").mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "configs/config.json", config)
    manifest_path = run_root / "manifest.json"
    if args.resume and manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest["status"] = "running"
    else:
        manifest = {
            "schema_version": "3.0",
            "run_id": run_id,
            "generated_at_utc": iso_now(),
            "status": "running",
            "artifacts": [],
        }
    atomic_write_json(manifest_path, manifest)
    randomizer = random.Random(args.seed)

    for phase in phases:
        phase_cases = select_cases(cases, phase, args.case_filter)
        if phase == "strong" and args.eligibility_manifest:
            eligible_by_system = {
                system: (
                    select_strong_cases_from_pilot(
                        phase_cases, args.eligibility_manifest, system
                    )
                    if system == "hybridcvdv"
                    else phase_cases
                )
                for system in systems
            }
            if "hybridcvdv" in systems and not eligible_by_system["hybridcvdv"]:
                raise RuntimeError(
                    "No HybridCVDV cases passed the 1-GPU GPU-scaling eligibility pilot"
                )
        else:
            eligible_by_system = {system: phase_cases for system in systems}
        work = [
            (system, case, gpu_count, repetition)
            for repetition in range(args.repetitions)
            for gpu_count in gpu_counts
            for system in systems
            for case in eligible_by_system[system]
            if not (system == "bqsim" and phase == "strong")
            and (
                successful_work is None
                or (system, phase, case["name"], gpu_count) in successful_work
            )
        ]
        randomizer.shuffle(work)
        for system, case, gpu_count, repetition in work:
            gpu_ids = list(range(gpu_count))
            output_path = (
                run_root
                / system
                / phase
                / f"g{gpu_count}"
                / case["name"]
                / f"r{repetition}"
                / "result.json"
            )
            if args.resume and output_path.exists():
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if is_known_host_bound_case(system, case["name"]):
                result = write_host_bound_skip_result(
                    system=system,
                    case=case,
                    phase=phase,
                    gpu_ids=gpu_ids,
                    repetition=repetition,
                    args=args,
                    output_path=output_path,
                )
            else:
                use_throughput_mode = phase == "throughput" or (
                    system == "bqsim" and phase in {"smoke", "full"}
                )
                if use_throughput_mode:
                    result = run_throughput(
                        system=system,
                        case=case,
                        phase=phase,
                        gpu_ids=gpu_ids,
                        repetition=repetition,
                        args=args,
                        qasm_meta=qasm_meta,
                        output_path=output_path,
                    )
                else:
                    result = run_native(
                        system=system,
                        case=case,
                        phase=phase,
                        gpu_ids=gpu_ids,
                        repetition=repetition,
                        args=args,
                        qasm_meta=qasm_meta,
                        output_path=output_path,
                    )
            manifest["artifacts"].append(
                {
                    "system": system,
                    "phase": phase,
                    "case_name": case["name"],
                    "gpu_count": gpu_count,
                    "repetition": repetition,
                    "status": result["status"],
                    "path": str(output_path),
                }
            )
            atomic_write_json(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = iso_now()
    atomic_write_json(manifest_path, manifest)
    print(f"Wrote distributed run to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
