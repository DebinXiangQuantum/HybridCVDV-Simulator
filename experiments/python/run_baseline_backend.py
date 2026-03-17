#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import math
import pathlib
import statistics
import time
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkloadSpec:
    workload: str
    cutoff: int
    num_modes: int
    num_qubits: int
    layers: int
    timesteps: int
    squeezing_r: float
    displacement_scale: float
    j_coupling: float
    omega_r: float
    omega_q: float
    g_coupling: float
    tau: float
    warmup_runs: int
    measured_runs: int


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def optional_import(name: str) -> Any | None:
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = quantile * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def make_qaoa_angles(layers: int) -> list[float]:
    return [2.0 * math.pi * float(i + 1) / float(2 * layers) for i in range(2 * layers)]


def flatten_norm(array: Any) -> float:
    if np is None:
        raise RuntimeError("numpy is required to materialize backend states")
    host = np.asarray(array)
    return float(np.linalg.norm(host.reshape(-1)))


def estimate_state_vector_bytes(cutoff: int, num_modes: int, num_qubits: int) -> int:
    return (2**num_qubits) * (cutoff**num_modes) * 16


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    total = [sample["total_ms"] for sample in samples]
    compute = [sample["compute_ms"] for sample in samples]
    communication = [sample["communication_ms"] for sample in samples]
    norms = [sample["output_norm"] for sample in samples]
    return {
        "median_total_ms": percentile(total, 0.5),
        "median_compute_ms": percentile(compute, 0.5),
        "median_communication_ms": percentile(communication, 0.5),
        "min_total_ms": min(total),
        "max_total_ms": max(total),
        "p25_total_ms": percentile(total, 0.25),
        "p75_total_ms": percentile(total, 0.75),
        "throughput_ops_per_sec": 1000.0 / percentile(total, 0.5) if percentile(total, 0.5) > 0 else 0.0,
        "median_output_norm": percentile(norms, 0.5),
    }


def benchmark_callable(
    case_name: str,
    category: str,
    backend_note: str,
    spec: WorkloadSpec,
    run_once,
) -> dict[str, Any]:
    for _ in range(spec.warmup_runs):
        run_once()

    samples = [run_once() for _ in range(spec.measured_runs)]
    metrics = summarize_samples(samples)
    metrics["state_vector_bytes_estimate"] = float(estimate_state_vector_bytes(spec.cutoff, spec.num_modes, spec.num_qubits))
    return {
        "name": case_name,
        "category": category,
        "status": "ok",
        "note": backend_note,
        "params": {
            "workload": spec.workload,
            "cutoff": str(spec.cutoff),
            "num_modes": str(spec.num_modes),
            "num_qubits": str(spec.num_qubits),
            "layers": str(spec.layers),
            "timesteps": str(spec.timesteps),
            "warmup_runs": str(spec.warmup_runs),
            "measured_runs": str(spec.measured_runs),
            "squeezing_r": str(spec.squeezing_r),
            "displacement_scale": str(spec.displacement_scale),
            "j_coupling": str(spec.j_coupling),
            "omega_r": str(spec.omega_r),
            "omega_q": str(spec.omega_q),
            "g_coupling": str(spec.g_coupling),
            "tau": str(spec.tau),
        },
        "metrics": metrics,
    }


def build_case_name(spec: WorkloadSpec) -> str:
    return f"{spec.workload}_nq{spec.num_qubits}_nm{spec.num_modes}_c{spec.cutoff}"


def build_metric_note(backend_name: str, workload: str) -> str:
    notes = {
        "vqe_circuit": "Full VQE circuit evaluation",
        "jch_simulation_circuit": "Full JCH simulation circuit evaluation",
        "cv_qaoa": "CV-QAOA bosonic-only circuit",
        "cat_state_circuit": "Cat state preparation via conditional displacement",
        "gkp_state_circuit": "GKP state preparation via iterative conditional displacement",
        "qaoa_circuit": "CV-QAOA circuit implementation",
        "qft_circuit": "Quantum Fourier Transform via CV-DV hybrid state transfer",
        "shors_circuit": "Shor's algorithm implementation using GKP and modular exponentiation",
        "state_transfer_CVtoDV_circuit": "Continuous-to-Discrete state transfer protocol",
        "state_transfer_DVtoCV_circuit": "Discrete-to-Continuous state transfer protocol",
    }
    return f"{backend_name}; {notes.get(workload, workload)}"


def run_strawberryfields_backend(spec: WorkloadSpec) -> dict[str, Any]:
    sf = optional_import("strawberryfields")
    tf = optional_import("tensorflow")
    if sf is None or tf is None:
        return unsupported_backend_payload("strawberryfields_tf", "missing strawberryfields or tensorflow")

    if spec.num_qubits > 0:
        return unsupported_backend_payload(
            "strawberryfields_tf",
            f"Strawberry Fields does not natively support hybrid DV+CV circuits (num_qubits={spec.num_qubits}). "
            "It lacks a native hybrid state space and discrete-continuous coupling gates (e.g. Jaynes-Cummings)."
        )

    from strawberryfields import ops

    def run_once() -> dict[str, float]:
        prog = sf.Program(spec.num_modes)
        with prog.context as registers:
            if spec.workload == "cv_qaoa" or spec.workload == "qaoa_circuit":
                params = make_qaoa_angles(spec.layers)
                for mode in range(spec.num_modes):
                    ops.Sgate(spec.squeezing_r, 0.0) | registers[mode]
                for layer in range(spec.layers):
                    gamma = params[layer]
                    eta = params[spec.layers + layer]
                    for mode in range(spec.num_modes):
                        ops.Dgate(spec.displacement_scale * gamma, 0.0) | registers[mode]
                    for mode in range(spec.num_modes):
                        ops.Sgate(eta, 0.0) | registers[mode]
            elif spec.workload == "jch_photonic_chain" or spec.workload == "jch_simulation_circuit":
                for _ in range(spec.timesteps):
                    for mode in range(spec.num_modes):
                        ops.Rgate(spec.omega_r * spec.tau) | registers[mode]
                    for mode in range(spec.num_modes - 1):
                        ops.BSgate(spec.j_coupling * spec.tau, 0.0) | (registers[mode], registers[mode + 1])
            else:
                # Other workloads are likely hybrid and caught by num_qubits > 0 check,
                # but if one slipped through, it's unsupported here.
                raise ValueError(f"unsupported Strawberry Fields workload: {spec.workload}")

        gpu_devices = configure_tensorflow_gpu(tf)
        engine = sf.Engine("tf", backend_options={"cutoff_dim": spec.cutoff})
        compute_start = time.perf_counter()
        result = engine.run(prog)
        compute_end = time.perf_counter()
        ket = result.state.ket()
        host = np.asarray(ket.numpy() if hasattr(ket, "numpy") else ket)
        total_end = time.perf_counter()
        return {
            "total_ms": (total_end - compute_start) * 1000.0,
            "compute_ms": (compute_end - compute_start) * 1000.0,
            "communication_ms": (total_end - compute_end) * 1000.0,
            "output_norm": flatten_norm(host),
        }

    return {
        "schema_version": "2.0",
        "generated_at_utc": iso_now(),
        "backend": "strawberryfields_tf",
        "status": "ok",
        "results": [benchmark_callable(build_case_name(spec), "baseline", build_metric_note("SF", spec.workload), spec, run_once)],
    }


def run_mrmustard_backend(spec: WorkloadSpec) -> dict[str, Any]:
    mm = optional_import("mrmustard")
    jax = optional_import("jax")
    if mm is None or jax is None:
        return unsupported_backend_payload("mrmustard_jax", "missing mrmustard or jax")

    if spec.num_qubits > 0:
        return unsupported_backend_payload(
            "mrmustard_jax",
            f"MrMustard does not natively support hybrid DV+CV circuits (num_qubits={spec.num_qubits}). "
            "It lacks a native hybrid state space and discrete-continuous coupling gates (e.g. Jaynes-Cummings)."
        )

    from mrmustard.lab import Vacuum, Dgate, Sgate, BSgate, Rgate
    import mrmustard.math as math_mod
    math_mod.change_backend("jax")

    def run_once() -> dict[str, float]:
        state = Vacuum(spec.num_modes)
        if spec.workload == "cv_qaoa" or spec.workload == "qaoa_circuit":
            params = make_qaoa_angles(spec.layers)
            for mode in range(spec.num_modes):
                state = state >> Sgate(modes=[mode], r=spec.squeezing_r)
            for layer in range(spec.layers):
                for mode in range(spec.num_modes):
                    state = state >> Dgate(modes=[mode], x=spec.displacement_scale * params[layer])
                for mode in range(spec.num_modes):
                    state = state >> Sgate(modes=[mode], r=params[spec.layers + layer])
        elif spec.workload == "jch_photonic_chain" or spec.workload == "jch_simulation_circuit":
            for _ in range(spec.timesteps):
                for mode in range(spec.num_modes):
                    state = state >> Rgate(modes=[mode], angle=spec.omega_r * spec.tau)
                for mode in range(spec.num_modes - 1):
                    state = state >> BSgate(modes=[mode, mode + 1], theta=spec.j_coupling * spec.tau)
        
        compute_start = time.perf_counter()
        ket = state.ket(cutoffs=[spec.cutoff] * spec.num_modes)
        ket = jax.block_until_ready(ket)
        compute_end = time.perf_counter()
        host = np.asarray(jax.device_get(ket))
        total_end = time.perf_counter()
        return {
            "total_ms": (total_end - compute_start) * 1000.0,
            "compute_ms": (compute_end - compute_start) * 1000.0,
            "communication_ms": (total_end - compute_end) * 1000.0,
            "output_norm": flatten_norm(host),
        }

    return {
        "schema_version": "2.0",
        "generated_at_utc": iso_now(),
        "backend": "mrmustard_jax",
        "status": "ok",
        "results": [benchmark_callable(build_case_name(spec), "baseline", build_metric_note("MM", spec.workload), spec, run_once)],
    }


def configure_tensorflow_gpu(tf):
    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    return [d.name for d in gpu_devices]


def unsupported_backend_payload(backend: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "generated_at_utc": iso_now(),
        "backend": backend,
        "single_gpu_focus": True,
        "status": "unsupported",
        "reason": reason,
        "results": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--workload", default="vqe_circuit")
    parser.add_argument("--cutoff", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--num-qubits", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--squeezing-r", type=float, default=0.5)
    parser.add_argument("--displacement-scale", type=float, default=1.0)
    parser.add_argument("--j-coupling", type=float, default=1.0)
    parser.add_argument("--omega-r", type=float, default=1.0)
    parser.add_argument("--omega-q", type=float, default=1.0)
    parser.add_argument("--g-coupling", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = WorkloadSpec(
        workload=args.workload,
        cutoff=args.cutoff,
        num_modes=args.num_modes,
        num_qubits=args.num_qubits,
        layers=args.layers,
        timesteps=args.timesteps,
        squeezing_r=args.squeezing_r,
        displacement_scale=args.displacement_scale,
        j_coupling=args.j_coupling,
        omega_r=args.omega_r,
        omega_q=args.omega_q,
        g_coupling=args.g_coupling,
        tau=args.tau,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )

    runners = {
        "strawberryfields_tf": run_strawberryfields_backend,
        "mrmustard_jax": run_mrmustard_backend,
    }

    backend = {
        "strawberryfields": "strawberryfields_tf",
        "mrmustard": "mrmustard_jax",
    }.get(args.backend, args.backend)

    try:
        payload = runners[backend](spec)
    except Exception as exc:
        payload = unsupported_backend_payload(backend, str(exc))

    write_json(pathlib.Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
