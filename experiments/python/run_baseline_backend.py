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
    layers: int
    timesteps: int
    squeezing_r: float
    displacement_scale: float
    j_coupling: float
    omega_r: float
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


def estimate_state_vector_bytes(cutoff: int, num_modes: int) -> int:
    return (cutoff**num_modes) * 16


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
    metrics["state_vector_bytes_estimate"] = float(estimate_state_vector_bytes(spec.cutoff, spec.num_modes))
    return {
        "name": case_name,
        "category": category,
        "status": "ok",
        "note": backend_note,
        "params": {
            "workload": spec.workload,
            "cutoff": str(spec.cutoff),
            "num_modes": str(spec.num_modes),
            "layers": str(spec.layers),
            "timesteps": str(spec.timesteps),
            "warmup_runs": str(spec.warmup_runs),
            "measured_runs": str(spec.measured_runs),
            "squeezing_r": str(spec.squeezing_r),
            "displacement_scale": str(spec.displacement_scale),
            "j_coupling": str(spec.j_coupling),
            "omega_r": str(spec.omega_r),
            "tau": str(spec.tau),
        },
        "metrics": metrics,
    }


def build_case_name(spec: WorkloadSpec) -> str:
    if spec.workload == "cv_qaoa":
        return f"cv_qaoa_modes_{spec.num_modes}_layers_{spec.layers}_cutoff_{spec.cutoff}"
    if spec.workload == "jch_photonic_chain":
        return (
            f"jch_photonic_chain_modes_{spec.num_modes}"
            f"_timesteps_{spec.timesteps}_cutoff_{spec.cutoff}"
        )
    raise ValueError(f"unsupported workload: {spec.workload}")


def build_metric_note(backend_name: str, workload: str) -> str:
    notes = {
        "cv_qaoa": "communication time is measured as final state materialization from device to host",
        "jch_photonic_chain": "derived from circuit/src/jch_simulation_circuit.cpp by keeping only bosonic R and BS terms",
    }
    return f"{backend_name}; {notes[workload]}"


def build_strawberryfields_program(sf, ops, spec: WorkloadSpec):
    prog = sf.Program(spec.num_modes)
    with prog.context as registers:
        if spec.workload == "cv_qaoa":
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
        elif spec.workload == "jch_photonic_chain":
            for _ in range(spec.timesteps):
                for mode in range(spec.num_modes):
                    ops.Rgate(spec.omega_r * spec.tau) | registers[mode]
                for mode in range(spec.num_modes - 1):
                    ops.BSgate(spec.j_coupling * spec.tau, 0.0) | (registers[mode], registers[mode + 1])
        else:
            raise ValueError(f"unsupported Strawberry Fields workload: {spec.workload}")
    return prog


def configure_tensorflow_gpu(tf) -> list[str]:
    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass
    return [device.name for device in gpu_devices]


def run_strawberryfields_backend(spec: WorkloadSpec) -> dict[str, Any]:
    sf = optional_import("strawberryfields")
    tf = optional_import("tensorflow")
    if sf is None or tf is None:
        missing = []
        if sf is None:
            missing.append("strawberryfields")
        if tf is None:
            missing.append("tensorflow")
        return unsupported_backend_payload(
            "strawberryfields_tf",
            f"missing importable package(s): {', '.join(missing)}",
        )

    gpu_devices = configure_tensorflow_gpu(tf)
    if not gpu_devices:
        return unsupported_backend_payload(
            "strawberryfields_tf",
            "TensorFlow did not report a visible GPU device",
        )

    from strawberryfields import ops  # type: ignore

    def run_once() -> dict[str, float]:
        program = build_strawberryfields_program(sf, ops, spec)
        engine = sf.Engine("tf", backend_options={"cutoff_dim": spec.cutoff})
        compute_start = time.perf_counter()
        result = engine.run(program)
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

    payload = benchmark_callable(
        case_name=build_case_name(spec),
        category="baseline_scaling",
        backend_note=build_metric_note("Strawberry Fields tf backend", spec.workload),
        spec=spec,
        run_once=run_once,
    )
    return {
        "schema_version": "2.0",
        "generated_at_utc": iso_now(),
        "backend": "strawberryfields_tf",
        "single_gpu_focus": True,
        "status": "ok",
        "backend_runtime": {
            "strawberryfields_version": getattr(sf, "__version__", "unknown"),
            "tensorflow_version": getattr(tf, "__version__", "unknown"),
            "visible_gpu_devices": gpu_devices,
        },
        "results": [payload],
    }


def import_mrmustard_symbols():
    math_module = importlib.import_module("mrmustard.math")
    try:
        from mrmustard.lab.states import Vacuum  # type: ignore
        from mrmustard.lab.transformations import BSgate, Dgate, Rgate, Sgate  # type: ignore
    except ImportError:
        from mrmustard.lab import BSgate, Dgate, Rgate, Sgate, Vacuum  # type: ignore
    return math_module, Vacuum, Dgate, Sgate, BSgate, Rgate


def call_with_supported_keywords(factory, **kwargs):
    try:
        signature = inspect.signature(factory)
        accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        if accepted:
            return factory(**accepted)
    except (TypeError, ValueError):
        pass
    return factory(**kwargs)


def make_mrmustard_vacuum(Vacuum, num_modes: int):
    for kwargs in (
        {"num_modes": num_modes},
        {"modes": tuple(range(num_modes))},
        {"modes": list(range(num_modes))},
    ):
        try:
            return call_with_supported_keywords(Vacuum, **kwargs)
        except TypeError:
            continue
    return Vacuum(num_modes)


def materialize_mrmustard_ket(state, cutoffs: list[int]):
    for kwargs in ({"cutoffs": cutoffs}, {"cutoff": cutoffs}, {"shape": cutoffs}):
        try:
            return state.ket(**kwargs)
        except TypeError:
            continue
    return state.ket(cutoffs)


def build_mrmustard_state(spec: WorkloadSpec, Vacuum, Dgate, Sgate, BSgate, Rgate):
    state = make_mrmustard_vacuum(Vacuum, spec.num_modes)
    if spec.workload == "cv_qaoa":
        params = make_qaoa_angles(spec.layers)
        for mode in range(spec.num_modes):
            state = state >> call_with_supported_keywords(Sgate, mode=mode, r=spec.squeezing_r, phi=0.0)
        for layer in range(spec.layers):
            gamma = params[layer]
            eta = params[spec.layers + layer]
            for mode in range(spec.num_modes):
                state = state >> call_with_supported_keywords(
                    Dgate,
                    mode=mode,
                    x=spec.displacement_scale * gamma,
                    y=0.0,
                )
            for mode in range(spec.num_modes):
                state = state >> call_with_supported_keywords(Sgate, mode=mode, r=eta, phi=0.0)
        return state

    if spec.workload == "jch_photonic_chain":
        for _ in range(spec.timesteps):
            for mode in range(spec.num_modes):
                state = state >> call_with_supported_keywords(
                    Rgate,
                    mode=mode,
                    angle=spec.omega_r * spec.tau,
                    phi=spec.omega_r * spec.tau,
                )
            for mode in range(spec.num_modes - 1):
                state = state >> call_with_supported_keywords(
                    BSgate,
                    modes=(mode, mode + 1),
                    theta=spec.j_coupling * spec.tau,
                    phi=0.0,
                )
        return state

    raise ValueError(f"unsupported MrMustard workload: {spec.workload}")


def run_mrmustard_backend(spec: WorkloadSpec) -> dict[str, Any]:
    mm = optional_import("mrmustard")
    jax = optional_import("jax")
    if mm is None or jax is None:
        missing = []
        if mm is None:
            missing.append("mrmustard")
        if jax is None:
            missing.append("jax")
        return unsupported_backend_payload(
            "mrmustard_jax",
            f"missing importable package(s): {', '.join(missing)}",
        )

    gpu_devices = [str(device) for device in jax.devices() if getattr(device, "platform", "") == "gpu"]
    if not gpu_devices:
        return unsupported_backend_payload(
            "mrmustard_jax",
            "JAX did not report a visible GPU device",
        )

    try:
        math_module, Vacuum, Dgate, Sgate, BSgate, Rgate = import_mrmustard_symbols()
    except Exception as exc:
        return unsupported_backend_payload("mrmustard_jax", f"failed to import API symbols: {exc}")

    if not hasattr(math_module, "change_backend"):
        return unsupported_backend_payload("mrmustard_jax", "mrmustard.math.change_backend is unavailable")

    math_module.change_backend("jax")

    def run_once() -> dict[str, float]:
        state = build_mrmustard_state(spec, Vacuum, Dgate, Sgate, BSgate, Rgate)
        compute_start = time.perf_counter()
        ket = materialize_mrmustard_ket(state, [spec.cutoff] * spec.num_modes)
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

    payload = benchmark_callable(
        case_name=build_case_name(spec),
        category="baseline_scaling",
        backend_note=build_metric_note("MrMustard jax backend", spec.workload),
        spec=spec,
        run_once=run_once,
    )
    return {
        "schema_version": "2.0",
        "generated_at_utc": iso_now(),
        "backend": "mrmustard_jax",
        "single_gpu_focus": True,
        "status": "ok",
        "backend_runtime": {
            "mrmustard_version": getattr(mm, "__version__", "unknown"),
            "jax_version": getattr(jax, "__version__", "unknown"),
            "visible_gpu_devices": gpu_devices,
            "default_backend": jax.default_backend(),
        },
        "results": [payload],
    }


def run_bosonic_gpu_backend(spec: WorkloadSpec) -> dict[str, Any]:
    if spec.workload != "cv_qaoa":
        return unsupported_backend_payload(
            "bosonic_gpu",
            "local bosonic baseline only supports legacy gate microbenchmarks",
        )

    try:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "baselines" / "bosonic"))
        import operators_gpu as bosonic_ops  # type: ignore
    except Exception as exc:
        return unsupported_backend_payload("bosonic_gpu", f"failed to import bosonic baseline: {exc}")

    cv_ops = bosonic_ops.CVOperatorsGPU()
    cutoff = spec.cutoff

    def run_once() -> dict[str, float]:
        compute_start = time.perf_counter()
        for _ in range(spec.layers):
            cv_ops.s(spec.squeezing_r + 0.0j, cutoff)
            cv_ops.d(spec.displacement_scale * 0.2 + 0.0j, cutoff)
        compute_end = time.perf_counter()
        total_end = compute_end
        return {
            "total_ms": (total_end - compute_start) * 1000.0,
            "compute_ms": (compute_end - compute_start) * 1000.0,
            "communication_ms": 0.0,
            "output_norm": 1.0,
        }

    payload = benchmark_callable(
        case_name=f"bosonic_proxy_layers_{spec.layers}_cutoff_{spec.cutoff}",
        category="baseline_scaling",
        backend_note="bosonic_gpu proxy uses repeated operator construction only",
        spec=spec,
        run_once=run_once,
    )
    return {
        "schema_version": "2.0",
        "generated_at_utc": iso_now(),
        "backend": "bosonic_gpu",
        "single_gpu_focus": True,
        "status": "ok",
        "results": [payload],
    }


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
    parser = argparse.ArgumentParser(description="Run one GPU baseline backend for a selected workload.")
    parser.add_argument(
        "--backend",
        required=True,
        choices=[
            "bosonic_gpu",
            "strawberryfields",
            "strawberryfields_tf",
            "mrmustard",
            "mrmustard_jax",
        ],
    )
    parser.add_argument(
        "--workload",
        choices=["cv_qaoa", "jch_photonic_chain"],
        default="cv_qaoa",
    )
    parser.add_argument("--cutoff", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--squeezing-r", type=float, default=0.5)
    parser.add_argument("--displacement-scale", type=float, default=1.0)
    parser.add_argument("--j-coupling", type=float, default=1.0)
    parser.add_argument("--omega-r", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = WorkloadSpec(
        workload=args.workload,
        cutoff=args.cutoff,
        num_modes=args.num_modes,
        layers=args.layers,
        timesteps=args.timesteps,
        squeezing_r=args.squeezing_r,
        displacement_scale=args.displacement_scale,
        j_coupling=args.j_coupling,
        omega_r=args.omega_r,
        tau=args.tau,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )

    normalized_backend = {
        "strawberryfields": "strawberryfields_tf",
        "mrmustard": "mrmustard_jax",
    }.get(args.backend, args.backend)

    runners = {
        "bosonic_gpu": run_bosonic_gpu_backend,
        "strawberryfields_tf": run_strawberryfields_backend,
        "mrmustard_jax": run_mrmustard_backend,
    }

    try:
        payload = runners[normalized_backend](spec)
    except Exception as exc:  # pragma: no cover
        payload = {
            "schema_version": "2.0",
            "generated_at_utc": iso_now(),
            "backend": normalized_backend,
            "single_gpu_focus": True,
            "status": "error",
            "reason": str(exc),
            "results": [],
        }

    write_json(pathlib.Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
