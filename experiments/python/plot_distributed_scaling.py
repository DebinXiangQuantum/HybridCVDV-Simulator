#!/usr/bin/env python3
"""Create the required distributed scaling figures from merged artifacts."""

from __future__ import annotations

import argparse
import pathlib
import statistics
from collections import defaultdict
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from distributed_common import read_json


SYSTEMS = ("hybridcvdv", "atlas", "bqsim")


def value(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def grouped(
    results: list[dict[str, Any]],
    phase: str,
    extractor: Callable[[dict[str, Any]], float | None],
) -> dict[str, dict[int, float]]:
    samples: dict[tuple[str, int], list[float]] = defaultdict(list)
    for result in results:
        if result.get("status") != "ok" or result.get("phase") != phase:
            continue
        if (
            phase == "strong"
            and result.get("system") == "hybridcvdv"
            and result.get("diagnostics", {}).get("gpu_scaling_eligible") is False
        ):
            continue
        extracted = extractor(result)
        if extracted is not None:
            samples[(result["system"], int(result["gpu_count"]))].append(extracted)
    output: dict[str, dict[int, float]] = defaultdict(dict)
    for (system, gpu_count), values in samples.items():
        output[system][gpu_count] = statistics.median(values)
    return output


def line_plot(data: dict[str, dict[int, float]], ylabel: str, output: pathlib.Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for system in SYSTEMS:
        points = data.get(system, {})
        if points:
            xs = sorted(points)
            axis.plot(xs, [points[x] for x in xs], marker="o", label=system)
    axis.set_xlabel("GPU count")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_timing(
    results: list[dict[str, Any]], phase: str, output: pathlib.Path
) -> None:
    simulation = grouped(results, phase, lambda result: value(result, "timing", "simulation_ms"))
    compute = grouped(results, phase, lambda result: value(result, "timing", "gpu_compute_ms"))
    communication = grouped(results, phase, lambda result: value(result, "timing", "communication_ms"))
    fig, (time_axis, speed_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, system in enumerate(SYSTEMS):
        points = simulation.get(system, {})
        for gpu_count in sorted(points):
            total = points[gpu_count]
            comp = min(total, compute.get(system, {}).get(gpu_count, 0.0))
            comm = min(max(0.0, total - comp), communication.get(system, {}).get(gpu_count, 0.0))
            sync = max(0.0, total - comp - comm)
            x = gpu_count + (index - 1) * 0.18
            time_axis.bar(x, comp, 0.17, label=f"{system} compute" if gpu_count == min(points) else None)
            time_axis.bar(x, comm, 0.17, bottom=comp, label=f"{system} communication" if gpu_count == min(points) else None)
            time_axis.bar(x, sync, 0.17, bottom=comp + comm)
        if 1 in points and points[1] > 0:
            xs = sorted(points)
            speed_axis.plot(xs, [points[1] / points[x] for x in xs], marker="o", label=system)
    time_axis.set_xlabel("GPU count")
    time_axis.set_ylabel("Simulation time (ms)")
    speed_axis.set_xlabel("GPU count")
    speed_axis.set_ylabel("Strong-scaling speedup")
    for axis in (time_axis, speed_axis):
        axis.grid(True, alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def memory_values(result: dict[str, Any]) -> float | None:
    per_gpu = result.get("memory", {}).get("per_gpu", {})
    values = [
        item.get("gpu_memory_peak_bytes", 0)
        for item in per_gpu.values()
        if isinstance(item, dict)
    ]
    if values:
        return float(sum(values)) / (1024**3)
    direct = value(result, "memory", "gpu_memory_peak_bytes")
    return direct / (1024**3) if direct is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("auto", "smoke", "strong", "capacity", "throughput", "full"),
        default="auto",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = read_json(args.manifest)["artifacts"]
    available_phases = {result.get("phase") for result in results}
    timing_phase = args.phase
    if timing_phase == "auto":
        timing_phase = "strong" if "strong" in available_phases else "smoke"
    throughput_phase = (
        "throughput"
        if args.phase == "auto" and "throughput" in available_phases
        else timing_phase
    )
    plot_timing(results, timing_phase, args.output_dir / "simulation_breakdown_speedup.png")
    line_plot(
        grouped(
            results,
            throughput_phase,
            lambda result: value(result, "throughput", "circuit_evaluations_per_sec"),
        ),
        "Circuit evaluations / s",
        args.output_dir / "throughput.png",
    )
    line_plot(
        grouped(results, timing_phase, memory_values),
        "Aggregate peak GPU memory (GiB)",
        args.output_dir / "memory.png",
    )
    print(
        f"Wrote plots to {args.output_dir} "
        f"(timing/memory={timing_phase}, throughput={throughput_phase})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
