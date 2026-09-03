#!/usr/bin/env python3
"""Generate paper-oriented figures from distributed 8xH800 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SYSTEM_LABELS = {
    "hybridcvdv": "HybridCVDV",
    "atlas": "ATLAS",
    "bqsim": "BQSim",
}
SYSTEM_COLORS = {
    "hybridcvdv": "#1f77b4",
    "atlas": "#d62728",
    "bqsim": "#2ca02c",
}
STATUS_COLORS = {
    "ok": "#2ca02c",
    "oom_single_gpu_pool": "#ff7f0e",
    "timeout": "#9467bd",
    "incorrect_result": "#d62728",
    "crash_host": "#8c564b",
    "host_bound_skipped": "#7f7f7f",
    "unsupported_gpu_count": "#bcbd22",
    "unsupported_backend": "#17becf",
    "other": "#c7c7c7",
}
CASE_ORDER = ["cat", "gkp", "jch", "qaoa", "qft", "shors", "transfer", "vqe"]


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_path(root: pathlib.Path, artifact: dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(str(artifact.get("path", "")))
    if path.is_absolute():
        parts = path.parts
        try:
            index = parts.index("experiments")
        except ValueError:
            return path
        return root.joinpath(*parts[index:])
    return root / path


def load_manifest(root: pathlib.Path, manifest: pathlib.Path) -> list[dict[str, Any]]:
    payload = read_json(manifest)
    rows: list[dict[str, Any]] = []
    for artifact in payload.get("artifacts", []):
        row = dict(artifact)
        row["run_id"] = payload.get("run_id")
        row["manifest_generated_at_utc"] = payload.get("generated_at_utc")
        row["manifest_status"] = payload.get("status")
        row["result_path"] = artifact_path(root, artifact)
        rows.append(row)
    return rows


def load_result(row: dict[str, Any]) -> dict[str, Any] | None:
    path = row.get("result_path")
    if not isinstance(path, pathlib.Path) or not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return None
    merged = dict(row)
    merged.update(payload)
    return merged


def median(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.median(clean) if clean else None


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - rank) + clean[hi] * (rank - lo)


def value(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def case_family(case_name: str) -> str:
    if "transfer_" in case_name:
        return "transfer"
    match = re.match(r"sc26_([^_]+)", case_name)
    return match.group(1) if match else "other"


def cutoff(case_name: str) -> int | None:
    match = re.search(r"_c(\d+)(?:$|_)", case_name)
    return int(match.group(1)) if match else None


def named_int(case_name: str, token: str) -> int | None:
    match = re.search(rf"_{token}(\d+)(?:$|_)", case_name)
    return int(match.group(1)) if match else None


def effective_dimension_log10(result: dict[str, Any]) -> float | None:
    c = int(result.get("cutoff") or cutoff(str(result.get("case_name", ""))) or 0)
    modes = int(result.get("num_modes") or named_int(str(result.get("case_name", "")), "nm") or 0)
    qubits = int(result.get("num_qubits") or named_int(str(result.get("case_name", "")), "nq") or 0)
    if c <= 1 or modes <= 0:
        return None
    return modes * math.log10(c) + qubits * math.log10(2)


def figure_out(fig: plt.Figure, output_dir: pathlib.Path, name: str) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_strong_speedup(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows: list[dict[str, Any]] = []
    for manifest in (
        root / "experiments/results/distributed_8xH800/phase-b-hybrid-fixed/manifest.json",
        root / "experiments/results/distributed_8xH800/phase-b-atlas/manifest.json",
    ):
        rows.extend(load_manifest(root, manifest))
    results = [payload for row in rows if (payload := load_result(row))]

    samples: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for result in results:
        if result.get("status") != "ok":
            continue
        if (
            result.get("system") == "hybridcvdv"
            and (result.get("diagnostics") or {}).get("gpu_scaling_eligible") is False
        ):
            continue
        sim = value(result, "timing", "simulation_ms")
        if sim is None or sim <= 0:
            sim = value(result, "timing", "total_wall_ms")
        if sim is None or sim <= 0:
            continue
        samples[(str(result["system"]), str(result["case_name"]), int(result["gpu_count"]))].append(sim)

    speedups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (system, case_name, gpu_count), timings in samples.items():
        base = median(samples.get((system, case_name, 1), []))
        current = median(timings)
        if base and current:
            speedups[(system, gpu_count)].append(base / current)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for system in ("hybridcvdv", "atlas"):
        xs = sorted(g for (s, g), values in speedups.items() if s == system and values)
        if not xs:
            continue
        ys = [median(speedups[(system, g)]) or 0 for g in xs]
        lo = [percentile(speedups[(system, g)], 0.25) or y for g, y in zip(xs, ys)]
        hi = [percentile(speedups[(system, g)], 0.75) or y for g, y in zip(xs, ys)]
        axis.plot(xs, ys, marker="o", lw=2.2, color=SYSTEM_COLORS[system], label=SYSTEM_LABELS[system])
        axis.fill_between(xs, lo, hi, color=SYSTEM_COLORS[system], alpha=0.16, linewidth=0)
    max_gpu = 8
    axis.plot([1, max_gpu], [1, max_gpu], "--", color="#555555", lw=1, label="Ideal")
    axis.set_xticks([1, 2, 4, 6, 8])
    axis.set_xlabel("GPU count")
    axis.set_ylabel("Strong-scaling speedup, T1 / Tg")
    axis.set_title("Strong Scaling on Representative Cases")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure_out(fig, output_dir, "fig1_strong_scaling_speedup")


def plot_throughput(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    results: list[dict[str, Any]] = []
    for system in ("hybridcvdv", "atlas", "bqsim"):
        manifest = root / f"experiments/results/distributed_8xH800/phase-d-{system}/manifest.json"
        results.extend(payload for row in load_manifest(root, manifest) if (payload := load_result(row)))

    by_case: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    wall_by_case: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for result in results:
        if result.get("status") != "ok":
            continue
        case_name = str(result["case_name"])
        if case_family(case_name) != "gkp":
            continue
        throughput = value(result, "throughput", "circuit_evaluations_per_sec")
        if throughput is not None and throughput > 0:
            by_case[(str(result["system"]), case_name, int(result["gpu_count"]))].append(throughput)
        wall = value(result, "timing", "total_wall_ms")
        if wall is not None and wall > 0:
            wall_by_case[(str(result["system"]), case_name, int(result["gpu_count"]))].append(wall)

    gkp_cases = sorted(
        {
            case
            for system, case, gpu in wall_by_case
            if system == "hybridcvdv" and gpu == 1
        }
        & {
            case
            for system, case, gpu in wall_by_case
            if system == "atlas" and gpu == 1
        }
        & {
            case
            for system, case, gpu in wall_by_case
            if system == "bqsim" and gpu == 1
        },
        key=lambda name: cutoff(name) or 0,
    )

    fig = plt.figure(figsize=(14.2, 7.4))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.35)
    runtime_axes = [fig.add_subplot(grid[0, index]) for index in range(5)]
    scaling_axis = fig.add_subplot(grid[1, :])
    width = 0.24
    positions = list(range(len(gkp_cases)))
    offsets = {"hybridcvdv": -width, "atlas": 0.0, "bqsim": width}
    gpu_counts = [1, 2, 4, 6, 8]
    for runtime_axis, gpu in zip(runtime_axes, gpu_counts):
        for system in ("hybridcvdv", "atlas", "bqsim"):
            values = []
            for case in gkp_cases:
                current = median(by_case.get((system, case, gpu), []))
                values.append(current or 0)
            runtime_axis.bar(
                [pos + offsets[system] for pos in positions],
                values,
                width=width,
                color=SYSTEM_COLORS[system],
                label=SYSTEM_LABELS[system],
            )
        runtime_axis.set_yscale("log")
        runtime_axis.set_title(f"g{gpu}", fontsize=10)
        runtime_axis.set_xticks(
            positions,
            [case.replace("sc26_", "") for case in gkp_cases],
            rotation=55,
            ha="right",
            fontsize=7,
        )
        runtime_axis.grid(True, axis="y", alpha=0.25)
        if gpu == 1:
            runtime_axis.set_ylabel("Throughput (circuit evals/s)")
    runtime_axes[2].set_xlabel("Fig. 2a: GKP-case absolute throughput by GPU count")

    bar_width = 0.22
    x_positions = list(range(len(gpu_counts)))
    scaling_offsets = {"hybridcvdv": -bar_width, "atlas": 0.0, "bqsim": bar_width}
    for system in ("hybridcvdv", "atlas", "bqsim"):
        ys = []
        for gpu in gpu_counts:
            ratios = []
            for case in gkp_cases:
                base = median(by_case.get((system, case, 1), []))
                current = median(by_case.get((system, case, gpu), []))
                if base and current:
                    ratios.append(current / base)
            ys.append(median(ratios) or 0)
        scaling_axis.bar(
            [x + scaling_offsets[system] for x in x_positions],
            ys,
            width=bar_width,
            color=SYSTEM_COLORS[system],
            label=SYSTEM_LABELS[system],
        )
    scaling_axis.axhline(1.0, color="#555555", lw=1, ls="--")
    scaling_axis.set_xticks(x_positions, [f"g{gpu}" for gpu in gpu_counts])
    scaling_axis.set_xlabel("GPU count")
    scaling_axis.set_ylabel("Throughput / 1 GPU throughput")
    scaling_axis.set_title("Fig. 2b: Normalized scaling on GKP cases")
    scaling_axis.grid(True, alpha=0.25)
    handles, labels = scaling_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Phase D GKP-Case Throughput Analysis")
    fig.subplots_adjust(bottom=0.12)
    figure_out(fig, output_dir, "fig08_throughput_scaling_diagnostic")


def plot_phase_e_family_gpu_heatmap(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    manifest = root / "experiments/results/distributed_8xH800/phase-e-feasibility-hybridcvdv/manifest.json"
    rows = load_manifest(root, manifest)
    families = [family for family in CASE_ORDER if any(case_family(str(row["case_name"])) == family for row in rows)]
    gpu_counts = [1, 2, 4, 6, 8]
    totals: dict[tuple[str, int], int] = Counter()
    oks: dict[tuple[str, int], int] = Counter()
    for row in rows:
        family = case_family(str(row["case_name"]))
        if family not in families:
            continue
        gpu = int(row["gpu_count"])
        totals[(family, gpu)] += 1
        if row.get("status") == "ok":
            oks[(family, gpu)] += 1

    matrix = []
    for family in families:
        line = []
        for gpu in gpu_counts:
            total = totals[(family, gpu)]
            line.append(100 * oks[(family, gpu)] / total if total else float("nan"))
        matrix.append(line)

    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    image = axis.imshow(matrix, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(gpu_counts)), [f"g{gpu}" for gpu in gpu_counts])
    axis.set_yticks(range(len(families)), families)
    axis.set_xlabel("GPU count")
    axis.set_ylabel("Case family")
    axis.set_title("Phase E Feasibility Success Rate by GPU Count")
    for y, family in enumerate(families):
        for x, gpu in enumerate(gpu_counts):
            total = totals[(family, gpu)]
            text = "n/a" if not total else f"{matrix[y][x]:.0f}%"
            axis.text(x, y, text, ha="center", va="center", fontsize=9, color="#111111")
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("ok combinations (%)")
    figure_out(fig, output_dir, "fig03_family_gpu_success_heatmap")


def plot_phase_e_status_by_family(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    manifests = [
        root / "experiments/results/distributed_8xH800/phase-e-feasibility-hybridcvdv/manifest.json",
        root / "experiments/results/distributed_8xH800/phase-e-hybridcvdv/manifest.json",
    ]
    panels = [load_manifest(root, manifest) for manifest in manifests]
    titles = ["Feasibility scan", "Formal rerun"]
    statuses = ["ok", "oom_single_gpu_pool", "timeout", "incorrect_result", "crash_host", "host_bound_skipped", "other"]
    families = [
        family
        for family in CASE_ORDER
        if any(case_family(str(row["case_name"])) == family for rows in panels for row in rows)
    ]
    fig, axes = plt.subplots(len(panels), len(families), figsize=(15.2, 5.8))
    for row_index, (rows, title) in enumerate(zip(panels, titles)):
        for col_index, family in enumerate(families):
            axis = axes[row_index][col_index]
            counter: Counter[str] = Counter()
            for row in rows:
                if case_family(str(row["case_name"])) != family:
                    continue
                status = str(row.get("status", "other"))
                if status not in statuses:
                    status = "other"
                counter[status] += 1
            labels = [status for status in statuses if counter[status]]
            values = [counter[status] for status in labels]
            if values:
                axis.pie(
                    values,
                    labels=None,
                    colors=[STATUS_COLORS[status] for status in labels],
                    startangle=90,
                    counterclock=False,
                    autopct=lambda pct: f"{pct:.0f}%" if pct >= 8 else "",
                    textprops={"fontsize": 7},
                )
            axis.set_title(f"{title}\n{family}, N={sum(values)}", fontsize=8)
    handles = [Patch(color=STATUS_COLORS[status], label=status) for status in statuses]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("HybridCVDV Phase E Robustness by Workload Family")
    fig.subplots_adjust(bottom=0.16, wspace=0.18, hspace=0.46)
    figure_out(fig, output_dir, "fig02_phase_e_family_status_pies")


def plot_solvable_frontier(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    base = root / "experiments/results/distributed_8xH800"
    rows: list[dict[str, Any]] = []
    for manifest in sorted(base.glob("phase-*/manifest.json")):
        rows.extend(load_manifest(root, manifest))
    results = [payload for row in rows if (payload := load_result(row))]
    systems = ["hybridcvdv", "atlas", "bqsim"]
    families = [
        family
        for family in CASE_ORDER
        if any(
            result.get("status") == "ok"
            and result.get("system") in systems
            and case_family(str(result["case_name"])) == family
            and effective_dimension_log10(result) is not None
            for result in results
        )
    ]
    frontier: dict[tuple[str, str], float] = {}
    for system in systems:
        for family in families:
            values = [
                dim
                for result in results
                if result.get("status") == "ok"
                and result.get("system") == system
                and case_family(str(result["case_name"])) == family
                and (dim := effective_dimension_log10(result)) is not None
            ]
            if values:
                frontier[(system, family)] = max(values)

    fig, axis = plt.subplots(figsize=(9.2, 4.6))
    width = 0.24
    positions = list(range(len(families)))
    offsets = {"hybridcvdv": -width, "atlas": 0.0, "bqsim": width}
    for system in systems:
        values = [frontier.get((system, family), 0.0) for family in families]
        axis.bar(
            [pos + offsets[system] for pos in positions],
            values,
            width=width,
            color=SYSTEM_COLORS[system],
            label=SYSTEM_LABELS[system],
        )
    axis.set_xticks(positions, families, rotation=35, ha="right")
    axis.set_ylabel("Max solved log10 state-space dimension")
    axis.set_xlabel("Case family")
    axis.set_title("Solved State-Space Frontier Across Systems")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    fig.subplots_adjust(bottom=0.22)
    figure_out(fig, output_dir, "fig11_solvable_frontier_by_system")


def plot_successful_coverage_by_system(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    base = root / "experiments/results/distributed_8xH800"
    rows: list[dict[str, Any]] = []
    for manifest in sorted(base.glob("phase-*/manifest.json")):
        rows.extend(load_manifest(root, manifest))
    systems = ["hybridcvdv", "atlas", "bqsim"]
    families = [
        family
        for family in CASE_ORDER
        if any(case_family(str(row["case_name"])) == family and row.get("status") == "ok" for row in rows)
    ]
    solved: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        if row.get("status") != "ok" or row.get("system") not in systems:
            continue
        solved[(str(row["system"]), case_family(str(row["case_name"])))].add(
            (str(row["case_name"]), int(row["gpu_count"]))
        )

    fig, axis = plt.subplots(figsize=(9.2, 4.6))
    width = 0.24
    positions = list(range(len(families)))
    offsets = {"hybridcvdv": -width, "atlas": 0.0, "bqsim": width}
    for system in systems:
        values = [len(solved[(system, family)]) for family in families]
        axis.bar(
            [pos + offsets[system] for pos in positions],
            values,
            width=width,
            color=SYSTEM_COLORS[system],
            label=SYSTEM_LABELS[system],
        )
    axis.set_yscale("symlog", linthresh=10)
    axis.set_xticks(positions, families, rotation=35, ha="right")
    axis.set_ylabel("Unique successful case x GPU configurations")
    axis.set_xlabel("Case family")
    axis.set_title("Successful Workload Coverage Across Distributed Experiments")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    fig.subplots_adjust(bottom=0.22)
    figure_out(fig, output_dir, "fig01_successful_coverage_by_system")


def plot_phase_e_coverage(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    manifests = [
        root / "experiments/results/distributed_8xH800/phase-e-feasibility-hybridcvdv/manifest.json",
        root / "experiments/results/distributed_8xH800/phase-e-hybridcvdv/manifest.json",
    ]
    panels = [load_manifest(root, manifest) for manifest in manifests]
    titles = ["Feasibility scan", "Formal rerun"]
    statuses = ["ok", "oom_single_gpu_pool", "timeout", "incorrect_result", "crash_host", "host_bound_skipped", "other"]
    gpu_counts = [1, 2, 4, 6, 8]
    fig, axes = plt.subplots(2, len(gpu_counts), figsize=(13.2, 5.8))
    for row_index, (rows, title) in enumerate(zip(panels, titles)):
        for col_index, gpu in enumerate(gpu_counts):
            axis = axes[row_index][col_index]
            counter: Counter[str] = Counter()
            for artifact in rows:
                if int(artifact["gpu_count"]) != gpu:
                    continue
                status = str(artifact.get("status", "other"))
                if status not in statuses:
                    status = "other"
                counter[status] += 1
            labels = [status for status in statuses if counter[status]]
            values = [counter[status] for status in labels]
            if values:
                axis.pie(
                    values,
                    labels=None,
                    colors=[STATUS_COLORS[status] for status in labels],
                    startangle=90,
                    counterclock=False,
                    autopct=lambda pct: f"{pct:.0f}%" if pct >= 8 else "",
                    textprops={"fontsize": 8},
                )
            axis.set_title(f"{title}\ng{gpu}, N={sum(values)}", fontsize=9)
    handles = [
        Patch(color=STATUS_COLORS[status], label=status)
        for status in statuses
        if any(Counter(str(row.get("status", "other")) for row in rows)[status] for rows in panels)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("HybridCVDV Phase E Coverage by GPU Count")
    fig.subplots_adjust(bottom=0.17, wspace=0.15, hspace=0.45)
    figure_out(fig, output_dir, "fig04_phase_e_gpu_coverage_pies")


def plot_runtime_breakdown(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    manifest = root / "experiments/results/distributed_8xH800/phase-e-hybridcvdv/manifest.json"
    results = [payload for row in load_manifest(root, manifest) if (payload := load_result(row))]
    parts: dict[int, dict[str, list[float]]] = {
        gpu: {"GPU compute": [], "Host orchestration": [], "Communication": [], "Other": []}
        for gpu in [1, 2, 4, 6, 8]
    }
    for result in results:
        if result.get("status") != "ok":
            continue
        total = value(result, "timing", "simulation_ms") or 0.0
        if total <= 0:
            continue
        compute = max(0.0, value(result, "timing", "gpu_compute_ms") or 0.0)
        host = max(0.0, value(result, "timing", "host_orchestration_ms") or 0.0)
        comm = max(0.0, value(result, "timing", "communication_ms") or 0.0)
        other = max(0.0, total - compute - host - comm)
        gpu = int(result["gpu_count"])
        parts[gpu]["GPU compute"].append(100 * compute / total)
        parts[gpu]["Host orchestration"].append(100 * host / total)
        parts[gpu]["Communication"].append(100 * comm / total)
        parts[gpu]["Other"].append(100 * other / total)

    colors = {
        "GPU compute": "#1f77b4",
        "Host orchestration": "#ff7f0e",
        "Communication": "#9467bd",
        "Other": "#7f7f7f",
    }
    gpu_counts = [1, 2, 4, 6, 8]
    fig, axes = plt.subplots(1, len(gpu_counts), figsize=(13.2, 3.2))
    labels = ["GPU compute", "Host orchestration", "Communication", "Other"]
    for axis, gpu in zip(axes, gpu_counts):
        values = [median(parts[gpu][label]) or 0.0 for label in labels]
        axis.pie(
            values,
            labels=None,
            colors=[colors[label] for label in labels],
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.0f}%" if pct >= 8 else "",
            textprops={"fontsize": 8},
        )
        axis.set_title(f"g{gpu}", fontsize=10)
    fig.suptitle("HybridCVDV Runtime Breakdown by GPU Count, Phase E")
    fig.legend(labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.22, wspace=0.15)
    figure_out(fig, output_dir, "fig07_runtime_breakdown_pies")


def plot_cutoff_success_heatmap(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    manifest = root / "experiments/results/distributed_8xH800/phase-e-feasibility-hybridcvdv/manifest.json"
    rows = load_manifest(root, manifest)
    families = [family for family in CASE_ORDER if any(case_family(str(row["case_name"])) == family for row in rows)]
    cutoffs = [4, 8, 16, 32]
    totals: dict[tuple[str, int], int] = Counter()
    oks: dict[tuple[str, int], int] = Counter()
    for row in rows:
        family = case_family(str(row["case_name"]))
        c = cutoff(str(row["case_name"]))
        if family not in families or c not in cutoffs:
            continue
        totals[(family, c)] += 1
        if row.get("status") == "ok":
            oks[(family, c)] += 1

    matrix = []
    for family in families:
        line = []
        for c in cutoffs:
            total = totals[(family, c)]
            line.append(100 * oks[(family, c)] / total if total else float("nan"))
        matrix.append(line)

    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(cutoffs)), [str(c) for c in cutoffs])
    axis.set_yticks(range(len(families)), families)
    axis.set_xlabel("Fock cutoff")
    axis.set_ylabel("Case family")
    axis.set_title("Phase E Feasibility Success Rate")
    for y, family in enumerate(families):
        for x, c in enumerate(cutoffs):
            total = totals[(family, c)]
            text = "n/a" if not total else f"{matrix[y][x]:.0f}%"
            axis.text(x, y, text, ha="center", va="center", fontsize=9, color="#111111")
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("ok combinations (%)")
    figure_out(fig, output_dir, "fig06_cutoff_family_success_heatmap")


def plot_capacity_status(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows: list[dict[str, Any]] = []
    for manifest in (
        root / "experiments/results/distributed_8xH800/phase-c-hybrid/manifest.json",
        root / "experiments/results/distributed_8xH800/phase-c-atlas/manifest.json",
    ):
        rows.extend(load_manifest(root, manifest))
    systems = ["hybridcvdv", "atlas"]
    statuses = ["ok", "oom_single_gpu_pool", "crash_host", "unsupported_gpu_count", "other"]
    gpu_counts = [1, 2, 4, 6, 8]
    fig, axes = plt.subplots(len(systems), len(gpu_counts), figsize=(13.2, 5.8))
    for row_index, system in enumerate(systems):
        for col_index, gpu in enumerate(gpu_counts):
            axis = axes[row_index][col_index]
            counter: Counter[str] = Counter()
            for artifact in rows:
                if artifact.get("system") != system or int(artifact["gpu_count"]) != gpu:
                    continue
                status = str(artifact.get("status", "other"))
                if status not in statuses:
                    status = "other"
                counter[status] += 1
            labels = [status for status in statuses if counter[status]]
            values = [counter[status] for status in labels]
            if values:
                axis.pie(
                    values,
                    labels=None,
                    colors=[STATUS_COLORS[status] for status in labels],
                    startangle=90,
                    counterclock=False,
                    autopct=lambda pct: f"{pct:.0f}%" if pct >= 8 else "",
                    textprops={"fontsize": 8},
                )
            axis.set_title(f"{SYSTEM_LABELS[system]} g{gpu}\nN={sum(values)}", fontsize=9)
    handles = [
        Patch(color=STATUS_COLORS[status], label=status)
        for status in statuses
        if any(row.get("status") == status for row in rows)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Capacity Scaling Status by GPU Count")
    fig.subplots_adjust(bottom=0.17, wspace=0.15, hspace=0.45)
    figure_out(fig, output_dir, "fig05_capacity_scaling_status_pies")


def write_index(output_dir: pathlib.Path) -> None:
    lines = [
        "# Paper Figure Index",
        "",
        "This directory uses `fig01` to `fig14` naming. The paper should center on coverage, success rate, solvable frontier, and robustness, not raw throughput.",
        "",
        "| Order | File | Use | What it supports |",
        "|---|---|---|---|",
        "| 1 | `fig01_successful_coverage_by_system.svg` | Main | HybridCVDV covers more successful case x GPU configurations than ATLAS/BQSim. |",
        "| 2 | `fig02_phase_e_family_status_pies.svg` | Main/Appendix | Family-level success/failure composition in Phase E. |",
        "| 3 | `fig03_family_gpu_success_heatmap.svg` | Main | Phase E success rate stability across GPU counts. |",
        "| 4 | `fig04_phase_e_gpu_coverage_pies.svg` | Appendix | Phase E GPU-level status composition. |",
        "| 5 | `fig05_capacity_scaling_status_pies.svg` | Main/Appendix | Capacity-scaling failure modes and baseline limitations. |",
        "| 6 | `fig06_cutoff_family_success_heatmap.svg` | Main | Feasible region over family and cutoff. |",
        "| 7 | `fig07_runtime_breakdown_pies.svg` | Appendix | Why more GPUs do not always translate to linear speedup. |",
        "| 8 | `fig08_throughput_scaling_diagnostic.svg` | Diagnostic | Throughput scaling by family; not a primary advantage figure. |",
        "| 9 | `fig09_manifest_status_summary.svg` | Appendix | Data completeness and batch status overview. |",
        "| 10 | `fig10_phase_d_throughput_medians.svg` | Diagnostic | Throughput overview across available circuit families. |",
        "| 11 | `fig11_solvable_frontier_by_system.svg` | Main/Appendix | Largest solved effective state-space frontier across systems. |",
        "| 12 | `fig12_family_success_rate_by_system.svg` | Main/Appendix | Aggregated success rate by family and system. |",
        "| 13 | `fig13_run_gpu_success_heatmap.svg` | Appendix | Failure concentration by run and GPU count. |",
        "| 14 | `fig14_successful_coverage_summary.svg` | Backup | Alternate coverage summary view. |",
        "",
        "## Recommended main set",
        "",
        "1. `fig01_successful_coverage_by_system.svg`",
        "2. `fig03_family_gpu_success_heatmap.svg`",
        "3. `fig06_cutoff_family_success_heatmap.svg`",
        "4. `fig11_solvable_frontier_by_system.svg`",
        "",
        "## Missing data or experiments",
        "",
        "| Missing | Why it matters |",
        "|---|---|",
        "| Strictly matched canonical case set across all three systems | Needed for a fair apples-to-apples comparison. |",
        "| More complete BQSim / ATLAS Phase E scans | Would strengthen the coverage gap claim. |",
        "| Multi-GPU memory peak / aggregate memory plots | Needed to show the value of multi-GPU beyond speed. |",
        "| Larger cutoff and frontier sweeps | Would better expose the actual boundary. |",
        "| Cross-system runtime breakdown | Would explain bottlenecks without relying on throughput alone. |",
        "| Repetition variance / IQR / error bars | Would make the figures more publication-grade. |",
        "| Correctness error versus scale | Would support stability claims at larger sizes. |",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("experiments/results/distributed_8xH800/paper_figures"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_throughput(root, output_dir)
    plot_phase_e_coverage(root, output_dir)
    plot_runtime_breakdown(root, output_dir)
    plot_cutoff_success_heatmap(root, output_dir)
    plot_capacity_status(root, output_dir)
    plot_phase_e_family_gpu_heatmap(root, output_dir)
    plot_phase_e_status_by_family(root, output_dir)
    plot_solvable_frontier(root, output_dir)
    plot_successful_coverage_by_system(root, output_dir)
    write_index(output_dir)
    print(f"Wrote paper figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
