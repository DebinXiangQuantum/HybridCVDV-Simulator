#!/usr/bin/env python3
"""Plot every CSV in the distributed paper-data summary as an SVG."""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = ["hybridcvdv", "atlas", "bqsim"]
SYSTEM_LABELS = {"hybridcvdv": "HybridCVDV", "atlas": "ATLAS", "bqsim": "BQSim"}
SYSTEM_COLORS = {"hybridcvdv": "#1f77b4", "atlas": "#d62728", "bqsim": "#2ca02c"}
GPU_COUNTS = [1, 2, 4, 6, 8]
FAMILIES = ["cat", "gkp", "jch", "qaoa", "qft", "shors", "transfer", "vqe"]
STATUS_COLORS = {
    "ok": "#2ca02c",
    "oom_single_gpu_pool": "#ff7f0e",
    "timeout": "#9467bd",
    "incorrect_result": "#d62728",
    "crash_host": "#8c564b",
    "host_bound_skipped": "#7f7f7f",
    "unsupported_gpu_count": "#bcbd22",
    "unsupported_backend": "#17becf",
}


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, output_dir: pathlib.Path, name: str) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_manifest_summary(summary_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows = read_csv(summary_dir / "manifest_summary.csv")
    labels = [row["run_id"] for row in rows]
    parsed_rows: list[dict[str, int]] = []
    for row in rows:
        parsed = {}
        for item in row["status_counts"].split("; "):
            if item:
                key, value = item.rsplit(":", 1)
                parsed[key] = int(value)
        parsed_rows.append(parsed)
    statuses = sorted({status for parsed in parsed_rows for status in parsed} | {"ok"}, key=lambda value: (value != "ok", value))
    fig, axis = plt.subplots(figsize=(12.5, 5.2))
    bottom = np.zeros(len(rows))
    for status in statuses:
        values = np.array([parsed.get(status, 0) for parsed in parsed_rows])
        axis.bar(labels, values, bottom=bottom, color=STATUS_COLORS.get(status, "#c7c7c7"), label=status)
        bottom += values
    axis.set_ylabel("Artifacts")
    axis.set_title("Distributed Experiment Manifest Status Summary")
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=4, loc="upper left")
    fig.subplots_adjust(bottom=0.28)
    save(fig, output_dir, "fig09_manifest_status_summary")


def plot_throughput(summary_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows = read_csv(summary_dir / "phase_d_throughput_medians.csv")
    samples: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        samples[(row["system"], row["family"], int(row["gpu_count"]))].append(float(row["median_circuit_evals_per_sec"]))
    families = [family for family in FAMILIES if any((system, family, gpu) in samples for system in SYSTEMS for gpu in GPU_COUNTS)]
    cols = 4
    rows_count = (len(families) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(14.2, 3.1 * rows_count), sharey=False)
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, family in zip(flat_axes, families):
        for system in SYSTEMS:
            values = []
            for gpu in GPU_COUNTS:
                vals = samples.get((system, family, gpu), [])
                values.append(float(np.median(vals)) if vals else np.nan)
            axis.plot(GPU_COUNTS, values, marker="o", linewidth=2, color=SYSTEM_COLORS[system], label=SYSTEM_LABELS[system])
        axis.set_title(f"{family.upper()} workloads")
        axis.set_xlabel("GPU count")
        axis.set_xticks(GPU_COUNTS)
        axis.set_ylabel("Median evals/s")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
    for axis in flat_axes[len(families):]:
        axis.axis("off")
    flat_axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Available Throughput Scaling by Circuit Family")
    fig.subplots_adjust(hspace=0.52, wspace=0.32)
    save(fig, output_dir, "fig10_phase_d_throughput_medians")


def plot_frontier(summary_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows = read_csv(summary_dir / "solvable_frontier_by_system.csv")
    values = {(row["system"], row["family"]): float(row["max_solved_log10_state_space_dimension"])
              for row in rows if row["max_solved_log10_state_space_dimension"]}
    families = [family for family in FAMILIES if any((system, family) in values for system in SYSTEMS)]
    fig, axis = plt.subplots(figsize=(10.2, 4.8))
    width = 0.24
    positions = np.arange(len(families))
    for index, system in enumerate(SYSTEMS):
        axis.bar(positions + (index - 1) * width, [values.get((system, family), 0) for family in families], width,
                 color=SYSTEM_COLORS[system], label=SYSTEM_LABELS[system])
    axis.set_xticks(positions, families, rotation=35, ha="right")
    axis.set_xlabel("Case family")
    axis.set_ylabel("Max solved log10 state-space dimension")
    axis.set_title("Solved State-Space Frontier Across Systems")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    save(fig, output_dir, "fig11_solvable_frontier_by_system")


def plot_status_by_family(summary_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows = read_csv(summary_dir / "status_by_family.csv")
    grouped: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row["system"], row["family"])
        grouped[key][0] += int(row["artifacts"])
        grouped[key][1] += int(row["ok"])
    families = [family for family in FAMILIES if any((system, family) in grouped for system in SYSTEMS)]
    fig, axis = plt.subplots(figsize=(10.2, 4.8))
    width = 0.24
    positions = np.arange(len(families))
    for index, system in enumerate(SYSTEMS):
        rates = [100 * grouped[(system, family)][1] / grouped[(system, family)][0]
                 if (system, family) in grouped and grouped[(system, family)][0] else np.nan for family in families]
        axis.bar(positions + (index - 1) * width, rates, width, color=SYSTEM_COLORS[system], label=SYSTEM_LABELS[system])
    axis.set_ylim(0, 105)
    axis.set_xticks(positions, families, rotation=35, ha="right")
    axis.set_xlabel("Case family")
    axis.set_ylabel("Aggregated success rate (%)")
    axis.set_title("Success Rate by Workload Family")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    save(fig, output_dir, "fig12_family_success_rate_by_system")


def plot_status_by_run_gpu(summary_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows = read_csv(summary_dir / "status_by_run_system_gpu.csv")
    runs = sorted({row["run_id"] for row in rows})
    systems = [system for system in SYSTEMS if any(row["system"] == system for row in rows)]
    fig, axes = plt.subplots(len(systems), 1, figsize=(13.0, 2.9 * len(systems)), sharex=True)
    if len(systems) == 1:
        axes = [axes]
    for axis, system in zip(axes, systems):
        matrix = np.full((len(runs), len(GPU_COUNTS)), np.nan)
        for row in rows:
            if row["system"] == system and int(row["gpu_count"]) in GPU_COUNTS:
                matrix[runs.index(row["run_id"]), GPU_COUNTS.index(int(row["gpu_count"]))] = float(row["success_rate_pct"])
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap="YlGn", aspect="auto")
        axis.set_yticks(range(len(runs)), runs, fontsize=7)
        axis.set_xticks(range(len(GPU_COUNTS)), [f"g{gpu}" for gpu in GPU_COUNTS])
        axis.set_title(SYSTEM_LABELS[system])
        for y in range(len(runs)):
            for x in range(len(GPU_COUNTS)):
                if not np.isnan(matrix[y, x]):
                    axis.text(x, y, f"{matrix[y, x]:.0f}%", ha="center", va="center", fontsize=7)
    axes[-1].set_xlabel("GPU count")
    fig.colorbar(image, ax=axes, label="Success rate (%)", shrink=0.85)
    fig.suptitle("Success Rate by Run, System, and GPU Count")
    save(fig, output_dir, "fig13_run_gpu_success_heatmap")


def plot_coverage(summary_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows = read_csv(summary_dir / "successful_coverage_by_system.csv")
    families = [family for family in FAMILIES if any(row["family"] == family for row in rows)]
    fig, axis = plt.subplots(figsize=(10.2, 4.8))
    width = 0.24
    positions = np.arange(len(families))
    for index, system in enumerate(SYSTEMS):
        values = [next((int(row["unique_successful_case_gpu_configs"]) for row in rows
                        if row["system"] == system and row["family"] == family), 0) for family in families]
        axis.bar(positions + (index - 1) * width, values, width, color=SYSTEM_COLORS[system], label=SYSTEM_LABELS[system])
    axis.set_yscale("symlog", linthresh=10)
    axis.set_xticks(positions, families, rotation=35, ha="right")
    axis.set_xlabel("Case family")
    axis.set_ylabel("Successful case x GPU configurations")
    axis.set_title("Successful Workload Coverage Across Systems")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    save(fig, output_dir, "fig14_successful_coverage_summary")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-dir",
        type=pathlib.Path,
        default=pathlib.Path("experiments/results/distributed_8xH800/paper_data_summary"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("experiments/results/distributed_8xH800/paper_figures"),
    )
    args = parser.parse_args()
    summary_dir = args.summary_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_manifest_summary(summary_dir, output_dir)
    plot_throughput(summary_dir, output_dir)
    plot_frontier(summary_dir, output_dir)
    plot_status_by_family(summary_dir, output_dir)
    plot_status_by_run_gpu(summary_dir, output_dir)
    plot_coverage(summary_dir, output_dir)
    print(f"Wrote summary plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
