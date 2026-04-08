#!/usr/bin/env python3
"""Render SC26 paper figures from a run_sc26_paper_eval summary."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time
from typing import Any

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
MPL_CONFIG_DIR = pathlib.Path(tempfile.gettempdir()) / "hybridcvdv_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt

sys.path.insert(0, str(REPO_ROOT))
from experiments.configs.paper_style import SINGLE_COLUMN_PT, apply_paper_style, save_figure


DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "experiments" / "results" / "remote-h100-baseline-sc26_baselines_20260316"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "SC26submission" / "expplots"
BYTES_PER_GIB = float(1024**3)
MODE_SWEEP = [2, 3, 4, 5, 6, 7]
HYBRID_MODE_SWEEP = [2, 3, 4, 5, 6]
CUTOFF_SWEEP = [4, 8, 12, 16, 24, 32]

REFERENCE_SPECS = [
    {
        "key": "cv_qaoa",
        "title": "Pure-CV QAOA",
        "task_id": lambda value: f"reference/sc26_cv_qaoa_nm{value}_c16",
        "color": "#1b9e77",
        "marker": "o",
        "baseline_series": [
            {
                "key": "sf_h100_ms",
                "label": "SF (archived H100)",
                "file": lambda value: f"strawberryfields_tf_cv_qaoa_nm{value}_c16.json",
                "color": "#4c4c4c",
                "marker": "x",
                "linestyle": "--",
            },
            {
                "key": "mrm_h100_ms",
                "label": "MrM (archived H100)",
                "file": lambda value: f"mrmustard_jax_cv_qaoa_nm{value}_c16.json",
                "color": "#7f7f7f",
                "marker": "+",
                "linestyle": ":",
            },
        ],
    },
    {
        "key": "cv_jch",
        "title": "Pure-CV JCH",
        "task_id": lambda value: f"reference/sc26_cv_jch_nm{value}_c16",
        "color": "#d95f02",
        "marker": "s",
        "baseline_series": [
            {
                "key": "sf_h100_ms",
                "label": "SF (archived H100)",
                "file": lambda value: f"strawberryfields_tf_jch_photonic_chain_nm{value}_c16.json",
                "color": "#4c4c4c",
                "marker": "x",
                "linestyle": "--",
            },
            {
                "key": "mrm_h100_ms",
                "label": "MrM (archived H100)",
                "file": lambda value: f"mrmustard_jax_jch_photonic_chain_nm{value}_c16.json",
                "color": "#7f7f7f",
                "marker": "+",
                "linestyle": ":",
            },
        ],
    },
]

MODE_ABLATION_SPECS = [
    {
        "key": "cv_qaoa",
        "title": "Pure-CV QAOA",
        "x_values": MODE_SWEEP,
        "series": [
            {
                "variant": "full",
                "label": "Full",
                "task_id": lambda value: f"reference/sc26_cv_qaoa_nm{value}_c16",
            },
            {
                "variant": "no_symbolic",
                "label": "No symbolic",
                "task_id": lambda value: f"ablation/sc26_cv_qaoa_nm{value}_c16/no_symbolic",
            },
            {
                "variant": "eager_materialize",
                "label": "Eager materialize",
                "task_id": lambda value: f"ablation/sc26_cv_qaoa_nm{value}_c16/eager_materialize",
            },
        ],
    },
    {
        "key": "hybrid_jch",
        "title": "Hybrid JCH (4 qubits + modes)",
        "x_values": HYBRID_MODE_SWEEP,
        "series": [
            {
                "variant": "full",
                "label": "Full",
                "task_id": lambda value: f"ablation/sc26_jch_nq4_nm{value}_c16/full",
            },
            {
                "variant": "no_symbolic",
                "label": "No symbolic",
                "task_id": lambda value: f"ablation/sc26_jch_nq4_nm{value}_c16/no_symbolic",
            },
            {
                "variant": "no_fusion",
                "label": "No fused diagonals",
                "task_id": lambda value: f"ablation/sc26_jch_nq4_nm{value}_c16/no_fusion",
            },
        ],
    },
]

CUTOFF_REFERENCE_SPECS = [
    {
        "key": "cv_qaoa_cutoff",
        "label": "QAOA, 5 modes",
        "task_id": lambda value: f"cutoff/sc26_cv_qaoa_nm5_c{value}",
        "color": "#1b9e77",
        "marker": "o",
    },
    {
        "key": "cv_jch_cutoff",
        "label": "JCH, 5 modes",
        "task_id": lambda value: f"cutoff/sc26_cv_jch_nm5_c{value}",
        "color": "#d95f02",
        "marker": "s",
    },
]

DIAGONAL_PROBE_SPECS = [
    {
        "variant": "full",
        "label": "Full",
        "task_id": lambda value: f"ablation/sc26_diagonal_mix_c{value}/full",
    },
    {
        "variant": "no_diagonal_mixture",
        "label": "No mixture path",
        "task_id": lambda value: f"ablation/sc26_diagonal_mix_c{value}/no_diagonal_mixture",
    },
    {
        "variant": "no_symbolic",
        "label": "No symbolic",
        "task_id": lambda value: f"ablation/sc26_diagonal_mix_c{value}/no_symbolic",
    },
]

MEMORY_CASES = [
    "hdd_vs_full_tensor_qubits_2",
    "hdd_vs_full_tensor_qubits_4",
    "hdd_vs_full_tensor_qubits_8",
    "hdd_vs_full_tensor_qubits_12",
    "hdd_vs_full_tensor_qubits_16",
    "hdd_vs_full_tensor_qubits_20",
]

BLOCK_BREAKDOWN_CASES = [
    {
        "label": "QAOA nm7 c16",
        "task_id": "reference/sc26_cv_qaoa_nm7_c16",
    },
    {
        "label": "Hybrid JCH nq4 nm6 c16",
        "task_id": "ablation/sc26_jch_nq4_nm6_c16/full",
    },
    {
        "label": "Diag. probe c32",
        "task_id": "ablation/sc26_diagonal_mix_c32/full",
    },
]

VARIANT_STYLES = {
    "full": {"color": "#1b9e77", "marker": "o"},
    "no_symbolic": {"color": "#d95f02", "marker": "s"},
    "eager_materialize": {"color": "#7570b3", "marker": "^"},
    "no_fusion": {"color": "#66a61e", "marker": "D"},
    "no_diagonal_mixture": {"color": "#e7298a", "marker": "v"},
}

BLOCK_ORDER = [
    "median_gaussian_symbolic_blocks",
    "median_diagonal_mixture_blocks",
    "median_exact_blocks",
    "median_qubit_only_blocks",
]
BLOCK_LABELS = {
    "median_gaussian_symbolic_blocks": "Gaussian symbolic",
    "median_diagonal_mixture_blocks": "Diagonal mixture",
    "median_exact_blocks": "Exact Fock",
    "median_qubit_only_blocks": "Qubit-only",
}
BLOCK_COLORS = {
    "median_gaussian_symbolic_blocks": "#66c2a5",
    "median_diagonal_mixture_blocks": "#fc8d62",
    "median_exact_blocks": "#8da0cb",
    "median_qubit_only_blocks": "#e78ac3",
}


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(task: dict[str, Any], key: str) -> float | None:
    metrics = task.get("metrics")
    value = None
    if isinstance(metrics, dict):
        value = metrics.get(key)
    
    # Support manifest.json format (wall_time_ms instead of median_total_ms in some cases)
    if value is None:
        if key == "median_total_ms":
            value = task.get("wall_time_ms") or task.get("runner_wall_time_ms")
        elif key == "median_memory_bytes":
            tel = task.get("telemetry_summary") or {}
            value = tel.get("gpu_peak_memory_used_mb")
            if value is not None:
                value = float(value) * 1024 * 1024 # MB to Bytes
        else:
            value = task.get(key)
            
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_task_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    task_map: dict[str, dict[str, Any]] = {}
    # Original 'tasks' list
    for task in summary.get("tasks", []):
        task_id = task.get("id")
        if isinstance(task_id, str):
            task_map[task_id] = task
            
    # New 'artifacts' list from manifest.json
    for artifact in summary.get("artifacts", []):
        name = artifact.get("case_name")
        if isinstance(name, str):
            task_map[name] = artifact
            # Map scaling names to possible reference IDs
            # Handles sc26_qaoa -> reference/sc26_cv_qaoa
            if "sc26_qaoa" in name:
                task_map[name.replace("sc26_qaoa", "reference/sc26_cv_qaoa")] = artifact
                task_map[name.replace("sc26_qaoa", "cutoff/sc26_cv_qaoa")] = artifact
            if "sc26_jch" in name:
                # For pure-cv JCH (nm only)
                if "nq" not in name:
                    task_map[name.replace("sc26_jch", "reference/sc26_cv_jch")] = artifact
                    task_map[name.replace("sc26_jch", "cutoff/sc26_cv_jch")] = artifact
                else:
                    # Hybrid JCH
                    task_map[name.replace("sc26_jch", "ablation/sc26_jch")] = artifact
            
            # Transfer
            if "transfer" in name:
                task_map[name.replace("sc26_transfer", "reference/sc26_transfer")] = artifact
                
    return task_map


def require_task(task_map: dict[str, dict[str, Any]], task_id: str) -> dict[str, Any] | None:
    return task_map.get(task_id)


def load_baseline_runtime(path: pathlib.Path) -> float | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if payload.get("status") != "ok":
        return None
    results = payload.get("results")
    if isinstance(results, list):
        for entry in results:
            if isinstance(entry, dict):
                runtime = metric(entry, "median_total_ms")
                if runtime is not None:
                    return runtime
    return metric(payload, "median_total_ms")


def infer_baseline_label(baseline_dir: pathlib.Path) -> str:
    manifest_path = baseline_dir / "manifest.json"
    if not manifest_path.exists():
        if "h100" in baseline_dir.name.lower():
            return "Archived H100 baseline"
        return "Archived baseline"
    manifest = load_json(manifest_path)
    hardware = manifest.get("hardware")
    if not isinstance(hardware, dict):
        return "Archived baseline"
    gpu_name = hardware.get("gpu_name")
    if isinstance(gpu_name, str) and gpu_name:
        return f"Archived {gpu_name} baseline"
    gpu_model = hardware.get("gpu_model")
    if isinstance(gpu_model, str) and gpu_model:
        return f"Archived {gpu_model} baseline"
    return "Archived baseline"


def make_series(
    task_map: dict[str, dict[str, Any]],
    x_values: list[int],
    task_builder,
    key: str = "median_total_ms",
) -> list[float | None]:
    values: list[float | None] = []
    for x_value in x_values:
        task = require_task(task_map, task_builder(x_value))
        if task is None:
            values.append(None)
        else:
            values.append(metric(task, key))
    return values


def plot_reference_runtime(
    task_map: dict[str, dict[str, Any]],
    baseline_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, panel_aspect=1.05)
    fig, axes = plt.subplots(2, 1, sharex=True)
    derived: dict[str, Any] = {
        "mode_sweep": MODE_SWEEP,
        "series": {},
    }

    legend_handles = []
    legend_labels = []
    for axis, spec in zip(axes, REFERENCE_SPECS):
        full_values = make_series(task_map, MODE_SWEEP, spec["task_id"])
        full_line = axis.plot(
            MODE_SWEEP,
            full_values,
            color=spec["color"],
            marker=spec["marker"],
            linewidth=1.8,
            markersize=4.5,
            label="HybridCVDV",
        )[0]
        if not legend_handles:
            legend_handles = [full_line]
            legend_labels = ["HybridCVDV"]

        baseline_values_by_key: dict[str, list[float | None]] = {}
        for baseline in spec["baseline_series"]:
            baseline_values = [
                load_baseline_runtime(baseline_dir / baseline["file"](mode)) for mode in MODE_SWEEP
            ]
            baseline_values_by_key[baseline["key"]] = baseline_values
            baseline_line = axis.plot(
                MODE_SWEEP,
                baseline_values,
                color=baseline["color"],
                marker=baseline["marker"],
                linewidth=1.4,
                markersize=4.5,
                linestyle=baseline["linestyle"],
                label=baseline["label"],
            )[0]
            if baseline["label"] not in legend_labels:
                legend_handles.append(baseline_line)
                legend_labels.append(baseline["label"])

        axis.set_title(spec["title"])
        axis.set_ylabel("Median runtime (ms)")
        axis.set_yscale("log")
        axis.set_xticks(MODE_SWEEP)
        axis.grid(True, axis="y", alpha=0.35)
        axis.grid(True, axis="x", alpha=0.15, linestyle=":")

        derived["series"][spec["key"]] = [
            {
                "modes": mode,
                "hybridcvdv_ms": hybrid,
                **{
                    baseline["key"]: baseline_values_by_key[baseline["key"]][index]
                    for baseline in spec["baseline_series"]
                },
            }
            for index, (mode, hybrid) in enumerate(zip(MODE_SWEEP, full_values))
        ]

    axes[-1].set_xlabel("Number of CV modes")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    fig.subplots_adjust(top=0.86, hspace=0.42)
    save_figure(fig, output_dir, "sc26_cv_baselines")
    plt.close(fig)
    return derived


def plot_memory_advantage(task_map: dict[str, dict[str, Any]], output_dir: pathlib.Path) -> dict[str, Any]:
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, panel_aspect=1.6)
    fig, ax = plt.subplots()

    qubits: list[int] = []
    naive_gib: list[float] = []
    hdd_gib: list[float] = []
    ratio: list[float] = []
    for case in MEMORY_CASES:
        task = require_task(task_map, f"memory/{case}")
        if task is None:
            continue
        qubit_count_val = (task.get("params") or {}).get("num_qubits")
        if qubit_count_val is None:
            continue
        qubit_count = int(qubit_count_val)
        naive_bytes = metric(task, "naive_full_tensor_bytes")
        hdd_reserved = metric(task, "state_pool_reserved_bytes")
        if naive_bytes is None or hdd_reserved is None:
            continue
        qubits.append(qubit_count)
        naive_gib.append(naive_bytes / BYTES_PER_GIB)
        hdd_gib.append(hdd_reserved / BYTES_PER_GIB)
        ratio.append(naive_bytes / max(hdd_reserved, 1.0))

    if not qubits:
        plt.close(fig)
        return {}

    ax.plot(
        qubits,
        naive_gib,
        color="#4c4c4c",
        marker="x",
        linewidth=1.5,
        markersize=4.5,
        linestyle="--",
        label="Naive full tensor",
    )
    ax.plot(
        qubits,
        hdd_gib,
        color="#1b9e77",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        label="HybridCVDV HDD pool",
    )
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Resident memory (GiB)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.15, linestyle=":")
    ax.legend(frameon=False)

    top_ratio = ratio[-1]
    ax.annotate(
        f"{top_ratio:,.0f}x smaller at {qubits[-1]} qubits",
        xy=(qubits[-1], hdd_gib[-1]),
        xytext=(-4, 12),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
    )

    save_figure(fig, output_dir, "sc26_memory_advantage")
    plt.close(fig)
    return {
        "qubits": qubits,
        "naive_full_tensor_gib": naive_gib,
        "hybridcvdv_hdd_reserved_gib": hdd_gib,
        "compression_ratio": ratio,
    }


def plot_mode_ablation(task_map: dict[str, dict[str, Any]], output_dir: pathlib.Path) -> dict[str, Any]:
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, panel_aspect=1.05)
    fig, axes = plt.subplots(2, 1, sharex=True)
    derived: dict[str, Any] = {"mode_sweep": MODE_SWEEP, "series": {}}

    legend_entries: dict[str, Any] = {}
    for axis, spec in zip(axes, MODE_ABLATION_SPECS):
        axis.set_title(spec["title"])
        axis.set_ylabel("Median runtime (ms)")
        axis.set_yscale("log")
        axis.set_xticks(MODE_SWEEP)
        axis.grid(True, axis="y", alpha=0.35)
        axis.grid(True, axis="x", alpha=0.15, linestyle=":")

        panel_rows: list[dict[str, Any]] = []
        for series in spec["series"]:
            style = VARIANT_STYLES[series["variant"]]
            values = make_series(task_map, spec["x_values"], series["task_id"])
            line = axis.plot(
                spec["x_values"],
                values,
                label=series["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.5,
            )[0]
            legend_entries.setdefault(series["label"], line)
            panel_rows.append(
                {
                    "variant": series["variant"],
                    "label": series["label"],
                    "runtime_ms": values,
                }
            )
        derived["series"][spec["key"]] = panel_rows

    axes[-1].set_xlabel("Number of modes")
    fig.legend(
        list(legend_entries.values()),
        list(legend_entries.keys()),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, hspace=0.42)
    save_figure(fig, output_dir, "sc26_ablation_runtime")
    plt.close(fig)
    return derived


def plot_cutoff_analysis(task_map: dict[str, dict[str, Any]], output_dir: pathlib.Path) -> dict[str, Any]:
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, panel_aspect=1.05)
    fig, axes = plt.subplots(2, 1, sharex=True)
    derived: dict[str, Any] = {"cutoff_sweep": CUTOFF_SWEEP, "series": {}}

    top_axis = axes[0]
    top_axis.set_title("Fixed-mode cutoff sweep")
    top_axis.set_ylabel("Median runtime (ms)")
    top_axis.set_yscale("log")
    top_axis.set_xticks(CUTOFF_SWEEP)
    top_axis.grid(True, axis="y", alpha=0.35)
    top_axis.grid(True, axis="x", alpha=0.15, linestyle=":")

    for spec in CUTOFF_REFERENCE_SPECS:
        values = make_series(task_map, CUTOFF_SWEEP, spec["task_id"])
        top_axis.plot(
            CUTOFF_SWEEP,
            values,
            label=spec["label"],
            color=spec["color"],
            marker=spec["marker"],
            linewidth=1.8,
            markersize=4.5,
        )
        derived["series"][spec["key"]] = values

    top_axis.legend(frameon=False)

    bottom_axis = axes[1]
    bottom_axis.set_title("Diagonal-mixture probe")
    bottom_axis.set_xlabel("Fock cutoff")
    bottom_axis.set_ylabel("Median runtime (ms)")
    bottom_axis.set_yscale("log")
    bottom_axis.set_xticks(CUTOFF_SWEEP)
    bottom_axis.grid(True, axis="y", alpha=0.35)
    bottom_axis.grid(True, axis="x", alpha=0.15, linestyle=":")

    for spec in DIAGONAL_PROBE_SPECS:
        style = VARIANT_STYLES[spec["variant"]]
        runtimes = make_series(task_map, CUTOFF_SWEEP, spec["task_id"])
        mixture_blocks = make_series(
            task_map,
            CUTOFF_SWEEP,
            spec["task_id"],
            key="median_diagonal_mixture_blocks",
        )
        bottom_axis.plot(
            CUTOFF_SWEEP,
            runtimes,
            label=spec["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=4.5,
        )
        derived["series"][spec["variant"]] = {
            "runtime_ms": runtimes,
            "median_diagonal_mixture_blocks": mixture_blocks,
        }

    bottom_axis.legend(frameon=False)
    fig.subplots_adjust(hspace=0.42)
    save_figure(fig, output_dir, "sc26_cutoff_analysis")
    plt.close(fig)
    return derived


def plot_block_breakdown(task_map: dict[str, dict[str, Any]], output_dir: pathlib.Path) -> dict[str, Any]:
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, panel_aspect=1.4)
    fig, ax = plt.subplots()

    derived_rows: list[dict[str, Any]] = []

    for spec in BLOCK_BREAKDOWN_CASES:
        task = require_task(task_map, spec["task_id"])
        if task is None:
            continue
        derived_rows.append(
            {
                "label": spec["label"],
                "task_id": spec["task_id"],
                **{key: metric(task, key) for key in BLOCK_ORDER},
                "median_symbolic_materializations": metric(task, "median_symbolic_materializations"),
            }
        )

    if not derived_rows:
        plt.close(fig)
        return {"cases": []}

    labels = [row["label"] for row in derived_rows]
    x_positions = list(range(len(derived_rows)))
    bottoms = [0.0] * len(derived_rows)

    for block_key in BLOCK_ORDER:
        heights = [float(row.get(block_key) or 0.0) for row in derived_rows]
        ax.bar(
            x_positions,
            heights,
            width=0.62,
            bottom=bottoms,
            color=BLOCK_COLORS[block_key],
            label=BLOCK_LABELS[block_key],
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Median block count")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    for x_position, row, total_height in zip(x_positions, derived_rows, bottoms):
        materializations = row.get("median_symbolic_materializations")
        if materializations is None:
            continue
        ax.text(
            x_position,
            total_height + 0.12,
            f"mat={materializations:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(fig, output_dir, "sc26_block_breakdown")
    plt.close(fig)
    return {"cases": derived_rows}


def write_derived_payload(derived: dict[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "figures": derived,
    }
    output_path = output_dir / "sc26_eval_derived.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path",
        type=pathlib.Path,
        required=True,
        help="Path to summary.json produced by run_sc26_paper_eval.py",
    )
    parser.add_argument(
        "--baseline-dir",
        type=pathlib.Path,
        default=DEFAULT_BASELINE_DIR,
        help="Directory containing archived baseline JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where rendered figures should be written.",
    )
    args = parser.parse_args()

    summary = load_json(args.summary_path)
    task_map = build_task_map(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    derived = {
        "sc26_cv_baselines": plot_reference_runtime(task_map, args.baseline_dir, args.output_dir),
        "sc26_memory_advantage": plot_memory_advantage(task_map, args.output_dir),
        "sc26_ablation_runtime": plot_mode_ablation(task_map, args.output_dir),
        "sc26_cutoff_analysis": plot_cutoff_analysis(task_map, args.output_dir),
        "sc26_block_breakdown": plot_block_breakdown(task_map, args.output_dir),
    }
    derived_path = write_derived_payload(derived, args.output_dir)

    print(f"Wrote figures to {args.output_dir}")
    print(f"Wrote derived metrics to {derived_path}")


if __name__ == "__main__":
    main()
