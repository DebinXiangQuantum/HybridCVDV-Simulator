#!/usr/bin/env python3
"""Render SC26 hybrid-baseline summary figures from archived raw results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
MPL_CONFIG_DIR = pathlib.Path(tempfile.gettempdir()) / "hybridcvdv_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(REPO_ROOT))
from experiments.configs.paper_style import DOUBLE_COLUMN_PT, apply_paper_style, save_figure


DEFAULT_OUR_DIR = REPO_ROOT / "experiments" / "results" / "sc26_scaling_full_20260405"
DEFAULT_CV_DIR = (
    REPO_ROOT / "experiments" / "results" / "remote-h100-baseline-sc26_baselines_hybrid_20260316"
)
DEFAULT_BQSIM_CSV = REPO_ROOT / "baselines" / "results" / "bqsim_results.csv"
DEFAULT_BOSONIC_CSV = REPO_ROOT / "baselines" / "results" / "bosonicGPU.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "SC26submission" / "expplots"
STUDY_CUTOFF = 16

WORKLOAD_ORDER = ["jch", "vqe"]
WORKLOAD_LABELS = {
    "jch": "JCH",
    "vqe": "VQE",
}
WORKLOAD_COLORS = {
    "jch": "#4c78a8",
    "vqe": "#f58518",
}
BACKEND_ORDER = [
    "hybridcvdv",
    "bqsim",
    "bosonic_gpu",
    "strawberryfields_tf",
    "mrmustard_jax",
]
BACKEND_LABELS = {
    "hybridcvdv": "HybridCVDV",
    "bqsim": "BQSim",
    "bosonic_gpu": "Bosonic-GPU",
    "strawberryfields_tf": "SF-TF",
    "mrmustard_jax": "MrM-JAX",
}
BACKEND_COLORS = {
    "hybridcvdv": "#1b9e77",
    "bqsim": "#7570b3",
    "bosonic_gpu": "#d95f02",
    "strawberryfields_tf": "#808080",
    "mrmustard_jax": "#555555",
}


@dataclass(frozen=True)
class BenchmarkKey:
    workload: str
    num_qubits: int
    num_modes: int
    cutoff: int


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def geometric_mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and float(value) > 0.0]
    if not clean:
        return None
    return math.exp(sum(math.log(value) for value in clean) / float(len(clean)))


def parse_our_results(results_dir: pathlib.Path) -> Tuple[Dict[BenchmarkKey, Dict[str, Any]], Dict[str, Any]]:
    # Regular pattern for old results and new scaling results
    pattern = re.compile(r"sc26_(jch|vqe)_nq(\d+)_nm(\d+)_c(\d+)(?:__hybridcvdv)?$")
    records: Dict[BenchmarkKey, Dict[str, Any]] = {}
    metadata = {
        "num_gpus": set(),
        "gpu_name": set(),
        "measured_runs": set(),
        "warmup_runs": set(),
        "gaussian_symbolic_mode_limit": set(),
        "use_interaction_picture": set(),
    }

    # Scan results_dir/*.json
    for path in sorted(results_dir.glob("*.json")):
        if path.name in ("manifest.json", "failure_summary.json"):
            continue
        payload = load_json(path)
        results = payload.get("results") or []
        if not results:
            continue
        record = results[0]
        name = str(record.get("name", path.stem))
        match = pattern.match(name)
        if match is None or record.get("status") not in ("ok", "success"):
            continue

        metrics = record.get("metrics") or {}
        runtime_ms = metric_float(metrics, "runner_wall_time_ms", "median_total_ms")
        if runtime_ms is None:
            continue

        key = BenchmarkKey(
            workload=match.group(1),
            num_qubits=int(match.group(2)),
            num_modes=int(match.group(3)),
            cutoff=int(match.group(4)),
        )
        records[key] = {
            "runtime_ms": float(runtime_ms),
            "compute_ms": float(metrics.get("median_compute_ms", 0.0)),
            "memory_bytes": float(metrics.get("median_memory_bytes", 0.0)),
            "params": record.get("params") or {},
        }
        metadata["num_gpus"].add(payload.get("num_gpus"))
        metadata["gpu_name"].add(payload.get("device", {}).get("name"))
        metadata["measured_runs"].add((record.get("params") or {}).get("measured_runs"))
        metadata["warmup_runs"].add((record.get("params") or {}).get("warmup_runs"))
        metadata["gaussian_symbolic_mode_limit"].add(payload.get("gaussian_symbolic_mode_limit"))
        metadata["use_interaction_picture"].add(payload.get("use_interaction_picture"))

    materialized_metadata = {}
    for key, values in metadata.items():
        materialized_metadata[key] = sorted(value for value in values if value is not None)
    return records, materialized_metadata


def metric_float(metrics: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        val = metrics.get(key)
        if val not in (None, "", "None"):
            return float(val)
    return None


def parse_cv_attempts(results_dir: pathlib.Path) -> Dict[Tuple[str, BenchmarkKey], Dict[str, Any]]:
    pattern = re.compile(
        r"(strawberryfields_tf|mrmustard_jax)_(vqe_circuit|jch_simulation_circuit)_nq(\d+)_nm(\d+)_c(\d+)\.json$"
    )
    attempts: Dict[Tuple[str, BenchmarkKey], Dict[str, Any]] = {}

    for path in sorted(results_dir.glob("*.json")):
        match = pattern.match(path.name)
        if match is None:
            continue
        backend = match.group(1)
        workload = "vqe" if match.group(2) == "vqe_circuit" else "jch"
        key = BenchmarkKey(
            workload=workload,
            num_qubits=int(match.group(3)),
            num_modes=int(match.group(4)),
            cutoff=int(match.group(5)),
        )
        payload = load_json(path)
        runtime_ms = None
        if payload.get("status") == "ok":
            results = payload.get("results") or []
            if results:
                runtime_ms = (results[0].get("metrics") or {}).get("median_total_ms")
        attempts[(backend, key)] = {
            "status": payload.get("status"),
            "runtime_ms": None if runtime_ms is None else float(runtime_ms),
            "reason": payload.get("reason", ""),
        }
    return attempts


def parse_bqsim_csv(csv_path: pathlib.Path) -> Dict[BenchmarkKey, Dict[str, Any]]:
    pattern = re.compile(r"sc26_(jch|vqe)_nq(\d+)_nm(\d+)_c(\d+)$")
    records: Dict[BenchmarkKey, Dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            match = pattern.match(row[0])
            if match is None:
                continue
            total_ms = float(row[2])
            records[
                BenchmarkKey(
                    workload=match.group(1),
                    num_qubits=int(match.group(2)),
                    num_modes=int(match.group(3)),
                    cutoff=int(match.group(4)),
                )
            ] = {
                "runtime_ms": None if total_ms < 0.0 else total_ms,
                "status": "ok" if total_ms >= 0.0 else "failed",
                "memory_bytes": None if float(row[5]) < 0.0 else float(row[5]),
            }
    return records


def parse_bosonic_csv(csv_path: pathlib.Path) -> Dict[BenchmarkKey, Dict[str, Any]]:
    records: Dict[BenchmarkKey, Dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row or row[0] not in ("jch_simulation_circuit", "vqe_circuit"):
                continue

            failure_msg = ""
            if row[-1] and re.fullmatch(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", row[-1], re.IGNORECASE) is None:
                failure_msg = row[-1]

            params_slice = row[1:-5] if not failure_msg else row[1:-6]
            params: Dict[str, str] = {}
            for item in params_slice:
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                params[key] = value

            num_qubits = params.get("num_qubits", params.get("qubits"))
            num_modes = params.get("num_modes", params.get("qumodes"))
            cutoff = params.get("cutoff")
            if num_qubits is None or num_modes is None or cutoff is None:
                continue

            if failure_msg:
                runtime_ms = None
                memory_bytes = None
            else:
                runtime_ms = float(row[-4])
                memory_bytes = float(row[-1])

            workload = "jch" if row[0] == "jch_simulation_circuit" else "vqe"
            records[
                BenchmarkKey(
                    workload=workload,
                    num_qubits=int(num_qubits),
                    num_modes=int(num_modes),
                    cutoff=int(cutoff),
                )
            ] = {
                "runtime_ms": runtime_ms,
                "status": "ok" if runtime_ms is not None and runtime_ms > 0.0 else "failed",
                "memory_bytes": memory_bytes,
                "failure_msg": failure_msg,
            }
    return records


def build_study_keys(our_results: Mapping[BenchmarkKey, Dict[str, Any]], cutoff: int) -> List[BenchmarkKey]:
    return sorted(
        [key for key in our_results.keys() if key.cutoff == cutoff],
        key=lambda item: (WORKLOAD_ORDER.index(item.workload), item.num_qubits, item.num_modes),
    )


def count_successes(
    study_keys: Sequence[BenchmarkKey],
    our_results: Mapping[BenchmarkKey, Dict[str, Any]],
    bqsim_results: Mapping[BenchmarkKey, Dict[str, Any]],
    bosonic_results: Mapping[BenchmarkKey, Dict[str, Any]],
    cv_attempts: Mapping[Tuple[str, BenchmarkKey], Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    success: Dict[str, Dict[str, int]] = {}
    for backend in BACKEND_ORDER:
        success[backend] = {workload: 0 for workload in WORKLOAD_ORDER}

    for key in study_keys:
        success["hybridcvdv"][key.workload] += 1
        if bqsim_results.get(key, {}).get("runtime_ms") is not None:
            success["bqsim"][key.workload] += 1
        if bosonic_results.get(key, {}).get("runtime_ms") is not None:
            success["bosonic_gpu"][key.workload] += 1
        for backend in ("strawberryfields_tf", "mrmustard_jax"):
            if cv_attempts.get((backend, key), {}).get("status") == "ok":
                success[backend][key.workload] += 1
    return success


def summarize_runtime_ratios(
    study_keys: Sequence[BenchmarkKey],
    our_results: Mapping[BenchmarkKey, Dict[str, Any]],
    baseline_name: str,
    baseline_results: Mapping[BenchmarkKey, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for workload in WORKLOAD_ORDER:
        ratios: List[float] = []
        faster_keys: List[BenchmarkKey] = []
        slower_keys: List[BenchmarkKey] = []
        for key in study_keys:
            if key.workload != workload:
                continue
            baseline_runtime = baseline_results.get(key, {}).get("runtime_ms")
            if baseline_runtime is None:
                continue
            ratio = float(baseline_runtime) / float(our_results[key]["runtime_ms"])
            ratios.append(ratio)
            if ratio > 1.0:
                faster_keys.append(key)
            else:
                slower_keys.append(key)

        summary[workload] = {
            "backend": baseline_name,
            "overlap_count": len(ratios),
            "geomean_ratio": geometric_mean(ratios),
            "min_ratio": min(ratios) if ratios else None,
            "max_ratio": max(ratios) if ratios else None,
            "ours_faster_count": len(faster_keys),
            "baseline_faster_count": len(slower_keys),
            "ours_faster_cases": [key.__dict__ for key in faster_keys],
            "baseline_faster_cases": [key.__dict__ for key in slower_keys],
        }
    return summary


def strongest_ratio_case(
    study_keys: Sequence[BenchmarkKey],
    our_results: Mapping[BenchmarkKey, Dict[str, Any]],
    baseline_name: str,
    baseline_results: Mapping[BenchmarkKey, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    best_payload = None
    best_ratio = None
    for key in study_keys:
        baseline_runtime = baseline_results.get(key, {}).get("runtime_ms")
        if baseline_runtime is None:
            continue
        ratio = float(baseline_runtime) / float(our_results[key]["runtime_ms"])
        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
            best_payload = {
                "backend": baseline_name,
                "key": key.__dict__,
                "baseline_runtime_ms": float(baseline_runtime),
                "hybridcvdv_runtime_ms": float(our_results[key]["runtime_ms"]),
                "ratio": ratio,
            }
    return best_payload


def make_coverage_and_runtime_figure(
    support_counts: Mapping[str, Mapping[str, int]],
    ratio_summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    study_case_count: int,
    study_workload_counts: Mapping[str, int],
    output_dir: pathlib.Path,
) -> List[pathlib.Path]:
    apply_paper_style(width_pt=DOUBLE_COLUMN_PT, ncols=2, panel_aspect=1.6)
    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax_support, ax_runtime = axes

    ax_support.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_support.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax_runtime.text(
        0.02,
        0.98,
        "(b)",
        transform=ax_runtime.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    x_positions = list(range(len(BACKEND_ORDER)))
    jch_values = [support_counts[backend]["jch"] for backend in BACKEND_ORDER]
    vqe_values = [support_counts[backend]["vqe"] for backend in BACKEND_ORDER]

    ax_support.bar(
        x_positions,
        jch_values,
        width=0.66,
        color=WORKLOAD_COLORS["jch"],
        label="JCH",
    )
    ax_support.bar(
        x_positions,
        vqe_values,
        width=0.66,
        bottom=jch_values,
        color=WORKLOAD_COLORS["vqe"],
        label="VQE",
    )
    for x_position, backend in zip(x_positions, BACKEND_ORDER):
        total = jch_values[x_position] + vqe_values[x_position]
        label_y = total + 0.6 if total > 0 else 0.8
        ax_support.text(
            x_position,
            label_y,
            "{}/{}".format(total, study_case_count),
            ha="center",
            va="bottom",
            fontsize=6.6,
        )

    ax_support.set_title("Cutoff-16 Backend Coverage")
    ax_support.set_ylabel("Successful study-set runs")
    ax_support.set_xticks(x_positions)
    ax_support.set_xticklabels([BACKEND_LABELS[backend] for backend in BACKEND_ORDER], rotation=18, ha="right")
    ax_support.set_ylim(0, study_case_count + 4)
    ax_support.text(
        0.98,
        0.98,
        "Study set: {} JCH + {} VQE".format(
            study_workload_counts["jch"],
            study_workload_counts["vqe"],
        ),
        transform=ax_support.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    group_positions = [0, 1]
    bar_width = 0.32
    for index, (backend, label) in enumerate((("bqsim", "BQSim"), ("bosonic_gpu", "Bosonic-GPU"))):
        ratios = [ratio_summaries[backend][workload]["geomean_ratio"] for workload in WORKLOAD_ORDER]
        offsets = [-0.5 * bar_width, 0.5 * bar_width]
        bars = ax_runtime.bar(
            [position + offsets[index] for position in group_positions],
            ratios,
            width=bar_width,
            color=BACKEND_COLORS[backend],
            label=label,
        )
        for bar, workload in zip(bars, WORKLOAD_ORDER):
            ratio = ratio_summaries[backend][workload]["geomean_ratio"]
            overlap = ratio_summaries[backend][workload]["overlap_count"]
            ax_runtime.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(ratio) + 0.18,
                "{:.1f}x\nn={}".format(float(ratio), overlap),
                ha="center",
                va="bottom",
                fontsize=6.3,
            )

    ax_runtime.axhline(1.0, color=BACKEND_COLORS["hybridcvdv"], linestyle="--", linewidth=1.0)
    ax_runtime.text(
        1.38,
        1.12,
        "HybridCVDV = 1",
        color=BACKEND_COLORS["hybridcvdv"],
        fontsize=6.4,
        ha="right",
        va="bottom",
    )
    ax_runtime.text(
        0.02,
        0.90,
        ">1 favors HybridCVDV",
        transform=ax_runtime.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
    )
    ax_runtime.set_title("Geometric-Mean Baseline Slowdown")
    ax_runtime.set_ylabel("Baseline runtime / HybridCVDV")
    ax_runtime.set_xticks(group_positions)
    ax_runtime.set_xticklabels([WORKLOAD_LABELS[workload] for workload in WORKLOAD_ORDER])
    ax_runtime.set_ylim(0, max(10.2, max(float(ratio_summaries[backend][workload]["geomean_ratio"]) for backend in ("bqsim", "bosonic_gpu") for workload in WORKLOAD_ORDER) + 1.2))
    fig.legend(
        handles=[
            Patch(facecolor=WORKLOAD_COLORS["jch"], label="JCH"),
            Patch(facecolor=WORKLOAD_COLORS["vqe"], label="VQE"),
            Patch(facecolor=BACKEND_COLORS["bqsim"], label="BQSim"),
            Patch(facecolor=BACKEND_COLORS["bosonic_gpu"], label="Bosonic-GPU"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        handlelength=0.8,
        handleheight=0.8,
        borderpad=0.2,
    )

    fig.subplots_adjust(wspace=0.34, top=0.88, bottom=0.2)
    saved_paths = save_figure(fig, output_dir, "sc26_hybrid_baseline_summary")
    plt.close(fig)
    return saved_paths


def write_derived_payload(payload: Mapping[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sc26_hybrid_eval_derived.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--our-dir", type=pathlib.Path, default=DEFAULT_OUR_DIR)
    parser.add_argument("--cv-dir", type=pathlib.Path, default=DEFAULT_CV_DIR)
    parser.add_argument("--bqsim-csv", type=pathlib.Path, default=DEFAULT_BQSIM_CSV)
    parser.add_argument("--bosonic-csv", type=pathlib.Path, default=DEFAULT_BOSONIC_CSV)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    our_results, our_metadata = parse_our_results(args.our_dir)
    cv_attempts = parse_cv_attempts(args.cv_dir)
    bqsim_results = parse_bqsim_csv(args.bqsim_csv)
    bosonic_results = parse_bosonic_csv(args.bosonic_csv)

    study_keys = build_study_keys(our_results, STUDY_CUTOFF)
    study_workload_counts = {
        workload: sum(1 for key in study_keys if key.workload == workload) for workload in WORKLOAD_ORDER
    }
    support_counts = count_successes(study_keys, our_results, bqsim_results, bosonic_results, cv_attempts)
    ratio_summaries = {
        "bqsim": summarize_runtime_ratios(study_keys, our_results, "BQSim", bqsim_results),
        "bosonic_gpu": summarize_runtime_ratios(study_keys, our_results, "Bosonic-GPU", bosonic_results),
    }
    strongest_cases = [
        payload
        for payload in (
            strongest_ratio_case(study_keys, our_results, "BQSim", bqsim_results),
            strongest_ratio_case(study_keys, our_results, "Bosonic-GPU", bosonic_results),
        )
        if payload is not None
    ]

    figure_paths = make_coverage_and_runtime_figure(
        support_counts=support_counts,
        ratio_summaries=ratio_summaries,
        study_case_count=len(study_keys),
        study_workload_counts=study_workload_counts,
        output_dir=args.output_dir,
    )

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "study_cutoff": STUDY_CUTOFF,
        "study_case_count": len(study_keys),
        "study_workload_counts": study_workload_counts,
        "hybridcvdv_metadata": our_metadata,
        "figure_paths": [str(path) for path in figure_paths],
        "support_counts": support_counts,
        "ratio_summaries": ratio_summaries,
        "strongest_hybridcvdv_advantage_cases": strongest_cases,
    }
    derived_path = write_derived_payload(payload, args.output_dir)

    print("Wrote figure to {}".format(", ".join(str(path) for path in figure_paths)))
    print("Wrote derived metrics to {}".format(derived_path))


if __name__ == "__main__":
    main()
