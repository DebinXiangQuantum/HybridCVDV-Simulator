#!/usr/bin/env python3
"""Collect distributed 8xH800 experiment data into paper-friendly tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any


CASE_ORDER = ["cat", "gkp", "jch", "qaoa", "qft", "shors", "transfer", "vqe"]
SYSTEMS = ["hybridcvdv", "atlas", "bqsim"]
GPU_COUNTS = [1, 2, 4, 6, 8]


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


def median(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.median(clean) if clean else None


def effective_dimension_log10(row: dict[str, Any]) -> float | None:
    case_name = str(row.get("case_name", ""))
    c = int(row.get("cutoff") or cutoff(case_name) or 0)
    modes = int(row.get("num_modes") or named_int(case_name, "nm") or 0)
    qubits = int(row.get("num_qubits") or named_int(case_name, "nq") or 0)
    if c <= 1 or modes <= 0:
        return None
    return modes * math.log10(c) + qubits * math.log10(2)


def throughput_per_sec(row: dict[str, Any]) -> float | None:
    throughput = row.get("throughput") or {}
    direct = throughput.get("circuit_evaluations_per_sec")
    if isinstance(direct, (int, float)) and direct > 0:
        return float(direct)
    completed = (
        throughput.get("completed_circuit_evaluations")
        or throughput.get("completed_input_states")
        or throughput.get("completed_batches")
    )
    timing = row.get("timing") or {}
    total_ms = timing.get("total_wall_ms") or timing.get("simulation_ms")
    if isinstance(completed, (int, float)) and isinstance(total_ms, (int, float)) and completed > 0 and total_ms > 0:
        return 1000.0 * float(completed) / float(total_ms)
    return None


def load_rows(root: pathlib.Path, result_root: pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for manifest in sorted(result_root.glob("*/manifest.json")):
        payload = read_json(manifest)
        run_id = str(payload.get("run_id") or manifest.parent.name)
        for artifact in payload.get("artifacts", []):
            row = dict(artifact)
            row["run_id"] = run_id
            row["manifest_generated_at_utc"] = payload.get("generated_at_utc")
            row["manifest_status"] = payload.get("status")
            row["result_path"] = artifact_path(root, artifact)
            row["family"] = case_family(str(row.get("case_name", "")))
            row["cutoff"] = row.get("cutoff") or cutoff(str(row.get("case_name", "")))
            manifest_rows.append(row)
            path = row["result_path"]
            if isinstance(path, pathlib.Path) and path.exists() and path.stat().st_size > 0:
                try:
                    result = read_json(path)
                except json.JSONDecodeError:
                    continue
                merged = dict(row)
                merged.update(result)
                merged["family"] = case_family(str(merged.get("case_name", "")))
                merged["cutoff"] = merged.get("cutoff") or cutoff(str(merged.get("case_name", "")))
                result_rows.append(merged)
    return manifest_rows, result_rows


def load_bqsim_supplemental_rows(root: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    supplemental_roots = [
        root / "experiments/results/distributed_smoke_all",
        root / "baselines/BQSim-main/experiments/results/distributed_smoke_all",
    ]
    seen_dirs: set[pathlib.Path] = set()
    for supplemental_root in supplemental_roots:
        if not supplemental_root.exists():
            continue
        for result_dir in sorted({path.parent for path in supplemental_root.rglob("*.json") if "/bqsim/" in str(path)}):
            if result_dir in seen_dirs:
                continue
            seen_dirs.add(result_dir)
            preferred = result_dir / "result.json"
            fallback = result_dir / "raw.json"
            path = preferred if preferred.exists() else fallback
            if not path.exists():
                continue
            try:
                row = read_json(path)
            except json.JSONDecodeError:
                continue
            if row.get("system") != "bqsim" or row.get("status") != "ok":
                continue
            if throughput_per_sec(row) is None:
                continue
            case_name = str(row.get("case_name", ""))
            row["run_id"] = "bqsim-supplemental-smoke"
            row["phase"] = row.get("phase") or "smoke"
            row["family"] = case_family(case_name)
            row["cutoff"] = row.get("cutoff") or cutoff(case_name)
            row["result_path"] = path
            row["manifest_generated_at_utc"] = ""
            row["manifest_status"] = "supplemental"
            rows.append(row)
    return rows


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize(root: pathlib.Path, output_dir: pathlib.Path) -> None:
    result_root = root / "experiments/results/distributed_8xH800"
    manifests, results = load_rows(root, result_root)
    supplemental_throughput_results = load_bqsim_supplemental_rows(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_summary = []
    for run_id in sorted({row["run_id"] for row in manifests}):
        rows = [row for row in manifests if row["run_id"] == run_id]
        statuses = Counter(str(row.get("status", "other")) for row in rows)
        systems = ",".join(sorted({str(row.get("system", "")) for row in rows if row.get("system")}))
        phases = ",".join(sorted({str(row.get("phase", "")) for row in rows if row.get("phase")}))
        manifest_summary.append(
            {
                "run_id": run_id,
                "generated_at_utc": rows[0].get("manifest_generated_at_utc"),
                "manifest_status": rows[0].get("manifest_status"),
                "systems": systems,
                "phases": phases,
                "artifacts": len(rows),
                "ok": statuses["ok"],
                "non_ok": len(rows) - statuses["ok"],
                "status_counts": "; ".join(f"{key}:{value}" for key, value in sorted(statuses.items())),
            }
        )
    write_csv(
        output_dir / "manifest_summary.csv",
        ["run_id", "generated_at_utc", "manifest_status", "systems", "phases", "artifacts", "ok", "non_ok", "status_counts"],
        manifest_summary,
    )

    status_by_run_system_gpu = []
    grouped: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    for row in manifests:
        grouped[(str(row["run_id"]), str(row.get("system", "")), int(row.get("gpu_count") or 0))][str(row.get("status", "other"))] += 1
    for (run_id, system, gpu_count), counts in sorted(grouped.items()):
        total = sum(counts.values())
        status_by_run_system_gpu.append(
            {
                "run_id": run_id,
                "system": system,
                "gpu_count": gpu_count,
                "artifacts": total,
                "ok": counts["ok"],
                "success_rate_pct": round(100 * counts["ok"] / total, 2) if total else "",
                "status_counts": "; ".join(f"{key}:{value}" for key, value in sorted(counts.items())),
            }
        )
    write_csv(
        output_dir / "status_by_run_system_gpu.csv",
        ["run_id", "system", "gpu_count", "artifacts", "ok", "success_rate_pct", "status_counts"],
        status_by_run_system_gpu,
    )

    status_by_family = []
    grouped_family: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in manifests:
        grouped_family[(str(row["run_id"]), str(row.get("system", "")), str(row["family"]))][str(row.get("status", "other"))] += 1
    for (run_id, system, family), counts in sorted(grouped_family.items()):
        total = sum(counts.values())
        status_by_family.append(
            {
                "run_id": run_id,
                "system": system,
                "family": family,
                "artifacts": total,
                "ok": counts["ok"],
                "success_rate_pct": round(100 * counts["ok"] / total, 2) if total else "",
                "status_counts": "; ".join(f"{key}:{value}" for key, value in sorted(counts.items())),
            }
        )
    write_csv(
        output_dir / "status_by_family.csv",
        ["run_id", "system", "family", "artifacts", "ok", "success_rate_pct", "status_counts"],
        status_by_family,
    )

    throughput_samples: dict[tuple[str, str, str, str, str, int], list[float]] = defaultdict(list)
    for row in results + supplemental_throughput_results:
        if row.get("status") != "ok":
            continue
        throughput = throughput_per_sec(row)
        if throughput is not None and throughput > 0:
            throughput_samples[
                (
                    str(row.get("run_id")),
                    str(row.get("phase")),
                    str(row.get("system")),
                    str(row.get("case_name")),
                    str(row.get("family")),
                    int(row.get("gpu_count") or 0),
                )
            ].append(throughput)
    throughput_rows = []
    for (run_id, phase, system, case_name, family, gpu_count), values in sorted(throughput_samples.items()):
        throughput_rows.append(
            {
                "run_id": run_id,
                "phase": phase,
                "system": system,
                "case_name": case_name,
                "family": family,
                "gpu_count": gpu_count,
                "samples": len(values),
                "median_circuit_evals_per_sec": round(median(values) or 0.0, 6),
            }
        )
    write_csv(
        output_dir / "phase_d_throughput_medians.csv",
        ["run_id", "phase", "system", "case_name", "family", "gpu_count", "samples", "median_circuit_evals_per_sec"],
        throughput_rows,
    )

    coverage_rows = []
    for system in SYSTEMS:
        for family in CASE_ORDER:
            solved = {
                (str(row.get("case_name")), int(row.get("gpu_count") or 0))
                for row in manifests
                if row.get("system") == system and row.get("status") == "ok" and row.get("family") == family
            }
            coverage_rows.append({"system": system, "family": family, "unique_successful_case_gpu_configs": len(solved)})
    write_csv(output_dir / "successful_coverage_by_system.csv", ["system", "family", "unique_successful_case_gpu_configs"], coverage_rows)

    frontier_rows = []
    for system in SYSTEMS:
        for family in CASE_ORDER:
            dims = [
                dim
                for row in results
                if row.get("system") == system
                and row.get("status") == "ok"
                and row.get("family") == family
                and (dim := effective_dimension_log10(row)) is not None
            ]
            frontier_rows.append(
                {
                    "system": system,
                    "family": family,
                    "max_solved_log10_state_space_dimension": round(max(dims), 6) if dims else "",
                }
            )
    write_csv(output_dir / "solvable_frontier_by_system.csv", ["system", "family", "max_solved_log10_state_space_dimension"], frontier_rows)

    readme = [
        "# Distributed 8xH800 Paper Data Summary",
        "",
        "This directory consolidates the distributed experiment manifests and result JSON files into tables used for paper figures.",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `manifest_summary.csv` | One row per run/manifest, with artifact counts and status counts. |",
        "| `status_by_run_system_gpu.csv` | Success rate and status counts by run, system, and GPU count. |",
        "| `status_by_family.csv` | Success rate and status counts by run, system, and workload family. |",
        "| `phase_d_throughput_medians.csv` | Median throughput for all successful runs with throughput telemetry. |",
        "| `successful_coverage_by_system.csv` | Unique successful case x GPU configurations, grouped by system and family. |",
        "| `solvable_frontier_by_system.csv` | Maximum solved effective state-space dimension, grouped by system and family. |",
        "",
        "Recommended paper use: use coverage and frontier tables for the main HybridCVDV advantage argument; use throughput only as a secondary scaling/diagnostic result.",
        "",
        "BQSim supplemental smoke results from `distributed_smoke_all` are included in throughput summaries when a valid full-result or raw-result record is available. These supplemental rows are not included in coverage/success-rate tables.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("experiments/results/distributed_8xH800/paper_data_summary"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    summarize(root, output_dir)
    print(f"Wrote distributed summary tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
