#!/usr/bin/env python3
"""Run all SC26 paper evaluation experiments.

Experiment groups:
  scaling    – Pure-CV and hybrid scaling reference (mode sweep, cutoff sweep)
  memory     – HDD vs full-tensor memory comparison
  ablation   – Fock hierarchy & Gaussian symbolic ablation study

Usage:
    # Run everything
    python run_sc26_paper_eval.py --results-dir results/ --summary-path summary.json

    # Run only ablation experiments
    python run_sc26_paper_eval.py --results-dir results/ --summary-path summary.json --group ablation

    # Dry-run to see what would be executed
    python run_sc26_paper_eval.py --results-dir results/ --summary-path summary.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = REPO_ROOT / "build-experiments"

# ---------------------------------------------------------------------------
# Scaling reference
# ---------------------------------------------------------------------------
CV_MODE_SWEEP = [2, 3, 4, 5, 6, 7]
HYBRID_NQ = [3, 4, 5, 10]
HYBRID_MODE_SWEEP = [3, 4, 5, 6, 7]
CUTOFF_SWEEP = [4, 8, 12, 16, 24, 32]

SCALING_CASES: list[dict[str, Any]] = [
    # Pure-CV QAOA mode sweep (cutoff=16)
    *[{"case": f"sc26_cv_qaoa_nm{nm}_c16", "flags": []} for nm in CV_MODE_SWEEP],
    # Pure-CV JCH mode sweep (cutoff=16)
    *[{"case": f"sc26_cv_jch_nm{nm}_c16", "flags": []} for nm in CV_MODE_SWEEP],
    # Hybrid JCH (various qubit counts, mode sweep)
    *[
        {"case": f"sc26_jch_nq{nq}_nm{nm}_c16", "flags": []}
        for nq in HYBRID_NQ
        for nm in HYBRID_MODE_SWEEP
    ],
    # Cutoff sweep for Pure-CV at nm=5
    *[{"case": f"sc26_cv_qaoa_nm5_c{c}", "flags": []} for c in CUTOFF_SWEEP],
    *[{"case": f"sc26_cv_jch_nm5_c{c}", "flags": []} for c in CUTOFF_SWEEP],
]

# ---------------------------------------------------------------------------
# Memory: HDD vs full-tensor
# ---------------------------------------------------------------------------
MEMORY_QUBIT_SWEEP = [2, 4, 8, 12, 16, 20]
MEMORY_CASES: list[dict[str, Any]] = [
    {"case": f"hdd_vs_full_tensor_qubits_{nq}", "flags": []}
    for nq in MEMORY_QUBIT_SWEEP
]

# ---------------------------------------------------------------------------
# Ablation study — 3 configurations × 2 workloads × nm 2-5
#
#   full                 : all optimizations enabled (Gaussian symbolic + ELL Fock)
#   no_symbolic          : --disable-gaussian-symbolic  (ELL Fock still active)
#   no_symbolic_dense_fock: --disable-gaussian-symbolic --force-dense-fock
#                           (removes both symbolic track AND sparse Fock hierarchy)
#
# This isolates:
#   (full → no_symbolic)          = Gaussian symbolic track contribution
#   (no_symbolic → no_sym+dense)  = ELL sparse Fock hierarchy contribution
# ---------------------------------------------------------------------------
ABLATION_MODE_SWEEP = [2, 3, 4, 5]
ABLATION_WORKLOADS = [
    "sc26_cv_qaoa_nm{nm}_c16",   # Pure-CV: only Displacement/Squeezing (L2)
    "sc26_jch_nq4_nm{nm}_c16",   # Hybrid: L0-L3 gates + qubit gates
]
ABLATION_CONFIGS: list[dict[str, Any]] = [
    {"variant": "full",                    "flags": []},
    {"variant": "no_symbolic",             "flags": ["--disable-gaussian-symbolic"]},
    {"variant": "no_symbolic_dense_fock",  "flags": ["--disable-gaussian-symbolic", "--force-dense-fock"]},
]


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def format_task_label(task: dict[str, Any]) -> str:
    return f"{task['id']} ({task['case']} / {task['variant']})"


def build_tasks(results_dir: pathlib.Path, groups: set[str] | None = None) -> list[dict[str, Any]]:
    """Build the full task list, optionally filtered by group name."""
    tasks: list[dict[str, Any]] = []
    all_groups = groups is None

    if all_groups or "scaling" in groups:
        for spec in SCALING_CASES:
            tasks.append(
                {
                    "id": f"scaling/{spec['case']}",
                    "group": "scaling",
                    "case": spec["case"],
                    "variant": "full",
                    "flags": list(spec["flags"]),
                    "output_path": str(
                        (results_dir / "scaling" / f"{spec['case']}.json").resolve()
                    ),
                }
            )

    if all_groups or "memory" in groups:
        for spec in MEMORY_CASES:
            tasks.append(
                {
                    "id": f"memory/{spec['case']}",
                    "group": "memory",
                    "case": spec["case"],
                    "variant": "full",
                    "flags": list(spec["flags"]),
                    "output_path": str(
                        (results_dir / "memory" / f"{spec['case']}.json").resolve()
                    ),
                }
            )

    if all_groups or "ablation" in groups:
        for workload_tpl in ABLATION_WORKLOADS:
            for nm in ABLATION_MODE_SWEEP:
                case = workload_tpl.format(nm=nm)
                for cfg in ABLATION_CONFIGS:
                    variant = cfg["variant"]
                    tasks.append(
                        {
                            "id": f"ablation/{case}/{variant}",
                            "group": "ablation",
                            "case": case,
                            "variant": variant,
                            "flags": list(cfg["flags"]),
                            "output_path": str(
                                (results_dir / "ablation_v2" / f"{case}__{variant}.json").resolve()
                            ),
                        }
                    )

    return tasks


def load_or_init_summary(
    summary_path: pathlib.Path,
    args: argparse.Namespace,
    groups: set[str] | None = None,
) -> dict[str, Any]:
    if summary_path.exists():
        summary = read_json(summary_path)
    else:
        summary = {
            "schema_version": "1.1",
            "generated_at_utc": iso_now(),
            "updated_at_utc": iso_now(),
            "build_dir": str(pathlib.Path(args.build_dir).resolve()),
            "results_dir": str(pathlib.Path(args.results_dir).resolve()),
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "gaussian_symbolic_mode_limit": args.gaussian_symbolic_mode_limit,
            "symbolic_branch_limit": args.symbolic_branch_limit,
            "tasks": [],
        }

    existing: dict[str, dict[str, Any]] = {}
    for task in summary.get("tasks", []):
        if isinstance(task, dict) and isinstance(task.get("id"), str):
            existing[task["id"]] = task

    tasks = build_tasks(pathlib.Path(args.results_dir), groups)
    merged: list[dict[str, Any]] = []
    for task in tasks:
        current = existing.get(task["id"], {})
        merged.append({**task, **current})

    summary["updated_at_utc"] = iso_now()
    summary["build_dir"] = str(pathlib.Path(args.build_dir).resolve())
    summary["results_dir"] = str(pathlib.Path(args.results_dir).resolve())
    summary["warmup_runs"] = args.warmup_runs
    summary["measured_runs"] = args.measured_runs
    summary["gaussian_symbolic_mode_limit"] = args.gaussian_symbolic_mode_limit
    summary["symbolic_branch_limit"] = args.symbolic_branch_limit
    summary["tasks"] = merged
    write_json(summary_path, summary)
    return summary


def make_command(
    binary_path: pathlib.Path,
    task: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(binary_path),
        "--suite",
        "scaling",
        "--name-filter",
        str(task["case"]),
        "--gaussian-symbolic-mode-limit",
        str(args.gaussian_symbolic_mode_limit),
        "--symbolic-branch-limit",
        str(args.symbolic_branch_limit),
        "--output",
        str(task["output_path"]),
        *[str(flag) for flag in task.get("flags", [])],
    ]


def extract_named_result(payload: dict[str, Any], case_name: str) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("experiment output is missing a results list")

    exact_match = [entry for entry in results if isinstance(entry, dict) and entry.get("name") == case_name]
    if len(exact_match) == 1:
        return exact_match[0]
    if len(results) == 1 and isinstance(results[0], dict):
        return results[0]
    raise ValueError(f"could not identify a unique result for case '{case_name}'")


def run_task(
    binary_path: pathlib.Path,
    task: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = pathlib.Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = make_command(binary_path, task, args)
    env = os.environ.copy()
    env["HYBRIDCVDV_SCALING_WARMUP_RUNS"] = str(args.warmup_runs)
    env["HYBRIDCVDV_SCALING_MEASURED_RUNS"] = str(args.measured_runs)

    started_at = iso_now()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    task["started_at_utc"] = started_at
    task["finished_at_utc"] = iso_now()
    task["command"] = command
    task["returncode"] = proc.returncode
    task["stdout_tail"] = proc.stdout[-4000:]
    task["stderr_tail"] = proc.stderr[-4000:]

    if proc.returncode != 0:
        task["status"] = "error"
        task["note"] = "experiment command failed"
        return task

    payload = read_json(output_path)
    result = extract_named_result(payload, str(task["case"]))
    task["status"] = str(result.get("status", "unknown"))
    task["note"] = str(result.get("note", ""))
    task["params"] = result.get("params", {})
    task["metrics"] = result.get("metrics", {})
    task["report"] = {
        "requested_suite": payload.get("requested_suite"),
        "gaussian_symbolic_mode_limit": payload.get("gaussian_symbolic_mode_limit"),
        "symbolic_branch_limit": payload.get("symbolic_branch_limit"),
        "gaussian_symbolic_enabled": payload.get("gaussian_symbolic_enabled"),
        "diagonal_mixture_enabled": payload.get("diagonal_mixture_enabled"),
        "fused_diagonal_enabled": payload.get("fused_diagonal_enabled"),
        "eager_symbolic_materialization_enabled": payload.get(
            "eager_symbolic_materialization_enabled"
        ),
        "use_interaction_picture": payload.get("use_interaction_picture"),
        "device": payload.get("device"),
        "num_gpus": payload.get("num_gpus"),
    }
    return task


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SC26 paper evaluation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Run everything
  %(prog)s --results-dir experiments/results --summary-path summary.json

  # Run only ablation experiments
  %(prog)s --results-dir experiments/results --summary-path summary.json --group ablation

  # Dry-run to see all tasks
  %(prog)s --results-dir experiments/results --summary-path summary.json --dry-run
""",
    )
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--gaussian-symbolic-mode-limit", type=int, default=16)
    parser.add_argument("--symbolic-branch-limit", type=int, default=64)
    parser.add_argument(
        "--group",
        action="append",
        choices=["scaling", "memory", "ablation"],
        help="Run only the specified group(s); can be repeated. Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print task list and exit without running.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run tasks even if they already have status=ok.",
    )
    args = parser.parse_args()

    groups: set[str] | None = set(args.group) if args.group else None

    build_dir = pathlib.Path(args.build_dir)
    binary_path = build_dir / "hybridcvdv_single_gpu_experiments"
    if not args.dry_run and not binary_path.exists():
        raise FileNotFoundError(f"benchmark binary not found: {binary_path}")

    summary_path = pathlib.Path(args.summary_path)
    summary = load_or_init_summary(summary_path, args, groups)

    if args.dry_run:
        print(f"{'#':>3}  {'Group':<10} {'Case':<40} {'Variant':<25} Flags")
        print("-" * 110)
        for i, task in enumerate(summary["tasks"], 1):
            flags_str = " ".join(task.get("flags", [])) or "(none)"
            status = task.get("status", "pending")
            print(
                f"{i:>3}  {task['group']:<10} {task['case']:<40} "
                f"{task['variant']:<25} {flags_str}  [{status}]"
            )
        print(f"\nTotal: {len(summary['tasks'])} tasks")
        return 0

    summary["status"] = "running"
    write_json(summary_path, summary)
    print(
        f"[{iso_now()}] starting sc26 eval run with {len(summary['tasks'])} tasks; "
        f"summary={summary_path}",
        flush=True,
    )

    for task in summary["tasks"]:
        if task.get("status") == "ok" and not args.force_rerun:
            continue
        print(f"[{iso_now()}] running {format_task_label(task)}", flush=True)
        updated = run_task(binary_path, task, args)
        task.update(updated)
        summary["updated_at_utc"] = iso_now()
        write_json(summary_path, summary)
        median_total_ms = None
        metrics = task.get("metrics")
        if isinstance(metrics, dict):
            median_total_ms = metrics.get("median_total_ms")
        print(
            f"[{iso_now()}] finished {format_task_label(task)} "
            f"status={task.get('status')} median_total_ms={median_total_ms}",
            flush=True,
        )
        if task.get("status") == "error":
            summary["updated_at_utc"] = iso_now()
            summary["status"] = "error"
            write_json(summary_path, summary)
            return 1

    summary["updated_at_utc"] = iso_now()
    summary["status"] = "ok"
    write_json(summary_path, summary)
    print(f"[{iso_now()}] completed sc26 eval run successfully", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
