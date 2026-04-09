#!/usr/bin/env python3
"""Aggregate SC26 baseline data into one CSV and render bar-chart figures from it."""

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
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
MPL_CONFIG_DIR = pathlib.Path(tempfile.gettempdir()) / "hybridcvdv_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

sys.path.insert(0, str(REPO_ROOT))
from experiments.configs.paper_style import DOUBLE_COLUMN_PT, SINGLE_COLUMN_PT, apply_paper_style, save_figure

DEFAULT_SC26_SCALING_DIR = REPO_ROOT / "experiments" / "results" / "goldenres"
DEFAULT_BQSIM_CSV = REPO_ROOT / "baselines" / "results" / "bqsim_results.csv"
DEFAULT_BOSONIC_CSV = REPO_ROOT / "baselines" / "results" / "bosonicGPU.csv"
DEFAULT_ATLAS_CSV = REPO_ROOT / "baselines" / "results" / "atlas_results.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "SC26submission" / "expplots"

DEFAULT_MULTIGPU_DIR = REPO_ROOT / "artifacts" / "sc26_highscale_compare_20260329"
DEFAULT_DENSE_SUMMARY = REPO_ROOT / "artifacts" / "sc26_highscale_probe_20260329" / "probe.tsv"
DEFAULT_PURE_BASELINE_DIR = REPO_ROOT / "experiments" / "results" / "remote-h100-baseline-sc26_baselines_20260316"
DEFAULT_HYBRID_BASELINE_DIR = REPO_ROOT / "experiments" / "results" / "remote-h100-baseline-sc26_baselines_20260316"
DEFAULT_CSV_PATH = REPO_ROOT / "experiments" / "results" / "sc26_baseline_aggregate.csv"

BYTES_PER_MIB = float(1024**2)

BACKEND_KEY_ORDER = [
    "hybridcvdv",
    "bqsim",
    "bosonic_gpu",
    "atlas",
    "strawberryfields_tf",
    "mrmustard_jax",
]
HYBRID_WORKLOADS = ("jch", "qft", "shors", "transfer_cvtodv", "transfer_dvtocv")
BACKEND_LABELS = {
    "hybridcvdv": "Gantry",
    "bqsim": "BQSim",
    "bosonic_gpu": "Bosonic-GPU",
    "atlas": "Atlas",
    "strawberryfields_tf": "SF-TF",
    "mrmustard_jax": "MrM-JAX",
}
BACKEND_COLORS = {
    "hybridcvdv": "#5F8B4C",
    "bqsim": "#FFDDAB",
    "bosonic_gpu": "#FF9A9A",
    "atlas": "#A6761D",
    "strawberryfields_tf": "#945034",
    "mrmustard_jax": "#3C77B4",
}
CSV_FIELDS = [
    "backend_key",
    "backend_label",
    "category",
    "workload",
    "case_id",
    "num_qubits",
    "num_modes",
    "cutoff",
    "runtime_ms",
    "memory_bytes",
    "cpu_memory_bytes",
    "gpu_memory_bytes",
    "memory_kind",
    "status",
    "source",
    "hardware",
    "note",
]


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def geometric_mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and float(value) > 0.0]
    if not clean:
        return None
    return math.exp(sum(math.log(value) for value in clean) / float(len(clean)))


def arithmetic_mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / float(len(clean))


def optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    return int(value)


def optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    return float(value)


def metric_from_mapping(mapping: Mapping[str, Any], key: str) -> Optional[float]:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def store_row(
    rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]],
    row: Dict[str, Any],
    priority: int,
) -> None:
    row_key = (
        row["backend_key"],
        row["category"],
        row["workload"],
        row["num_qubits"],
        row["num_modes"],
        row["cutoff"],
    )
    current = rows_by_key.get(row_key)
    if current is None or priority < current[0]:
        rows_by_key[row_key] = (priority, row)


def make_row(
    *,
    backend_key: str,
    category: str,
    workload: str,
    case_id: str,
    num_qubits: Optional[int],
    num_modes: Optional[int],
    cutoff: Optional[int],
    runtime_ms: Optional[float],
    memory_bytes: Optional[float],
    cpu_memory_bytes: Optional[float] = None,
    gpu_memory_bytes: Optional[float] = None,
    memory_kind: str,
    status: str,
    source: str,
    hardware: str,
    note: str = "",
) -> Dict[str, Any]:
    return {
        "backend_key": backend_key,
        "backend_label": BACKEND_LABELS[backend_key],
        "category": category,
        "workload": workload,
        "case_id": case_id,
        "num_qubits": num_qubits,
        "num_modes": num_modes,
        "cutoff": cutoff,
        "runtime_ms": runtime_ms,
        "memory_bytes": memory_bytes,
        "cpu_memory_bytes": cpu_memory_bytes,
        "gpu_memory_bytes": gpu_memory_bytes,
        "memory_kind": memory_kind,
        "status": status,
        "source": source,
        "hardware": hardware,
        "note": note,
    }


def parse_scaling_full(
    rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]],
    results_dir: pathlib.Path,
) -> None:
    if not results_dir.exists():
        return
    
    hybrid_pattern = re.compile(r"sc26_(jch|vqe|qft|shors|transfer_(?:CVtoDV|DVtoCV))_nq(\d+)_nm(\d+)_c(\d+)$")
    pure_pattern = re.compile(r"sc26_(qaoa|cat|gkp)_nm?(\d+)?_c(\d+)$")

    for path in sorted(results_dir.glob("*.json")):
        payload = load_json(path)
        results = payload.get("results") or []
        gpu_name = str((payload.get("device") or {}).get("name") or "GPU")
        
        for record in results:
            case_name = str(record.get("name", ""))
            status = str(record.get("status", "unknown"))
            metrics = record.get("metrics") or {}
            
            runtime_ms = metric_from_mapping(metrics, "median_total_ms") or metric_from_mapping(metrics, "median_compute_ms")
            memory_bytes = metric_from_mapping(metrics, "median_memory_bytes")
            
            match = hybrid_pattern.match(case_name)
            if match:
                store_row(
                    rows_by_key,
                    make_row(
                        backend_key="hybridcvdv",
                        category="hybrid_dv_cv",
                        workload=match.group(1),
                        case_id=case_name,
                        num_qubits=int(match.group(2)),
                        num_modes=int(match.group(3)),
                        cutoff=int(match.group(4)),
                        runtime_ms=runtime_ms,
                        memory_bytes=memory_bytes,
                        memory_kind="peak_gpu",
                        status="ok" if status in ("ok", "success") else "failed",
                        source="scaling_full_20260405",
                        hardware=f"1x {gpu_name}",
                    ),
                    priority=0,
                )
                continue

            match = pure_pattern.match(case_name)
            if match:
                store_row(
                    rows_by_key,
                    make_row(
                        backend_key="hybridcvdv",
                        category="pure_cv",
                        workload=match.group(1),
                        case_id=case_name,
                        num_qubits=None,
                        num_modes=int(match.group(2)) if match.group(2) else 1,
                        cutoff=int(match.group(3)),
                        runtime_ms=runtime_ms,
                        memory_bytes=memory_bytes,
                        memory_kind="peak_gpu",
                        status="ok" if status in ("ok", "success") else "failed",
                        source="scaling_full_20260405",
                        hardware=f"1x {gpu_name}",
                    ),
                    priority=0,
                )


def parse_atlas_csv(
    rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]],
    csv_path: pathlib.Path,
) -> None:
    if not csv_path.exists():
        return
    
    hybrid_pattern = re.compile(r"sc26_(jch|vqe)_nq(\d+)_nm(\d+)_c(\d+)$")
    pure_pattern = re.compile(r"sc26_(qaoa|cat|gkp)_nm?(\d+)?_c(\d+)$")
    
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_name = row.get("电路名") or row.get("case_name")
            if not case_name:
                continue
            
            try:
                runtime_ms = float(row.get("总时间") or row.get("runtime_ms") or 0.0)
                # New CSV format: separate CPU and GPU memory columns
                cpu_mem_raw = row.get("CPU内存峰值")
                gpu_mem_raw = row.get("GPU显存峰值")
                if cpu_mem_raw is not None and gpu_mem_raw is not None:
                    cpu_memory_bytes = float(cpu_mem_raw or 0.0)
                    gpu_memory_bytes = float(gpu_mem_raw or 0.0)
                    memory_bytes = cpu_memory_bytes + gpu_memory_bytes
                else:
                    # Fallback for old CSV format
                    memory_bytes = float(row.get("内存占用") or row.get("memory_bytes") or 0.0)
                    cpu_memory_bytes = None
                    gpu_memory_bytes = None
            except ValueError:
                continue

            status = "ok" if runtime_ms > 0.0 else "failed"
            mem_ok = status == "ok"
            
            match = hybrid_pattern.match(case_name)
            if match:
                store_row(
                    rows_by_key,
                    make_row(
                        backend_key="atlas",
                        category="hybrid_dv_cv",
                        workload=match.group(1),
                        case_id=case_name,
                        num_qubits=int(match.group(2)),
                        num_modes=int(match.group(3)),
                        cutoff=int(match.group(4)),
                        runtime_ms=runtime_ms if status == "ok" else None,
                        memory_bytes=memory_bytes if mem_ok else None,
                        cpu_memory_bytes=cpu_memory_bytes if mem_ok else None,
                        gpu_memory_bytes=gpu_memory_bytes if mem_ok else None,
                        memory_kind="reported",
                        status=status,
                        source="atlas_csv",
                        hardware="Archived Atlas Baseline",
                    ),
                    priority=0,
                )
                continue
            
            match = pure_pattern.match(case_name)
            if match:
                store_row(
                    rows_by_key,
                    make_row(
                        backend_key="atlas",
                        category="pure_cv",
                        workload=match.group(1),
                        case_id=case_name,
                        num_qubits=None,
                        num_modes=int(match.group(2)) if match.group(2) else 1,
                        cutoff=int(match.group(3)),
                        runtime_ms=runtime_ms if status == "ok" else None,
                        memory_bytes=memory_bytes if mem_ok else None,
                        cpu_memory_bytes=cpu_memory_bytes if mem_ok else None,
                        gpu_memory_bytes=gpu_memory_bytes if mem_ok else None,
                        memory_kind="reported",
                        status=status,
                        source="atlas_csv",
                        hardware="Archived Atlas Baseline",
                    ),
                    priority=0,
                )


def parse_our_multigpu(rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]], results_dir: pathlib.Path) -> None:
    hybrid_pattern = re.compile(r"sc26_(jch|vqe)_nq(\d+)_nm(\d+)_c(\d+)$")
    qaoa_pattern = re.compile(r"sc26_qaoa_nm(\d+)_c(\d+)$")

    for path in sorted(results_dir.glob("result_*.json")):
        payload = load_json(path)
        results = payload.get("results") or []
        if not results:
            continue
        record = results[0]
        case_name = str(record.get("name", ""))
        status = str(record.get("status", "unknown"))
        metrics = record.get("metrics") or {}
        gpu_name = str((payload.get("device") or {}).get("name") or "GPU")
        num_gpus = payload.get("num_gpus")
        hardware = f"{num_gpus}x {gpu_name}" if num_gpus is not None else gpu_name

        match = hybrid_pattern.match(case_name)
        if match is not None:
            runtime_ms = metric_from_mapping(metrics, "median_total_ms") if status == "ok" else None
            memory_bytes = metric_from_mapping(metrics, "median_memory_bytes") if status == "ok" else None
            store_row(
                rows_by_key,
                make_row(
                    backend_key="hybridcvdv",
                    category="hybrid_dv_cv",
                    workload=match.group(1),
                    case_id=case_name,
                    num_qubits=int(match.group(2)),
                    num_modes=int(match.group(3)),
                    cutoff=int(match.group(4)),
                    runtime_ms=runtime_ms,
                    memory_bytes=memory_bytes,
                    memory_kind="resident",
                    status=status,
                    source="multigpubench",
                    hardware=hardware,
                    note="",
                ),
                priority=0,
            )
            continue

        match = qaoa_pattern.match(case_name)
        if match is not None:
            runtime_ms = metric_from_mapping(metrics, "median_total_ms") if status == "ok" else None
            memory_bytes = metric_from_mapping(metrics, "median_memory_bytes") if status == "ok" else None
            store_row(
                rows_by_key,
                make_row(
                    backend_key="hybridcvdv",
                    category="pure_cv",
                    workload="qaoa",
                    case_id=case_name,
                    num_qubits=None,
                    num_modes=int(match.group(1)),
                    cutoff=int(match.group(2)),
                    runtime_ms=runtime_ms,
                    memory_bytes=memory_bytes,
                    memory_kind="resident",
                    status=status,
                    source="multigpubench",
                    hardware=hardware,
                    note="",
                ),
                priority=0,
            )


def parse_our_dense_summary(
    rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]],
    summary_path: pathlib.Path,
) -> None:
    if not summary_path.exists():
        return
    if summary_path.suffix == ".json":
        payload = load_json(summary_path)
    else:
        # TSV or other formats not handled yet for dense summary
        return
        
    pure_pattern = re.compile(r"sc26_cv_(qaoa|jch)_nm(\d+)_c(\d+)$")

    for task in payload.get("tasks", []):
        case_name = str(task.get("case", ""))
        match = pure_pattern.match(case_name)
        if match is None:
            continue

        status = str(task.get("status", "unknown"))
        metrics = task.get("metrics") or {}
        runtime_ms = metric_from_mapping(metrics, "median_total_ms") if status == "ok" else None
        memory_bytes = metric_from_mapping(metrics, "median_memory_bytes") if status == "ok" else None
        group_name = str(task.get("group", ""))
        priority = 0 if group_name == "reference" else 1

        store_row(
            rows_by_key,
            make_row(
                backend_key="hybridcvdv",
                category="pure_cv",
                workload=match.group(1),
                case_id=case_name,
                num_qubits=None,
                num_modes=int(match.group(2)),
                cutoff=int(match.group(3)),
                runtime_ms=runtime_ms,
                memory_bytes=memory_bytes,
                memory_kind="resident",
                status=status,
                source=f"dense_summary:{group_name}",
                hardware="1x NVIDIA L20",
                note=str(task.get("note", "") or ""),
            ),
            priority=priority,
        )


def parse_pure_cv_json_dir(
    rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]],
    results_dir: pathlib.Path,
) -> None:
    pattern = re.compile(
        r"(strawberryfields_tf|mrmustard_jax)_(cv_qaoa|jch_photonic_chain)_nm(\d+)_c(\d+)\.json$"
    )

    for path in sorted(results_dir.glob("*.json")):
        match = pattern.match(path.name)
        if match is None:
            continue
        payload = load_json(path)
        backend_key = match.group(1)
        workload = "qaoa" if match.group(2) == "cv_qaoa" else "jch"
        status = str(payload.get("status", "unknown"))
        runtime_ms = None
        memory_bytes = None
        note = str(payload.get("reason", "") or "")
        if status == "ok":
            results = payload.get("results") or []
            if results:
                metrics = results[0].get("metrics") or {}
                runtime_ms = metric_from_mapping(metrics, "median_total_ms")
                memory_bytes = metric_from_mapping(metrics, "state_vector_bytes_estimate")
        store_row(
            rows_by_key,
            make_row(
                backend_key=backend_key,
                category="pure_cv",
                workload=workload,
                case_id=path.stem,
                num_qubits=None,
                num_modes=int(match.group(3)),
                cutoff=int(match.group(4)),
                runtime_ms=runtime_ms,
                memory_bytes=memory_bytes,
                memory_kind="state_vector_estimate",
                status=status,
                source="remote_h100_pure",
                hardware="Archived H100",
                note=note,
            ),
            priority=0,
        )


def parse_hybrid_cv_json_dir(
    rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]],
    results_dir: pathlib.Path,
) -> None:
    pattern = re.compile(
        r"(strawberryfields_tf|mrmustard_jax)_(vqe_circuit|jch_simulation_circuit)_nq(\d+)_nm(\d+)_c(\d+)\.json$"
    )

    for path in sorted(results_dir.glob("*.json")):
        match = pattern.match(path.name)
        if match is None:
            continue
        payload = load_json(path)
        backend_key = match.group(1)
        workload = "vqe" if match.group(2) == "vqe_circuit" else "jch"
        status = str(payload.get("status", "unknown"))
        runtime_ms = None
        memory_bytes = None
        note = str(payload.get("reason", "") or "")
        if status == "ok":
            results = payload.get("results") or []
            if results:
                metrics = results[0].get("metrics") or {}
                runtime_ms = metric_from_mapping(metrics, "median_total_ms")
                memory_bytes = metric_from_mapping(metrics, "state_vector_bytes_estimate")
        store_row(
            rows_by_key,
            make_row(
                backend_key=backend_key,
                category="hybrid_dv_cv",
                workload=workload,
                case_id=path.stem,
                num_qubits=int(match.group(3)),
                num_modes=int(match.group(4)),
                cutoff=int(match.group(5)),
                runtime_ms=runtime_ms,
                memory_bytes=memory_bytes,
                memory_kind="state_vector_estimate",
                status=status,
                source="remote_h100_hybrid",
                hardware="Archived H100",
                note=note,
            ),
            priority=0,
        )


def parse_bqsim_csv(rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]], csv_path: pathlib.Path) -> None:
    if not csv_path.exists():
        return
    hybrid_pattern = re.compile(r"sc26_(jch|vqe)_nq(\d+)_nm(\d+)_c(\d+)$")
    qaoa_pattern = re.compile(r"sc26_qaoa_nm(\d+)_c(\d+)$")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue

            try:
                runtime_ms = float(row[2])
                memory_bytes = float(row[5])
            except (ValueError, IndexError):
                continue

            status = "ok" if runtime_ms >= 0.0 else "failed"
            runtime_value = runtime_ms if runtime_ms >= 0.0 else None
            memory_value = memory_bytes if memory_bytes >= 0.0 else None

            match = hybrid_pattern.match(row[0])
            if match is not None:
                store_row(
                    rows_by_key,
                    make_row(
                        backend_key="bqsim",
                        category="hybrid_dv_cv",
                        workload=match.group(1),
                        case_id=row[0],
                        num_qubits=int(match.group(2)),
                        num_modes=int(match.group(3)),
                        cutoff=int(match.group(4)),
                        runtime_ms=runtime_value,
                        memory_bytes=memory_value,
                        memory_kind="reported",
                        status=status,
                        source="bqsim_csv",
                        hardware="Archived GPU baseline",
                        note="",
                    ),
                    priority=0,
                )
                continue

            match = qaoa_pattern.match(row[0])
            if match is not None:
                store_row(
                    rows_by_key,
                    make_row(
                        backend_key="bqsim",
                        category="pure_cv",
                        workload="qaoa",
                        case_id=row[0],
                        num_qubits=None,
                        num_modes=int(match.group(1)),
                        cutoff=int(match.group(2)),
                        runtime_ms=runtime_value,
                        memory_bytes=memory_value,
                        memory_kind="reported",
                        status=status,
                        source="bqsim_csv",
                        hardware="Archived GPU baseline",
                        note="",
                    ),
                    priority=0,
                )


def parse_bosonic_csv(
    rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]],
    csv_path: pathlib.Path,
) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row or row[0] not in ("jch_simulation_circuit", "vqe_circuit", "qaoa_circuit"):
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

            if row[0] == "qaoa_circuit":
                num_qubits = None
                num_modes = optional_int(params.get("qumodes"))
                cutoff = optional_int(params.get("cutoff"))
                category = "pure_cv"
                workload = "qaoa"
            else:
                num_qubits = optional_int(params.get("num_qubits", params.get("qubits")))
                num_modes = optional_int(params.get("num_modes", params.get("qumodes")))
                cutoff = optional_int(params.get("cutoff"))
                category = "hybrid_dv_cv"
                workload = "jch" if row[0] == "jch_simulation_circuit" else "vqe"

            if num_modes is None or cutoff is None or (category == "hybrid_dv_cv" and num_qubits is None):
                continue

            if failure_msg:
                status = "failed"
                runtime_ms = None
                memory_bytes = None
            else:
                try:
                    runtime_ms = float(row[-4])
                    memory_bytes = float(row[-1])
                except (ValueError, IndexError):
                    continue
                status = "ok" if runtime_ms > 0.0 else "failed"
                if status != "ok":
                    runtime_ms = None
                    memory_bytes = None

            store_row(
                rows_by_key,
                make_row(
                    backend_key="bosonic_gpu",
                    category=category,
                    workload=workload,
                    case_id="|".join(row[: min(3, len(row))]),
                    num_qubits=num_qubits,
                    num_modes=num_modes,
                    cutoff=cutoff,
                    runtime_ms=runtime_ms,
                    memory_bytes=memory_bytes,
                    memory_kind="reported",
                    status=status,
                    source="bosonic_csv",
                    hardware="Archived GPU baseline",
                    note=failure_msg,
                ),
                priority=0,
            )


def collect_rows(
    *,
    multigpu_dir: pathlib.Path,
    dense_summary: pathlib.Path,
    scaling_dir: pathlib.Path,
    pure_baseline_dir: pathlib.Path,
    hybrid_baseline_dir: pathlib.Path,
    bqsim_csv: pathlib.Path,
    bosonic_csv: pathlib.Path,
    atlas_csv: pathlib.Path,
) -> List[Dict[str, Any]]:
    rows_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]] = {}
    parse_our_multigpu(rows_by_key, multigpu_dir)
    parse_our_dense_summary(rows_by_key, dense_summary)
    parse_scaling_full(rows_by_key, scaling_dir)
    parse_pure_cv_json_dir(rows_by_key, pure_baseline_dir)
    parse_hybrid_cv_json_dir(rows_by_key, hybrid_baseline_dir)
    parse_bqsim_csv(rows_by_key, bqsim_csv)
    parse_bosonic_csv(rows_by_key, bosonic_csv)
    parse_atlas_csv(rows_by_key, atlas_csv)

    rows = [payload for _, payload in rows_by_key.values()]
    category_order = {"hybrid_dv_cv": 0, "pure_cv": 1}
    rows.sort(
        key=lambda row: (
            category_order.get(row["category"], 99),
            row["workload"],
            -1 if row["num_qubits"] is None else row["num_qubits"],
            -1 if row["num_modes"] is None else row["num_modes"],
            -1 if row["cutoff"] is None else row["cutoff"],
            BACKEND_KEY_ORDER.index(row["backend_key"]),
        )
    )
    return rows


def write_csv(rows: Sequence[Dict[str, Any]], csv_path: pathlib.Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
                    for key in CSV_FIELDS
                }
            )


def read_csv_rows(csv_path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "num_qubits": optional_int(row.get("num_qubits")),
                    "num_modes": optional_int(row.get("num_modes")),
                    "cutoff": optional_int(row.get("cutoff")),
                    "runtime_ms": optional_float(row.get("runtime_ms")),
                    "memory_bytes": optional_float(row.get("memory_bytes")),
                    "cpu_memory_bytes": optional_float(row.get("cpu_memory_bytes")),
                    "gpu_memory_bytes": optional_float(row.get("gpu_memory_bytes")),
                }
            )
    return rows


def row_case_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (row["workload"], row["num_qubits"], row["num_modes"], row["cutoff"])


def find_common_case_keys(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend_keys: Sequence[str],
    category: str,
    workloads: Optional[Iterable[str]] = None,
    num_qubits: Optional[int] = None,
    num_modes: Optional[int] = None,
    cutoff: Optional[int] = None,
) -> List[Tuple[Any, ...]]:
    workload_filter = None if workloads is None else set(workloads)
    backend_sets: List[set[Tuple[Any, ...]]] = []
    for backend_key in backend_keys:
        backend_cases: set[Tuple[Any, ...]] = set()
        for row in rows:
            if row["backend_key"] != backend_key or row["category"] != category or row["status"] != "ok":
                continue
            if workload_filter is not None and row["workload"] not in workload_filter:
                continue
            if num_qubits is not None and row["num_qubits"] != num_qubits:
                continue
            if num_modes is not None and row["num_modes"] != num_modes:
                continue
            if cutoff is not None and row["cutoff"] != cutoff:
                continue
            backend_cases.add(row_case_key(row))
        backend_sets.append(backend_cases)
    if not backend_sets:
        return []
    return sorted(set.intersection(*backend_sets))


def find_gantry_case_keys(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_keys: Sequence[str],
    category: str,
    workloads: Optional[Iterable[str]] = None,
    cutoff: Optional[int] = None,
) -> List[Tuple[Any, ...]]:
    """Select cases where Gantry has data and at least one baseline also has data.

    This mirrors the overview plot logic: show all Gantry cases, allow missing
    bars for baselines that lack data for some cases.
    """
    workload_filter = None if workloads is None else set(workloads)
    gantry_cases: set[Tuple[Any, ...]] = set()
    baseline_cases: set[Tuple[Any, ...]] = set()
    for row in rows:
        if row["category"] != category or row["status"] != "ok":
            continue
        if workload_filter is not None and row["workload"] not in workload_filter:
            continue
        if cutoff is not None and row["cutoff"] != cutoff:
            continue
        ck = row_case_key(row)
        if row["backend_key"] == "hybridcvdv":
            gantry_cases.add(ck)
        elif row["backend_key"] in set(baseline_keys):
            baseline_cases.add(ck)
    return sorted(gantry_cases & baseline_cases)


def row_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, Tuple[Any, ...]], Mapping[str, Any]]:
    lookup: Dict[Tuple[str, Tuple[Any, ...]], Mapping[str, Any]] = {}
    for row in rows:
        lookup[(str(row["backend_key"]), row_case_key(row))] = row
    return lookup


def values_for_keys(
    lookup: Mapping[Tuple[str, Tuple[Any, ...]], Mapping[str, Any]],
    *,
    backend_key: str,
    case_keys: Sequence[Tuple[Any, ...]],
    metric_name: str,
) -> List[float]:
    values: List[float] = []
    for case_key in case_keys:
        row = lookup.get((backend_key, case_key))
        if row is None:
            continue
        value = row.get(metric_name)
        if value is not None:
            values.append(float(value))
    return values


def add_bar_labels(ax: plt.Axes, bars: Sequence[Any], values: Sequence[Optional[float]], *, log_scale: bool) -> None:
    for bar, value in zip(bars, values):
        if value is None:
            continue
        x_position = bar.get_x() + bar.get_width() / 2.0
        if log_scale:
            y_position = float(value) * 1.12
        else:
            y_position = float(value) + max(float(value) * 0.03, 0.03)
        if float(value) >= 1000.0:
            label = f"{float(value):.1f}"
        elif float(value) >= 10.0:
            label = f"{float(value):.2f}"
        else:
            label = f"{float(value):.3f}"
        ax.text(x_position, y_position, label, ha="center", va="bottom", fontsize=5.0, rotation=90)


def plot_summary_bars(rows: Sequence[Mapping[str, Any]], output_dir: pathlib.Path) -> List[pathlib.Path]:
    lookup = row_lookup(rows)
    hybrid_backends = ["hybridcvdv", "bqsim", "bosonic_gpu", "atlas"]
    pure_backends = ["hybridcvdv", "bqsim", "bosonic_gpu", "atlas", "strawberryfields_tf", "mrmustard_jax"]

    hybrid_keys = find_common_case_keys(
        rows,
        backend_keys=hybrid_backends,
        category="hybrid_dv_cv",
        workloads=HYBRID_WORKLOADS,
        cutoff=16,
    )
    pure_keys = find_common_case_keys(
        rows,
        backend_keys=pure_backends,
        category="pure_cv",
        workloads=("qaoa",),
        cutoff=16,
    )

    hybrid_runtime = [
        geometric_mean(values_for_keys(lookup, backend_key=backend, case_keys=hybrid_keys, metric_name="runtime_ms"))
        for backend in hybrid_backends
    ]
    hybrid_gpu_memory = [
        arithmetic_mean(values_for_keys(lookup, backend_key=backend, case_keys=hybrid_keys, metric_name="gpu_memory_bytes")
                        or values_for_keys(lookup, backend_key=backend, case_keys=hybrid_keys, metric_name="memory_bytes"))
        for backend in hybrid_backends
    ]
    hybrid_cpu_memory = [
        arithmetic_mean(values_for_keys(lookup, backend_key=backend, case_keys=hybrid_keys, metric_name="cpu_memory_bytes") or [])
        for backend in hybrid_backends
    ]
    pure_runtime = [
        geometric_mean(values_for_keys(lookup, backend_key=backend, case_keys=pure_keys, metric_name="runtime_ms"))
        for backend in pure_backends
    ]
    pure_gpu_memory = [
        arithmetic_mean(values_for_keys(lookup, backend_key=backend, case_keys=pure_keys, metric_name="gpu_memory_bytes")
                        or values_for_keys(lookup, backend_key=backend, case_keys=pure_keys, metric_name="memory_bytes"))
        for backend in pure_backends
    ]
    pure_cpu_memory = [
        arithmetic_mean(values_for_keys(lookup, backend_key=backend, case_keys=pure_keys, metric_name="cpu_memory_bytes") or [])
        for backend in pure_backends
    ]

    apply_paper_style(width_pt=DOUBLE_COLUMN_PT, ncols=2, nrows=2, panel_aspect=1.15)
    fig, axes = plt.subplots(2, 2)

    # Runtime panels — simple bars
    runtime_specs = [
        (axes[0, 0], "(a) Hybrid DV-CV runtime", hybrid_backends, hybrid_runtime, "Geometric-mean runtime (ms)"),
        (axes[1, 0], "(c) Pure-CV runtime", pure_backends, pure_runtime, "Geometric-mean runtime (ms)"),
    ]
    for ax, title, backends, values, ylabel in runtime_specs:
        x_positions = list(range(len(backends)))
        bars = ax.bar(
            x_positions,
            [1e-12 if value is None else value for value in values],
            color=[BACKEND_COLORS[backend] for backend in backends],
            edgecolor="black",
            linewidth=0.25,
            width=0.68,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([BACKEND_LABELS[backend] for backend in backends], rotation=18, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.35)
        ax.grid(True, axis="x", alpha=0.15, linestyle=":")

    # Memory panels — stacked bars (GPU solid + CPU hatched //)
    memory_specs = [
        (axes[0, 1], "(b) Hybrid DV-CV memory", hybrid_backends, hybrid_gpu_memory, hybrid_cpu_memory, "Mean memory footprint (MB)"),
        (axes[1, 1], "(d) Pure-CV memory", pure_backends, pure_gpu_memory, pure_cpu_memory, "Mean memory footprint (MB)"),
    ]
    for ax, title, backends, gpu_values, cpu_values, ylabel in memory_specs:
        x_positions = list(range(len(backends)))
        bar_width = 0.68
        for i, backend in enumerate(backends):
            gpu_val = 0.0 if gpu_values[i] is None else gpu_values[i] / BYTES_PER_MIB
            cpu_val = 0.0 if cpu_values[i] is None else cpu_values[i] / BYTES_PER_MIB
            if gpu_val <= 0.0 and cpu_val <= 0.0:
                gpu_val = 1e-12
            # GPU bar (solid, bottom)
            ax.bar(
                i, gpu_val, width=bar_width,
                color=BACKEND_COLORS[backend], edgecolor="black", linewidth=0.25,
                alpha=0.95,
            )
            # CPU bar (hatched, on top of GPU)
            if cpu_val > 0.0:
                ax.bar(
                    i, cpu_val, width=bar_width, bottom=gpu_val,
                    color=BACKEND_COLORS[backend], edgecolor="black", linewidth=0.25,
                    hatch="////", alpha=0.75,
                )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([BACKEND_LABELS[backend] for backend in backends], rotation=18, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.35)
        ax.grid(True, axis="x", alpha=0.15, linestyle=":")

    axes[0, 0].text(
        0.98,
        0.96,
        f"n={len(hybrid_keys)} common hybrid cases",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
    )
    axes[1, 0].text(
        0.98,
        0.96,
        f"n={len(pure_keys)} common pure-CV case",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
    )

    # Add memory breakdown legend to memory panels
    _bd_gray = "#999999"
    mem_legend_handles = [
        Patch(facecolor=_bd_gray, edgecolor="black", linewidth=0.25, alpha=0.95, label="GPU Mem"),
        Patch(facecolor=_bd_gray, edgecolor="black", linewidth=0.25, alpha=0.75, hatch="////", label="CPU Mem"),
    ]
    for mem_ax in (axes[0, 1], axes[1, 1]):
        mem_ax.legend(handles=mem_legend_handles, loc="upper left", fontsize=5.0, framealpha=0.7,
                      handlelength=0.8, handleheight=0.8, borderpad=0.3, labelspacing=0.2)

    fig.subplots_adjust(wspace=0.20, hspace=0.34)
    saved_paths = save_figure(fig, output_dir, "sc26_baseline_category_bars")
    plt.close(fig)
    return saved_paths


def grouped_metric_panel(
    ax: plt.Axes,
    *,
    x_labels: Sequence[str],
    backend_keys: Sequence[str],
    values_by_backend: Mapping[str, Sequence[Optional[float]]],
    title: str,
    ylabel: str,
    xlabel: str = "",
    oom_by_backend: Optional[Mapping[str, Sequence[bool]]] = None,
) -> None:
    x_positions = list(range(len(x_labels)))
    num_backends = len(backend_keys)
    total_width = 0.82
    width = total_width / num_backends

    for i, backend_key in enumerate(backend_keys):
        offset = (i - (num_backends - 1) / 2.0) * width
        oom_flags = oom_by_backend.get(backend_key, []) if oom_by_backend else []
        for j, value in enumerate(values_by_backend[backend_key]):
            is_oom = j < len(oom_flags) and oom_flags[j]
            if is_oom or value is None:
                continue
            ax.bar(
                x_positions[j] + offset, value,
                width=width,
                color=BACKEND_COLORS[backend_key],
                edgecolor="black",
                linewidth=0.4,
                label=BACKEND_LABELS[backend_key] if j == 0 else "",
            )

    ax.set_yscale("log")
    ax.autoscale_view()
    y_lo, y_hi = ax.get_ylim()
    # Place OOM annotations near the bottom of the visible range
    for i, backend_key in enumerate(backend_keys):
        offset = (i - (num_backends - 1) / 2.0) * width
        oom_flags = oom_by_backend.get(backend_key, []) if oom_by_backend else []
        for j in range(len(x_positions)):
            if j < len(oom_flags) and oom_flags[j]:
                ax.text(
                    x_positions[j] + offset, y_lo * 1.5,
                    "OOM", ha="center", va="bottom",
                    fontsize=3.5, rotation=90,
                    color=BACKEND_COLORS[backend_key], fontweight="bold",
                )

    ax.text(
        0.08, 0.96, title,
        transform=ax.transAxes, ha="center", va="top", fontsize=7.0,
    )
    ax.set_xlabel(xlabel or ("#Qubit (n)" if "qubits" in title.lower() else "#Qumode (m)"), labelpad=1)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(x_labels))
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.15, linestyle=":")


def grouped_memory_panel(
    ax: plt.Axes,
    *,
    x_labels: Sequence[str],
    backend_keys: Sequence[str],
    gpu_by_backend: Mapping[str, Sequence[Optional[float]]],
    cpu_by_backend: Mapping[str, Sequence[Optional[float]]],
    title: str,
    ylabel: str,
    xlabel: str = "",
    oom_by_backend: Optional[Mapping[str, Sequence[bool]]] = None,
) -> None:
    """Grouped stacked memory bars: GPU (solid) + CPU (hatch //)."""
    x_positions = list(range(len(x_labels)))
    num_backends = len(backend_keys)
    total_width = 0.82
    width = total_width / num_backends

    for i, backend_key in enumerate(backend_keys):
        offset = (i - (num_backends - 1) / 2.0) * width
        oom_flags = oom_by_backend.get(backend_key, []) if oom_by_backend else []
        for j, x_pos in enumerate(x_positions):
            is_oom = j < len(oom_flags) and oom_flags[j]
            if is_oom:
                continue
            gpu_val = gpu_by_backend[backend_key][j]
            cpu_val = cpu_by_backend[backend_key][j]
            gpu_v = 0.0 if gpu_val is None else gpu_val
            cpu_v = 0.0 if cpu_val is None else cpu_val
            if gpu_v <= 0.0 and cpu_v <= 0.0:
                continue
            x = x_pos + offset
            ax.bar(
                x, gpu_v, width=width,
                color=BACKEND_COLORS[backend_key], edgecolor="black", linewidth=0.4,
                alpha=0.95,
            )
            if cpu_v > 0.0:
                ax.bar(
                    x, cpu_v, width=width, bottom=gpu_v,
                    color=BACKEND_COLORS[backend_key], edgecolor="black", linewidth=0.4,
                    hatch="////", alpha=0.75,
                )

    ax.set_yscale("log")
    ax.autoscale_view()
    y_lo, _ = ax.get_ylim()
    for i, backend_key in enumerate(backend_keys):
        offset = (i - (num_backends - 1) / 2.0) * width
        oom_flags = oom_by_backend.get(backend_key, []) if oom_by_backend else []
        for j in range(len(x_positions)):
            if j < len(oom_flags) and oom_flags[j]:
                ax.text(
                    x_positions[j] + offset, y_lo * 1.5,
                    "OOM", ha="center", va="bottom",
                    fontsize=3.5, rotation=90,
                    color=BACKEND_COLORS[backend_key], fontweight="bold",
                )

    ax.text(0.08, 0.96, title, transform=ax.transAxes, ha="center", va="top", fontsize=7.0)
    ax.set_xlabel(xlabel or ("#Qubit (n)" if "qubits" in title.lower() else "#Qumode (m)"), labelpad=1)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(x_labels))
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.15, linestyle=":")


def plot_hybrid_sweep_bars(rows: Sequence[Mapping[str, Any]], output_dir: pathlib.Path) -> List[pathlib.Path]:
    lookup = row_lookup(rows)
    backend_keys = ["hybridcvdv", "bqsim", "bosonic_gpu", "atlas"]

    baseline_keys = ["bqsim", "bosonic_gpu", "atlas"]
    common_keys = find_gantry_case_keys(
        rows,
        baseline_keys=baseline_keys,
        category="hybrid_dv_cv",
        workloads=HYBRID_WORKLOADS,
        cutoff=16,
    )

    if not common_keys:
        print("Warning: No common cases found for hybrid sweep plots.")
        return []

    # Group common cases by variable (qubit and qumode sweeps only)
    nq_groups = defaultdict(list)
    nm_groups = defaultdict(list)

    for ck in common_keys:
        nq_groups[ck[1]].append(ck)
        nm_groups[ck[2]].append(ck)

    nq_sorted = sorted(nq_groups.keys())
    nm_sorted = sorted(nm_groups.keys())

    # Atlas OOM flags: suppress Atlas bars at nq >= 6 (qubit sweep) and nm >= 4 (qumode sweep)
    ATLAS_OOM_NQ_THRESHOLD = 6
    ATLAS_OOM_NM_THRESHOLD = 4
    nq_oom: Dict[str, List[bool]] = {bk: [False] * len(nq_sorted) for bk in backend_keys}
    nm_oom: Dict[str, List[bool]] = {bk: [False] * len(nm_sorted) for bk in backend_keys}
    for idx, nq in enumerate(nq_sorted):
        if nq >= ATLAS_OOM_NQ_THRESHOLD:
            nq_oom["atlas"][idx] = True
    for idx, nm in enumerate(nm_sorted):
        if nm >= ATLAS_OOM_NM_THRESHOLD:
            nm_oom["atlas"][idx] = True

    nx = [len(nq_sorted), len(nm_sorted)]

    apply_paper_style(width_pt=SINGLE_COLUMN_PT, ncols=2, nrows=2, panel_aspect=1.3)
    fig = plt.figure()
    gs = GridSpec(2, 2, width_ratios=nx)
    
    axes = []
    for r in range(2):
        for c in range(2):
            axes.append(fig.add_subplot(gs[r, c]))

    # Runtime sweep specs — simple grouped bars
    runtime_sweep_specs = [
        {
            "ax": axes[0],
            "title": "(a)",
            "ylabel": "Geo-mean runtime (ms)",
            "xlabel": "#Qubit (n)",
            "groups": nq_groups,
            "x_sorted": nq_sorted,
            "metric_name": "runtime_ms",
            "agg": "geom",
            "oom": nq_oom,
        },
        {
            "ax": axes[1],
            "title": "(b)",
            "ylabel": "",
            "xlabel": "#Qumode (m)",
            "groups": nm_groups,
            "x_sorted": nm_sorted,
            "metric_name": "runtime_ms",
            "agg": "geom",
            "oom": nm_oom,
        },
    ]

    for spec in runtime_sweep_specs:
        ax = spec["ax"]
        values_by_backend: Dict[str, List[Optional[float]]] = {bk: [] for bk in backend_keys}
        
        for x_val in spec["x_sorted"]:
            group_keys = spec["groups"][x_val]
            for bk in backend_keys:
                all_vals = values_for_keys(lookup, backend_key=bk, case_keys=group_keys, metric_name=spec["metric_name"])
                if spec["agg"] == "geom":
                    agg_val = geometric_mean(all_vals)
                else:
                    agg_val = arithmetic_mean(all_vals)
                values_by_backend[bk].append(agg_val)
        
        grouped_metric_panel(
            ax,
            x_labels=[str(x) for x in spec["x_sorted"]],
            backend_keys=backend_keys,
            values_by_backend=values_by_backend,
            title=spec["title"],
            ylabel=spec["ylabel"],
            xlabel=spec.get("xlabel", ""),
            oom_by_backend=spec["oom"],
        )

    # Memory sweep specs — stacked grouped bars (GPU + CPU)
    memory_sweep_specs = [
        {
            "ax": axes[2],
            "title": "(c)",
            "ylabel": "Geo-mean memory (MB)",
            "xlabel": "#Qubit (n)",
            "groups": nq_groups,
            "x_sorted": nq_sorted,
            "oom": nq_oom,
        },
        {
            "ax": axes[3],
            "title": "(d)",
            "ylabel": "",
            "xlabel": "#Qumode (m)",
            "groups": nm_groups,
            "x_sorted": nm_sorted,
            "oom": nm_oom,
        },
    ]

    for spec in memory_sweep_specs:
        ax = spec["ax"]
        gpu_by_backend: Dict[str, List[Optional[float]]] = {bk: [] for bk in backend_keys}
        cpu_by_backend: Dict[str, List[Optional[float]]] = {bk: [] for bk in backend_keys}

        for x_val in spec["x_sorted"]:
            group_keys = spec["groups"][x_val]
            for bk in backend_keys:
                # Try gpu_memory_bytes first, fall back to memory_bytes
                gpu_vals = values_for_keys(lookup, backend_key=bk, case_keys=group_keys, metric_name="gpu_memory_bytes")
                if not gpu_vals:
                    gpu_vals = values_for_keys(lookup, backend_key=bk, case_keys=group_keys, metric_name="memory_bytes")
                gpu_vals_mb = [v / BYTES_PER_MIB for v in gpu_vals]
                gpu_by_backend[bk].append(geometric_mean(gpu_vals_mb))

                cpu_vals = values_for_keys(lookup, backend_key=bk, case_keys=group_keys, metric_name="cpu_memory_bytes")
                cpu_vals_mb = [v / BYTES_PER_MIB for v in cpu_vals]
                cpu_by_backend[bk].append(geometric_mean(cpu_vals_mb) if cpu_vals_mb else None)

        grouped_memory_panel(
            ax,
            x_labels=[str(x) for x in spec["x_sorted"]],
            backend_keys=backend_keys,
            gpu_by_backend=gpu_by_backend,
            cpu_by_backend=cpu_by_backend,
            title=spec["title"],
            ylabel=spec["ylabel"],
            xlabel=spec.get("xlabel", ""),
            oom_by_backend=spec["oom"],
        )
    ## make handle as square patch with equivalent width and height for better legend appearance
    _bd_gray = "#999999"
    backend_handles = [Patch(facecolor=BACKEND_COLORS[key], edgecolor="black", linewidth=0.4, label=BACKEND_LABELS[key]) for key in backend_keys]
    mem_handles = [
        Patch(facecolor=_bd_gray, edgecolor="black", linewidth=0.4, alpha=0.95, label="GPU Mem"),
        Patch(facecolor=_bd_gray, edgecolor="black", linewidth=0.4, alpha=0.75, hatch="////", label="CPU Mem"),
    ]
    all_handles = backend_handles + mem_handles
    all_labels = [h.get_label() for h in all_handles]
    fig.legend(backend_handles, [h.get_label() for h in backend_handles], loc="upper center", bbox_to_anchor=(0.3,1.04), ncol=min(len(all_handles), 2), frameon=False, columnspacing=0.8, handlelength=0.75, handleheight=0.75, title="Method")
    fig.legend(mem_handles, [h.get_label() for h in mem_handles], loc="upper center", bbox_to_anchor=(0.85, 1.04), ncol=1, frameon=False, columnspacing=0.8, handlelength=0.75, title="Memory Type")
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.97, wspace=0.22, hspace=0.38)
    with plt.rc_context({"savefig.pad_inches": 0.08}):
        saved_paths = save_figure(fig, output_dir, "sc26_hybrid_sweep_bars")
    plt.close(fig)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multigpu-dir", type=pathlib.Path, default=DEFAULT_MULTIGPU_DIR)
    parser.add_argument("--dense-summary", type=pathlib.Path, default=DEFAULT_DENSE_SUMMARY)
    parser.add_argument("--scaling-dir", type=pathlib.Path, default=DEFAULT_SC26_SCALING_DIR)
    parser.add_argument("--pure-baseline-dir", type=pathlib.Path, default=DEFAULT_PURE_BASELINE_DIR)
    parser.add_argument("--hybrid-baseline-dir", type=pathlib.Path, default=DEFAULT_HYBRID_BASELINE_DIR)
    parser.add_argument("--bqsim-csv", type=pathlib.Path, default=DEFAULT_BQSIM_CSV)
    parser.add_argument("--bosonic-csv", type=pathlib.Path, default=DEFAULT_BOSONIC_CSV)
    parser.add_argument("--atlas-csv", type=pathlib.Path, default=DEFAULT_ATLAS_CSV)
    parser.add_argument("--csv-path", type=pathlib.Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = collect_rows(
        multigpu_dir=args.multigpu_dir,
        dense_summary=args.dense_summary,
        scaling_dir=args.scaling_dir,
        pure_baseline_dir=args.pure_baseline_dir,
        hybrid_baseline_dir=args.hybrid_baseline_dir,
        bqsim_csv=args.bqsim_csv,
        bosonic_csv=args.bosonic_csv,
        atlas_csv=args.atlas_csv,
    )
    write_csv(rows, args.csv_path)
    csv_rows = read_csv_rows(args.csv_path)
    summary_paths = plot_summary_bars(csv_rows, args.output_dir)
    sweep_paths = plot_hybrid_sweep_bars(csv_rows, args.output_dir)

    print(f"Wrote CSV: {args.csv_path}")
    print("Wrote figures:")
    for path in summary_paths + sweep_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
