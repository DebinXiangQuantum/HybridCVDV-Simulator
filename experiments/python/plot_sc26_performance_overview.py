#!/usr/bin/env python3
"""Generate SC26 overview bar charts and refresh the aggregate CSV."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Make hatch lines thinner globally
plt.rcParams['hatch.linewidth'] = 0.4

sys.path.insert(0, str(REPO_ROOT))
from experiments.configs.paper_style import (
    BASE_FONT_SIZE,
    DOUBLE_COLUMN_PT,
    SINGLE_COLUMN_PT,
    apply_paper_style,
    save_figure,
)


DEFAULT_CSV_PATH = REPO_ROOT / "SC26submission" / "expplots" / "sc26_baseline_plot_data.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "SC26submission" / "expplots"
DEFAULT_BENCHMARK_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "goldenres"
DEFAULT_BQSIM_CSV = REPO_ROOT / "baselines" / "results" / "bqsim_results.csv"
DEFAULT_BOSONIC_CSV = REPO_ROOT / "baselines" / "results" / "bosonicGPU.csv"
DEFAULT_ATLAS_CSV = REPO_ROOT / "baselines" / "results" / "atlas_results.csv"
DEFAULT_REMOTE_H100_PURE_DIR = REPO_ROOT / "experiments" / "results" / "remote-h100-baseline-sc26_baselines_20260316"

BYTES_PER_MIB = float(1024**2)
STUDY_CUTOFF = 16
MAX_HYBRID_QUBITS = 10

CUTOFF_FIGURE_CUTOFFS = [4, 8, 16]
PURE_CUTOFF_CASE = ("qaoa", 4)

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
    "compute_ms",
    "communication_ms",
    "memory_bytes",
    "cpu_memory_bytes",
    "gpu_memory_bytes",
    "memory_kind",
    "status",
    "source",
    "hardware",
    "note",
]

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
HYBRID_BACKENDS = ["hybridcvdv", "bqsim", "bosonic_gpu", "atlas"]
PURE_CV_BACKENDS = ["hybridcvdv", "bqsim", "bosonic_gpu", "atlas", "strawberryfields_tf", "mrmustard_jax"]

HYBRID_WORKLOAD_ORDER = [
    "transfer_cvtodv",
    "transfer_dvtocv",
    "qft",
    "shors",
    "jch",
]
PURE_WORKLOAD_ORDER = ["qaoa"]
WORKLOAD_LABELS = {
    "transfer_cvtodv": "CV->DV",
    "transfer_dvtocv": "DV->CV",
    "qft": "QFT",
    "shors": "Shor",
    "jch": "JCH",
    "vqe": "VQE",
    "qaoa": "QAOA",
    "cat": "Cat",
    "gkp": "GKP",
}
GROUP_LABELS = {
    "transfer_cvtodv": "CV->DV",
    "transfer_dvtocv": "DV->CV",
    "qft": "QFT",
    "shors": "Shor",
    "jch": "JCH",
    "vqe": "VQE",
    "qaoa": "QAOA",
    "cat": "Cat",
    "gkp": "GKP",
}
WORKLOAD_SHADE = {
    "transfer_cvtodv": "#eef5ea",
    "transfer_dvtocv": "#f6f0e6",
    "qft": "#fff3dd",
    "shors": "#fdf0f0",
    "jch": "#f7e8dd",
    "vqe": "#eaf1f8",
    "qaoa": "#eef5ea",
    "cat": "#f6f0e6",
    "gkp": "#fff3dd",
}
WORKLOAD_ORDER_FOR_SORT = {name: index for index, name in enumerate(HYBRID_WORKLOAD_ORDER + PURE_WORKLOAD_ORDER)}
SPEEDUP_MARKERS = {
    "bqsim": "o",
    "bosonic_gpu": "s",
    "strawberryfields_tf": "^",
    "mrmustard_jax": "D",
}


@dataclass(frozen=True)
class CaseKey:
    category: str
    workload: str
    num_qubits: Optional[int]
    num_modes: Optional[int]
    cutoff: int


def optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    return int(value)


def optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    return float(value)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_float(metrics: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in metrics and metrics.get(key) not in (None, "", "None"):
            return optional_float(metrics.get(key))
    return None


def is_number_text(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


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
    compute_ms: Optional[float],
    communication_ms: Optional[float],
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
        "compute_ms": compute_ms,
        "communication_ms": communication_ms,
        "memory_bytes": memory_bytes,
        "cpu_memory_bytes": cpu_memory_bytes,
        "gpu_memory_bytes": gpu_memory_bytes,
        "memory_kind": memory_kind,
        "status": status,
        "source": source,
        "hardware": hardware,
        "note": note,
    }


def row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row["backend_key"],
        row["category"],
        row["workload"],
        row["num_qubits"],
        row["num_modes"],
        row["cutoff"],
    )


def store_row(rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]], row: Dict[str, Any]) -> None:
    rows_by_key[row_key(row)] = row


def read_existing_rows(csv_path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "backend_label": BACKEND_LABELS.get(str(row.get("backend_key", "")), str(row.get("backend_label", ""))),
                    "num_qubits": optional_int(row.get("num_qubits")),
                    "num_modes": optional_int(row.get("num_modes")),
                    "cutoff": optional_int(row.get("cutoff")),
                    "runtime_ms": optional_float(row.get("runtime_ms")),
                    "compute_ms": optional_float(row.get("compute_ms")),
                    "communication_ms": optional_float(row.get("communication_ms")),
                    "memory_bytes": optional_float(row.get("memory_bytes")),
                    "cpu_memory_bytes": optional_float(row.get("cpu_memory_bytes")),
                    "gpu_memory_bytes": optional_float(row.get("gpu_memory_bytes")),
                }
            )
    return rows


def parse_extra_our_benchmark_results(
    rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    benchmark_results_dir: pathlib.Path,
) -> None:
    workload_map = {
        "qft_circuit": "qft",
        "shors_circuit": "shors",
        "state_transfer_CVtoDV_circuit": "transfer_cvtodv",
        "state_transfer_DVtoCV_circuit": "transfer_dvtocv",
        "jch_simulation_circuit": "jch",
        "vqe_circuit": "vqe",
        "qaoa_circuit": "qaoa",
        "cat_state_circuit": "cat",
        "gkp_state_circuit": "gkp",
    }
    qft_name_pattern = re.compile(r"sc26_qft_nq(\d+)_c(\d+)$")
    shors_name_pattern = re.compile(r"sc26_shors_c(\d+)$")
    transfer_name_pattern = re.compile(r"sc26_transfer_(CVtoDV|DVtoCV)_nq(\d+)_c(\d+)$")

    for path in sorted(benchmark_results_dir.glob("*.json")):
        payload = load_json(path)
        results = payload.get("results") or []
        for record in results:
            params = record.get("params") or {}
            case_id = str(record.get("name", path.stem.replace("result_", "")))
            mapped_workload = workload_map.get(str(params.get("workload", "")))

            num_qubits = optional_int(params.get("num_qubits"))
            num_modes = optional_int(params.get("num_qumodes"))
            cutoff = optional_int(params.get("cutoff"))
            if mapped_workload is None or cutoff is None:
                qft_match = qft_name_pattern.match(case_id)
                if qft_match is not None:
                    mapped_workload = "qft"
                    num_qubits = int(qft_match.group(1))
                    num_modes = 1
                    cutoff = int(qft_match.group(2))
                else:
                    shors_match = shors_name_pattern.match(case_id)
                    if shors_match is not None:
                        mapped_workload = "shors"
                        num_qubits = 1
                        num_modes = 3
                        cutoff = int(shors_match.group(1))
                    else:
                        transfer_match = transfer_name_pattern.match(case_id)
                        if transfer_match is not None:
                            mapped_workload = "transfer_cvtodv" if transfer_match.group(1) == "CVtoDV" else "transfer_dvtocv"
                            num_qubits = int(transfer_match.group(2))
                            num_modes = 1
                            cutoff = int(transfer_match.group(3))

            if mapped_workload is None or cutoff is None:
                continue
            
            if num_qubits is None and mapped_workload in ("jch", "vqe", "qft", "shors", "transfer_cvtodv", "transfer_dvtocv"):
                num_qubits = 1
            if num_modes is None and mapped_workload in ("qft", "transfer_cvtodv", "transfer_dvtocv", "cat", "gkp"):
                num_modes = 1
            
            category = "hybrid_dv_cv" if mapped_workload in ("jch", "vqe", "qft", "shors", "transfer_cvtodv", "transfer_dvtocv") else "pure_cv"
            
            if category == "pure_cv":
                num_qubits = None
                if num_modes is None:
                    num_modes = 1
                
            status = str(record.get("status", "unknown"))
            metrics = record.get("metrics") or {}
            runtime_ms = metric_float(metrics, "median_total_ms", "runner_wall_time_ms") if status == "ok" else None
            compute_ms = metric_float(metrics, "median_compute_ms") if status == "ok" else None
            communication_ms = metric_float(metrics, "median_transfer_ms", "median_communication_ms") if status == "ok" else None
            memory_bytes = metric_float(metrics, "median_memory_bytes", "gpu_peak_memory_used_mb") if status == "ok" else None
            
            # Convert MB to bytes if needed
            if memory_bytes is not None and "mb" in str(metrics.keys()).lower() and "median_memory_bytes" not in metrics:
                memory_bytes *= 1024 * 1024
            
            gpu_name = str((payload.get("device") or {}).get("name") or "GPU")

            store_row(
                rows_by_key,
                make_row(
                    backend_key="hybridcvdv",
                    category=category,
                    workload=mapped_workload,
                    case_id=case_id,
                    num_qubits=num_qubits,
                    num_modes=num_modes,
                    cutoff=cutoff,
                    runtime_ms=runtime_ms,
                    compute_ms=compute_ms,
                    communication_ms=communication_ms,
                    memory_bytes=memory_bytes,
                    memory_kind="resident",
                    status=status,
                    source="benchmark_results",
                    hardware="1x {}".format(gpu_name),
                    note=str(record.get("note", "") or ""),
                ),
            )



def parse_atlas_csv(
    rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    csv_path: pathlib.Path,
) -> None:
    if not csv_path.exists():
        return

    jch_pattern = re.compile(r"sc26_jch_nq(\d+)_nm(\d+)_c(\d+)$")
    vqe_pattern = re.compile(r"sc26_vqe_nq(\d+)_nm(\d+)_c(\d+)$")
    qaoa_pattern = re.compile(r"sc26_qaoa_nm(\d+)_c(\d+)$")
    qft_pattern = re.compile(r"sc26_qft_nq(\d+)_c(\d+)$")
    shors_pattern = re.compile(r"sc26_shors_c(\d+)$")
    cat_pattern = re.compile(r"sc26_cat_c(\d+)$")
    gkp_pattern = re.compile(r"sc26_gkp_c(\d+)$")
    transfer_pattern = re.compile(r"sc26_transfer_(CVtoDV|DVtoCV)_nq(\d+)_c(\d+)$")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            case_id = row[0]
            runtime_ms = float(row[2])
            communication_ms = float(row[3])
            compute_ms = float(row[4])
            cpu_memory_bytes = float(row[5])
            gpu_memory_bytes = float(row[6]) if len(row) > 6 else 0.0
            memory_bytes = cpu_memory_bytes + gpu_memory_bytes
            
            note = ""
            if len(row) > 7:
                note = row[7]
            
            status = "ok" if runtime_ms > 0.0 else "failed"
            if "超出可用范围" in note or "OOM" in note.upper():
                status = "failed"
                note = "OOM"
            
            runtime_value = runtime_ms if status == "ok" else None
            compute_value = compute_ms if status == "ok" else None
            communication_value = communication_ms if status == "ok" else None
            memory_value = memory_bytes if status == "ok" else None
            cpu_mem_value = cpu_memory_bytes if status == "ok" else None
            gpu_mem_value = gpu_memory_bytes if status == "ok" else None

            match = jch_pattern.match(case_id)
            if match:
                category, workload = "hybrid_dv_cv", "jch"
                num_qubits, num_modes, cutoff = int(match.group(1)), int(match.group(2)), int(match.group(3))
            elif (match := vqe_pattern.match(case_id)):
                category, workload = "hybrid_dv_cv", "vqe"
                num_qubits, num_modes, cutoff = int(match.group(1)), int(match.group(2)), int(match.group(3))
            elif (match := qaoa_pattern.match(case_id)):
                category, workload = "pure_cv", "qaoa"
                num_qubits, num_modes, cutoff = None, int(match.group(1)), int(match.group(2))
            elif (match := qft_pattern.match(case_id)):
                category, workload = "hybrid_dv_cv", "qft"
                num_qubits, num_modes, cutoff = int(match.group(1)), 1, int(match.group(2))
            elif (match := shors_pattern.match(case_id)):
                category, workload = "hybrid_dv_cv", "shors"
                num_qubits, num_modes, cutoff = 1, 3, int(match.group(1))
            elif (match := cat_pattern.match(case_id)):
                category, workload = "pure_cv", "cat"
                num_qubits, num_modes, cutoff = None, 1, int(match.group(1))
            elif (match := gkp_pattern.match(case_id)):
                category, workload = "pure_cv", "gkp"
                num_qubits, num_modes, cutoff = None, 1, int(match.group(1))
            elif (match := transfer_pattern.match(case_id)):
                category = "hybrid_dv_cv"
                workload = "transfer_cvtodv" if match.group(1) == "CVtoDV" else "transfer_dvtocv"
                num_qubits, num_modes, cutoff = int(match.group(2)), 1, int(match.group(3))
            else:
                continue

            store_row(
                rows_by_key,
                make_row(
                    backend_key="atlas",
                    category=category,
                    workload=workload,
                    case_id=case_id,
                    num_qubits=num_qubits,
                    num_modes=num_modes,
                    cutoff=cutoff,
                    runtime_ms=runtime_value,
                    compute_ms=compute_value,
                    communication_ms=communication_value,
                    memory_bytes=memory_value,
                    cpu_memory_bytes=cpu_mem_value,
                    gpu_memory_bytes=gpu_mem_value,
                    memory_kind="reported",
                    status=status,
                    source="atlas_csv",
                    hardware="Atlas baseline",
                    note=note,
                ),
            )


def parse_remote_reference_results(
    rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    results_dir: pathlib.Path,
) -> None:
    if not results_dir.exists():
        return

    workload_map = {
        "cv_qaoa": "qaoa",
        "jch_photonic_chain": "jch",
    }

    for path in sorted(results_dir.glob("*.json")):
        payload = load_json(path)
        backend_key = str(payload.get("backend", ""))
        if backend_key not in BACKEND_LABELS:
            continue
            
        results = payload.get("results") or []
        for record in results:
            params = record.get("params") or {}
            workload_raw = str(params.get("workload", ""))
            mapped_workload = workload_map.get(workload_raw)
            if not mapped_workload:
                continue
                
            cutoff = optional_int(params.get("cutoff"))
            num_modes = optional_int(params.get("num_modes"))
            
            # QAOA is pure CV, JCH is hybrid (we use 1 qubit for comparison)
            if mapped_workload == "qaoa":
                category = "pure_cv"
                num_qubits = None
            else:
                category = "hybrid_dv_cv"
                num_qubits = 1
                
            status = str(record.get("status", "unknown"))
            metrics = record.get("metrics") or {}
            mem_val = metric_float(metrics, "state_vector_bytes_estimate")
            
            store_row(
                rows_by_key,
                make_row(
                    backend_key=backend_key,
                    category=category,
                    workload=mapped_workload,
                    case_id=path.stem,
                    num_qubits=num_qubits,
                    num_modes=num_modes,
                    cutoff=cutoff,
                    runtime_ms=metric_float(metrics, "median_total_ms"),
                    compute_ms=metric_float(metrics, "median_compute_ms"),
                    communication_ms=metric_float(metrics, "median_communication_ms"),
                    memory_bytes=mem_val,
                    memory_kind="estimated",
                    status=status,
                    source="remote_h100_pure",
                    hardware="NVIDIA H100 (Reference)",
                    note=str(record.get("note", "") or ""),
                ),
            )


def parse_bqsim_csv(
    rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    csv_path: pathlib.Path,
) -> None:
    jch_pattern = re.compile(r"sc26_jch_nq(\d+)_nm(\d+)_c(\d+)$")
    vqe_pattern = re.compile(r"sc26_vqe_nq(\d+)_nm(\d+)_c(\d+)$")
    qaoa_pattern = re.compile(r"sc26_qaoa_nm(\d+)_c(\d+)$")
    cat_pattern = re.compile(r"sc26_cat_c(\d+)$")
    gkp_pattern = re.compile(r"sc26_gkp_c(\d+)$")
    qft_pattern = re.compile(r"sc26_qft_nq(\d+)_c(\d+)$")
    shors_pattern = re.compile(r"sc26_shors_c(\d+)$")
    transfer_pattern = re.compile(r"sc26_transfer_(CVtoDV|DVtoCV)_nq(\d+)_c(\d+)$")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            case_id = row[0]
            runtime_ms = float(row[2])
            communication_ms = float(row[3])
            compute_ms = float(row[4])
            memory_bytes = float(row[5])
            status = "ok" if runtime_ms >= 0.0 else "failed"
            runtime_value = runtime_ms if status == "ok" else None
            compute_value = compute_ms if status == "ok" and compute_ms >= 0.0 else None
            communication_value = communication_ms if status == "ok" and communication_ms >= 0.0 else None
            memory_value = memory_bytes if status == "ok" else None

            match = jch_pattern.match(case_id)
            if match is not None:
                category = "hybrid_dv_cv"
                workload = "jch"
                num_qubits = int(match.group(1))
                num_modes = int(match.group(2))
                cutoff = int(match.group(3))
            else:
                match = vqe_pattern.match(case_id)
                if match is not None:
                    category = "hybrid_dv_cv"
                    workload = "vqe"
                    num_qubits = int(match.group(1))
                    num_modes = int(match.group(2))
                    cutoff = int(match.group(3))
                else:
                    match = qaoa_pattern.match(case_id)
                    if match is not None:
                        category = "pure_cv"
                        workload = "qaoa"
                        num_qubits = None
                        num_modes = int(match.group(1))
                        cutoff = int(match.group(2))
                    elif (match := cat_pattern.match(case_id)):
                        category = "pure_cv"
                        workload = "cat"
                        num_qubits = None
                        num_modes = 1
                        cutoff = int(match.group(1))
                    elif (match := gkp_pattern.match(case_id)):
                        category = "pure_cv"
                        workload = "gkp"
                        num_qubits = None
                        num_modes = 1
                        cutoff = int(match.group(1))
                    else:
                        match = qft_pattern.match(case_id)
                        if match is not None:
                            category = "hybrid_dv_cv"
                            workload = "qft"
                            num_qubits = int(match.group(1))
                            num_modes = 1
                            cutoff = int(match.group(2))
                        else:
                            match = shors_pattern.match(case_id)
                            if match is not None:
                                category = "hybrid_dv_cv"
                                workload = "shors"
                                num_qubits = 1
                                num_modes = 3
                                cutoff = int(match.group(1))
                            else:
                                match = transfer_pattern.match(case_id)
                                if match is None:
                                    continue
                                category = "hybrid_dv_cv"
                                workload = "transfer_cvtodv" if match.group(1) == "CVtoDV" else "transfer_dvtocv"
                                num_qubits = int(match.group(2))
                                num_modes = 1
                                cutoff = int(match.group(3))

            store_row(
                rows_by_key,
                make_row(
                    backend_key="bqsim",
                    category=category,
                    workload=workload,
                    case_id=case_id,
                    num_qubits=num_qubits,
                    num_modes=num_modes,
                    cutoff=cutoff,
                    runtime_ms=runtime_value,
                    compute_ms=compute_value,
                    communication_ms=communication_value,
                    memory_bytes=memory_value,
                    memory_kind="reported",
                    status=status,
                    source="bqsim_csv",
                    hardware="Archived GPU baseline",
                ),
            )


def parse_bosonic_csv(
    rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    csv_path: pathlib.Path,
) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue

            note = ""
            tail = row
            if not is_number_text(row[-1]):
                note = row[-1]
                tail = row[:-1]
            if len(tail) < 6:
                continue

            runtime_ms = float(tail[-4])
            communication_ms = float(tail[-3])
            compute_ms = float(tail[-2])
            memory_bytes = float(tail[-1])
            status = "ok" if runtime_ms > 0.0 else "failed"
            runtime_value = runtime_ms if status == "ok" else None
            compute_value = compute_ms if status == "ok" else None
            communication_value = communication_ms if status == "ok" else None
            memory_value = memory_bytes if status == "ok" else None

            params: Dict[str, str] = {}
            for item in row[1:]:
                if "=" in item:
                    key, value = item.split("=", 1)
                    params[key] = value

            category: Optional[str] = None
            workload: Optional[str] = None
            num_qubits: Optional[int] = None
            num_modes: Optional[int] = None
            cutoff: Optional[int] = None
            kind = row[0]

            if kind == "jch_simulation_circuit":
                category = "hybrid_dv_cv"
                workload = "jch"
                num_qubits = optional_int(params.get("num_qubits"))
                num_modes = optional_int(params.get("num_modes") or params.get("qumodes") or params.get("num_qumodes"))
                cutoff = optional_int(params.get("cutoff"))
            elif kind == "vqe_circuit":
                category = "hybrid_dv_cv"
                workload = "vqe"
                num_qubits = optional_int(params.get("qubits"))
                num_modes = optional_int(params.get("qumodes") or params.get("num_qumodes"))
                cutoff = optional_int(params.get("cutoff"))
            elif kind == "qaoa_circuit":
                category = "pure_cv"
                workload = "qaoa"
                num_qubits = None
                num_modes = optional_int(params.get("qumodes") or params.get("num_qumodes"))
                cutoff = optional_int(params.get("cutoff"))
            elif kind == "cat_state_circuit":
                category = "pure_cv"
                workload = "cat"
                num_qubits = None
                num_modes = optional_int(params.get("num_qumodes") or params.get("qumodes"))
                cutoff = optional_int(params.get("cutoff"))
            elif kind == "gkp_state_circuit":
                category = "pure_cv"
                workload = "gkp"
                num_qubits = None
                num_modes = optional_int(params.get("num_qumodes") or params.get("qumodes"))
                cutoff = optional_int(params.get("cutoff"))
            elif kind == "qft_circuit":
                category = "hybrid_dv_cv"
                workload = "qft"
                n_value = optional_int(params.get("n"))
                append_value = optional_int(params.get("append"))
                cutoff = optional_int(params.get("cutoff"))
                if n_value is not None and append_value is not None:
                    num_qubits = int(n_value + append_value + 1)
                    num_modes = 1
            elif kind == "state_transfer_circuit":
                category = "hybrid_dv_cv"
                workload = "transfer_cvtodv" if row[1] == "CVtoDV" else "transfer_dvtocv"
                num_qubits = optional_int(params.get("qubits"))
                num_modes = 1
                cutoff = optional_int(params.get("cutoff"))

            if category is None or workload is None or cutoff is None:
                continue

            if category == "pure_cv":
                num_qubits = None
                if num_modes is None:
                    num_modes = 1

            store_row(
                rows_by_key,
                make_row(
                    backend_key="bosonic_gpu",
                    category=category,
                    workload=workload,
                    case_id="|".join(row[: min(4, len(row))]),
                    num_qubits=num_qubits,
                    num_modes=num_modes,
                    cutoff=cutoff,
                    runtime_ms=runtime_value,
                    compute_ms=compute_value,
                    communication_ms=communication_value,
                    memory_bytes=memory_value,
                    memory_kind="reported",
                    status=status,
                    source="bosonic_csv",
                    hardware="Archived GPU baseline",
                    note=note,
                ),
            )


def sorted_rows(rows_by_key: Mapping[Tuple[Any, ...], Dict[str, Any]]) -> List[Dict[str, Any]]:
    category_order = {"hybrid_dv_cv": 0, "pure_cv": 1}
    backend_order = {key: index for index, key in enumerate(PURE_CV_BACKENDS)}
    rows = list(rows_by_key.values())
    rows.sort(
        key=lambda row: (
            category_order.get(str(row["category"]), 99),
            WORKLOAD_ORDER_FOR_SORT.get(str(row["workload"]), 99),
            -1 if row["num_qubits"] is None else int(row["num_qubits"]),
            -1 if row["num_modes"] is None else int(row["num_modes"]),
            -1 if row["cutoff"] is None else int(row["cutoff"]),
            backend_order.get(str(row["backend_key"]), 99),
        )
    )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], csv_path: pathlib.Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in CSV_FIELDS})


def update_row_from_json_metrics(row: Dict[str, Any], json_path: pathlib.Path) -> bool:
    if not json_path.exists():
        return False
    payload = load_json(json_path)
    results = payload.get("results") or []
    if not results:
        return False
    record = results[0]
    metrics = record.get("metrics") or {}
    status = str(record.get("status", row.get("status") or "unknown"))
    row["status"] = status
    row["backend_label"] = BACKEND_LABELS.get(str(row["backend_key"]), str(row.get("backend_label", "")))
    row["runtime_ms"] = metric_float(metrics, "median_total_ms") if status == "ok" else None
    row["compute_ms"] = metric_float(metrics, "median_compute_ms") if status == "ok" else None
    row["communication_ms"] = metric_float(metrics, "median_transfer_ms", "median_communication_ms") if status == "ok" else None
    memory_value = metric_float(metrics, "median_memory_bytes", "state_vector_bytes_estimate")
    if memory_value is not None:
        row["memory_bytes"] = memory_value
    note = str(record.get("note", "") or "")
    if note:
        row["note"] = note
    return True


def candidate_json_paths_for_row(row: Mapping[str, Any]) -> List[pathlib.Path]:
    case_id = str(row.get("case_id", "") or "")
    source = str(row.get("source", "") or "")
    if not case_id:
        return []
    if source == "multigpubench":
        return [DEFAULT_MULTIGPUBENCH_DIR / "result_{}.json".format(case_id)]
    if source == "benchmark_results":
        return [DEFAULT_BENCHMARK_RESULTS_DIR / "result_{}.json".format(case_id)]
    if source == "remote_h100_pure":
        return [DEFAULT_REMOTE_H100_PURE_DIR / "{}.json".format(case_id)]
    if source == "dense_summary:reference":
        return [DEFAULT_SC26_DENSE_ROOT / "hybrid_reference" / "{}.json".format(case_id)]
    if source == "dense_summary:cutoff":
        return [DEFAULT_SC26_DENSE_ROOT / "cutoff" / "{}.json".format(case_id)]
    return []


def enrich_rows_with_json_breakdown(rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]]) -> None:
    for row in rows_by_key.values():
        for candidate in candidate_json_paths_for_row(row):
            if update_row_from_json_metrics(row, candidate):
                break


def build_case_key(row: Mapping[str, Any]) -> CaseKey:
    return CaseKey(
        category=str(row["category"]),
        workload=str(row["workload"]),
        num_qubits=optional_int(row["num_qubits"]) if not isinstance(row["num_qubits"], int) else int(row["num_qubits"]),
        num_modes=optional_int(row["num_modes"]) if not isinstance(row["num_modes"], int) else int(row["num_modes"]),
        cutoff=int(row["cutoff"]),
    )


def row_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, CaseKey], Mapping[str, Any]]:
    lookup: Dict[Tuple[str, CaseKey], Mapping[str, Any]] = {}
    for row in rows:
        lookup[(str(row["backend_key"]), build_case_key(row))] = row
    return lookup


def select_hybrid_cases(rows: Sequence[Mapping[str, Any]], cutoff: int, max_qubits: int) -> List[CaseKey]:
    # Group rows by case key to check cross-backend presence
    rows_by_case: Dict[CaseKey, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if int(row["cutoff"]) != cutoff:
            continue
        if row["category"] != "hybrid_dv_cv":
            continue
        if row["workload"] not in HYBRID_WORKLOAD_ORDER:
            continue
        ck = build_case_key(row)
        if ck not in rows_by_case:
            rows_by_case[ck] = {}
        rows_by_case[ck][str(row["backend_key"])] = row

    case_set = set()
    for ck, backends in rows_by_case.items():
        # 1. JCH m > 6 filter
        if ck.workload == "jch" and ck.num_modes is not None and ck.num_modes > 6:
            continue
            
        # 2. Gantry missing runtime filter
        gantry_row = backends.get("hybridcvdv")
        if not gantry_row or gantry_row.get("status") != "ok" or gantry_row.get("runtime_ms") is None:
            continue
            
        # 3. Only Gantry data filter (Must have at least one baseline)
        has_baseline = False
        for b_key in ["bqsim", "bosonic_gpu", "atlas"]:
            b_row = backends.get(b_key)
            if b_row:
                if b_row.get("status") == "ok" or row_is_oom(b_row):
                    has_baseline = True
                    break
        
        if not has_baseline:
            continue
            
        if ck.num_qubits is not None and ck.num_qubits <= max_qubits:
            case_set.add(ck)
            
    return sorted(
        case_set,
        key=lambda case: (
            HYBRID_WORKLOAD_ORDER.index(case.workload),
            case.num_qubits if case.num_qubits is not None else -1,
            case.num_modes if case.num_modes is not None else -1,
        ),
    )


def select_pure_cv_cases(rows: Sequence[Mapping[str, Any]], cutoff: int) -> List[CaseKey]:
    # Group rows by case key
    rows_by_case: Dict[CaseKey, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if int(row["cutoff"]) != cutoff:
            continue
        if row["category"] != "pure_cv":
            continue
        if row["workload"] not in PURE_WORKLOAD_ORDER:
            continue
        ck = build_case_key(row)
        if ck not in rows_by_case:
            rows_by_case[ck] = {}
        rows_by_case[ck][str(row["backend_key"])] = row

    case_set = set()
    for ck, backends in rows_by_case.items():
        # 1. Gantry presence filter (include even if failed/OOM for visibility)
        gantry_row = backends.get("hybridcvdv")
        if not gantry_row:
            continue
            
        # 2. Only Gantry data filter
        has_baseline = False
        for b_key in ["bqsim", "bosonic_gpu", "atlas", "strawberryfields_tf", "mrmustard_jax"]:
            b_row = backends.get(b_key)
            if b_row:
                if b_row.get("status") == "ok" or row_is_oom(b_row):
                    has_baseline = True
                    break
        
        if has_baseline:
            case_set.add(ck)
            
    return sorted(
        case_set,
        key=lambda case: (
            PURE_WORKLOAD_ORDER.index(case.workload) if case.workload in PURE_WORKLOAD_ORDER else 99,
            case.num_modes if case.num_modes is not None else -1,
        ),
    )


def case_label(case: CaseKey) -> str:
    if case.workload == "transfer_cvtodv":
        return "CV2DV-q{}m{}".format(case.num_qubits, case.num_modes)
    if case.workload == "transfer_dvtocv":
        return "DV2CV-q{}m{}".format(case.num_qubits, case.num_modes)
    if case.workload == "qft":
        return "QFT-q{}m{}".format(case.num_qubits, case.num_modes)
    if case.workload == "shors":
        return "Shor-q{}m{}".format(case.num_qubits, case.num_modes)
    if case.category == "hybrid_dv_cv":
        return "{}-q{}m{}".format(WORKLOAD_LABELS[case.workload], case.num_qubits, case.num_modes)
    return "{}-m{}".format(WORKLOAD_LABELS[case.workload], case.num_modes)


def workload_ranges(cases: Sequence[CaseKey], workload_order: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not cases:
        return []
    ranges: List[Tuple[int, int, str]] = []
    start = 0
    while start < len(cases):
        workload = cases[start].workload
        end = start
        while end + 1 < len(cases) and cases[end + 1].workload == workload:
            end += 1
        ranges.append((start, end, workload))
        start = end + 1
    ranges.sort(key=lambda item: workload_order.index(item[2]))
    return ranges


def series_values(
    cases: Sequence[CaseKey],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    backend_key: str,
    metric_name: str,
    scale: float,
) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for case in cases:
        row = lookup.get((backend_key, case))
        if row is None or row.get("status") != "ok":
            values.append(None)
            continue
        metric = row.get(metric_name)
        if metric in (None, "", "None"):
            values.append(None)
            continue
        values.append(float(metric) / scale)
    return values


def count_missing_cases(
    cases: Sequence[CaseKey],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    backend_keys: Iterable[str],
) -> Dict[str, int]:
    missing: Dict[str, int] = {}
    for backend_key in backend_keys:
        count = 0
        for case in cases:
            row = lookup.get((backend_key, case))
            if row is None or row.get("status") != "ok":
                count += 1
        missing[backend_key] = count
    return missing


def add_group_background(
    ax: plt.Axes,
    cases: Sequence[CaseKey],
    workload_order: Sequence[str],
    *,
    show_group_labels: bool = True,
) -> None:
    for start, end, workload in workload_ranges(cases, workload_order):
        ax.axvspan(start - 0.5, end + 0.5, color=WORKLOAD_SHADE.get(workload, "#f7f7f7"), alpha=0.55, zorder=0)
        if show_group_labels:
            ax.text(
                0.5 * (start + end),
                1.012,
                "{}\n(n={})".format(GROUP_LABELS.get(workload, workload), end - start + 1),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6,
                linespacing=0.9,
            )
    for start, end, _ in workload_ranges(cases, workload_order)[:-1]:
        del start
        ax.axvline(end + 0.5, color="#777777", linestyle="--", linewidth=0.6, alpha=0.8)


def plot_grouped_bars(
    ax: plt.Axes,
    cases: Sequence[CaseKey],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    backend_keys: Sequence[str],
    metric_name: str,
    ylabel: str,
    *,
    scale: float = 1.0,
) -> None:
    all_values: List[float] = []
    backend_series: Dict[str, List[Optional[float]]] = {}
    for backend_key in backend_keys:
        values = series_values(cases, lookup, backend_key, metric_name, scale)
        backend_series[backend_key] = values
        all_values.extend(value for value in values if value is not None and value > 0.0)

    if not all_values:
        return {"base": 1.0, "ymax": 1.0, "bar_width": 0.2, "offset_start": -0.3}

    base = min(all_values) / 4.0
    ymax = max(all_values) * 1.8
    x_positions = list(range(len(cases)))
    group_width = 0.80
    bar_width = min(0.22, group_width / max(1, len(backend_keys)))
    offset_start = -0.5 * group_width + 0.5 * bar_width

    for index, backend_key in enumerate(backend_keys):
        x_values: List[float] = []
        heights: List[float] = []
        for case_index, value in enumerate(backend_series[backend_key]):
            if value is None or value <= 0.0:
                continue
            x_values.append(x_positions[case_index] + offset_start + index * bar_width)
            heights.append(value - base)
        ax.bar(
            x_values,
            heights,
            width=bar_width * 0.94,
            bottom=base,
            color=BACKEND_COLORS[backend_key],
            edgecolor="black",
            linewidth=0.25,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_ylim(base, ymax)
    ax.set_ylabel(ylabel)
    return {
        "base": base,
        "ymax": ymax,
        "bar_width": bar_width,
        "offset_start": offset_start,
    }


def runtime_breakdown_values(row: Optional[Mapping[str, Any]], *, scale: float = 1.0) -> Optional[Dict[str, float]]:
    if row is None or row.get("status") != "ok":
        return None
    total_ms = optional_float(row.get("runtime_ms"))
    if total_ms is None or total_ms <= 0.0:
        return None
    compute_ms = optional_float(row.get("compute_ms")) or 0.0
    communication_ms = optional_float(row.get("communication_ms")) or 0.0
    other_ms = max(total_ms - compute_ms - communication_ms, 0.0)
    return {
        "total": total_ms / scale,
        "other": other_ms / scale,
        "compute": max(compute_ms, 0.0) / scale,
        "communication": max(communication_ms, 0.0) / scale,
    }


def plot_runtime_bars_with_breakdown(
    ax: plt.Axes,
    cases: Sequence[CaseKey],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    backend_keys: Sequence[str],
    ylabel: str,
) -> Dict[str, float]:
    all_values: List[float] = []
    runtime_series: Dict[str, List[Optional[Dict[str, float]]]] = {}
    for backend_key in backend_keys:
        values: List[Optional[Dict[str, float]]] = []
        for case in cases:
            value = runtime_breakdown_values(lookup.get((backend_key, case)))
            values.append(value)
            if value is not None and value["total"] > 0.0:
                all_values.append(value["total"])
        runtime_series[backend_key] = values

    if not all_values:
        return {"base": 1.0, "ymax": 1.0, "bar_width": 0.2, "offset_start": -0.3}

    base = min(all_values) / 4.0
    ymax = (max(all_values) + base) * 1.8
    x_positions = list(range(len(cases)))
    group_width = 0.80
    bar_width = min(0.22, group_width / max(1, len(backend_keys)))
    offset_start = -0.5 * group_width + 0.5 * bar_width
    
    # Order: compute (bottom), communication (middle), other (top)
    segment_styles = [
        ("compute", 0.95, "////////"),
        ("communication", 0.75, "xxxxx"),
        ("other", 0.35, None),
    ]

    for index, backend_key in enumerate(backend_keys):
        for case_index, value in enumerate(runtime_series[backend_key]):
            if value is None:
                continue
            x = x_positions[case_index] + offset_start + index * bar_width
            bottom = base
            for segment_key, alpha, hatch in segment_styles:
                segment_value = value[segment_key]
                if segment_value <= 0.0:
                    continue
                ax.bar(
                    x,
                    segment_value,
                    width=bar_width * 0.94,
                    bottom=bottom,
                    color=BACKEND_COLORS[backend_key],
                    edgecolor="black",
                    linewidth=0.2,
                    hatch=hatch,
                    alpha=alpha,
                    zorder=3,
                )
                bottom += segment_value

    ax.set_yscale("log")
    ax.set_ylim(base, ymax)
    ax.set_ylabel(ylabel)
    return {
        "base": base,
        "ymax": ymax,
        "bar_width": bar_width,
        "offset_start": offset_start,
    }


def memory_breakdown_values(row: Optional[Mapping[str, Any]], *, scale: float = 1.0) -> Optional[Dict[str, float]]:
    """Return {"total", "gpu", "cpu"} memory breakdown in scaled units. GPU on bottom, CPU on top."""
    if row is None or row.get("status") != "ok":
        return None
    total = optional_float(row.get("memory_bytes"))
    if total is None or total <= 0.0:
        return None
    gpu = optional_float(row.get("gpu_memory_bytes"))
    cpu = optional_float(row.get("cpu_memory_bytes"))
    if gpu is not None and cpu is not None:
        return {"total": total / scale, "gpu": gpu / scale, "cpu": cpu / scale}
    # Non-atlas backends: treat the single memory value as GPU memory
    return {"total": total / scale, "gpu": total / scale, "cpu": 0.0}


def plot_memory_bars_with_breakdown(
    ax: plt.Axes,
    cases: Sequence[CaseKey],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    backend_keys: Sequence[str],
    ylabel: str,
    *,
    scale: float = BYTES_PER_MIB,
) -> Dict[str, float]:
    """Stacked memory bars: GPU memory (solid, bottom) + CPU memory (hatch //, top)."""
    all_values: List[float] = []
    memory_series: Dict[str, List[Optional[Dict[str, float]]]] = {}
    for backend_key in backend_keys:
        values: List[Optional[Dict[str, float]]] = []
        for case in cases:
            value = memory_breakdown_values(lookup.get((backend_key, case)), scale=scale)
            values.append(value)
            if value is not None and value["total"] > 0.0:
                all_values.append(value["total"])
        memory_series[backend_key] = values

    if not all_values:
        return {"base": 1.0, "ymax": 1.0, "bar_width": 0.2, "offset_start": -0.3}

    base = min(all_values) / 4.0
    ymax = (max(all_values) + base) * 1.8
    x_positions = list(range(len(cases)))
    group_width = 0.80
    bar_width = min(0.22, group_width / max(1, len(backend_keys)))
    offset_start = -0.5 * group_width + 0.5 * bar_width

    # GPU (solid, bottom) then CPU (hatch //, top)
    segment_styles = [
        ("gpu", 0.95, None),
        ("cpu", 0.75, "////"),
    ]

    for index, backend_key in enumerate(backend_keys):
        for case_index, value in enumerate(memory_series[backend_key]):
            if value is None:
                continue
            x = x_positions[case_index] + offset_start + index * bar_width
            bottom = base
            for segment_key, alpha, hatch in segment_styles:
                segment_value = value[segment_key]
                if segment_value <= 0.0:
                    continue
                ax.bar(
                    x,
                    segment_value,
                    width=bar_width * 0.94,
                    bottom=bottom,
                    color=BACKEND_COLORS[backend_key],
                    edgecolor="black",
                    linewidth=0.2,
                    hatch=hatch,
                    alpha=alpha,
                    zorder=3,
                )
                bottom += segment_value

    ax.set_yscale("log")
    ax.set_ylim(base, ymax)
    ax.set_ylabel(ylabel)
    return {
        "base": base,
        "ymax": ymax,
        "bar_width": bar_width,
        "offset_start": offset_start,
    }


def add_speedup_lines(
    ax: plt.Axes,
    cases: Sequence[CaseKey],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    backend_keys: Sequence[str],
) -> Tuple[Optional[plt.Axes], Dict[str, List[Optional[float]]]]:
    speedup_series: Dict[str, List[Optional[float]]] = {}
    all_values: List[float] = []
    ours_key = "hybridcvdv"
    x_positions = list(range(len(cases)))

    for backend_key in backend_keys:
        if backend_key == ours_key:
            continue
        values: List[Optional[float]] = []
        for case in cases:
            ours_row = lookup.get((ours_key, case))
            other_row = lookup.get((backend_key, case))
            ours_runtime = optional_float(ours_row.get("runtime_ms")) if ours_row and ours_row.get("status") == "ok" else None
            other_runtime = optional_float(other_row.get("runtime_ms")) if other_row and other_row.get("status") == "ok" else None
            if ours_runtime is None or other_runtime is None or ours_runtime <= 0.0 or other_runtime <= 0.0:
                values.append(None)
                continue
            speedup = other_runtime / ours_runtime
            values.append(speedup)
            all_values.append(speedup)
        speedup_series[backend_key] = values

    if not all_values:
        return None, speedup_series

    ax_speed = ax.twinx()
    workload_order = HYBRID_WORKLOAD_ORDER if cases and cases[0].category == "hybrid_dv_cv" else PURE_WORKLOAD_ORDER
    ranges = workload_ranges(cases, workload_order)
    for backend_key, values in speedup_series.items():
        for start, end, _ in ranges:
            xs = [x_positions[index] for index in range(start, end + 1) if values[index] is not None]
            ys = [float(values[index]) for index in range(start, end + 1) if values[index] is not None]
            if not xs:
                continue
            ax_speed.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=0.85,
                color=BACKEND_COLORS[backend_key],
                marker=SPEEDUP_MARKERS.get(backend_key, "o"),
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.8,
                zorder=5,
                alpha=0.92,
            )
    ax_speed.set_yscale("log")
    ax_speed.set_ylabel("Speedup (x)")
    ax_speed.set_ylim(min(1.0, min(all_values) / 1.4), max(all_values) * 1.8)
    ax_speed.grid(False)
    return ax_speed, speedup_series


def has_rows_for_cutoffs(
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    *,
    category: str,
    workload: str,
    num_qubits: Optional[int],
    num_modes: Optional[int],
    backend_keys: Sequence[str],
    cutoffs: Sequence[int],
) -> bool:
    for backend_key in backend_keys:
        for cutoff in cutoffs:
            case = CaseKey(category, workload, num_qubits, num_modes, cutoff)
            if lookup.get((backend_key, case)) is None:
                return False
    return True


def row_status(
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    *,
    category: str,
    workload: str,
    num_qubits: Optional[int],
    num_modes: Optional[int],
    backend_key: str,
    cutoff: int,
) -> Optional[str]:
    row = lookup.get((backend_key, CaseKey(category, workload, num_qubits, num_modes, cutoff)))
    return None if row is None else str(row.get("status"))


HYBRID_CUTOFF_CASES: List[Dict[str, Any]] = [
    {"workload": "jch", "num_qubits": 10, "num_modes": 3, "kind": "qubit_dominated"},
    {"workload": "jch", "num_qubits": 4,  "num_modes": 4, "kind": "balanced"},
    {"workload": "jch", "num_qubits": 3,  "num_modes": 6, "kind": "mode_dominated"},
]


def select_hybrid_cutoff_representative_cases(
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return HYBRID_CUTOFF_CASES


def row_is_oom(row: Optional[Mapping[str, Any]]) -> bool:
    if row is None:
        return False
    status = str(row.get("status", "")).lower()
    note = str(row.get("note", "") or "").lower()
    if "allocate gpu memory" in note or "failed to allocate" in note or "oom" in note:
        return True
    if "无法分配" in str(row.get("note", "")) or "内存" in str(row.get("note", "")):
        return True
    # For baselines, if it failed and has no runtime, it's likely an OOM/resource issue we want to label
    if status == "failed" and row.get("runtime_ms") is None:
        return True
    return False


def annotate_oom_missing_bars(
    ax: plt.Axes,
    cases: Sequence[CaseKey],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    backend_keys: Sequence[str],
    geometry: Mapping[str, float],
) -> None:
    x_positions = list(range(len(cases)))
    bar_width = float(geometry["bar_width"])
    offset_start = float(geometry["offset_start"])
    base = float(geometry["base"])
    text_y = base * 1.6

    for case_index, case in enumerate(cases):
        for backend_index, backend_key in enumerate(backend_keys):
            row = lookup.get((backend_key, case))
            if row is None or row.get("status") == "ok":
                continue
            if not row_is_oom(row):
                continue
            x = x_positions[case_index] + offset_start + backend_index * bar_width
            ax.text(
                x,
                text_y,
                "OOM",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=BACKEND_COLORS[backend_key],
                fontweight="bold",
                zorder=4,
            )


def annotate_cutoff_oom(
    ax: plt.Axes,
    *,
    category: str,
    workload: str,
    num_qubits: Optional[int],
    num_modes: Optional[int],
    backend_keys: Sequence[str],
    cutoffs: Sequence[int],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
) -> None:
    for backend_index, backend_key in enumerate(backend_keys):
        y_frac = 0.05 + 0.08 * backend_index
        for cutoff in cutoffs:
            row = lookup.get((backend_key, CaseKey(category, workload, num_qubits, num_modes, cutoff)))
            if row is None or row.get("status") == "ok" or not row_is_oom(row):
                continue
            ax.text(
                cutoff,
                y_frac,
                "OOM",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=5.6,
                color=BACKEND_COLORS[backend_key],
                fontweight="bold",
                zorder=6,
            )


def padded_cutoff_xlim(
    cutoffs: Sequence[int],
    *,
    pad_fraction: float = 0.12,
    min_pad: float = 1.4,
) -> Tuple[float, float]:
    ordered = sorted(cutoffs)
    if not ordered:
        return (-0.5, 0.5)
    span = max(float(ordered[-1] - ordered[0]), 1.0)
    pad = max(min_pad, span * pad_fraction)
    return (float(ordered[0]) - pad, float(ordered[-1]) + pad)


def add_case_header(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.5,
        1.02,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=BASE_FONT_SIZE,
        fontweight="bold",
        clip_on=False,
        zorder=7,
    )


def finish_axis(
    ax: plt.Axes,
    cases: Sequence[CaseKey],
    *,
    workload_order: Sequence[str],
    show_group_labels: bool,
    show_labels: bool,
    rotation: float,
) -> None:
    x_positions = list(range(len(cases)))
    add_group_background(ax, cases, workload_order, show_group_labels=show_group_labels)
    ax.set_xlim(-0.5, len(cases) - 0.5)
    ax.set_xticks(x_positions)
    if show_labels:
        ax.set_xticklabels([case_label(case) for case in cases], rotation=rotation, ha="center")
    else:
        ax.set_xticklabels([])


def collect_oom_annotations(
    cases: Sequence[CaseKey],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    backend_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Any]] = []
    for case in cases:
        for backend_key in backend_keys:
            row = lookup.get((backend_key, case))
            if row is None or row.get("status") == "ok" or not row_is_oom(row):
                continue
            annotations.append(
                {
                    "backend_key": backend_key,
                    "backend_label": BACKEND_LABELS[backend_key],
                    "case_label": case_label(case),
                    "workload": case.workload,
                    "num_qubits": case.num_qubits,
                    "num_modes": case.num_modes,
                    "cutoff": case.cutoff,
                    "note": str(row.get("note", "") or ""),
                    "source": str(row.get("source", "") or ""),
                    "status": str(row.get("status", "") or ""),
                }
            )
    return annotations


def backend_patch_handles(backend_keys: Sequence[str]) -> List[Patch]:
    return [
        Patch(facecolor=BACKEND_COLORS[key], edgecolor="black", linewidth=0.25, label=BACKEND_LABELS[key])
        for key in backend_keys
    ]


_BREAKDOWN_GRAY = "#999999"


def runtime_breakdown_handles() -> List[Patch]:
    return [
        Patch(facecolor=_BREAKDOWN_GRAY, edgecolor="black", linewidth=0.25, alpha=0.95, hatch="////////", label="Compute"),
        Patch(facecolor=_BREAKDOWN_GRAY, edgecolor="black", linewidth=0.25, alpha=0.75, hatch="xxxxx", label="Communication"),
        Patch(facecolor=_BREAKDOWN_GRAY, edgecolor="black", linewidth=0.25, alpha=0.35, label="Other"),
    ]


def memory_breakdown_handles() -> List[Patch]:
    return [
        Patch(facecolor=_BREAKDOWN_GRAY, edgecolor="black", linewidth=0.25, alpha=0.95, label="GPU Memory"),
        Patch(facecolor=_BREAKDOWN_GRAY, edgecolor="black", linewidth=0.25, alpha=0.75, hatch="////", label="CPU Memory"),
    ]


def serialize_speedup_series(
    cases: Sequence[CaseKey],
    speedup_series: Mapping[str, Sequence[Optional[float]]],
) -> Dict[str, List[Dict[str, Any]]]:
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for backend_key, values in speedup_series.items():
        payload[BACKEND_LABELS[backend_key]] = [
            {
                "case_label": case_label(case),
                "speedup": None if value is None else float(value),
            }
            for case, value in zip(cases, values)
        ]
    return payload


def parse_scaling_full_results(
    rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    results_dir: pathlib.Path,
) -> None:
    if not results_dir.exists():
        return
    
    # Pattern: sc26_{workload}_nq{q}_nm{m}_c{c}__hybridcvdv.json or sc26_{workload}_c{c}__hybridcvdv.json
    hybrid_pattern = re.compile(r"sc26_(jch|vqe|qft|shors|transfer_(?:CVtoDV|DVtoCV))_nq(\d+)_nm(\d+)_c(\d+)__hybridcvdv\.json$")
    pure_pattern = re.compile(r"sc26_(qaoa|cat|gkp)_nm?(\d+)?_c(\d+)__hybridcvdv\.json$")
    simple_hybrid_pattern = re.compile(r"sc26_(shors|cat|gkp)_c(\d+)__hybridcvdv\.json$")

    for path in sorted(results_dir.glob("*.json")):
        case_id = path.stem.replace("__hybridcvdv", "")
        payload = load_json(path)
        results = payload.get("results") or []
        if not results:
            continue
        record = results[0]
        status = str(record.get("status", "unknown"))
        if status == "error":
            status = "failed"
        
        metrics = record.get("metrics") or {}
        # New result format uses runner_wall_time_ms or median_total_ms
        runtime_ms = metric_float(metrics, "runner_wall_time_ms", "median_total_ms")
        memory_bytes = metric_float(metrics, "gpu_peak_memory_used_mb")
        if memory_bytes is not None:
            memory_bytes *= 1024 * 1024 # Convert MB to Bytes
        
        gpu_name = str((payload.get("device") or {}).get("name") or "GPU")

        match = hybrid_pattern.match(path.name)
        if match:
            workload = match.group(1)
            num_qubits = int(match.group(2))
            num_modes = int(match.group(3))
            cutoff = int(match.group(4))
            category = "hybrid_dv_cv"
        else:
            match = pure_pattern.match(path.name)
            if match:
                workload = match.group(1)
                num_modes = int(match.group(2)) if match.group(2) else 1
                cutoff = int(match.group(3))
                num_qubits = None
                category = "pure_cv"
            else:
                match = simple_hybrid_pattern.match(path.name)
                if match:
                    workload = match.group(1)
                    cutoff = int(match.group(2))
                    num_qubits = 1 if workload == "shors" else None
                    num_modes = 3 if workload == "shors" else 1
                    category = "hybrid_dv_cv" if workload == "shors" else "pure_cv"
                else:
                    continue

        store_row(
            rows_by_key,
            make_row(
                backend_key="hybridcvdv",
                category=category,
                workload=workload,
                case_id=case_id,
                num_qubits=num_qubits,
                num_modes=num_modes,
                cutoff=cutoff,
                runtime_ms=runtime_ms,
                compute_ms=metric_float(metrics, "median_compute_ms"),
                communication_ms=metric_float(metrics, "median_transfer_ms"),
                memory_bytes=memory_bytes,
                memory_kind="peak_gpu",
                status="ok" if status in ("ok", "success") else "failed",
                source="scaling_full_20260405",
                hardware="1x {}".format(gpu_name),
                note=str(record.get("note", "") or ""),
            ),
        )


def render_hybrid_figure(
    rows: Sequence[Mapping[str, Any]],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    output_dir: pathlib.Path,
    cutoff: int,
    max_qubits: int,
) -> Tuple[List[pathlib.Path], Dict[str, Any]]:
    cases = select_hybrid_cases(rows, cutoff, max_qubits)
    missing = count_missing_cases(cases, lookup, HYBRID_BACKENDS)
    oom_annotations = collect_oom_annotations(cases, lookup, HYBRID_BACKENDS)

    apply_paper_style(width_pt=DOUBLE_COLUMN_PT, nrows=2, panel_aspect=7, font_size=BASE_FONT_SIZE)
    fig, (ax_runtime, ax_memory) = plt.subplots(2, 1, sharex=True)

    runtime_geometry = plot_runtime_bars_with_breakdown(ax_runtime, cases, lookup, HYBRID_BACKENDS, "Runtime (ms)")
    _, speedup_series = add_speedup_lines(ax_runtime, cases, lookup, HYBRID_BACKENDS)
    plot_memory_bars_with_breakdown(ax_memory, cases, lookup, HYBRID_BACKENDS, "Memory (MB)")
    annotate_oom_missing_bars(ax_runtime, cases, lookup, HYBRID_BACKENDS, runtime_geometry)

    ax_runtime.text(0.01, 0.98, "(a)", transform=ax_runtime.transAxes, ha="left", va="top", fontweight="bold")
    ax_memory.text(0.01, 0.98, "(b)", transform=ax_memory.transAxes, ha="left", va="top", fontweight="bold")

    finish_axis(
        ax_runtime,
        cases,
        workload_order=HYBRID_WORKLOAD_ORDER,
        show_group_labels=True,
        show_labels=False,
        rotation=90.0,
    )
    finish_axis(
        ax_memory,
        cases,
        workload_order=HYBRID_WORKLOAD_ORDER,
        show_group_labels=False,
        show_labels=True,
        rotation=90.0,
    )
    ax_memory.set_xlabel("Hybrid benchmark instances (cutoff = 16)", labelpad=2.0)

    # all_handles = backend_patch_handles(HYBRID_BACKENDS) + runtime_breakdown_handles() + memory_breakdown_handles()
    fig.legend(
        handles=backend_patch_handles(HYBRID_BACKENDS), title="Method",
        loc="upper center",
        bbox_to_anchor=(0.25, 1.17),
        ncol=min(len(HYBRID_BACKENDS), 2),
        handlelength=0.75,
        handleheight=0.75,
        borderpad=0.2,
        columnspacing=0.8,
    )
    fig.legend(
        handles=runtime_breakdown_handles(), title="Runtime Breakdown",
        loc="upper center",
        bbox_to_anchor=(0.55, 1.17),
        ncol=min(len(runtime_breakdown_handles()), 2),
        handlelength=0.75,
        handleheight=0.75,
        borderpad=0.2,
        columnspacing=0.8,
    )
    fig.legend(
        handles=memory_breakdown_handles(), title="Memory Breakdown",
        loc="upper center",
        bbox_to_anchor=(0.8, 1.17),
        ncol=min(len(memory_breakdown_handles()), 1),
        handlelength=0.75,
        handleheight=0.75,
        borderpad=0.2,
        columnspacing=0.8,
    )
    fig.subplots_adjust(top=0.86, bottom=0.15, hspace=0.12, right=0.91)
    figure_paths = save_figure(fig, output_dir, "sc26_hybrid_performance_overview")
    plt.close(fig)

    payload = {
        "cutoff": cutoff,
        "max_qubits": max_qubits,
        "case_count": len(cases),
        "cases": [
            {
                "label": case_label(case),
                "workload": case.workload,
                "num_qubits": case.num_qubits,
                "num_modes": case.num_modes,
                "cutoff": case.cutoff,
            }
            for case in cases
        ],
        "missing_case_counts": missing,
        "oom_annotations": oom_annotations,
        "speedup_series": serialize_speedup_series(cases, speedup_series),
        "figure_paths": [str(path) for path in figure_paths],
    }
    return figure_paths, payload


def render_pure_cv_figure(
    rows: Sequence[Mapping[str, Any]],
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    output_dir: pathlib.Path,
    cutoff: int,
) -> Tuple[List[pathlib.Path], Dict[str, Any]]:
    cases = select_pure_cv_cases(rows, cutoff)
    missing = count_missing_cases(cases, lookup, PURE_CV_BACKENDS)
    oom_annotations = collect_oom_annotations(cases, lookup, PURE_CV_BACKENDS)

    apply_paper_style(width_pt=SINGLE_COLUMN_PT, nrows=2, panel_aspect=3.75, font_size=BASE_FONT_SIZE)
    fig, (ax_runtime, ax_memory) = plt.subplots(2, 1, sharex=True)

    runtime_geometry = plot_runtime_bars_with_breakdown(ax_runtime, cases, lookup, PURE_CV_BACKENDS, "Runtime(ms)")
    _, speedup_series = add_speedup_lines(ax_runtime, cases, lookup, PURE_CV_BACKENDS)
    plot_memory_bars_with_breakdown(ax_memory, cases, lookup, PURE_CV_BACKENDS, "Memory(MB)")
    annotate_oom_missing_bars(ax_runtime, cases, lookup, PURE_CV_BACKENDS, runtime_geometry)

    ax_runtime.text(0.01, 0.98, "(a)", transform=ax_runtime.transAxes, ha="left", va="top", fontweight="bold")
    ax_memory.text(0.01, 0.98, "(b)", transform=ax_memory.transAxes, ha="left", va="top", fontweight="bold")

    finish_axis(
        ax_runtime,
        cases,
        workload_order=PURE_WORKLOAD_ORDER,
        show_group_labels=False,
        show_labels=False,
        rotation=55.0,
    )
    finish_axis(
        ax_memory,
        cases,
        workload_order=PURE_WORKLOAD_ORDER,
        show_group_labels=False,
        show_labels=True,
        rotation=0,
    )
    ax_memory.set_xlabel("Pure-CV QAOA instances", labelpad=2.0)

    fig.legend(
        handles=backend_patch_handles(HYBRID_BACKENDS), title="Method",
        loc="upper center",
        bbox_to_anchor=(0.2, 1.27),
        ncol=min(len(HYBRID_BACKENDS), 1),
        handlelength=0.75,
        handleheight=0.75,
        borderpad=0.2,
        columnspacing=0.8,
    )
    fig.legend(
        handles=runtime_breakdown_handles(), title="Runtime Breakdown",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.27),
        ncol=min(len(runtime_breakdown_handles()), 1),
        handlelength=0.75,
        handleheight=0.75,
        borderpad=0.2,
        columnspacing=0.8,
    )
    fig.legend(
        handles=memory_breakdown_handles(), title="Memory Breakdown",
        loc="upper center",
        bbox_to_anchor=(0.8, 1.27),
        ncol=min(len(memory_breakdown_handles()), 1),
        handlelength=0.75,
        handleheight=0.75,
        borderpad=0.2,
        columnspacing=0.8,
    )
    fig.subplots_adjust(top=0.84, bottom=0.20, hspace=0.15, right=0.88)
    figure_paths = save_figure(fig, output_dir, "sc26_pure_cv_performance_overview")
    plt.close(fig)

    payload = {
        "cutoff": cutoff,
        "case_count": len(cases),
        "cases": [
            {
                "label": case_label(case),
                "workload": case.workload,
                "num_qubits": case.num_qubits,
                "num_modes": case.num_modes,
                "cutoff": case.cutoff,
            }
            for case in cases
        ],
        "missing_case_counts": missing,
        "oom_annotations": oom_annotations,
        "speedup_series": serialize_speedup_series(cases, speedup_series),
        "figure_paths": [str(path) for path in figure_paths],
    }
    return figure_paths, payload


def render_hybrid_cutoff_figure(
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    output_dir: pathlib.Path,
) -> Tuple[List[pathlib.Path], Dict[str, Any]]:
    selected_cases = select_hybrid_cutoff_representative_cases(lookup)
    ncols = len(selected_cases)
    hybrid_xlim = padded_cutoff_xlim(CUTOFF_FIGURE_CUTOFFS, pad_fraction=0.14, min_pad=1.6)

    apply_paper_style(
        width_pt=SINGLE_COLUMN_PT,
        ncols=ncols,
        nrows=2,
        panel_aspect=1.35,
        font_size=BASE_FONT_SIZE,
    )
    fig, axes = plt.subplots(2, ncols, sharex=False)
    if ncols == 1:
        axes = [[axes[0]], [axes[1]]]

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    selected_payload: List[Dict[str, Any]] = []

    for column, item in enumerate(selected_cases):
        workload = str(item["workload"])
        num_qubits = int(item["num_qubits"])
        num_modes = int(item["num_modes"])
        case_header = "{}-q{}m{}".format(WORKLOAD_LABELS[workload], num_qubits, num_modes)
        runtime_ax = axes[0][column]
        memory_ax = axes[1][column]

        runtime_values_all: List[float] = []
        memory_values_all: List[float] = []
        for backend_key in HYBRID_BACKENDS:
            xs_runtime: List[int] = []
            ys_runtime: List[float] = []
            xs_memory: List[int] = []
            ys_memory: List[float] = []
            for cutoff in CUTOFF_FIGURE_CUTOFFS:
                case = CaseKey("hybrid_dv_cv", workload, num_qubits, num_modes, cutoff)
                row = lookup.get((backend_key, case))
                if row is None or row.get("status") != "ok":
                    continue
                runtime_value = optional_float(row.get("runtime_ms"))
                memory_value = optional_float(row.get("memory_bytes"))
                if runtime_value is not None and runtime_value > 0.0:
                    xs_runtime.append(cutoff)
                    ys_runtime.append(runtime_value)
                    runtime_values_all.append(runtime_value)
                if memory_value is not None and memory_value > 0.0:
                    xs_memory.append(cutoff)
                    ys_memory.append(memory_value / BYTES_PER_MIB)
                    memory_values_all.append(memory_value / BYTES_PER_MIB)
            runtime_ax.plot(
                xs_runtime,
                ys_runtime,
                color=BACKEND_COLORS[backend_key],
                marker=SPEEDUP_MARKERS.get(backend_key, "o"),
                linewidth=0.6,
                markersize=2.8,
                markeredgewidth=0.45,
            )
            memory_ax.plot(
                xs_memory,
                ys_memory,
                color=BACKEND_COLORS[backend_key],
                marker=SPEEDUP_MARKERS.get(backend_key, "o"),
                linewidth=0.6,
                markersize=2.8,
                markeredgewidth=0.45,
            )

        annotate_cutoff_oom(
            runtime_ax,
            category="hybrid_dv_cv",
            workload=workload,
            num_qubits=num_qubits,
            num_modes=num_modes,
            backend_keys=HYBRID_BACKENDS,
            cutoffs=CUTOFF_FIGURE_CUTOFFS,
            lookup=lookup,
        )

        if runtime_values_all:
            runtime_ax.set_yscale("log")
            runtime_ax.set_ylim(min(runtime_values_all) / 2.5, max(runtime_values_all) * 3.0)
        if memory_values_all:
            memory_ax.set_yscale("log")
            memory_ax.set_ylim(min(memory_values_all) / 2.5, max(memory_values_all) * 3.0)
        runtime_ax.text(0.02, 0.96, panel_labels[column], transform=runtime_ax.transAxes, ha="left", va="top", fontweight="bold")
        add_case_header(runtime_ax, case_header)
        memory_ax.text(0.02, 0.96, panel_labels[ncols + column], transform=memory_ax.transAxes, ha="left", va="top", fontweight="bold")
        runtime_ax.set_xticks(CUTOFF_FIGURE_CUTOFFS)
        runtime_ax.set_xlim(*hybrid_xlim)
        memory_ax.set_xticks(CUTOFF_FIGURE_CUTOFFS)
        memory_ax.set_xlim(*hybrid_xlim)
        memory_ax.set_xlabel("Cutoff", labelpad=1.2)
        if column == 0:
            runtime_ax.set_ylabel("Runtime (ms)")
            memory_ax.set_ylabel("Memory (MB)")
        selected_payload.append(
            {
                "label": case_header,
                "workload": workload,
                "num_qubits": num_qubits,
                "num_modes": num_modes,
                "kind": str(item.get("kind", "")),
                "cutoffs": list(CUTOFF_FIGURE_CUTOFFS),
            }
        )

    fig.legend(
        handles=[
            Line2D([], [], color=BACKEND_COLORS[key], marker=SPEEDUP_MARKERS.get(key, "o"), linewidth=0.6, markersize=2.8, label=BACKEND_LABELS[key])
            for key in HYBRID_BACKENDS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        columnspacing=0.6,
        handlelength=1.2,
        borderpad=0.15,
    )
    fig.subplots_adjust(top=0.85, bottom=0.10, left=0.14, right=0.97, wspace=0.30, hspace=0.35)
    figure_paths = save_figure(fig, output_dir, "sc26_hybrid_cutoff_sensitivity")
    plt.close(fig)

    payload = {
        "selected_cases": selected_payload,
        "figure_paths": [str(path) for path in figure_paths],
    }
    return figure_paths, payload


def render_pure_cv_cutoff_figure(
    lookup: Mapping[Tuple[str, CaseKey], Mapping[str, Any]],
    output_dir: pathlib.Path,
) -> Tuple[List[pathlib.Path], Dict[str, Any]]:
    workload, num_modes = PURE_CUTOFF_CASE
    cutoffs = [4, 8, 12, 16, 24, 32]
    pure_xlim = padded_cutoff_xlim(cutoffs, pad_fraction=0.14, min_pad=2.2)
    case_header = "{}-m{}".format(WORKLOAD_LABELS[workload], num_modes)

    apply_paper_style(width_pt=SINGLE_COLUMN_PT, ncols=1, nrows=2, panel_aspect=3.525, font_size=BASE_FONT_SIZE)
    fig, (ax_runtime, ax_memory) = plt.subplots(2, 1, sharex=True)

    line_backends = ["hybridcvdv"]
    reference_backends = ["strawberryfields_tf", "mrmustard_jax"]

    runtime_values_all: List[float] = []
    memory_values_all: List[float] = []
    for backend_key in line_backends:
        xs_runtime: List[int] = []
        ys_runtime: List[float] = []
        xs_memory: List[int] = []
        ys_memory: List[float] = []
        for cutoff in cutoffs:
            row = lookup.get((backend_key, CaseKey("pure_cv", workload, None, num_modes, cutoff)))
            if row is None or row.get("status") != "ok":
                continue
            runtime_value = optional_float(row.get("runtime_ms"))
            memory_value = optional_float(row.get("memory_bytes"))
            if runtime_value is not None and runtime_value > 0.0:
                xs_runtime.append(cutoff)
                ys_runtime.append(runtime_value)
                runtime_values_all.append(runtime_value)
            if memory_value is not None and memory_value > 0.0:
                xs_memory.append(cutoff)
                ys_memory.append(memory_value / BYTES_PER_MIB)
                memory_values_all.append(memory_value / BYTES_PER_MIB)
        ax_runtime.plot(xs_runtime, ys_runtime, color=BACKEND_COLORS[backend_key], marker="o", linewidth=0.4, markersize=2.0, markeredgewidth=0.45)
        ax_memory.plot(xs_memory, ys_memory, color=BACKEND_COLORS[backend_key], marker="o", linewidth=0.4, markersize=2.0, markeredgewidth=0.45)

    for backend_key in reference_backends:
        row = lookup.get((backend_key, CaseKey("pure_cv", workload, None, num_modes, 16)))
        if row is None or row.get("status") != "ok":
            continue
        runtime_value = optional_float(row.get("runtime_ms"))
        memory_value = optional_float(row.get("memory_bytes"))
        if runtime_value is not None and runtime_value > 0.0:
            ax_runtime.plot([16], [runtime_value], linestyle="None", marker=SPEEDUP_MARKERS.get(backend_key, "D"), markersize=2.0, color=BACKEND_COLORS[backend_key], markeredgewidth=0.45)
            runtime_values_all.append(runtime_value)
        if memory_value is not None and memory_value > 0.0:
            ax_memory.plot([16], [memory_value / BYTES_PER_MIB], linestyle="None", marker=SPEEDUP_MARKERS.get(backend_key, "D"), markersize=2.0, color=BACKEND_COLORS[backend_key], markeredgewidth=0.45)
            memory_values_all.append(memory_value / BYTES_PER_MIB)

    if runtime_values_all:
        ax_runtime.set_yscale("log")
        ax_runtime.set_ylim(min(runtime_values_all) / 2.0, max(runtime_values_all) * 2.0)
    if memory_values_all:
        ax_memory.set_yscale("log")
        ax_memory.set_ylim(min(memory_values_all) / 2.0, max(memory_values_all) * 2.0)
    ax_runtime.set_ylabel("Runtime (ms)")
    ax_memory.set_ylabel("Memory (MB)")
    ax_runtime.text(0.01, 0.98, "(a)", transform=ax_runtime.transAxes, ha="left", va="top", fontweight="bold")
    add_case_header(ax_runtime, case_header)
    ax_memory.text(0.01, 0.98, "(b)", transform=ax_memory.transAxes, ha="left", va="top", fontweight="bold")
    ax_memory.set_xlabel("Cutoff", labelpad=1.2)
    ax_memory.set_xticks(cutoffs)
    ax_runtime.set_xlim(*pure_xlim)
    ax_memory.set_xlim(*pure_xlim)

    fig.legend(
        handles=[
            Line2D([], [], color=BACKEND_COLORS["hybridcvdv"], marker="o", linewidth=1.8, markersize=6.0, label=BACKEND_LABELS["hybridcvdv"]),
            Line2D([], [], color=BACKEND_COLORS["strawberryfields_tf"], marker=SPEEDUP_MARKERS.get("strawberryfields_tf", "^"), linewidth=0.0, markersize=6.2, label=BACKEND_LABELS["strawberryfields_tf"]),
            Line2D([], [], color=BACKEND_COLORS["mrmustard_jax"], marker=SPEEDUP_MARKERS.get("mrmustard_jax", "D"), linewidth=0.0, markersize=6.2, label=BACKEND_LABELS["mrmustard_jax"]),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.4,
        borderpad=0.2,
    )
    fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.16)
    # figure_paths = save_figure(fig, output_dir, "sc26_pure_cv_cutoff_sensitivity")
    # plt.close(fig)

    # payload = {
    #     "selected_case": {
    #         "label": case_header,
    #         "workload": workload,
    #         "num_modes": num_modes,
    #         "cutoffs": cutoffs,
    #     },
    #     "figure_paths": [str(path) for path in figure_paths],
    # }
    return [], {}


def write_payload(output_dir: pathlib.Path, payload: Mapping[str, Any]) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sc26_performance_overview_derived.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-results-dir", type=pathlib.Path, default=DEFAULT_BENCHMARK_RESULTS_DIR)
    parser.add_argument("--bqsim-csv", type=pathlib.Path, default=DEFAULT_BQSIM_CSV)
    parser.add_argument("--bosonic-csv", type=pathlib.Path, default=DEFAULT_BOSONIC_CSV)
    parser.add_argument("--atlas-csv", type=pathlib.Path, default=DEFAULT_ATLAS_CSV)
    parser.add_argument("--remote-h100-dir", type=pathlib.Path, default=DEFAULT_REMOTE_H100_PURE_DIR)
    parser.add_argument("--cutoff", type=int, default=STUDY_CUTOFF)
    parser.add_argument("--max-hybrid-qubits", type=int, default=MAX_HYBRID_QUBITS)
    args = parser.parse_args()

    rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in read_existing_rows(args.csv):
        store_row(rows_by_key, row)
    parse_extra_our_benchmark_results(rows_by_key, args.benchmark_results_dir)
    parse_bqsim_csv(rows_by_key, args.bqsim_csv)
    parse_bosonic_csv(rows_by_key, args.bosonic_csv)
    parse_atlas_csv(rows_by_key, args.atlas_csv)
    parse_remote_reference_results(rows_by_key, args.remote_h100_dir)
    enrich_rows_with_json_breakdown(rows_by_key)

    rows = sorted_rows(rows_by_key)
    write_csv(rows, args.csv)
    lookup = row_lookup(rows)

    hybrid_paths, hybrid_payload = render_hybrid_figure(rows, lookup, args.output_dir, args.cutoff, args.max_hybrid_qubits)
    pure_paths, pure_payload = render_pure_cv_figure(rows, lookup, args.output_dir, args.cutoff)
    cutoff_paths, cutoff_payload = render_hybrid_cutoff_figure(lookup, args.output_dir)
    pure_cutoff_paths, pure_cutoff_payload = render_pure_cv_cutoff_figure(lookup, args.output_dir)

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv_path": str(args.csv),
        "hybrid_overview": hybrid_payload,
        "pure_cv_overview": pure_payload,
        "hybrid_cutoff_sensitivity": cutoff_payload,
        "pure_cv_cutoff_sensitivity": pure_cutoff_payload,
        "all_figure_paths": [str(path) for path in hybrid_paths + pure_paths + cutoff_paths + pure_cutoff_paths],
        "absent_data_notes": {
            "original_csv_missing_workloads": ["qft", "shors", "transfer_cvtodv", "transfer_dvtocv"],
            "bosonic_gpu_has_no_shors_rows": True,
            "pure_cv_has_no_multi_cutoff_case_shared_by_gantry_bqsim_bosonic_gpu": True,
            "pure_cv_cutoff_case_uses_gantry_sweep_with_c16_sf_tf_and_mrmustard_references": True,
        },
    }
    derived_path = write_payload(args.output_dir, payload)

    print("Wrote refreshed CSV to {}".format(args.csv))
    print("Wrote figures to {}".format(", ".join(str(path) for path in hybrid_paths + pure_paths + cutoff_paths + pure_cutoff_paths)))
    print("Wrote derived metrics to {}".format(derived_path))


if __name__ == "__main__":
    main()
