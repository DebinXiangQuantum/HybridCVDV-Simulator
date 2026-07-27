#!/usr/bin/env python3
"""Capture reproducibility metadata for a distributed GPU experiment."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from distributed_common import REPO_ROOT, atomic_write_json, command_output, iso_now


COMMANDS: dict[str, list[str]] = {
    "git_commit": ["git", "rev-parse", "HEAD"],
    "git_status": ["git", "status", "--short"],
    "gpu_inventory": [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,power.limit",
        "--format=csv",
    ],
    "gpu_processes": [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv",
    ],
    "gpu_topology": ["nvidia-smi", "topo", "-m"],
    "nvcc_version": ["nvcc", "--version"],
    "cmake_version": ["cmake", "--version"],
    "mpi_version": ["mpirun", "--version"],
    "cpu": ["lscpu"],
    "numa": ["numactl", "--hardware"],
}


def collect(output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"generated_at_utc": iso_now(), "commands": {}}
    for name, command in COMMANDS.items():
        result = command_output(command)
        metadata["commands"][name] = {"command": command, **result}
        suffix = "txt"
        (output_dir / f"{name}.{suffix}").write_text(
            str(result.get("output", result.get("error", ""))) + "\n",
            encoding="utf-8",
        )
    atomic_write_json(output_dir / "environment.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metadata = collect(pathlib.Path(args.output_dir))
    failures = [name for name, value in metadata["commands"].items() if not value["ok"]]
    if failures:
        print("Metadata collection completed with unavailable commands: " + ", ".join(failures))
    else:
        print(f"Wrote environment metadata to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
