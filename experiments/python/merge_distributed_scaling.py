#!/usr/bin/env python3
"""Validate and merge distributed scaling artifacts."""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
from collections import defaultdict
from typing import Any

from distributed_common import atomic_write_json, read_json, validate_result


def nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def merge(run_root: pathlib.Path, output_dir: pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/*/g*/*/r*/result.json")):
        payload = read_json(path)
        errors = validate_result(payload)
        payload["_artifact_path"] = str(path)
        if errors:
            payload["status"] = "configuration_error"
            payload["schema_errors"] = errors
        results.append(payload)
    checksums: dict[tuple[str, str, str], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for result in results:
        checksum = nested(result, "correctness", "checksum")
        if result["status"] == "ok" and isinstance(checksum, (int, float)):
            checksums[(result["system"], result["phase"], result["case_name"])].append(
                (result, float(checksum))
            )
    for grouped_results in checksums.values():
        reference = grouped_results[0][1]
        tolerance = 1e-8 * max(1.0, abs(reference))
        for result, checksum in grouped_results[1:]:
            if abs(checksum - reference) > tolerance:
                result["status"] = "incorrect_result"
                result.setdefault("correctness", {})["checksum_reference"] = reference

    for result in results:
        if result["status"] != "ok":
            failures.append(
                {
                    "system": result.get("system"),
                    "phase": result.get("phase"),
                    "case_name": result.get("case_name"),
                    "gpu_count": result.get("gpu_count"),
                    "status": result.get("status"),
                    "path": result["_artifact_path"],
                }
            )

    strong_times: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for result in results:
        value = nested(result, "timing", "simulation_ms")
        eligible = nested(result, "diagnostics", "gpu_scaling_eligible")
        if (
            result["status"] == "ok"
            and result["phase"] == "strong"
            and eligible is not False
            and isinstance(value, (int, float))
        ):
            strong_times[(result["system"], result["case_name"], result["gpu_count"])].append(float(value))
    medians = {key: statistics.median(values) for key, values in strong_times.items()}
    for result in results:
        if result["phase"] != "strong" or result["status"] != "ok":
            continue
        if nested(result, "diagnostics", "gpu_scaling_eligible") is False:
            continue
        key = (result["system"], result["case_name"], result["gpu_count"])
        baseline = medians.get((result["system"], result["case_name"], 1))
        current = medians.get(key)
        if baseline and current:
            result.setdefault("scaling", {})["speedup"] = baseline / current
            result["scaling"]["parallel_efficiency"] = baseline / current / result["gpu_count"]

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "3.0",
            "artifact_count": len(results),
            "ok_count": len(results) - len(failures),
            "failure_count": len(failures),
            "artifacts": results,
        },
    )
    atomic_write_json(output_dir / "failures.json", {"failures": failures})
    rows = []
    for result in results:
        rows.append(
            {
                "system": result["system"],
                "phase": result["phase"],
                "case_name": result["case_name"],
                "gpu_count": result["gpu_count"],
                "repetition": result.get("process_repetition"),
                "status": result["status"],
                "simulation_ms": nested(result, "timing", "simulation_ms"),
                "compute_ms": nested(result, "timing", "gpu_compute_ms"),
                "host_orchestration_ms": nested(result, "timing", "host_orchestration_ms"),
                "host_fraction": nested(result, "diagnostics", "host_fraction"),
                "scaling_role": nested(result, "diagnostics", "scaling_role"),
                "gpu_scaling_eligible": nested(
                    result, "diagnostics", "gpu_scaling_eligible"
                ),
                "communication_ms": nested(result, "timing", "communication_ms"),
                "throughput_circuit_s": nested(result, "throughput", "circuit_evaluations_per_sec"),
                "speedup": nested(result, "scaling", "speedup"),
                "parallel_efficiency": nested(result, "scaling", "parallel_efficiency"),
                "p2p_bytes": nested(result, "communication", "p2p_bytes"),
                "host_staged_bytes": nested(result, "communication", "host_staged_bytes"),
                "checksum": nested(result, "correctness", "checksum"),
                "artifact_path": result["_artifact_path"],
            }
        )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["system"])
        writer.writeheader()
        writer.writerows(rows)
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_root / "merged"
    results, failures = merge(args.run_root, output)
    print(f"Merged {len(results)} artifacts ({len(failures)} non-ok) into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
