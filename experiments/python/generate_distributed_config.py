#!/usr/bin/env python3
"""Generate the complete, deduplicated distributed SC26 case matrix."""

from __future__ import annotations

import argparse
import copy
import pathlib
import re
from typing import Any

from distributed_common import atomic_write_json, iso_now, read_json


CUTOFF_TOKEN = re.compile(r"_c\d+(?=_|$)")


def parse_cutoffs(raw: str) -> list[int]:
    values = sorted({int(value) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0 or value & (value - 1) for value in values):
        raise ValueError("cutoffs must be positive powers of two")
    return values


def with_cutoff_name(name: str, cutoff: int) -> str:
    if CUTOFF_TOKEN.search(name):
        return CUTOFF_TOKEN.sub(f"_c{cutoff}", name, count=1)
    return f"{name}_c{cutoff}"


def template_key(case: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    ignored = {
        "name",
        "cutoff",
        "internal_name_filter",
        "warmup_runs",
        "measured_runs",
        "backends",
        "systems",
    }
    return tuple(
        sorted(
            (key, repr(value))
            for key, value in case.items()
            if key not in ignored
        )
    )


def generate(source: dict[str, Any], cutoffs: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_cases = source.get("cases")
    if not isinstance(source_cases, list) or not source_cases:
        raise ValueError("source config must contain a non-empty cases list")

    templates: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    existing_by_template: dict[tuple[tuple[str, str], ...], dict[int, dict[str, Any]]] = {}
    for index, raw_case in enumerate(source_cases):
        if not isinstance(raw_case, dict) or not raw_case.get("name"):
            raise ValueError(f"invalid source case #{index}")
        if not isinstance(raw_case.get("cutoff"), int):
            raise ValueError(f"case {raw_case['name']} has no integer cutoff")
        key = template_key(raw_case)
        templates.setdefault(key, raw_case)
        existing_by_template.setdefault(key, {})[raw_case["cutoff"]] = raw_case

    output_cases: list[dict[str, Any]] = []
    derived_count = 0
    for key, template in templates.items():
        existing = existing_by_template[key]
        for cutoff in cutoffs:
            if cutoff in existing:
                case = copy.deepcopy(existing[cutoff])
            else:
                case = copy.deepcopy(template)
                case["name"] = with_cutoff_name(str(template["name"]), cutoff)
                case["cutoff"] = cutoff
                case["internal_name_filter"] = case["name"]
                derived_count += 1
            case.setdefault("internal_name_filter", case["name"])
            case["backends"] = ["hybridcvdv"]
            case["systems"] = ["hybridcvdv", "atlas", "bqsim"]
            output_cases.append(case)

    output_cases.sort(key=lambda case: case["name"])
    names = [case["name"] for case in output_cases]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"derived config contains duplicate names: {duplicates[:5]}")

    config = {
        "schema_version": "3.0",
        "generated_at_utc": iso_now(),
        "source_schema_version": source.get("schema_version"),
        "telemetry_interval_ms": 100,
        "cutoffs": cutoffs,
        "cases": output_cases,
    }
    summary = {
        "source_case_count": len(source_cases),
        "template_count": len(templates),
        "derived_case_count": derived_count,
        "output_case_count": len(output_cases),
        "cutoffs": cutoffs,
    }
    return config, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cutoffs", default="4,8,16,32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    source_path = pathlib.Path(args.source)
    output_path = pathlib.Path(args.output)
    config, summary = generate(read_json(source_path), parse_cutoffs(args.cutoffs))
    atomic_write_json(output_path, config)
    summary_path = pathlib.Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    atomic_write_json(summary_path, summary)
    print(f"Wrote {summary['output_case_count']} cases to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
