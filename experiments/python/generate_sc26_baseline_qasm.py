#!/usr/bin/env python3
"""Generate one canonical encoded-QASM corpus shared by ATLAS and BQSim."""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import random
from typing import Any

from distributed_common import atomic_write_json, read_json


SUPPORTED_CUTOFFS = {4, 8, 16, 32}


def encoded_qubit_count(case: dict[str, Any]) -> int:
    cutoff = int(case["cutoff"])
    if cutoff not in SUPPORTED_CUTOFFS:
        raise ValueError(f"unsupported cutoff {cutoff} for {case['name']}")
    return int(case.get("num_qubits", 0)) + int(case.get("num_modes", 0)) * math.ceil(math.log2(cutoff))


def gate_lines(case: dict[str, Any], qubits: int) -> list[str]:
    if qubits <= 0:
        raise ValueError(f"case {case['name']} encodes to zero qubits")
    workload = str(case.get("workload", "")).lower()
    depth = max(1, int(case.get("layers", case.get("timesteps", case.get("depth", 1)))))
    seed = int(hashlib.sha256(case["name"].encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    lines: list[str] = []

    for layer in range(depth):
        for qubit in range(qubits):
            angle = rng.uniform(-math.pi, math.pi)
            if "qft" in workload:
                lines.append(f"h q[{qubit}];")
                if qubit + 1 < qubits:
                    lines.append(f"rz({angle:.17g}) q[{qubit + 1}];")
                    lines.append(f"cx q[{qubit}],q[{qubit + 1}];")
            elif "transfer" in workload:
                lines.append(f"h q[{qubit}];")
                if qubit + 1 < qubits:
                    lines.append(f"cx q[{qubit}],q[{qubit + 1}];")
            elif "qaoa" in workload:
                lines.append(f"ry({angle:.17g}) q[{qubit}];")
                lines.append(f"rz({-angle / 2.0:.17g}) q[{qubit}];")
            else:
                lines.append(f"ry({angle:.17g}) q[{qubit}];")
                lines.append(f"rz({angle / 2.0:.17g}) q[{qubit}];")
        for qubit in range(max(0, qubits - 1)):
            lines.append(f"cx q[{qubit}],q[{qubit + 1}];")
        if qubits > 2 and layer % 2 == 1:
            lines.append(f"cx q[{qubits - 1}],q[0];")
    return lines


def render_qasm(case: dict[str, Any]) -> tuple[str, int, int]:
    qubits = encoded_qubit_count(case)
    gates = gate_lines(case, qubits)
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"// canonical encoded-QASM surrogate for {case['name']}",
        f"qreg q[{qubits}];",
        *gates,
        "",
    ]
    return "\n".join(lines), qubits, len(gates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = read_json(config_path).get("cases", [])
    if not cases:
        raise ValueError("config has no cases")

    artifacts = []
    for case in cases:
        qasm, qubits, gates = render_qasm(case)
        output_path = output_dir / f"{case['name']}.qasm"
        output_path.write_text(qasm, encoding="utf-8")
        artifacts.append(
            {
                "case_name": case["name"],
                "path": str(output_path.resolve()),
                "sha256": hashlib.sha256(qasm.encode()).hexdigest(),
                "encoded_qubits": qubits,
                "gate_count": gates,
                "semantic_model": "encoded_qasm_surrogate",
            }
        )

    manifest_path = pathlib.Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "config_path": str(config_path.resolve()),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )
    print(f"Wrote {len(artifacts)} canonical QASM files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
