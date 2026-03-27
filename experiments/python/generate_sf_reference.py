#!/usr/bin/env python3
"""
Generate reference state vectors from Strawberry Fields TF backend.

Applies individual CV gates to vacuum state and dumps the resulting
Fock-space state vectors as text files. These are used by the C++ test
`test_sf_precision.cpp` to cross-validate our GPU implementation.

Usage:
    python generate_sf_reference.py --cutoff 16 --output-dir ../sf_reference_data
    python generate_sf_reference.py --cutoff 32 --output-dir ../sf_reference_data --gpu
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import strawberryfields as sf
    from strawberryfields import ops
except ImportError:
    print("ERROR: strawberryfields not installed. Run: pip install strawberryfields tensorflow")
    sys.exit(1)

try:
    import tensorflow as _tf
except ImportError:
    print("ERROR: tensorflow not installed. Run: pip install tensorflow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Gate test case definitions
# ---------------------------------------------------------------------------

@dataclass
class GateTestCase:
    """One gate applied to vacuum, producing a reference state vector."""
    name: str               # e.g. "displacement_r0.5_phi0.3"
    num_modes: int
    gate_fn: Any            # callable(registers) -> None
    description: str = ""
    params: dict = field(default_factory=dict)


def build_single_mode_cases() -> list[GateTestCase]:
    """Single-mode CV gates applied to vacuum |0>."""
    cases = []

    # --- Displacement D(alpha) with alpha = r * exp(i*phi) ---
    for r, phi in [(0.3, 0.0), (0.5, 0.0), (1.0, 0.0),
                   (0.3, 0.5), (0.5, 1.2), (0.1, 3.14)]:
        name = f"displacement_r{r}_phi{phi:.2f}"
        def make_fn(r_=r, phi_=phi):
            return lambda regs: ops.Dgate(r_, phi_) | regs[0]
        cases.append(GateTestCase(
            name=name, num_modes=1, gate_fn=make_fn(),
            description=f"Displacement D(r={r}, phi={phi})",
            params={"gate": "DISPLACEMENT", "r": r, "phi": phi}
        ))

    # --- Squeezing S(r, phi) ---
    for r, phi in [(0.2, 0.0), (0.5, 0.0), (0.8, 0.0),
                   (0.3, 0.7), (0.5, 1.5)]:
        name = f"squeezing_r{r}_phi{phi:.2f}"
        def make_fn(r_=r, phi_=phi):
            return lambda regs: ops.Sgate(r_, phi_) | regs[0]
        cases.append(GateTestCase(
            name=name, num_modes=1, gate_fn=make_fn(),
            description=f"Squeezing S(r={r}, phi={phi})",
            params={"gate": "SQUEEZING", "r": r, "phi": phi}
        ))

    # --- Phase Rotation R(theta) ---
    for theta in [0.3, 0.7, 1.5, 3.14159]:
        name = f"phase_rotation_theta{theta:.4f}"
        def make_fn(t=theta):
            return lambda regs: ops.Rgate(t) | regs[0]
        cases.append(GateTestCase(
            name=name, num_modes=1, gate_fn=make_fn(),
            description=f"Phase rotation R(theta={theta})",
            params={"gate": "PHASE_ROTATION", "theta": theta}
        ))

    # --- Kerr gate K(kappa) ---
    for kappa in [0.1, 0.3, 0.5, 1.0]:
        name = f"kerr_kappa{kappa}"
        def make_fn(k=kappa):
            return lambda regs: ops.Kgate(k) | regs[0]
        cases.append(GateTestCase(
            name=name, num_modes=1, gate_fn=make_fn(),
            description=f"Kerr K(kappa={kappa})",
            params={"gate": "KERR_GATE", "kappa": kappa}
        ))

    # --- Composed: Displacement then Kerr (tests non-trivial state) ---
    for d_r, kappa in [(0.5, 0.3), (1.0, 0.1)]:
        name = f"displace_then_kerr_r{d_r}_k{kappa}"
        def make_fn(dr=d_r, k=kappa):
            def fn(regs):
                ops.Dgate(dr, 0.0) | regs[0]
                ops.Kgate(k) | regs[0]
            return fn
        cases.append(GateTestCase(
            name=name, num_modes=1, gate_fn=make_fn(),
            description=f"D(r={d_r}) then K(kappa={kappa})",
            params={"gate": "COMPOSED_D_K", "d_r": d_r, "kappa": kappa}
        ))

    # --- Composed: Squeezing then Displacement ---
    for s_r, d_r in [(0.3, 0.5), (0.5, 0.3)]:
        name = f"squeeze_then_displace_sr{s_r}_dr{d_r}"
        def make_fn(sr=s_r, dr=d_r):
            def fn(regs):
                ops.Sgate(sr, 0.0) | regs[0]
                ops.Dgate(dr, 0.0) | regs[0]
            return fn
        cases.append(GateTestCase(
            name=name, num_modes=1, gate_fn=make_fn(),
            description=f"S(r={s_r}) then D(r={d_r})",
            params={"gate": "COMPOSED_S_D", "s_r": s_r, "d_r": d_r}
        ))

    return cases


def build_two_mode_cases() -> list[GateTestCase]:
    """Two-mode CV gates applied to various input states."""
    cases = []

    # --- Beam Splitter BS(theta, phi) on displaced vacuum ---
    for theta, phi in [(0.3, 0.0), (0.7854, 0.0), (0.5, 0.3)]:
        name = f"beamsplitter_theta{theta:.4f}_phi{phi:.2f}"
        def make_fn(t=theta, p=phi):
            def fn(regs):
                ops.Dgate(0.5, 0.0) | regs[0]
                ops.BSgate(t, p) | (regs[0], regs[1])
            return fn
        cases.append(GateTestCase(
            name=name, num_modes=2, gate_fn=make_fn(),
            description=f"D(0.5)|0> then BS(theta={theta}, phi={phi})",
            params={"gate": "BEAM_SPLITTER", "theta": theta, "phi": phi}
        ))

    # --- Cross-Kerr CK(kappa) on two displaced modes ---
    for kappa in [0.1, 0.3]:
        name = f"cross_kerr_kappa{kappa}"
        def make_fn(k=kappa):
            def fn(regs):
                ops.Dgate(0.3, 0.0) | regs[0]
                ops.Dgate(0.4, 0.0) | regs[1]
                ops.CKgate(k) | (regs[0], regs[1])
            return fn
        cases.append(GateTestCase(
            name=name, num_modes=2, gate_fn=make_fn(),
            description=f"D|0>D|0> then CK(kappa={kappa})",
            params={"gate": "CROSS_KERR_GATE", "kappa": kappa}
        ))

    # --- Beam Splitter on squeezed+displaced ---
    name = "bs_on_squeezed_displaced"
    def make_bs_sq(regs):
        ops.Sgate(0.3, 0.0) | regs[0]
        ops.Dgate(0.4, 0.0) | regs[1]
        ops.BSgate(0.5, 0.2) | (regs[0], regs[1])
    cases.append(GateTestCase(
        name=name, num_modes=2, gate_fn=make_bs_sq,
        description="S|0> + D|0> then BS",
        params={"gate": "BEAM_SPLITTER_COMPOSED", "s_r": 0.3, "d_r": 0.4,
                "theta": 0.5, "phi": 0.2}
    ))

    return cases


# ---------------------------------------------------------------------------
# State vector I/O
# ---------------------------------------------------------------------------

def save_state_vector(filepath: str, state: np.ndarray, metadata: dict):
    """Save state vector in simple key=value header format readable by C++.

    Format:
        # key=value header lines (one per metadata field)
        # blank line separator
        # N (element count)
        # real imag  (one per element)
    """
    flat = state.flatten()
    with open(filepath, 'w') as f:
        for k, v in metadata.items():
            f.write(f"# {k}={v}\n")
        f.write("\n")
        f.write(f"{len(flat)}\n")
        for c in flat:
            f.write(f"{c.real:.17e} {c.imag:.17e}\n")


def run_sf_gate(case: GateTestCase, cutoff: int, use_gpu: bool) -> np.ndarray:
    """Run a single gate test case through SF TF backend, return ket."""
    prog = sf.Program(case.num_modes)
    with prog.context as registers:
        case.gate_fn(registers)

    backend_opts = {"cutoff_dim": cutoff}
    engine = sf.Engine("tf", backend_options=backend_opts)
    result = engine.run(prog)
    ket = result.state.ket()
    state = np.asarray(ket.numpy() if hasattr(ket, "numpy") else ket)
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SF reference state vectors for precision comparison")
    parser.add_argument("--cutoff", type=int, default=16,
                        help="Fock space cutoff dimension (default: 16)")
    parser.add_argument("--output-dir", type=str,
                        default=str(pathlib.Path(__file__).resolve().parent.parent / "sf_reference_data"),
                        help="Output directory for reference data")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU for TF backend")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Configure TF GPU
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cases = build_single_mode_cases() + build_two_mode_cases()
    manifest = {"cutoff": args.cutoff, "cases": []}

    print(f"Generating {len(cases)} SF reference state vectors (cutoff={args.cutoff})...")
    print(f"Output directory: {args.output_dir}")
    print()

    for i, case in enumerate(cases):
        filename = f"sf_ref_{case.name}_c{args.cutoff}.txt"
        filepath = os.path.join(args.output_dir, filename)

        t0 = time.perf_counter()
        try:
            state = run_sf_gate(case, args.cutoff, args.gpu)
            elapsed = time.perf_counter() - t0
            norm = float(np.linalg.norm(state.flatten()))

            metadata = {
                "name": case.name,
                "description": case.description,
                "num_modes": case.num_modes,
                "cutoff": args.cutoff,
                "norm": norm,
                **case.params
            }
            save_state_vector(filepath, state, metadata)

            manifest["cases"].append({
                "name": case.name,
                "file": filename,
                "num_modes": case.num_modes,
                "norm": norm,
                "elapsed_ms": elapsed * 1000,
                **case.params
            })

            status = "OK" if abs(norm - 1.0) < 1e-4 else f"WARN norm={norm:.6f}"
            print(f"  [{i+1}/{len(cases)}] {case.name}: {status} ({elapsed*1000:.1f}ms)")

        except Exception as e:
            print(f"  [{i+1}/{len(cases)}] {case.name}: FAILED - {e}")
            manifest["cases"].append({
                "name": case.name,
                "file": None,
                "error": str(e),
                **case.params
            })

    # Save manifest
    manifest_path = os.path.join(args.output_dir, f"manifest_c{args.cutoff}.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    ok_count = sum(1 for c in manifest["cases"] if c.get("file") is not None)
    print(f"\nDone: {ok_count}/{len(cases)} succeeded. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
