# HybridCVDV-Simulator

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CUDA](https://img.shields.io/badge/CUDA-11.x%2F12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.18%2B-brightgreen.svg)](https://cmake.org/)



HybridCVDV-Simulator is a C++/CUDA research codebase for simulating hybrid discrete-variable and continuous-variable quantum circuits. The implementation follows the Hybrid Tensor-DD architecture described in `docs/architecture.md`: the CPU manages discrete branching with a decision diagram, while the GPU executes the continuous-variable payload with specialized kernels and execution paths.

## Overview

The current repository focuses on three layers:

- **Hybrid circuit runtime**: a `QuantumCircuit` API for qubits, qumodes, and hybrid gates.
- **Execution backends**: exact Fock-space kernels, Gaussian symbolic updates, and diagonal non-Gaussian mixture approximations.
- **Experiment tooling**: single-GPU benchmark drivers, baseline runners, plotting inputs, and SC26 evaluation assets.

## Current capabilities

- **Control/physical separation** through a CPU-resident HDD plus GPU-resident CV state storage.
- **Dual-track CV execution** with symbolic Gaussian evolution and exact tensor-based execution for non-Gaussian paths.
- **Gate coverage across DV, CV, and hybrid layers**, including:
  - qubit gates such as Hadamard, Pauli, rotations, CNOT, and CZ
  - CV gates such as phase rotation, Kerr, SNAP, cross-Kerr, displacement, squeezing, beam splitter, creation, and annihilation
  - hybrid gates such as conditional displacement/squeezing/beam splitter, Jaynes-Cummings, anti-Jaynes-Cummings, Rabi interaction, and selective qubit rotation
- **Execution utilities** such as checkpointing, execution-block compilation, batching, instruction fusion, and CUDA graph-based scheduling.
- **Benchmark infrastructure** under `experiments/` for scaling runs, baseline comparison, and GPU telemetry collection.
- **Experimental noisy-simulation sandbox** under `src/noisy/`, kept intentionally separate from the default runtime while the API is still evolving.

## Supported gates

The current public gate constructors are exposed through the `Gates::...` helpers in `include/quantum_circuit.h`.

| Category | Gate helpers currently exposed |
| --- | --- |
| **Qubit gates** | `Hadamard`, `PauliX`, `PauliY`, `PauliZ`, `RotationX`, `RotationY`, `RotationZ`, `PhaseGateS`, `PhaseGateT`, `CNOT`, `CZ` |
| **Pure CV gates** | `PhaseRotation`, `KerrGate`, `ConditionalParity`, `Snap`, `MultiSNAP`, `CrossKerr`, `CreationOperator`, `AnnihilationOperator`, `Displacement`, `Squeezing`, `BeamSplitter` |
| **Hybrid controlled gates** | `ConditionalDisplacement`, `ConditionalSqueezing`, `ConditionalBeamSplitter`, `ConditionalTwoModeSqueezing`, `ConditionalSUM` |
| **Hybrid interaction gates** | `RabiInteraction`, `JaynesCummings`, `AntiJaynesCummings`, `SelectiveQubitRotation` |

From an execution-path perspective, these gates map onto four practical groups:

- **Discrete-only control** for qubit rewrites on the HDD.
- **Pure CV evolution** for Fock-space and Gaussian-state updates on qumodes.
- **Conditionally applied hybrid gates** that branch on qubit state and act on one or more qumodes.
- **Qubit-qumode mixing interactions** that explicitly couple DV and CV degrees of freedom.

This section is intentionally limited to the gate surface that is present in the current codebase, rather than older paper-only or placeholder functionality.

## Repository layout

| Path | Purpose |
| --- | --- |
| `include/` | Public headers for the simulator library |
| `src/core/` | CV state pool, Gaussian state handling, HDD nodes, and shared math |
| `src/cpu/` | Circuit construction, compilation, and CPU-side control flow |
| `src/gpu/` | CUDA kernels for qubit, CV, and hybrid gate execution |
| `src/scheduler/` | Batch scheduling and instruction fusion |
| `src/noisy/` | Experimental noisy-runtime subsystem |
| `examples/` | Small standalone usage example |
| `tests/` | Unit tests and analysis/benchmark drivers |
| `experiments/` | Single-GPU benchmark harness, configs, and result-processing scripts |
| `baselines/` | External baseline simulators and related setup material |
| `docs/` | Architecture and design documentation |

## Requirements

- **CMake** 3.18 or newer
- **C++ compiler** with C++17 support
- **CUDA Toolkit** 11.x or 12.x for GPU acceleration
- **GCC 9** is the recommended CUDA host compiler when available; otherwise the build falls back to the system GCC/Clang toolchain
- **Linux-like environment** is the primary tested target

If CUDA is not detected, CMake can still configure a CPU-only build for limited development and inspection workflows, but the main simulator is designed around GPU execution.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To build the GoogleTest-based test suite as well:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHYBRIDCVDV_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run the included executables

After a successful build, the repository provides these common entry points:

```bash
./build/HybridCVDV-Simulator_main
./build/HybridCVDV-Simulator_examples
```

When CUDA is available, the single-GPU experiment driver is also built:

```bash
./build/hybridcvdv_single_gpu_experiments
```

## Minimal usage example

The example below mirrors `examples/basic_usage.cpp`:

```cpp
#include <complex>
#include "quantum_circuit.h"

int main() {
    constexpr double pi = 3.14159265358979323846;
    QuantumCircuit circuit(1, 2, 8, 16);
    circuit.build();

    circuit.add_gate(Gates::PhaseRotation(0, pi / 4.0));
    circuit.add_gate(Gates::Displacement(0, std::complex<double>(0.3, 0.1)));
    circuit.add_gate(Gates::Squeezing(1, std::complex<double>(0.2, 0.0)));
    circuit.add_gate(Gates::BeamSplitter(0, 1, pi / 3.0));
    circuit.add_gate(Gates::CreationOperator(0));

    circuit.execute();
    auto stats = circuit.get_stats();
    return stats.active_states >= 0 ? 0 : 1;
}
```

## Benchmarks and experiment flow

The benchmark harness in `experiments/` is the current entry point for paper-style GPU runs.

Typical flow:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target hybridcvdv_single_gpu_experiments -j
python experiments/python/run_gpu_benchmark_matrix.py
```

For setup details, benchmark matrix formats, and baseline environment preparation, see [Experiment Harness](experiments/README.md)
## Notes on the current codebase

- `src/noisy/` is a sandbox for the next noisy-simulation subsystem and is not wired into the root `CMakeLists.txt` yet.
- The repository contains both library-style APIs and paper-oriented experiment drivers; not every directory is meant for end-user installation.
- The previous top-level README contained stale benchmark numbers and placeholder metadata. This version intentionally focuses on the current implementation and documented build/run flow instead.

## License

This project is released under the [MIT License](LICENSE).
