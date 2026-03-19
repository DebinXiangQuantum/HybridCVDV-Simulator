# Implementation Details and Audit Notes

This note complements `architecture.md` with the current implementation status in `src/`, plus a short audit of details that already exist in code but were not explicitly captured in the architecture summary.

For the noisy-simulation research conclusion and the recommended extension path for this repository, see `docs/noisy_simulation_strategy.md`.
For the class-level implementation plan and the isolated `src/noisy/` subsystem layout, see `docs/noisy_implementation_plan.md`.

## 1. Architecture Coverage in `src/`

### 1.1 CPU control layer / HDD rewriting
- The immutable HDD rewrite path is implemented in `src/cpu/quantum_circuit.cpp`.
- `DD_Add` is implemented as `QuantumCircuit::hdd_add(...)`, including:
  - terminal-state linear combination on the GPU,
  - recursive alignment on mismatched qubit levels,
  - branch-wise recombination for superposition gates.
- Pure qubit gates (`X/Y/Z/H/Rx/Ry/Rz/S/T/CNOT/CZ`) are handled through recursive HDD rewrites and `HDDNodeManager`.

### 1.2 GPU physical layer / TET path
- Level-0 diagonal CV gates are implemented in `src/gpu/diagonal_gates.cu`.
- Level-1 ladder gates are implemented in `src/gpu/ladder_gates.cu`.
- Level-2 single-mode gates are implemented in `src/gpu/single_mode_gates.cu` and `src/gpu/squeezing_gate.cu`.
- Level-3 two-mode gates are implemented in `src/gpu/two_mode_gates.cu`.

### 1.3 Hybrid CPU+GPU gates
- Separable hybrid gates are implemented through HDD traversal plus copy-on-write terminal duplication:
  - `CD`, `CS`, `CBS`, `CTMS`, `SUM`.
- Qubit-mixing hybrid gates are implemented through paired low/high branch recursion:
  - `RB`, `JC`, `AJC`, `SQR`.

### 1.4 Dual-track engine / EDE
- The Gaussian symbolic track is implemented via symbolic terminal sidecars and `GaussianStatePool`.
- Gaussian blocks are compiled and attempted through `try_execute_gaussian_block_with_ede(...)`.
- Diagonal non-Gaussian blocks can stay on the symbolic side through Gaussian mixture decomposition in `try_execute_diagonal_non_gaussian_block_with_mixture(...)`.
- Symbolic branches are materialized back to Fock states by `project_symbolic_terminal_to_fock_state(...)` and `materialize_symbolic_terminals_to_fock()`.

## 2. Extra Implementation Details Already Present in Code

These details are useful for the paper/docs because they materially affect runtime behavior, but they were not spelled out in `architecture.md`.

- Block compilation pipeline:
  `QuantumCircuit::compile_execution_block(...)` precomputes Gaussian updates, mixture metadata, downstream non-Gaussianity estimates, and fidelity/error bounds.

- Symbolic terminal sidecar design:
  symbolic terminals use negative logical IDs and store branch-local Gaussian state IDs plus replayable Gaussian gate history.

- Vacuum-ray classification:
  before promoting a Fock terminal to the Gaussian track, the code checks whether it is a zero ray or a scaled vacuum ray, avoiding invalid symbolic promotion.

- Scratch-buffer reuse:
  `CVStatePool` owns reusable GPU scratch buffers (`scratch_target_ids`, `scratch_temp`, `scratch_aux`) to remove repeated `cudaMalloc/cudaFree` from the hot path.

- Dynamic storage reuse:
  `CVStatePool` uses reusable storage blocks with block merging and best-fit reuse, instead of forcing one fixed allocation per active state.

- Scheduler-side optimizations:
  `BatchScheduler` supports task merging, prepared level-0 batches, and CUDA Graph capture for eligible workloads.

- Symbolic branch safety limits:
  the runtime caps symbolic mixture branch count and prunes very small branch weights to keep EDE fallback behavior bounded.

## 3. Audit Fixes Applied in This Pass

- Fixed HDD unique-table correctness:
  `HDDNodeManager` no longer treats a hash collision as structural equality. Colliding nodes now share a hash bucket and are compared structurally before deduplication.

- Fixed `CD` semantics to match the architecture:
  the two qubit branches now apply `D(+alpha)` and `D(-alpha)` respectively, instead of leaving the `|0>` branch unchanged.

- Replaced the previous `SQR` placeholder behavior:
  `SQR` now performs real low/high branch mixing through paired recursion and expands the `(theta_n, phi_n)` profile onto the flattened multi-mode Fock layout using the selected control qumode.

- Replaced placeholder ELL builders:
  `prepare_ell_operator(...)` and `prepare_squeezing_ell_operator(...)` now build real dense reference matrices and convert them into `FockELLOperator` form.

- Removed the dead duplicate beam-splitter source:
  `src/gpu/beamsplitter_recursive.cu` was an unused placeholder copy and was removed. The active implementation remains in `src/gpu/two_mode_gates.cu`.

- Removed the dead Level-4 placeholder path:
  `execute_level4_gate(...)` now delegates to the real hybrid execution path instead of calling a stub.

- Hardened the legacy GPU stub:
  `apply_hybrid_control_gate(...)` now throws an explicit deprecation error if it is ever called, instead of silently doing nothing.

- Aligned the reference implementation:
  `src/reference/reference_gates.*` now uses the same `sigma_z` branch semantics as the main runtime for `CD` and `CS`.

## 4. Current Reading of Remaining Non-Core Utilities

- `src/memory/memory_pool.cpp` and `src/memory/garbage_collector.cpp` are support utilities, not part of the main `QuantumCircuit` execution path described in `architecture.md`.
- They compile successfully, but they should still be treated as auxiliary infrastructure rather than part of the paper's core execution engine unless/until they are wired into the runtime.
