# Noisy Implementation Plan

This document turns the noisy-simulation research conclusion into a repository-specific implementation plan. The goal is to add a new noisy-runtime subsystem under `src/noisy/` without changing the current execution path in `src/cpu/`, `src/core/`, or `src/gpu/`.

## 1. Design Goal

The implementation should satisfy four constraints:

1. Keep the current pure-state simulator unchanged and usable.
2. Put all new noisy-simulation code under `src/noisy/`.
3. Reuse the current pure-state engine as much as possible.
4. Separate the exact Gaussian-channel path from the stochastic trajectory path.

The first milestone is not a full replacement runtime. It is a parallel subsystem that can be developed and validated in isolation.

## 2. Proposed Directory Layout

Create a new subsystem rooted at:

- `src/noisy/`

Current structured skeleton:

- `src/noisy/README.md`
- `src/noisy/types.h`
- `src/noisy/gaussian_channels.h`
- `src/noisy/runtime.h`
- `src/noisy/core/`
- `src/noisy/channels/`
- `src/noisy/executors/`
- `src/noisy/observables/`
- `src/noisy/runtime/`
- `src/noisy/internal/`

Concrete files already laid down:

- `src/noisy/core/types.h`
- `src/noisy/core/gaussian_moment_state.h`
- `src/noisy/core/noisy_program.h`
- `src/noisy/channels/gaussian_channels.h`
- `src/noisy/executors/gaussian_channel_executor.h`
- `src/noisy/executors/trajectory_executor.h`
- `src/noisy/observables/observable_accumulator.h`
- `src/noisy/runtime/noisy_runtime.h`
- `src/noisy/internal/matrix_utils.h`
- `src/noisy/internal/gate_conversion.h`

Planned later files:

- `src/noisy/reference_density_backend.cpp`
- `src/noisy/phase_space_mixture.cpp`

The new subsystem should remain out of `CMakeLists.txt` until the first internal API is stable.

## 3. Runtime Split

The noisy subsystem should have two main execution modes.

### 3.1 Gaussian-channel mode

Use exact Gaussian-channel evolution on symbolic CV states:

- `d' = X d + c`
- `Sigma' = X Sigma X^T + Y`

This mode handles:

- pure loss
- thermal loss
- additive Gaussian noise
- Gaussian amplifier noise

This mode should be the default whenever the active CV state is still Gaussian and the requested noise is Gaussian.

### 3.2 Trajectory mode

Use stochastic pure-state trajectories for general Markovian noise:

- DV amplitude damping
- DV dephasing
- CV photon loss outside the pure Gaussian-only path
- jump-style Lindblad evolution for hybrid DV-CV circuits

Each trajectory should reuse the existing pure-state simulator rather than introducing a new full density-matrix state representation.

## 4. Core Data Structures

### 4.1 `src/noisy/core/types.h`

This file should define the shared value types for the new subsystem.

Recommended types:

- `enum class NoiseChannelKind`
- `enum class NoisyExecutionMode`
- `enum class ObservableKind`
- `struct GaussianMomentUpdate`
- `struct NoiseInstruction`
- `struct TrajectoryConfig`
- `struct ObservableSpec`
- `struct ObservableEstimate`
- `struct NoisyRunConfig`
- `struct NoisyRunResult`

Suggested shape:

- `NoiseChannelKind`
  - `GaussianLoss`
  - `ThermalLoss`
  - `AdditiveGaussianNoise`
  - `GaussianAmplifier`
  - `DVAmplitudeDamping`
  - `DVDephasing`
  - `CustomJumpOperator`
- `NoisyExecutionMode`
  - `GaussianChannelsOnly`
  - `QuantumTrajectories`
  - `ReferenceDensityMatrix`
- `ObservableKind`
  - `Expectation`
  - `Variance`
  - `Probability`
  - `PhotonNumber`
  - `Parity`
  - `Fidelity`
  - `SampleHistogram`

`GaussianMomentUpdate` should carry:

- `std::vector<double> X`
- `std::vector<double> Y`
- `std::vector<double> c`
- `int num_qumodes`
- `std::vector<int> target_qumodes`

`NoiseInstruction` should carry:

- channel kind
- target qubits
- target qumodes
- real parameters
- complex parameters
- optional jump weight or rate

`TrajectoryConfig` should carry:

- number of trajectories
- RNG seed
- max jump count per shot
- whether to store per-shot samples

`ObservableSpec` should carry:

- observable kind
- target qubits
- target qumodes
- optional reference-state label or operator coefficients

### 4.2 Why these types are separate

These types should not be mixed into the current `GateParams` yet.

Reason:

- `GateParams` is currently a pure-gate container for the existing runtime.
- Noisy evolution needs a larger semantic space than unitary gates.
- Keeping the noisy types separate prevents accidental coupling to the current execution planner.

## 5. Gaussian Channel API

### 5.1 `src/noisy/channels/gaussian_channels.h`

This file should host the Gaussian-channel constructors and validation utilities.

Recommended declarations:

- `struct GaussianChannel`
- `class GaussianChannelFactory`
- `bool validate_gaussian_channel(const GaussianChannel&)`
- `GaussianChannel compose_gaussian_channels(const GaussianChannel&, const GaussianChannel&)`

Suggested factory functions:

- `GaussianChannelFactory::pure_loss(double eta, int num_qumodes, int target_qumode)`
- `GaussianChannelFactory::thermal_loss(double eta, double n_th, int num_qumodes, int target_qumode)`
- `GaussianChannelFactory::additive_noise(double variance, int num_qumodes, int target_qumode)`
- `GaussianChannelFactory::phase_insensitive_amplifier(double gain, double n_env, int num_qumodes, int target_qumode)`

Required validation:

- matrix dimension checks
- target-mode bounds
- complete-positivity consistency checks for `Y`
- numerical tolerance handling for nearly singular channels

## 6. Execution Classes

### 6.1 `src/noisy/runtime/noisy_runtime.h`

This file should define the top-level orchestration classes.

Recommended classes:

- `class NoisyProgram`
- `class GaussianChannelExecutor`
- `class TrajectoryExecutor`
- `class ObservableAccumulator`
- `class NoisyRuntime`

#### `NoisyProgram`

Purpose:

- own the sequence of deterministic gates and noise instructions
- stay separate from the current `QuantumCircuit`

Recommended public API:

- `void add_gate(const GateParams&)`
- `void add_noise(const NoiseInstruction&)`
- `void add_observable(const ObservableSpec&)`
- `const std::vector<std::variant<GateParams, NoiseInstruction>>& instructions() const`
- `const std::vector<ObservableSpec>& observables() const`

#### `GaussianChannelExecutor`

Purpose:

- evolve symbolic Gaussian states without leaving the Gaussian moment representation

Recommended public API:

- `void apply_channel(int gaussian_state_id, const GaussianChannel&)`
- `bool can_remain_symbolic(const GaussianChannel&) const`
- `void apply_channel_sequence(int gaussian_state_id, const std::vector<GaussianChannel>&)`

Recommended later internal helpers:

- `apply_embedded_X_to_state(...)`
- `apply_embedded_Y_to_state(...)`
- `embed_single_mode_channel(...)`
- `embed_two_mode_channel(...)`

#### `TrajectoryExecutor`

Purpose:

- run many pure-state trajectories on top of the current simulator

Recommended public API:

- `NoisyRunResult run(const NoisyProgram&, const TrajectoryConfig&)`

Recommended internal helpers:

- `QuantumCircuit make_trajectory_circuit(const NoisyProgram&) const`
- `void apply_no_jump_segment(QuantumCircuit&, const NoiseInstruction&, double dt)`
- `void sample_jump_and_apply(QuantumCircuit&, const NoiseInstruction&, uint64_t* rng_state)`
- `void finalize_trajectory_observables(QuantumCircuit&, ObservableAccumulator&)`

#### `ObservableAccumulator`

Purpose:

- average trajectory results and expose confidence intervals

Recommended public API:

- `void begin_run(const std::vector<ObservableSpec>&)`
- `void record_trajectory_value(size_t observable_index, double value)`
- `void record_trajectory_complex_value(size_t observable_index, std::complex<double> value)`
- `NoisyRunResult finish() const`

#### `NoisyRuntime`

Purpose:

- choose the execution mode
- route Gaussian-only workloads into exact Gaussian-channel execution
- route general noisy workloads into trajectories

Recommended public API:

- `NoisyRunResult execute(const NoisyProgram&, const NoisyRunConfig&)`

Recommended mode-selection helpers:

- `NoisyExecutionMode select_mode(const NoisyProgram&, const NoisyRunConfig&) const`
- `bool is_gaussian_only_program(const NoisyProgram&) const`
- `bool requires_reference_density_backend(const NoisyProgram&) const`

## 7. Execution Flow

### 7.1 Gaussian-channel-only flow

1. Build a `NoisyProgram`.
2. Check that every operation is Gaussian and every noise instruction is a Gaussian channel.
3. Initialize symbolic Gaussian states.
4. Apply deterministic Gaussian gates using the existing symplectic machinery.
5. Apply noisy Gaussian channels using `(X, Y, c)` updates.
6. Evaluate observables that can be computed from moments.
7. Return `NoisyRunResult`.

### 7.2 General trajectory flow

1. Build a `NoisyProgram`.
2. Split the instruction list into deterministic and noisy intervals.
3. For each trajectory:
4. Initialize a fresh pure-state `QuantumCircuit`.
5. Replay deterministic gates through the current runtime.
6. At each noisy interval, sample no-jump or jump evolution.
7. Continue until the instruction sequence completes.
8. Evaluate observables on the resulting pure state.
9. Accumulate results across trajectories.
10. Return means, variances, and confidence intervals.

## 8. Integration Boundary With Existing Code

The new subsystem should reuse current code only through narrow boundaries.

Allowed reuse:

- `GateParams`
- `QuantumCircuit`
- Gaussian symbolic state pool concepts
- existing Fock-state kernels
- existing gate factory functions

Not allowed in phase 1:

- editing the current `execute()` path
- editing the current execution-block compiler
- changing the current `GateType`
- changing the current symbolic terminal semantics

This preserves a clean rollback path.

## 9. API Direction

The noisy subsystem should not expose amplitudes as the primary result.

Primary user-facing result types should be:

- expectation values
- variances
- photon-number moments
- parity values
- bitstring and photon-count histograms
- trajectory sample sets
- fidelity estimates

The current amplitude API can remain untouched in the old runtime.

## 10. Milestones

### Milestone 1

- add `src/noisy/` skeleton
- add plan and type definitions
- keep everything out of the build

### Milestone 2

- implement Gaussian-channel data structures
- implement single-mode exact Gaussian-channel updates
- add unit tests for `pure_loss`, `thermal_loss`, and `additive_noise`

### Milestone 3

- implement trajectory executor on top of `QuantumCircuit`
- support DV amplitude damping and dephasing
- support CV loss as trajectory noise on Fock states

### Milestone 4

- add observable accumulation
- add statistical error bars
- add small-system validation against a reference density-matrix solver

### Milestone 5

- evaluate whether a phase-space Gaussian-mixture backend is needed
- only add it if cat/GKP-heavy workloads become important

## 11. Recommended First Code To Write

The first concrete implementation step should be:

1. fill out `src/noisy/types.h`
2. implement `src/noisy/gaussian_channels.h`
3. add a simple `NoisyProgram` container
4. add a `GaussianChannelExecutor` that operates on symbolic moment states only

This gives the project a correct first noisy-CV capability without disturbing the current pure-state simulator.
