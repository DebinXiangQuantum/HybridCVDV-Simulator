# GPU Benchmark Harness

This directory contains the single-GPU experiment harness used for the HPC paper benchmark section.
All outputs are JSON so the same artifacts can feed tables, plots, and appendix dumps.

## Layout

- `cpp/single_gpu_experiments.cpp`
  - Internal HybridCVDV runner for correctness, microbench, runtime ablation, and scaling cases derived from `circuit/src/`.
- `python/run_baseline_backend.py`
  - Per-case baseline runner for `strawberryfields_tf` and `mrmustard_jax`.
- `python/run_gpu_benchmark_matrix.py`
  - Matrix orchestrator. Runs one case at a time and samples GPU telemetry with `nvidia-smi`.
- `scripts/setup_gpu_baseline_envs.sh`
  - Checkpoint-aware remote bootstrap for `.venv-sf-gpu` and `.venv-mm-gpu`.
- `scripts/resume_gpu_baseline_envs.sh`
  - Resume entrypoint for the baseline environment bootstrap task.
- `scripts/run_gpu_benchmark_matrix.sh`
  - Checkpoint-aware wrapper for remote long runs.
- `scripts/resume_gpu_benchmark_matrix.sh`
  - Resume entrypoint when a remote long run is interrupted after build.
- `configs/gpu_benchmark_matrix.json`
  - Default benchmark matrix used by the orchestrator.
- `results/<run-id>/`
  - Timestamped outputs for each backend/case pair plus raw telemetry samples.

## Experiment Split

- Common CV layer
  - Workloads that all three GPU backends can express fairly.
- Current defaults:
    - `cv_qaoa` from [qaoa_circuit.cpp](/Users/xiangdebin/Library/Mobile%20Documents/com~apple~CloudDocs/codes/quantumcomputing/HybridCVDV-Simulator/circuit/src/qaoa_circuit.cpp)
    - `jch_photonic_chain`, derived from the bosonic `R` and `BS` terms in [jch_simulation_circuit.cpp](/Users/xiangdebin/Library/Mobile%20Documents/com~apple~CloudDocs/codes/quantumcomputing/HybridCVDV-Simulator/circuit/src/jch_simulation_circuit.cpp)
- The experiment configs set `gaussian_symbolic_mode_limit=16` for the internal runner so large-mode Gaussian blocks are not cut off by the legacy default of 4.
- Internal hybrid layer
  - Workloads that depend on hybrid DV-CV gates and therefore only run on HybridCVDV.
  - Current defaults include cat-state, GKP, state-transfer, full JCH, and VQE cases.

## Metrics

Per case, the harness records:

- total time
- compute time
- communication time
- estimated state-vector bytes
- sampled GPU util, memory, power, temperature, and SM clock

Communication time for `strawberryfields_tf` and `mrmustard_jax` is defined as final-state materialization from device to host.
This keeps the metric observable and comparable across frameworks, even though their internal runtime stacks differ.

## Typical Flow

1. Build the internal runner:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target hybridcvdv_single_gpu_experiments -j
```

2. Prepare backend-specific Python environments for baselines:

```bash
bash experiments/scripts/setup_gpu_baseline_envs.sh
```

3. Run the benchmark matrix:

```bash
python experiments/python/run_gpu_benchmark_matrix.py
```

4. Inspect the manifest:

```bash
cat experiments/results/<run-id>/manifest.json
```

## Notes

- The harness is single-GPU only.
- By default `configs/gpu_benchmark_matrix.json` maps `strawberryfields_tf` to `.venv-sf-gpu/bin/python`
  and `mrmustard_jax` to `.venv-mm-gpu/bin/python`.
- `strawberryfields_tf` requires TensorFlow to see a CUDA GPU.
- `mrmustard_jax` requires a JAX GPU install plus a MrMustard version that exposes the JAX backend.
- Unsupported cases are written explicitly to JSON instead of being skipped silently.
