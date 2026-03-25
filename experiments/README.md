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
- `scripts/launch_gpu_benchmark_matrix_systemd.sh`
  - Launches the matrix as a `systemd-run --user` service so it survives SSH disconnects and CLI/session teardown.
- `scripts/resume_gpu_benchmark_matrix_systemd.sh`
  - Relaunches an interrupted run through the same user-systemd path.
- `scripts/launch_gpu_benchmark_matrix_cron.sh`
  - Installs a user-cron launcher for hosts where session-scoped `tmux`, `nohup`, or `systemd --user` jobs do not survive the SSH session. It waits for an idle GPU, runs a smoke preflight, then starts/resumes the checkpointed matrix. The same script also exposes `status` and `uninstall` subcommands for inspection/cleanup.
- `configs/sc26_scaling.json`
  - Default SC26 benchmark matrix used by the orchestrator and remote wrapper scripts.
- `configs/gpu_benchmark_matrix.json`
  - Smaller development matrix that can still be selected explicitly with `--config`.
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

This now defaults to `experiments/configs/sc26_scaling.json`. For the smaller development matrix, pass:

```bash
python experiments/python/run_gpu_benchmark_matrix.py \
  --config experiments/configs/gpu_benchmark_matrix.json
```

4. Inspect the manifest:

```bash
cat experiments/results/<run-id>/manifest.json
```

## Notes

- The harness is single-GPU only.
- `scripts/run_gpu_benchmark_matrix.sh` and `scripts/resume_gpu_benchmark_matrix.sh` keep a stable
  `run_id`, persist the active `build_dir` in the checkpoint, update the checkpoint after every
  completed benchmark case, and resume from unfinished cases instead of restarting the full matrix.
- Set `SKIP_BUILD=1` when you already trust the binary in `BUILD_DIR` and want to reuse it without
  re-running `cmake` / `cmake --build` before the matrix step.
- For smoke/debug runs, `HYBRIDCVDV_SCALING_WARMUP_RUNS` and `HYBRIDCVDV_SCALING_MEASURED_RUNS`
  can override the scaling suite's default `2` warmups and `10` measured runs without changing the
  default benchmark methodology.
- When the remote host has a working `systemd --user` session, prefer
  `scripts/launch_gpu_benchmark_matrix_systemd.sh` (and the matching resume wrapper) for true
  long-lived execution that survives SSH disconnects.
- On hosts where session-scoped background jobs are torn down with the SSH session, use
  `scripts/launch_gpu_benchmark_matrix_cron.sh install` instead. The installed cron tick keeps a
  short coordinator lock, waits for zero active GPU compute apps, and launches detached
  preflight/matrix workers tracked by pid/rc files so minute-level cron ticks cannot relaunch the
  same work while it is still running. Those rc files are also written when a worker shell is
  interrupted by `HUP`/`INT`/`QUIT`/`TERM`, so stale-worker failures are visible on the next tick.
  The default smoke preflight remains
  `sc26_vqe_nq3_nm7_c16` with `0/1` warmup/measured repetitions. Set
  `CRON_SECONDARY_DELAY_SECONDS=30` if you want a second half-minute tick without depending on a
  long-lived session-scoped waiter.
- Inspect a cron-managed launch with
  `scripts/launch_gpu_benchmark_matrix_cron.sh status --env-file <results>/cron/<run-id>/launcher.env`.
- The cron launcher also keeps a machine-readable status file at
  `<results>/cron/<run-id>/status.json`, updated on each tick with the current launcher state.
- `configs/gpu_benchmark_matrix.json` still maps `strawberryfields_tf` to `.venv-sf-gpu/bin/python`
  and `mrmustard_jax` to `.venv-mm-gpu/bin/python` when you explicitly choose that config.
- `strawberryfields_tf` requires TensorFlow to see a CUDA GPU.
- `mrmustard_jax` requires a JAX GPU install plus a MrMustard version that exposes the JAX backend.
- Unsupported cases are written explicitly to JSON instead of being skipped silently.
