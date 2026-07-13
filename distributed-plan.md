# 8×H800 多 GPU 扩展性实验计划

## 1. 实验目标

本实验用于回答以下问题：

1. HybridCVDV-Simulator 从 1 张 GPU 扩展到 2、4、6、8 张 GPU 后，单个电路的模拟时间、计算时间、通信时间、吞吐量和显存占用如何变化。
2. 当前多 GPU `CVStatePool` 能否把多个完整 Fock state 均匀分布到多张 GPU，并扩大可模拟问题的容量上限。
3. 多 GPU 增加的是单任务速度、可模拟容量，还是仅增加并发吞吐量。
4. HybridCVDV 与 ATLAS、BQSim 在相同硬件、相同输入和相同计时边界下的性能差异。

正式扫描使用：

```text
GPU_COUNT = 1, 2, 4, 6, 8
```

## 2. 当前实现状态和实验边界

### 2.1 HybridCVDV-Simulator

当前 `CVStatePool` 已支持在一个进程内管理多张 GPU，并把不同的完整 state 分配到不同 GPU。它当前不是“把单个 state vector 切片到多张 GPU”。因此：

- 当一个 workload 同时存在多个 Fock states 时，多 GPU 可以分摊状态池容量和部分计算。
- 当 workload 只有一个大 state 时，该 state 仍必须完整放入一张 GPU。
- 如果单态大小超过单卡显存，例如 `16^8 × sizeof(complex<double>) = 64 GiB`，即使 8 张 GPU 可见也仍会 OOM。
- 小 case 可能只使用一张 GPU；这不是实验错误，但不能据此宣称单任务获得了多 GPU speedup。

因此 HybridCVDV 必须同时报告：

- strong scaling：固定 case，增加 GPU 数；
- capacity scaling：增加 GPU 后，原来因状态池总容量不足而失败的 case 是否成功；
- throughput scaling：固定总任务数，增加 GPU 后单位时间完成多少个独立 circuit evaluation；
- state placement：每张 GPU 上的 state 数量和 state bytes。

### 2.2 现有 HybridCVDV benchmark runner 的限制

现有入口是：

- `experiments/scripts/run_gpu_benchmark_matrix.sh`
- `experiments/python/run_gpu_benchmark_matrix.py`
- `experiments/configs/sc26_scaling.json`

当前 runner 可以让被测进程看到多张 GPU，但仍有以下限制：

1. `run_gpu_benchmark_matrix.py` 只通过 `--gpu-index` 采样一张 GPU，不能记录所有可见 GPU 的利用率、显存和功耗。
2. 结果中的 `num_gpus` 只说明进程看到了多少张 GPU，不说明每张 GPU 实际承担了多少计算。
3. `throughput_ops_per_sec = 1000 / median_total_ms` 实际更接近 circuit evaluations/s，不是严格的 gate operations/s。
4. `median_transfer_ms` 不能完整代表 GPU 间通信；`PeerTransfer` 的 bytes、时间、P2P/host-staged 策略尚未汇总到 benchmark JSON。
5. 当前 runner 串行执行 case，不支持统一的多进程吞吐实验。

正式实验前必须扩展 runner，详见第 8 节。

### 2.3 ATLAS

ATLAS 原生包含 MPI、NCCL 和多 GPU state-vector 路径，理论上适合测单电路 strong scaling。但仓库中的当前版本还不能直接作为正式结果使用：

1. `baselines/atlas-main/CMakeLists.txt` 硬编码了旧 CUDA、cuQuantum 和 NCCL 路径，且 CUDA architecture 未包含集群实际 GPU 架构。
2. `PERFORMANCE_MONITORING.md` 中列出的 H2D、D2H、D2D 和 compute 计时点尚未真正接入 `src/base/simulator.cu`。
3. 当前 `InitStateMulti` 仅在 `nRanks > 1` 时初始化 NCCL；单节点 `mpirun -np 1 --device N` 会跳过 NCCL 通信。
4. `all2all()` 在 `nRanks == 1` 时直接返回，这会使单节点多 GPU shuffle 缺少真实数据交换。
5. 设备划分使用位掩码和 `log2(device_count)` 逻辑，原生分布式路径应视为要求 GPU 总数为 2 的幂。`6 GPU` 必须先做正确性验证；如果不支持，应明确记录为 `unsupported`，不能静默改成 4 或 8。

ATLAS 正式 strong-scaling 主曲线优先使用 `1/2/4/8 GPU`。`6 GPU` 仍执行 smoke test 并保留结果状态。

### 2.4 BQSim

BQSim 当前是单 GPU batch simulator，没有发现单个 BQSim 进程把同一 simulation 分布到多张 GPU 的实现。因此不能把 `CUDA_VISIBLE_DEVICES=0,1,...` 当作 BQSim 的 native multi-GPU scaling。

BQSim 的多 GPU 实验定义为“独立进程数据并行”：

- 每张 GPU 启动一个 BQSim 进程；
- 把固定的总 batch 数均匀分给 N 个进程；
- 记录整个批任务的 makespan 和 aggregate throughput；
- 不计算 BQSim 的单任务 native multi-GPU speedup。

此外，BQSim 当前计时有两个必须修复的问题：

- 当 transfer time 为 0 时会写入固定的 `8 ms`，该值不能用于论文结果。
- `peak_memory_usage` 当前读取的是进程 CPU RSS，不是 GPU 显存峰值。

正式实验前应使用 CUDA events 记录 H2D、kernel、D2H，并通过 NVML 或外部 telemetry 记录每张 GPU 的显存。

## 3. 公平比较原则

### 3.1 相同输入

实验的 workload source of truth 是：

```text
experiments/configs/sc26_scaling.json
```

当前配置共有 409 个 case：

| Cutoff | Case 数量 |
|---:|---:|
| 4 | 96 |
| 8 | 97 |
| 16 | 117 |
| 32 | 99 |

JCH 和 VQE 已包含完整的 `c4/c8/c16/c32` 参数扫描。其他 workload 的 cutoff 不完全齐全。分布式实验不要手工修改原配置，而应由脚本生成派生配置：

```text
experiments/configs/sc26_distributed_scaling.json
```

生成规则：

1. 保留原配置的全部 case 和参数。
2. 对需要完整 cutoff 扫描的 workload，从同一参数模板派生 `4/8/16/32`，避免重复 name。
3. 生成后的 config 写入结果目录，作为实验 artifact 保存。
4. 不根据旧结果目录反向生成 case，避免遗漏 QFT 等 workload。

### 3.2 Baseline 使用同一份 QASM

ATLAS 和 BQSim 目录下目前各自有一套 QASM generator，而且两个 generator 对同一 CV gate 的编码并不完全一致。这会破坏公平比较。

正式实验需要建立唯一的 canonical QASM generator，例如：

```text
experiments/python/generate_sc26_baseline_qasm.py
```

生成一次后，将同一份、字节一致的 QASM 输入同时交给 ATLAS 和 BQSim。建议使用 ATLAS 和 BQSim 都支持的 OpenQASM 2.0 gate 子集。

当前两个 baseline generator 都设置了：

```python
ALLOWED_CUTOFFS = [4, 8, 16, 32]
```

需要注意：将 qumode 截断空间编码为 `ceil(log2(cutoff))` 个 qubit 是 baseline surrogate，不等价于 HybridCVDV 的原生 CV gate。论文中应将 ATLAS/BQSim 标记为 encoded-QASM baselines；只有在 gate mapping 和输出正确性通过后，才能声称算法语义完全等价。

### 3.3 相同精度和计时边界

- 三套系统均使用 double precision complex state，除非某 baseline 明确不支持；任何精度差异必须写入结果。
- `compile/setup time` 与 `simulation time` 分开报告。
- speedup 使用 simulation-only 时间；end-to-end 表格另报 total wall time。
- warmup 不计入 measured time。
- 所有系统使用相同 case、相同 GPU 集合、相同 warmup/measured 次数和相同 timeout。

## 4. 集群预检查

在 8×H800 节点上首先保存硬件、拓扑和软件环境：

```bash
mkdir -p experiments/results/distributed_8xH800/metadata

nvidia-smi -L \
  > experiments/results/distributed_8xH800/metadata/nvidia-smi-L.txt
nvidia-smi topo -m \
  > experiments/results/distributed_8xH800/metadata/nvidia-smi-topo.txt
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,power.limit \
  --format=csv \
  > experiments/results/distributed_8xH800/metadata/gpu-inventory.csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv \
  > experiments/results/distributed_8xH800/metadata/gpu-processes-before.csv

nvcc --version \
  > experiments/results/distributed_8xH800/metadata/nvcc-version.txt
cmake --version \
  > experiments/results/distributed_8xH800/metadata/cmake-version.txt
mpirun --version \
  > experiments/results/distributed_8xH800/metadata/mpi-version.txt
lscpu \
  > experiments/results/distributed_8xH800/metadata/lscpu.txt
numactl --hardware \
  > experiments/results/distributed_8xH800/metadata/numa.txt
```

然后执行多 GPU 正确性和 P2P smoke test：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ctest --test-dir build-H800 -R 'MultiGPU|PeerTransfer' --output-on-failure
```

必须记录：

- GPU 之间是否支持 CUDA P2P；
- 实际通信路径是 P2P、PCIe 还是 host-staged；
- GPU 与 CPU NUMA 关系；
- 运行前是否有其他进程占用 GPU；
- driver、CUDA、cuQuantum、NCCL、MPI 和 git commit。

如果集群不支持 GPU P2P，HybridCVDV 当前的部分跨卡操作可能失败，ATLAS 的 NCCL 性能也会明显不同。此时不能隐藏该事实，应把拓扑作为实验条件报告。

## 5. 实验矩阵

实验分五层执行，避免直接运行巨大矩阵后才发现计时或正确性错误。

### 5.1 Phase A：正确性和可运行性 smoke test

GPU 数：`1,2,4,6,8`

每类选择一个小 case 和一个中等 case，`warmup=0, measured=1`：

- JCH
- VQE
- QAOA
- QFT
- CV→DV transfer
- DV→CV transfer

检查：

- 退出状态；
- state norm、checksum 或可比较输出；
- `num_gpus` 与请求 GPU 数一致；
- 每张 GPU 是否有 telemetry；
- 是否发生 P2P 或 host-staged 通信；
- ATLAS 的 6 GPU 是否正确支持。

Phase A 任一系统出现 illegal memory access、错误输出或计时字段为空，先修复，不进入正式实验。

### 5.2 Phase B：强扩展实验

固定 workload，不改变 cutoff、mode、qubit、depth 和 batch，扫描 GPU 数。

主 GPU 数：

```text
HybridCVDV: 1, 2, 4, 6, 8
ATLAS:      1, 2, 4, 8；6 作为兼容性点
BQSim:      不适用 native strong scaling
```

代表性 case 不应只选小 case。建议从 Phase A 和历史结果中按以下规则自动选择：

1. small：1 GPU 能运行，simulation time 约 10–100 ms；
2. medium：1 GPU 能运行，simulation time 约 0.1–5 s；
3. large：1 GPU 能运行且峰值显存达到单卡的 50%–80%；
4. branch-rich：active Fock states 足够多，能触发跨 GPU state placement；
5. communication-heavy：存在频繁 state combine/migration；
6. single-state control：只有一个主要 state，用于展示当前多 GPU 不会加速单态。

每个 cutoff `4/8/16/32` 至少选择 2 个 case；JCH、VQE、transfer 和 QFT 均应出现在主表中。候选包括：

```text
sc26_jch_nq10_nm4_c{4,8,16,32}
sc26_jch_nq3_nm6_c{4,8,16,32}
sc26_vqe_nq8_nm5_c{4,8,16,32}
sc26_vqe_nq3_nm6_c{4,8,16,32}
sc26_qft_nq9_c{4,8,16,32}
sc26_transfer_DVtoCV_nq16_c{4,8,16,32}
```

候选最终以“能在 1 GPU 正确运行”作为 strong-scaling 入选条件。1 GPU OOM 的 case 放入 capacity scaling，不用于计算 `T1/TN`。

正式参数：

```text
warmup_runs = 2
measured_runs = 10
process_repetitions = 3
telemetry_interval_ms = 100
```

三个 process repetition 的 GPU-count 顺序应交错或使用固定随机种子打乱，减少温度、boost clock 和运行顺序偏差。

### 5.3 Phase C：容量和弱扩展实验

容量实验使用在 1 GPU 上 OOM、但单个 state 小于单卡显存的 workload。目标是验证多个完整 state 能否分布到更多 GPU。

每个 GPU 数选择一个使“每 GPU 目标显存占用”接近恒定的问题规模，报告：

- 最大成功 cutoff；
- 最大成功 mode 数；
- 最大 active state 数；
- aggregate state-pool bytes；
- max per-GPU memory；
- memory imbalance；
- OOM 首次出现位置。

单态超过单卡显存的 case，例如 `sc26_qaoa_nm8_c16`，应单独标记为：

```text
unsupported_single_state_too_large
```

它不能用于证明多 GPU state pool 的容量扩展失败，因为当前设计本来没有实现单态切片。

### 5.4 Phase D：吞吐扩展实验

固定总任务量 `B_total`，分别使用 `1/2/4/6/8 GPU` 完成相同数量的独立 circuit evaluations。

建议：

```text
B_total = 256 或 1024
每张 GPU 一个进程
每进程任务数 = ceil(B_total / GPU_COUNT)
```

三套系统都执行该实验：

- HybridCVDV：每个进程只看一张 GPU，运行独立 case/repetition；
- ATLAS：吞吐模式下每个进程运行独立单 GPU simulation，不使用 distributed state vector；
- BQSim：每张 GPU 一个 BQSim 进程，分片 `num_batch`。

吞吐模式与 native strong scaling 必须放在不同图中，不能把“8 个独立进程”解释成“单电路 8 GPU 加速”。

### 5.5 Phase E：全量 case 覆盖扫描

代表性子集用于得到稳定、可画 speedup 曲线的数据；此外还需要对派生配置中的全部 case 执行覆盖扫描，保证每个 workload、cutoff 和 GPU 数都有状态记录。

派生配置按第 3.1 节补齐 cutoff 后，当前配置预计约有 468 个 case，最终数量以生成器的去重和校验结果为准。完整扫描为：

```text
HybridCVDV: all cases × 1/2/4/6/8 GPU
ATLAS:      all canonical-QASM cases × 1/2/4/8 GPU，6 GPU 记录兼容性状态
BQSim:      all canonical-QASM cases × 1/2/4/6/8 GPU throughput mode
```

全量扫描参数建议为：

```text
warmup_runs = 1
measured_runs = 3
process_repetitions = 1
per_case_timeout = 30 min；large case 可单独放宽到 2 h
```

如果总运行时间过长，先用 `warmup=0, measured=1` 完成 feasibility pass，再只对 `ok` case 执行 `warmup=1, measured=3`。全量结果主要用于：

- 成功率和失败分类；
- 最大可模拟规模；
- OOM 边界；
- GPU 使用和显存分布；
- 发现异常 scaling 的 case。

论文主性能数字仍应来自 Phase B 中 `2 warmup + 10 measured × 3 process repetitions` 的代表性子集，不能用单次 feasibility 时间替代。

## 6. 指标和计算方法

### 6.1 时间

每个 case、每个 GPU 数记录：

| 字段 | 定义 |
|---|---|
| `total_wall_ms` | 从进程启动到结果落盘的端到端时间 |
| `setup_ms` | runtime、state pool、stream、handle 初始化 |
| `compile_ms` | 电路解析、优化、fusion、schedule 构建 |
| `simulation_ms` | 正式模拟执行时间 |
| `gpu_compute_ms` | GPU kernel 的累计或 critical-path 时间 |
| `h2d_ms` | CPU→GPU 数据传输时间 |
| `d2h_ms` | GPU→CPU 数据传输时间 |
| `p2p_ms` | GPU→GPU P2P 时间 |
| `host_staged_ms` | 通过 pinned host memory 的跨卡传输时间 |
| `sync_wait_ms` | stream/device synchronization 等待时间 |

异步 CUDA 时间应使用 CUDA events；总时间使用 monotonic wall clock。多 stream 的时间不能简单相加后当作 wall time，需同时报告累计 device time 和 critical-path wall time。

### 6.2 扩展性

固定 case 的 strong-scaling 指标：

```text
Speedup(N)              = T(1) / T(N)
ParallelEfficiency(N)   = Speedup(N) / N
CommunicationRatio(N)   = T_comm(N) / T_simulation(N)
ComputeRatio(N)         = T_compute(N) / T_simulation(N)
```

其中：

```text
T_comm = p2p_ms + host_staged_ms + communication_sync_ms
```

如果某 case 在 1 GPU OOM，则不计算 strong speedup，只报告 capacity gain。

### 6.3 吞吐量

统一使用以下定义：

```text
CircuitThroughput = completed_circuit_evaluations / makespan_seconds
StateThroughput   = completed_input_states / makespan_seconds
GateThroughput    = completed_gate_applications / makespan_seconds
```

不同系统只有在分子语义一致时才能直接比较。BQSim 的 batch state throughput 不应与 HybridCVDV 的单次 circuit throughput 混为同一个指标。

### 6.4 内存

必须记录 per-GPU，而不只记录总和：

```text
gpu_memory_peak_bytes[gpu]
gpu_memory_avg_bytes[gpu]
state_pool_reserved_bytes[gpu]
state_pool_active_bytes[gpu]
active_state_count[gpu]
scratch_bytes[gpu]
```

派生指标：

```text
AggregatePeakMemory = sum(per_gpu_peak)
MaxPerGpuMemory     = max(per_gpu_peak)
MemoryImbalance     = max(per_gpu_active_bytes) / mean(per_gpu_active_bytes)
```

### 6.5 通信

HybridCVDV 和 ATLAS 均记录：

```text
p2p_bytes
host_staged_bytes
transfer_count
state_migration_count
effective_bandwidth_gbps
communication_time_ms
```

同时保存 GPU topology，区分 NVLink、PCIe P2P 和 host staging。

### 6.6 GPU 遥测和能耗

对所有可见 GPU 每 100 ms 采样：

- GPU utilization；
- memory utilization；
- memory used；
- power
- temperature；
- SM clock；
- PCIe TX/RX throughput，如果驱动支持；
- NVLink TX/RX throughput，如果硬件支持。

派生：

```text
EnergyJoules = integral(power_watts, time)
GpuUtilImbalance = max(avg_util_per_gpu) - min(avg_util_per_gpu)
```

## 7. 结果状态分类

所有失败必须使用机器可读状态：

```text
ok
oom_single_gpu_pool
oom_single_state_too_large
oom_aggregate
unsupported_gpu_count
unsupported_backend
timeout
crash_cuda
crash_host
incorrect_result
missing_telemetry
configuration_error
```

禁止把 runner 没生成 case、进程未启动和真实 CUDA OOM 都统一写成 `error`。

## 8. 开跑前需要完成的脚本和计时改造

### 8.1 统一 orchestrator

建议新增：

```text
experiments/scripts/run_distributed_scaling.sh
experiments/python/run_distributed_scaling.py
experiments/python/merge_distributed_scaling.py
experiments/python/plot_distributed_scaling.py
```

顶层接口建议为：

```bash
bash experiments/scripts/run_distributed_scaling.sh \
  --systems hybridcvdv,atlas,bqsim \
  --gpu-counts 1,2,4,6,8 \
  --config experiments/configs/sc26_distributed_scaling.json \
  --result-root experiments/results/distributed_8xH800 \
  --warmup-runs 2 \
  --measured-runs 10 \
  --repetitions 3 \
  --telemetry-interval-ms 100
```

该脚本负责：

1. 根据 GPU 数生成 `CUDA_VISIBLE_DEVICES=0,...,N-1`；
2. 检查所选 GPU 是否空闲；
3. 顺序运行 native strong-scaling，避免多个系统互相干扰；
4. 启动独立的 all-GPU telemetry collector；
5. 按 system/case/GPU count/repetition 写独立结果；
6. 支持 checkpoint、resume、timeout 和失败分类；
7. 最后生成 merged manifest 和 CSV。

### 8.2 扩展现有 HybridCVDV runner

复用 `run_gpu_benchmark_matrix.py` 的 config、checkpoint 和 artifact 逻辑，增加：

```text
--gpu-indices 0,1,2,3
--execution-mode native|throughput
--process-repetition N
--timeout-seconds N
```

telemetry 从单个 dict 改为：

```json
{
  "per_gpu": {
    "0": {"samples": [], "summary": {}},
    "1": {"samples": [], "summary": {}}
  },
  "aggregate": {}
}
```

HybridCVDV benchmark JSON 还应加入：

```text
states_per_gpu
active_bytes_per_gpu
reserved_bytes_per_gpu
p2p_bytes
p2p_time_ms
host_staged_bytes
host_staged_time_ms
state_migrations
```

### 8.3 ATLAS 改造

开跑前完成：

1. 移除 CMake 中 CUDA、cuQuantum、NCCL 的硬编码路径，改用环境变量或 CMake cache 参数。
2. 使用集群 CUDA 编译器支持的 native GPU architecture。
3. 单节点多 GPU 时，只要 `nRanks * n_devices > 1` 就初始化 NCCL。
4. 修复 `all2all()` 对 `nRanks == 1` 的错误跳过逻辑。
5. 在 `ApplyGate`、`InitStateMulti`、`ApplyRecordedShuffle` 和结果复制处加入 CUDA event 计时。
6. CSV/JSON 加入 `gpu_count`、compute、H2D、D2H、D2D、通信 bytes 和 per-GPU memory。
7. 对 1/2/4/8 GPU 做 state checksum 正确性验证后再跑性能。

修复后的单节点命令目标形式：

```bash
GPU_COUNT=4
CUDA_VISIBLE_DEVICES=0,1,2,3 \
mpirun -np 1 --bind-to none \
  baselines/atlas-main/build/examples/mpi-based/run_generated_qasm \
  --import-circuit sc26_jch_nq10_nm4_c16 \
  --n 26 \
  --local 24 \
  --device "${GPU_COUNT}" \
  --use-ilp
```

`--n` 和 `--local` 必须由 wrapper 根据 canonical QASM 的实际 qubit 数和 GPU partition 计算，不应沿用 `run_all_qasm.sh` 中的 `num_qubits/2` 经验值。

现有 `baselines/atlas-main/run_all_qasm.sh` 只是并行启动多个 `mpirun -n 1 --device 0` 任务，不是 ATLAS native multi-GPU scaling，不能直接用于本实验。

### 8.4 BQSim 改造

开跑前完成：

1. 删除 transfer time 的固定 `8 ms` fallback。
2. 用 CUDA events 分别测量 H2D、kernel 和 D2H。
3. 加入 `--output`，避免多进程并发写同一个 CSV。
4. 输出实际 GPU memory peak，不再把 CPU RSS 当 GPU memory。
5. 输出 completed states、gate count 和 batch count，供计算吞吐量。

单 GPU 命令目标形式：

```bash
CUDA_VISIBLE_DEVICES=0 \
baselines/BQSim-main/build/apps/BQSim \
  --ps \
  --batch_size 256 \
  --num_batch 200 \
  --conversion_type 2 \
  --file baselines/BQSim-main/qasm/sc26_jch_nq10_nm4_c16.qasm \
  --output experiments/results/distributed_8xH800/bqsim/g1/result.json
```

多 GPU throughput 模式由 orchestrator 启动 N 个进程，每个进程绑定一张 GPU，且输出路径唯一。

## 9. 正式运行流程

### Step 1：冻结代码和环境

记录 git commit、dirty diff、container image 或依赖版本。正式三套系统测试期间不要更新 driver、CUDA、编译参数或代码。

### Step 2：生成派生 config 和 canonical QASM

```bash
python3 experiments/python/generate_distributed_config.py \
  --source experiments/configs/sc26_scaling.json \
  --cutoffs 4,8,16,32 \
  --output experiments/configs/sc26_distributed_scaling.json

python3 experiments/python/generate_sc26_baseline_qasm.py \
  --config experiments/configs/sc26_distributed_scaling.json \
  --output-dir experiments/generated/sc26_baseline_qasm
```

这两个脚本是计划中的新增脚本，未实现前不要直接执行上述命令。

### Step 3：编译三套系统

HybridCVDV：

```bash
cmake -S . -B build-H800 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-H800 --target hybridcvdv_single_gpu_experiments -j "$(nproc)"
```

ATLAS 和 BQSim 必须使用相同 CUDA toolchain、Release 模式和明确记录的 architecture flags。

### Step 4：Phase A smoke

先运行 1/2/4/6/8 GPU 的小矩阵。检查结果正确性、每卡 telemetry 和通信计数。

### Step 5：Phase B/C/D/E 正式实验

正式运行建议每次只运行一个 system，GPU 节点上不要同时运行其他 workload：

```bash
bash experiments/scripts/run_distributed_scaling.sh \
  --systems hybridcvdv \
  --gpu-counts 1,2,4,6,8 \
  --phase strong,capacity,throughput,full \
  --config experiments/configs/sc26_distributed_scaling.json \
  --result-root experiments/results/distributed_8xH800
```

然后分别运行 ATLAS 和 BQSim。长任务必须使用 checkpoint/resume，不依赖单个 SSH 会话。

### Step 6：合并和画图

至少生成以下图表：

1. simulation time (stacked bar with compute/communication/sync breakdown) vs GPU count, also a speedup line vs GPU count.Methods including HybridCVDV, ATLAS, BQSim with different colors.
1. total throughput vs GPU count，Methods including HybridCVDV, ATLAS, BQSim with different colors.
2. per-GPU peak memory and total memory，Methods including HybridCVDV, ATLAS, BQSim with different colors.

## 10. 结果目录和统一 schema

建议目录：

```text
experiments/results/distributed_8xH800_<timestamp>/
  metadata/
  configs/
  qasm/
  hybridcvdv/
    strong/g1/
    strong/g2/
    strong/g4/
    strong/g6/
    strong/g8/
    capacity/
    throughput/
  atlas/
  bqsim/
  telemetry/
  logs/
  checkpoints/
  merged/
    manifest.json
    results.csv
    failures.json
    plots/
```

统一结果至少包含：

```json
{
  "system": "hybridcvdv",
  "case_name": "sc26_jch_nq10_nm4_c16",
  "phase": "strong",
  "gpu_count": 4,
  "gpu_ids": [0, 1, 2, 3],
  "status": "ok",
  "cutoff": 16,
  "num_modes": 4,
  "num_qubits": 10,
  "warmup_runs": 2,
  "measured_runs": 10,
  "simulation_ms": {},
  "compute_ms": {},
  "communication_ms": {},
  "throughput": {},
  "memory": {"per_gpu": {}, "aggregate": {}},
  "communication": {},
  "telemetry": {"per_gpu": {}, "aggregate": {}},
  "environment": {}
}
```

## 11. 结果验收标准
指标要求：
加速效果要有，要保证比其他baselines的总模拟时间短。

正式结果必须满足：

1. 每个 `ok` case 都有正确性检查，不只是进程返回 0。
2. 每个 GPU 数都有所有可见 GPU 的 telemetry。
3. HybridCVDV 结果能说明 state 实际分布在哪些 GPU，而不仅是 `num_gpus=N`。
4. communication time 和 bytes 由内部计时点产生，不由 `nvidia-smi` 推断。
5. BQSim 不再输出占位 transfer time。
6. ATLAS 单节点 NCCL 通信经过正确性验证。
7. ATLAS/BQSim 使用同一份 canonical QASM。
8. OOM、unsupported、timeout 和 crash 分开统计。
9. speedup 只对 1 GPU 能成功运行的相同 case 计算。
10. native strong scaling 和多进程 throughput scaling 分图报告。

## 12. 推荐执行顺序

1. 先扩展 all-GPU telemetry 和统一结果 schema。
2. 给 HybridCVDV 增加 per-GPU state-pool 和 P2P 统计。
3. 修复 ATLAS 单节点 NCCL 以及内部 compute/communication 计时。
4. 修复 BQSim 占位计时和输出冲突。
5. 生成统一 config 和 canonical QASM。
6. 在 8×H800 上完成 Phase A。
7. Phase A 全部正确后，再跑 Phase B/C/D/E。

当前不建议直接把已有 `run_gpu_benchmark_matrix.sh` 放到 8 卡节点运行完整矩阵，因为它只能采样一张 GPU，也不能证明多卡是否实际参与计算。应先完成第 8 节的最小脚本和计时改造。
