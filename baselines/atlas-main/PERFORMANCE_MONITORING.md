# 性能监控集成指南

本文档说明如何在 Atlas 模拟器中集成精确的性能测量功能。

## 已完成的工作

### 1. 新增文件

- `include/performance_monitor.h` - 性能监控类定义
- `src/base/performance_monitor.cc` - 性能监控实现
- `include/performance_monitor_global.h` - 全局访问接口
- `src/base/performance_monitor_global.cc` - 全局访问实现

### 2. 修改的文件

- `examples/mpi-based/run_generated_qasm.cc` - 主程序集成性能监控
- `CMakeLists.txt` - 添加新源文件到编译

### 3. 功能特性

✅ **已实现：**
- CPU 时间测量（编译、模拟、总时间）
- CPU 内存峰值监控
- GPU 内存峰值监控
- CUDA 事件精确计时框架

⚠️ **待实现：**
- GPU-CPU 传输时间测量（需要在 simulator.cu 中添加）
- GPU 计算时间测量（需要在 simulator.cu 中添加）
- Device-to-Device 传输时间（NCCL 通信）

## 如何在 Simulator 中添加性能测量

### 方法 1: 在 InitStateMulti 中测量 H2D 传输

在 `src/base/simulator.cu` 的 `InitStateMulti` 函数中添加：

```cpp
#include "performance_monitor_global.h"

template <typename DT>
bool SimulatorCuQuantum<DT>::InitStateMulti(std::vector<unsigned> const &init_perm) {
  auto* perf = sim::GlobalPerfMonitor::get();
  
  // ... 现有代码 ...
  
  // 在 cudaMemcpy H2D 之前
  if (perf) {
    for (int i = 0; i < n_devices; i++) {
      perf->record_event_start(i, "h2d_init");
    }
  }
  
  // H2D 传输
  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_sv[i], h_sv[i], subSvSize, 
                                      cudaMemcpyHostToDevice, s[i]));
  }
  
  // 记录结束
  if (perf) {
    for (int i = 0; i < n_devices; i++) {
      perf->record_event_end(i, "h2d_init");
      float time_ms = perf->get_event_elapsed_time(i, "h2d_init");
      auto metrics = perf->get_metrics();
      metrics.h2d_transfer_time += time_ms / 1000.0; // 转换为秒
      perf->set_metrics(metrics);
    }
  }
  
  // ... 现有代码 ...
}
```

### 方法 2: 在 ApplyGate 中测量 GPU 计算时间

```cpp
template <typename DT>
bool SimulatorCuQuantum<DT>::ApplyGate(Gate<DT> &gate, int device_id) {
  auto* perf = sim::GlobalPerfMonitor::get();
  
  // 记录计算开始
  if (perf) {
    perf->record_event_start(device_id, "compute_gate");
  }
  
  // ... 现有的门应用代码 ...
  HANDLE_ERROR(custatevecApplyMatrix(...));
  
  // 记录计算结束
  if (perf) {
    perf->record_event_end(device_id, "compute_gate");
    float time_ms = perf->get_event_elapsed_time(device_id, "compute_gate");
    auto metrics = perf->get_metrics();
    metrics.compute_time += time_ms / 1000.0;
    perf->set_metrics(metrics);
  }
  
  return true;
}
```

### 方法 3: 在 ApplyRecordedShuffle 中测量 D2D 传输（NCCL）

```cpp
template <typename DT>
bool SimulatorCuQuantum<DT>::ApplyRecordedShuffle(unsigned global_swap, 
                                                   const std::vector<int2> &local_swap) {
  auto* perf = sim::GlobalPerfMonitor::get();
  
  // ... 现有代码 ...
  
  // 在 NCCL all2all 之前
  if (perf) {
    for (int i = 0; i < n_devices; i++) {
      perf->record_event_start(i, "d2d_shuffle");
    }
  }
  
  // NCCL 通信
  for (int i = 0; i < n_devices; i++) {
    all2all(d_sv[i], sendcount, datatype, recv_buf[i], recvcount, 
            datatype, comms[i], s[i], global_swap, myncclrank);
  }
  
  // 记录结束
  if (perf) {
    for (int i = 0; i < n_devices; i++) {
      perf->record_event_end(i, "d2d_shuffle");
      float time_ms = perf->get_event_elapsed_time(i, "d2d_shuffle");
      auto metrics = perf->get_metrics();
      metrics.d2d_transfer_time += time_ms / 1000.0;
      perf->set_metrics(metrics);
    }
  }
  
  // ... 现有代码 ...
}
```

### 方法 4: 测量 GPU 内存分配

在分配 GPU 内存后立即更新：

```cpp
// 在 cudaMalloc 之后
HANDLE_CUDA_ERROR(cudaMalloc(&d_sv[i], subSvSize));

auto* perf = sim::GlobalPerfMonitor::get();
if (perf) {
  auto metrics = perf->get_metrics();
  metrics.gpu_memory_allocated += subSvSize;
  perf->set_metrics(metrics);
  perf->update_memory_peak();
}
```

## 编译和运行

### 编译

```bash
cd build
cmake ..
make -j 12
```

### 运行

```bash
cd examples/mpi-based
mpirun -np 2 ./run_generated_qasm --import-circuit qft_26 --n 26 --local 24 --device 4 --use-ilp
```

### 输出

程序会生成两个文件：
1. `result/atlas_results_detailed.csv` - 详细的性能指标（包括传输时间、内存等）
2. 控制台输出 - 格式化的性能报告

## 性能指标说明

| 指标 | 说明 | 单位 |
|------|------|------|
| total_time | 总执行时间 | 秒 |
| compile_time | 电路编译时间 | 秒 |
| simulate_time | 模拟执行时间 | 秒 |
| compute_time | 纯 GPU 计算时间 | 秒 |
| h2d_transfer_time | Host to Device 传输时间 | 秒 |
| d2h_transfer_time | Device to Host 传输时间 | 秒 |
| d2d_transfer_time | Device to Device 传输时间（NCCL） | 秒 |
| cpu_memory_peak | CPU 内存峰值 | MB |
| gpu_memory_peak | GPU 显存峰值（所有设备总和） | MB |
| gpu_memory_allocated | GPU 显存分配总量 | MB |

## 注意事项

1. **空指针检查**：在 simulator 中使用 `GlobalPerfMonitor::get()` 时，务必检查返回值是否为 nullptr
2. **多设备支持**：对于多 GPU 场景，需要为每个设备分别记录事件
3. **异步操作**：CUDA 事件会自动处理异步操作的同步
4. **开销**：性能监控本身有轻微开销（< 1%），主要来自事件创建和同步

## 下一步

要完整实现所有性能测量，需要：
1. 在 `src/base/simulator.cu` 中添加上述测量点
2. 重新编译项目
3. 运行测试验证数据准确性

建议先在一个小电路上测试，确保测量逻辑正确后再应用到大规模电路。
