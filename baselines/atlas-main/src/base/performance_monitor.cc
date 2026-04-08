#include "performance_monitor.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/resource.h>
#include <algorithm>
#include <cstdlib>

namespace sim {

PerformanceMonitor::PerformanceMonitor(int num_devices)
    : num_devices_(num_devices) {
  gpu_events_.resize(num_devices_);
  gpu_memory_baseline_.resize(num_devices_, 0);
  gpu_memory_peak_.resize(num_devices_, 0);

  // 记录初始GPU内存状态
  for (int i = 0; i < num_devices_; i++) {
    cudaSetDevice(i);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    gpu_memory_baseline_[i] = total_mem - free_mem;
  }

  // 初始化 NVML（获取 GPU 功耗、利用率等）
  nvmlReturn_t nvml_ret = nvmlInit_v2();
  if (nvml_ret == NVML_SUCCESS) {
    nvml_initialized_ = true;
    nvml_devices_.resize(num_devices_);
    // 用 CUDA_VISIBLE_DEVICES 映射：NVML 索引 ≠ CUDA 索引
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    for (int i = 0; i < num_devices_; i++) {
      unsigned int nvml_idx = i;
      if (visible) {
        // 解析 CUDA_VISIBLE_DEVICES 获取真实 GPU 索引
        std::string vis(visible);
        int idx = 0;
        size_t start = 0;
        for (int j = 0; j <= i; j++) {
          size_t comma = vis.find(',', start);
          if (j == i) {
            nvml_idx = std::stoi(vis.substr(start, comma - start));
          }
          start = (comma != std::string::npos) ? comma + 1 : vis.size();
        }
      }
      nvmlDeviceGetHandleByIndex_v2(nvml_idx, &nvml_devices_[i]);
    }
  } else {
    std::cerr << "Warning: NVML initialization failed (code " << nvml_ret
              << "), GPU power/utilization metrics unavailable\n";
  }

  metrics_.num_devices = num_devices_;
}

PerformanceMonitor::~PerformanceMonitor() {
  // 清理所有CUDA事件
  for (int dev = 0; dev < num_devices_; dev++) {
    cudaSetDevice(dev);
    for (auto& pair : gpu_events_[dev]) {
      if (pair.second.is_recorded) {
        cudaEventDestroy(pair.second.start_event);
        cudaEventDestroy(pair.second.end_event);
      }
    }
  }
  // 关闭 NVML
  if (nvml_initialized_) {
    nvmlShutdown();
  }
}

void PerformanceMonitor::start_timer(const std::string& name) {
  timers_[name].start = std::chrono::high_resolution_clock::now();
  timers_[name].is_running = true;
}

void PerformanceMonitor::stop_timer(const std::string& name) {
  if (timers_.find(name) != timers_.end() && timers_[name].is_running) {
    timers_[name].end = std::chrono::high_resolution_clock::now();
    timers_[name].is_running = false;
  }
}

double PerformanceMonitor::get_elapsed_time(const std::string& name) const {
  auto it = timers_.find(name);
  if (it != timers_.end() && !it->second.is_running) {
    return std::chrono::duration<double>(it->second.end - it->second.start).count();
  }
  return 0.0;
}

void PerformanceMonitor::record_event_start(int device_id, const std::string& event_name) {
  if (device_id >= num_devices_) return;

  cudaSetDevice(device_id);
  auto& event_info = gpu_events_[device_id][event_name];

  if (!event_info.is_recorded) {
    cudaEventCreate(&event_info.start_event);
    cudaEventCreate(&event_info.end_event);
    event_info.is_recorded = true;
  }

  cudaEventRecord(event_info.start_event);
}

void PerformanceMonitor::record_event_end(int device_id, const std::string& event_name) {
  if (device_id >= num_devices_) return;

  cudaSetDevice(device_id);
  auto it = gpu_events_[device_id].find(event_name);
  if (it != gpu_events_[device_id].end() && it->second.is_recorded) {
    cudaEventRecord(it->second.end_event);
  }
}

float PerformanceMonitor::get_event_elapsed_time(int device_id, const std::string& event_name) {
  if (device_id >= num_devices_) return 0.0f;

  auto it = gpu_events_[device_id].find(event_name);
  if (it != gpu_events_[device_id].end() && it->second.is_recorded) {
    cudaEventSynchronize(it->second.end_event);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, it->second.start_event, it->second.end_event);
    return elapsed_ms;
  }
  return 0.0f;
}

size_t PerformanceMonitor::get_cpu_memory_usage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  // ru_maxrss 在 Linux 上是 KB，在 macOS 上是字节
#ifdef __APPLE__
  return usage.ru_maxrss;
#else
  return usage.ru_maxrss * 1024;
#endif
}

size_t PerformanceMonitor::get_gpu_memory_usage(int device_id) {
  if (device_id >= num_devices_) return 0;

  cudaSetDevice(device_id);
  cudaDeviceSynchronize();
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t current = total_mem - free_mem;
  // 减去 CUDA context 等基线开销，得到实际应用分配的 GPU 显存
  if (current > gpu_memory_baseline_[device_id]) {
    return current - gpu_memory_baseline_[device_id];
  }
  return 0;
}

size_t PerformanceMonitor::get_gpu_memory_peak(int device_id) {
  if (device_id >= num_devices_) return 0;
  return gpu_memory_peak_[device_id];
}

void PerformanceMonitor::update_memory_peak() {
  // 更新CPU内存峰值
  size_t cpu_mem = get_cpu_memory_usage();
  if (cpu_mem > metrics_.cpu_memory_peak) {
    metrics_.cpu_memory_peak = cpu_mem;
  }

  // 更新GPU内存峰值
  size_t total_gpu_mem = 0;
  for (int i = 0; i < num_devices_; i++) {
    size_t gpu_mem = get_gpu_memory_usage(i);
    if (gpu_mem > gpu_memory_peak_[i]) {
      gpu_memory_peak_[i] = gpu_mem;
    }
    total_gpu_mem += gpu_mem;
  }

  if (total_gpu_mem > metrics_.gpu_memory_peak) {
    metrics_.gpu_memory_peak = total_gpu_mem;
  }

  // 同时采样 GPU 功耗和利用率
  sample_gpu_metrics();
}

void PerformanceMonitor::sample_gpu_metrics() {
  if (!nvml_initialized_) return;

  double max_power_w = 0.0;
  unsigned int max_util_pct = 0;
  double sum_power_w = 0.0;
  unsigned int sum_util_pct = 0;

  for (int i = 0; i < num_devices_; i++) {
    // 功耗 (毫瓦 → 瓦)
    unsigned int power_mw = 0;
    if (nvmlDeviceGetPowerUsage(nvml_devices_[i], &power_mw) == NVML_SUCCESS) {
      double power_w = power_mw / 1000.0;
      sum_power_w += power_w;
      if (power_w > max_power_w) max_power_w = power_w;
    }

    // GPU 利用率
    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(nvml_devices_[i], &util) == NVML_SUCCESS) {
      sum_util_pct += util.gpu;
      if (util.gpu > max_util_pct) max_util_pct = util.gpu;
    }
  }

  // 更新峰值
  if (max_power_w > metrics_.gpu_power_peak_w) {
    metrics_.gpu_power_peak_w = max_power_w;
  }
  if (max_util_pct > metrics_.gpu_utilization_peak_pct) {
    metrics_.gpu_utilization_peak_pct = max_util_pct;
  }

  // 累积用于计算平均值
  gpu_power_sum_w_ += sum_power_w;
  gpu_util_sum_pct_ += sum_util_pct;
  gpu_sample_count_++;

  // 计算平均值
  if (gpu_sample_count_ > 0) {
    metrics_.gpu_power_avg_w = gpu_power_sum_w_ / gpu_sample_count_;
    metrics_.gpu_utilization_avg_pct = static_cast<double>(gpu_util_sum_pct_) / gpu_sample_count_;
  }
}

void PerformanceMonitor::print_report(const std::string& circuit_name) const {
  std::cout << "\n========== 性能分析报告 ==========\n";
  std::cout << "电路名称: " << circuit_name << "\n";
  std::cout << "量子比特数: " << metrics_.num_qubits << "\n";
  std::cout << "门数量: " << metrics_.num_gates << "\n";
  std::cout << "GPU设备数: " << metrics_.num_devices << "\n";
  std::cout << "\n--- 时间指标 (秒) ---\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "总时间:           " << metrics_.total_time << "\n";
  std::cout << "编译时间:         " << metrics_.compile_time << "\n";
  std::cout << "模拟时间:         " << metrics_.simulate_time << "\n";

  std::cout << "\n--- 内存指标 ---\n";
  std::cout << "CPU内存峰值:   " << (metrics_.cpu_memory_peak / 1024.0 / 1024.0) << " MB\n";
  std::cout << "GPU显存峰值:   " << (metrics_.gpu_memory_peak / 1024.0 / 1024.0) << " MB\n";

  std::cout << "\n--- GPU 运行指标 ---\n";
  std::cout << std::setprecision(1);
  std::cout << "GPU功耗峰值:   " << metrics_.gpu_power_peak_w << " W\n";
  std::cout << "GPU平均功耗:   " << metrics_.gpu_power_avg_w << " W\n";
  std::cout << "GPU利用率峰值: " << metrics_.gpu_utilization_peak_pct << " %\n";
  std::cout << "GPU平均利用率: " << metrics_.gpu_utilization_avg_pct << " %\n";
  std::cout << "===================================\n\n";
}


} // namespace sim
