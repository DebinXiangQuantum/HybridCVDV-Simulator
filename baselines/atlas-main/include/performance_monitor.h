#ifndef _PERFORMANCE_MONITOR_H_
#define _PERFORMANCE_MONITOR_H_

#include <cuda_runtime.h>
#include <nvml.h>
#include <vector>
#include <string>
#include <chrono>
#include <map>

namespace sim {

struct PerformanceMetrics {
  // 时间指标 (秒)
  double total_time = 0.0;
  double compile_time = 0.0;
  double simulate_time = 0.0;
  double h2d_transfer_time = 0.0;  // Host to Device
  double d2h_transfer_time = 0.0;  // Device to Host
  double d2d_transfer_time = 0.0;  // Device to Device (NCCL)
  double compute_time = 0.0;       // 纯GPU计算时间
  size_t h2d_bytes = 0;
  size_t d2h_bytes = 0;
  size_t d2d_bytes = 0;
  size_t transfer_count = 0;

  // 内存指标 (字节)
  size_t cpu_memory_peak = 0;      // CPU内存峰值
  size_t gpu_memory_peak = 0;      // GPU显存峰值
  size_t gpu_memory_allocated = 0; // GPU显存分配总量
  std::vector<size_t> gpu_memory_peak_per_device;

  // GPU 运行指标
  double gpu_power_peak_w = 0.0;        // GPU 功耗峰值 (瓦)
  unsigned int gpu_utilization_peak_pct = 0; // GPU 利用率峰值 (%)
  double gpu_power_avg_w = 0.0;         // GPU 平均功耗 (瓦)
  double gpu_utilization_avg_pct = 0.0; // GPU 平均利用率 (%)

  // 其他指标
  int num_gates = 0;
  int num_qubits = 0;
  int num_devices = 0;
};

class PerformanceMonitor {
public:
  PerformanceMonitor(int num_devices = 1);
  ~PerformanceMonitor();

  // 时间测量
  void start_timer(const std::string& name);
  void stop_timer(const std::string& name);
  double get_elapsed_time(const std::string& name) const;

  // GPU事件测量
  void record_event_start(int device_id, const std::string& event_name);
  void record_event_end(int device_id, const std::string& event_name);
  float get_event_elapsed_time(int device_id, const std::string& event_name);

  // 内存监控
  size_t get_cpu_memory_usage();
  size_t get_gpu_memory_usage(int device_id);
  size_t get_gpu_memory_peak(int device_id);
  void update_memory_peak();   // 同时更新内存、功耗、利用率

  // GPU 运行指标
  void sample_gpu_metrics();   // 采样 GPU 功耗和利用率

  // 获取完整指标
  PerformanceMetrics get_metrics() const { return metrics_; }
  void set_metrics(const PerformanceMetrics& metrics) { metrics_ = metrics; }
  void add_compute_time_ms(double elapsed_ms) { metrics_.compute_time += elapsed_ms / 1000.0; }
  void add_h2d_time_ms(double elapsed_ms, size_t bytes) {
    metrics_.h2d_transfer_time += elapsed_ms / 1000.0;
    metrics_.h2d_bytes += bytes;
    ++metrics_.transfer_count;
  }
  void add_d2h_time_ms(double elapsed_ms, size_t bytes) {
    metrics_.d2h_transfer_time += elapsed_ms / 1000.0;
    metrics_.d2h_bytes += bytes;
    ++metrics_.transfer_count;
  }
  void add_d2d_time_ms(double elapsed_ms, size_t bytes) {
    metrics_.d2d_transfer_time += elapsed_ms / 1000.0;
    metrics_.d2d_bytes += bytes;
    ++metrics_.transfer_count;
  }

  // 打印报告
  void print_report(const std::string& circuit_name) const;
  void save_to_csv(const std::string& filename, const std::string& circuit_name,
                   const std::string& circuit_type) const;

private:
  struct TimerInfo {
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    bool is_running = false;
  };

  struct EventInfo {
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    bool is_recorded = false;
  };

  int num_devices_;
  PerformanceMetrics metrics_;

  // CPU计时器
  std::map<std::string, TimerInfo> timers_;

  // GPU事件 (per device)
  std::vector<std::map<std::string, EventInfo>> gpu_events_;

  // 内存追踪
  std::vector<size_t> gpu_memory_baseline_;
  std::vector<size_t> gpu_memory_peak_;

  // NVML 设备句柄
  std::vector<nvmlDevice_t> nvml_devices_;
  bool nvml_initialized_ = false;

  // GPU 指标采样累积（用于计算平均值）
  double gpu_power_sum_w_ = 0.0;
  unsigned long long gpu_util_sum_pct_ = 0;
  int gpu_sample_count_ = 0;
};

} // namespace sim

#endif // _PERFORMANCE_MONITOR_H_
