#include "circuit.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace gpu {

Circuit::Circuit(int num_qubits, int num_qumodes, int cutoff)
    : num_qubits_(num_qubits), num_qumodes_(num_qumodes), cutoff_(cutoff),
      transfer_time_ms_(0.0), computation_time_ms_(0.0),
      memory_usage_bytes_(0) {
    state_ = std::make_unique<QuantumState>(num_qubits, num_qumodes, cutoff);
    
    // 测量初始化的传输时间
    auto transfer_start = std::chrono::high_resolution_clock::now();
    state_->initialize_ground();
    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ms_ = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
}

void Circuit::add_gate(std::unique_ptr<Gate> gate) {
    gates_.push_back(std::move(gate));
}

void Circuit::build() {
    std::cout << "电路构建完成，共 " << gates_.size() << " 个门" << std::endl;
}

size_t Circuit::get_system_memory_usage() const {
    size_t current_usage = 0;
    
    // 从系统中获取实际内存使用情况
    #ifdef __linux__
    // Linux系统：读取/proc/self/status文件
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            size_t rss_kb;
            iss >> rss_kb;
            current_usage = rss_kb * 1024; // 转换为字节
            break;
        }
    }
    #elif __APPLE__
    // macOS系统：使用sysctl获取内存使用情况
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
        current_usage = info.resident_size;
    }
    #else
    // 其他系统：返回0
    current_usage = 0;
    #endif
    
    return current_usage;
}

void Circuit::execute() {
    start_time_ = std::chrono::high_resolution_clock::now();
    
    // 重置内存使用统计
    memory_usage_bytes_ = 0;
    
    for (const auto& gate : gates_) {
        auto gate_start = std::chrono::high_resolution_clock::now();
        gate->apply(*state_);
        auto gate_end = std::chrono::high_resolution_clock::now();
        computation_time_ms_ += std::chrono::duration<double, std::milli>(gate_end - gate_start).count();
        
        // 每个门操作后检测内存使用情况，更新峰值
        size_t current_usage = get_system_memory_usage();
        if (current_usage > memory_usage_bytes_) {
            memory_usage_bytes_ = current_usage;
        }
    }
    
    end_time_ = std::chrono::high_resolution_clock::now();
    
    // 最后再次检测内存使用情况，确保捕获到峰值
    size_t final_usage = get_system_memory_usage();
    if (final_usage > memory_usage_bytes_) {
        memory_usage_bytes_ = final_usage;
    }
}

CircuitStats Circuit::get_stats() const {
    CircuitStats stats;
    stats.num_gates = gates_.size();
    stats.total_time_ms = std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    stats.transfer_time_ms = transfer_time_ms_;
    stats.computation_time_ms = computation_time_ms_;
    stats.memory_usage_bytes = memory_usage_bytes_;
    return stats;
}

} // namespace gpu
