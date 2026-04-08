#include "circuit.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

namespace gpu {

Circuit::Circuit(int num_qubits, int num_qumodes, int cutoff)
    : num_qubits_(num_qubits), num_qumodes_(num_qumodes), cutoff_(cutoff),
      transfer_time_ms_(0.0), computation_time_ms_(0.0),
      memory_usage_bytes_(0),
      memory_monitor_running_(false) {
    state_ = std::make_unique<QuantumState>(num_qubits, num_qumodes, cutoff);
    
    // 测量初始化的传输时间
    auto transfer_start = std::chrono::high_resolution_clock::now();
    memory_usage_bytes_ = state_->initialize_ground();
    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ms_ = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
    
    std::cout << "电路初始化完成，GPU 内存占用: " << memory_usage_bytes_ / (1024 * 1024) << " MB" << std::endl;
}

void Circuit::add_gate(std::unique_ptr<Gate> gate) {
    gates_.push_back(std::move(gate));
}

void Circuit::build() {
    std::cout << "电路构建完成，共 " << gates_.size() << " 个门" << std::endl;
}



void Circuit::execute() {
    start_time_ = std::chrono::high_resolution_clock::now();   
    
    for (const auto& gate : gates_) {
        auto gate_start = std::chrono::high_resolution_clock::now();
        gate->apply(*state_);
        auto gate_end = std::chrono::high_resolution_clock::now();
        computation_time_ms_ += std::chrono::duration<double, std::milli>(gate_end - gate_start).count();
    }
    end_time_ = std::chrono::high_resolution_clock::now();
    std::cout << "电路执行完成，GPU 内存峰值: " << memory_usage_bytes_ << " 字节" << std::endl;
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
