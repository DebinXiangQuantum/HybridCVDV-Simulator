#ifndef CIRCUIT_H
#define CIRCUIT_H

#include "quantum_state.h"
#include "gates/gates.h"
#include <vector>
#include <memory>
#include <chrono>

namespace gpu {

struct CircuitStats {
    int num_gates;
    double total_time_ms;
    double transfer_time_ms;
    double computation_time_ms;
    size_t memory_usage_bytes;
};

class Circuit {
public:
    Circuit(int num_qubits, int num_qumodes, int cutoff);
    
    void add_gate(std::unique_ptr<Gate> gate);
    void build();
    void execute();
    
    CircuitStats get_stats() const;
    const QuantumState& get_state() const { return *state_; }
    
private:
    size_t get_system_memory_usage() const;
    
    int num_qubits_;
    int num_qumodes_;
    int cutoff_;
    std::vector<std::unique_ptr<Gate>> gates_;
    std::unique_ptr<QuantumState> state_;
    
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    double transfer_time_ms_;
    double computation_time_ms_;
    size_t memory_usage_bytes_;
};

} // namespace gpu

#endif // CIRCUIT_H
