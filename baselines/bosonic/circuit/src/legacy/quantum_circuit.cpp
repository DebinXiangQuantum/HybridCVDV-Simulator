#include "quantum_circuit.h"
#include "quantum_kernels.cuh"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while (0)

namespace gpu {

// 计算量子态空间维度
int calculate_state_dimension(int num_qubits, int num_qumodes, int cutoff) {
    int qubit_dim = 1 << num_qubits;
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    return qubit_dim * qumode_dim;
}

// 初始化量子态为|00...0>
CUDASparseMatrix initialize_state(int num_qubits, int num_qumodes, int cutoff, double& transfer_time) {
    int state_dim = calculate_state_dimension(num_qubits, num_qumodes, cutoff);
    CUDASparseMatrix state(state_dim, 1);
    state.set(0, 0, Complex(1.0, 0.0));
    
    // 测量上传到GPU的时间
    auto start = std::chrono::high_resolution_clock::now();
    state.uploadToDevice();
    auto end = std::chrono::high_resolution_clock::now();
    transfer_time += std::chrono::duration<double, std::milli>(end - start).count();
    
    return state;
}

// Hadamard门实现
CUDASparseMatrix HadamardGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2; // 假设有2个量子比特
    int qumode_dim = 1;
    for (int i = 0; i < 3; ++i) { // 假设有3个量子模态
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_hadamard_kernel<<<blocks, threads>>>(state.device_data_.values, qubit, qubit_dim, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// S门实现
CUDASparseMatrix PhaseGateS::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int qumode_dim = 1;
    for (int i = 0; i < 3; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_phase_s_kernel<<<blocks, threads>>>(state.device_data_.values, qubit, qubit_dim, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// 条件位移门实现
CUDASparseMatrix ConditionalDisplacementGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int num_qumodes = 3;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    Complex alpha_cuda(displacement_param.real, displacement_param.imag);
    apply_conditional_displacement_kernel<<<blocks, threads>>>(state.device_data_.values, qubit, qumode, alpha_cuda, cutoff, qubit_dim, num_qumodes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// 压缩门实现
CUDASparseMatrix SqueezingGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int num_qumodes = 3;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    Complex xi_cuda(squeezing_param.real, squeezing_param.imag);
    apply_squeezing_kernel<<<blocks, threads>>>(state.device_data_.values, qumode, xi_cuda, cutoff, qubit_dim, num_qumodes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// 相位旋转门实现
CUDASparseMatrix PhaseRotationGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int num_qumodes = 3;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_phase_rotation_kernel<<<blocks, threads>>>(state.device_data_.values, qumode, angle, cutoff, qubit_dim, num_qumodes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// Z旋转门实现
CUDASparseMatrix RotationZGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int qumode_dim = 1;
    for (int i = 0; i < 3; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_rotation_z_kernel<<<blocks, threads>>>(state.device_data_.values, qubit, angle, qubit_dim, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// Jaynes-Cummings门实现
CUDASparseMatrix JaynesCummingsGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int num_qumodes = 3;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_jaynes_cummings_kernel<<<blocks, threads>>>(state.device_data_.values, qubit, qumode, angle, cutoff, qubit_dim, num_qumodes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// 分束器门实现
CUDASparseMatrix BeamSplitterGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int num_qumodes = 3;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_beam_splitter_kernel<<<blocks, threads>>>(state.device_data_.values, qumode1, qumode2, angle, cutoff, qubit_dim, num_qumodes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// 位移门实现
CUDASparseMatrix DisplacementGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int num_qumodes = 3;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    Complex alpha_cuda(displacement_param.real, displacement_param.imag);
    apply_displacement_kernel<<<blocks, threads>>>(state.device_data_.values, qumode, alpha_cuda, cutoff, qubit_dim, num_qumodes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// Pauli Z门实现
CUDASparseMatrix PauliZGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int qumode_dim = 1;
    for (int i = 0; i < 3; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_pauli_z_kernel<<<blocks, threads>>>(state.device_data_.values, qubit, qubit_dim, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// Pauli X门实现
CUDASparseMatrix PauliXGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int qumode_dim = 1;
    for (int i = 0; i < 3; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_pauli_x_kernel<<<blocks, threads>>>(state.device_data_.values, qubit, qubit_dim, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// X旋转门实现
CUDASparseMatrix RotationXGate::apply(const CUDASparseMatrix& state, int cutoff) const {
    int qubit_dim = 1 << 2;
    int qumode_dim = 1;
    for (int i = 0; i < 3; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    apply_rotation_x_kernel<<<blocks, threads>>>(state.device_data_.values, qubit, angle, qubit_dim, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return state;
}

// 量子电路构造函数
QuantumCircuit::QuantumCircuit(int n_qubits, int n_qumodes, int c, int max_states)
    : num_qubits(n_qubits), num_qumodes(n_qumodes), cutoff(c), max_active_states(max_states), transfer_time(0.0), computation_time(0.0) {
    state = initialize_state(num_qubits, num_qumodes, cutoff, transfer_time);
}

// 添加门操作
void QuantumCircuit::add_gate(std::unique_ptr<Gate> gate) {
    gates.push_back(std::move(gate));
}

// 构建电路
void QuantumCircuit::build() {
    // 构建过程，这里可以添加额外的优化
    std::cout << "电路构建完成，共" << gates.size() << "个门" << std::endl;
}

// 执行电路
void QuantumCircuit::execute() {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& gate : gates) {
        auto gate_start = std::chrono::high_resolution_clock::now();
        state = gate->apply(state, cutoff);
        auto gate_end = std::chrono::high_resolution_clock::now();
        computation_time += std::chrono::duration<double, std::milli>(gate_end - gate_start).count();
    }
    
    // 测量从GPU下载数据的时间
    auto download_start = std::chrono::high_resolution_clock::now();
    state.downloadFromDevice();
    auto download_end = std::chrono::high_resolution_clock::now();
    transfer_time += std::chrono::duration<double, std::milli>(download_end - download_start).count();
    
    auto end = std::chrono::high_resolution_clock::now();
    start_time = std::chrono::duration<double, std::milli>(start.time_since_epoch()).count();
    end_time = std::chrono::duration<double, std::milli>(end.time_since_epoch()).count();
}

// 获取电路统计信息
CircuitStats QuantumCircuit::get_stats() const {
    CircuitStats stats;
    stats.num_gates = gates.size();
    stats.active_states = 1; // 简化实现
    return stats;
}

// 获取时间统计信息
TimeStats QuantumCircuit::get_time_stats() const {
    TimeStats stats;
    stats.total_time = end_time - start_time;
    stats.transfer_time = transfer_time;
    stats.computation_time = computation_time;
    return stats;
}

// 获取量子态
const CUDASparseMatrix& QuantumCircuit::get_state() const {
    return state;
}

} // namespace gpu
