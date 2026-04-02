#include "quantum_circuit_new.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while (0)

namespace gpu {

// GPU内核声明
extern "C" {
    void hadamard_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim);
    void phase_s_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim);
    void rotation_z_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim, double angle);
    void rotation_x_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim, double angle);
    void pauli_x_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim);
    void pauli_z_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim);
    void phase_rotation_kernel(cuDoubleComplex* state, int qumode, int num_qubits, int num_qumodes, int cutoff, double angle);
    void displacement_kernel(cuDoubleComplex* state, int qumode, int num_qubits, int num_qumodes, int cutoff, double alpha_real, double alpha_imag);
    void squeezing_kernel(cuDoubleComplex* state, int qumode, int num_qubits, int num_qumodes, int cutoff, double r_real, double r_imag);
    void jaynes_cummings_kernel(cuDoubleComplex* state, int qubit, int qumode, int num_qubits, int num_qumodes, int cutoff, double angle);
    void beam_splitter_kernel(cuDoubleComplex* state, int qumode1, int qumode2, int num_qubits, int num_qumodes, int cutoff, double angle);
    void conditional_displacement_kernel(cuDoubleComplex* state, int qubit, int qumode, int num_qubits, int num_qumodes, int cutoff, double alpha_real, double alpha_imag);
}

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
std::unique_ptr<GPUState> initialize_state(int num_qubits, int num_qumodes, int cutoff, double& transfer_time) {
    int state_dim = calculate_state_dimension(num_qubits, num_qumodes, cutoff);
    auto state = std::make_unique<GPUState>(state_dim);
    
    std::vector<std::complex<double>> initial_state(state_dim, std::complex<double>(0.0, 0.0));
    initial_state[0] = std::complex<double>(1.0, 0.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    state->uploadFromHost(initial_state);
    auto end = std::chrono::high_resolution_clock::now();
    transfer_time += std::chrono::duration<double, std::milli>(end - start).count();
    
    return state;
}

// Hadamard门实现
void HadamardGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int qubit_dim = 1 << num_qubits;
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    hadamard_kernel<<<blocks, threads>>>(state.device_data(), qubit, num_qubits, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// S门实现
void PhaseGateS::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int qubit_dim = 1 << num_qubits;
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    phase_s_kernel<<<blocks, threads>>>(state.device_data(), qubit, num_qubits, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 条件位移门实现
void ConditionalDisplacementGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int total_dim = calculate_state_dimension(num_qubits, num_qumodes, cutoff);
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    conditional_displacement_kernel<<<blocks, threads>>>(
        state.device_data(), qubit, qumode, num_qubits, num_qumodes, cutoff,
        displacement_param.real(), displacement_param.imag()
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 压缩门实现
void SqueezingGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int total_dim = calculate_state_dimension(num_qubits, num_qumodes, cutoff);
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    squeezing_kernel<<<blocks, threads>>>(
        state.device_data(), qumode, num_qubits, num_qumodes, cutoff,
        squeezing_param.real(), squeezing_param.imag()
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 相位旋转门实现
void PhaseRotationGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int total_dim = calculate_state_dimension(num_qubits, num_qumodes, cutoff);
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    phase_rotation_kernel<<<blocks, threads>>>(
        state.device_data(), qumode, num_qubits, num_qumodes, cutoff, angle
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Z旋转门实现
void RotationZGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int qubit_dim = 1 << num_qubits;
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    rotation_z_kernel<<<blocks, threads>>>(state.device_data(), qubit, num_qubits, qumode_dim, angle);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Jaynes-Cummings门实现
void JaynesCummingsGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int total_dim = calculate_state_dimension(num_qubits, num_qumodes, cutoff);
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    jaynes_cummings_kernel<<<blocks, threads>>>(
        state.device_data(), qubit, qumode, num_qubits, num_qumodes, cutoff, angle
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 分束器门实现
void BeamSplitterGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int total_dim = calculate_state_dimension(num_qubits, num_qumodes, cutoff);
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    beam_splitter_kernel<<<blocks, threads>>>(
        state.device_data(), qumode1, qumode2, num_qubits, num_qumodes, cutoff, angle
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 位移门实现
void DisplacementGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int total_dim = calculate_state_dimension(num_qubits, num_qumodes, cutoff);
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    displacement_kernel<<<blocks, threads>>>(
        state.device_data(), qumode, num_qubits, num_qumodes, cutoff,
        displacement_param.real(), displacement_param.imag()
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Pauli Z门实现
void PauliZGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int qubit_dim = 1 << num_qubits;
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    pauli_z_kernel<<<blocks, threads>>>(state.device_data(), qubit, num_qubits, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Pauli X门实现
void PauliXGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int qubit_dim = 1 << num_qubits;
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    pauli_x_kernel<<<blocks, threads>>>(state.device_data(), qubit, num_qubits, qumode_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// X旋转门实现
void RotationXGate::apply(GPUState& state, int num_qubits, int num_qumodes, int cutoff) const {
    int qubit_dim = 1 << num_qubits;
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    int total_dim = qubit_dim * qumode_dim;
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    rotation_x_kernel<<<blocks, threads>>>(state.device_data(), qubit, num_qubits, qumode_dim, angle);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
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
    std::cout << "电路构建完成，共" << gates.size() << "个门" << std::endl;
}

// 执行电路
void QuantumCircuit::execute() {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& gate : gates) {
        auto gate_start = std::chrono::high_resolution_clock::now();
        gate->apply(*state, num_qubits, num_qumodes, cutoff);
        auto gate_end = std::chrono::high_resolution_clock::now();
        computation_time += std::chrono::duration<double, std::milli>(gate_end - gate_start).count();
    }
    
    auto download_start = std::chrono::high_resolution_clock::now();
    std::vector<std::complex<double>> host_data;
    state->downloadToHost(host_data);
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
    stats.active_states = 1;
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
const GPUState& QuantumCircuit::get_state() const {
    return *state;
}

} // namespace gpu
