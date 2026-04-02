#include "quantum_state.h"
#include <stdexcept>
#include <cmath>

namespace gpu {

QuantumState::QuantumState(int num_qubits, int num_qumodes, int cutoff)
    : num_qubits_(num_qubits), num_qumodes_(num_qumodes), cutoff_(cutoff) {
    
    // 计算总维度: 2^num_qubits * cutoff^num_qumodes
    dim_ = 1 << num_qubits;
    for (int i = 0; i < num_qumodes; ++i) {
        dim_ *= cutoff;
    }
    
    cudaError_t err = cudaMalloc(&d_data_, dim_ * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory");
    }
}

QuantumState::~QuantumState() {
    if (d_data_) {
        cudaFree(d_data_);
    }
}

void QuantumState::initialize_zero() {
    cudaMemset(d_data_, 0, dim_ * sizeof(cuDoubleComplex));
}

void QuantumState::initialize_ground() {
    initialize_zero();
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cudaMemcpy(d_data_, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
}

void QuantumState::upload(const std::vector<std::complex<double>>& host_data) {
    if (host_data.size() != static_cast<size_t>(dim_)) {
        throw std::invalid_argument("Host data size mismatch");
    }
    
    std::vector<cuDoubleComplex> temp(dim_);
    for (int i = 0; i < dim_; ++i) {
        temp[i] = make_cuDoubleComplex(host_data[i].real(), host_data[i].imag());
    }
    
    cudaMemcpy(d_data_, temp.data(), dim_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
}

void QuantumState::download(std::vector<std::complex<double>>& host_data) const {
    host_data.resize(dim_);
    
    std::vector<cuDoubleComplex> temp(dim_);
    cudaMemcpy(temp.data(), d_data_, dim_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < dim_; ++i) {
        host_data[i] = std::complex<double>(cuCreal(temp[i]), cuCimag(temp[i]));
    }
}

} // namespace gpu
