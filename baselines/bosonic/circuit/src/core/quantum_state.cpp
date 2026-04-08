#include "quantum_state.h"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace gpu {

std::atomic<size_t> QuantumState::process_allocated_gpu_bytes_(0);

QuantumState::QuantumState(int num_qubits, int num_qumodes, int cutoff)
    : num_qubits_(num_qubits), num_qumodes_(num_qumodes), cutoff_(cutoff) {
    
    // 计算总维度: 2^num_qubits * cutoff^num_qumodes
    dim_ = 1 << num_qubits;
    for (int i = 0; i < num_qumodes; ++i) {
        dim_ *= cutoff;
    }
    
    size_t bytes_to_alloc = dim_ * sizeof(cuDoubleComplex);
    cudaError_t err = cudaMalloc(&d_data_, bytes_to_alloc);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory");
    }
    
    process_allocated_gpu_bytes_ += bytes_to_alloc;
    std::cout << "当前进程 GPU 内存分配: " << bytes_to_alloc / (1024 * 1024) << " MB"
              << " (累计: " << process_allocated_gpu_bytes_.load() / (1024 * 1024) << " MB)" << std::endl;
}

QuantumState::~QuantumState() {
    if (d_data_) {
        size_t bytes_to_free = dim_ * sizeof(cuDoubleComplex);
        cudaFree(d_data_);
        process_allocated_gpu_bytes_ -= bytes_to_free;
        std::cout << "当前进程 GPU 内存释放: " << bytes_to_free  << " 字节"
                  << " (剩余: " << process_allocated_gpu_bytes_.load() << " 字节)" << std::endl;
    }
}

void QuantumState::initialize_zero() {
    cudaMemset(d_data_, 0, dim_ * sizeof(cuDoubleComplex));
}

size_t QuantumState::initialize_ground() {
    initialize_zero();
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cudaMemcpy(d_data_, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    return process_allocated_gpu_bytes_.load();
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
