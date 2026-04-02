#include "gpu_state.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace gpu {

GPUState::GPUState(int dim) : dim_(dim), d_data_(nullptr) {
    if (dim_ <= 0) {
        throw std::invalid_argument("Dimension must be positive");
    }
    
    cudaError_t err = cudaMalloc(&d_data_, dim_ * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory: " + std::string(cudaGetErrorString(err)));
    }
}

GPUState::~GPUState() {
    if (d_data_ != nullptr) {
        cudaFree(d_data_);
    }
}

void GPUState::uploadFromHost(const std::vector<std::complex<double>>& host_data) {
    if (host_data.size() != static_cast<size_t>(dim_)) {
        throw std::invalid_argument("Host data size does not match state dimension");
    }
    
    std::vector<cuDoubleComplex> device_data(dim_);
    for (int i = 0; i < dim_; ++i) {
        device_data[i] = make_cuDoubleComplex(host_data[i].real(), host_data[i].imag());
    }
    
    cudaError_t err = cudaMemcpy(d_data_, device_data.data(), dim_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to upload data to GPU: " + std::string(cudaGetErrorString(err)));
    }
}

void GPUState::downloadToHost(std::vector<std::complex<double>>& host_data) const {
    if (host_data.size() != static_cast<size_t>(dim_)) {
        host_data.resize(dim_);
    }
    
    std::vector<cuDoubleComplex> device_data(dim_);
    cudaError_t err = cudaMemcpy(device_data.data(), d_data_, dim_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to download data from GPU: " + std::string(cudaGetErrorString(err)));
    }
    
    for (int i = 0; i < dim_; ++i) {
        host_data[i] = std::complex<double>(cuCreal(device_data[i]), cuCimag(device_data[i]));
    }
}

void GPUState::copyFrom(const GPUState& other) {
    if (other.dim_ != dim_) {
        throw std::invalid_argument("Cannot copy states with different dimensions");
    }
    
    cudaError_t err = cudaMemcpy(d_data_, other.d_data_, dim_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data on GPU: " + std::string(cudaGetErrorString(err)));
    }
}

void GPUState::setZero() {
    cudaError_t err = cudaMemset(d_data_, 0, dim_ * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to zero GPU memory: " + std::string(cudaGetErrorString(err)));
    }
}

} // namespace gpu
