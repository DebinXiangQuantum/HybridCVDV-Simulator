#include <cuda_runtime.h>
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include <iostream>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

__global__ void displacement_simple_kernel(
    double** d_ptrs,
    const double* d_old_buffer,
    int dim,
    const int* state_ids,
    int batch_size,
    const double* S,
    const double* dg
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int state_id = state_ids[batch_idx];
    double* d = d_ptrs[state_id];
    const double* d_old = d_old_buffer + (static_cast<size_t>(batch_idx) * dim);

    if (!d) return;
    
    for (int i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += S[i * dim + j] * d_old[j];
        }
        d[i] = sum + dg[i];
    }
}

__global__ void copy_displacement_kernel(
    double** d_ptrs,
    double* d_old_buffer,
    int dim,
    const int* state_ids,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int state_id = state_ids[batch_idx];
    double* d = d_ptrs[state_id];
    double* d_old = d_old_buffer + (static_cast<size_t>(batch_idx) * dim);

    if (!d) return;

    for (int i = 0; i < dim; ++i) {
        d_old[i] = d[i];
    }
}

__global__ void covariance_step1_kernel(
    double** sigma_ptrs,
    double* temp_buffer, 
    int dim,
    const int* state_ids,
    int batch_size,
    const double* S
) {
    int batch_idx = blockIdx.z;
    int i = blockIdx.y;
    int l = blockIdx.x;
    
    if (batch_idx >= batch_size || i >= dim || l >= dim) return;
    
    int state_id = state_ids[batch_idx];
    double* sig = sigma_ptrs[state_id];
    double* temp = temp_buffer + (static_cast<size_t>(batch_idx) * dim * dim);

    if (!sig) return;
    
    double sum = 0.0;
    for (int k = 0; k < dim; ++k) {
        sum += S[i * dim + k] * sig[k * dim + l];
    }
    temp[i * dim + l] = sum;
}

__global__ void covariance_step2_kernel(
    double** sigma_ptrs,
    const double* temp_buffer,
    int dim,
    const int* state_ids,
    int batch_size,
    const double* S
) {
    int batch_idx = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x;
    
    if (batch_idx >= batch_size || i >= dim || j >= dim) return;
    
    int state_id = state_ids[batch_idx];
    double* sig = sigma_ptrs[state_id];
    const double* temp = temp_buffer + (static_cast<size_t>(batch_idx) * dim * dim);

    if (!sig) return;
    
    double sum = 0.0;
    for (int l = 0; l < dim; ++l) {
        sum += temp[i * dim + l] * S[j * dim + l]; // S^T element is S[j][l]
    }
    sig[i * dim + j] = sum;
}

void apply_batched_symplectic_update(
    GaussianStatePool* pool,
    const int* d_state_ids, // Device pointer
    int batch_size,
    const double* d_S,    // Device pointer
    const double* d_dg    // Device pointer
) {
    if (batch_size <= 0) return;
    
    int dim = 2 * pool->get_num_qumodes();
    double** d_ptrs = pool->get_d_ptrs_device();
    double** sigma_ptrs = pool->get_sig_ptrs_device();

    double* d_old = nullptr;
    CHECK_CUDA(cudaMalloc(&d_old, static_cast<size_t>(batch_size) * dim * sizeof(double)));

    copy_displacement_kernel<<<batch_size, 1>>>(
        d_ptrs, d_old, dim, d_state_ids, batch_size
    );
    CHECK_CUDA(cudaGetLastError());

    displacement_simple_kernel<<<batch_size, 1>>>(
        d_ptrs, d_old, dim, d_state_ids, batch_size, d_S, d_dg
    );
    CHECK_CUDA(cudaGetLastError());
    
    double* d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, (size_t)batch_size * dim * dim * sizeof(double)));
    
    dim3 grid(dim, dim, batch_size);
    covariance_step1_kernel<<<grid, 1>>>(
        sigma_ptrs, d_temp, dim, d_state_ids, batch_size, d_S
    );
    CHECK_CUDA(cudaGetLastError());
    
    covariance_step2_kernel<<<grid, 1>>>(
        sigma_ptrs, d_temp, dim, d_state_ids, batch_size, d_S
    );
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaFree(d_old));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaDeviceSynchronize());
}
