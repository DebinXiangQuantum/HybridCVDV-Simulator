#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <stdexcept>
#include "cv_state_pool.h"

/**
 * Level 0: 对角门 (Diagonal Gates) GPU内核
 */

__global__ void apply_kerr_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    double chi
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    cuDoubleComplex* psi = &all_states_data[state_idx * d_trunc];

    double phase = chi * n * n;
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), -sin(phase));

    cuDoubleComplex current_val = psi[n];
    psi[n] = make_cuDoubleComplex(
        current_val.x * phase_factor.x - current_val.y * phase_factor.y,
        current_val.x * phase_factor.y + current_val.y * phase_factor.x
    );
}

__global__ void apply_phase_rotation_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    double theta
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    cuDoubleComplex* psi = &all_states_data[state_idx * d_trunc];

    double phase = theta * n;
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), -sin(phase));

    cuDoubleComplex current_val = psi[n];
    psi[n] = make_cuDoubleComplex(
        current_val.x * phase_factor.x - current_val.y * phase_factor.y,
        current_val.x * phase_factor.y + current_val.y * phase_factor.x
    );
}

__global__ void apply_conditional_parity_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    double parity
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    cuDoubleComplex* psi = &all_states_data[state_idx * d_trunc];

    double phase = parity * M_PI * (n % 2);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), -sin(phase));

    cuDoubleComplex current_val = psi[n];
    psi[n] = make_cuDoubleComplex(
        current_val.x * phase_factor.x - current_val.y * phase_factor.y,
        current_val.x * phase_factor.y + current_val.y * phase_factor.x
    );
}

void apply_phase_rotation(CVStatePool* state_pool, const int* target_indices,
                         int batch_size, double theta) {
    // 假设 target_indices 是设备指针
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, batch_size);

    apply_phase_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size, theta
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Phase Rotation kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void apply_kerr_gate(CVStatePool* state_pool, const int* target_indices,
                    int batch_size, double chi) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, batch_size);

    apply_kerr_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size, chi
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kerr gate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void apply_conditional_parity(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, double parity) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, batch_size);

    apply_conditional_parity_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size, parity
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Conditional Parity kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
