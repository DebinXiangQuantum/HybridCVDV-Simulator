#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "cv_state_pool.h"

/**
 * 量子比特门 GPU 实现
 * 
 * 实现标准的单量子比特门：
 * - Pauli X, Y, Z
 * - S+ (升算符 σ+)
 * - S- (降算符 σ-)
 * - P0 (投影到 |0⟩)
 * - P1 (投影到 |1⟩)
 * 
 * 这些门作用于 2 维希尔伯特空间（量子比特）
 */

/**
 * Pauli X 门内核
 * X = |0⟩⟨1| + |1⟩⟨0|
 * 交换 |0⟩ 和 |1⟩ 态
 */
__global__ void apply_pauli_x_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // 量子比特维度固定为 2
    if (idx >= 2) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // 交换 |0⟩ 和 |1⟩
    if (idx == 0) {
        cuDoubleComplex temp = psi[0];
        psi[0] = psi[1];
        psi[1] = temp;
    }
}

/**
 * Pauli Y 门内核
 * Y = i|0⟩⟨1| - i|1⟩⟨0|
 */
__global__ void apply_pauli_y_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 2) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    if (idx == 0) {
        cuDoubleComplex psi0 = psi[0];
        cuDoubleComplex psi1 = psi[1];
        
        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
        psi[0] = make_cuDoubleComplex(cuCimag(psi1), -cuCreal(psi1));  // -i * psi1
        psi[1] = make_cuDoubleComplex(-cuCimag(psi0), cuCreal(psi0));  // i * psi0
    }
}

/**
 * Pauli Z 门内核
 * Z = |0⟩⟨0| - |1⟩⟨1|
 * 对角门：保持 |0⟩，翻转 |1⟩ 的相位
 */
__global__ void apply_pauli_z_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 2) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    if (idx == 1) {
        psi[1] = make_cuDoubleComplex(-cuCreal(psi[1]), -cuCimag(psi[1]));
    }
}

/**
 * S+ 门内核（升算符）
 * σ+ = |1⟩⟨0| = (X + iY)/2
 */
__global__ void apply_sigma_plus_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 2) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    if (idx == 0) {
        cuDoubleComplex psi0 = psi[0];
        // σ+|0⟩ = |1⟩, σ+|1⟩ = 0
        psi[0] = make_cuDoubleComplex(0.0, 0.0);
        psi[1] = psi0;
    }
}

/**
 * S- 门内核（降算符）
 * σ- = |0⟩⟨1| = (X - iY)/2
 */
__global__ void apply_sigma_minus_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 2) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    if (idx == 0) {
        cuDoubleComplex psi1 = psi[1];
        // σ-|0⟩ = 0, σ-|1⟩ = |0⟩
        psi[0] = psi1;
        psi[1] = make_cuDoubleComplex(0.0, 0.0);
    }
}

/**
 * P0 门内核（投影到 |0⟩）
 * P0 = |0⟩⟨0| = (I + Z)/2
 */
__global__ void apply_projector_0_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 2) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // P0|0⟩ = |0⟩, P0|1⟩ = 0
    if (idx == 1) {
        psi[1] = make_cuDoubleComplex(0.0, 0.0);
    }
}

/**
 * P1 门内核（投影到 |1⟩）
 * P1 = |1⟩⟨1| = (I - Z)/2
 */
__global__ void apply_projector_1_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 2) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // P1|0⟩ = 0, P1|1⟩ = |1⟩
    if (idx == 0) {
        psi[0] = make_cuDoubleComplex(0.0, 0.0);
    }
}

// ==================== 主机端接口 ====================
// Qubit states are 2-dimensional; use block_dim(32) (one warp) instead of 256.

static void launch_qubit_gate_and_sync(cudaError_t launch_err, const char* gate_name) {
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(std::string(gate_name) + " kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(gate_name) + " kernel synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void apply_pauli_x(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(32);
    dim3 grid_dim(1, batch_size);
    apply_pauli_x_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->state_offsets,
        state_pool->state_dims, target_indices, batch_size);
    launch_qubit_gate_and_sync(cudaGetLastError(), "Pauli X");
}

void apply_pauli_y(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(32);
    dim3 grid_dim(1, batch_size);
    apply_pauli_y_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->state_offsets,
        state_pool->state_dims, target_indices, batch_size);
    launch_qubit_gate_and_sync(cudaGetLastError(), "Pauli Y");
}

void apply_pauli_z(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(32);
    dim3 grid_dim(1, batch_size);
    apply_pauli_z_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->state_offsets,
        state_pool->state_dims, target_indices, batch_size);
    launch_qubit_gate_and_sync(cudaGetLastError(), "Pauli Z");
}

void apply_sigma_plus(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(32);
    dim3 grid_dim(1, batch_size);
    apply_sigma_plus_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->state_offsets,
        state_pool->state_dims, target_indices, batch_size);
    launch_qubit_gate_and_sync(cudaGetLastError(), "Sigma Plus");
}

void apply_sigma_minus(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(32);
    dim3 grid_dim(1, batch_size);
    apply_sigma_minus_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->state_offsets,
        state_pool->state_dims, target_indices, batch_size);
    launch_qubit_gate_and_sync(cudaGetLastError(), "Sigma Minus");
}

void apply_projector_0(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(32);
    dim3 grid_dim(1, batch_size);
    apply_projector_0_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->state_offsets,
        state_pool->state_dims, target_indices, batch_size);
    launch_qubit_gate_and_sync(cudaGetLastError(), "Projector 0");
}

void apply_projector_1(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(32);
    dim3 grid_dim(1, batch_size);
    apply_projector_1_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->state_offsets,
        state_pool->state_dims, target_indices, batch_size);
    launch_qubit_gate_and_sync(cudaGetLastError(), "Projector 1");
}
