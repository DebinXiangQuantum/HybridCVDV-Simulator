#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <stdexcept>
#include "cv_state_pool.h"

/**
 * Level 1: 梯算符门 (Ladder/Shift Gates) GPU内核
 */

/**
 * 光子创建算符 a† 内核 (使用共享内存避免Race Condition)
 * ψ_out[n] = √n · ψ_in[n-1]
 */
__global__ void apply_creation_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const int* target_indices,
    int batch_size
) {
    // 动态分配共享内存用于存储当前状态向量
    extern __shared__ cuDoubleComplex shared_state[];

    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int tid = threadIdx.x;
    int n = tid; // 假设一个Block处理一个状态 (BlockSize >= d_trunc)

    // 将全局内存数据加载到共享内存
    if (n < d_trunc) {
        shared_state[n] = all_states_data[state_idx * d_trunc + n];
    } else {
        // 超出维度的部分清零（如果BlockSize > d_trunc）
        // 虽然不需要，但是个好习惯
    }
    __syncthreads(); // 确保加载完成

    if (n < d_trunc) {
        cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
        if (n > 0) {
            double coeff = sqrt(static_cast<double>(n));
            // 从共享内存读取旧值，避免Race Condition
            cuDoubleComplex prev_val = shared_state[n - 1];
            result = make_cuDoubleComplex(
                coeff * prev_val.x,
                coeff * prev_val.y
            );
        }
        // 写回全局内存
        all_states_data[state_idx * d_trunc + n] = result;
    }
}

/**
 * 光子湮灭算符 a 内核 (使用共享内存避免Race Condition)
 * ψ_out[n] = √(n+1) · ψ_in[n+1]
 */
__global__ void apply_annihilation_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const int* target_indices,
    int batch_size
) {
    extern __shared__ cuDoubleComplex shared_state[];

    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int tid = threadIdx.x;
    int n = tid;

    if (n < d_trunc) {
        shared_state[n] = all_states_data[state_idx * d_trunc + n];
    }
    __syncthreads();

    if (n < d_trunc) {
        cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
        if (n < d_trunc - 1) {
            double coeff = sqrt(static_cast<double>(n + 1));
            cuDoubleComplex next_val = shared_state[n + 1];
            result = make_cuDoubleComplex(
                coeff * next_val.x,
                coeff * next_val.y
            );
        }
        all_states_data[state_idx * d_trunc + n] = result;
    }
}

/**
 * 主机端接口：应用光子创建算符 a†
 */
void apply_creation_operator(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    // 假设 d_trunc 不超过 1024 (最大线程数)
    // 如果超过，需要分块处理（这里简化处理，仅支持 d_trunc <= 1024）
    if (state_pool->d_trunc > 1024) {
        throw std::runtime_error("Current implementation only supports d_trunc <= 1024");
    }

    dim3 block_dim(state_pool->d_trunc); // 每个Block处理一个完整状态
    // 向上取整到32的倍数以优化warp
    if (block_dim.x < 32) block_dim.x = 32;
    else if (block_dim.x % 32 != 0) block_dim.x = ((block_dim.x + 31) / 32) * 32;

    dim3 grid_dim(1, batch_size); // grid.x=1 因为一个Block处理整个状态
    size_t shared_mem_size = state_pool->d_trunc * sizeof(cuDoubleComplex);

    apply_creation_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Creation operator kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用光子湮灭算符 a
 */
void apply_annihilation_operator(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    if (state_pool->d_trunc > 1024) {
        throw std::runtime_error("Current implementation only supports d_trunc <= 1024");
    }

    dim3 block_dim(state_pool->d_trunc);
    if (block_dim.x < 32) block_dim.x = 32;
    else if (block_dim.x % 32 != 0) block_dim.x = ((block_dim.x + 31) / 32) * 32;

    dim3 grid_dim(1, batch_size);
    size_t shared_mem_size = state_pool->d_trunc * sizeof(cuDoubleComplex);

    apply_annihilation_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Annihilation operator kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
