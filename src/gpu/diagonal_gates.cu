#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include "cv_state_pool.h"

/**
 * Level 0: 对角门 (Diagonal Gates) GPU内核
 *
 * 特性：对角门在Fock基底是对角阵，不涉及矩阵乘法，仅做Element-wise相乘
 * 典型门：Phase Rotation R(θ), Kerr K(χ), Conditional Parity CP
 *
 * 数学公式：ψ_out[n] = ψ_in[n] · e^(-i · f(n))
 *   - R(θ): f(n) = θ · n
 *   - Kerr: f(n) = χ · n²
 */

/**
 * 相位旋转门内核 R(θ) = exp(-i θ n)
 * 支持动态维度
 */
__global__ void apply_phase_rotation_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double theta
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // 获取该状态的维度
    int current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    // 获取状态向量指针 (使用偏移量)
    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // 计算相位因子: exp(-i * theta * n)
    double phase = -theta * static_cast<double>(n);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));

    // 应用相位旋转
    cuDoubleComplex current_val = psi[n];
    psi[n] = cuCmul(current_val, phase_factor);
}

/**
 * Kerr门内核 K(χ) = exp(-i χ n²)
 * 支持动态维度
 */
__global__ void apply_kerr_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double chi
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // 获取该状态的维度
    int current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    // 获取状态向量指针 (使用偏移量)
    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // f(n, chi) = chi * n * n
    double phase = chi * static_cast<double>(n * n);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), -sin(phase)); // e^(-i * phase)

    cuDoubleComplex current_val = psi[n];
    psi[n] = make_cuDoubleComplex(
        current_val.x * phase_factor.x - current_val.y * phase_factor.y,
        current_val.x * phase_factor.y + current_val.y * phase_factor.x
    );
}

/**
 * 条件奇偶校验门内核 CP
 * f(n, parity) = parity · π · (n % 2)
 * 支持动态维度
 */
__global__ void apply_conditional_parity_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double parity
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // 获取该状态的维度
    int current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    // 获取状态向量指针 (使用偏移量)
    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // f(n, parity) = parity * pi * (n % 2)
    double phase = parity * M_PI * static_cast<double>(n % 2);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), -sin(phase)); // e^(-i * phase)

    cuDoubleComplex current_val = psi[n];
    psi[n] = make_cuDoubleComplex(
        current_val.x * phase_factor.x - current_val.y * phase_factor.y,
        current_val.x * phase_factor.y + current_val.y * phase_factor.x
    );
}

/**
 * 主机端接口：应用相位旋转门 R(θ)
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_phase_rotation(CVStatePool* state_pool, const int* target_indices,
                         int batch_size, double theta) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_phase_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, theta
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Phase Rotation kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Phase Rotation kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用Kerr门 K(χ)
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_kerr_gate(CVStatePool* state_pool, const int* target_indices,
                    int batch_size, double chi) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_kerr_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, chi
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kerr gate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kerr gate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用条件奇偶校验门 CP
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_conditional_parity(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, double parity) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_conditional_parity_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, parity
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Conditional Parity kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Conditional Parity kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 状态加法内核：result = w1 * state1 + w2 * state2
 */
__global__ void add_states_kernel(
    cuDoubleComplex* all_states_data,
    int total_dim,
    const int* src1_indices,
    const cuDoubleComplex* weights1,
    const int* src2_indices,
    const cuDoubleComplex* weights2,
    const int* dst_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int src1_idx = src1_indices[batch_id];
    int src2_idx = src2_indices[batch_id];
    int dst_idx = dst_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= total_dim) return;

    cuDoubleComplex w1 = weights1[batch_id];
    cuDoubleComplex w2 = weights2[batch_id];

    cuDoubleComplex* state1 = &all_states_data[src1_idx * total_dim];
    cuDoubleComplex* state2 = &all_states_data[src2_idx * total_dim];
    cuDoubleComplex* result = &all_states_data[dst_idx * total_dim];

    // result[n] = w1 * state1[n] + w2 * state2[n]
    cuDoubleComplex val1 = cuCmul(w1, state1[n]);
    cuDoubleComplex val2 = cuCmul(w2, state2[n]);
    result[n] = cuCadd(val1, val2);
}

/**
 * 状态加法函数：result = w1 * state1 + w2 * state2
 * @param state_pool 状态池
 * @param src1_indices 源状态1的ID数组（设备指针）
 * @param weights1 权重1数组（设备指针）
 * @param src2_indices 源状态2的ID数组（设备指针）
 * @param weights2 权重2数组（设备指针）
 * @param dst_indices 目标状态ID数组（设备指针）
 * @param batch_size 批次大小
 */
void add_states(CVStatePool* state_pool,
                const int* src1_indices,
                const cuDoubleComplex* weights1,
                const int* src2_indices,
                const cuDoubleComplex* weights2,
                const int* dst_indices,
                int batch_size) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    add_states_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->total_dim,
        src1_indices, weights1,
        src2_indices, weights2,
        dst_indices, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Add states kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
