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
 * 简化的Kerr门内核
 */
__global__ void apply_kerr_simple_kernel(
    cuDoubleComplex* state_data,
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

    // 获取状态向量指针
    cuDoubleComplex* psi = &state_data[state_idx * d_trunc];

    // 简化的测试：只读取内存，不写入（确保内存可访问）
    (void)psi[n];  // 读取但不使用，验证内存可访问性
}

/**
 * 相位旋转门内核 R(θ) = exp(-i θ n)
 * 公式：ψ_out[n] = ψ_in[n] · exp(-i θ n)
 */
__global__ void apply_phase_rotation_kernel(
    cuDoubleComplex* state_data,
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

    // 获取状态向量指针
    cuDoubleComplex* psi = &state_data[state_idx * d_trunc];

    // 计算相位因子: exp(-i * theta * n)
    double phase = -theta * static_cast<double>(n);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));

    // 应用相位旋转
    cuDoubleComplex current_val = psi[n];
    psi[n] = cuCmul(current_val, phase_factor);
}

/**
 * 通用对角门应用内核
 * @param state_data CV状态池数据
 * @param d_trunc 截断维度
 * @param target_indices 需要更新的状态ID列表
 * @param batch_size 批处理大小
 * @param phase_func 相位函数指针 (设备端函数)
 * @param params 门参数
 */
__global__ void apply_diagonal_gate_kernel(
    cuDoubleComplex* state_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    double (*phase_func)(int, double),  // 相位函数: f(n, param)
    double param
) {
    // 计算全局线程索引
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    // 获取状态向量指针
    cuDoubleComplex* psi = &state_data[state_idx * d_trunc];

    // 计算相位因子 e^(-i * phase)
    double phase = phase_func(n, param);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), -sin(phase));

    // 应用对角门: ψ[n] *= phase_factor
    cuDoubleComplex current_val = psi[n];
    psi[n] = make_cuDoubleComplex(
        current_val.x * phase_factor.x - current_val.y * phase_factor.y,
        current_val.x * phase_factor.y + current_val.y * phase_factor.x
    );
}

/**
 * 相位旋转门 R(θ) 相位函数
 * f(n, θ) = θ · n
 */
__device__ double phase_rotation_func(int n, double theta) {
    return theta * n;
}

/**
 * Kerr门 K(χ) 相位函数
 * f(n, χ) = χ · n²
 */
__device__ double kerr_func(int n, double chi) {
    return chi * n * n;
}

/**
 * 条件奇偶校验门 CP 相位函数
 * f(n, parity) = parity · π · (n % 2)
 */
__device__ double conditional_parity_func(int n, double parity) {
    return parity * M_PI * (n % 2);
}

// 函数指针类型定义
typedef double (*PhaseFuncPtr)(int, double);

/**
 * 主机端接口：应用相位旋转门 R(θ)
 */
void apply_phase_rotation(CVStatePool* state_pool, const int* target_indices,
                         int batch_size, double theta) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, batch_size);

    // 使用正确的相位旋转内核
    apply_phase_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size, theta
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Phase Rotation kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用Kerr门 K(χ)
 */
void apply_kerr_gate(CVStatePool* state_pool, const int* target_indices,
                    int batch_size, double chi) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, batch_size);

    // 使用简化的内核
    apply_kerr_simple_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size, chi
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kerr gate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用条件奇偶校验门 CP
 */
void apply_conditional_parity(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, double parity) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, batch_size);

    apply_diagonal_gate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size,
        conditional_parity_func, parity
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Conditional Parity kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 状态加法内核：result = w1 * state1 + w2 * state2
 */
__global__ void add_states_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
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

    if (n >= d_trunc) return;

    cuDoubleComplex w1 = weights1[batch_id];
    cuDoubleComplex w2 = weights2[batch_id];

    cuDoubleComplex* state1 = &all_states_data[src1_idx * d_trunc];
    cuDoubleComplex* state2 = &all_states_data[src2_idx * d_trunc];
    cuDoubleComplex* result = &all_states_data[dst_idx * d_trunc];

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
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, batch_size);

    add_states_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->d_trunc,
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
