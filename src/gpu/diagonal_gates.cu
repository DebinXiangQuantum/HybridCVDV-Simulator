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
    CVStatePool* state_pool,
    const int* target_indices,
    int batch_size,
    double chi
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= state_pool->d_trunc) return;

    // 获取状态向量指针
    cuDoubleComplex* psi = &state_pool->data[state_idx * state_pool->d_trunc];

    // 简化的测试：只读取内存，不写入
    cuDoubleComplex current_val = psi[n];
    // 不做任何修改
}

/**
 * 简化的相位旋转门内核 (避免函数指针)
 */
__global__ void apply_phase_rotation_simple_kernel(
    CVStatePool* state_pool,
    const int* target_indices,
    int batch_size,
    double theta
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= state_pool->d_trunc) return;

    // 获取状态向量指针
    cuDoubleComplex* psi = &state_pool->data[state_idx * state_pool->d_trunc];

    // 最简单的测试：什么都不做，只是检查内存访问
    if (n < state_pool->d_trunc) {
        // 什么都不做，只是检查能否访问内存
        cuDoubleComplex temp = psi[n];
        // 不做任何修改
    }
}

/**
 * 通用对角门应用内核
 * @param state_pool CV状态池
 * @param target_indices 需要更新的状态ID列表
 * @param batch_size 批处理大小
 * @param phase_func 相位函数指针 (设备端函数)
 * @param params 门参数
 */
__global__ void apply_diagonal_gate_kernel(
    CVStatePool* state_pool,
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

    if (n >= state_pool->d_trunc) return;

    // 获取状态向量指针
    cuDoubleComplex* psi = &state_pool->data[state_idx * state_pool->d_trunc];

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

    // 使用简化的内核，避免函数指针
    apply_phase_rotation_simple_kernel<<<grid_dim, block_dim>>>(
        state_pool, target_indices, batch_size, theta
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
        state_pool, target_indices, batch_size, chi
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
        state_pool, target_indices, batch_size,
        conditional_parity_func, parity
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Conditional Parity kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
