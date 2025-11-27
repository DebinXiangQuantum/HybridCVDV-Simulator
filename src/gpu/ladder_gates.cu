#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include "cv_state_pool.h"

/**
 * Level 1: 梯算符门 (Ladder/Shift Gates) GPU内核
 *
 * 特性：矩阵仅有一条非零对角线（次对角线）。无需存储矩阵，系数实时计算。
 * 典型门：Photon Creation a†, Annihilation a
 *
 * 数学公式：
 *   - Creation: ψ_out[n] = √n · ψ_in[n-1]
 *   - Annihilation: ψ_out[n] = √(n+1) · ψ_in[n+1]
 */

/**
 * 光子创建算符 a† 内核
 * ψ_out[n] = √n · ψ_in[n-1] (n >= 1)
 * ψ_out[0] = 0
 */
__global__ void apply_creation_operator_kernel(
    cuDoubleComplex* state_data,
    int d_trunc,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    cuDoubleComplex* psi_in = &state_data[state_idx * d_trunc];
    cuDoubleComplex* psi_out = psi_in;  // 原地操作

    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

    if (n > 0) {
        // ψ_out[n] = √n · ψ_in[n-1]
        double coeff = sqrt(static_cast<double>(n));
        cuDoubleComplex input_val = psi_in[n - 1];
        result = make_cuDoubleComplex(
            coeff * cuCreal(input_val),
            coeff * cuCimag(input_val)
        );
    }
    // n == 0 时，结果为0

    psi_out[n] = result;
}

/**
 * 光子湮灭算符 a 内核
 * ψ_out[n] = √(n+1) · ψ_in[n+1] (n < D-1)
 * ψ_out[D-1] = 0
 */
__global__ void apply_annihilation_operator_kernel(
    cuDoubleComplex* state_data,
    int d_trunc,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    cuDoubleComplex* psi_in = &state_data[state_idx * d_trunc];
    cuDoubleComplex* psi_out = psi_in;  // 原地操作

    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

    if (n < d_trunc - 1) {
        // ψ_out[n] = √(n+1) · ψ_in[n+1]
        double coeff = sqrt(static_cast<double>(n + 1));
        cuDoubleComplex input_val = psi_in[n + 1];
        result = make_cuDoubleComplex(
            coeff * cuCreal(input_val),
            coeff * cuCimag(input_val)
        );
    }
    // n == D-1 时，结果为0

    psi_out[n] = result;
}

/**
 * 通用梯算符门内核 (使用shuffle指令优化)
 * 支持创建和湮灭算符的warp级优化版本
 */
__global__ void apply_ladder_operator_warp_kernel(
    cuDoubleComplex* state_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    bool is_creation  // true: 创建算符, false: 湮灭算符
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int tid = threadIdx.x;
    int n = blockIdx.x * blockDim.x + tid;

    if (n >= d_trunc) return;

    cuDoubleComplex* psi = &state_data[state_idx * d_trunc];

    if (is_creation) {
        // 创建算符: ψ[n] = √n · ψ[n-1]
        cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

        if (n > 0) {
            // 从前一个线程获取数据
            double coeff = sqrt(static_cast<double>(n));

            // 使用shuffle获取前一个元素
            float real_part, imag_part;
            if (tid > 0) {
                // 同warp内的线程可以直接shuffle
                real_part = __shfl_up_sync(0xFFFFFFFF, cuCreal(psi[n]), 1);
                imag_part = __shfl_up_sync(0xFFFFFFFF, cuCimag(psi[n]), 1);
            } else {
                // 跨warp边界，需要从全局内存读取
                if (n > 0) {
                    cuDoubleComplex prev_val = psi[n - 1];
                    real_part = cuCreal(prev_val);
                    imag_part = cuCimag(prev_val);
                } else {
                    real_part = 0.0f;
                    imag_part = 0.0f;
                }
            }

            result = make_cuDoubleComplex(coeff * real_part, coeff * imag_part);
        }

        psi[n] = result;
    } else {
        // 湮灭算符: ψ[n] = √(n+1) · ψ[n+1]
        cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

        if (n < d_trunc - 1) {
            // 从后一个线程获取数据
            double coeff = sqrt(static_cast<double>(n + 1));

            // 使用shuffle获取后一个元素
            float real_part, imag_part;
            if (tid < blockDim.x - 1) {
                // 同warp内的线程可以直接shuffle
                real_part = __shfl_down_sync(0xFFFFFFFF, cuCreal(psi[n]), 1);
                imag_part = __shfl_down_sync(0xFFFFFFFF, cuCimag(psi[n]), 1);
            } else {
                // 跨warp边界，需要从全局内存读取
                if (n < d_trunc - 1) {
                    cuDoubleComplex next_val = psi[n + 1];
                    real_part = cuCreal(next_val);
                    imag_part = cuCimag(next_val);
                } else {
                    real_part = 0.0f;
                    imag_part = 0.0f;
                }
            }

            result = make_cuDoubleComplex(coeff * real_part, coeff * imag_part);
        }

        psi[n] = result;
    }
}

/**
 * 主机端接口：应用光子创建算符 a†
 */
void apply_creation_operator(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_creation_operator_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->total_dim, target_indices, batch_size
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
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_annihilation_operator_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->total_dim, target_indices, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Annihilation operator kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
