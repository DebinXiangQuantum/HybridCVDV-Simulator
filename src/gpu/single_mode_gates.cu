#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include "cv_state_pool.h"
#include "fock_ell_operator.h"

/**
 * Level 2: 通用单模门 (General Single-Mode Gates) GPU内核
 *
 * 特性：矩阵为带状稀疏矩阵，使用Fock-ELL格式存储
 * 典型门：Displacement D(α), Squeezing S(ξ)
 *
 * 数学公式：ψ_out[n] = Σ_{k=0}^{K-1} ELL_Val[n][k] · ψ_in[ELL_Col[n][k]]
 */

/**
 * ELL格式稀疏矩阵向量乘法内核
 * 实现单模门的通用应用
 */
__global__ void apply_ell_spmv_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const cuDoubleComplex* ell_val,
    const int* ell_col,
    int dim,
    int max_bandwidth,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex* temp_buffer
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim) return;

    cuDoubleComplex* psi_in = &all_states_data[state_idx * d_trunc];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * d_trunc];

    // 初始化结果
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    // 遍历该行的非零元素
    for (int k = 0; k < max_bandwidth; ++k) {
        int col_idx = ell_col[row * max_bandwidth + k];

        if (col_idx == -1) break;  // 该行结束
        if (col_idx < 0 || col_idx >= d_trunc) continue; // Bounds check

        cuDoubleComplex val = ell_val[row * max_bandwidth + k];
        cuDoubleComplex psi_val = psi_in[col_idx];

        // 累加：sum += val * psi_in[col]
        sum = cuCadd(sum, cuCmul(val, psi_val));
    }

    psi_out[row] = sum;
}

/**
 * 优化版本：使用共享内存的ELL-SpMV
 * 对于小矩阵，可以将ELL算符加载到共享内存中
 */
__global__ void apply_ell_spmv_shared_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const cuDoubleComplex* ell_val,
    const int* ell_col,
    int dim,
    int max_bandwidth,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex* temp_buffer
) {
    extern __shared__ cuDoubleComplex shared_ell_val[];
    int* shared_ell_col = (int*)&shared_ell_val[dim * max_bandwidth];

    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    cuDoubleComplex* psi_in = &all_states_data[state_idx * d_trunc];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * d_trunc];

    // 将ELL算符加载到共享内存
    int total_elements = dim * max_bandwidth;
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        shared_ell_val[i] = ell_val[i];
        shared_ell_col[i] = ell_col[i];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;

    // 计算该行的贡献
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 0; k < max_bandwidth; ++k) {
        int col_idx = shared_ell_col[row * max_bandwidth + k];

        if (col_idx == -1) break;
        if (col_idx < 0 || col_idx >= d_trunc) continue;

        cuDoubleComplex val = shared_ell_val[row * max_bandwidth + k];
        cuDoubleComplex psi_val = psi_in[col_idx];

        sum = cuCadd(sum, cuCmul(val, psi_val));
    }

    psi_out[row] = sum;
}

/**
 * Displacement门 D(α) 的专用内核
 * 对于小的α值，矩阵带宽很小，可以特别优化
 *
 * D(α) = exp(α*a† - α*a)
 * 矩阵元素：<n|D(α)|m> = √(n!/m!) * (α)^(n-m) * exp(-|α|²/2) * L_m^(n-m)(|α|²)
 * 其中 L 是Laguerre多项式
 */
__global__ void apply_displacement_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex alpha,
    cuDoubleComplex* temp_buffer
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    cuDoubleComplex* psi_in = &all_states_data[state_idx * d_trunc];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * d_trunc];

    // 计算Displacement矩阵的第n行
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    double alpha_norm_sq = cuCreal(cuCmul(alpha, cuConj(alpha)));
    double exp_factor = exp(-alpha_norm_sq / 2.0);

    // 简化的位移门实现 - 只处理主要的矩阵元素
    // 对于小的位移参数，主要的贡献来自对角线和相邻元素
    int start_m = max(0, n - 2);
    int end_m = min(d_trunc - 1, n + 2);

    for (int m = start_m; m <= end_m; ++m) {
        double coeff = 0.0;

        if (n == m) {
            // 对角线元素: <n|D|n> = e^(-|α|²/2)
            coeff = exp_factor;
        } else if (n == m + 1) {
            // <n|D|n-1> = e^(-|α|²/2) * √n * Re(α)
            coeff = exp_factor * sqrt((double)n) * cuCreal(alpha);
        } else if (n == m - 1) {
            // <n|D|n+1> = e^(-|α|²/2) * √(n+1) * (-Re(α))
            coeff = -exp_factor * sqrt((double)n + 1.0) * cuCreal(alpha);
        }

        if (coeff != 0.0) {
            cuDoubleComplex matrix_elem = make_cuDoubleComplex(coeff, 0.0);
            sum = cuCadd(sum, cuCmul(matrix_elem, psi_in[m]));
        }
    }

    psi_out[n] = sum;
}

__global__ void copy_back_kernel(
    cuDoubleComplex* all_states_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    const cuDoubleComplex* temp_buffer
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;
    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= d_trunc) return;
    
    all_states_data[state_idx * d_trunc + n] = temp_buffer[batch_id * d_trunc + n];
}

/**
 * 主机端接口：应用通用单模门 (ELL格式)
 */
void apply_single_mode_gate(CVStatePool* state_pool, FockELLOperator* ell_op,
                           const int* target_indices, int batch_size) {
    dim3 block_dim(256);
    dim3 grid_dim((ell_op->dim + block_dim.x - 1) / block_dim.x, batch_size);

    // Allocate temporary buffer
    cuDoubleComplex* temp_buffer = nullptr;
    size_t buffer_size = batch_size * state_pool->d_trunc * sizeof(cuDoubleComplex);
    cudaError_t err = cudaMalloc(&temp_buffer, buffer_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate temp buffer: " + std::string(cudaGetErrorString(err)));
    }

    // 选择合适的内核版本
    size_t shared_mem_size = ell_op->dim * ell_op->max_bandwidth *
                           (sizeof(cuDoubleComplex) + sizeof(int));

    if (shared_mem_size < 48 * 1024) {  // 48KB shared memory limit
        apply_ell_spmv_shared_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            state_pool->data, state_pool->d_trunc, 
            ell_op->ell_val, ell_op->ell_col, ell_op->dim, ell_op->max_bandwidth,
            target_indices, batch_size, temp_buffer
        );
    } else {
        apply_ell_spmv_kernel<<<grid_dim, block_dim>>>(
            state_pool->data, state_pool->d_trunc, 
            ell_op->ell_val, ell_op->ell_col, ell_op->dim, ell_op->max_bandwidth,
            target_indices, batch_size, temp_buffer
        );
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(temp_buffer);
        throw std::runtime_error("Single-mode gate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
    
    // Copy back
    copy_back_kernel<<<grid_dim, block_dim>>>(state_pool->data, state_pool->d_trunc, target_indices, batch_size, temp_buffer);
    
    cudaFree(temp_buffer);
}

/**
 * 主机端接口：应用Displacement门 D(α)
 */
void apply_displacement_gate(CVStatePool* state_pool, const int* target_indices,
                           int batch_size, cuDoubleComplex alpha) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, batch_size);

    cuDoubleComplex* temp_buffer = nullptr;
    size_t buffer_size = batch_size * state_pool->d_trunc * sizeof(cuDoubleComplex);
    cudaError_t err = cudaMalloc(&temp_buffer, buffer_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate temp buffer: " + std::string(cudaGetErrorString(err)));
    }

    apply_displacement_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, state_pool->d_trunc, target_indices, batch_size, alpha, temp_buffer
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(temp_buffer);
        throw std::runtime_error("Displacement kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
    
    copy_back_kernel<<<grid_dim, block_dim>>>(state_pool->data, state_pool->d_trunc, target_indices, batch_size, temp_buffer);
    
    cudaFree(temp_buffer);
}

/**
 * 主机端接口：应用Squeezing门 S(ξ)
 * S(ξ) = exp(ξ*a²/2 - ξ*(a†)²/2)
 */
void apply_squeezing_gate(CVStatePool& state_pool, FockELLOperator& ell_op,
                         const int* target_indices, int batch_size) {
    // Squeezing门也使用ELL格式的通用实现
    apply_single_mode_gate(&state_pool, &ell_op, target_indices, batch_size);
}

