#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include "cv_state_pool.h"
#include "fock_ell_operator.h"

namespace {

int max_batch_launch_y() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 65535;
    }

    int max_grid_y = 65535;
    if (cudaDeviceGetAttribute(&max_grid_y, cudaDevAttrMaxGridDimY, device) != cudaSuccess || max_grid_y <= 0) {
        return 65535;
    }
    return max_grid_y;
}

}  // namespace

/**
 * 位移门内核 D(α) = exp(α a† - α* a)
 * 使用ELL格式SpMV实现
 */
__global__ void apply_displacement_kernel(
    cuDoubleComplex* state_data,
    int d_trunc,
    FockELLOperator* ell_op,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    // 获取输入和输出状态向量指针
    cuDoubleComplex* psi_in = &state_data[state_idx * d_trunc];
    cuDoubleComplex* psi_out = psi_in; // 原地更新

    // ELL格式SpMV
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    for (int k = 0; k < ell_op->max_bandwidth; ++k) {
        int col = ell_op->ell_col[n * ell_op->max_bandwidth + k];
        if (col == -1) break; // ELL填充

        cuDoubleComplex val = ell_op->ell_val[n * ell_op->max_bandwidth + k];
        cuDoubleComplex input_val = psi_in[col];

        sum = cuCadd(sum, cuCmul(val, input_val));
    }
    psi_out[n] = sum;
}

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
 * 使用动态状态偏移量支持不同维度的状态
 */
__global__ void apply_ell_spmv_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const cuDoubleComplex* ell_val,
    const int* ell_col,
    int ell_dim,
    int ell_bandwidth,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= ell_dim) return;
    
    // 获取状态的偏移量和维度
    size_t offset = state_offsets[state_idx];
    int64_t state_dim = state_dims[state_idx];
    
    // 验证维度匹配
    if (state_dim < ell_dim) {
        return;  // 状态维度不足，跳过
    }

    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = psi_in;  // 原地操作

    // 初始化结果
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    // 遍历该行的非零元素
    for (int k = 0; k < ell_bandwidth; ++k) {
        int col_idx = ell_col[row * ell_bandwidth + k];

        if (col_idx == -1) break;  // 该行结束
        
        // 验证列索引在有效范围内
        if (col_idx >= state_dim) continue;

        cuDoubleComplex val = ell_val[row * ell_bandwidth + k];
        cuDoubleComplex psi_val = psi_in[col_idx];

        // 累加：sum += val * psi_in[col]
        sum = cuCadd(sum, cuCmul(val, psi_val));
    }

    psi_out[row] = sum;
}

/**
 * 优化版本：使用共享内存的ELL-SpMV
 * 对于小矩阵，可以将ELL算符加载到共享内存中
 * 使用动态状态偏移量支持不同维度的状态
 */
__global__ void apply_ell_spmv_shared_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const cuDoubleComplex* ell_val,
    const int* ell_col,
    int ell_dim,
    int ell_bandwidth,
    const int* target_indices,
    int batch_size
) {
    extern __shared__ cuDoubleComplex shared_ell_val[];
    int* shared_ell_col = (int*)&shared_ell_val[ell_dim * ell_bandwidth];

    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    
    // 获取状态的偏移量和维度
    size_t offset = state_offsets[state_idx];
    int64_t state_dim = state_dims[state_idx];
    
    // 验证维度匹配
    if (state_dim < ell_dim) {
        return;  // 状态维度不足，跳过
    }
    
    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = psi_in;

    // 将ELL算符加载到共享内存
    int total_elements = ell_dim * ell_bandwidth;
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        shared_ell_val[i] = ell_val[i];
        shared_ell_col[i] = ell_col[i];
    }
    __syncthreads();

    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ell_dim) return;

    // 计算该行的贡献
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 0; k < ell_bandwidth; ++k) {
        int col_idx = shared_ell_col[row * ell_bandwidth + k];

        if (col_idx == -1) break;  // 该行结束
        
        // 验证列索引在有效范围内
        if (col_idx >= state_dim) continue;

        cuDoubleComplex val = shared_ell_val[row * ell_bandwidth + k];
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
__global__ void apply_displacement_direct_kernel(
    const cuDoubleComplex* in_data,
    cuDoubleComplex* out_data,
    int d_trunc,
    cuDoubleComplex alpha
) {
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= d_trunc) return;

    const cuDoubleComplex* psi_in = in_data;
    cuDoubleComplex* psi_out = out_data;

    // 计算Displacement矩阵的第n行
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    double alpha_real = cuCreal(alpha);
    double alpha_imag = cuCimag(alpha);
    double alpha_norm_sq = alpha_real*alpha_real + alpha_imag*alpha_imag;
    double exp_factor = exp(-alpha_norm_sq / 2.0);

    for (int m = 0; m < d_trunc; ++m) {
        cuDoubleComplex term_val;
        int min_nm = min(static_cast<int>(n), m);
        int max_nm = max(static_cast<int>(n), m);
        int diff = max_nm - min_nm; // |n-m|

        // 计算因子 √(min!/max!)
        double sqrt_fact_ratio = 1.0;
        for (int k = min_nm + 1; k <= max_nm; ++k) {
            sqrt_fact_ratio /= sqrt((double)k);
        }

        // 计算幂次项
        cuDoubleComplex power_term = make_cuDoubleComplex(1.0, 0.0);
        if (n >= m) {
            // n >= m: D_{nm} = coeff * α^(n-m) * L
            for(int k=0; k<diff; ++k) power_term = cuCmul(power_term, alpha);
        } else {
            // m > n: D_{nm} = coeff * (-conj(α))^(m-n) * L
            cuDoubleComplex minus_conj_alpha = make_cuDoubleComplex(-alpha_real, alpha_imag);
            for(int k=0; k<diff; ++k) power_term = cuCmul(power_term, minus_conj_alpha);
        }

        // 计算拉盖尔多项式 L_lower^{(diff)}(|α|^2) where lower = min(n, m)
        double laguerre = 0.0;
        double x = alpha_norm_sq;
        double x_pow_j = 1.0; // x^0
        double fact_j = 1.0;  // 0!

        for (int j = 0; j <= min_nm; ++j) {
            if (j > 0) {
                x_pow_j *= x;
                fact_j *= j;
            }

            // binom(max_nm, min_nm - j)
            double binom = 1.0;
            for (int i = 0; i < min_nm - j; ++i) {
                binom = binom * (max_nm - i) / (i + 1);
            }

            double term = binom * x_pow_j / fact_j;
            if (j % 2 == 1) term = -term;

            laguerre += term;
        }

        double real_part = exp_factor * sqrt_fact_ratio * laguerre;
        term_val = cuCmul(power_term, make_cuDoubleComplex(real_part, 0.0));

        sum = cuCadd(sum, cuCmul(term_val, psi_in[m]));
    }

    psi_out[n] = sum;
}

__global__ void apply_displacement_batched_kernel(
    const cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex alpha,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    const cuDoubleComplex* psi_in = &state_data[state_offsets[state_idx]];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    double alpha_real = cuCreal(alpha);
    double alpha_imag = cuCimag(alpha);
    double alpha_norm_sq = alpha_real * alpha_real + alpha_imag * alpha_imag;
    double exp_factor = exp(-alpha_norm_sq / 2.0);

    for (int m = 0; m < current_dim; ++m) {
        cuDoubleComplex term_val;
        int min_nm = min(static_cast<int>(n), m);
        int max_nm = max(static_cast<int>(n), m);
        int diff = max_nm - min_nm;

        double sqrt_fact_ratio = 1.0;
        for (int k = min_nm + 1; k <= max_nm; ++k) {
            sqrt_fact_ratio /= sqrt((double)k);
        }

        cuDoubleComplex power_term = make_cuDoubleComplex(1.0, 0.0);
        if (n >= m) {
            for (int k = 0; k < diff; ++k) power_term = cuCmul(power_term, alpha);
        } else {
            cuDoubleComplex minus_conj_alpha = make_cuDoubleComplex(-alpha_real, alpha_imag);
            for (int k = 0; k < diff; ++k) power_term = cuCmul(power_term, minus_conj_alpha);
        }

        double laguerre = 0.0;
        double x = alpha_norm_sq;
        double x_pow_j = 1.0;
        double fact_j = 1.0;

        for (int j = 0; j <= min_nm; ++j) {
            if (j > 0) {
                x_pow_j *= x;
                fact_j *= j;
            }

            double binom = 1.0;
            int choose = min_nm - j;
            for (int k = 0; k < choose; ++k) {
                binom *= (double)(max_nm - k) / (double)(k + 1);
            }

            double term = binom * ((j % 2 == 0) ? 1.0 : -1.0) * x_pow_j / fact_j;
            laguerre += term;
        }

        double scale = exp_factor * sqrt_fact_ratio * laguerre;
        term_val = make_cuDoubleComplex(scale * cuCreal(power_term), scale * cuCimag(power_term));
        sum = cuCadd(sum, cuCmul(term_val, psi_in[m]));
    }

    psi_out[n] = sum;
}

__global__ void copy_back_displacement_batched_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    const cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    cuDoubleComplex* psi = &state_data[state_offsets[state_idx]];
    psi[n] = temp_buffer[batch_id * buffer_stride + n];
}

/**
 * 主机端接口：应用通用单模门 (ELL格式)
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_single_mode_gate(CVStatePool* state_pool, FockELLOperator* ell_op,
                           const int* target_indices, int batch_size,
                           cudaStream_t stream, bool synchronize) {
    // 验证输入参数
    if (!state_pool || !ell_op || !target_indices || batch_size <= 0) {
        throw std::runtime_error("apply_single_mode_gate: 无效的输入参数");
    }

    // 检查ELL算符是否有效
    if (!ell_op->ell_val || !ell_op->ell_col || ell_op->dim <= 0 || ell_op->max_bandwidth <= 0) {
        throw std::runtime_error("apply_single_mode_gate: ELL算符无效或未初始化");
    }

    // 检查状态池数据指针
    if (!state_pool->data || !state_pool->state_offsets || !state_pool->state_dims) {
        throw std::runtime_error("apply_single_mode_gate: 状态池未正确初始化");
    }

    dim3 block_dim(256);
    const unsigned int grid_x = static_cast<unsigned int>((ell_op->dim + block_dim.x - 1) / block_dim.x);
    const int max_launch_batch = max_batch_launch_y();

    // 选择合适的内核版本
    size_t shared_mem_size = ell_op->dim * ell_op->max_bandwidth *
                           (sizeof(cuDoubleComplex) + sizeof(int));

    cudaError_t err = cudaSuccess;
    for (int batch_offset = 0; batch_offset < batch_size; batch_offset += max_launch_batch) {
        const int launch_batch = std::min(max_launch_batch, batch_size - batch_offset);
        const int* launch_targets = target_indices + batch_offset;
        dim3 grid_dim(grid_x, static_cast<unsigned int>(launch_batch));

        cudaGetLastError();

        if (shared_mem_size < 48 * 1024) {  // 48KB shared memory limit
            apply_ell_spmv_shared_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
                state_pool->data, state_pool->state_offsets, state_pool->state_dims,
                ell_op->ell_val, ell_op->ell_col, ell_op->dim, ell_op->max_bandwidth,
                launch_targets, launch_batch
            );
        } else {
            apply_ell_spmv_kernel<<<grid_dim, block_dim, 0, stream>>>(
                state_pool->data, state_pool->state_offsets, state_pool->state_dims,
                ell_op->ell_val, ell_op->ell_col, ell_op->dim, ell_op->max_bandwidth,
                launch_targets, launch_batch
            );
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Single-mode gate kernel launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    if (synchronize) {
        err = stream != nullptr ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("Single-mode gate kernel synchronization failed: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
}

/**
 * Multi-mode ELL SpMV kernel — applies a D×D ELL operator on a single target mode
 * within a multi-mode tensor product state (D^nm dimensional).
 * Each thread handles one flat_index in the full state vector.
 */
__global__ void apply_ell_spmv_multimode_kernel(
    const cuDoubleComplex* __restrict__ state_data,
    const size_t*          __restrict__ state_offsets,
    const int64_t*         __restrict__ state_dims,
    const cuDoubleComplex* __restrict__ ell_val,
    const int*             __restrict__ ell_col,
    int ell_dim,           // D (truncation dim of target mode)
    int ell_bandwidth,     // K_eff
    const int* __restrict__ target_indices,
    int batch_size,
    cuDoubleComplex* __restrict__ temp_buffer,
    size_t buffer_stride,
    int target_mode_right_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    size_t offset = state_offsets[state_idx];
    const cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t mode_block = static_cast<size_t>(ell_dim) * right_stride;
    if (mode_block == 0) return;

    // Decompose flat_index into (group_base, n, intra_offset)
    const size_t group_start =
        (flat_index / mode_block) * mode_block + (flat_index % right_stride);
    const int n = static_cast<int>((flat_index / right_stride) % ell_dim);

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 0; k < ell_bandwidth; ++k) {
        int col = ell_col[n * ell_bandwidth + k];
        if (col == -1) break;
        cuDoubleComplex val = ell_val[n * ell_bandwidth + k];
        size_t src_idx = group_start + static_cast<size_t>(col) * right_stride;
        sum = cuCadd(sum, cuCmul(val, psi_in[src_idx]));
    }
    psi_out[flat_index] = sum;
}

__global__ void copy_back_multimode_kernel(
    cuDoubleComplex* __restrict__ state_data,
    const size_t*    __restrict__ state_offsets,
    const int64_t*   __restrict__ state_dims,
    const int*       __restrict__ target_indices,
    int batch_size,
    const cuDoubleComplex* __restrict__ temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    size_t offset = state_offsets[state_idx];
    state_data[offset + flat_index] = temp_buffer[batch_id * buffer_stride + flat_index];
}

/**
 * Host interface: apply a D×D ELL operator on a specific mode within multi-mode states.
 * The ELL val/col arrays must already reside on the GPU.
 */
void apply_ell_gate_on_mode(
    CVStatePool* state_pool,
    const int* d_target_ids,
    int batch_size,
    const cuDoubleComplex* d_ell_val,
    const int* d_ell_col,
    int ell_dim,
    int ell_bandwidth,
    int target_qumode,
    int num_qumodes,
    cudaStream_t stream,
    bool synchronize
) {
    if (!state_pool || !d_target_ids || batch_size <= 0) return;

    int right_stride = 1;
    for (int m = target_qumode + 1; m < num_qumodes; ++m) {
        right_stride *= state_pool->d_trunc;
    }

    const size_t buffer_stride = state_pool->max_total_dim;
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(
            static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex)));

    dim3 block_dim(256);
    const unsigned int grid_x = static_cast<unsigned int>(
        (state_pool->max_total_dim + block_dim.x - 1) / block_dim.x);
    const int max_launch_batch = max_batch_launch_y();

    for (int batch_offset = 0; batch_offset < batch_size; batch_offset += max_launch_batch) {
        const int launch_batch = std::min(max_launch_batch, batch_size - batch_offset);
        const int* launch_targets = d_target_ids + batch_offset;
        cuDoubleComplex* launch_temp =
            temp_buffer + static_cast<size_t>(batch_offset) * buffer_stride;
        dim3 grid_dim(grid_x, static_cast<unsigned int>(launch_batch));

        apply_ell_spmv_multimode_kernel<<<grid_dim, block_dim, 0, stream>>>(
            state_pool->data, state_pool->state_offsets, state_pool->state_dims,
            d_ell_val, d_ell_col, ell_dim, ell_bandwidth,
            launch_targets, launch_batch,
            launch_temp, buffer_stride, right_stride);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("ELL multimode SpMV kernel launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        copy_back_multimode_kernel<<<grid_dim, block_dim, 0, stream>>>(
            state_pool->data, state_pool->state_offsets, state_pool->state_dims,
            launch_targets, launch_batch,
            launch_temp, buffer_stride);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("ELL multimode copy-back kernel failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    if (synchronize) {
        cudaError_t err = stream != nullptr ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("ELL multimode gate synchronization failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
}

/**
 * 主机端接口：应用Displacement门 D(α)
 * @param target_indices 设备端指针或主机端指针（根据调用者不同）
 * 注意：这个函数需要特殊处理，因为它直接访问target_indices
 */
void apply_displacement_gate(CVStatePool* state_pool, const int* target_indices,
                           int batch_size, cuDoubleComplex alpha,
                           cudaStream_t stream, bool synchronize) {
    if (batch_size <= 0) {
        return;
    }

    const size_t buffer_stride = static_cast<size_t>(state_pool->max_total_dim);
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(static_cast<size_t>(batch_size) * buffer_stride *
                                        sizeof(cuDoubleComplex)));

    dim3 block_dim(256);
    const unsigned int grid_x = static_cast<unsigned int>((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x);
    const int max_launch_batch = max_batch_launch_y();
    cudaError_t err = cudaSuccess;

    for (int batch_offset = 0; batch_offset < batch_size; batch_offset += max_launch_batch) {
        const int launch_batch = std::min(max_launch_batch, batch_size - batch_offset);
        const int* launch_targets = target_indices + batch_offset;
        cuDoubleComplex* launch_temp_buffer =
            temp_buffer + static_cast<size_t>(batch_offset) * buffer_stride;
        dim3 grid_dim(grid_x, static_cast<unsigned int>(launch_batch));

        cudaGetLastError();

        apply_displacement_batched_kernel<<<grid_dim, block_dim, 0, stream>>>(
            state_pool->data,
            state_pool->state_offsets,
            state_pool->state_dims,
            launch_targets,
            launch_batch,
            alpha,
            launch_temp_buffer,
            buffer_stride);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Displacement gate kernel launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        copy_back_displacement_batched_kernel<<<grid_dim, block_dim, 0, stream>>>(
            state_pool->data,
            state_pool->state_offsets,
            state_pool->state_dims,
            launch_targets,
            launch_batch,
            launch_temp_buffer,
            buffer_stride);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Displacement gate write-back failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    if (synchronize) {
        err = stream != nullptr ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("Displacement gate synchronization failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
}

/**
 * 主机端接口：应用Squeezing门 S(ξ)
 * S(ξ) = exp(ξ*a²/2 - ξ*(a†)²/2)
 */
void apply_squeezing_gate(CVStatePool& state_pool, FockELLOperator& ell_op,
                         const int* target_indices, int batch_size) {
    // Squeezing门也使用ELL格式的通用实现
    apply_single_mode_gate(&state_pool, &ell_op, target_indices, batch_size, nullptr, true);
}
