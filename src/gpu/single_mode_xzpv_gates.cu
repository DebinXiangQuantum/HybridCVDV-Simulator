#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include "cv_state_pool.h"
#include "fock_ell_operator.h"

// ==========================================
// Strawberry Fields 单模门扩展
// ==========================================

/**
 * Xgate (位置位移) 内核
 * X(x) = exp(i x p̂/ℏ) = D(x/√(2ℏ))
 * 
 * 在 Fock 基中，这等价于位移算符
 */
__global__ void apply_xgate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double x,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int64_t current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    // X(x) = D(α) where α = x/√(2ℏ), 设 ℏ = 1
    // 使用位移算符的矩阵元素
    double alpha_real = x / sqrt(2.0);
    double alpha_imag = 0.0;
    double alpha_norm_sq = alpha_real * alpha_real;
    double prefactor = exp(-alpha_norm_sq / 2.0);

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    for (int m = 0; m < current_dim; ++m) {
        // 计算 D_nm = <n|D(α)|m>
        double sqrt_fact_ratio = 1.0;
        if (n > m) {
            for (int k = m + 1; k <= n; ++k) sqrt_fact_ratio *= sqrt((double)k);
            sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
        } else if (m > n) {
            for (int k = n + 1; k <= m; ++k) sqrt_fact_ratio *= sqrt((double)k);
            sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
        }

        int lower = (n < m) ? n : m;
        int upper = (n > m) ? n : m;
        int k = upper - lower;

        double laguerre = 0.0;
        double term = 1.0;
        double binom = 1.0;
        for (int i = 1; i <= lower; ++i) binom = binom * (upper - i + 1) / i;
        term = binom;
        laguerre += term;

        for (int j = 1; j <= lower; ++j) {
            term = term * (-alpha_norm_sq) * (lower - j + 1) / ((k + j) * j);
            laguerre += term;
        }

        cuDoubleComplex power_val = make_cuDoubleComplex(1.0, 0.0);
        if (n >= m) {
            for (int p = 0; p < k; ++p) {
                double new_real = power_val.x * alpha_real - power_val.y * alpha_imag;
                double new_imag = power_val.x * alpha_imag + power_val.y * alpha_real;
                power_val.x = new_real;
                power_val.y = new_imag;
            }
        } else {
            cuDoubleComplex minus_alpha_conj = make_cuDoubleComplex(-alpha_real, alpha_imag);
            for (int p = 0; p < k; ++p) {
                power_val = cuCmul(power_val, minus_alpha_conj);
            }
        }

        double real_scale = prefactor * sqrt_fact_ratio * laguerre;
        cuDoubleComplex d_nm = make_cuDoubleComplex(
            real_scale * cuCreal(power_val),
            real_scale * cuCimag(power_val)
        );

        sum = cuCadd(sum, cuCmul(d_nm, psi_in[m]));
    }
    psi_out[n] = sum;
}

/**
 * Zgate (动量位移) 内核
 * Z(p) = exp(-i p x̂/ℏ) = D(ip/√(2ℏ))
 */
__global__ void apply_zgate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double p,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int64_t current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    // Z(p) = D(α) where α = ip/√(2ℏ), 设 ℏ = 1
    double alpha_real = 0.0;
    double alpha_imag = p / sqrt(2.0);
    double alpha_norm_sq = alpha_imag * alpha_imag;
    double prefactor = exp(-alpha_norm_sq / 2.0);

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    for (int m = 0; m < current_dim; ++m) {
        double sqrt_fact_ratio = 1.0;
        if (n > m) {
            for (int k = m + 1; k <= n; ++k) sqrt_fact_ratio *= sqrt((double)k);
            sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
        } else if (m > n) {
            for (int k = n + 1; k <= m; ++k) sqrt_fact_ratio *= sqrt((double)k);
            sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
        }

        int lower = (n < m) ? n : m;
        int upper = (n > m) ? n : m;
        int k = upper - lower;

        double laguerre = 0.0;
        double term = 1.0;
        double binom = 1.0;
        for (int i = 1; i <= lower; ++i) binom = binom * (upper - i + 1) / i;
        term = binom;
        laguerre += term;

        for (int j = 1; j <= lower; ++j) {
            term = term * (-alpha_norm_sq) * (lower - j + 1) / ((k + j) * j);
            laguerre += term;
        }

        cuDoubleComplex power_val = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex alpha = make_cuDoubleComplex(alpha_real, alpha_imag);
        if (n >= m) {
            for (int p = 0; p < k; ++p) power_val = cuCmul(power_val, alpha);
        } else {
            cuDoubleComplex minus_alpha_conj = make_cuDoubleComplex(-alpha_real, alpha_imag);
            for (int p = 0; p < k; ++p) power_val = cuCmul(power_val, minus_alpha_conj);
        }

        double real_scale = prefactor * sqrt_fact_ratio * laguerre;
        cuDoubleComplex d_nm = make_cuDoubleComplex(
            real_scale * cuCreal(power_val),
            real_scale * cuCimag(power_val)
        );

        sum = cuCadd(sum, cuCmul(d_nm, psi_in[m]));
    }
    psi_out[n] = sum;
}

/**
 * Pgate (二次相位) 内核
 * P(s) = exp(i s x̂²/(2ℏ))
 * 
 * 在 Fock 基中是对角的
 */
__global__ void apply_pgate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double s
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int64_t current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // P(s)|n⟩ = exp(i s (n + 1/2)/2) |n⟩
    // 设 ℏ = 1
    double phase = s * (n + 0.5) / 2.0;
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));

    psi[n] = cuCmul(psi[n], phase_factor);
}

/**
 * Vgate (三次相位) 内核
 * V(γ) = exp(i γ x̂³/(3ℏ))
 * 
 * 需要在位置基中计算，然后变换回 Fock 基
 * 这里使用近似方法
 */
__global__ void apply_vgate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double gamma,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int64_t current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    // V gate 的矩阵元素计算（简化版本）
    // 完整实现需要在位置基中计算
    // 这里使用一阶近似
    cuDoubleComplex sum = psi_in[n];

    // 添加耦合项（简化）
    if (n > 0 && n < current_dim - 1) {
        double coeff = gamma * sqrt((double)n) * sqrt((double)(n + 1));
        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(0.0, coeff), psi_in[n]));
    }

    psi_out[n] = sum;
}

/**
 * Fouriergate 内核
 * F = exp(i π/2 (x̂² + p̂²)/(2ℏ))
 * 
 * 在 Fock 基中：F|n⟩ = exp(i π/2 (n + 1/2)) |n⟩
 */
__global__ void apply_fouriergate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int64_t current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // F|n⟩ = exp(i π/2 (n + 1/2)) |n⟩ = i^(n+1/2) |n⟩
    // = i^n · √i |n⟩
    double phase = M_PI / 2.0 * (n + 0.5);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));

    psi[n] = cuCmul(psi[n], phase_factor);
}

// ==================== 主机端接口 ====================

/**
 * 应用 Xgate (位置位移)
 */
void apply_xgate(CVStatePool* state_pool, const int* target_indices,
                int batch_size, double x) {
    size_t buffer_stride = state_pool->max_total_dim;
    size_t buffer_size = batch_size * buffer_stride * sizeof(cuDoubleComplex);
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(buffer_size));

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_xgate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, x,
        temp_buffer, buffer_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Xgate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Xgate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 复制结果回原位置
    for (int b = 0; b < batch_size; ++b) {
        int state_idx;
        cudaMemcpy(&state_idx, &target_indices[b], sizeof(int), cudaMemcpyDeviceToHost);

        // 从GPU复制state_dims和state_offsets到CPU
        int64_t state_dim;
        size_t offset;
        cudaMemcpy(&state_dim, &state_pool->state_dims[state_idx], sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&offset, &state_pool->state_offsets[state_idx], sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaMemcpy(&state_pool->data[offset], &temp_buffer[b * buffer_stride],
                   state_dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }
}

/**
 * 应用 Zgate (动量位移)
 */
void apply_zgate(CVStatePool* state_pool, const int* target_indices,
                int batch_size, double p) {
    size_t buffer_stride = state_pool->max_total_dim;
    size_t buffer_size = batch_size * buffer_stride * sizeof(cuDoubleComplex);
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(buffer_size));

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_zgate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, p,
        temp_buffer, buffer_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Zgate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Zgate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 复制结果回原位置
    for (int b = 0; b < batch_size; ++b) {
        int state_idx;
        cudaMemcpy(&state_idx, &target_indices[b], sizeof(int), cudaMemcpyDeviceToHost);

        // 从GPU复制state_dims和state_offsets到CPU
        int64_t state_dim;
        size_t offset;
        cudaMemcpy(&state_dim, &state_pool->state_dims[state_idx], sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&offset, &state_pool->state_offsets[state_idx], sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaMemcpy(&state_pool->data[offset], &temp_buffer[b * buffer_stride],
                   state_dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }
}

/**
 * 应用 Pgate (二次相位)
 */
void apply_pgate(CVStatePool* state_pool, const int* target_indices,
                int batch_size, double s) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_pgate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, s
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Pgate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Pgate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 应用 Vgate (三次相位)
 */
void apply_vgate(CVStatePool* state_pool, const int* target_indices,
                int batch_size, double gamma) {
    size_t buffer_stride = state_pool->max_total_dim;
    size_t buffer_size = batch_size * buffer_stride * sizeof(cuDoubleComplex);
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(buffer_size));

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_vgate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, gamma,
        temp_buffer, buffer_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Vgate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Vgate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 复制结果回原位置
    for (int b = 0; b < batch_size; ++b) {
        int state_idx;
        cudaMemcpy(&state_idx, &target_indices[b], sizeof(int), cudaMemcpyDeviceToHost);

        // 从GPU复制state_dims和state_offsets到CPU
        int64_t state_dim;
        size_t offset;
        cudaMemcpy(&state_dim, &state_pool->state_dims[state_idx], sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&offset, &state_pool->state_offsets[state_idx], sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaMemcpy(&state_pool->data[offset], &temp_buffer[b * buffer_stride],
                   state_dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }
}

/**
 * 应用 Fouriergate
 */
void apply_fouriergate(CVStatePool* state_pool, const int* target_indices,
                      int batch_size) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_fouriergate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Fouriergate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Fouriergate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

// ==========================================
// HPC Optimization: Kernel/Operator Fusion
// ==========================================

/**
 * 深度融合内核 (Phase -> Displacement -> Squeezing)
 * 将三个连续的单模门操作融合为一个 Kernel。
 * 数据仅从 Global Memory 加载一次，中间结果保存在 Shared Memory / Registers 中，
 * 最后一次性写回 Global Memory。这降低了 66% 的全局显存带宽需求。
 */
__global__ void apply_fused_rds_kernel(
    cuDoubleComplex* state_data, 
    const size_t* state_offsets, 
    const int64_t* state_dims,
    const int* target_indices, 
    int batch_size,
    double theta,                         // Phase 参数
    const cuDoubleComplex* disp_ell_val, const int* disp_ell_col, int disp_bw, // Disp ELL 格式
    const cuDoubleComplex* sqz_ell_val,  const int* sqz_ell_col,  int sqz_bw   // Squeezing ELL 格式
) {
    extern __shared__ cuDoubleComplex smem_buffer[]; 
    // 分配 2 * current_dim 的 Shared Memory 用于 Double Buffering

    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = threadIdx.x; 
    
    int64_t current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    cuDoubleComplex* psi_global = &state_data[state_offsets[state_idx]];
    cuDoubleComplex* smem_in = smem_buffer;
    cuDoubleComplex* smem_out = &smem_buffer[current_dim];

    // 1. Load & Phase Rotation (R) 
    double phase = -theta * static_cast<double>(n);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));
    smem_in[n] = cuCmul(psi_global[n], phase_factor);
    __syncthreads();

    // 2. Displacement (D)
    cuDoubleComplex sum_d = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 0; k < disp_bw; ++k) {
        int col = disp_ell_col[n * disp_bw + k];
        if (col != -1 && col < current_dim) {
            sum_d = cuCadd(sum_d, cuCmul(disp_ell_val[n * disp_bw + k], smem_in[col]));
        }
    }
    smem_out[n] = sum_d;
    __syncthreads();

    // 3. Squeezing (S)
    cuDoubleComplex sum_s = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 0; k < sqz_bw; ++k) {
        int col = sqz_ell_col[n * sqz_bw + k];
        if (col != -1 && col < current_dim) {
            sum_s = cuCadd(sum_s, cuCmul(sqz_ell_val[n * sqz_bw + k], smem_out[col]));
        }
    }

    // 4. Store Back
    psi_global[n] = sum_s;
}

/**
 * 主机端接口：执行深度融合的 R -> D -> S 门
 */
void apply_fused_rds_gate(
    CVStatePool* state_pool, 
    const int* target_indices, 
    int batch_size,
    double phase_theta,
    FockELLOperator* disp_ell,
    FockELLOperator* sqz_ell
) {
    if (!state_pool || !disp_ell || !sqz_ell || !target_indices || batch_size <= 0) return;

    // 假设每个 block 处理一个态，利用 Shared Memory 做 Buffer
    unsigned int block_x = static_cast<unsigned int>(std::min<int64_t>(state_pool->max_total_dim, 1024));
    if (block_x == 0) block_x = 1;
    dim3 block_dim(block_x);
    
    dim3 grid_dim(1, batch_size);

    size_t shared_mem_size = 2 * static_cast<size_t>(state_pool->max_total_dim) * sizeof(cuDoubleComplex);

    apply_fused_rds_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        state_pool->data, state_pool->state_offsets, state_pool->state_dims,
        target_indices, batch_size,
        phase_theta,
        disp_ell->ell_val, disp_ell->ell_col, disp_ell->max_bandwidth,
        sqz_ell->ell_val, sqz_ell->ell_col, sqz_ell->max_bandwidth
    );

    cudaDeviceSynchronize();
}
