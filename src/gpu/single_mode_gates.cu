#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cv_state_pool.h"
#include "fock_ell_operator.h"

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
    int n = blockIdx.x * blockDim.x + threadIdx.x;

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
    const int* state_dims,
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= ell_dim) return;
    
    // 获取状态的偏移量和维度
    size_t offset = state_offsets[state_idx];
    int state_dim = state_dims[state_idx];
    
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
        if (col_idx < 0 || col_idx >= state_dim || col_idx >= ell_dim) continue;

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
    const int* state_dims,
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
    int state_dim = state_dims[state_idx];
    
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

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ell_dim) return;

    // 计算该行的贡献
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 0; k < ell_bandwidth; ++k) {
        int col_idx = shared_ell_col[row * ell_bandwidth + k];

        if (col_idx == -1) break;  // 该行结束
        
        // 验证列索引在有效范围内
        if (col_idx < 0 || col_idx >= state_dim || col_idx >= ell_dim) continue;

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
    int n = blockIdx.x * blockDim.x + threadIdx.x;

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
        int min_nm = min(n, m);
        int max_nm = max(n, m);
        int diff = max_nm - min_nm; // |n-m|

        // 计算因子 √(min!/max!)
        double sqrt_fact_ratio = 1.0;
        if (max_nm > min_nm) {
            for (int k = min_nm + 1; k <= max_nm; ++k) {
                if (k > 0) {
                    sqrt_fact_ratio /= sqrt((double)k);
                }
            }
        } else if (max_nm < min_nm) {
            // 这种情况不应该发生，但为了安全起见
            for (int k = max_nm + 1; k <= min_nm; ++k) {
                if (k > 0) {
                    sqrt_fact_ratio *= sqrt((double)k);
                }
            }
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
            int binom_iters = min_nm - j;
            if (binom_iters > 0 && binom_iters <= max_nm) {
                for (int i = 0; i < binom_iters; ++i) {
                    binom = binom * (max_nm - i) / (i + 1);
                }
            } else if (binom_iters < 0) {
                binom = 0.0;  // 无效的二项式系数
            }

            // 避免除零错误
            if (fact_j == 0.0) fact_j = 1.0;
            
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

/**
 * 主机端接口：应用通用单模门 (ELL格式)
 */
void apply_single_mode_gate(CVStatePool* state_pool, FockELLOperator* ell_op,
                           const int* target_indices, int batch_size) {
    // 验证输入参数
    if (!state_pool || !ell_op || !target_indices || batch_size <= 0) {
        throw std::runtime_error("apply_single_mode_gate: 无效的输入参数");
    }
    
    // 检查ELL算符是否有效（允许空算符，但不允许空指针）
    if (!ell_op->ell_val || !ell_op->ell_col || ell_op->dim <= 0 || ell_op->max_bandwidth <= 0) {
        throw std::runtime_error("apply_single_mode_gate: ELL算符无效或未初始化");
    }
    
    // 检查状态池数据指针
    if (!state_pool->data || !state_pool->state_offsets || !state_pool->state_dims) {
        throw std::runtime_error("apply_single_mode_gate: 状态池未正确初始化");
    }
    
    // 验证每个目标状态的维度是否匹配ELL算符维度
    for (int i = 0; i < batch_size; ++i) {
        int state_id = target_indices[i];
        if (state_id < 0 || state_id >= state_pool->capacity) {
            throw std::runtime_error("apply_single_mode_gate: 无效的状态ID: " + std::to_string(state_id));
        }
    }
    
    dim3 block_dim(256);
    dim3 grid_dim((ell_op->dim + block_dim.x - 1) / block_dim.x, batch_size);

    // 选择合适的内核版本
    size_t shared_mem_size = ell_op->dim * ell_op->max_bandwidth *
                           (sizeof(cuDoubleComplex) + sizeof(int));

    // 清除之前的CUDA错误
    cudaGetLastError();
    
    if (shared_mem_size < 48 * 1024) {  // 48KB shared memory limit
        apply_ell_spmv_shared_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            state_pool->data, state_pool->state_offsets, state_pool->state_dims,
            ell_op->ell_val, ell_op->ell_col, ell_op->dim, ell_op->max_bandwidth,
            target_indices, batch_size
        );
    } else {
        apply_ell_spmv_kernel<<<grid_dim, block_dim>>>(
            state_pool->data, state_pool->state_offsets, state_pool->state_dims,
            ell_op->ell_val, ell_op->ell_col, ell_op->dim, ell_op->max_bandwidth,
            target_indices, batch_size
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Single-mode gate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用Displacement门 D(α)
 */
void apply_displacement_gate(CVStatePool* state_pool, const int* target_indices,
                           int batch_size, cuDoubleComplex alpha) {
    // 验证输入参数
    if (!state_pool || !target_indices || batch_size <= 0) {
        throw std::runtime_error("apply_displacement_gate: 无效的输入参数");
    }
    
    // 将目标索引从设备复制到主机
    std::vector<int> host_indices(batch_size);
    cudaError_t err = cudaMemcpy(host_indices.data(), target_indices,
                                 batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法复制目标索引到主机: " + std::string(cudaGetErrorString(err)));
    }

    // 清除之前的CUDA错误
    cudaGetLastError();

    // 对于动态张量积管理，为每个状态单独处理
    for (int i = 0; i < batch_size; ++i) {
        int state_id = host_indices[i];
        
        // 验证状态ID
        if (!state_pool->is_valid_state(state_id)) {
            std::cerr << "警告：跳过无效的状态ID: " << state_id << std::endl;
            continue;
        }
        
        int state_dim = state_pool->get_state_dim(state_id);
        if (state_dim <= 0 || state_dim > state_pool->get_max_total_dim()) {
            std::cerr << "警告：状态维度无效: " << state_dim << std::endl;
            continue;
        }

        cuDoubleComplex* state_ptr = state_pool->get_state_ptr(state_id);
        if (!state_ptr) {
            std::cerr << "警告：无法获取状态指针，状态ID: " << state_id << std::endl;
            continue;
        }

        dim3 block_dim(256);
        dim3 grid_dim((state_dim + block_dim.x - 1) / block_dim.x, 1);
        
        // 确保grid维度有效
        if (grid_dim.x == 0) {
            std::cerr << "警告：grid维度为0，跳过状态ID: " << state_id << std::endl;
            continue;
        }

        // 创建临时缓冲区用于输出
        cuDoubleComplex* temp_buffer = nullptr;
        err = cudaMalloc(&temp_buffer, state_dim * sizeof(cuDoubleComplex));
        if (err != cudaSuccess) {
            std::cerr << "警告：无法分配临时缓冲区，状态ID: " << state_id 
                      << ", 错误: " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        // 清除之前的CUDA错误
        cudaGetLastError();

        // 调用内核，传递输入和输出指针
        apply_displacement_direct_kernel<<<grid_dim, block_dim>>>(
            state_ptr, temp_buffer, state_dim, alpha
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(temp_buffer);
            throw std::runtime_error("Displacement gate kernel launch failed: " +
                                    std::string(cudaGetErrorString(err)));
        }

        // 同步以确保内核完成
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(temp_buffer);
            throw std::runtime_error("GPU同步失败: " + std::string(cudaGetErrorString(err)));
        }

        // 复制结果回原位置
        err = cudaMemcpy(state_ptr, temp_buffer, state_dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cudaFree(temp_buffer);
            throw std::runtime_error("无法复制结果: " + std::string(cudaGetErrorString(err)));
        }
        
        cudaFree(temp_buffer);
        if (cudaGetLastError() != cudaSuccess) {
            // 忽略释放错误，继续执行
        }
    }
    
    // 最终同步
    cudaDeviceSynchronize();
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
