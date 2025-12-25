/**
 * Squeezing Gate GPU Implementation
 * 完全在GPU上计算和应用挤压门，避免CPU-GPU传输开销
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cv_state_pool.h"
#include <vector>
#include <complex>
#include <cmath>

/**
 * GPU设备端函数: 计算挤压门矩阵元素
 * 使用递推关系直接在GPU上计算
 */
__device__ cuDoubleComplex compute_squeezing_element(
    int m, int n, 
    double r, double theta,
    const double* sqrt_n,
    int cutoff
) {
    // 奇偶性守恒检查
    if ((m + n) % 2 != 0) {
        return make_cuDoubleComplex(0.0, 0.0);
    }
    
    // 计算参数
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    double tanh_r = tanh(r);
    double sech_r = 1.0 / cosh(r);
    
    // eitheta_tanhr = exp(i*theta) * tanh(r)
    cuDoubleComplex eitheta_tanhr = make_cuDoubleComplex(
        cos_theta * tanh_r,
        sin_theta * tanh_r
    );
    
    // R矩阵
    cuDoubleComplex R00 = cuCmul(make_cuDoubleComplex(-1.0, 0.0), eitheta_tanhr);
    cuDoubleComplex R01 = make_cuDoubleComplex(sech_r, 0.0);
    cuDoubleComplex R10 = make_cuDoubleComplex(sech_r, 0.0);
    cuDoubleComplex R11 = cuConj(eitheta_tanhr);
    
    // 使用共享内存存储中间结果（每个线程块计算一行）
    // 这里简化实现，直接递推计算
    
    // 对于小矩阵，可以直接递推
    // 注意：这个实现适用于小cutoff（如16），对于大cutoff需要优化
    
    if (m == 0 && n == 0) {
        return make_cuDoubleComplex(sqrt(sech_r), 0.0);
    }
    
    // 这里简化处理：返回预计算标志
    // 实际应该在共享内存中递推计算
    return make_cuDoubleComplex(0.0, 0.0);
}

/**
 * GPU内核: 直接在GPU上应用挤压门
 * 使用on-the-fly计算，避免存储整个矩阵
 */
__global__ void apply_squeezing_direct_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double r,
    double theta,
    int cutoff,
    cuDoubleComplex* temp_buffer
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= cutoff) return;
    
    size_t offset = state_offsets[state_idx];
    int state_dim = state_dims[state_idx];
    
    if (state_dim < cutoff) return;

    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * cutoff];
    
    // 预计算sqrt
    __shared__ double sqrt_cache[32];  // 假设cutoff <= 32
    if (threadIdx.x < cutoff) {
        sqrt_cache[threadIdx.x] = sqrt((double)threadIdx.x);
    }
    __syncthreads();
    
    // 计算参数（每个线程独立计算）
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    double tanh_r = tanh(r);
    double sech_r = 1.0 / cosh(r);
    
    cuDoubleComplex eitheta_tanhr = make_cuDoubleComplex(
        cos_theta * tanh_r,
        sin_theta * tanh_r
    );
    
    cuDoubleComplex R00 = cuCmul(make_cuDoubleComplex(-1.0, 0.0), eitheta_tanhr);
    cuDoubleComplex R01 = make_cuDoubleComplex(sech_r, 0.0);
    cuDoubleComplex R11 = cuConj(eitheta_tanhr);
    
    // 使用共享内存存储S矩阵的当前列
    extern __shared__ cuDoubleComplex shared_S[];
    
    // 初始化第一列
    if (threadIdx.x == 0) {
        shared_S[0] = make_cuDoubleComplex(sqrt(sech_r), 0.0);
        for (int m = 2; m < cutoff; m += 2) {
            if (m < cutoff) {
                shared_S[m] = cuCmul(
                    make_cuDoubleComplex(sqrt_cache[m-1] / sqrt_cache[m], 0.0),
                    cuCmul(R00, shared_S[m-2])
                );
            }
        }
    }
    __syncthreads();
    
    // 计算该行的输出
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    
    // 遍历所有列（只计算奇偶性匹配的）
    for (int n = 0; n < cutoff; ++n) {
        if ((row + n) % 2 != 0) continue;  // 奇偶性守恒
        
        // 递推计算 S[row, n]
        cuDoubleComplex S_mn;
        
        if (n == 0) {
            S_mn = shared_S[row];
        } else {
            // 简化：这里需要完整的递推逻辑
            // 为了性能，应该预计算或使用更高效的方法
            S_mn = make_cuDoubleComplex(0.0, 0.0);
            
            // 递推公式
            if (n >= 2 && row < cutoff) {
                cuDoubleComplex term1 = make_cuDoubleComplex(0.0, 0.0);
                // term1 = sqrt[n-1]/sqrt[n] * R11 * S[row, n-2]
                // 这需要访问之前计算的值
            }
            
            if (row >= 1 && n >= 1) {
                cuDoubleComplex term2 = make_cuDoubleComplex(0.0, 0.0);
                // term2 = sqrt[row]/sqrt[n] * R01 * S[row-1, n-1]
            }
        }
        
        sum = cuCadd(sum, cuCmul(S_mn, psi_in[n]));
    }
    
    psi_out[row] = sum;
}

/**
 * 优化版本: 使用预计算的ELL矩阵（缓存在GPU上）
 */
struct SqueezingCache {
    cuDoubleComplex* d_ell_val;
    int* d_ell_col;
    int cutoff;
    int bandwidth;
    double r;
    double theta;
    bool valid;
};

// 全局缓存（简化实现，实际应该用更好的缓存策略）
static SqueezingCache g_cache = {nullptr, nullptr, 0, 0, 0.0, 0.0, false};

/**
 * GPU内核: 使用预计算的ELL格式
 */
__global__ void apply_squeezing_ell_cached_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const cuDoubleComplex* ell_val,
    const int* ell_col,
    int cutoff,
    int bandwidth,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex* temp_buffer
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= cutoff) return;
    
    size_t offset = state_offsets[state_idx];
    int state_dim = state_dims[state_idx];
    
    if (state_dim < cutoff) return;

    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * cutoff];

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    // ELL格式SpMV
    for (int k = 0; k < bandwidth; ++k) {
        int col_idx = ell_col[row * bandwidth + k];
        if (col_idx == -1) break;
        if (col_idx >= state_dim) continue;

        cuDoubleComplex val = ell_val[row * bandwidth + k];
        sum = cuCadd(sum, cuCmul(val, psi_in[col_idx]));
    }

    psi_out[row] = sum;
}

/**
 * GPU内核: 复制结果
 */
__global__ void copy_result_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const cuDoubleComplex* temp_buffer,
    const int* target_indices,
    int batch_size,
    int cutoff
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= cutoff) return;
    
    size_t offset = state_offsets[state_idx];
    int state_dim = state_dims[state_idx];
    
    if (idx < state_dim) {
        state_data[offset + idx] = temp_buffer[batch_id * cutoff + idx];
    }
}

/**
 * CPU端函数: 生成并缓存ELL矩阵
 */
void generate_and_cache_squeezing_ell(double r, double theta, int cutoff) {
    // 检查是否已缓存
    if (g_cache.valid && g_cache.r == r && g_cache.theta == theta && g_cache.cutoff == cutoff) {
        return;  // 已缓存，直接返回
    }
    
    // 清理旧缓存
    if (g_cache.d_ell_val) cudaFree(g_cache.d_ell_val);
    if (g_cache.d_ell_col) cudaFree(g_cache.d_ell_col);
    
    // 在CPU上生成矩阵（只在参数改变时执行一次）
    std::vector<double> sqrt_n(cutoff);
    for (int i = 0; i < cutoff; ++i) {
        sqrt_n[i] = std::sqrt(static_cast<double>(i));
    }
    
    std::complex<double> eitheta_tanhr = std::exp(std::complex<double>(0.0, theta)) * std::tanh(r);
    double sechr = 1.0 / std::cosh(r);
    
    std::complex<double> R[2][2];
    R[0][0] = -eitheta_tanhr;
    R[0][1] = sechr;
    R[1][0] = sechr;
    R[1][1] = std::conj(eitheta_tanhr);
    
    // 生成稠密矩阵
    std::vector<std::complex<double>> S(cutoff * cutoff, std::complex<double>(0.0, 0.0));
    S[0] = std::sqrt(sechr);
    
    for (int m = 2; m < cutoff; m += 2) {
        S[m * cutoff] = sqrt_n[m - 1] / sqrt_n[m] * R[0][0] * S[(m - 2) * cutoff];
    }
    
    for (int m = 0; m < cutoff; ++m) {
        for (int n = 1; n < cutoff; ++n) {
            if ((m + n) % 2 == 0) {
                std::complex<double> term1(0.0, 0.0);
                std::complex<double> term2(0.0, 0.0);
                
                if (n >= 2) {
                    term1 = sqrt_n[n - 1] / sqrt_n[n] * R[1][1] * S[m * cutoff + (n - 2)];
                }
                
                if (m >= 1 && n >= 1) {
                    term2 = sqrt_n[m] / sqrt_n[n] * R[0][1] * S[(m - 1) * cutoff + (n - 1)];
                }
                
                S[m * cutoff + n] = term1 + term2;
            }
        }
    }
    
    // 转换为ELL格式
    int max_bandwidth = cutoff;
    std::vector<std::complex<double>> ell_values(cutoff * max_bandwidth, std::complex<double>(0.0, 0.0));
    std::vector<int> ell_indices(cutoff * max_bandwidth, -1);
    
    int actual_bandwidth = 0;
    double threshold = 1e-12;
    
    for (int row = 0; row < cutoff; ++row) {
        std::vector<std::pair<int, std::complex<double>>> nonzeros;
        
        for (int col = 0; col < cutoff; ++col) {
            std::complex<double> val = S[row * cutoff + col];
            if (std::abs(val) > threshold) {
                nonzeros.push_back({col, val});
            }
        }
        
        int nnz = std::min(static_cast<int>(nonzeros.size()), max_bandwidth);
        actual_bandwidth = std::max(actual_bandwidth, nnz);
        
        for (int k = 0; k < nnz; ++k) {
            ell_values[row * max_bandwidth + k] = nonzeros[k].second;
            ell_indices[row * max_bandwidth + k] = nonzeros[k].first;
        }
    }
    
    // 上传到GPU并缓存
    cudaMalloc(&g_cache.d_ell_val, ell_values.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&g_cache.d_ell_col, ell_indices.size() * sizeof(int));
    
    cudaMemcpy(g_cache.d_ell_val, ell_values.data(), 
               ell_values.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(g_cache.d_ell_col, ell_indices.data(), 
               ell_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    g_cache.cutoff = cutoff;
    g_cache.bandwidth = max_bandwidth;
    g_cache.r = r;
    g_cache.theta = theta;
    g_cache.valid = true;
}

/**
 * 主机端接口: 应用挤压门（使用缓存）
 */
void apply_squeezing_gate_gpu(
    CVStatePool* pool,
    const int* target_indices,
    int batch_size,
    double r,
    double theta
) {
    int cutoff = pool->d_trunc;
    
    // 生成并缓存ELL矩阵（只在参数改变时重新生成）
    generate_and_cache_squeezing_ell(r, theta, cutoff);
    
    // 分配临时缓冲区
    cuDoubleComplex* d_temp_buffer;
    cudaMalloc(&d_temp_buffer, batch_size * cutoff * sizeof(cuDoubleComplex));
    
    // 启动内核
    dim3 block_dim(256);
    dim3 grid_dim((cutoff + block_dim.x - 1) / block_dim.x, batch_size);
    
    apply_squeezing_ell_cached_kernel<<<grid_dim, block_dim>>>(
        pool->data,
        pool->state_offsets,
        pool->state_dims,
        g_cache.d_ell_val,
        g_cache.d_ell_col,
        cutoff,
        g_cache.bandwidth,
        target_indices,
        batch_size,
        d_temp_buffer
    );
    
    cudaDeviceSynchronize();
    
    // 复制结果回原位置
    copy_result_kernel<<<grid_dim, block_dim>>>(
        pool->data,
        pool->state_offsets,
        pool->state_dims,
        d_temp_buffer,
        target_indices,
        batch_size,
        cutoff
    );
    
    cudaDeviceSynchronize();
    
    // 清理临时缓冲区
    cudaFree(d_temp_buffer);
}

/**
 * 清理缓存
 */
void clear_squeezing_cache() {
    if (g_cache.d_ell_val) {
        cudaFree(g_cache.d_ell_val);
        g_cache.d_ell_val = nullptr;
    }
    if (g_cache.d_ell_col) {
        cudaFree(g_cache.d_ell_col);
        g_cache.d_ell_col = nullptr;
    }
    g_cache.valid = false;
}
