#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include "cv_state_pool.h"

/**
 * Level 3: 双模混合门 (Two-Mode Mixing Gates) GPU内核
 *
 * 使用Strawberry Fields的递推方法计算Beam Splitter
 * 
 * 优势：
 * - 数值稳定（无阶乘、无指数运算）
 * - 精确（保持幺正性）
 * - 高效（O(D^4)预计算 + O(D^4)应用）
 */

/**
 * CPU端：使用递推关系构建完整的BS矩阵
 * 
 * 基于Strawberry Fields的实现：
 * Z[m,n,p,q] = <m,n| BS(θ,φ) |p,q>
 * 
 * 递推关系：
 * Rank 3 (q=0): Z[m,n,p,0] = (ct*√m/√p)*Z[m-1,n,p-1,0] + (st*e^(iφ)*√n/√p)*Z[m,n-1,p-1,0]
 * Rank 4 (q>0): Z[m,n,p,q] = (-st*e^(-iφ)*√m/√q)*Z[m-1,n,p,q-1] + (ct*√n/√q)*Z[m,n-1,p,q-1]
 */
void build_bs_matrix_recursive(
    std::vector<cuDoubleComplex>& Z,
    int cutoff,
    double theta,
    double phi) {
    
    int D = cutoff;
    Z.resize(D * D * D * D, make_cuDoubleComplex(0.0, 0.0));
    
    double ct = std::cos(theta);
    double st = std::sin(theta);
    cuDoubleComplex phase = make_cuDoubleComplex(std::cos(phi), std::sin(phi));
    
    // 预计算sqrt表
    std::vector<double> sqrt_table(D);
    for (int i = 0; i < D; ++i) {
        sqrt_table[i] = std::sqrt((double)i);
    }
    
    // 基础情况
    Z[0] = make_cuDoubleComplex(1.0, 0.0);  // Z[0,0,0,0] = 1
    
    // Rank 3: 填充 Z[m,n,p,0]
    for (int m = 0; m < D; ++m) {
        for (int n = 0; n < D - m; ++n) {
            int p = m + n;
            if (p > 0 && p < D) {
                int idx = m*D*D*D + n*D*D + p*D + 0;
                cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
                
                // 第一项: ct * sqrt(m) / sqrt(p) * Z[m-1,n,p-1,0]
                if (m > 0) {
                    int idx1 = (m-1)*D*D*D + n*D*D + (p-1)*D + 0;
                    double coeff = ct * sqrt_table[m] / sqrt_table[p];
                    sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), Z[idx1]));
                }
                
                // 第二项: st * e^(iφ) * sqrt(n) / sqrt(p) * Z[m,n-1,p-1,0]
                if (n > 0) {
                    int idx2 = m*D*D*D + (n-1)*D*D + (p-1)*D + 0;
                    double coeff = st * sqrt_table[n] / sqrt_table[p];
                    cuDoubleComplex term = cuCmul(phase, make_cuDoubleComplex(coeff, 0.0));
                    sum = cuCadd(sum, cuCmul(term, Z[idx2]));
                }
                
                Z[idx] = sum;
            }
        }
    }
    
    // Rank 4: 填充 Z[m,n,p,q] for q > 0
    for (int m = 0; m < D; ++m) {
        for (int n = 0; n < D; ++n) {
            for (int p = 0; p < D; ++p) {
                int q = m + n - p;  // 光子数守恒
                if (q > 0 && q < D) {
                    int idx = m*D*D*D + n*D*D + p*D + q;
                    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
                    
                    // 第一项: -st * e^(-iφ) * sqrt(m) / sqrt(q) * Z[m-1,n,p,q-1]
                    if (m > 0) {
                        int idx1 = (m-1)*D*D*D + n*D*D + p*D + (q-1);
                        double coeff = -st * sqrt_table[m] / sqrt_table[q];
                        cuDoubleComplex conj_phase = cuConj(phase);
                        cuDoubleComplex term = cuCmul(conj_phase, make_cuDoubleComplex(coeff, 0.0));
                        sum = cuCadd(sum, cuCmul(term, Z[idx1]));
                    }
                    
                    // 第二项: ct * sqrt(n) / sqrt(q) * Z[m,n-1,p,q-1]
                    if (n > 0) {
                        int idx2 = m*D*D*D + (n-1)*D*D + p*D + (q-1);
                        double coeff = ct * sqrt_table[n] / sqrt_table[q];
                        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), Z[idx2]));
                    }
                    
                    Z[idx] = sum;
                }
            }
        }
    }
}

/**
 * GPU Kernel: 应用BS矩阵到状态向量
 * 使用标准张量积格式（不需要转换！）
 * 
 * output[m,n] = sum_{p,q} Z[m,n,p,q] * input[p,q]
 */
__global__ void apply_bs_tensor_kernel(
    const cuDoubleComplex* input_state,
    cuDoubleComplex* output_state,
    const cuDoubleComplex* bs_matrix,
    int cutoff) {
    
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (m >= cutoff || n >= cutoff) return;
    
    cuDoubleComplex amplitude = make_cuDoubleComplex(0.0, 0.0);
    
    // 对所有输入基态 |p,q⟩ 求和
    for (int p = 0; p < cutoff; ++p) {
        for (int q = 0; q < cutoff; ++q) {
            // Z[m,n,p,q]
            int z_idx = m*cutoff*cutoff*cutoff + n*cutoff*cutoff + p*cutoff + q;
            cuDoubleComplex z_elem = bs_matrix[z_idx];
            
            // input[p,q]
            int input_idx = p * cutoff + q;
            
            amplitude = cuCadd(amplitude, cuCmul(z_elem, input_state[input_idx]));
        }
    }
    
    int output_idx = m * cutoff + n;
    output_state[output_idx] = amplitude;
}

// 全局缓存：存储预计算的BS矩阵
static cuDoubleComplex* d_bs_matrix_cache = nullptr;
static int cached_cutoff = 0;
static double cached_theta = -999.0;
static double cached_phi = -999.0;

// 全局缓存：存储预计算的Two-Mode Squeezing矩阵
static cuDoubleComplex* d_tms_matrix_cache = nullptr;
static int cached_tms_cutoff = 0;
static double cached_tms_r = -999.0;
static double cached_tms_theta = -999.0;

/**
 * Helper: 计算 log(n!)
 */
__host__ __device__ double log_factorial(int n) {
    if (n <= 1) return 0.0;
    double result = 0.0;
    for (int i = 2; i <= n; ++i) {
        result += log((double)i);
    }
    return result;
}

/**
 * Beam Splitter矩阵计算函数 - 使用对数空间保证数值稳定性
 */
__host__ __device__ cuDoubleComplex compute_bs_matrix_element(int L, int m, int n, double theta, double phi) {
    if (m < 0 || m > L || n < 0 || n > L) {
        return make_cuDoubleComplex(0.0, 0.0);
    }
    
    double ct = cos(theta);
    double st = sin(theta);
    
    // 计算 log(sqrt(C(L,m) * C(L,n)))
    double log_binom_m = log_factorial(L) - log_factorial(m) - log_factorial(L - m);
    double log_binom_n = log_factorial(L) - log_factorial(n) - log_factorial(L - n);
    double log_prefactor = 0.5 * (log_binom_m + log_binom_n);
    
    // 计算求和项
    int k_min = (m + n > L) ? (m + n - L) : 0;
    int k_max = (m < n) ? m : n;
    
    double sum_val = 0.0;
    
    for (int k = k_min; k <= k_max; ++k) {
        // 计算 log(1 / (k! * (m-k)! * (n-k)! * (L-m-n+k)!))
        double log_factorial_term = -(log_factorial(k) + log_factorial(m - k) + 
                                     log_factorial(n - k) + log_factorial(L - m - n + k));
        
        // 计算 log(cos^(m+n-2k) * sin^(L-m-n+2k))
        int pow_cos = m + n - 2 * k;
        int pow_sin = L - m - n + 2 * k;
        
        double log_trig = 0.0;
        double sign = 1.0;
        
        // 处理cos项
        if (pow_cos > 0) {
            if (fabs(ct) > 1e-15) {
                log_trig += pow_cos * log(fabs(ct));
            } else {
                // cos接近0，这一项贡献很小，跳过
                continue;
            }
            if (ct < 0 && pow_cos % 2 == 1) sign = -sign;
        }
        
        // 处理sin项
        if (pow_sin > 0) {
            if (fabs(st) > 1e-15) {
                log_trig += pow_sin * log(fabs(st));
            } else {
                // sin接近0，这一项贡献很小，跳过
                continue;
            }
            if (st < 0 && pow_sin % 2 == 1) sign = -sign;
        }
        
        // (-1)^k
        if (k % 2 == 1) sign = -sign;
        
        // 组合所有项
        double log_term = log_prefactor + log_factorial_term + log_trig;
        
        // 检查是否会溢出
        if (log_term > -100.0 && log_term < 100.0) {  // 避免exp溢出
            double term = sign * exp(log_term);
            if (isfinite(term)) {
                sum_val += term;
            }
        }
    }
    
    // 相位因子 exp(i*φ*(n-m))
    double phase = phi * (n - m);
    double result_real = sum_val * cos(phase);
    double result_imag = sum_val * sin(phase);
    
    return make_cuDoubleComplex(result_real, result_imag);
}

/**
 * 双模混合门内核 - Block per Subspace 版本
 * 每个CUDA Block处理一个光子数子空间
 */
__global__ void apply_two_mode_gate_kernel(
    cuDoubleComplex* state_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    int max_photon_number,  // 最大总光子数
    double param1,          // 门参数1 (θ for BS)
    double param2           // 门参数2 (φ for BS)
) {
    // 共享内存用于存储子空间矩阵和向量
    extern __shared__ cuDoubleComplex shared_mem[];
    cuDoubleComplex* sub_matrix = shared_mem;
    cuDoubleComplex* sub_vec_in = &shared_mem[max_photon_number * max_photon_number];
    cuDoubleComplex* sub_vec_out = &sub_vec_in[max_photon_number];

    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    cuDoubleComplex* psi = &state_data[state_idx * d_trunc];

    // 计算当前block处理的子空间
    int L = blockIdx.x;  // 总光子数L
    
    if (L > max_photon_number) return;

    int sub_dim = L + 1;  // 子空间维度 (L+1)

    // 线程组织：每个线程负责矩阵的一个元素
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 加载子空间矩阵到共享内存
    // 矩阵元素 U^(L)_{m,n} 对应 ⟨L-m, m| BS |L-n, n⟩
    if (tid < sub_dim * sub_dim) {
        int m = tid / sub_dim;  // 输出态中mode b的光子数
        int n = tid % sub_dim;  // 输入态中mode b的光子数
        // 计算Beam Splitter矩阵元素
        sub_matrix[m * sub_dim + n] = compute_bs_matrix_element(L, m, n, param1, param2);
    }

    // 加载输入向量到共享内存
    // 状态按光子数分组存储: 索引 = L*(L+1)/2 + k
    // 其中k是mode b的光子数 (状态为 |L-k, k⟩)
    if (tid < sub_dim) {
        int global_idx = L * (L + 1) / 2 + tid;
        if (global_idx < d_trunc) {
            sub_vec_in[tid] = psi[global_idx];
        } else {
            sub_vec_in[tid] = make_cuDoubleComplex(0.0, 0.0);
        }
    }

    __syncthreads();

    // 执行稠密矩阵向量乘法
    // ψ_out[m] = Σ_n U^(L)_{m,n} * ψ_in[n]
    if (tid < sub_dim) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        for (int n = 0; n < sub_dim; ++n) {
            cuDoubleComplex matrix_elem = sub_matrix[tid * sub_dim + n];
            cuDoubleComplex vec_elem = sub_vec_in[n];
            sum = cuCadd(sum, cuCmul(matrix_elem, vec_elem));
        }

        sub_vec_out[tid] = sum;
    }

    __syncthreads();

    // 将结果写回全局内存
    if (tid < sub_dim) {
        int global_idx = L * (L + 1) / 2 + tid;
        if (global_idx < d_trunc) {
            psi[global_idx] = sub_vec_out[tid];
        }
    }
}

/**
 * 优化版本：预计算Beam Splitter矩阵
 * 对于常用的BS参数，可以预计算矩阵存储在常量内存中
 */
#define MAX_SUBSPACE_DIM 16
__constant__ cuDoubleComplex bs_matrix_const[MAX_SUBSPACE_DIM * MAX_SUBSPACE_DIM];

__global__ void apply_two_mode_gate_fast_kernel(
    cuDoubleComplex* state_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    int max_photon_number
) {
    extern __shared__ cuDoubleComplex shared_vec[];

    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    cuDoubleComplex* psi = &state_data[state_idx * d_trunc];

    int subspace_k = blockIdx.x;
    if (subspace_k >= max_photon_number) return;

    int sub_dim = subspace_k + 1;
    int tid = threadIdx.x;

    // 加载输入向量到共享内存
    if (tid < sub_dim) {
        int global_idx = subspace_k * (subspace_k + 1) / 2 + tid;
        shared_vec[tid] = (global_idx < d_trunc) ? psi[global_idx] :
                         make_cuDoubleComplex(0.0, 0.0);
        shared_vec[sub_dim + tid] = make_cuDoubleComplex(0.0, 0.0);  // 输出向量
    }

    __syncthreads();

    // 执行矩阵向量乘法 (使用常量内存中的矩阵)
    if (tid < sub_dim) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        for (int j = 0; j < sub_dim; ++j) {
            cuDoubleComplex matrix_elem = bs_matrix_const[tid * MAX_SUBSPACE_DIM + j];
            cuDoubleComplex vec_elem = shared_vec[j];
            sum = cuCadd(sum, cuCmul(matrix_elem, vec_elem));
        }

        shared_vec[sub_dim + tid] = sum;
    }

    __syncthreads();

    // 写回结果
    if (tid < sub_dim) {
        int global_idx = subspace_k * (subspace_k + 1) / 2 + tid;
        if (global_idx < d_trunc) {
            psi[global_idx] = shared_vec[sub_dim + tid];
        }
    }
}

/**
 * 主机端：预计算Beam Splitter矩阵到常量内存
 */
void prepare_bs_matrix(double theta, double phi, int max_k) {
    int max_dim = max_k + 1;
    if (max_dim > MAX_SUBSPACE_DIM) {
        throw std::runtime_error("子空间维度超过常量内存限制");
    }

    std::vector<cuDoubleComplex> host_matrix(MAX_SUBSPACE_DIM * MAX_SUBSPACE_DIM,
                                           make_cuDoubleComplex(0.0, 0.0));

    // 计算所有子空间的BS矩阵
    for (int k = 0; k <= max_k; ++k) {
        int sub_dim = k + 1;
        int offset = k * (k + 1) / 2;  // 子空间在全局矩阵中的偏移

        for (int i = 0; i < sub_dim; ++i) {
            for (int j = 0; j < sub_dim; ++j) {
                cuDoubleComplex elem = compute_bs_matrix_element(k, i, j, theta, phi);
                int global_row = offset + i;
                int global_col = offset + j;
                host_matrix[global_row * MAX_SUBSPACE_DIM + global_col] = elem;
            }
        }
    }

    // 复制到常量内存
    cudaMemcpyToSymbol(bs_matrix_const, host_matrix.data(),
                      MAX_SUBSPACE_DIM * MAX_SUBSPACE_DIM * sizeof(cuDoubleComplex));
}

/**
 * 主机端接口：应用Beam Splitter门 BS(θ,φ)
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_beam_splitter(CVStatePool* state_pool, const int* target_indices,
                        int batch_size, double theta, double phi, int max_photon_number) {
    // 计算共享内存大小
    int max_sub_dim = max_photon_number + 1;
    size_t shared_mem_size = max_sub_dim * max_sub_dim * sizeof(cuDoubleComplex) +  // 矩阵
                           2 * max_sub_dim * sizeof(cuDoubleComplex);  // 输入输出向量

    dim3 block_dim(16, 16);  // 16x16 = 256 threads per block
    // Grid dimension: need to process subspaces L=0 to L=max_photon_number (inclusive)
    dim3 grid_dim(max_photon_number + 1, 1, batch_size);

    // 使用Block per Subspace版本
    apply_two_mode_gate_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        state_pool->data, state_pool->total_dim, target_indices, batch_size, max_photon_number, theta, phi
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Beam Splitter kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 确保GPU操作完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Beam Splitter kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用优化版Beam Splitter门 (使用常量内存)
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_beam_splitter_fast(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, int max_photon_number) {
    int max_sub_dim = max_photon_number + 1;
    size_t shared_mem_size = 2 * max_sub_dim * sizeof(cuDoubleComplex);

    dim3 block_dim(256);
    dim3 grid_dim(max_photon_number, 1, batch_size);

    apply_two_mode_gate_fast_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        state_pool->data, state_pool->total_dim, target_indices, batch_size, max_photon_number
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Fast Beam Splitter kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Fast Beam Splitter kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用通用双模混合门
 */
void apply_two_mode_gate(CVStatePool* state_pool, const int* target_indices,
                        int batch_size, double param1, double param2, int max_photon_number) {
    apply_beam_splitter(state_pool, target_indices, batch_size,
                       param1, param2, max_photon_number);
}

/**
 * CPU端：使用递推关系构建Two-Mode Squeezing矩阵
 * 
 * 基于Strawberry Fields的实现：
 * Z[m,n,p,q] = <m,n| S2(r,θ) |p,q>
 * 
 * 参数：
 * - r: 挤压强度
 * - theta: 挤压角度
 * 
 * R矩阵（注意整体有负号）：
 * R = -[[0, -conj(eiptr), -sc, 0],
 *      [-conj(eiptr), 0, 0, -sc],
 *      [-sc, 0, 0, eiptr],
 *      [0, -sc, eiptr, 0]]
 * 
 * 递推关系：
 * sc = 1/cosh(r)
 * eiptr = exp(-iθ) * tanh(r)
 * 
 * Rank 1: Z[0,0,0,0] = sc
 * Rank 2: Z[n,n,0,0] = R[0,1] * Z[n-1,n-1,0,0] = conj(eiptr) * Z[n-1,n-1,0,0]
 * Rank 3: Z[m,n,p,0] = R[0,2] * sqrt(m)/sqrt(p) * Z[m-1,n,p-1,0] = sc * sqrt(m)/sqrt(p) * Z[m-1,n,p-1,0]
 * Rank 4: Z[m,n,p,q] = R[1,3] * sqrt(n)/sqrt(q) * Z[m,n-1,p,q-1]
 *                     + R[2,3] * sqrt(p)/sqrt(q) * Z[m,n,p-1,q-1]
 *                     = sc * sqrt(n)/sqrt(q) * Z[m,n-1,p,q-1]
 *                     - eiptr * sqrt(p)/sqrt(q) * Z[m,n,p-1,q-1]
 */
void build_tms_matrix_recursive(
    std::vector<cuDoubleComplex>& Z,
    int cutoff,
    double r,
    double theta) {
    
    int D = cutoff;
    Z.resize(D * D * D * D, make_cuDoubleComplex(0.0, 0.0));
    
    double sc = 1.0 / std::cosh(r);
    double tanh_r = std::tanh(r);
    cuDoubleComplex eiptr = make_cuDoubleComplex(
        std::cos(-theta) * tanh_r,
        std::sin(-theta) * tanh_r
    );
    cuDoubleComplex conj_eiptr = cuConj(eiptr);
    
    // 预计算sqrt表
    std::vector<double> sqrt_table(D);
    for (int i = 0; i < D; ++i) {
        sqrt_table[i] = std::sqrt((double)i);
    }
    
    // Rank 1: 基础情况
    Z[0] = make_cuDoubleComplex(sc, 0.0);  // Z[0,0,0,0] = sc
    
    // Rank 2: 填充 Z[n,n,0,0]
    // R[0,1] = conj(eiptr) (因为R矩阵整体有负号，-(-conj(eiptr)) = conj(eiptr))
    // 但实际上应该产生 (-tanh r)^n 的效果，所以需要负号
    for (int n = 1; n < D; ++n) {
        int idx = n*D*D*D + n*D*D + 0*D + 0;
        int idx_prev = (n-1)*D*D*D + (n-1)*D*D + 0*D + 0;
        // Z[n,n,0,0] = -conj(eiptr) * Z[n-1,n-1,0,0]
        cuDoubleComplex neg_conj_eiptr = cuCmul(make_cuDoubleComplex(-1.0, 0.0), conj_eiptr);
        Z[idx] = cuCmul(neg_conj_eiptr, Z[idx_prev]);
    }
    
    // Rank 3: 填充 Z[m,n,p,0] for m > n
    // R[0,2] = sc (因为R矩阵整体有负号，-(-sc) = sc)
    for (int m = 0; m < D; ++m) {
        for (int n = 0; n < m; ++n) {  // n < m
            int p = m - n;
            if (p > 0 && p < D) {
                int idx = m*D*D*D + n*D*D + p*D + 0;
                
                if (m > 0) {
                    int idx_prev = (m-1)*D*D*D + n*D*D + (p-1)*D + 0;
                    // Z[m,n,p,0] = sc * sqrt(m)/sqrt(p) * Z[m-1,n,p-1,0]
                    double coeff = sc * sqrt_table[m] / sqrt_table[p];
                    Z[idx] = cuCmul(make_cuDoubleComplex(coeff, 0.0), Z[idx_prev]);
                }
            }
        }
    }
    
    // Rank 4: 填充 Z[m,n,p,q] for q > 0
    // R[1,3] = sc (因为R矩阵整体有负号，-(-sc) = sc)
    // R[2,3] = -eiptr (因为R矩阵整体有负号，-(eiptr) = -eiptr)
    for (int m = 0; m < D; ++m) {
        for (int n = 0; n < D; ++n) {
            for (int p = 0; p < D; ++p) {
                int q = p - (m - n);  // 从递推关系推导
                if (q > 0 && q < D) {
                    int idx = m*D*D*D + n*D*D + p*D + q;
                    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
                    
                    // 第一项: sc * sqrt(n)/sqrt(q) * Z[m,n-1,p,q-1]
                    if (n > 0) {
                        int idx1 = m*D*D*D + (n-1)*D*D + p*D + (q-1);
                        double coeff = sc * sqrt_table[n] / sqrt_table[q];
                        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), Z[idx1]));
                    }
                    
                    // 第二项: -eiptr * sqrt(p)/sqrt(q) * Z[m,n,p-1,q-1]
                    if (p > 0) {
                        int idx2 = m*D*D*D + n*D*D + (p-1)*D + (q-1);
                        double coeff = -sqrt_table[p] / sqrt_table[q];
                        cuDoubleComplex term = cuCmul(eiptr, make_cuDoubleComplex(coeff, 0.0));
                        sum = cuCadd(sum, cuCmul(term, Z[idx2]));
                    }
                    
                    Z[idx] = sum;
                }
            }
        }
    }
}

/**
 * 主机端接口：应用Two-Mode Squeezing门 S2(r,θ)
 * 
 * 使用Strawberry Fields递推方法 + GPU缓存策略
 * 
 * 物理意义：
 * - 双模挤压态是量子光学中的重要资源
 * - 用于产生纠缠态和压缩光
 * - 应用于量子通信和量子计算
 * 
 * @param state_pool 状态池（维度应为 D^2，其中D是单模cutoff）
 * @param target_indices 设备端指针，指向目标状态ID数组
 * @param batch_size 批次大小
 * @param r 挤压强度
 * @param theta 挤压角度
 */
void apply_two_mode_squeezing_recursive(CVStatePool* state_pool, const int* target_indices,
                                       int batch_size, double r, double theta) {
    int cutoff = state_pool->total_dim;  // 两模态总维度 = D^2
    int single_mode_cutoff = (int)std::sqrt((double)cutoff);
    
    // 检查缓存是否有效
    bool cache_valid = (d_tms_matrix_cache != nullptr &&
                       cached_tms_cutoff == single_mode_cutoff &&
                       std::abs(cached_tms_r - r) < 1e-10 &&
                       std::abs(cached_tms_theta - theta) < 1e-10);
    
    if (!cache_valid) {
        // 释放旧缓存
        if (d_tms_matrix_cache != nullptr) {
            cudaFree(d_tms_matrix_cache);
            d_tms_matrix_cache = nullptr;
        }
        
        // 在CPU上构建TMS矩阵（使用递推方法）
        std::vector<cuDoubleComplex> h_tms_matrix;
        build_tms_matrix_recursive(h_tms_matrix, single_mode_cutoff, r, theta);
        
        // 分配GPU内存并上传
        size_t matrix_size = h_tms_matrix.size() * sizeof(cuDoubleComplex);
        cudaMalloc(&d_tms_matrix_cache, matrix_size);
        cudaMemcpy(d_tms_matrix_cache, h_tms_matrix.data(), matrix_size, cudaMemcpyHostToDevice);
        
        // 更新缓存参数
        cached_tms_cutoff = single_mode_cutoff;
        cached_tms_r = r;
        cached_tms_theta = theta;
    }
    
    // 为每个batch中的状态应用TMS
    for (int b = 0; b < batch_size; ++b) {
        // 获取状态索引
        int state_idx;
        cudaMemcpy(&state_idx, &target_indices[b], sizeof(int), cudaMemcpyDeviceToHost);
        
        cuDoubleComplex* input_state = &state_pool->data[state_idx * cutoff];
        
        // 需要临时缓冲区避免读写冲突
        cuDoubleComplex* d_temp;
        cudaMalloc(&d_temp, cutoff * sizeof(cuDoubleComplex));
        
        // 启动GPU kernel应用TMS矩阵（复用BS的kernel）
        dim3 block_dim(16, 16);
        dim3 grid_dim((single_mode_cutoff + 15) / 16, (single_mode_cutoff + 15) / 16);
        
        apply_bs_tensor_kernel<<<grid_dim, block_dim>>>(
            input_state, d_temp, d_tms_matrix_cache, single_mode_cutoff);
        
        // 复制结果回原位置
        cudaMemcpy(input_state, d_temp, cutoff * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        cudaFree(d_temp);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Two-Mode Squeezing kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Two-Mode Squeezing kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用Beam Splitter门 BS(θ,φ) - 新递推方法
 * 
 * 使用Strawberry Fields递推方法 + GPU缓存策略（类似挤压门）
 * 
 * 优势：
 * - 数值稳定（无阶乘、无指数）
 * - 完美保真度（保持幺正性）
 * - 使用标准张量积格式（无需转换）
 * - GPU缓存复用（相同参数时无需重新计算）
 * 
 * @param state_pool 状态池（维度应为 D^2，其中D是单模cutoff）
 * @param target_indices 设备端指针，指向目标状态ID数组
 * @param batch_size 批次大小
 * @param theta BS参数θ（透射角）
 * @param phi BS参数φ（反射相位）
 */
void apply_beam_splitter_recursive(CVStatePool* state_pool, const int* target_indices,
                                   int batch_size, double theta, double phi) {
    int cutoff = state_pool->total_dim;  // 两模态总维度 = D^2
    // 从total_dim推导单模cutoff
    int single_mode_cutoff = (int)std::sqrt((double)cutoff);
    
    // 检查缓存是否有效
    bool cache_valid = (d_bs_matrix_cache != nullptr &&
                       cached_cutoff == single_mode_cutoff &&
                       std::abs(cached_theta - theta) < 1e-10 &&
                       std::abs(cached_phi - phi) < 1e-10);
    
    if (!cache_valid) {
        // 释放旧缓存
        if (d_bs_matrix_cache != nullptr) {
            cudaFree(d_bs_matrix_cache);
            d_bs_matrix_cache = nullptr;
        }
        
        // 在CPU上构建BS矩阵（使用递推方法）
        std::vector<cuDoubleComplex> h_bs_matrix;
        build_bs_matrix_recursive(h_bs_matrix, single_mode_cutoff, theta, phi);
        
        // 分配GPU内存并上传
        size_t matrix_size = h_bs_matrix.size() * sizeof(cuDoubleComplex);
        cudaMalloc(&d_bs_matrix_cache, matrix_size);
        cudaMemcpy(d_bs_matrix_cache, h_bs_matrix.data(), matrix_size, cudaMemcpyHostToDevice);
        
        // 更新缓存参数
        cached_cutoff = single_mode_cutoff;
        cached_theta = theta;
        cached_phi = phi;
    }
    
    // 为每个batch中的状态应用BS
    for (int b = 0; b < batch_size; ++b) {
        // 获取状态索引
        int state_idx;
        cudaMemcpy(&state_idx, &target_indices[b], sizeof(int), cudaMemcpyDeviceToHost);
        
        cuDoubleComplex* input_state = &state_pool->data[state_idx * cutoff];
        
        // 需要临时缓冲区避免读写冲突
        cuDoubleComplex* d_temp;
        cudaMalloc(&d_temp, cutoff * sizeof(cuDoubleComplex));
        
        // 启动GPU kernel应用BS矩阵
        dim3 block_dim(16, 16);
        dim3 grid_dim((single_mode_cutoff + 15) / 16, (single_mode_cutoff + 15) / 16);
        
        apply_bs_tensor_kernel<<<grid_dim, block_dim>>>(
            input_state, d_temp, d_bs_matrix_cache, single_mode_cutoff);
        
        // 复制结果回原位置
        cudaMemcpy(input_state, d_temp, cutoff * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        cudaFree(d_temp);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Beam Splitter recursive kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Beam Splitter recursive kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
