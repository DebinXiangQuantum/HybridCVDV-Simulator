#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "cv_state_pool.h"

// Functions defined in two_mode_gates.cu, used by specialized gate implementations.
void apply_cached_two_mode_tensor_gate(CVStatePool* state_pool,
                                       const int* target_indices,
                                       int batch_size,
                                       int single_mode_cutoff,
                                       const cuDoubleComplex* tensor_matrix,
                                       int target_qumode1,
                                       int target_qumode2,
                                       int num_qumodes,
                                       cudaStream_t stream,
                                       bool synchronize);

void build_bs_matrix_recursive(std::vector<cuDoubleComplex>& Z,
                               int cutoff, double theta, double phi);

std::vector<cuDoubleComplex> build_tms_tensor_dense(int cutoff, double r, double theta);
std::vector<cuDoubleComplex> build_sum_tensor_dense(int cutoff, double scale);

namespace {

using HostComplex = std::complex<double>;

struct TensorMatrixCacheEntry {
    int cutoff = 0;
    double param1 = 0.0;
    double param2 = 0.0;
    cuDoubleComplex* device_matrix = nullptr;
    uint64_t last_use = 0;
};

constexpr int kTwoModeTensorCacheCapacity = 8;
constexpr double kTwoModeTensorCacheTolerance = 1e-10;

bool cache_entry_matches(const TensorMatrixCacheEntry& entry,
                         int cutoff,
                         double param1,
                         double param2) {
    return entry.device_matrix != nullptr &&
           entry.cutoff == cutoff &&
           std::abs(entry.param1 - param1) < kTwoModeTensorCacheTolerance &&
           std::abs(entry.param2 - param2) < kTwoModeTensorCacheTolerance;
}

template <typename Builder>
cuDoubleComplex* get_or_build_two_mode_tensor_cache(
    std::vector<TensorMatrixCacheEntry>* cache_entries,
    uint64_t* use_counter,
    int cutoff,
    double param1,
    double param2,
    const char* allocation_error_prefix,
    const char* upload_error_prefix,
    Builder&& builder) {
    for (auto& entry : *cache_entries) {
        if (cache_entry_matches(entry, cutoff, param1, param2)) {
            entry.last_use = ++(*use_counter);
            return entry.device_matrix;
        }
    }

    const std::vector<cuDoubleComplex> host_matrix = builder();
    cuDoubleComplex* device_matrix = nullptr;
    cudaError_t err = cudaMalloc(&device_matrix, host_matrix.size() * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(allocation_error_prefix) + cudaGetErrorString(err));
    }

    err = cudaMemcpy(device_matrix,
                     host_matrix.data(),
                     host_matrix.size() * sizeof(cuDoubleComplex),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_matrix);
        throw std::runtime_error(std::string(upload_error_prefix) + cudaGetErrorString(err));
    }

    TensorMatrixCacheEntry new_entry;
    new_entry.cutoff = cutoff;
    new_entry.param1 = param1;
    new_entry.param2 = param2;
    new_entry.device_matrix = device_matrix;
    new_entry.last_use = ++(*use_counter);

    if (cache_entries->size() < static_cast<size_t>(kTwoModeTensorCacheCapacity)) {
        cache_entries->push_back(new_entry);
        return device_matrix;
    }

    auto lru_it = std::min_element(
        cache_entries->begin(),
        cache_entries->end(),
        [](const TensorMatrixCacheEntry& lhs, const TensorMatrixCacheEntry& rhs) {
            return lhs.last_use < rhs.last_use;
        });
    if (lru_it != cache_entries->end() && lru_it->device_matrix != nullptr) {
        cudaFree(lru_it->device_matrix);
    }
    *lru_it = new_entry;
    return device_matrix;
}

int compute_mode_right_stride(int trunc_dim, int target_qumode, int num_qumodes) {
    if (trunc_dim <= 0) {
        throw std::invalid_argument("truncation dimension must be positive");
    }
    if (num_qumodes <= 0) {
        throw std::invalid_argument("number of qumodes must be positive");
    }
    if (target_qumode < 0 || target_qumode >= num_qumodes) {
        throw std::out_of_range("target qumode is out of range");
    }

    int right_stride = 1;
    for (int mode = target_qumode + 1; mode < num_qumodes; ++mode) {
        right_stride *= trunc_dim;
    }
    return right_stride;
}

std::vector<int> copy_target_indices_to_host(const int* target_indices, int batch_size) {
    std::vector<int> host_targets(batch_size, -1);
    if (batch_size <= 0) {
        return host_targets;
    }

    cudaError_t err = cudaMemcpy(host_targets.data(), target_indices,
                                 static_cast<size_t>(batch_size) * sizeof(int),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy target indices: " +
                                 std::string(cudaGetErrorString(err)));
    }

    return host_targets;
}

int infer_single_mode_cutoff(const CVStatePool* state_pool, int num_qumodes) {
    if (num_qumodes <= 0) {
        throw std::invalid_argument("number of qumodes must be positive");
    }
    const double inferred =
        std::pow(static_cast<double>(state_pool->max_total_dim), 1.0 / static_cast<double>(num_qumodes));
    const int cutoff = static_cast<int>(std::llround(inferred));
    if (cutoff <= 0) {
        throw std::runtime_error("failed to infer single-mode cutoff");
    }
    return cutoff;
}

}  // namespace

// Global caches for tensor gate matrices, shared across gate calls.
static std::vector<TensorMatrixCacheEntry> g_bs_matrix_cache_entries;
static uint64_t g_bs_matrix_cache_use_counter = 0;
static std::vector<TensorMatrixCacheEntry> g_tms_matrix_cache_entries;
static uint64_t g_tms_matrix_cache_use_counter = 0;
static std::vector<TensorMatrixCacheEntry> g_sum_matrix_cache_entries;
static uint64_t g_sum_matrix_cache_use_counter = 0;

// ==========================================
// Exponential SWAP Gate (eSWAP)
// ==========================================

/**
 * Exponential SWAP 门内核
 * eSWAP(θ) = exp(iθ * SWAP)
 * 
 * SWAP 交换两个模式的光子数
 * |m, n⟩ → |n, m⟩
 */
__global__ void apply_eswap_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double theta,
    int cutoff_a,
    int cutoff_b
) {
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= cutoff_a || n >= cutoff_b) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // 计算索引
    int idx_mn = m * cutoff_b + n;  // |m, n⟩
    int idx_nm = n * cutoff_a + m;  // |n, m⟩

    // eSWAP 矩阵元素
    // 对角元素：cos(θ)
    // 非对角元素（交换）：i*sin(θ)
    
    if (m == n) {
        // 对角元素：保持不变（cos(θ) ≈ 1 for small θ）
        double cos_theta = cos(theta);
        psi[idx_mn] = cuCmul(psi[idx_mn], make_cuDoubleComplex(cos_theta, 0.0));
    } else if (m < n) {
        // 只处理上三角，避免重复
        cuDoubleComplex psi_mn = psi[idx_mn];
        cuDoubleComplex psi_nm = psi[idx_nm];
        
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        
        // 新的 |m,n⟩ = cos(θ)|m,n⟩ + i*sin(θ)|n,m⟩
        psi[idx_mn] = make_cuDoubleComplex(
            cos_theta * cuCreal(psi_mn) - sin_theta * cuCimag(psi_nm),
            cos_theta * cuCimag(psi_mn) + sin_theta * cuCreal(psi_nm)
        );
        
        // 新的 |n,m⟩ = i*sin(θ)|m,n⟩ + cos(θ)|n,m⟩
        psi[idx_nm] = make_cuDoubleComplex(
            -sin_theta * cuCimag(psi_mn) + cos_theta * cuCreal(psi_nm),
            sin_theta * cuCreal(psi_mn) + cos_theta * cuCimag(psi_nm)
        );
    }
}

void apply_exponential_swap(CVStatePool* state_pool, const int* target_indices,
                           int batch_size, double theta, int cutoff_a, int cutoff_b) {
    dim3 block_dim(16, 16);
    dim3 grid_dim((cutoff_a + 15) / 16, (cutoff_b + 15) / 16, batch_size);

    apply_eswap_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        theta, cutoff_a, cutoff_b
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Exponential SWAP kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Exponential SWAP kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

// ==========================================
// SUM Gate
// ==========================================

/**
 * SUM 门内核
 * SUM(s) = exp[s/2 * (a + a†) ⊗ (b† - b)]
 * 
 * 这是一个双模门，耦合两个模式
 */
__global__ void apply_sum_gate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double scale,
    int cutoff_a,
    int cutoff_b,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= cutoff_a || n >= cutoff_b) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    // SUM 门的矩阵元素计算（简化版本）
    // 完整实现需要矩阵指数，这里使用近似
    
    int idx = m * cutoff_b + n;
    
    // 简化：只实现一阶近似
    // 实际应该使用完整的矩阵指数
    cuDoubleComplex sum = psi_in[idx];
    
    // 添加耦合项的贡献
    if (m > 0 && n < cutoff_b - 1) {
        double coeff = scale * sqrt((double)m) * sqrt((double)(n + 1));
        int idx_coupled = (m - 1) * cutoff_b + (n + 1);
        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), psi_in[idx_coupled]));
    }
    
    if (m < cutoff_a - 1 && n > 0) {
        double coeff = scale * sqrt((double)(m + 1)) * sqrt((double)n);
        int idx_coupled = (m + 1) * cutoff_b + (n - 1);
        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), psi_in[idx_coupled]));
    }
    
    psi_out[idx] = sum;
}

void apply_sum_gate(CVStatePool* state_pool, const int* target_indices,
                   int batch_size, double scale, int cutoff_a, int cutoff_b,
                   int target_qumode1, int target_qumode2, int num_qumodes,
                   cudaStream_t stream, bool synchronize) {
    if (batch_size <= 0) {
        return;
    }

    if (cutoff_a != cutoff_b) {
        throw std::runtime_error("SUM gate currently requires identical mode cutoffs");
    }

    const int single_mode_cutoff = cutoff_a;
    const cuDoubleComplex* sum_matrix = get_or_build_two_mode_tensor_cache(
        &g_sum_matrix_cache_entries,
        &g_sum_matrix_cache_use_counter,
        single_mode_cutoff,
        scale,
        0.0,
        "SUM gate matrix allocation failed: ",
        "SUM gate matrix upload failed: ",
        [&]() {
            return build_sum_tensor_dense(single_mode_cutoff, scale);
        });

    apply_cached_two_mode_tensor_gate(
        state_pool,
        target_indices,
        batch_size,
        single_mode_cutoff,
        sum_matrix,
        target_qumode1,
        target_qumode2,
        num_qumodes,
        stream,
        synchronize);
}

// ==========================================
// Three-Mode Squeezing (S3)
// ==========================================

/**
 * 三模压缩门内核
 * S3(θ) = exp[θ * a† b† c† - θ* a b c]
 * 
 * 这是一个三模门，产生三模纠缠
 */
__global__ void apply_three_mode_squeezing_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex theta,
    int cutoff_a,
    int cutoff_b,
    int cutoff_c,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = threadIdx.z;

    if (i >= cutoff_a || j >= cutoff_b || k >= cutoff_c) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    int idx = i * cutoff_b * cutoff_c + j * cutoff_c + k;
    
    // 简化实现：只包含主要项
    // 完整实现需要矩阵指数
    cuDoubleComplex sum = psi_in[idx];
    
    // 添加三模耦合项
    if (i < cutoff_a - 1 && j < cutoff_b - 1 && k < cutoff_c - 1) {
        double coeff = cuCreal(theta) * sqrt((double)((i+1)*(j+1)*(k+1)));
        int idx_up = (i+1) * cutoff_b * cutoff_c + (j+1) * cutoff_c + (k+1);
        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), psi_in[idx_up]));
    }
    
    if (i > 0 && j > 0 && k > 0) {
        double coeff = -cuCreal(theta) * sqrt((double)(i*j*k));
        int idx_down = (i-1) * cutoff_b * cutoff_c + (j-1) * cutoff_c + (k-1);
        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), psi_in[idx_down]));
    }
    
    psi_out[idx] = sum;
}

void apply_three_mode_squeezing(CVStatePool* state_pool, const int* target_indices,
                                int batch_size, cuDoubleComplex theta,
                                int cutoff_a, int cutoff_b, int cutoff_c) {
    size_t buffer_stride = cutoff_a * cutoff_b * cutoff_c;
    const size_t temp_bytes = static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex);
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(temp_bytes));

    dim3 block_dim(8, 8, 4);
    dim3 grid_dim((cutoff_a + 7) / 8, (cutoff_b + 7) / 8, batch_size);

    apply_three_mode_squeezing_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        theta, cutoff_a, cutoff_b, cutoff_c,
        temp_buffer, buffer_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Three-mode squeezing kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Three-mode squeezing kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    const std::vector<int> host_targets = copy_target_indices_to_host(target_indices, batch_size);
    for (int b = 0; b < batch_size; ++b) {
        cuDoubleComplex* state_ptr = state_pool->get_state_ptr(host_targets[b]);
        if (!state_ptr) {
            throw std::runtime_error("Invalid state ID for three-mode squeezing");
        }

        cudaError_t copy_err = cudaMemcpy(state_ptr, &temp_buffer[static_cast<size_t>(b) * buffer_stride],
                                          buffer_stride * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error("Three-mode squeezing write-back failed: " +
                                     std::string(cudaGetErrorString(copy_err)));
        }
    }
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
                                       int batch_size, double r, double theta,
                                       int target_qumode1, int target_qumode2, int num_qumodes,
                                       cudaStream_t stream, bool synchronize) {
    if (batch_size <= 0) {
        return;
    }

    const int single_mode_cutoff = infer_single_mode_cutoff(state_pool, num_qumodes);
    const cuDoubleComplex* tms_matrix = get_or_build_two_mode_tensor_cache(
        &g_tms_matrix_cache_entries,
        &g_tms_matrix_cache_use_counter,
        single_mode_cutoff,
        r,
        theta,
        "Two-Mode Squeezing matrix allocation failed: ",
        "Two-Mode Squeezing matrix upload failed: ",
        [&]() {
            return build_tms_tensor_dense(single_mode_cutoff, r, theta);
        });
    
    apply_cached_two_mode_tensor_gate(
        state_pool,
        target_indices,
        batch_size,
        single_mode_cutoff,
        tms_matrix,
        target_qumode1,
        target_qumode2,
        num_qumodes,
        stream,
        synchronize);
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
                                   int batch_size, double theta, double phi,
                                   int target_qumode1, int target_qumode2, int num_qumodes,
                                   cudaStream_t stream, bool synchronize) {
    if (batch_size <= 0) {
        return;
    }

    int single_mode_cutoff = infer_single_mode_cutoff(state_pool, num_qumodes);
    const cuDoubleComplex* bs_matrix = get_or_build_two_mode_tensor_cache(
        &g_bs_matrix_cache_entries,
        &g_bs_matrix_cache_use_counter,
        single_mode_cutoff,
        theta,
        phi,
        "Beam Splitter matrix allocation failed: ",
        "Beam Splitter matrix upload failed: ",
        [&]() {
            std::vector<cuDoubleComplex> h_bs_matrix;
            build_bs_matrix_recursive(h_bs_matrix, single_mode_cutoff, theta, phi);
            return h_bs_matrix;
        });
    
    apply_cached_two_mode_tensor_gate(
        state_pool,
        target_indices,
        batch_size,
        single_mode_cutoff,
        bs_matrix,
        target_qumode1,
        target_qumode2,
        num_qumodes,
        stream,
        synchronize);
}

// ==========================================
// Strawberry Fields 双模门扩展
// ==========================================

/**
 * MZgate (Mach-Zehnder 干涉仪)
 * MZ(φ_in, φ_ex) = BS(π/4, 0) · [R(φ_ex) ⊗ R(φ_in)] · BS(π/4, 0)
 */
void apply_mzgate(CVStatePool* state_pool, const int* target_indices,
                 int batch_size, double phi_in, double phi_ex, int cutoff_a, int cutoff_b) {
    // MZ 门分解为：BS(π/4) -> R(φ) -> BS(π/4)
    apply_beam_splitter_recursive(
        state_pool, target_indices, batch_size, M_PI / 4, 0.0, 0, 1, 2, nullptr, true);
    // 中间的相位旋转需要单独实现
    apply_beam_splitter_recursive(
        state_pool, target_indices, batch_size, M_PI / 4, 0.0, 0, 1, 2, nullptr, true);
}

/**
 * CZgate (受控相位) 内核
 * CZ(s) = exp(i s x̂₁ ⊗ x̂₂/ℏ)
 */
__global__ void apply_czgate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double s,
    int cutoff_a,
    int cutoff_b
) {
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= cutoff_a || n >= cutoff_b) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    int idx = m * cutoff_b + n;

    // CZ|m,n⟩ = exp(i s m n)|m,n⟩
    double phase = s * m * n;
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));

    psi[idx] = cuCmul(psi[idx], phase_factor);
}

void apply_czgate(CVStatePool* state_pool, const int* target_indices,
                 int batch_size, double s, int cutoff_a, int cutoff_b) {
    dim3 block_dim(16, 16);
    dim3 grid_dim((cutoff_a + 15) / 16, (cutoff_b + 15) / 16, batch_size);

    apply_czgate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        s, cutoff_a, cutoff_b
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CZgate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CZgate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * CKgate (Cross-Kerr) 内核
 * CK(κ) = exp(i κ n̂₁ ⊗ n̂₂)
 */
__global__ void apply_ckgate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double kappa,
    int cutoff_a,
    int cutoff_b
) {
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= cutoff_a || n >= cutoff_b) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    int idx = m * cutoff_b + n;

    // CK(κ)|m,n⟩ = exp(i κ m n) |m,n⟩
    double phase = kappa * m * n;
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));

    psi[idx] = cuCmul(psi[idx], phase_factor);
}

__global__ void apply_ckgate_multimode_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double kappa,
    int first_mode_dim,
    int first_mode_right_stride,
    int second_mode_dim,
    int second_mode_right_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    const int state_idx = target_indices[batch_id];
    const size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    cuDoubleComplex* psi = &state_data[state_offsets[state_idx]];
    const int first_photon =
        static_cast<int>((flat_index / static_cast<size_t>(first_mode_right_stride)) % first_mode_dim);
    const int second_photon =
        static_cast<int>((flat_index / static_cast<size_t>(second_mode_right_stride)) % second_mode_dim);

    const double phase = kappa * static_cast<double>(first_photon * second_photon);
    const cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));
    psi[flat_index] = cuCmul(psi[flat_index], phase_factor);
}

void apply_ckgate(CVStatePool* state_pool, const int* target_indices,
                 int batch_size, double kappa, int cutoff_a, int cutoff_b) {
    dim3 block_dim(16, 16);
    dim3 grid_dim((cutoff_a + 15) / 16, (cutoff_b + 15) / 16, batch_size);

    apply_ckgate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        kappa, cutoff_a, cutoff_b
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CKgate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CKgate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void apply_ckgate_on_modes(CVStatePool* state_pool, const int* target_indices,
                           int batch_size, double kappa,
                           int target_qumode1, int target_qumode2,
                           int num_qumodes) {
    if (target_qumode1 == target_qumode2) {
        throw std::invalid_argument("Cross-Kerr requires two distinct target qumodes");
    }

    const int first_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode1, num_qumodes);
    const int second_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode2, num_qumodes);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_ckgate_multimode_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices,
        batch_size,
        kappa,
        state_pool->d_trunc,
        first_mode_right_stride,
        state_pool->d_trunc,
        second_mode_right_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Cross-Kerr multi-mode kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Cross-Kerr multi-mode kernel synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}
