#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
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

__global__ void apply_bs_tensor_kernel(
    const cuDoubleComplex* input_state,
    cuDoubleComplex* output_state,
    const cuDoubleComplex* bs_matrix,
    int cutoff);

namespace {

using HostComplex = std::complex<double>;
using HostMatrix = std::vector<HostComplex>;

HostComplex to_host_complex(cuDoubleComplex value) {
    return HostComplex(cuCreal(value), cuCimag(value));
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

cuDoubleComplex to_device_complex(const HostComplex& value) {
    return make_cuDoubleComplex(value.real(), value.imag());
}

HostMatrix create_identity_matrix(int dim) {
    HostMatrix identity(static_cast<size_t>(dim) * dim, HostComplex(0.0, 0.0));
    for (int i = 0; i < dim; ++i) {
        identity[static_cast<size_t>(i) * dim + i] = HostComplex(1.0, 0.0);
    }
    return identity;
}

double matrix_max_abs(const HostMatrix& matrix) {
    double max_abs = 0.0;
    for (const auto& value : matrix) {
        max_abs = std::max(max_abs, std::abs(value));
    }
    return max_abs;
}

double matrix_one_norm(const HostMatrix& matrix, int dim) {
    double max_sum = 0.0;
    for (int col = 0; col < dim; ++col) {
        double sum = 0.0;
        for (int row = 0; row < dim; ++row) {
            sum += std::abs(matrix[static_cast<size_t>(row) * dim + col]);
        }
        max_sum = std::max(max_sum, sum);
    }
    return max_sum;
}

HostMatrix matrix_multiply(const HostMatrix& lhs, const HostMatrix& rhs, int dim) {
    HostMatrix result(static_cast<size_t>(dim) * dim, HostComplex(0.0, 0.0));
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            HostComplex sum(0.0, 0.0);
            for (int k = 0; k < dim; ++k) {
                sum += lhs[static_cast<size_t>(row) * dim + k] *
                       rhs[static_cast<size_t>(k) * dim + col];
            }
            result[static_cast<size_t>(row) * dim + col] = sum;
        }
    }
    return result;
}

HostMatrix matrix_scale(const HostMatrix& matrix, double scale) {
    HostMatrix result = matrix;
    for (auto& value : result) {
        value *= scale;
    }
    return result;
}

void matrix_add_inplace(HostMatrix& target, const HostMatrix& source) {
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] += source[i];
    }
}

HostMatrix matrix_exponential(const HostMatrix& matrix, int dim) {
    if (matrix.empty()) {
        return {};
    }

    const double norm = matrix_one_norm(matrix, dim);
    const int scaling_power = norm > 1.0 ? static_cast<int>(std::ceil(std::log2(norm))) : 0;
    const double scale = std::ldexp(1.0, scaling_power);
    const HostMatrix scaled_matrix = matrix_scale(matrix, 1.0 / scale);

    HostMatrix result = create_identity_matrix(dim);
    HostMatrix term = result;

    for (int order = 1; order <= 80; ++order) {
        term = matrix_multiply(term, scaled_matrix, dim);
        const double inv_order = 1.0 / static_cast<double>(order);
        for (auto& value : term) {
            value *= inv_order;
        }
        matrix_add_inplace(result, term);
        if (matrix_max_abs(term) < 1e-14) {
            break;
        }
    }

    for (int i = 0; i < scaling_power; ++i) {
        result = matrix_multiply(result, result, dim);
    }

    return result;
}

std::vector<cuDoubleComplex> dense_matrix_to_two_mode_tensor(const HostMatrix& matrix, int cutoff) {
    const int dim = cutoff * cutoff;
    std::vector<cuDoubleComplex> tensor(static_cast<size_t>(dim) * dim, make_cuDoubleComplex(0.0, 0.0));

    for (int m = 0; m < cutoff; ++m) {
        for (int n = 0; n < cutoff; ++n) {
            const int out_idx = m * cutoff + n;
            for (int p = 0; p < cutoff; ++p) {
                for (int q = 0; q < cutoff; ++q) {
                    const int in_idx = p * cutoff + q;
                    const size_t tensor_idx =
                        static_cast<size_t>(m) * cutoff * cutoff * cutoff +
                        static_cast<size_t>(n) * cutoff * cutoff +
                        static_cast<size_t>(p) * cutoff +
                        static_cast<size_t>(q);
                    tensor[tensor_idx] = to_device_complex(matrix[static_cast<size_t>(out_idx) * dim + in_idx]);
                }
            }
        }
    }

    return tensor;
}

std::vector<cuDoubleComplex> build_tms_tensor_dense(int cutoff, double r, double theta) {
    const int dim = cutoff * cutoff;
    HostMatrix generator(static_cast<size_t>(dim) * dim, HostComplex(0.0, 0.0));
    const HostComplex xi = std::polar(r, theta);

    for (int p = 0; p < cutoff; ++p) {
        for (int q = 0; q < cutoff; ++q) {
            const int input_idx = p * cutoff + q;

            if (p + 1 < cutoff && q + 1 < cutoff) {
                const int output_idx = (p + 1) * cutoff + (q + 1);
                const double coeff = 0.5 * std::sqrt(static_cast<double>((p + 1) * (q + 1)));
                generator[static_cast<size_t>(output_idx) * dim + input_idx] += std::conj(xi) * coeff;
            }

            if (p > 0 && q > 0) {
                const int output_idx = (p - 1) * cutoff + (q - 1);
                const double coeff = -0.5 * std::sqrt(static_cast<double>(p * q));
                generator[static_cast<size_t>(output_idx) * dim + input_idx] += xi * coeff;
            }
        }
    }

    return dense_matrix_to_two_mode_tensor(matrix_exponential(generator, dim), cutoff);
}

std::vector<cuDoubleComplex> build_sum_tensor_dense(int cutoff, double scale) {
    const int dim = cutoff * cutoff;
    HostMatrix generator(static_cast<size_t>(dim) * dim, HostComplex(0.0, 0.0));
    const double prefactor = 0.5 * scale;

    for (int p = 0; p < cutoff; ++p) {
        for (int q = 0; q < cutoff; ++q) {
            const int input_idx = p * cutoff + q;

            if (p > 0 && q + 1 < cutoff) {
                const int output_idx = (p - 1) * cutoff + (q + 1);
                const double coeff = prefactor * std::sqrt(static_cast<double>(p * (q + 1)));
                generator[static_cast<size_t>(output_idx) * dim + input_idx] += coeff;
            }

            if (p > 0 && q > 0) {
                const int output_idx = (p - 1) * cutoff + (q - 1);
                const double coeff = -prefactor * std::sqrt(static_cast<double>(p * q));
                generator[static_cast<size_t>(output_idx) * dim + input_idx] += coeff;
            }

            if (p + 1 < cutoff && q + 1 < cutoff) {
                const int output_idx = (p + 1) * cutoff + (q + 1);
                const double coeff = prefactor * std::sqrt(static_cast<double>((p + 1) * (q + 1)));
                generator[static_cast<size_t>(output_idx) * dim + input_idx] += coeff;
            }

            if (p + 1 < cutoff && q > 0) {
                const int output_idx = (p + 1) * cutoff + (q - 1);
                const double coeff = -prefactor * std::sqrt(static_cast<double>((p + 1) * q));
                generator[static_cast<size_t>(output_idx) * dim + input_idx] += coeff;
            }
        }
    }

    return dense_matrix_to_two_mode_tensor(matrix_exponential(generator, dim), cutoff);
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

__global__ void apply_two_mode_tensor_gate_on_modes_kernel(
    const cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    const cuDoubleComplex* tensor_matrix,
    int single_mode_cutoff,
    int mode1_stride,
    int mode2_stride,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    const size_t offset = state_offsets[state_idx];
    const cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    const size_t stride1 = static_cast<size_t>(mode1_stride);
    const size_t stride2 = static_cast<size_t>(mode2_stride);
    const int out_mode1 = static_cast<int>((flat_index / stride1) % single_mode_cutoff);
    const int out_mode2 = static_cast<int>((flat_index / stride2) % single_mode_cutoff);
    const size_t base_index =
        flat_index - static_cast<size_t>(out_mode1) * stride1 - static_cast<size_t>(out_mode2) * stride2;

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int in_mode1 = 0; in_mode1 < single_mode_cutoff; ++in_mode1) {
        for (int in_mode2 = 0; in_mode2 < single_mode_cutoff; ++in_mode2) {
            const size_t matrix_index =
                static_cast<size_t>(out_mode1) * single_mode_cutoff * single_mode_cutoff * single_mode_cutoff +
                static_cast<size_t>(out_mode2) * single_mode_cutoff * single_mode_cutoff +
                static_cast<size_t>(in_mode1) * single_mode_cutoff +
                static_cast<size_t>(in_mode2);
            const size_t source_index =
                base_index + static_cast<size_t>(in_mode1) * stride1 + static_cast<size_t>(in_mode2) * stride2;
            sum = cuCadd(sum, cuCmul(tensor_matrix[matrix_index], psi_in[source_index]));
        }
    }

    psi_out[flat_index] = sum;
}

__global__ void copy_back_two_mode_tensor_gate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    const cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    const size_t offset = state_offsets[state_idx];
    state_data[offset + flat_index] = temp_buffer[batch_id * buffer_stride + flat_index];
}

void apply_cached_two_mode_tensor_gate(CVStatePool* state_pool,
                                       const int* target_indices,
                                       int batch_size,
                                       int single_mode_cutoff,
                                       const cuDoubleComplex* tensor_matrix,
                                       int target_qumode1,
                                       int target_qumode2,
                                       int num_qumodes) {
    if (batch_size <= 0) {
        return;
    }
    if (target_qumode1 == target_qumode2) {
        throw std::invalid_argument("two-mode gate targets must be distinct");
    }

    const int mode1_stride =
        compute_mode_right_stride(single_mode_cutoff, target_qumode1, num_qumodes);
    const int mode2_stride =
        compute_mode_right_stride(single_mode_cutoff, target_qumode2, num_qumodes);
    const size_t buffer_stride = state_pool->max_total_dim;

    const size_t temp_bytes = static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex);
    cuDoubleComplex* temp_state = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(temp_bytes));

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_two_mode_tensor_gate_on_modes_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices,
        batch_size,
        tensor_matrix,
        single_mode_cutoff,
        mode1_stride,
        mode2_stride,
        temp_state,
        buffer_stride
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Two-mode tensor kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Two-mode tensor kernel synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    copy_back_two_mode_tensor_gate_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices,
        batch_size,
        temp_state,
        buffer_stride
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Two-mode tensor write-back failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Two-mode tensor write-back synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

}  // namespace

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

// 全局缓存：存储预计算的SUM矩阵
static cuDoubleComplex* d_sum_matrix_cache = nullptr;
static int cached_sum_cutoff = 0;
static double cached_sum_scale = -999.0;

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
__global__ void apply_two_mode_gate_kernel_cached(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    int single_mode_cutoff,
    const int* target_indices,
    int batch_size,
    int max_photon_number,
    const cuDoubleComplex* cached_matrices
) {
    extern __shared__ cuDoubleComplex shared_mem[];

    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    cuDoubleComplex* psi = &state_data[state_offsets[state_idx]];

    int L = blockIdx.x;  
    if (L > max_photon_number) return;

    int sub_dim = L + 1;  
    int padded_stride = sub_dim + (sub_dim % 2 == 0 ? 1 : 0);

    cuDoubleComplex* sub_matrix = shared_mem;
    cuDoubleComplex* sub_vec_in = &shared_mem[sub_dim * padded_stride];
    cuDoubleComplex* sub_vec_out = &sub_vec_in[sub_dim];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Load sub-matrix from global cache
    if (tid < sub_dim * sub_dim) {
        int m = tid / sub_dim;  
        int n = tid % sub_dim;  
        int offset = L * (L + 1) * (2 * L + 1) / 6;
        sub_matrix[m * padded_stride + n] = cached_matrices[offset + tid];
    }

    // Load state vector mapped from tensor product grid
    if (tid < sub_dim) {
        int p = L - tid;
        int q = tid;
        if (p < single_mode_cutoff && q < single_mode_cutoff) {
            int global_idx = p * single_mode_cutoff + q;
            sub_vec_in[tid] = psi[global_idx];
        } else {
            sub_vec_in[tid] = make_cuDoubleComplex(0.0, 0.0);
        }
    }

    __syncthreads();

    // GEMV
    if (tid < sub_dim) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (int n = 0; n < sub_dim; ++n) {
            cuDoubleComplex matrix_elem = sub_matrix[tid * padded_stride + n];
            cuDoubleComplex vec_elem = sub_vec_in[n];
            sum = cuCadd(sum, cuCmul(matrix_elem, vec_elem));
        }
        sub_vec_out[tid] = sum;
    }

    __syncthreads();

    // Store back to tensor product grid
    if (tid < sub_dim) {
        int p = L - tid;
        int q = tid;
        if (p < single_mode_cutoff && q < single_mode_cutoff) {
            int global_idx = p * single_mode_cutoff + q;
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

static cuDoubleComplex* d_bs_subspace_cache = nullptr;
static double cached_bs_subspace_theta = -999.0;
static double cached_bs_subspace_phi = -999.0;
static int cached_bs_subspace_max_k = -1;

void prepare_bs_subspace_matrices(double theta, double phi, int max_k) {
    int total_elements = (max_k + 1) * (max_k + 2) * (2 * max_k + 3) / 6;
    
    std::vector<cuDoubleComplex> h_matrices(total_elements);
    
    for (int k = 0; k <= max_k; ++k) {
        int sub_dim = k + 1;
        int offset = k * (k + 1) * (2 * k + 1) / 6;
        for (int m = 0; m < sub_dim; ++m) {
            for (int n = 0; n < sub_dim; ++n) {
                h_matrices[offset + m * sub_dim + n] = compute_bs_matrix_element(k, m, n, theta, phi);
            }
        }
    }
    
    if (d_bs_subspace_cache == nullptr || cached_bs_subspace_max_k < max_k) {
        if (d_bs_subspace_cache) cudaFree(d_bs_subspace_cache);
        cudaMalloc(&d_bs_subspace_cache, total_elements * sizeof(cuDoubleComplex));
    }
    
    cudaMemcpy(d_bs_subspace_cache, h_matrices.data(), total_elements * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    cached_bs_subspace_theta = theta;
    cached_bs_subspace_phi = phi;
    cached_bs_subspace_max_k = max_k;
}

void apply_beam_splitter(CVStatePool* state_pool, const int* target_indices,
                        int batch_size, double theta, double phi, int max_photon_number) {
    int single_mode_cutoff = state_pool->d_trunc;
    
    // To avoid truncation of tensor product space, max_photon_number MUST be enough to cover it
    int safe_max_photon = 2 * single_mode_cutoff - 2;
    if (max_photon_number < safe_max_photon) {
        max_photon_number = safe_max_photon;
    }

    if (d_bs_subspace_cache == nullptr || 
        std::abs(theta - cached_bs_subspace_theta) > 1e-10 || 
        std::abs(phi - cached_bs_subspace_phi) > 1e-10 || 
        max_photon_number > cached_bs_subspace_max_k) {
        prepare_bs_subspace_matrices(theta, phi, max_photon_number);
    }

    int max_sub_dim = max_photon_number + 1;
    int max_padded_stride = max_sub_dim + (max_sub_dim % 2 == 0 ? 1 : 0);
    size_t shared_mem_size = max_sub_dim * max_padded_stride * sizeof(cuDoubleComplex) + 
                           2 * max_sub_dim * sizeof(cuDoubleComplex);

    dim3 block_dim(16, 16); 
    dim3 grid_dim(max_photon_number + 1, 1, batch_size);

    apply_two_mode_gate_kernel_cached<<<grid_dim, block_dim, shared_mem_size>>>(
        state_pool->data, state_pool->state_offsets, single_mode_cutoff, target_indices, batch_size, max_photon_number, d_bs_subspace_cache
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Beam Splitter kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Beam Splitter kernel synchronization failed: " + std::string(cudaGetErrorString(err)));
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
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double theta,
    int cutoff_a,
    int cutoff_b
) {
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int m = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int* state_dims,
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
    int m = blockIdx.x * blockDim.x + threadIdx.x;
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
                   int target_qumode1, int target_qumode2, int num_qumodes) {
    if (batch_size <= 0) {
        return;
    }

    if (cutoff_a != cutoff_b) {
        throw std::runtime_error("SUM gate currently requires identical mode cutoffs");
    }

    const int single_mode_cutoff = cutoff_a;
    const bool cache_valid = (d_sum_matrix_cache != nullptr &&
                              cached_sum_cutoff == single_mode_cutoff &&
                              std::abs(cached_sum_scale - scale) < 1e-10);

    if (!cache_valid) {
        if (d_sum_matrix_cache != nullptr) {
            cudaFree(d_sum_matrix_cache);
            d_sum_matrix_cache = nullptr;
        }

        const std::vector<cuDoubleComplex> h_sum_matrix = build_sum_tensor_dense(single_mode_cutoff, scale);
        cudaError_t err = cudaMalloc(&d_sum_matrix_cache,
                                     h_sum_matrix.size() * sizeof(cuDoubleComplex));
        if (err != cudaSuccess) {
            throw std::runtime_error("SUM gate matrix allocation failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMemcpy(d_sum_matrix_cache, h_sum_matrix.data(),
                         h_sum_matrix.size() * sizeof(cuDoubleComplex),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_sum_matrix_cache);
            d_sum_matrix_cache = nullptr;
            throw std::runtime_error("SUM gate matrix upload failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        cached_sum_cutoff = single_mode_cutoff;
        cached_sum_scale = scale;
    }

    apply_cached_two_mode_tensor_gate(
        state_pool,
        target_indices,
        batch_size,
        single_mode_cutoff,
        d_sum_matrix_cache,
        target_qumode1,
        target_qumode2,
        num_qumodes);
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
    const int* state_dims,
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
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
                                       int target_qumode1, int target_qumode2, int num_qumodes) {
    if (batch_size <= 0) {
        return;
    }

    const int single_mode_cutoff = infer_single_mode_cutoff(state_pool, num_qumodes);
    
    // 检查缓存是否有效
    const bool cache_valid = (d_tms_matrix_cache != nullptr &&
                              cached_tms_cutoff == single_mode_cutoff &&
                              std::abs(cached_tms_r - r) < 1e-10 &&
                              std::abs(cached_tms_theta - theta) < 1e-10);
    
    if (!cache_valid) {
        // 释放旧缓存
        if (d_tms_matrix_cache != nullptr) {
            cudaFree(d_tms_matrix_cache);
            d_tms_matrix_cache = nullptr;
        }
        
        // 在CPU上构建TMS矩阵
        const std::vector<cuDoubleComplex> h_tms_matrix =
            build_tms_tensor_dense(single_mode_cutoff, r, theta);
        
        // 分配GPU内存并上传
        const size_t matrix_size = h_tms_matrix.size() * sizeof(cuDoubleComplex);
        cudaError_t err = cudaMalloc(&d_tms_matrix_cache, matrix_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Two-Mode Squeezing matrix allocation failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMemcpy(d_tms_matrix_cache, h_tms_matrix.data(), matrix_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_tms_matrix_cache);
            d_tms_matrix_cache = nullptr;
            throw std::runtime_error("Two-Mode Squeezing matrix upload failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        
        // 更新缓存参数
        cached_tms_cutoff = single_mode_cutoff;
        cached_tms_r = r;
        cached_tms_theta = theta;
    }
    
    apply_cached_two_mode_tensor_gate(
        state_pool,
        target_indices,
        batch_size,
        single_mode_cutoff,
        d_tms_matrix_cache,
        target_qumode1,
        target_qumode2,
        num_qumodes);
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
                                   int target_qumode1, int target_qumode2, int num_qumodes) {
    if (batch_size <= 0) {
        return;
    }

    int single_mode_cutoff = infer_single_mode_cutoff(state_pool, num_qumodes);
    
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
    
    apply_cached_two_mode_tensor_gate(
        state_pool,
        target_indices,
        batch_size,
        single_mode_cutoff,
        d_bs_matrix_cache,
        target_qumode1,
        target_qumode2,
        num_qumodes);
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
    apply_beam_splitter_recursive(state_pool, target_indices, batch_size, M_PI / 4, 0.0, 0, 1, 2);
    // 中间的相位旋转需要单独实现
    apply_beam_splitter_recursive(state_pool, target_indices, batch_size, M_PI / 4, 0.0, 0, 1, 2);
}

/**
 * CZgate (受控相位) 内核
 * CZ(s) = exp(i s x̂₁ ⊗ x̂₂/ℏ)
 */
__global__ void apply_czgate_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double s,
    int cutoff_a,
    int cutoff_b
) {
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int m = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double kappa,
    int cutoff_a,
    int cutoff_b
) {
    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int m = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int* state_dims,
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
    const int current_dim = state_dims[state_idx];
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
