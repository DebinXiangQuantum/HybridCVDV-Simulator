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
// Two-Mode Mixing Gates: beam splitter, tensor gate application, and related utilities.
// Specialized two-mode gates (eSWAP, SUM, CZ, CK, etc.) are in two_mode_special_gates.cu.

__global__ void apply_bs_tensor_kernel(
    const cuDoubleComplex* input_state, cuDoubleComplex* output_state,
    const cuDoubleComplex* bs_matrix, int cutoff);

namespace {

using HostComplex = std::complex<double>;
using HostMatrix = std::vector<HostComplex>;

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

void synchronize_if_requested(cudaStream_t stream,
                              bool synchronize,
                              const char* error_prefix) {
    if (!synchronize) {
        return;
    }

    const cudaError_t err =
        stream != nullptr ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(error_prefix) + cudaGetErrorString(err));
    }
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

__global__ void apply_two_mode_tensor_gate_on_modes_kernel(
    const cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
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
    int64_t current_dim = state_dims[state_idx];
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

__global__ void apply_two_mode_tensor_gate_to_output_kernel(
    const cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    const size_t* output_offsets,
    int batch_size,
    const cuDoubleComplex* tensor_matrix,
    int single_mode_cutoff,
    int mode1_stride,
    int mode2_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    const int state_idx = target_indices[batch_id];
    const size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    const size_t input_offset = state_offsets[state_idx];
    const size_t output_offset = output_offsets[batch_id];
    const cuDoubleComplex* psi_in = &state_data[input_offset];
    cuDoubleComplex* psi_out = const_cast<cuDoubleComplex*>(&state_data[output_offset]);

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
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    const cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    const size_t offset = state_offsets[state_idx];
    state_data[offset + flat_index] = temp_buffer[batch_id * buffer_stride + flat_index];
}

}  // namespace

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

void apply_cached_two_mode_tensor_gate(CVStatePool* state_pool,
                                       const int* target_indices,
                                       int batch_size,
                                       int single_mode_cutoff,
                                       const cuDoubleComplex* tensor_matrix,
                                       int target_qumode1,
                                       int target_qumode2,
                                       int num_qumodes,
                                       cudaStream_t stream,
                                       bool synchronize) {
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
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    if (!synchronize) {
        const size_t buffer_stride = state_pool->max_total_dim;
        const size_t temp_bytes = static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex);
        cuDoubleComplex* temp_state = static_cast<cuDoubleComplex*>(
            state_pool->scratch_temp.ensure(temp_bytes));

        apply_two_mode_tensor_gate_on_modes_kernel<<<grid_dim, block_dim, 0, stream>>>(
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

        copy_back_two_mode_tensor_gate_kernel<<<grid_dim, block_dim, 0, stream>>>(
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
        return;
    }

    const std::vector<int> host_targets = copy_target_indices_to_host(target_indices, batch_size);
    std::vector<int> state_dims(batch_size, 0);
    std::vector<size_t> output_offsets(batch_size, 0);
    std::vector<bool> output_owned(batch_size, false);

    try {
        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            const int state_id = host_targets[batch_id];
            if (!state_pool->is_valid_state(state_id)) {
                throw std::invalid_argument("Two-mode tensor target state is invalid");
            }

            const int64_t state_dim = state_pool->get_state_dim(state_id);
            state_dims[batch_id] = state_dim;
            output_offsets[batch_id] =
                state_pool->allocate_detached_storage(static_cast<size_t>(state_dim));
            output_owned[batch_id] = true;
        }

        const size_t* d_output_offsets =
            state_pool->upload_values_to_buffer(output_offsets.data(),
                                                output_offsets.size(),
                                                state_pool->scratch_aux);

        apply_two_mode_tensor_gate_to_output_kernel<<<grid_dim, block_dim, 0, stream>>>(
            state_pool->data,
            state_pool->state_offsets,
            state_pool->state_dims,
            target_indices,
            d_output_offsets,
            batch_size,
            tensor_matrix,
            single_mode_cutoff,
            mode1_stride,
            mode2_stride
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Two-mode tensor direct-write launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        synchronize_if_requested(stream, true, "Two-mode tensor direct-write sync failed: ");

        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            state_pool->replace_state_storage(host_targets[batch_id],
                                              output_offsets[batch_id],
                                              static_cast<size_t>(state_dims[batch_id]),
                                              state_dims[batch_id]);
            output_owned[batch_id] = false;
        }
    } catch (...) {
        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            if (!output_owned[batch_id]) {
                continue;
            }
            state_pool->release_detached_storage(output_offsets[batch_id],
                                                static_cast<size_t>(state_dims[batch_id]));
        }
        throw;
    }
}

// Build BS matrix using Strawberry Fields recurrence relation.
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

// GPU Kernel: output[m,n] = sum_{p,q} Z[m,n,p,q] * input[p,q]
__global__ void apply_bs_tensor_kernel(
    const cuDoubleComplex* input_state,
    cuDoubleComplex* output_state,
    const cuDoubleComplex* bs_matrix,
    int cutoff) {
    
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
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
    int64_t d_trunc,
    const int* target_indices,
    int batch_size,
    int max_photon_number
) {
    extern __shared__ cuDoubleComplex shared_vec[];

    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    cuDoubleComplex* psi = &state_data[static_cast<size_t>(state_idx) * d_trunc];

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

