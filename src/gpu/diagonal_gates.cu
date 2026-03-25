#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "cv_state_pool.h"

namespace {

struct VacuumClassificationScratch {
    int is_zero;
    int is_scaled_vacuum;
    cuDoubleComplex scale;
};

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

void check_kernel_launch(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(message) + cudaGetErrorString(err));
    }
}

void synchronize_if_requested(cudaStream_t stream, bool synchronize, const char* message) {
    if (!synchronize) {
        return;
    }
    const cudaError_t err =
        stream != nullptr ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
    check_kernel_launch(err, message);
}

}  // namespace

/**
 * Level 0: 对角门 (Diagonal Gates) GPU内核
 *
 * 特性：对角门在Fock基底是对角阵，不涉及矩阵乘法，仅做Element-wise相乘
 * 典型门：Phase Rotation R(θ), Kerr K(χ), Conditional Parity CP
 *
 * 数学公式：ψ_out[n] = ψ_in[n] · e^(-i · f(n))
 *   - R(θ): f(n) = θ · n
 *   - Kerr: f(n) = χ · n²
 */

/**
 * 相位旋转门内核 R(θ) = exp(-i θ n)
 * 支持动态维度
 */
__global__ void apply_phase_rotation_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double theta,
    int target_mode_dim,
    int target_mode_right_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 获取该状态的维度
    int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    // 获取状态向量指针 (使用偏移量)
    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const int target_n = static_cast<int>((flat_index / right_stride) % target_mode_dim);

    // 计算相位因子: exp(-i * theta * n)  (与参考实现一致，使用负号)
    double phase = -theta * static_cast<double>(target_n);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));

    // 应用相位旋转
    cuDoubleComplex current_val = psi[flat_index];
    psi[flat_index] = cuCmul(current_val, phase_factor);
}

/**
 * Kerr门内核 K(χ) = exp(-i χ n²)
 * 支持动态维度
 */
__global__ void apply_kerr_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double chi,
    int target_mode_dim,
    int target_mode_right_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 获取该状态的维度
    int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    // 获取状态向量指针 (使用偏移量)
    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const int target_n = static_cast<int>((flat_index / right_stride) % target_mode_dim);

    // f(n, chi) = -chi * n * n
    // Kerr gate: exp(-i * chi * n^2)  (与参考实现一致，使用负号)
    double phase = -chi * static_cast<double>(target_n * target_n);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase)); // e^(i * phase)

    cuDoubleComplex current_val = psi[flat_index];
    psi[flat_index] = cuCmul(current_val, phase_factor);
}

/**
 * 条件奇偶校验门内核 CP
 * f(n, parity) = parity · π · (n % 2)
 * 支持动态维度
 */
__global__ void apply_conditional_parity_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    double parity,
    int target_mode_dim,
    int target_mode_right_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 获取该状态的维度
    int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    // 获取状态向量指针 (使用偏移量)
    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const int target_n = static_cast<int>((flat_index / right_stride) % target_mode_dim);

    // f(n, parity) = -parity * pi * (n % 2)
    double phase = -parity * M_PI * static_cast<double>(target_n % 2);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase)); // e^(i * phase)

    cuDoubleComplex current_val = psi[flat_index];
    psi[flat_index] = cuCmul(current_val, phase_factor);
}

/**
 * 主机端接口：应用相位旋转门 R(θ)
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_phase_rotation_on_mode(CVStatePool* state_pool, const int* target_indices,
                                  int batch_size, double theta,
                                  int target_qumode, int num_qumodes,
                                  cudaStream_t stream, bool synchronize) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_phase_rotation_kernel<<<grid_dim, block_dim, 0, stream>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, theta
        , state_pool->d_trunc, target_mode_right_stride
    );

    check_kernel_launch(cudaGetLastError(), "Phase Rotation kernel launch failed: ");
    if (synchronize) {
        check_kernel_launch(
            cudaStreamSynchronize(stream),
            "Phase Rotation kernel synchronization failed: ");
    }
}

void apply_phase_rotation(CVStatePool* state_pool, const int* target_indices,
                         int batch_size, double theta) {
    apply_phase_rotation_on_mode(
        state_pool, target_indices, batch_size, theta, 0, 1, nullptr, true);
}

/**
 * 主机端接口：应用Kerr门 K(χ)
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_kerr_gate_on_mode(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, double chi,
                             int target_qumode, int num_qumodes,
                             cudaStream_t stream, bool synchronize) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_kerr_kernel<<<grid_dim, block_dim, 0, stream>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, chi
        , state_pool->d_trunc, target_mode_right_stride
    );

    check_kernel_launch(cudaGetLastError(), "Kerr gate kernel launch failed: ");
    if (synchronize) {
        check_kernel_launch(
            cudaStreamSynchronize(stream),
            "Kerr gate kernel synchronization failed: ");
    }
}

void apply_kerr_gate(CVStatePool* state_pool, const int* target_indices,
                    int batch_size, double chi) {
    apply_kerr_gate_on_mode(
        state_pool, target_indices, batch_size, chi, 0, 1, nullptr, true);
}

/**
 * 主机端接口：应用条件奇偶校验门 CP
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_conditional_parity_on_mode(CVStatePool* state_pool, const int* target_indices,
                                      int batch_size, double parity,
                                      int target_qumode, int num_qumodes,
                                      cudaStream_t stream, bool synchronize) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_conditional_parity_kernel<<<grid_dim, block_dim, 0, stream>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, parity
        , state_pool->d_trunc, target_mode_right_stride
    );

    check_kernel_launch(cudaGetLastError(), "Conditional Parity kernel launch failed: ");
    if (synchronize) {
        check_kernel_launch(
            cudaStreamSynchronize(stream),
            "Conditional Parity kernel synchronization failed: ");
    }
}

void apply_conditional_parity(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, double parity) {
    apply_conditional_parity_on_mode(
        state_pool, target_indices, batch_size, parity, 0, 1, nullptr, true);
}

/**
 * 状态加法内核：result = w1 * state1 + w2 * state2
 */
__global__ void add_states_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* src1_indices,
    const cuDoubleComplex* weights1,
    const int* src2_indices,
    const cuDoubleComplex* weights2,
    const int* dst_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int src1_idx = src1_indices[batch_id];
    int src2_idx = src2_indices[batch_id];
    int dst_idx = dst_indices[batch_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    const int64_t current_dim = state_dims[src1_idx];
    const int64_t src2_dim = state_dims[src2_idx];
    const int64_t dst_dim = state_dims[dst_idx];
    if (current_dim != src2_dim || current_dim != dst_dim || n >= current_dim) return;

    cuDoubleComplex w1 = weights1[batch_id];
    cuDoubleComplex w2 = weights2[batch_id];

    cuDoubleComplex* state1 = &all_states_data[state_offsets[src1_idx]];
    cuDoubleComplex* state2 = &all_states_data[state_offsets[src2_idx]];
    cuDoubleComplex* result = &all_states_data[state_offsets[dst_idx]];

    // result[n] = w1 * state1[n] + w2 * state2[n]
    cuDoubleComplex val1 = cuCmul(w1, state1[n]);
    cuDoubleComplex val2 = cuCmul(w2, state2[n]);
    result[n] = cuCadd(val1, val2);
}

__global__ void combine_states_kernel(const cuDoubleComplex* src1,
                                      cuDoubleComplex weight1,
                                      const cuDoubleComplex* src2,
                                      cuDoubleComplex weight2,
                                      cuDoubleComplex* dst,
                                      int64_t state_dim) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }

    dst[idx] = cuCadd(cuCmul(weight1, src1[idx]), cuCmul(weight2, src2[idx]));
}

/**
 * 状态加法函数：result = w1 * state1 + w2 * state2
 * @param state_pool 状态池
 * @param src1_indices 源状态1的ID数组（设备指针）
 * @param weights1 权重1数组（设备指针）
 * @param src2_indices 源状态2的ID数组（设备指针）
 * @param weights2 权重2数组（设备指针）
 * @param dst_indices 目标状态ID数组（设备指针，调用方需预先保留与源状态匹配的存储）
 * @param batch_size 批次大小
 */
void add_states(CVStatePool* state_pool,
                const int* src1_indices,
                const cuDoubleComplex* weights1,
                const int* src2_indices,
                const cuDoubleComplex* weights2,
                const int* dst_indices,
                int batch_size) {
    if (batch_size <= 0) {
        return;
    }

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    add_states_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        src1_indices, weights1,
        src2_indices, weights2,
        dst_indices, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Add states kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void combine_states_device(CVStatePool* state_pool,
                           int src1_state_id,
                           cuDoubleComplex weight1,
                           int src2_state_id,
                           cuDoubleComplex weight2,
                           int dst_state_id,
                           cudaStream_t stream,
                           bool synchronize) {
    const int src1_dim = state_pool->get_state_dim(src1_state_id);
    const int src2_dim = state_pool->get_state_dim(src2_state_id);
    if (src1_dim != src2_dim) {
        throw std::runtime_error("Combine states requires matching source dimensions");
    }
    if (src1_dim <= 0) {
        return;
    }

    state_pool->reserve_state_storage(dst_state_id, src1_dim);

    const cuDoubleComplex* src1_ptr = state_pool->get_state_ptr(src1_state_id);
    const cuDoubleComplex* src2_ptr = state_pool->get_state_ptr(src2_state_id);
    cuDoubleComplex* dst_ptr = state_pool->get_state_ptr(dst_state_id);
    if (!src1_ptr || !src2_ptr || !dst_ptr) {
        throw std::runtime_error("Combine states requires valid GPU state pointers");
    }

    dim3 block_dim(256);
    dim3 grid_dim((src1_dim + block_dim.x - 1) / block_dim.x);
    combine_states_kernel<<<grid_dim, block_dim, 0, stream>>>(
        src1_ptr, weight1, src2_ptr, weight2, dst_ptr, src1_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Combine states kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    synchronize_if_requested(stream, synchronize, "Combine states kernel synchronization failed: ");
}

__global__ void zero_state_vector_kernel(cuDoubleComplex* state, int64_t state_dim) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    state[idx] = make_cuDoubleComplex(0.0, 0.0);
}

__global__ void initialize_vacuum_state_vector_kernel(cuDoubleComplex* state, int64_t state_dim) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    state[idx] = idx == 0 ? make_cuDoubleComplex(1.0, 0.0) : make_cuDoubleComplex(0.0, 0.0);
}

__global__ void copy_state_vector_kernel(const cuDoubleComplex* src,
                                         cuDoubleComplex* dst,
                                         int64_t state_dim) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    dst[idx] = src[idx];
}

__global__ void copy_scale_state_vector_kernel(const cuDoubleComplex* src,
                                               cuDoubleComplex* dst,
                                               cuDoubleComplex weight,
                                               int64_t state_dim) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    dst[idx] = cuCmul(weight, src[idx]);
}

__global__ void scale_state_vector_kernel(cuDoubleComplex* state,
                                          cuDoubleComplex weight,
                                          int64_t state_dim) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    state[idx] = cuCmul(weight, state[idx]);
}

__global__ void axpy_state_vector_kernel(const cuDoubleComplex* src,
                                         cuDoubleComplex* dst,
                                         cuDoubleComplex weight,
                                         int64_t state_dim) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    dst[idx] = cuCadd(dst[idx], cuCmul(weight, src[idx]));
}

__global__ void inspect_scaled_vacuum_kernel(const cuDoubleComplex* state,
                                             int64_t state_dim,
                                             double tolerance,
                                             int* is_zero,
                                             int* is_scaled_vacuum,
                                             cuDoubleComplex* scale) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }

    const cuDoubleComplex value = state[idx];
    const double magnitude = hypot(cuCreal(value), cuCimag(value));
    if (idx == 0) {
        *scale = value;
        if (magnitude > tolerance) {
            atomicExch(is_zero, 0);
        }
        return;
    }

    if (magnitude > tolerance) {
        atomicExch(is_zero, 0);
        atomicExch(is_scaled_vacuum, 0);
    }
}

void zero_state_device(CVStatePool* state_pool, int state_id,
                       cudaStream_t stream, bool synchronize) {
    cuDoubleComplex* state_ptr = state_pool->get_state_ptr(state_id);
    const int64_t state_dim = state_pool->get_state_dim(state_id);
    if (!state_ptr || state_dim <= 0) {
        return;
    }

    // Use cudaMemsetAsync instead of a custom kernel — avoids kernel launch overhead
    cudaError_t err = cudaMemsetAsync(state_ptr, 0,
                                      static_cast<size_t>(state_dim) * sizeof(cuDoubleComplex),
                                      stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Zero state memset failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    synchronize_if_requested(stream, synchronize, "Zero state synchronization failed: ");
}

void initialize_vacuum_state_device(CVStatePool* state_pool, int state_id, int64_t state_dim,
                                    cudaStream_t stream, bool synchronize) {
    state_pool->reserve_state_storage(state_id, state_dim);
    cuDoubleComplex* state_ptr = state_pool->get_state_ptr(state_id);
    if (!state_ptr || state_dim <= 0) {
        return;
    }

    // Use cudaMemsetAsync to zero all elements, then set element [0] = (1,0)
    cudaError_t err = cudaMemsetAsync(state_ptr, 0,
                                      static_cast<size_t>(state_dim) * sizeof(cuDoubleComplex),
                                      stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Initialize vacuum memset failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Set state[0] = 1.0 + 0.0i via async device-to-device copy from a host value
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    err = cudaMemcpyAsync(state_ptr, &one, sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Initialize vacuum element[0] copy failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    synchronize_if_requested(stream, synchronize,
                             "Initialize vacuum synchronization failed: ");
}

void copy_state_device(CVStatePool* state_pool, int src_state_id, int dst_state_id,
                       cudaStream_t stream, bool synchronize) {
    const int64_t state_dim = state_pool->get_state_dim(src_state_id);
    state_pool->reserve_state_storage(dst_state_id, state_dim);

    const cuDoubleComplex* src_ptr = state_pool->get_state_ptr(src_state_id);
    cuDoubleComplex* dst_ptr = state_pool->get_state_ptr(dst_state_id);
    if (!src_ptr || !dst_ptr || state_dim <= 0) {
        return;
    }

    constexpr size_t kCopyChunkBytes = 1ULL << 30;  // kernel chunking avoids unreliable large D2D memcpy paths
    const size_t total_elements = static_cast<size_t>(state_dim);
    const size_t chunk_elements =
        std::max<size_t>(1, kCopyChunkBytes / sizeof(cuDoubleComplex));

    dim3 block_dim(256);
    size_t copied_elements = 0;
    while (copied_elements < total_elements) {
        const size_t elements_this_chunk =
            std::min(chunk_elements, total_elements - copied_elements);
        dim3 grid_dim((elements_this_chunk + block_dim.x - 1) / block_dim.x);
        copy_state_vector_kernel<<<grid_dim, block_dim, 0, stream>>>(
            src_ptr + copied_elements,
            dst_ptr + copied_elements,
            static_cast<int64_t>(elements_this_chunk));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Copy state kernel launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        copied_elements += elements_this_chunk;
    }

    synchronize_if_requested(stream, synchronize, "Copy state synchronization failed: ");
}

void copy_scale_state_device(CVStatePool* state_pool,
                             int src_state_id,
                             int dst_state_id,
                             cuDoubleComplex weight,
                             cudaStream_t stream,
                             bool synchronize) {
    const int64_t state_dim = state_pool->get_state_dim(src_state_id);
    state_pool->reserve_state_storage(dst_state_id, state_dim);

    const cuDoubleComplex* src_ptr = state_pool->get_state_ptr(src_state_id);
    cuDoubleComplex* dst_ptr = state_pool->get_state_ptr(dst_state_id);
    if (!src_ptr || !dst_ptr || state_dim <= 0) {
        return;
    }

    constexpr size_t kVectorChunkBytes = 1ULL << 30;  // keep large-state kernels below 1 GiB each
    const size_t total_elements = static_cast<size_t>(state_dim);
    const size_t chunk_elements =
        std::max<size_t>(1, kVectorChunkBytes / sizeof(cuDoubleComplex));

    dim3 block_dim(256);
    for (size_t copied_elements = 0; copied_elements < total_elements;) {
        const size_t elements_this_chunk =
            std::min(chunk_elements, total_elements - copied_elements);
        dim3 grid_dim((elements_this_chunk + block_dim.x - 1) / block_dim.x);
        copy_scale_state_vector_kernel<<<grid_dim, block_dim, 0, stream>>>(
            src_ptr + copied_elements,
            dst_ptr + copied_elements,
            weight,
            static_cast<int64_t>(elements_this_chunk));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Copy-scale state kernel launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        copied_elements += elements_this_chunk;
    }

    synchronize_if_requested(stream, synchronize, "Copy-scale state kernel synchronization failed: ");
}

void scale_state_device(CVStatePool* state_pool, int state_id, cuDoubleComplex weight,
                        cudaStream_t stream, bool synchronize) {
    cuDoubleComplex* state_ptr = state_pool->get_state_ptr(state_id);
    const int64_t state_dim = state_pool->get_state_dim(state_id);
    if (!state_ptr || state_dim <= 0) {
        return;
    }

    constexpr size_t kVectorChunkBytes = 1ULL << 30;
    const size_t total_elements = static_cast<size_t>(state_dim);
    const size_t chunk_elements =
        std::max<size_t>(1, kVectorChunkBytes / sizeof(cuDoubleComplex));

    dim3 block_dim(256);
    for (size_t scaled_elements = 0; scaled_elements < total_elements;) {
        const size_t elements_this_chunk =
            std::min(chunk_elements, total_elements - scaled_elements);
        dim3 grid_dim((elements_this_chunk + block_dim.x - 1) / block_dim.x);
        scale_state_vector_kernel<<<grid_dim, block_dim, 0, stream>>>(
            state_ptr + scaled_elements,
            weight,
            static_cast<int64_t>(elements_this_chunk));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Scale state kernel launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        scaled_elements += elements_this_chunk;
    }

    synchronize_if_requested(stream, synchronize, "Scale state kernel synchronization failed: ");
}

void axpy_state_device(CVStatePool* state_pool,
                       int src_state_id,
                       int dst_state_id,
                       cuDoubleComplex weight,
                       cudaStream_t stream,
                       bool synchronize) {
    const int64_t state_dim = state_pool->get_state_dim(src_state_id);
    const int64_t dst_dim = state_pool->get_state_dim(dst_state_id);
    if (state_dim != dst_dim) {
        throw std::invalid_argument("AXPY requires matching state dimensions");
    }
    if (state_dim <= 0) {
        return;
    }

    const cuDoubleComplex* src_ptr = state_pool->get_state_ptr(src_state_id);
    cuDoubleComplex* dst_ptr = state_pool->get_state_ptr(dst_state_id);
    if (!src_ptr || !dst_ptr) {
        throw std::runtime_error("AXPY requires valid GPU state pointers");
    }

    constexpr size_t kVectorChunkBytes = 1ULL << 30;
    const size_t total_elements = static_cast<size_t>(state_dim);
    const size_t chunk_elements =
        std::max<size_t>(1, kVectorChunkBytes / sizeof(cuDoubleComplex));

    dim3 block_dim(256);
    for (size_t processed_elements = 0; processed_elements < total_elements;) {
        const size_t elements_this_chunk =
            std::min(chunk_elements, total_elements - processed_elements);
        dim3 grid_dim((elements_this_chunk + block_dim.x - 1) / block_dim.x);
        axpy_state_vector_kernel<<<grid_dim, block_dim, 0, stream>>>(
            src_ptr + processed_elements,
            dst_ptr + processed_elements,
            weight,
            static_cast<int64_t>(elements_this_chunk));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("AXPY state kernel launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        processed_elements += elements_this_chunk;
    }

    synchronize_if_requested(stream, synchronize, "AXPY state kernel synchronization failed: ");
}

void classify_vacuum_ray_device(CVStatePool* state_pool,
                                int state_id,
                                double tolerance,
                                int* is_zero,
                                int* is_scaled_vacuum,
                                cuDoubleComplex* scale) {
    if (!is_zero || !is_scaled_vacuum || !scale) {
        throw std::invalid_argument("vacuum classification output pointers must not be null");
    }

    const cuDoubleComplex* state_ptr = state_pool->get_state_ptr(state_id);
    const int64_t state_dim = state_pool->get_state_dim(state_id);
    *is_zero = 0;
    *is_scaled_vacuum = 0;
    *scale = make_cuDoubleComplex(0.0, 0.0);

    if (!state_ptr || state_dim <= 0) {
        return;
    }

    auto* scratch = static_cast<VacuumClassificationScratch*>(
        state_pool->scratch_aux.ensure(sizeof(VacuumClassificationScratch)));
    auto* host_scratch = static_cast<VacuumClassificationScratch*>(
        state_pool->host_transfer_staging.ensure(sizeof(VacuumClassificationScratch)));
    host_scratch->is_zero = 1;
    host_scratch->is_scaled_vacuum = 1;
    host_scratch->scale = make_cuDoubleComplex(0.0, 0.0);
    cudaMemcpy(scratch, host_scratch, sizeof(VacuumClassificationScratch), cudaMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_dim((state_dim + block_dim.x - 1) / block_dim.x);
    inspect_scaled_vacuum_kernel<<<grid_dim, block_dim>>>(
        state_ptr, state_dim, tolerance,
        &scratch->is_zero, &scratch->is_scaled_vacuum, &scratch->scale);

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        throw std::runtime_error("Vacuum classification kernel failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    cudaMemcpy(host_scratch, scratch, sizeof(VacuumClassificationScratch), cudaMemcpyDeviceToHost);
    *is_zero = host_scratch->is_zero;
    *is_scaled_vacuum = host_scratch->is_scaled_vacuum;
    *scale = host_scratch->scale;
}

// =====================================================================
// Cross-mode fused diagonal kernel
// Applies PhaseRotation + Kerr + ConditionalParity across ALL target
// modes in a single pass, eliminating per-mode kernel launch overhead.
// =====================================================================

/**
 * Per-mode descriptor uploaded to device constant or global memory.
 * phase(n) = -(theta * n + chi * n^2 + parity * pi * (n%2))
 */
struct FusedDiagonalOp {
    int right_stride;    // D^(modes_to_the_right)
    double theta;        // PhaseRotation parameter
    double chi;          // Kerr parameter
    double parity;       // ConditionalParity parameter
};

__global__ void apply_fused_diagonal_kernel(
    cuDoubleComplex* __restrict__ state_data,
    const size_t* __restrict__ state_offsets,
    const int64_t* __restrict__ state_dims,
    const int* __restrict__ target_indices,
    int batch_size,
    const FusedDiagonalOp* __restrict__ ops,
    int num_ops,
    int trunc_dim
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // Accumulate total phase from all ops
    double total_phase = 0.0;
    for (int op_idx = 0; op_idx < num_ops; ++op_idx) {
        const FusedDiagonalOp& op = ops[op_idx];
        int n = static_cast<int>((flat_index / static_cast<size_t>(op.right_stride)) % trunc_dim);
        double dn = static_cast<double>(n);
        total_phase += -(op.theta * dn + op.chi * dn * dn +
                         op.parity * M_PI * static_cast<double>(n % 2));
    }

    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(total_phase), sin(total_phase));
    cuDoubleComplex current_val = psi[flat_index];
    psi[flat_index] = cuCmul(current_val, phase_factor);
}

/**
 * Host-side: apply fused diagonal operations across multiple modes in one kernel.
 *
 * @param ops_host  Vector of per-mode descriptors (theta/chi/parity per mode)
 * @param target_indices  Device pointer to target state IDs
 * @param batch_size  Number of states
 */
void apply_fused_diagonal_gates(
    CVStatePool* state_pool,
    const int* target_indices,
    int batch_size,
    const std::vector<FusedDiagonalOp>& ops_host,
    int num_qumodes,
    cudaStream_t stream,
    bool synchronize
) {
    if (ops_host.empty() || batch_size <= 0) return;

    // Upload ops to device via scratch_aux
    FusedDiagonalOp* d_ops = state_pool->upload_vector_to_buffer(
        ops_host, state_pool->scratch_aux);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_fused_diagonal_kernel<<<grid_dim, block_dim, 0, stream>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        d_ops, static_cast<int>(ops_host.size()),
        state_pool->d_trunc
    );

    check_kernel_launch(cudaGetLastError(), "Fused diagonal kernel launch failed: ");
    synchronize_if_requested(stream, synchronize, "Fused diagonal kernel sync failed: ");
}
