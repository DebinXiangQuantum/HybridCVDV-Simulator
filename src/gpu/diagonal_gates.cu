#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "cv_state_pool.h"

namespace {

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
    const int* state_dims,
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
    int current_dim = state_dims[state_idx];
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
    const int* state_dims,
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
    int current_dim = state_dims[state_idx];
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
    const int* state_dims,
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
    int current_dim = state_dims[state_idx];
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
                                  int target_qumode, int num_qumodes) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_phase_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, theta
        , state_pool->d_trunc, target_mode_right_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Phase Rotation kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Phase Rotation kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void apply_phase_rotation(CVStatePool* state_pool, const int* target_indices,
                         int batch_size, double theta) {
    apply_phase_rotation_on_mode(state_pool, target_indices, batch_size, theta, 0, 1);
}

/**
 * 主机端接口：应用Kerr门 K(χ)
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_kerr_gate_on_mode(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, double chi,
                             int target_qumode, int num_qumodes) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_kerr_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, chi
        , state_pool->d_trunc, target_mode_right_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kerr gate kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kerr gate kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void apply_kerr_gate(CVStatePool* state_pool, const int* target_indices,
                    int batch_size, double chi) {
    apply_kerr_gate_on_mode(state_pool, target_indices, batch_size, chi, 0, 1);
}

/**
 * 主机端接口：应用条件奇偶校验门 CP
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_conditional_parity_on_mode(CVStatePool* state_pool, const int* target_indices,
                                      int batch_size, double parity,
                                      int target_qumode, int num_qumodes) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_conditional_parity_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size, parity
        , state_pool->d_trunc, target_mode_right_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Conditional Parity kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Conditional Parity kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void apply_conditional_parity(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, double parity) {
    apply_conditional_parity_on_mode(state_pool, target_indices, batch_size, parity, 0, 1);
}

/**
 * 状态加法内核：result = w1 * state1 + w2 * state2
 */
__global__ void add_states_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int* state_dims,
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
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    const int current_dim = state_dims[src1_idx];
    if (n >= current_dim) return;

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

/**
 * 状态加法函数：result = w1 * state1 + w2 * state2
 * @param state_pool 状态池
 * @param src1_indices 源状态1的ID数组（设备指针）
 * @param weights1 权重1数组（设备指针）
 * @param src2_indices 源状态2的ID数组（设备指针）
 * @param weights2 权重2数组（设备指针）
 * @param dst_indices 目标状态ID数组（设备指针）
 * @param batch_size 批次大小
 */
void add_states(CVStatePool* state_pool,
                const int* src1_indices,
                const cuDoubleComplex* weights1,
                const int* src2_indices,
                const cuDoubleComplex* weights2,
                const int* dst_indices,
                int batch_size) {
    std::vector<int> host_src1(batch_size);
    std::vector<int> host_src2(batch_size);
    std::vector<int> host_dst(batch_size);

    cudaError_t err = cudaMemcpy(host_src1.data(), src1_indices,
                                 batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Add states读取源状态1失败: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaMemcpy(host_src2.data(), src2_indices,
                     batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Add states读取源状态2失败: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaMemcpy(host_dst.data(), dst_indices,
                     batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Add states读取目标状态失败: " +
                                 std::string(cudaGetErrorString(err)));
    }

    int max_dim = 0;
    for (int batch = 0; batch < batch_size; ++batch) {
        const int src1_dim = state_pool->get_state_dim(host_src1[batch]);
        const int src2_dim = state_pool->get_state_dim(host_src2[batch]);
        if (src1_dim != src2_dim) {
            throw std::runtime_error("Add states要求源状态维度一致");
        }
        state_pool->reserve_state_storage(host_dst[batch], src1_dim);
        max_dim = std::max(max_dim, src1_dim);
    }

    dim3 block_dim(256);
    dim3 grid_dim((max_dim + block_dim.x - 1) / block_dim.x, batch_size);

    add_states_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        src1_indices, weights1,
        src2_indices, weights2,
        dst_indices, batch_size
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Add states kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

__global__ void zero_state_vector_kernel(cuDoubleComplex* state, int state_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    state[idx] = make_cuDoubleComplex(0.0, 0.0);
}

__global__ void initialize_vacuum_state_vector_kernel(cuDoubleComplex* state, int state_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    state[idx] = idx == 0 ? make_cuDoubleComplex(1.0, 0.0) : make_cuDoubleComplex(0.0, 0.0);
}

__global__ void copy_state_vector_kernel(const cuDoubleComplex* src,
                                         cuDoubleComplex* dst,
                                         int state_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    dst[idx] = src[idx];
}

__global__ void axpy_state_vector_kernel(const cuDoubleComplex* src,
                                         cuDoubleComplex* dst,
                                         cuDoubleComplex weight,
                                         int state_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) {
        return;
    }
    dst[idx] = cuCadd(dst[idx], cuCmul(weight, src[idx]));
}

__global__ void inspect_scaled_vacuum_kernel(const cuDoubleComplex* state,
                                             int state_dim,
                                             double tolerance,
                                             int* is_zero,
                                             int* is_scaled_vacuum,
                                             cuDoubleComplex* scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

void zero_state_device(CVStatePool* state_pool, int state_id) {
    cuDoubleComplex* state_ptr = state_pool->get_state_ptr(state_id);
    const int state_dim = state_pool->get_state_dim(state_id);
    if (!state_ptr || state_dim <= 0) {
        return;
    }

    dim3 block_dim(256);
    dim3 grid_dim((state_dim + block_dim.x - 1) / block_dim.x);
    zero_state_vector_kernel<<<grid_dim, block_dim>>>(state_ptr, state_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Zero state kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Zero state kernel synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void initialize_vacuum_state_device(CVStatePool* state_pool, int state_id, int state_dim) {
    state_pool->reserve_state_storage(state_id, state_dim);
    cuDoubleComplex* state_ptr = state_pool->get_state_ptr(state_id);
    if (!state_ptr || state_dim <= 0) {
        return;
    }

    dim3 block_dim(256);
    dim3 grid_dim((state_dim + block_dim.x - 1) / block_dim.x);
    initialize_vacuum_state_vector_kernel<<<grid_dim, block_dim>>>(state_ptr, state_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Initialize vacuum kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Initialize vacuum kernel synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void copy_state_device(CVStatePool* state_pool, int src_state_id, int dst_state_id) {
    const int state_dim = state_pool->get_state_dim(src_state_id);
    state_pool->reserve_state_storage(dst_state_id, state_dim);

    const cuDoubleComplex* src_ptr = state_pool->get_state_ptr(src_state_id);
    cuDoubleComplex* dst_ptr = state_pool->get_state_ptr(dst_state_id);
    if (!src_ptr || !dst_ptr || state_dim <= 0) {
        return;
    }

    dim3 block_dim(256);
    dim3 grid_dim((state_dim + block_dim.x - 1) / block_dim.x);
    copy_state_vector_kernel<<<grid_dim, block_dim>>>(src_ptr, dst_ptr, state_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Copy state kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Copy state kernel synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void axpy_state_device(CVStatePool* state_pool,
                       int src_state_id,
                       int dst_state_id,
                       cuDoubleComplex weight) {
    const int state_dim = state_pool->get_state_dim(src_state_id);
    const int dst_dim = state_pool->get_state_dim(dst_state_id);
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

    dim3 block_dim(256);
    dim3 grid_dim((state_dim + block_dim.x - 1) / block_dim.x);
    axpy_state_vector_kernel<<<grid_dim, block_dim>>>(src_ptr, dst_ptr, weight, state_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("AXPY state kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("AXPY state kernel synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void classify_vacuum_ray_device(const CVStatePool* state_pool,
                                int state_id,
                                double tolerance,
                                int* is_zero,
                                int* is_scaled_vacuum,
                                cuDoubleComplex* scale) {
    if (!is_zero || !is_scaled_vacuum || !scale) {
        throw std::invalid_argument("vacuum classification output pointers must not be null");
    }

    const cuDoubleComplex* state_ptr = state_pool->get_state_ptr(state_id);
    const int state_dim = state_pool->get_state_dim(state_id);
    *is_zero = 0;
    *is_scaled_vacuum = 0;
    *scale = make_cuDoubleComplex(0.0, 0.0);

    if (!state_ptr || state_dim <= 0) {
        return;
    }

    int host_is_zero = 1;
    int host_is_scaled_vacuum = 1;
    int* d_is_zero = nullptr;
    int* d_is_scaled_vacuum = nullptr;
    cuDoubleComplex* d_scale = nullptr;

    cudaError_t err = cudaMalloc(&d_is_zero, sizeof(int));
    if (err != cudaSuccess) {
        throw std::runtime_error("Vacuum classification alloc failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_is_scaled_vacuum, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_is_zero);
        throw std::runtime_error("Vacuum classification alloc failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_scale, sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        cudaFree(d_is_zero);
        cudaFree(d_is_scaled_vacuum);
        throw std::runtime_error("Vacuum classification alloc failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    cudaMemcpy(d_is_zero, &host_is_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_scaled_vacuum, &host_is_scaled_vacuum, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_scale, 0, sizeof(cuDoubleComplex));

    dim3 block_dim(256);
    dim3 grid_dim((state_dim + block_dim.x - 1) / block_dim.x);
    inspect_scaled_vacuum_kernel<<<grid_dim, block_dim>>>(
        state_ptr, state_dim, tolerance, d_is_zero, d_is_scaled_vacuum, d_scale);

    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        cudaFree(d_is_zero);
        cudaFree(d_is_scaled_vacuum);
        cudaFree(d_scale);
        throw std::runtime_error("Vacuum classification kernel failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    cudaMemcpy(is_zero, d_is_zero, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(is_scaled_vacuum, d_is_scaled_vacuum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(scale, d_scale, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_is_zero);
    cudaFree(d_is_scaled_vacuum);
    cudaFree(d_scale);
}
