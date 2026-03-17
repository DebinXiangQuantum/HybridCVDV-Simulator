#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <stdexcept>
#include <string>
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
 * Level 1: 梯算符门 (Ladder/Shift Gates) GPU内核
 *
 * 特性：矩阵仅有一条非零对角线（次对角线）。无需存储矩阵，系数实时计算。
 * 典型门：Photon Creation a†, Annihilation a
 *
 * 数学公式：
 *   - Creation: ψ_out[n] = √n · ψ_in[n-1]
 *   - Annihilation: ψ_out[n] = √(n+1) · ψ_in[n+1]
 */

/**
 * 光子创建算符 a† 内核
 * ψ_out[n] = √n · ψ_in[n-1] (n >= 1)
 * ψ_out[0] = 0
 * 支持动态维度
 */
__global__ void apply_creation_operator_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride,
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
    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const int n = static_cast<int>((flat_index / right_stride) % target_mode_dim);
    const size_t base_index = flat_index - static_cast<size_t>(n) * right_stride;

    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

    if (n > 0) {
        // ψ_out[n] = √n · ψ_in[n-1]
        double coeff = sqrt(static_cast<double>(n));
        cuDoubleComplex input_val = psi_in[base_index + static_cast<size_t>(n - 1) * right_stride];
        result = make_cuDoubleComplex(
            coeff * cuCreal(input_val),
            coeff * cuCimag(input_val)
        );
    }
    // n == 0 时，结果为0

    psi_out[flat_index] = result;
}

/**
 * 光子湮灭算符 a 内核
 * ψ_out[n] = √(n+1) · ψ_in[n+1] (n < D-1)
 * ψ_out[D-1] = 0
 * 支持动态维度
 */
__global__ void apply_annihilation_operator_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride,
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
    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const int n = static_cast<int>((flat_index / right_stride) % target_mode_dim);
    const size_t base_index = flat_index - static_cast<size_t>(n) * right_stride;

    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

    if (n < target_mode_dim - 1) {
        // ψ_out[n] = √(n+1) · ψ_in[n+1]
        double coeff = sqrt(static_cast<double>(n + 1));
        cuDoubleComplex input_val = psi_in[base_index + static_cast<size_t>(n + 1) * right_stride];
        result = make_cuDoubleComplex(
            coeff * cuCreal(input_val),
            coeff * cuCimag(input_val)
        );
    }
    // n == D-1 时，结果为0

    psi_out[flat_index] = result;
}

__global__ void copy_back_ladder_kernel(
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

    size_t offset = state_offsets[state_idx];
    state_data[offset + flat_index] = temp_buffer[batch_id * buffer_stride + flat_index];
}

/**
 * 主机端接口：应用光子创建算符 a†
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_creation_operator_on_mode(CVStatePool* state_pool, const int* target_indices, int batch_size,
                                     int target_qumode, int num_qumodes) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);
    const size_t buffer_stride = state_pool->max_total_dim;
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(
            static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex)));

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_creation_operator_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        temp_buffer, buffer_stride, state_pool->d_trunc, target_mode_right_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Creation operator kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Creation operator kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    copy_back_ladder_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        temp_buffer, buffer_stride
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Creation operator write-back failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Creation operator write-back synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void apply_creation_operator(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    apply_creation_operator_on_mode(state_pool, target_indices, batch_size, 0, 1);
}

/**
 * 主机端接口：应用光子湮灭算符 a
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_annihilation_operator_on_mode(CVStatePool* state_pool, const int* target_indices, int batch_size,
                                         int target_qumode, int num_qumodes) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);
    const size_t buffer_stride = state_pool->max_total_dim;
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(
            static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex)));

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_annihilation_operator_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        temp_buffer, buffer_stride, state_pool->d_trunc, target_mode_right_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Annihilation operator kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Annihilation operator kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    copy_back_ladder_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        temp_buffer, buffer_stride
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Annihilation operator write-back failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Annihilation operator write-back synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void apply_annihilation_operator(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    apply_annihilation_operator_on_mode(state_pool, target_indices, batch_size, 0, 1);
}

/**
 * 数算符内核 n = a† a
 * ψ_out[n] = n · ψ_in[n]
 * 支持动态维度
 */
__global__ void apply_number_operator_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // 获取该状态的维度
    int current_dim = state_dims[state_idx];
    if (n >= current_dim) return;

    // 获取状态向量指针 (使用偏移量)
    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // 数算符：n|n⟩ = n|n⟩
    // ψ_out[n] = n · ψ_in[n]
    double photon_number = static_cast<double>(n);
    cuDoubleComplex current_val = psi[n];
    psi[n] = make_cuDoubleComplex(
        photon_number * cuCreal(current_val),
        photon_number * cuCimag(current_val)
    );
}

/**
 * 主机端接口：应用数算符 n
 * @param target_indices 设备端指针，指向目标状态ID数组
 */
void apply_number_operator(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_number_operator_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Number operator kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Number operator kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
