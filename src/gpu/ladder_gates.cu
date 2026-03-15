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
 * 通用梯算符门内核 (使用shuffle指令优化)
 * 支持创建和湮灭算符的warp级优化版本
 */
__global__ void apply_ladder_operator_warp_kernel(
    cuDoubleComplex* state_data,
    int d_trunc,
    const int* target_indices,
    int batch_size,
    bool is_creation  // true: 创建算符, false: 湮灭算符
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int tid = threadIdx.x;
    int n = blockIdx.x * blockDim.x + tid;

    if (n >= d_trunc) return;

    cuDoubleComplex* psi = &state_data[state_idx * d_trunc];

    if (is_creation) {
        // 创建算符: ψ[n] = √n · ψ[n-1]
        cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

        if (n > 0) {
            // 从前一个线程获取数据
            double coeff = sqrt(static_cast<double>(n));

            // 使用shuffle获取前一个元素
            float real_part, imag_part;
            if (tid > 0) {
                // 同warp内的线程可以直接shuffle
                real_part = __shfl_up_sync(0xFFFFFFFF, cuCreal(psi[n]), 1);
                imag_part = __shfl_up_sync(0xFFFFFFFF, cuCimag(psi[n]), 1);
            } else {
                // 跨warp边界，需要从全局内存读取
                if (n > 0) {
                    cuDoubleComplex prev_val = psi[n - 1];
                    real_part = cuCreal(prev_val);
                    imag_part = cuCimag(prev_val);
                } else {
                    real_part = 0.0f;
                    imag_part = 0.0f;
                }
            }

            result = make_cuDoubleComplex(coeff * real_part, coeff * imag_part);
        }

        psi[n] = result;
    } else {
        // 湮灭算符: ψ[n] = √(n+1) · ψ[n+1]
        cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

        if (n < d_trunc - 1) {
            // 从后一个线程获取数据
            double coeff = sqrt(static_cast<double>(n + 1));

            // 使用shuffle获取后一个元素
            float real_part, imag_part;
            if (tid < blockDim.x - 1) {
                // 同warp内的线程可以直接shuffle
                real_part = __shfl_down_sync(0xFFFFFFFF, cuCreal(psi[n]), 1);
                imag_part = __shfl_down_sync(0xFFFFFFFF, cuCimag(psi[n]), 1);
            } else {
                // 跨warp边界，需要从全局内存读取
                if (n < d_trunc - 1) {
                    cuDoubleComplex next_val = psi[n + 1];
                    real_part = cuCreal(next_val);
                    imag_part = cuCimag(next_val);
                } else {
                    real_part = 0.0f;
                    imag_part = 0.0f;
                }
            }

            result = make_cuDoubleComplex(coeff * real_part, coeff * imag_part);
        }

        psi[n] = result;
    }
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
    cuDoubleComplex* temp_buffer = nullptr;
    cudaError_t err = cudaMalloc(&temp_buffer, static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate creation buffer: " +
                                 std::string(cudaGetErrorString(err)));
    }

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_creation_operator_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        temp_buffer, buffer_stride, state_pool->d_trunc, target_mode_right_stride
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(temp_buffer);
        throw std::runtime_error("Creation operator kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(temp_buffer);
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
        cudaFree(temp_buffer);
        throw std::runtime_error("Creation operator write-back failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    cudaFree(temp_buffer);
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
    cuDoubleComplex* temp_buffer = nullptr;
    cudaError_t err = cudaMalloc(&temp_buffer, static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate annihilation buffer: " +
                                 std::string(cudaGetErrorString(err)));
    }

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_annihilation_operator_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        temp_buffer, buffer_stride, state_pool->d_trunc, target_mode_right_stride
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(temp_buffer);
        throw std::runtime_error("Annihilation operator kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 同步等待内核完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(temp_buffer);
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
        cudaFree(temp_buffer);
        throw std::runtime_error("Annihilation operator write-back failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    cudaFree(temp_buffer);
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
