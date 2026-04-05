/**
 * Squeezing Gate GPU Implementation
 * 主线实现保持矩阵生成与应用都在 GPU 侧，避免 CPU dense/ELL 构建和回传。
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cv_state_pool.h"
#include <cmath>
#include <stdexcept>
#include <string>

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

struct SqueezingCache {
    cuDoubleComplex* d_dense_matrix = nullptr;
    int device_id = -1;
    int cutoff = 0;
    double r = 0.0;
    double theta = 0.0;
    bool valid = false;
};

static SqueezingCache g_cache;

void check_cuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void release_squeezing_cache_storage() {
    if (g_cache.d_dense_matrix) {
        int current_device = 0;
        cudaError_t get_device_err = cudaGetDevice(&current_device);
        if (get_device_err != cudaSuccess) {
            current_device = 0;
            cudaGetLastError();
        }
        if (g_cache.device_id >= 0 && current_device != g_cache.device_id) {
            check_cuda(cudaSetDevice(g_cache.device_id), "failed to switch device for squeezing cache release");
        }
        check_cuda(cudaFree(g_cache.d_dense_matrix), "failed to release squeezing cache");
        g_cache.d_dense_matrix = nullptr;
        if (g_cache.device_id >= 0 && current_device != g_cache.device_id) {
            check_cuda(cudaSetDevice(current_device), "failed to restore device after squeezing cache release");
        }
    }
    g_cache.device_id = -1;
    g_cache.cutoff = 0;
    g_cache.r = 0.0;
    g_cache.theta = 0.0;
    g_cache.valid = false;
}

bool matches_cache(double r, double theta, int cutoff, int device_id) {
    return g_cache.valid &&
           g_cache.device_id == device_id &&
           g_cache.cutoff == cutoff &&
           g_cache.r == r &&
           g_cache.theta == theta;
}

__global__ void generate_squeezing_dense_kernel(cuDoubleComplex* dense_matrix,
                                                int cutoff,
                                                double r,
                                                double theta) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const int matrix_size = cutoff * cutoff;
    for (int idx = 0; idx < matrix_size; ++idx) {
        dense_matrix[idx] = make_cuDoubleComplex(0.0, 0.0);
    }

    const double tanh_r = tanh(r);
    const double sech_r = 1.0 / cosh(r);
    const cuDoubleComplex eitheta_tanhr =
        make_cuDoubleComplex(cos(theta) * tanh_r, sin(theta) * tanh_r);
    const cuDoubleComplex R00 = make_cuDoubleComplex(-cuCreal(eitheta_tanhr),
                                                     -cuCimag(eitheta_tanhr));
    const cuDoubleComplex R01 = make_cuDoubleComplex(sech_r, 0.0);
    const cuDoubleComplex R11 = cuConj(eitheta_tanhr);

    dense_matrix[0] = make_cuDoubleComplex(sqrt(sech_r), 0.0);

    for (int m = 2; m < cutoff; m += 2) {
        const double factor = sqrt(static_cast<double>(m - 1)) /
                              sqrt(static_cast<double>(m));
        dense_matrix[m * cutoff] = cuCmul(
            make_cuDoubleComplex(factor, 0.0),
            cuCmul(R00, dense_matrix[(m - 2) * cutoff]));
    }

    for (int m = 0; m < cutoff; ++m) {
        for (int n = 1; n < cutoff; ++n) {
            if (((m + n) & 1) != 0) {
                continue;
            }

            cuDoubleComplex value = make_cuDoubleComplex(0.0, 0.0);

            if (n >= 2) {
                const double factor = sqrt(static_cast<double>(n - 1)) /
                                      sqrt(static_cast<double>(n));
                value = cuCadd(
                    value,
                    cuCmul(make_cuDoubleComplex(factor, 0.0),
                           cuCmul(R11, dense_matrix[m * cutoff + (n - 2)])));
            }

            if (m >= 1) {
                const double factor = sqrt(static_cast<double>(m)) /
                                      sqrt(static_cast<double>(n));
                value = cuCadd(
                    value,
                    cuCmul(make_cuDoubleComplex(factor, 0.0),
                           cuCmul(R01, dense_matrix[(m - 1) * cutoff + (n - 1)])));
            }

            dense_matrix[m * cutoff + n] = value;
        }
    }
}

__global__ void apply_squeezing_dense_cached_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const cuDoubleComplex* dense_matrix,
    int cutoff,
    const int* target_indices,
    int batch_size,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride,
    int target_mode_right_stride
) {
    const int batch_id = blockIdx.y;
    if (batch_id >= batch_size) {
        return;
    }

    const int state_idx = target_indices[batch_id];
    const size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t offset = state_offsets[state_idx];
    const int64_t state_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(state_dim)) {
        return;
    }

    cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const int row = static_cast<int>((flat_index / right_stride) % cutoff);
    const size_t base_index = flat_index - static_cast<size_t>(row) * right_stride;

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int col = row & 1; col < cutoff; col += 2) {
        sum = cuCadd(
            sum,
            cuCmul(dense_matrix[row * cutoff + col],
                   psi_in[base_index + static_cast<size_t>(col) * right_stride]));
    }

    psi_out[flat_index] = sum;
}

__global__ void copy_result_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const cuDoubleComplex* temp_buffer,
    const int* target_indices,
    int batch_size,
    size_t buffer_stride
) {
    const int batch_id = blockIdx.y;
    if (batch_id >= batch_size) {
        return;
    }

    const int state_idx = target_indices[batch_id];
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t offset = state_offsets[state_idx];
    const int64_t state_dim = state_dims[state_idx];
    if (idx >= static_cast<size_t>(state_dim)) {
        return;
    }

    state_data[offset + idx] = temp_buffer[batch_id * buffer_stride + idx];
}

void generate_and_cache_squeezing_dense(double r,
                                        double theta,
                                        int cutoff,
                                        cudaStream_t stream) {
    int current_device = 0;
    cudaError_t get_device_err = cudaGetDevice(&current_device);
    if (get_device_err != cudaSuccess) {
        current_device = 0;
        cudaGetLastError();
    }

    if (matches_cache(r, theta, cutoff, current_device)) {
        return;
    }

    release_squeezing_cache_storage();

    const size_t matrix_bytes =
        static_cast<size_t>(cutoff) * static_cast<size_t>(cutoff) * sizeof(cuDoubleComplex);
    check_cuda(cudaMalloc(&g_cache.d_dense_matrix, matrix_bytes),
               "failed to allocate squeezing dense cache");

    generate_squeezing_dense_kernel<<<1, 1, 0, stream>>>(g_cache.d_dense_matrix, cutoff, r, theta);
    check_cuda(cudaPeekAtLastError(), "failed to launch squeezing cache generation kernel");

    g_cache.device_id = current_device;
    g_cache.cutoff = cutoff;
    g_cache.r = r;
    g_cache.theta = theta;
    g_cache.valid = true;
}

}  // namespace

void apply_squeezing_gate_gpu(
    CVStatePool* pool,
    const int* target_indices,
    int batch_size,
    double r,
    double theta,
    int target_qumode,
    int num_qumodes,
    cudaStream_t stream,
    bool synchronize
) {
    if (!pool) {
        throw std::invalid_argument("state pool pointer is null");
    }
    if (batch_size <= 0) {
        return;
    }

    const int cutoff = pool->d_trunc;
    const int target_mode_right_stride =
        compute_mode_right_stride(pool->d_trunc, target_qumode, num_qumodes);

    generate_and_cache_squeezing_dense(r, theta, cutoff, stream);

    const size_t buffer_stride = pool->max_total_dim;
    cuDoubleComplex* d_temp_buffer = static_cast<cuDoubleComplex*>(
        pool->scratch_temp.ensure(static_cast<size_t>(batch_size) * buffer_stride *
                                  sizeof(cuDoubleComplex)));

    const dim3 block_dim(256);
    const dim3 grid_dim(
        static_cast<unsigned int>((pool->max_total_dim + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>(batch_size));

    apply_squeezing_dense_cached_kernel<<<grid_dim, block_dim, 0, stream>>>(
        pool->data,
        pool->state_offsets,
        pool->state_dims,
        g_cache.d_dense_matrix,
        cutoff,
        target_indices,
        batch_size,
        d_temp_buffer,
        buffer_stride,
        target_mode_right_stride);
    check_cuda(cudaPeekAtLastError(), "failed to launch squeezing apply kernel");

    copy_result_kernel<<<grid_dim, block_dim, 0, stream>>>(
        pool->data,
        pool->state_offsets,
        pool->state_dims,
        d_temp_buffer,
        target_indices,
        batch_size,
        buffer_stride);
    check_cuda(cudaPeekAtLastError(), "failed to launch squeezing copy kernel");

    if (synchronize) {
        const cudaError_t err =
            stream != nullptr ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        check_cuda(err, "squeezing gate synchronization failed");
    }
}

void clear_squeezing_cache() {
    release_squeezing_cache_storage();
}
