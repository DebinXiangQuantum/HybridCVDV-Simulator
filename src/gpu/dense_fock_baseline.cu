/**
 * Dense Fock Baseline GPU kernels — 消融实验用
 *
 * 提供 generic dense MatVec / tensor contraction 替代 L0-L3 专用 kernel,
 * 用于消融实验量化 Fock ELL 分级执行的性能增益。
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include <string>
#include "cv_state_pool.h"
#include "dense_fock_baseline.h"

namespace {

int compute_mode_right_stride_dense(int trunc_dim, int target_qumode, int num_qumodes) {
    int right_stride = 1;
    for (int mode = target_qumode + 1; mode < num_qumodes; ++mode) {
        right_stride *= trunc_dim;
    }
    return right_stride;
}

int infer_single_mode_cutoff_dense(const CVStatePool* state_pool, int num_qumodes) {
    const double inferred =
        std::pow(static_cast<double>(state_pool->max_total_dim),
                 1.0 / static_cast<double>(num_qumodes));
    return static_cast<int>(std::llround(inferred));
}

void check_cuda_dense(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

// --------------------------------------------------------------------------
// Single-mode dense D×D gemv kernel
// For a multi-mode state |n_0, ..., n_{M-1}> stored as flat array of
// size D^M, the target mode index maps to a stride in the flat layout.
// --------------------------------------------------------------------------

__global__ void dense_single_mode_gemv_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    const cuDoubleComplex* dense_matrix,   // D × D row-major
    int D,
    int mode_right_stride,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    const int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    const int state_idx = target_indices[batch_id];
    const size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t offset = state_offsets[state_idx];
    const int64_t state_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(state_dim)) return;

    const cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    // Decompose flat_index to find the row index in the target mode dimension
    const size_t rs = static_cast<size_t>(mode_right_stride);
    const int row = static_cast<int>((flat_index / rs) % D);
    const size_t base = flat_index - static_cast<size_t>(row) * rs;

    // Dense matrix-vector multiply along the target mode
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int col = 0; col < D; ++col) {
        cuDoubleComplex mat_val = dense_matrix[row * D + col];
        cuDoubleComplex psi_val = psi_in[base + static_cast<size_t>(col) * rs];
        sum = cuCadd(sum, cuCmul(mat_val, psi_val));
    }
    psi_out[flat_index] = sum;
}

__global__ void dense_copy_back_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    const cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    const int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    const int state_idx = target_indices[batch_id];
    const size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t state_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(state_dim)) return;

    state_data[state_offsets[state_idx] + flat_index] =
        temp_buffer[batch_id * buffer_stride + flat_index];
}

void apply_dense_single_mode_gate_gpu(
    CVStatePool* state_pool,
    const int* d_target_indices,
    int batch_size,
    const cuDoubleComplex* d_dense_matrix,
    int matrix_dim,
    int target_qumode,
    int num_qumodes,
    cudaStream_t stream,
    bool synchronize
) {
    if (batch_size <= 0) return;

    const int cutoff = infer_single_mode_cutoff_dense(state_pool, num_qumodes);
    const int mode_right_stride = compute_mode_right_stride_dense(cutoff, target_qumode, num_qumodes);

    const size_t buffer_stride = static_cast<size_t>(state_pool->max_total_dim);
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(
            static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex)));

    dim3 block_dim(256);
    const unsigned int grid_x =
        static_cast<unsigned int>((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x);
    dim3 grid_dim(grid_x, static_cast<unsigned int>(batch_size));

    dense_single_mode_gemv_kernel<<<grid_dim, block_dim, 0, stream>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_target_indices,
        batch_size,
        d_dense_matrix,
        matrix_dim,
        mode_right_stride,
        temp_buffer,
        buffer_stride);
    check_cuda_dense(cudaGetLastError(), "dense single-mode gemv launch failed");

    dense_copy_back_kernel<<<grid_dim, block_dim, 0, stream>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_target_indices,
        batch_size,
        temp_buffer,
        buffer_stride);
    check_cuda_dense(cudaGetLastError(), "dense copy-back launch failed");

    if (synchronize) {
        const cudaError_t err =
            stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        check_cuda_dense(err, "dense single-mode sync failed");
    }
}

// --------------------------------------------------------------------------
// Two-mode dense D²×D² tensor contraction kernel
// T[m1, m2, n1, n2] applied to state(n1, n2) → state(m1, m2)
// --------------------------------------------------------------------------

__global__ void dense_two_mode_tensor_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_indices,
    int batch_size,
    const cuDoubleComplex* tensor,   // D^4 elements: T[m1*D³ + m2*D² + n1*D + n2]
    int D,
    int stride1,
    int stride2,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    const int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    const int state_idx = target_indices[batch_id];
    const size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t offset = state_offsets[state_idx];
    const int64_t state_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(state_dim)) return;

    const cuDoubleComplex* psi_in = &state_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_id * buffer_stride];

    // Decompose flat_index to extract mode1/mode2 indices
    const size_t s1 = static_cast<size_t>(stride1);
    const size_t s2 = static_cast<size_t>(stride2);
    const int m1 = static_cast<int>((flat_index / s1) % D);
    const int m2 = static_cast<int>((flat_index / s2) % D);
    const size_t base = flat_index
        - static_cast<size_t>(m1) * s1
        - static_cast<size_t>(m2) * s2;

    // Full D² contraction over input modes
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    const int D2 = D * D;
    const int D3 = D2 * D;
    for (int n1 = 0; n1 < D; ++n1) {
        for (int n2 = 0; n2 < D; ++n2) {
            int t_idx = m1 * D3 + m2 * D2 + n1 * D + n2;
            cuDoubleComplex t_val = tensor[t_idx];
            size_t in_idx = base + static_cast<size_t>(n1) * s1 + static_cast<size_t>(n2) * s2;
            if (in_idx < static_cast<size_t>(state_dim)) {
                sum = cuCadd(sum, cuCmul(t_val, psi_in[in_idx]));
            }
        }
    }
    psi_out[flat_index] = sum;
}

void apply_dense_two_mode_gate_gpu(
    CVStatePool* state_pool,
    const int* d_target_indices,
    int batch_size,
    const cuDoubleComplex* d_tensor,
    int cutoff,
    int target_qumode1,
    int target_qumode2,
    int num_qumodes,
    cudaStream_t stream,
    bool synchronize
) {
    if (batch_size <= 0) return;

    const int stride1 = compute_mode_right_stride_dense(cutoff, target_qumode1, num_qumodes);
    const int stride2 = compute_mode_right_stride_dense(cutoff, target_qumode2, num_qumodes);

    const size_t buffer_stride = static_cast<size_t>(state_pool->max_total_dim);
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(
            static_cast<size_t>(batch_size) * buffer_stride * sizeof(cuDoubleComplex)));

    dim3 block_dim(256);
    const unsigned int grid_x =
        static_cast<unsigned int>((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x);
    dim3 grid_dim(grid_x, static_cast<unsigned int>(batch_size), 1);

    dense_two_mode_tensor_kernel<<<grid_dim, block_dim, 0, stream>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_target_indices,
        batch_size,
        d_tensor,
        cutoff,
        stride1,
        stride2,
        temp_buffer,
        buffer_stride);
    check_cuda_dense(cudaGetLastError(), "dense two-mode tensor launch failed");

    dense_copy_back_kernel<<<grid_dim, block_dim, 0, stream>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_target_indices,
        batch_size,
        temp_buffer,
        buffer_stride);
    check_cuda_dense(cudaGetLastError(), "dense two-mode copy-back launch failed");

    if (synchronize) {
        const cudaError_t err =
            stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        check_cuda_dense(err, "dense two-mode sync failed");
    }
}
