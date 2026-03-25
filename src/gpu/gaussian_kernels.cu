#include <cuda_runtime.h>
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include <iostream>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

// Max phase-space dimension we support in shared memory (dim = 2*num_qumodes).
// For 32 qumodes this is 64, covering all practical cases.
// Shared memory per block: 64*64*8 = 32KB for S matrix.
constexpr int MAX_DIM_SHARED = 64;

// ============================================================================
// Optimized: Fused copy + displacement kernel
// Each block handles one batch element; dim threads parallelize over rows.
// S matrix loaded into shared memory once, reused by all threads.
// ============================================================================
__global__ void fused_copy_displacement_kernel(
    double** d_ptrs,
    double* d_old_buffer,
    int dim,
    const int* state_ids,
    int batch_size,
    const double* __restrict__ S,
    const double* __restrict__ dg
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int state_id = state_ids[batch_idx];
    double* d = d_ptrs[state_id];
    double* d_old = d_old_buffer + (static_cast<size_t>(batch_idx) * dim);

    if (!d) return;

    // Load S matrix into shared memory (cooperative load by all threads)
    extern __shared__ double smem[];
    double* s_S = smem;  // dim * dim doubles
    int total_S = dim * dim;
    for (int idx = tid; idx < total_S; idx += blockDim.x) {
        s_S[idx] = S[idx];
    }

    // Copy d -> d_old (each thread copies its assigned elements)
    for (int i = tid; i < dim; i += blockDim.x) {
        d_old[i] = d[i];
    }
    __syncthreads();

    // Compute d_new[i] = sum_j S[i][j] * d_old[j] + dg[i]
    for (int i = tid; i < dim; i += blockDim.x) {
        double sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += s_S[i * dim + j] * d_old[j];
        }
        d[i] = sum + dg[i];
    }
}

// ============================================================================
// Optimized: Fused covariance update kernel
// Computes sigma' = S * sigma * S^T in one kernel launch.
// Each block handles one (row, batch) pair; threads parallelize over columns.
// S is loaded into shared memory. Temp row stored in shared memory.
// ============================================================================
__global__ void fused_covariance_kernel(
    double** sigma_ptrs,
    double* temp_buffer,
    int dim,
    const int* state_ids,
    int batch_size,
    const double* __restrict__ S
) {
    int batch_idx = blockIdx.y;
    int row_i = blockIdx.x;
    if (batch_idx >= batch_size || row_i >= dim) return;

    int tid = threadIdx.x;
    int state_id = state_ids[batch_idx];
    double* sig = sigma_ptrs[state_id];
    double* temp_row = temp_buffer + (static_cast<size_t>(batch_idx) * dim * dim)
                       + static_cast<size_t>(row_i) * dim;

    if (!sig) return;

    // Load S matrix into shared memory
    extern __shared__ double smem[];
    double* s_S = smem;
    int total_S = dim * dim;
    for (int idx = tid; idx < total_S; idx += blockDim.x) {
        s_S[idx] = S[idx];
    }
    __syncthreads();

    // Step 1: temp[row_i][l] = sum_k S[row_i][k] * sig[k][l]
    for (int l = tid; l < dim; l += blockDim.x) {
        double sum = 0.0;
        for (int k = 0; k < dim; ++k) {
            sum += s_S[row_i * dim + k] * sig[k * dim + l];
        }
        temp_row[l] = sum;
    }
    __syncthreads();

    // Step 2: sig[row_i][j] = sum_l temp[row_i][l] * S^T[l][j]
    //                        = sum_l temp[row_i][l] * S[j][l]
    for (int j = tid; j < dim; j += blockDim.x) {
        double sum = 0.0;
        for (int l = 0; l < dim; ++l) {
            sum += temp_row[l] * s_S[j * dim + l];
        }
        sig[row_i * dim + j] = sum;
    }
}

// ============================================================================
// Legacy kernels kept for API compatibility (used by tests or other callers)
// ============================================================================
__global__ void displacement_simple_kernel(
    double** d_ptrs,
    const double* d_old_buffer,
    int dim,
    const int* state_ids,
    int batch_size,
    const double* S,
    const double* dg
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int state_id = state_ids[batch_idx];
    double* d = d_ptrs[state_id];
    const double* d_old = d_old_buffer + (static_cast<size_t>(batch_idx) * dim);

    if (!d) return;
    
    for (int i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += S[i * dim + j] * d_old[j];
        }
        d[i] = sum + dg[i];
    }
}

__global__ void copy_displacement_kernel(
    double** d_ptrs,
    double* d_old_buffer,
    int dim,
    const int* state_ids,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int state_id = state_ids[batch_idx];
    double* d = d_ptrs[state_id];
    double* d_old = d_old_buffer + (static_cast<size_t>(batch_idx) * dim);

    if (!d) return;

    for (int i = 0; i < dim; ++i) {
        d_old[i] = d[i];
    }
}

__global__ void covariance_step1_kernel(
    double** sigma_ptrs,
    double* temp_buffer, 
    int dim,
    const int* state_ids,
    int batch_size,
    const double* S
) {
    int batch_idx = blockIdx.z;
    int i = blockIdx.y;
    int l = blockIdx.x;
    
    if (batch_idx >= batch_size || i >= dim || l >= dim) return;
    
    int state_id = state_ids[batch_idx];
    double* sig = sigma_ptrs[state_id];
    double* temp = temp_buffer + (static_cast<size_t>(batch_idx) * dim * dim);

    if (!sig) return;
    
    double sum = 0.0;
    for (int k = 0; k < dim; ++k) {
        sum += S[i * dim + k] * sig[k * dim + l];
    }
    temp[i * dim + l] = sum;
}

__global__ void covariance_step2_kernel(
    double** sigma_ptrs,
    const double* temp_buffer,
    int dim,
    const int* state_ids,
    int batch_size,
    const double* S
) {
    int batch_idx = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x;
    
    if (batch_idx >= batch_size || i >= dim || j >= dim) return;
    
    int state_id = state_ids[batch_idx];
    double* sig = sigma_ptrs[state_id];
    const double* temp = temp_buffer + (static_cast<size_t>(batch_idx) * dim * dim);

    if (!sig) return;
    
    double sum = 0.0;
    for (int l = 0; l < dim; ++l) {
        sum += temp[i * dim + l] * S[j * dim + l];
    }
    sig[i * dim + j] = sum;
}

// ============================================================================
// Optimized host-side launch function
// ============================================================================
void apply_batched_symplectic_update(
    GaussianStatePool* pool,
    const int* d_state_ids,
    int batch_size,
    const double* d_S,
    const double* d_dg,
    double* d_old_buffer,
    double* d_temp_buffer,
    cudaStream_t stream,
    bool synchronize
) {
    if (batch_size <= 0) return;
    
    int dim = 2 * pool->get_num_qumodes();
    double** d_ptrs = pool->get_d_ptrs_device();
    double** sigma_ptrs = pool->get_sig_ptrs_device();

    double* d_old = d_old_buffer;
    bool owns_old_buffer = false;
    if (!d_old) {
        CHECK_CUDA(cudaMalloc(&d_old, static_cast<size_t>(batch_size) * dim * sizeof(double)));
        owns_old_buffer = true;
    }

    double* d_temp = d_temp_buffer;
    bool owns_temp_buffer = false;
    if (!d_temp) {
        CHECK_CUDA(cudaMalloc(&d_temp, (size_t)batch_size * dim * dim * sizeof(double)));
        owns_temp_buffer = true;
    }

    // Choose block size: at least dim threads, rounded up to warp size
    int block_threads = ((dim + 31) / 32) * 32;
    if (block_threads > 1024) block_threads = 1024;
    if (block_threads < 32) block_threads = 32;
    size_t smem_bytes = static_cast<size_t>(dim) * dim * sizeof(double);

    // Fused copy + displacement: 1 kernel launch instead of 2
    fused_copy_displacement_kernel<<<batch_size, block_threads, smem_bytes, stream>>>(
        d_ptrs, d_old, dim, d_state_ids, batch_size, d_S, d_dg
    );
    CHECK_CUDA(cudaGetLastError());

    // Fused covariance: 1 kernel launch instead of 2
    // Grid: (dim rows) x (batch_size)
    dim3 cov_grid(dim, batch_size);
    fused_covariance_kernel<<<cov_grid, block_threads, smem_bytes, stream>>>(
        sigma_ptrs, d_temp, dim, d_state_ids, batch_size, d_S
    );
    CHECK_CUDA(cudaGetLastError());

    if (synchronize || owns_old_buffer || owns_temp_buffer) {
        CHECK_CUDA(stream != nullptr ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize());
    }
    if (owns_old_buffer) {
        CHECK_CUDA(cudaFree(d_old));
    }
    if (owns_temp_buffer) {
        CHECK_CUDA(cudaFree(d_temp));
    }
}
