#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include "cv_state_pool.h"
#include "hdd_node.h"

// Helper for CUDA error checking
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while (0)

namespace {

bool hybrid_debug_logging_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("HYBRIDCVDV_FALLBACK_DEBUG");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

void validate_paired_state_lists(const char* gate_name,
                                 const CVStatePool* state_pool,
                                 const std::vector<int>& qubit0_states,
                                 const std::vector<int>& qubit1_states,
                                 int target_qumode,
                                 int target_mode_dim,
                                 int target_mode_right_stride) {
    const int64_t mode_span =
        static_cast<int64_t>(target_mode_dim) * static_cast<int64_t>(target_mode_right_stride);
    int64_t max_dim = 0;
    for (size_t i = 0; i < qubit0_states.size(); ++i) {
        const int id0 = qubit0_states[i];
        const int id1 = qubit1_states[i];
        const int64_t dim0 = state_pool->get_state_dim(id0);
        const int64_t dim1 = state_pool->get_state_dim(id1);
        if (dim0 != dim1) {
            throw std::runtime_error(
                std::string(gate_name) + " paired state dims mismatch: id0=" +
                std::to_string(id0) + " dim0=" + std::to_string(dim0) +
                ", id1=" + std::to_string(id1) + " dim1=" + std::to_string(dim1));
        }
        if (dim0 <= 0) {
            throw std::runtime_error(
                std::string(gate_name) + " paired state has non-positive dim: id0=" +
                std::to_string(id0) + ", id1=" + std::to_string(id1) +
                ", dim=" + std::to_string(dim0));
        }
        if (mode_span > 0 && (dim0 % mode_span) != 0) {
            throw std::runtime_error(
                std::string(gate_name) + " paired state dim is not aligned with target mode span: id0=" +
                std::to_string(id0) + ", id1=" + std::to_string(id1) +
                ", dim=" + std::to_string(dim0) +
                ", mode_span=" + std::to_string(mode_span));
        }
        max_dim = std::max(max_dim, dim0);
    }

    if (!hybrid_debug_logging_enabled()) {
        return;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const bool have_mem_info = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
    std::cout << "[hybrid-debug] " << gate_name
              << " pairs=" << qubit0_states.size()
              << " target_qumode=" << target_qumode
              << " right_stride=" << target_mode_right_stride
              << " mode_span=" << mode_span
              << " max_dim=" << max_dim;
    if (have_mem_info) {
        std::cout << " free_mb=" << (free_bytes / (1024 * 1024))
                  << " total_mb=" << (total_bytes / (1024 * 1024));
    }
    std::cout << std::endl;

    const size_t preview = std::min<size_t>(qubit0_states.size(), 4);
    for (size_t i = 0; i < preview; ++i) {
        const int id0 = qubit0_states[i];
        const int id1 = qubit1_states[i];
        std::cout << "[hybrid-debug] " << gate_name
                  << " pair[" << i << "] ids=(" << id0 << "," << id1 << ")"
                  << " dims=(" << state_pool->host_state_dims[id0]
                  << "," << state_pool->host_state_dims[id1] << ")"
                  << " offsets=(" << state_pool->host_state_offsets[id0]
                  << "," << state_pool->host_state_offsets[id1] << ")"
                  << std::endl;
    }
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

void upload_host_bytes(CVStatePool* state_pool,
                       void* device_ptr,
                       const void* host_ptr,
                       size_t bytes) {
    if (bytes == 0) {
        return;
    }
    void* staged = state_pool->host_transfer_staging.ensure(bytes);
    std::memcpy(staged, host_ptr, bytes);
    CHECK_CUDA(cudaMemcpy(device_ptr, staged, bytes, cudaMemcpyHostToDevice));
}

void upload_int_pair_vectors(CVStatePool* state_pool,
                             void* device_ptr,
                             const std::vector<int>& first,
                             const std::vector<int>& second) {
    if (first.size() != second.size()) {
        throw std::invalid_argument("pair upload requires equal-sized vectors");
    }
    const size_t first_bytes = first.size() * sizeof(int);
    const size_t total_bytes = first_bytes * 2;
    int* staged = static_cast<int*>(state_pool->host_transfer_staging.ensure(total_bytes));
    std::memcpy(staged, first.data(), first_bytes);
    std::memcpy(staged + first.size(), second.data(), first_bytes);
    CHECK_CUDA(cudaMemcpy(device_ptr, staged, total_bytes, cudaMemcpyHostToDevice));
}

}  // namespace

/**
 * Level 4: 混合控制门 (Hybrid Control Gates) GPU内核
 */

// ==========================================
// 1. Conditional Displacement (CD)
// ==========================================

__global__ void apply_controlled_displacement_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    int capacity,
    const int* target_state_ids,
    int num_states,
    cuDoubleComplex alpha,
    cuDoubleComplex* temp_buffer,
    size_t buffer_stride,
    int target_mode_dim,
    int target_mode_right_stride
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_states) return;

    int state_id = target_state_ids[batch_idx];
    if (state_id < 0 || state_id >= capacity) return;

    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t current_dim = state_dims[state_id]; // 获取当前状态的维度
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    size_t offset = state_offsets[state_id];
    cuDoubleComplex* psi = &all_states_data[offset];
    cuDoubleComplex* psi_out = &temp_buffer[batch_idx * buffer_stride];

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t mode_block = static_cast<size_t>(target_mode_dim) * right_stride;
    if (mode_block == 0) return;

    const size_t group_start =
        (flat_index / mode_block) * mode_block + (flat_index % right_stride);
    const int n = static_cast<int>((flat_index / right_stride) % target_mode_dim);

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    double alpha_real = cuCreal(alpha);
    double alpha_imag = cuCimag(alpha);
    double alpha_norm_sq = alpha_real*alpha_real + alpha_imag*alpha_imag;
    double prefactor = exp(-alpha_norm_sq / 2.0);

    for (int m = 0; m < target_mode_dim; ++m) {
        // Compute D_nm = <n|D(alpha)|m>
        double sqrt_fact_ratio = 1.0;
        if (n > m) {
            for (int k = m + 1; k <= n; ++k) sqrt_fact_ratio *= sqrt((double)k);
            sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
        } else if (m > n) {
            for (int k = n + 1; k <= m; ++k) sqrt_fact_ratio *= sqrt((double)k);
            sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
        }
        
        int lower = (n < m) ? n : m;
        int upper = (n > m) ? n : m;
        int k = upper - lower; // abs(n-m)
        
        double laguerre = 0.0;
        double term = 1.0;
        double binom = 1.0;
        // binom(upper, lower)
        for(int i=1; i<=lower; ++i) binom = binom * (upper - i + 1) / i;
        term = binom;
        laguerre += term;
        
        for(int j=1; j<=lower; ++j) {
            term = term * (-alpha_norm_sq) * (lower - j + 1) / ((k + j) * j);
            laguerre += term;
        }
        
        cuDoubleComplex power_val = make_cuDoubleComplex(1.0, 0.0);
        if (n >= m) {
            for(int p=0; p<k; ++p) power_val = cuCmul(power_val, alpha);
        } else {
            cuDoubleComplex minus_alpha_conj = make_cuDoubleComplex(-alpha_real, alpha_imag);
            for(int p=0; p<k; ++p) power_val = cuCmul(power_val, minus_alpha_conj);
        }
        
        double real_scale = prefactor * sqrt_fact_ratio * laguerre;
        cuDoubleComplex d_nm = make_cuDoubleComplex(real_scale * cuCreal(power_val), real_scale * cuCimag(power_val));

        const size_t source_index = group_start + static_cast<size_t>(m) * right_stride;
        if (source_index < static_cast<size_t>(current_dim)) {
            sum = cuCadd(sum, cuCmul(d_nm, psi[source_index]));
        }
    }
    psi_out[flat_index] = sum;
}

__global__ void apply_controlled_displacement_inplace_shared_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    int capacity,
    const int* target_state_ids,
    int num_states,
    cuDoubleComplex alpha,
    int target_mode_dim,
    int target_mode_right_stride,
    int slices_per_block
) {
    extern __shared__ cuDoubleComplex shared_slices[];

    const int batch_idx = blockIdx.y;
    if (batch_idx >= num_states) return;

    const int state_id = target_state_ids[batch_idx];
    if (state_id < 0 || state_id >= capacity) return;

    const int local_thread = threadIdx.x;
    const int slice_slot = local_thread / target_mode_dim;
    const int n = local_thread % target_mode_dim;
    if (slice_slot >= slices_per_block) return;

    const int64_t current_dim = state_dims[state_id];
    if (current_dim <= 0 || current_dim % target_mode_dim != 0 || target_mode_right_stride <= 0) {
        return;
    }

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t mode_block = static_cast<size_t>(target_mode_dim) * right_stride;
    const size_t num_slices = static_cast<size_t>(current_dim / target_mode_dim);
    const size_t slice_idx = static_cast<size_t>(blockIdx.x) * slices_per_block + slice_slot;

    cuDoubleComplex* shared_slice = shared_slices + slice_slot * target_mode_dim;
    size_t target_index = 0;
    bool slice_active = false;

    const size_t offset = state_offsets[state_id];
    cuDoubleComplex* psi = &all_states_data[offset];
    if (slice_idx < num_slices) {
        const size_t outer_group = slice_idx / right_stride;
        const size_t inner_offset = slice_idx % right_stride;
        const size_t group_start = outer_group * mode_block + inner_offset;
        target_index = group_start + static_cast<size_t>(n) * right_stride;
        shared_slice[n] = psi[target_index];
        slice_active = true;
    } else {
        shared_slice[n] = make_cuDoubleComplex(0.0, 0.0);
    }

    __syncthreads();

    if (!slice_active) {
        return;
    }

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    const double alpha_real = cuCreal(alpha);
    const double alpha_imag = cuCimag(alpha);
    const double alpha_norm_sq = alpha_real * alpha_real + alpha_imag * alpha_imag;
    const double prefactor = exp(-alpha_norm_sq / 2.0);

    for (int m = 0; m < target_mode_dim; ++m) {
        double sqrt_fact_ratio = 1.0;
        if (n > m) {
            for (int k = m + 1; k <= n; ++k) sqrt_fact_ratio *= sqrt((double)k);
            sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
        } else if (m > n) {
            for (int k = n + 1; k <= m; ++k) sqrt_fact_ratio *= sqrt((double)k);
            sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
        }

        const int lower = (n < m) ? n : m;
        const int upper = (n > m) ? n : m;
        const int k = upper - lower;

        double laguerre = 0.0;
        double term = 1.0;
        double binom = 1.0;
        for (int i = 1; i <= lower; ++i) binom = binom * (upper - i + 1) / i;
        term = binom;
        laguerre += term;

        for (int j = 1; j <= lower; ++j) {
            term = term * (-alpha_norm_sq) * (lower - j + 1) / ((k + j) * j);
            laguerre += term;
        }

        cuDoubleComplex power_val = make_cuDoubleComplex(1.0, 0.0);
        if (n >= m) {
            for (int p = 0; p < k; ++p) power_val = cuCmul(power_val, alpha);
        } else {
            const cuDoubleComplex minus_alpha_conj = make_cuDoubleComplex(-alpha_real, alpha_imag);
            for (int p = 0; p < k; ++p) power_val = cuCmul(power_val, minus_alpha_conj);
        }

        const double real_scale = prefactor * sqrt_fact_ratio * laguerre;
        const cuDoubleComplex d_nm = make_cuDoubleComplex(real_scale * cuCreal(power_val),
                                                          real_scale * cuCimag(power_val));
        sum = cuCadd(sum, cuCmul(d_nm, shared_slice[m]));
    }

    psi[target_index] = sum;
}

__global__ void copy_back_hybrid_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    int capacity,
    const int* target_state_ids,
    int num_states,
    const cuDoubleComplex* temp_buffer,
    size_t buffer_stride
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_states) return;

    int state_id = target_state_ids[batch_idx];
    if (state_id < 0 || state_id >= capacity) return;

    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t current_dim = state_dims[state_id];
    
    if (n >= current_dim) return;

    size_t offset = state_offsets[state_id];
    all_states_data[offset + n] = temp_buffer[batch_idx * buffer_stride + n];
}

void apply_controlled_displacement_on_mode(CVStatePool* state_pool,
                                           const std::vector<int>& controlled_states,
                                           cuDoubleComplex alpha,
                                           int target_qumode,
                                           int num_qumodes) {
    if (controlled_states.empty()) return;
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);
    const int target_mode_dim = state_pool->d_trunc;

    // 验证所有状态ID的有效性
    for (int state_id : controlled_states) {
        if (!state_pool->is_valid_state(state_id)) {
            throw std::runtime_error("无效的状态ID: " + std::to_string(state_id));
        }
    }

    const size_t ids_bytes = controlled_states.size() * sizeof(int);
    int* d_state_ids = state_pool->upload_vector_to_buffer(
        controlled_states, state_pool->scratch_target_ids);

    bool can_use_inplace_shared = target_mode_dim > 0 && target_mode_dim <= 256;
    size_t max_slices = 0;
    if (can_use_inplace_shared) {
        for (int state_id : controlled_states) {
            const int64_t state_dim = state_pool->host_state_dims[state_id];
            if (state_dim <= 0 || state_dim % target_mode_dim != 0) {
                can_use_inplace_shared = false;
                break;
            }
            max_slices = std::max(max_slices, static_cast<size_t>(state_dim / target_mode_dim));
        }
    }

    if (can_use_inplace_shared) {
        const int slices_per_block = std::max(1, 256 / target_mode_dim);
        const int threads_per_block = slices_per_block * target_mode_dim;
        dim3 block_dim(threads_per_block);
        const unsigned int grid_x =
            static_cast<unsigned int>((max_slices + static_cast<size_t>(slices_per_block) - 1) /
                                      static_cast<size_t>(slices_per_block));
        const size_t shared_bytes =
            static_cast<size_t>(threads_per_block) * sizeof(cuDoubleComplex);
        constexpr size_t kMaxSharedBlocksPerLaunch = 1u << 20;
        size_t states_per_launch = controlled_states.size();
        if (grid_x != 0) {
            states_per_launch =
                std::max<size_t>(1, kMaxSharedBlocksPerLaunch / static_cast<size_t>(grid_x));
        }
        states_per_launch = std::min(states_per_launch, controlled_states.size());

        if (hybrid_debug_logging_enabled()) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            const bool have_mem_info = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
            std::cout << "[hybrid-debug] Displacement path=shared"
                      << " states=" << controlled_states.size()
                      << " target_qumode=" << target_qumode
                      << " right_stride=" << target_mode_right_stride
                      << " target_mode_dim=" << target_mode_dim
                      << " max_slices=" << max_slices
                      << " slices_per_block=" << slices_per_block
                      << " threads_per_block=" << threads_per_block
                      << " grid_x=" << grid_x
                      << " states_per_launch=" << states_per_launch
                      << " shared_bytes=" << shared_bytes;
            if (have_mem_info) {
                std::cout << " free_mb=" << (free_bytes / (1024 * 1024))
                          << " total_mb=" << (total_bytes / (1024 * 1024));
            }
            std::cout << std::endl;
        }

        for (size_t state_offset = 0; state_offset < controlled_states.size();
             state_offset += states_per_launch) {
            const int chunk_states = static_cast<int>(
                std::min(states_per_launch, controlled_states.size() - state_offset));
            dim3 grid_dim(grid_x, static_cast<unsigned int>(chunk_states));
            apply_controlled_displacement_inplace_shared_kernel<<<grid_dim, block_dim, shared_bytes>>>(
                state_pool->data,
                state_pool->state_offsets,
                state_pool->state_dims,
                state_pool->capacity,
                d_state_ids + state_offset,
                chunk_states,
                alpha,
                target_mode_dim,
                target_mode_right_stride,
                slices_per_block);
            CHECK_CUDA(cudaGetLastError());
            if (hybrid_debug_logging_enabled()) {
                CHECK_CUDA(cudaDeviceSynchronize());
                std::cout << "[hybrid-debug] Displacement shared chunk complete"
                          << " state_offset=" << state_offset
                          << " chunk_states=" << chunk_states
                          << " target_qumode=" << target_qumode
                          << std::endl;
            }
        }
        if (hybrid_debug_logging_enabled()) {
            std::cout << "[hybrid-debug] Displacement shared path complete"
                      << " states=" << controlled_states.size()
                      << " target_qumode=" << target_qumode
                      << std::endl;
        }
        return;
    }

    size_t buffer_stride = state_pool->max_total_dim;
    size_t buffer_size = controlled_states.size() * buffer_stride * sizeof(cuDoubleComplex);
    if (hybrid_debug_logging_enabled()) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const bool have_mem_info = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
        std::cout << "[hybrid-debug] Displacement path=temp"
                  << " states=" << controlled_states.size()
                  << " target_qumode=" << target_qumode
                  << " right_stride=" << target_mode_right_stride
                  << " target_mode_dim=" << target_mode_dim
                  << " buffer_stride=" << buffer_stride
                  << " buffer_bytes=" << buffer_size;
        if (have_mem_info) {
            std::cout << " free_mb=" << (free_bytes / (1024 * 1024))
                      << " total_mb=" << (total_bytes / (1024 * 1024));
        }
        std::cout << std::endl;
    }
    cuDoubleComplex* temp_buffer = static_cast<cuDoubleComplex*>(
        state_pool->scratch_temp.ensure(buffer_size));

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, controlled_states.size());

    apply_controlled_displacement_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        state_pool->capacity,
        d_state_ids,
        controlled_states.size(),
        alpha,
        temp_buffer,
        buffer_stride,
        target_mode_dim,
        target_mode_right_stride
    );

    CHECK_CUDA(cudaGetLastError());
    if (hybrid_debug_logging_enabled()) {
        CHECK_CUDA(cudaDeviceSynchronize());
        std::cout << "[hybrid-debug] Displacement temp path kernel complete"
                  << " states=" << controlled_states.size()
                  << " target_qumode=" << target_qumode
                  << std::endl;
    }

    copy_back_hybrid_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        state_pool->capacity,
        d_state_ids, controlled_states.size(), temp_buffer, buffer_stride
    );
    CHECK_CUDA(cudaGetLastError());
    if (hybrid_debug_logging_enabled()) {
        CHECK_CUDA(cudaDeviceSynchronize());
        std::cout << "[hybrid-debug] Displacement temp path copy-back complete"
                  << " states=" << controlled_states.size()
                  << " target_qumode=" << target_qumode
                  << std::endl;
    }
}

void apply_controlled_displacement(CVStatePool* state_pool,
                                   const std::vector<int>& controlled_states,
                                   cuDoubleComplex alpha) {
    apply_controlled_displacement_on_mode(state_pool, controlled_states, alpha, 0, 1);
}


// ==========================================
// 2. Conditional Rotation (CR) & Parity (CP)
// ==========================================

__global__ void apply_controlled_rotation_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* target_state_ids,
    int num_states,
    double theta
) {
    int state_id = target_state_ids[blockIdx.y];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int64_t current_dim = state_dims[state_id];
    if (n >= current_dim) return;

    size_t offset = state_offsets[state_id];
    cuDoubleComplex* psi = &all_states_data[offset];
    
    // CR(theta) = exp[-i theta/2 sigma_z n]
    double phase = (theta / 2.0) * (double)n;
    double cos_val = cos(phase);
    double sin_val = sin(phase);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_val, sin_val);
    
    psi[n] = cuCmul(psi[n], phase_factor);
}

void apply_controlled_rotation(CVStatePool* state_pool,
                               const std::vector<int>& controlled_states,
                               double theta) {
    if (controlled_states.empty()) return;
    int* d_ids = state_pool->upload_vector_to_buffer(
        controlled_states, state_pool->scratch_target_ids);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, controlled_states.size());

    apply_controlled_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_ids, controlled_states.size(), theta
    );
}

void apply_controlled_parity(CVStatePool* state_pool, const std::vector<int>& controlled_states) {
    apply_controlled_rotation(state_pool, controlled_states, M_PI);
}

// ==========================================
// 3. Rabi Interaction (RB)
// ==========================================

// Kernel to mix v0 and v1 (Hadamard-like or general rotation mixing)
// Used for Rabi: Basis transform -> Displacement -> Basis transform
__global__ void batch_mix_rabi_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* id0_list,
    const int* id1_list,
    int num_pairs,
    bool inverse
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;

    int id0 = id0_list[batch_idx];
    int id1 = id1_list[batch_idx];
    
    // 假设两个状态维度相同
    int64_t current_dim = state_dims[id0];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= current_dim) return;

    size_t offset0 = state_offsets[id0];
    size_t offset1 = state_offsets[id1];

    cuDoubleComplex* p0 = &all_states_data[offset0];
    cuDoubleComplex* p1 = &all_states_data[offset1];
    
    cuDoubleComplex v0 = p0[n];
    cuDoubleComplex v1 = p1[n];
    double inv_sqrt2 = 0.7071067811865475;
    
    p0[n] = make_cuDoubleComplex(
        (cuCreal(v0) + cuCreal(v1)) * inv_sqrt2,
        (cuCimag(v0) + cuCimag(v1)) * inv_sqrt2
    );
    p1[n] = make_cuDoubleComplex(
        (cuCreal(v0) - cuCreal(v1)) * inv_sqrt2,
        (cuCimag(v0) - cuCimag(v1)) * inv_sqrt2
    );
}

void apply_rabi_interaction_on_mode(CVStatePool* state_pool,
                                    const std::vector<int>& qubit0_states,
                                    const std::vector<int>& qubit1_states,
                                    double theta,
                                    int target_qumode,
                                    int num_qumodes) {
    if (qubit0_states.size() != qubit1_states.size()) throw std::invalid_argument("State lists mismatch");
    if (qubit0_states.empty()) return;
    
    size_t n = qubit0_states.size();
    const size_t pair_bytes = 2 * n * sizeof(int);
    char* aux = static_cast<char*>(state_pool->scratch_aux.ensure(pair_bytes));
    int* d_id0 = reinterpret_cast<int*>(aux);
    int* d_id1 = reinterpret_cast<int*>(aux + n * sizeof(int));
    upload_int_pair_vectors(state_pool, aux, qubit0_states, qubit1_states);

    dim3 bd(256);
    dim3 gd((state_pool->max_total_dim + bd.x - 1)/bd.x, n);

    // 1. Transform to X basis
    batch_mix_rabi_kernel<<<gd, bd>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_id0, d_id1, n, false
    );
    CHECK_CUDA(cudaGetLastError());

    cuDoubleComplex alpha0 = make_cuDoubleComplex(0.0, -theta);
    cuDoubleComplex alpha1 = make_cuDoubleComplex(0.0, theta);

    apply_controlled_displacement_on_mode(
        state_pool, qubit0_states, alpha0, target_qumode, num_qumodes);
    apply_controlled_displacement_on_mode(
        state_pool, qubit1_states, alpha1, target_qumode, num_qumodes);

    // Re-obtain pointers (scratch_aux may have been reallocated by displacement calls)
    aux = static_cast<char*>(state_pool->scratch_aux.ensure(pair_bytes));
    d_id0 = reinterpret_cast<int*>(aux);
    d_id1 = reinterpret_cast<int*>(aux + n * sizeof(int));
    upload_int_pair_vectors(state_pool, aux, qubit0_states, qubit1_states);

    // 3. Transform back
    batch_mix_rabi_kernel<<<gd, bd>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_id0, d_id1, n, true
    );
    CHECK_CUDA(cudaGetLastError());
}

void apply_rabi_interaction(CVStatePool* state_pool,
                            const std::vector<int>& qubit0_states,
                            const std::vector<int>& qubit1_states,
                            double theta) {
    apply_rabi_interaction_on_mode(state_pool, qubit0_states, qubit1_states, theta, 0, 1);
}

