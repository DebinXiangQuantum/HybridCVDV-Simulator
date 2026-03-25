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

// ==========================================
// 4. Jaynes-Cummings (JC)
// ==========================================

__global__ void apply_jc_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* id0_list,
    const int* id1_list,
    int num_pairs,
    double theta,
    double phi,
    int target_mode_dim,
    int target_mode_right_stride
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;

    int id0 = id0_list[batch_idx];
    int id1 = id1_list[batch_idx];

    int64_t current_dim = state_dims[id0];
    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t pair_group_span = static_cast<size_t>(target_mode_dim - 1) * right_stride;
    if (pair_group_span == 0) return;

    const size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t group_count =
        static_cast<size_t>(current_dim) / (static_cast<size_t>(target_mode_dim) * right_stride);
    const size_t total_pairs = group_count * pair_group_span;
    if (pair_index >= total_pairs) return;

    size_t offset0 = state_offsets[id0];
    size_t offset1 = state_offsets[id1];

    cuDoubleComplex* v0 = &all_states_data[offset0];
    cuDoubleComplex* v1 = &all_states_data[offset1];

    const size_t group_index = pair_index / pair_group_span;
    const size_t within_group = pair_index % pair_group_span;
    const int n = static_cast<int>(within_group / right_stride);
    const size_t right_index = within_group % right_stride;
    const size_t group_start =
        group_index * static_cast<size_t>(target_mode_dim) * right_stride + right_index;
    const size_t idx1 = group_start + static_cast<size_t>(n) * right_stride;
    const size_t idx0 = group_start + static_cast<size_t>(n + 1) * right_stride;
    
    double omega = theta * sqrt((double)(n + 1));
    double cos_w = cos(omega);
    double sin_w = sin(omega);
    
    cuDoubleComplex c1 = v1[idx1];
    cuDoubleComplex c0 = v0[idx0];
    
    // ... (rest of calculation remains the same)
    cuDoubleComplex factor01 = make_cuDoubleComplex(-sin(phi) * sin_w, -cos(phi) * sin_w);
    cuDoubleComplex factor10 = make_cuDoubleComplex(sin(phi) * sin_w, -cos(phi) * sin_w);
    
    cuDoubleComplex new_c0 = cuCadd(make_cuDoubleComplex(cos_w * cuCreal(c0), cos_w * cuCimag(c0)),
                                    cuCmul(factor01, c1));
                                    
    cuDoubleComplex new_c1 = cuCadd(cuCmul(factor10, c0),
                                    make_cuDoubleComplex(cos_w * cuCreal(c1), cos_w * cuCimag(c1)));
                                    
    v0[idx0] = new_c0;
    v1[idx1] = new_c1;
}

__global__ void apply_jc_single_pair_kernel(
    cuDoubleComplex* v0,
    cuDoubleComplex* v1,
    int64_t current_dim,
    double theta,
    double phi,
    int target_mode_dim,
    int target_mode_right_stride
) {
    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t pair_group_span = static_cast<size_t>(target_mode_dim - 1) * right_stride;
    if (pair_group_span == 0) return;

    const size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t group_count =
        static_cast<size_t>(current_dim) / (static_cast<size_t>(target_mode_dim) * right_stride);
    const size_t total_pairs = group_count * pair_group_span;
    if (pair_index >= total_pairs) return;

    const size_t group_index = pair_index / pair_group_span;
    const size_t within_group = pair_index % pair_group_span;
    const int n = static_cast<int>(within_group / right_stride);
    const size_t right_index = within_group % right_stride;
    const size_t group_start =
        group_index * static_cast<size_t>(target_mode_dim) * right_stride + right_index;
    const size_t idx1 = group_start + static_cast<size_t>(n) * right_stride;
    const size_t idx0 = group_start + static_cast<size_t>(n + 1) * right_stride;

    double omega = theta * sqrt((double)(n + 1));
    double cos_w = cos(omega);
    double sin_w = sin(omega);

    cuDoubleComplex c1 = v1[idx1];
    cuDoubleComplex c0 = v0[idx0];

    cuDoubleComplex factor01 = make_cuDoubleComplex(-sin(phi) * sin_w, -cos(phi) * sin_w);
    cuDoubleComplex factor10 = make_cuDoubleComplex(sin(phi) * sin_w, -cos(phi) * sin_w);

    cuDoubleComplex new_c0 = cuCadd(make_cuDoubleComplex(cos_w * cuCreal(c0), cos_w * cuCimag(c0)),
                                    cuCmul(factor01, c1));
    cuDoubleComplex new_c1 = cuCadd(cuCmul(factor10, c0),
                                    make_cuDoubleComplex(cos_w * cuCreal(c1), cos_w * cuCimag(c1)));

    v0[idx0] = new_c0;
    v1[idx1] = new_c1;
}

__global__ void apply_jc_pointer_batch_kernel(
    cuDoubleComplex* const* v0_list,
    cuDoubleComplex* const* v1_list,
    const int64_t* state_dims,
    int num_pairs,
    double theta,
    double phi,
    int target_mode_dim,
    int target_mode_right_stride
) {
    const int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;

    cuDoubleComplex* v0 = v0_list[batch_idx];
    cuDoubleComplex* v1 = v1_list[batch_idx];
    const int64_t current_dim = state_dims[batch_idx];
    if (!v0 || !v1 || current_dim <= 0) return;

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t pair_group_span = static_cast<size_t>(target_mode_dim - 1) * right_stride;
    if (pair_group_span == 0) return;

    const size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t group_count =
        static_cast<size_t>(current_dim) / (static_cast<size_t>(target_mode_dim) * right_stride);
    const size_t total_pairs = group_count * pair_group_span;
    if (pair_index >= total_pairs) return;

    const size_t group_index = pair_index / pair_group_span;
    const size_t within_group = pair_index % pair_group_span;
    const int n = static_cast<int>(within_group / right_stride);
    const size_t right_index = within_group % right_stride;
    const size_t group_start =
        group_index * static_cast<size_t>(target_mode_dim) * right_stride + right_index;
    const size_t idx1 = group_start + static_cast<size_t>(n) * right_stride;
    const size_t idx0 = group_start + static_cast<size_t>(n + 1) * right_stride;

    const double omega = theta * sqrt((double)(n + 1));
    const double cos_w = cos(omega);
    const double sin_w = sin(omega);

    const cuDoubleComplex c1 = v1[idx1];
    const cuDoubleComplex c0 = v0[idx0];

    const cuDoubleComplex factor01 = make_cuDoubleComplex(-sin(phi) * sin_w, -cos(phi) * sin_w);
    const cuDoubleComplex factor10 = make_cuDoubleComplex(sin(phi) * sin_w, -cos(phi) * sin_w);

    const cuDoubleComplex new_c0 =
        cuCadd(make_cuDoubleComplex(cos_w * cuCreal(c0), cos_w * cuCimag(c0)),
               cuCmul(factor01, c1));
    const cuDoubleComplex new_c1 =
        cuCadd(cuCmul(factor10, c0),
               make_cuDoubleComplex(cos_w * cuCreal(c1), cos_w * cuCimag(c1)));

    v0[idx0] = new_c0;
    v1[idx1] = new_c1;
}

void apply_jaynes_cummings_on_mode(CVStatePool* state_pool,
                                   const std::vector<int>& qubit0_states,
                                   const std::vector<int>& qubit1_states,
                                   double theta,
                                   double phi,
                                   int target_qumode,
                                   int num_qumodes) {
    if (qubit0_states.size() != qubit1_states.size()) {
        throw std::invalid_argument("State lists mismatch");
    }
    if (qubit0_states.empty()) return;

    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);
    const int target_mode_dim = state_pool->d_trunc;
    size_t n = qubit0_states.size();
    validate_paired_state_lists(
        "JaynesCummings", state_pool, qubit0_states, qubit1_states,
        target_qumode, target_mode_dim, target_mode_right_stride);
    dim3 bd(256);
    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t pair_group_span = static_cast<size_t>(target_mode_dim - 1) * right_stride;
    if (pair_group_span == 0) {
        return;
    }

    auto align_offset = [](size_t offset, size_t alignment) {
        const size_t mask = alignment - 1;
        return (offset + mask) & ~mask;
    };

    size_t upload_bytes = 0;
    upload_bytes = align_offset(upload_bytes, alignof(cuDoubleComplex*));
    const size_t v0_offset = upload_bytes;
    upload_bytes += n * sizeof(cuDoubleComplex*);
    upload_bytes = align_offset(upload_bytes, alignof(cuDoubleComplex*));
    const size_t v1_offset = upload_bytes;
    upload_bytes += n * sizeof(cuDoubleComplex*);
    upload_bytes = align_offset(upload_bytes, alignof(int64_t));
    const size_t dims_offset = upload_bytes;
    upload_bytes += n * sizeof(int64_t);

    char* aux = static_cast<char*>(state_pool->scratch_aux.ensure(upload_bytes));
    char* staged = static_cast<char*>(state_pool->host_transfer_staging.ensure(upload_bytes));
    auto* d_v0 = reinterpret_cast<cuDoubleComplex**>(aux + v0_offset);
    auto* d_v1 = reinterpret_cast<cuDoubleComplex**>(aux + v1_offset);
    auto* d_dims = reinterpret_cast<int64_t*>(aux + dims_offset);
    auto* h_v0 = reinterpret_cast<cuDoubleComplex**>(staged + v0_offset);
    auto* h_v1 = reinterpret_cast<cuDoubleComplex**>(staged + v1_offset);
    auto* h_dims = reinterpret_cast<int64_t*>(staged + dims_offset);

    for (size_t pair_idx = 0; pair_idx < n; ++pair_idx) {
        const int id0 = qubit0_states[pair_idx];
        const int id1 = qubit1_states[pair_idx];
        const int64_t current_dim = state_pool->get_state_dim(id0);
        cuDoubleComplex* v0 = state_pool->get_state_ptr(id0);
        cuDoubleComplex* v1 = state_pool->get_state_ptr(id1);
        if (!v0 || !v1 || current_dim <= 0) {
            throw std::runtime_error("JaynesCummings encountered invalid paired state pointers");
        }

        h_v0[pair_idx] = v0;
        h_v1[pair_idx] = v1;
        h_dims[pair_idx] = current_dim;

        const size_t group_count =
            static_cast<size_t>(current_dim) /
            (static_cast<size_t>(target_mode_dim) * right_stride);
        const size_t total_pairs = group_count * pair_group_span;

        if (hybrid_debug_logging_enabled()) {
            std::cout << "[hybrid-debug] JaynesCummings launching pair[" << pair_idx << "]"
                      << " ids=(" << id0 << "," << id1 << ")"
                      << " ptrs=(" << static_cast<const void*>(v0)
                      << "," << static_cast<const void*>(v1) << ")"
                      << " total_pairs=" << total_pairs
                      << std::endl;
        }
    }

    if (hybrid_debug_logging_enabled()) {
        for (size_t pair_idx = 0; pair_idx < n; ++pair_idx) {
            const int64_t current_dim = h_dims[pair_idx];
            const size_t group_count =
                static_cast<size_t>(current_dim) /
                (static_cast<size_t>(target_mode_dim) * right_stride);
            const size_t total_pairs = group_count * pair_group_span;
            dim3 gd((total_pairs + bd.x - 1) / bd.x);
            apply_jc_single_pair_kernel<<<gd, bd>>>(
                h_v0[pair_idx],
                h_v1[pair_idx],
                current_dim,
                theta,
                phi,
                target_mode_dim,
                target_mode_right_stride);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            std::cout << "[hybrid-debug] JaynesCummings pair[" << pair_idx << "] complete"
                      << std::endl;
        }
        return;
    }

    CHECK_CUDA(cudaMemcpy(aux, staged, upload_bytes, cudaMemcpyHostToDevice));
    const size_t max_pairs =
        static_cast<size_t>(state_pool->max_total_dim / state_pool->d_trunc) *
        static_cast<size_t>(state_pool->d_trunc - 1);
    dim3 gd((max_pairs + bd.x - 1) / bd.x, n);
    apply_jc_pointer_batch_kernel<<<gd, bd>>>(
        d_v0, d_v1, d_dims, static_cast<int>(n), theta, phi, target_mode_dim, target_mode_right_stride);
    CHECK_CUDA(cudaGetLastError());
}

void apply_jaynes_cummings(CVStatePool* state_pool,
                           const std::vector<int>& qubit0_states,
                           const std::vector<int>& qubit1_states,
                           double theta,
                           double phi) {
    apply_jaynes_cummings_on_mode(state_pool, qubit0_states, qubit1_states, theta, phi, 0, 1);
}

// ==========================================
// 5. Anti-Jaynes-Cummings (AJC)
// ==========================================

__global__ void apply_ajc_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* id0_list,
    const int* id1_list,
    int num_pairs,
    double theta,
    double phi,
    int target_mode_dim,
    int target_mode_right_stride
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;
    int id0 = id0_list[batch_idx];
    int id1 = id1_list[batch_idx];
    
    int64_t current_dim = state_dims[id0];
    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t pair_group_span = static_cast<size_t>(target_mode_dim - 1) * right_stride;
    if (pair_group_span == 0) return;

    const size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t group_count =
        static_cast<size_t>(current_dim) / (static_cast<size_t>(target_mode_dim) * right_stride);
    const size_t total_pairs = group_count * pair_group_span;
    if (pair_index >= total_pairs) return;

    size_t offset0 = state_offsets[id0];
    size_t offset1 = state_offsets[id1];

    cuDoubleComplex* v0 = &all_states_data[offset0];
    cuDoubleComplex* v1 = &all_states_data[offset1];

    const size_t group_index = pair_index / pair_group_span;
    const size_t within_group = pair_index % pair_group_span;
    const int n = static_cast<int>(within_group / right_stride);
    const size_t right_index = within_group % right_stride;
    const size_t group_start =
        group_index * static_cast<size_t>(target_mode_dim) * right_stride + right_index;
    const size_t idx0 = group_start + static_cast<size_t>(n) * right_stride;
    const size_t idx1 = group_start + static_cast<size_t>(n + 1) * right_stride;
    
    double omega = theta * sqrt((double)(n + 1));
    double cos_w = cos(omega);
    double sin_w = sin(omega);
    
    cuDoubleComplex c0 = v0[idx0];
    cuDoubleComplex c1 = v1[idx1];
    
    cuDoubleComplex factor01 = make_cuDoubleComplex(-sin(phi) * sin_w, -cos(phi) * sin_w);
    cuDoubleComplex factor10 = make_cuDoubleComplex(sin(phi) * sin_w, -cos(phi) * sin_w);
    
    v0[idx0] = cuCadd(make_cuDoubleComplex(cos_w * cuCreal(c0), cos_w * cuCimag(c0)),
                      cuCmul(factor01, c1));
    v1[idx1] = cuCadd(cuCmul(factor10, c0),
                      make_cuDoubleComplex(cos_w * cuCreal(c1), cos_w * cuCimag(c1)));
}

__global__ void apply_ajc_single_pair_kernel(
    cuDoubleComplex* v0,
    cuDoubleComplex* v1,
    int64_t current_dim,
    double theta,
    double phi,
    int target_mode_dim,
    int target_mode_right_stride
) {
    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t pair_group_span = static_cast<size_t>(target_mode_dim - 1) * right_stride;
    if (pair_group_span == 0) return;

    const size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t group_count =
        static_cast<size_t>(current_dim) / (static_cast<size_t>(target_mode_dim) * right_stride);
    const size_t total_pairs = group_count * pair_group_span;
    if (pair_index >= total_pairs) return;

    const size_t group_index = pair_index / pair_group_span;
    const size_t within_group = pair_index % pair_group_span;
    const int n = static_cast<int>(within_group / right_stride);
    const size_t right_index = within_group % right_stride;
    const size_t group_start =
        group_index * static_cast<size_t>(target_mode_dim) * right_stride + right_index;
    const size_t idx0 = group_start + static_cast<size_t>(n) * right_stride;
    const size_t idx1 = group_start + static_cast<size_t>(n + 1) * right_stride;

    double omega = theta * sqrt((double)(n + 1));
    double cos_w = cos(omega);
    double sin_w = sin(omega);

    cuDoubleComplex c0 = v0[idx0];
    cuDoubleComplex c1 = v1[idx1];

    cuDoubleComplex factor01 = make_cuDoubleComplex(-sin(phi) * sin_w, -cos(phi) * sin_w);
    cuDoubleComplex factor10 = make_cuDoubleComplex(sin(phi) * sin_w, -cos(phi) * sin_w);

    v0[idx0] = cuCadd(make_cuDoubleComplex(cos_w * cuCreal(c0), cos_w * cuCimag(c0)),
                      cuCmul(factor01, c1));
    v1[idx1] = cuCadd(cuCmul(factor10, c0),
                      make_cuDoubleComplex(cos_w * cuCreal(c1), cos_w * cuCimag(c1)));
}

__global__ void apply_ajc_pointer_batch_kernel(
    cuDoubleComplex* const* v0_list,
    cuDoubleComplex* const* v1_list,
    const int64_t* state_dims,
    int num_pairs,
    double theta,
    double phi,
    int target_mode_dim,
    int target_mode_right_stride
) {
    const int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;

    cuDoubleComplex* v0 = v0_list[batch_idx];
    cuDoubleComplex* v1 = v1_list[batch_idx];
    const int64_t current_dim = state_dims[batch_idx];
    if (!v0 || !v1 || current_dim <= 0) return;

    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t pair_group_span = static_cast<size_t>(target_mode_dim - 1) * right_stride;
    if (pair_group_span == 0) return;

    const size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t group_count =
        static_cast<size_t>(current_dim) / (static_cast<size_t>(target_mode_dim) * right_stride);
    const size_t total_pairs = group_count * pair_group_span;
    if (pair_index >= total_pairs) return;

    const size_t group_index = pair_index / pair_group_span;
    const size_t within_group = pair_index % pair_group_span;
    const int n = static_cast<int>(within_group / right_stride);
    const size_t right_index = within_group % right_stride;
    const size_t group_start =
        group_index * static_cast<size_t>(target_mode_dim) * right_stride + right_index;
    const size_t idx0 = group_start + static_cast<size_t>(n) * right_stride;
    const size_t idx1 = group_start + static_cast<size_t>(n + 1) * right_stride;

    const double omega = theta * sqrt((double)(n + 1));
    const double cos_w = cos(omega);
    const double sin_w = sin(omega);

    const cuDoubleComplex c0 = v0[idx0];
    const cuDoubleComplex c1 = v1[idx1];

    const cuDoubleComplex factor01 = make_cuDoubleComplex(-sin(phi) * sin_w, -cos(phi) * sin_w);
    const cuDoubleComplex factor10 = make_cuDoubleComplex(sin(phi) * sin_w, -cos(phi) * sin_w);

    v0[idx0] = cuCadd(make_cuDoubleComplex(cos_w * cuCreal(c0), cos_w * cuCimag(c0)),
                      cuCmul(factor01, c1));
    v1[idx1] = cuCadd(cuCmul(factor10, c0),
                      make_cuDoubleComplex(cos_w * cuCreal(c1), cos_w * cuCimag(c1)));
}

void apply_anti_jaynes_cummings_on_mode(CVStatePool* state_pool,
                                        const std::vector<int>& qubit0_states,
                                        const std::vector<int>& qubit1_states,
                                        double theta,
                                        double phi,
                                        int target_qumode,
                                        int num_qumodes) {
    if (qubit0_states.size() != qubit1_states.size()) {
        throw std::invalid_argument("State lists mismatch");
    }
    if (qubit0_states.empty()) return;

    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);
    const int target_mode_dim = state_pool->d_trunc;
    size_t n = qubit0_states.size();
    validate_paired_state_lists(
        "AntiJaynesCummings", state_pool, qubit0_states, qubit1_states,
        target_qumode, target_mode_dim, target_mode_right_stride);
    dim3 bd(256);
    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const size_t pair_group_span = static_cast<size_t>(target_mode_dim - 1) * right_stride;
    if (pair_group_span == 0) {
        return;
    }

    auto align_offset = [](size_t offset, size_t alignment) {
        const size_t mask = alignment - 1;
        return (offset + mask) & ~mask;
    };

    size_t upload_bytes = 0;
    upload_bytes = align_offset(upload_bytes, alignof(cuDoubleComplex*));
    const size_t v0_offset = upload_bytes;
    upload_bytes += n * sizeof(cuDoubleComplex*);
    upload_bytes = align_offset(upload_bytes, alignof(cuDoubleComplex*));
    const size_t v1_offset = upload_bytes;
    upload_bytes += n * sizeof(cuDoubleComplex*);
    upload_bytes = align_offset(upload_bytes, alignof(int64_t));
    const size_t dims_offset = upload_bytes;
    upload_bytes += n * sizeof(int64_t);

    char* aux = static_cast<char*>(state_pool->scratch_aux.ensure(upload_bytes));
    char* staged = static_cast<char*>(state_pool->host_transfer_staging.ensure(upload_bytes));
    auto* d_v0 = reinterpret_cast<cuDoubleComplex**>(aux + v0_offset);
    auto* d_v1 = reinterpret_cast<cuDoubleComplex**>(aux + v1_offset);
    auto* d_dims = reinterpret_cast<int64_t*>(aux + dims_offset);
    auto* h_v0 = reinterpret_cast<cuDoubleComplex**>(staged + v0_offset);
    auto* h_v1 = reinterpret_cast<cuDoubleComplex**>(staged + v1_offset);
    auto* h_dims = reinterpret_cast<int64_t*>(staged + dims_offset);

    for (size_t pair_idx = 0; pair_idx < n; ++pair_idx) {
        const int id0 = qubit0_states[pair_idx];
        const int id1 = qubit1_states[pair_idx];
        const int64_t current_dim = state_pool->get_state_dim(id0);
        cuDoubleComplex* v0 = state_pool->get_state_ptr(id0);
        cuDoubleComplex* v1 = state_pool->get_state_ptr(id1);
        if (!v0 || !v1 || current_dim <= 0) {
            throw std::runtime_error("AntiJaynesCummings encountered invalid paired state pointers");
        }

        h_v0[pair_idx] = v0;
        h_v1[pair_idx] = v1;
        h_dims[pair_idx] = current_dim;

        const size_t group_count =
            static_cast<size_t>(current_dim) /
            (static_cast<size_t>(target_mode_dim) * right_stride);
        const size_t total_pairs = group_count * pair_group_span;
        if (hybrid_debug_logging_enabled()) {
            std::cout << "[hybrid-debug] AntiJaynesCummings launching pair[" << pair_idx << "]"
                      << " ids=(" << id0 << "," << id1 << ")"
                      << " ptrs=(" << static_cast<const void*>(v0)
                      << "," << static_cast<const void*>(v1) << ")"
                      << " total_pairs=" << total_pairs
                      << std::endl;
        }
    }

    if (hybrid_debug_logging_enabled()) {
        for (size_t pair_idx = 0; pair_idx < n; ++pair_idx) {
            const int64_t current_dim = h_dims[pair_idx];
            const size_t group_count =
                static_cast<size_t>(current_dim) /
                (static_cast<size_t>(target_mode_dim) * right_stride);
            const size_t total_pairs = group_count * pair_group_span;
            dim3 gd((total_pairs + bd.x - 1) / bd.x);
            apply_ajc_single_pair_kernel<<<gd, bd>>>(
                h_v0[pair_idx],
                h_v1[pair_idx],
                current_dim,
                theta,
                phi,
                target_mode_dim,
                target_mode_right_stride);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            std::cout << "[hybrid-debug] AntiJaynesCummings pair[" << pair_idx << "] complete"
                      << std::endl;
        }
        return;
    }

    CHECK_CUDA(cudaMemcpy(aux, staged, upload_bytes, cudaMemcpyHostToDevice));
    const size_t max_pairs =
        static_cast<size_t>(state_pool->max_total_dim / state_pool->d_trunc) *
        static_cast<size_t>(state_pool->d_trunc - 1);
    dim3 gd((max_pairs + bd.x - 1) / bd.x, n);
    apply_ajc_pointer_batch_kernel<<<gd, bd>>>(
        d_v0, d_v1, d_dims, static_cast<int>(n), theta, phi, target_mode_dim, target_mode_right_stride);
    CHECK_CUDA(cudaGetLastError());
}

void apply_anti_jaynes_cummings(CVStatePool* state_pool,
                                const std::vector<int>& qubit0_states,
                                const std::vector<int>& qubit1_states,
                                double theta,
                                double phi) {
    apply_anti_jaynes_cummings_on_mode(
        state_pool, qubit0_states, qubit1_states, theta, phi, 0, 1);
}

// ==========================================
// 6. SQR (Selective Qubit Rotation)
// ==========================================

__global__ void apply_sqr_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* id0_list,
    const int* id1_list,
    int num_pairs,
    const double* thetas,
    const double* phis
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;
    int id0 = id0_list[batch_idx];
    int id1 = id1_list[batch_idx];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    int64_t current_dim = state_dims[id0];
    if (n >= current_dim) return;

    size_t offset0 = state_offsets[id0];
    size_t offset1 = state_offsets[id1];

    cuDoubleComplex* v0 = &all_states_data[offset0];
    cuDoubleComplex* v1 = &all_states_data[offset1];
    
    // 注意：thetas和phis也是每个状态的数组。如果它们是统一大小，则需要小心索引。
    // 假设它们是针对 max_dim 分配的或者是压缩的。
    // apply_sqr 中分配了 total_dim (max_total_dim)，所以使用 n 作为索引应该是安全的，只要 n < max_total_dim
    // 但这里 n < current_dim <= max_total_dim，所以安全。
    
    double theta = thetas[n];
    double phi = phis[n];
    
    // ... (rest of calculation)
    double cos_t = cos(theta / 2.0);
    double sin_t = sin(theta / 2.0);
    cuDoubleComplex alpha = make_cuDoubleComplex(cos_t, 0.0);
    
    cuDoubleComplex beta = make_cuDoubleComplex(-cos(phi) * sin_t, sin(phi) * sin_t);
    
    cuDoubleComplex c0 = v0[n];
    cuDoubleComplex c1 = v1[n];
    
    v0[n] = cuCadd(cuCmul(alpha, c0), cuCmul(beta, c1));
    
    cuDoubleComplex minus_beta_conj = make_cuDoubleComplex(-cuCreal(beta), cuCimag(beta));
    cuDoubleComplex alpha_conj = make_cuDoubleComplex(cuCreal(alpha), -cuCimag(alpha));
    
    v1[n] = cuCadd(cuCmul(minus_beta_conj, c0), cuCmul(alpha_conj, c1));
}

void apply_sqr(CVStatePool* state_pool,
               const std::vector<int>& qubit0_states,
               const std::vector<int>& qubit1_states,
               const std::vector<double>& thetas,
               const std::vector<double>& phis) {
    if (qubit0_states.size() != qubit1_states.size()) return;
    size_t n_pairs = qubit0_states.size();
    const size_t ids_bytes = 2 * n_pairs * sizeof(int);
    const size_t params_bytes = 2 * static_cast<size_t>(state_pool->max_total_dim) * sizeof(double);
    const size_t total_bytes = ids_bytes + params_bytes;

    char* buf = static_cast<char*>(state_pool->scratch_aux.ensure(total_bytes));
    int* d0 = reinterpret_cast<int*>(buf);
    int* d1 = reinterpret_cast<int*>(buf + n_pairs * sizeof(int));
    double* d_thetas = reinterpret_cast<double*>(buf + ids_bytes);
    double* d_phis = reinterpret_cast<double*>(buf + ids_bytes +
                     static_cast<size_t>(state_pool->max_total_dim) * sizeof(double));
    char* staged = static_cast<char*>(state_pool->host_transfer_staging.ensure(total_bytes));
    std::memcpy(staged, qubit0_states.data(), n_pairs * sizeof(int));
    std::memcpy(staged + n_pairs * sizeof(int),
                qubit1_states.data(),
                n_pairs * sizeof(int));
    std::memcpy(staged + ids_bytes,
                thetas.data(),
                static_cast<size_t>(state_pool->max_total_dim) * sizeof(double));
    std::memcpy(staged + ids_bytes + static_cast<size_t>(state_pool->max_total_dim) * sizeof(double),
                phis.data(),
                static_cast<size_t>(state_pool->max_total_dim) * sizeof(double));
    CHECK_CUDA(cudaMemcpy(buf, staged, total_bytes, cudaMemcpyHostToDevice));

    dim3 bd(256);
    dim3 gd((state_pool->max_total_dim + bd.x - 1)/bd.x, n_pairs);

    apply_sqr_kernel<<<gd, bd>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d0, d1, n_pairs, d_thetas, d_phis
    );
}

// ==========================================
// 7. Utility: Copy States
// ==========================================

__global__ void copy_states_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int64_t* state_dims,
    const int* source_ids,
    const int* dest_ids,
    int num_copies
) {
    int copy_id = blockIdx.y;
    if (copy_id >= num_copies) return;
    int src_id = source_ids[copy_id];
    int dst_id = dest_ids[copy_id];
    
    // 假设源和目标维度一致
    int64_t current_dim = state_dims[src_id];
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= current_dim) return;

    size_t src_offset = state_offsets[src_id];
    size_t dst_offset = state_offsets[dst_id];

    all_states_data[dst_offset + n] = all_states_data[src_offset + n];
}

void copy_states(CVStatePool* state_pool,
                const std::vector<int>& source_ids,
                const std::vector<int>& dest_ids) {
    if (source_ids.size() != dest_ids.size()) return;
    if (source_ids.empty()) return;
    const size_t pair_bytes = 2 * source_ids.size() * sizeof(int);
    char* aux = static_cast<char*>(state_pool->scratch_aux.ensure(pair_bytes));
    int* d_src = reinterpret_cast<int*>(aux);
    int* d_dst = reinterpret_cast<int*>(aux + source_ids.size() * sizeof(int));
    upload_int_pair_vectors(state_pool, aux, source_ids, dest_ids);

    dim3 bd(256);
    dim3 gd((state_pool->max_total_dim + bd.x - 1)/bd.x, source_ids.size());

    copy_states_kernel<<<gd, bd>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_src, d_dst, source_ids.size()
    );
}

// ==========================================
// 8. Echo Controlled Displacement (ECD)
// ==========================================

void apply_echo_controlled_displacement(CVStatePool* state_pool,
                                       const std::vector<int>& controlled_states,
                                       cuDoubleComplex theta) {
    apply_controlled_displacement(state_pool, controlled_states, theta);
}

// ==========================================
// 9. Controlled X/Y Rotation (CRX, CRY)
// ==========================================

__global__ void apply_controlled_x_rotation_kernel(
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
    
    // CRX(theta) = exp[-i theta/2 sigma_x ⊗ n]
    // sigma_x 作用于 qubit，n 作用于 qumode
    double phase = (theta / 2.0) * (double)n;
    double cos_val = cos(phase);
    double sin_val = sin(phase);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_val, sin_val);
    
    psi[n] = cuCmul(psi[n], phase_factor);
}

void apply_controlled_x_rotation(CVStatePool* state_pool,
                                 const std::vector<int>& controlled_states,
                                 double theta) {
    if (controlled_states.empty()) return;
    int* d_ids = state_pool->upload_vector_to_buffer(
        controlled_states, state_pool->scratch_target_ids);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, controlled_states.size());

    apply_controlled_x_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_ids, controlled_states.size(), theta
    );
}

__global__ void apply_controlled_y_rotation_kernel(
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
    
    // CRY(theta) = exp[-i theta/2 sigma_y ⊗ n]
    double phase = (theta / 2.0) * (double)n;
    double cos_val = cos(phase);
    double sin_val = sin(phase);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_val, sin_val);
    
    psi[n] = cuCmul(psi[n], phase_factor);
}

void apply_controlled_y_rotation(CVStatePool* state_pool,
                                 const std::vector<int>& controlled_states,
                                 double theta) {
    if (controlled_states.empty()) return;
    int* d_ids = state_pool->upload_vector_to_buffer(
        controlled_states, state_pool->scratch_target_ids);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, controlled_states.size());

    apply_controlled_y_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        d_ids, controlled_states.size(), theta
    );
}

// Legacy Interface Wrapper (to satisfy existing calls)
void apply_hybrid_control_gate(HDDNode* root_node,
                             CVStatePool* state_pool,
                             HDDNodeManager& node_manager,
                             const std::string& gate_type,
                             cuDoubleComplex param) {
    (void)root_node;
    (void)state_pool;
    (void)node_manager;
    (void)param;
    throw std::runtime_error(
        "apply_hybrid_control_gate is deprecated; dispatch hybrid gates through QuantumCircuit::execute_hybrid_gate for gate type " +
        gate_type);
}
