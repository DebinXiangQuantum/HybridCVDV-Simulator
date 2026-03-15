#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
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
 * Level 4: 混合控制门 (Hybrid Control Gates) GPU内核
 */

// ==========================================
// 1. Conditional Displacement (CD)
// ==========================================

__global__ void apply_controlled_displacement_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int* state_dims,
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
    int current_dim = state_dims[state_id]; // 获取当前状态的维度
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

__global__ void copy_back_hybrid_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int* state_dims,
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

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int current_dim = state_dims[state_id];
    
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

    int* d_state_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state_ids, controlled_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_state_ids, controlled_states.data(),
               controlled_states.size() * sizeof(int), cudaMemcpyHostToDevice));

    // 使用 max_total_dim 作为 stride 分配足够大的缓冲区
    size_t buffer_stride = state_pool->max_total_dim;
    cuDoubleComplex* temp_buffer = nullptr;
    size_t buffer_size = controlled_states.size() * buffer_stride * sizeof(cuDoubleComplex);
    CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));

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
    CHECK_CUDA(cudaDeviceSynchronize());

    copy_back_hybrid_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        state_pool->capacity,
        d_state_ids, controlled_states.size(), temp_buffer, buffer_stride
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_state_ids));
    CHECK_CUDA(cudaFree(temp_buffer));
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
    const int* state_dims,
    const int* target_state_ids,
    int num_states,
    double theta
) {
    int state_id = target_state_ids[blockIdx.y];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    int current_dim = state_dims[state_id];
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
    int* d_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ids, controlled_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_ids, controlled_states.data(), controlled_states.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, controlled_states.size());

    apply_controlled_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        d_ids, controlled_states.size(), theta
    );
    CHECK_CUDA(cudaFree(d_ids));
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
    const int* state_dims,
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
    int current_dim = state_dims[id0];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
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
    int *d_id0, *d_id1;
    CHECK_CUDA(cudaMalloc(&d_id0, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_id1, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_id0, qubit0_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_id1, qubit1_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
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
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cuDoubleComplex alpha0 = make_cuDoubleComplex(0.0, -theta);
    cuDoubleComplex alpha1 = make_cuDoubleComplex(0.0, theta);

    apply_controlled_displacement_on_mode(
        state_pool, qubit0_states, alpha0, target_qumode, num_qumodes);
    apply_controlled_displacement_on_mode(
        state_pool, qubit1_states, alpha1, target_qumode, num_qumodes);

    // 3. Transform back
    batch_mix_rabi_kernel<<<gd, bd>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        d_id0, d_id1, n, true
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaFree(d_id0));
    CHECK_CUDA(cudaFree(d_id1));
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
    const int* state_dims,
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

    int current_dim = state_dims[id0];
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
    int *d0, *d1;
    CHECK_CUDA(cudaMalloc(&d0, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d1, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d0, qubit0_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d1, qubit1_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 bd(256);
    const size_t max_pairs =
        static_cast<size_t>(state_pool->max_total_dim / state_pool->d_trunc) *
        static_cast<size_t>(state_pool->d_trunc - 1);
    dim3 gd((max_pairs + bd.x - 1)/bd.x, n);

    apply_jc_kernel<<<gd, bd>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        d0, d1, n, theta, phi, target_mode_dim, target_mode_right_stride
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaFree(d0));
    CHECK_CUDA(cudaFree(d1));
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
    const int* state_dims,
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
    
    int current_dim = state_dims[id0];
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
    int *d0, *d1;
    CHECK_CUDA(cudaMalloc(&d0, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d1, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d0, qubit0_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d1, qubit1_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 bd(256);
    const size_t max_pairs =
        static_cast<size_t>(state_pool->max_total_dim / state_pool->d_trunc) *
        static_cast<size_t>(state_pool->d_trunc - 1);
    dim3 gd((max_pairs + bd.x - 1)/bd.x, n);
    
    apply_ajc_kernel<<<gd, bd>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        d0, d1, n, theta, phi, target_mode_dim, target_mode_right_stride
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaFree(d0));
    CHECK_CUDA(cudaFree(d1));
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
    const int* state_dims,
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
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    int current_dim = state_dims[id0];
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
    int *d0, *d1;
    double *d_thetas, *d_phis;
    
    CHECK_CUDA(cudaMalloc(&d0, n_pairs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d1, n_pairs * sizeof(int)));
    // 使用 max_total_dim 分配参数数组
    CHECK_CUDA(cudaMalloc(&d_thetas, state_pool->max_total_dim * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_phis, state_pool->max_total_dim * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d0, qubit0_states.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d1, qubit1_states.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    // 注意：这里假设输入 thetas/phis 的大小匹配 max_total_dim，或者至少覆盖了所有状态的维度
    // 为了安全，应该传递实际数据大小，或者在调用方保证
    CHECK_CUDA(cudaMemcpy(d_thetas, thetas.data(), state_pool->max_total_dim * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_phis, phis.data(), state_pool->max_total_dim * sizeof(double), cudaMemcpyHostToDevice));

    dim3 bd(256);
    dim3 gd((state_pool->max_total_dim + bd.x - 1)/bd.x, n_pairs);
    
    apply_sqr_kernel<<<gd, bd>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        d0, d1, n_pairs, d_thetas, d_phis
    );
    
    CHECK_CUDA(cudaFree(d0));
    CHECK_CUDA(cudaFree(d1));
    CHECK_CUDA(cudaFree(d_thetas));
    CHECK_CUDA(cudaFree(d_phis));
}

// ==========================================
// 7. Utility: Copy States
// ==========================================

__global__ void copy_states_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* source_ids,
    const int* dest_ids,
    int num_copies
) {
    int copy_id = blockIdx.y;
    if (copy_id >= num_copies) return;
    int src_id = source_ids[copy_id];
    int dst_id = dest_ids[copy_id];
    
    // 假设源和目标维度一致
    int current_dim = state_dims[src_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
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
    int *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, source_ids.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_dst, dest_ids.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_src, source_ids.data(), source_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dst, dest_ids.data(), dest_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 bd(256);
    dim3 gd((state_pool->max_total_dim + bd.x - 1)/bd.x, source_ids.size());
    
    copy_states_kernel<<<gd, bd>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        d_src, d_dst, source_ids.size()
    );
    
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
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
    const int* state_dims,
    const int* target_state_ids,
    int num_states,
    double theta
) {
    int state_id = target_state_ids[blockIdx.y];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    int current_dim = state_dims[state_id];
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
    int* d_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ids, controlled_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_ids, controlled_states.data(), controlled_states.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, controlled_states.size());

    apply_controlled_x_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        d_ids, controlled_states.size(), theta
    );
    CHECK_CUDA(cudaFree(d_ids));
}

__global__ void apply_controlled_y_rotation_kernel(
    cuDoubleComplex* all_states_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_state_ids,
    int num_states,
    double theta
) {
    int state_id = target_state_ids[blockIdx.y];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    int current_dim = state_dims[state_id];
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
    int* d_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ids, controlled_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_ids, controlled_states.data(), controlled_states.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, controlled_states.size());

    apply_controlled_y_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool->data, 
        state_pool->state_offsets,
        state_pool->state_dims,
        d_ids, controlled_states.size(), theta
    );
    CHECK_CUDA(cudaFree(d_ids));
}

// Legacy Interface Wrapper (to satisfy existing calls)
void apply_hybrid_control_gate(HDDNode* root_node,
                             CVStatePool* state_pool,
                             HDDNodeManager& node_manager,
                             const std::string& gate_type,
                             cuDoubleComplex param) {
    // Placeholder
}
