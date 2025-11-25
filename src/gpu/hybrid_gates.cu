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

/**
 * Level 4: 混合控制门 (Hybrid Control Gates) GPU内核
 */

// ==========================================
// 1. Conditional Displacement (CD)
// ==========================================

__global__ void apply_controlled_displacement_kernel(
    CVStatePool* state_pool,
    const int* target_state_ids,
    int num_states,
    cuDoubleComplex alpha
) {
    int state_id = target_state_ids[blockIdx.y];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= state_pool->d_trunc) return;

    cuDoubleComplex* psi = &state_pool->data[state_id * state_pool->d_trunc];

    // Simple Displacement Implementation (Bandwidth limited)
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    double alpha_norm_sq = cuCreal(cuCmul(alpha, cuConj(alpha)));
    double exp_factor = exp(-alpha_norm_sq / 2.0);
    int bandwidth = min(10, (int)sqrt(alpha_norm_sq * 4) + 2);

    for (int m = max(0, n - bandwidth); m <= min(state_pool->d_trunc - 1, n + bandwidth); ++m) {
        double coeff = exp_factor;
        if (n >= m) {
            double sqrt_factorial_ratio = 1.0;
            for (int k = m + 1; k <= n; ++k) sqrt_factorial_ratio *= sqrt((double)k);
            coeff *= sqrt_factorial_ratio;
        }
        
        if (abs(n - m) <= bandwidth) {
            cuDoubleComplex power_term = make_cuDoubleComplex(1.0, 0.0);
            int power = n - m;
            if (power > 0) {
                for (int p = 0; p < power; ++p) power_term = cuCmul(power_term, alpha);
            } else if (power < 0) {
                cuDoubleComplex alpha_conj = cuConj(alpha);
                for (int p = 0; p < -power; ++p) power_term = cuCmul(power_term, alpha_conj);
            }

            double laguerre = 1.0;
            if (abs(power) > 0) {
                laguerre = exp(-alpha_norm_sq / 2.0) * pow(alpha_norm_sq, abs(power) / 2.0);
            }
            
            cuDoubleComplex matrix_elem = make_cuDoubleComplex(coeff * laguerre, 0.0);
            matrix_elem = cuCmul(matrix_elem, power_term);
            sum = cuCadd(sum, cuCmul(matrix_elem, psi[m]));
        }
    }
    psi[n] = sum;
}

void apply_controlled_displacement(CVStatePool* state_pool,
                                 const std::vector<int>& controlled_states,
                                 cuDoubleComplex alpha) {
    if (controlled_states.empty()) return;

    int* d_state_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state_ids, controlled_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_state_ids, controlled_states.data(),
               controlled_states.size() * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, controlled_states.size());

    apply_controlled_displacement_kernel<<<grid_dim, block_dim>>>(
        state_pool, d_state_ids, controlled_states.size(), alpha
    );

    CHECK_CUDA(cudaFree(d_state_ids));
}

// ==========================================
// 2. Conditional Rotation (CR) & Parity (CP)
// ==========================================

__global__ void apply_controlled_rotation_kernel(
    CVStatePool* state_pool,
    const int* target_state_ids,
    int num_states,
    double theta // This should be theta (for Branch 1) or -theta (for Branch 0, if handled)
) {
    int state_id = target_state_ids[blockIdx.y];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= state_pool->d_trunc) return;
    
    cuDoubleComplex* psi = &state_pool->data[state_id * state_pool->d_trunc];
    
    // CR(theta) = exp[-i theta/2 sigma_z n]
    // For Q=1 (sigma_z=-1): exp[+i theta/2 n]
    // Assuming this kernel is called for Branch 1
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
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x, controlled_states.size());
    
    apply_controlled_rotation_kernel<<<grid_dim, block_dim>>>(
        state_pool, d_ids, controlled_states.size(), theta
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
    CVStatePool* state_pool,
    const int* id0_list,
    const int* id1_list,
    int num_pairs,
    bool inverse
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;
    
    int id0 = id0_list[batch_idx];
    int id1 = id1_list[batch_idx];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= state_pool->d_trunc) return;
    
    cuDoubleComplex* p0 = &state_pool->data[id0 * state_pool->d_trunc];
    cuDoubleComplex* p1 = &state_pool->data[id1 * state_pool->d_trunc];
    
    cuDoubleComplex v0 = p0[n];
    cuDoubleComplex v1 = p1[n];
    double inv_sqrt2 = 0.7071067811865475;
    
    // X basis transform: |+> = (|0>+|1>)/rt2, |-> = (|0>-|1>)/rt2
    // If inverse: same matrix (H is symmetric and self-inverse)
    
    p0[n] = make_cuDoubleComplex(
        (cuCreal(v0) + cuCreal(v1)) * inv_sqrt2,
        (cuCimag(v0) + cuCimag(v1)) * inv_sqrt2
    );
    p1[n] = make_cuDoubleComplex(
        (cuCreal(v0) - cuCreal(v1)) * inv_sqrt2,
        (cuCimag(v0) - cuCimag(v1)) * inv_sqrt2
    );
}

void apply_rabi_interaction(CVStatePool* state_pool,
                          const std::vector<int>& qubit0_states,
                          const std::vector<int>& qubit1_states,
                          double theta) {
    if (qubit0_states.size() != qubit1_states.size()) throw std::invalid_argument("State lists mismatch");
    if (qubit0_states.empty()) return;
    
    size_t n = qubit0_states.size();
    int *d_id0, *d_id1;
    CHECK_CUDA(cudaMalloc(&d_id0, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_id1, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_id0, qubit0_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_id1, qubit1_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 bd(256);
    dim3 gd((state_pool->d_trunc + bd.x - 1)/bd.x, n);
    
    // 1. Transform to X basis
    batch_mix_rabi_kernel<<<gd, bd>>>(state_pool, d_id0, d_id1, n, false);
    
    // 2. Apply Displacement D(-i theta) to v0 (|0>+|1>) and D(i theta) to v1 (|0>-|1>)
    // Note: In X basis, sigma_x becomes sigma_z. exp[-i sigma_x H] -> exp[-i sigma_z H].
    // H = theta(a^dag + a).
    // Branch 0 (eigenvalue +1 of sigma_x): exp[-i H] = D(-i theta)
    // Branch 1 (eigenvalue -1 of sigma_x): exp[+i H] = D(i theta)
    
    cuDoubleComplex alpha0 = make_cuDoubleComplex(0.0, -theta);
    cuDoubleComplex alpha1 = make_cuDoubleComplex(0.0, theta);
    
    apply_controlled_displacement_kernel<<<gd, bd>>>(state_pool, d_id0, n, alpha0);
    apply_controlled_displacement_kernel<<<gd, bd>>>(state_pool, d_id1, n, alpha1);
    
    // 3. Transform back
    batch_mix_rabi_kernel<<<gd, bd>>>(state_pool, d_id0, d_id1, n, true);
    
    CHECK_CUDA(cudaFree(d_id0));
    CHECK_CUDA(cudaFree(d_id1));
}

// ==========================================
// 4. Jaynes-Cummings (JC)
// ==========================================

__global__ void apply_jc_kernel(
    CVStatePool* state_pool,
    const int* id0_list,
    const int* id1_list,
    int num_pairs,
    double theta,
    double phi
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;
    
    int id0 = id0_list[batch_idx];
    int id1 = id1_list[batch_idx];
    
    // Thread n handles subspace coupling |1, n> and |0, n+1>
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds. We need to access v0[n+1].
    if (n >= state_pool->d_trunc - 1) return;
    
    cuDoubleComplex* v0 = &state_pool->data[id0 * state_pool->d_trunc];
    cuDoubleComplex* v1 = &state_pool->data[id1 * state_pool->d_trunc];
    
    double omega = theta * sqrt((double)(n + 1));
    double cos_w = cos(omega);
    double sin_w = sin(omega);
    
    // |1, n> (v1[n]) and |0, n+1> (v0[n+1])
    cuDoubleComplex c1 = v1[n];
    cuDoubleComplex c0 = v0[n+1];
    
    // |0, n+1> -> cos(w)|0, n+1> - i e^{-i phi} sin(w) |1, n>
    double cos_phi = cos(-phi);
    double sin_phi = sin(-phi);
    cuDoubleComplex ie_miphi_sin = make_cuDoubleComplex(sin_phi * sin_w, cos_phi * -sin_w); // -i * (cos-isin) = -i cos - sin?
    // -i * e^{-i phi} = -i (cos(phi) - i sin(phi)) = -i cos(phi) - sin(phi)
    // Using -phi:
    // -i * (cos(-phi) + i sin(-phi)) = -i cos(phi) + sin(phi) ?
    // Let's be explicit:
    // factor = -i * exp(-i phi) * sin(w)
    // exp(-i phi) = cos(phi) - i sin(phi)
    // -i * exp(...) = -sin(phi) - i cos(phi)
    
    cuDoubleComplex factor01 = make_cuDoubleComplex(-sin(phi) * sin_w, -cos(phi) * sin_w);
    
    // |1, n> -> -i e^{i phi} sin(w) |0, n+1> + cos(w) |1, n>
    // factor = -i * exp(i phi) * sin(w)
    // exp(i phi) = cos(phi) + i sin(phi)
    // -i * exp(...) = sin(phi) - i cos(phi)
    cuDoubleComplex factor10 = make_cuDoubleComplex(sin(phi) * sin_w, -cos(phi) * sin_w);
    
    cuDoubleComplex new_c0 = cuCadd(make_cuDoubleComplex(cos_w * cuCreal(c0), cos_w * cuCimag(c0)),
                                    cuCmul(factor01, c1));
                                    
    cuDoubleComplex new_c1 = cuCadd(cuCmul(factor10, c0),
                                    make_cuDoubleComplex(cos_w * cuCreal(c1), cos_w * cuCimag(c1)));
                                    
    v0[n+1] = new_c0;
    v1[n] = new_c1;
}

void apply_jaynes_cummings(CVStatePool* state_pool,
                         const std::vector<int>& qubit0_states,
                         const std::vector<int>& qubit1_states,
                         double theta, double phi) {
    if (qubit0_states.size() != qubit1_states.size()) return;
    size_t n = qubit0_states.size();
    int *d0, *d1;
    CHECK_CUDA(cudaMalloc(&d0, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d1, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d0, qubit0_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d1, qubit1_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 bd(256);
    dim3 gd((state_pool->d_trunc + bd.x - 1)/bd.x, n);
    
    apply_jc_kernel<<<gd, bd>>>(state_pool, d0, d1, n, theta, phi);
    
    CHECK_CUDA(cudaFree(d0));
    CHECK_CUDA(cudaFree(d1));
}

// ==========================================
// 5. Anti-Jaynes-Cummings (AJC)
// ==========================================

__global__ void apply_ajc_kernel(
    CVStatePool* state_pool,
    const int* id0_list,
    const int* id1_list,
    int num_pairs,
    double theta,
    double phi
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= num_pairs) return;
    int id0 = id0_list[batch_idx];
    int id1 = id1_list[batch_idx];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= state_pool->d_trunc - 1) return;
    
    cuDoubleComplex* v0 = &state_pool->data[id0 * state_pool->d_trunc];
    cuDoubleComplex* v1 = &state_pool->data[id1 * state_pool->d_trunc];
    
    // Couples |0, n> and |1, n+1>
    double omega = theta * sqrt((double)(n + 1));
    double cos_w = cos(omega);
    double sin_w = sin(omega);
    
    cuDoubleComplex c0 = v0[n];
    cuDoubleComplex c1 = v1[n+1];
    
    // |0, n> -> cos(w)|0, n> - i e^{-i phi} sin(w) |1, n+1>
    cuDoubleComplex factor01 = make_cuDoubleComplex(-sin(phi) * sin_w, -cos(phi) * sin_w);
    
    // |1, n+1> -> -i e^{i phi} sin(w) |0, n> + cos(w) |1, n+1>
    cuDoubleComplex factor10 = make_cuDoubleComplex(sin(phi) * sin_w, -cos(phi) * sin_w);
    
    v0[n] = cuCadd(make_cuDoubleComplex(cos_w * cuCreal(c0), cos_w * cuCimag(c0)),
                   cuCmul(factor01, c1));
    v1[n+1] = cuCadd(cuCmul(factor10, c0),
                     make_cuDoubleComplex(cos_w * cuCreal(c1), cos_w * cuCimag(c1)));
}

void apply_anti_jaynes_cummings(CVStatePool* state_pool,
                              const std::vector<int>& qubit0_states,
                              const std::vector<int>& qubit1_states,
                              double theta, double phi) {
    if (qubit0_states.size() != qubit1_states.size()) return;
    size_t n = qubit0_states.size();
    int *d0, *d1;
    CHECK_CUDA(cudaMalloc(&d0, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d1, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d0, qubit0_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d1, qubit1_states.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 bd(256);
    dim3 gd((state_pool->d_trunc + bd.x - 1)/bd.x, n);
    apply_ajc_kernel<<<gd, bd>>>(state_pool, d0, d1, n, theta, phi);
    CHECK_CUDA(cudaFree(d0));
    CHECK_CUDA(cudaFree(d1));
}

// ==========================================
// 6. SQR (Selective Qubit Rotation)
// ==========================================

__global__ void apply_sqr_kernel(
    CVStatePool* state_pool,
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
    if (n >= state_pool->d_trunc) return;
    
    cuDoubleComplex* v0 = &state_pool->data[id0 * state_pool->d_trunc];
    cuDoubleComplex* v1 = &state_pool->data[id1 * state_pool->d_trunc];
    
    double theta = thetas[n];
    double phi = phis[n];
    
    double cos_t = cos(theta / 2.0);
    double sin_t = sin(theta / 2.0);
    cuDoubleComplex alpha = make_cuDoubleComplex(cos_t, 0.0);
    
    // beta = -e^{-i phi} sin(theta/2)
    // - (cos(phi) - i sin(phi)) * sin(t) = -cos(phi)sin(t) + i sin(phi)sin(t)
    cuDoubleComplex beta = make_cuDoubleComplex(-cos(phi) * sin_t, sin(phi) * sin_t);
    
    cuDoubleComplex c0 = v0[n];
    cuDoubleComplex c1 = v1[n];
    
    // v0 = alpha c0 + beta c1
    // v1 = -beta* c0 + alpha* c1
    
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
    CHECK_CUDA(cudaMalloc(&d_thetas, state_pool->d_trunc * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_phis, state_pool->d_trunc * sizeof(double)));
    
    CHECK_CUDA(cudaMemcpy(d0, qubit0_states.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d1, qubit1_states.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_thetas, thetas.data(), state_pool->d_trunc * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_phis, phis.data(), state_pool->d_trunc * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 bd(256);
    dim3 gd((state_pool->d_trunc + bd.x - 1)/bd.x, n_pairs);
    apply_sqr_kernel<<<gd, bd>>>(state_pool, d0, d1, n_pairs, d_thetas, d_phis);
    
    CHECK_CUDA(cudaFree(d0));
    CHECK_CUDA(cudaFree(d1));
    CHECK_CUDA(cudaFree(d_thetas));
    CHECK_CUDA(cudaFree(d_phis));
}

// ==========================================
// 7. Utility: Copy States
// ==========================================

__global__ void copy_states_kernel(
    CVStatePool* state_pool,
    const int* source_ids,
    const int* dest_ids,
    int num_copies
) {
    int copy_id = blockIdx.y;
    if (copy_id >= num_copies) return;
    int src_id = source_ids[copy_id];
    int dst_id = dest_ids[copy_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= state_pool->d_trunc) return;
    
    state_pool->data[dst_id * state_pool->d_trunc + n] = 
        state_pool->data[src_id * state_pool->d_trunc + n];
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
    dim3 gd((state_pool->d_trunc + bd.x - 1)/bd.x, source_ids.size());
    copy_states_kernel<<<gd, bd>>>(state_pool, d_src, d_dst, source_ids.size());
    
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
}

// Legacy Interface Wrapper (to satisfy existing calls)
void apply_hybrid_control_gate(HDDNode* root_node,
                             CVStatePool* state_pool,
                             HDDNodeManager& node_manager,
                             const std::string& gate_type,
                             cuDoubleComplex param) {
    // Placeholder
}
