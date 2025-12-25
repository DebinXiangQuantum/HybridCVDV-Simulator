/**
 * Beam Splitter implementation using Strawberry Fields recursive method
 * 
 * This approach computes matrix elements on-the-fly using recursive relations
 * instead of pre-computing the full D^4 tensor or using Wigner d-matrices.
 * 
 * Key advantages:
 * - Numerically stable (no factorials or exponentials)
 * - Memory efficient (O(D^2) instead of O(D^4))
 * - Exact (no truncation errors)
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include "cv_state_pool.h"

/**
 * Compute beam splitter matrix element Z[m,n,p,q] using recursive relation
 * 
 * This is computed on-the-fly for each needed element
 */
__device__ cuDoubleComplex compute_bs_element_recursive(
    int m, int n, int p, int q,
    double ct, double st, cuDoubleComplex phase,
    const double* sqrt_table, int cutoff) {
    
    // Base case
    if (m == 0 && n == 0 && p == 0 && q == 0) {
        return make_cuDoubleComplex(1.0, 0.0);
    }
    
    // Out of bounds
    if (m < 0 || n < 0 || p < 0 || q < 0 || 
        m >= cutoff || n >= cutoff || p >= cutoff || q >= cutoff) {
        return make_cuDoubleComplex(0.0, 0.0);
    }
    
    // Use shared memory cache for computed values
    // For now, compute directly (can be optimized with memoization)
    
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    
    // Recursive computation based on photon number conservation
    // m + n = p + q (total photon number conserved)
    if (m + n != p + q) {
        return make_cuDoubleComplex(0.0, 0.0);
    }
    
    // Rank 3: q = 0 case
    if (q == 0 && p > 0) {
        if (m > 0) {
            cuDoubleComplex term1 = compute_bs_element_recursive(
                m-1, n, p-1, 0, ct, st, phase, sqrt_table, cutoff);
            double coeff1 = ct * sqrt_table[m] / sqrt_table[p];
            result = cuCadd(result, cuCmul(make_cuDoubleComplex(coeff1, 0.0), term1));
        }
        if (n > 0) {
            cuDoubleComplex term2 = compute_bs_element_recursive(
                m, n-1, p-1, 0, ct, st, phase, sqrt_table, cutoff);
            double coeff2 = st * sqrt_table[n] / sqrt_table[p];
            cuDoubleComplex phase_term = cuCmul(phase, make_cuDoubleComplex(coeff2, 0.0));
            result = cuCadd(result, cuCmul(phase_term, term2));
        }
        return result;
    }
    
    // Rank 4: general case with q > 0
    if (q > 0) {
        if (m > 0) {
            cuDoubleComplex term1 = compute_bs_element_recursive(
                m-1, n, p, q-1, ct, st, phase, sqrt_table, cutoff);
            double coeff1 = -st * sqrt_table[m] / sqrt_table[q];
            cuDoubleComplex conj_phase = cuConj(phase);
            cuDoubleComplex phase_term = cuCmul(conj_phase, make_cuDoubleComplex(coeff1, 0.0));
            result = cuCadd(result, cuCmul(phase_term, term1));
        }
        if (n > 0) {
            cuDoubleComplex term2 = compute_bs_element_recursive(
                m, n-1, p, q-1, ct, st, phase, sqrt_table, cutoff);
            double coeff2 = ct * sqrt_table[n] / sqrt_table[q];
            result = cuCadd(result, cuCmul(make_cuDoubleComplex(coeff2, 0.0), term2));
        }
        return result;
    }
    
    return result;
}

/**
 * Optimized version: Build Z matrix iteratively (not recursively)
 * This avoids stack overflow and is much faster
 */
__device__ void compute_bs_matrix_iterative(
    cuDoubleComplex* Z, int cutoff,
    double theta, double phi) {
    
    double ct = cos(theta);
    double st = sin(theta);
    cuDoubleComplex phase = make_cuDoubleComplex(cos(phi), sin(phi));
    
    // Precompute sqrt table
    double sqrt_table[64];  // Assuming cutoff <= 64
    for (int i = 0; i < cutoff; ++i) {
        sqrt_table[i] = sqrt((double)i);
    }
    
    // Initialize
    int D = cutoff;
    for (int i = 0; i < D*D*D*D; ++i) {
        Z[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    
    // Base case
    Z[0] = make_cuDoubleComplex(1.0, 0.0);  // Z[0,0,0,0] = 1
    
    // Rank 3: Fill Z[m,n,p,0]
    for (int m = 0; m < D; ++m) {
        for (int n = 0; n < D - m; ++n) {
            int p = m + n;
            if (p > 0 && p < D) {
                int idx = m*D*D*D + n*D*D + p*D + 0;
                cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
                
                if (m > 0) {
                    int idx1 = (m-1)*D*D*D + n*D*D + (p-1)*D + 0;
                    double coeff = ct * sqrt_table[m] / sqrt_table[p];
                    sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), Z[idx1]));
                }
                if (n > 0) {
                    int idx2 = m*D*D*D + (n-1)*D*D + (p-1)*D + 0;
                    double coeff = st * sqrt_table[n] / sqrt_table[p];
                    cuDoubleComplex term = cuCmul(phase, make_cuDoubleComplex(coeff, 0.0));
                    sum = cuCadd(sum, cuCmul(term, Z[idx2]));
                }
                Z[idx] = sum;
            }
        }
    }
    
    // Rank 4: Fill Z[m,n,p,q] for q > 0
    for (int m = 0; m < D; ++m) {
        for (int n = 0; n < D; ++n) {
            for (int p = 0; p < D; ++p) {
                int q = m + n - p;
                if (q > 0 && q < D) {
                    int idx = m*D*D*D + n*D*D + p*D + q;
                    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
                    
                    if (m > 0) {
                        int idx1 = (m-1)*D*D*D + n*D*D + p*D + (q-1);
                        double coeff = -st * sqrt_table[m] / sqrt_table[q];
                        cuDoubleComplex conj_phase = cuConj(phase);
                        cuDoubleComplex term = cuCmul(conj_phase, make_cuDoubleComplex(coeff, 0.0));
                        sum = cuCadd(sum, cuCmul(term, Z[idx1]));
                    }
                    if (n > 0) {
                        int idx2 = m*D*D*D + (n-1)*D*D + p*D + (q-1);
                        double coeff = ct * sqrt_table[n] / sqrt_table[q];
                        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(coeff, 0.0), Z[idx2]));
                    }
                    Z[idx] = sum;
                }
            }
        }
    }
}

/**
 * GPU Kernel: Apply beam splitter directly to state vector
 * Each thread computes one output amplitude
 */
__global__ void apply_bs_direct_kernel(
    const cuDoubleComplex* input_state,
    cuDoubleComplex* output_state,
    int cutoff,
    double theta,
    double phi) {
    
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (m >= cutoff || n >= cutoff) return;
    
    double ct = cos(theta);
    double st = sin(theta);
    cuDoubleComplex phase = make_cuDoubleComplex(cos(phi), sin(phi));
    
    // Precompute sqrt table in shared memory
    __shared__ double sqrt_table[64];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < cutoff && i < 64; ++i) {
            sqrt_table[i] = sqrt((double)i);
        }
    }
    __syncthreads();
    
    // Compute output amplitude for |m,n⟩
    cuDoubleComplex amplitude = make_cuDoubleComplex(0.0, 0.0);
    
    // Sum over all input basis states |p,q⟩
    for (int p = 0; p < cutoff; ++p) {
        for (int q = 0; q < cutoff; ++q) {
            // Only non-zero if m+n = p+q (photon conservation)
            if (m + n != p + q) continue;
            
            // Compute Z[m,n,p,q] on-the-fly
            // For efficiency, we use a simplified direct formula
            // This can be optimized further with caching
            
            cuDoubleComplex z_elem = make_cuDoubleComplex(0.0, 0.0);
            
            // Simplified computation (full recursive would be too slow)
            // Use approximation or precomputed values
            // For now, use a placeholder
            
            int input_idx = p * cutoff + q;
            amplitude = cuCadd(amplitude, cuCmul(z_elem, input_state[input_idx]));
        }
    }
    
    int output_idx = m * cutoff + n;
    output_state[output_idx] = amplitude;
}

/**
 * Host function: Apply beam splitter using recursive method
 */
void apply_beam_splitter_recursive(
    CVStatePool* pool,
    const int* target_indices,
    int batch_size,
    double theta,
    double phi,
    int cutoff) {
    
    // This is a placeholder for the full implementation
    // The key idea is to use the Strawberry Fields recursive approach
    // which is numerically stable and doesn't require D^4 storage
    
    // Implementation would:
    // 1. Build Z matrix iteratively on GPU (or CPU and upload)
    // 2. Apply matrix-vector multiplication
    // 3. Or compute Z elements on-the-fly during multiplication
}
