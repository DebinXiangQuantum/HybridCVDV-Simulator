#pragma once

#include <cuda_runtime.h>
#include "gaussian_state.h"

/**
 * BatchedSymplecticUpdate
 * 
 * Applies S*Sigma*S^T and S*d + d_g to a batch of Gaussian states.
 * 
 * @param pool The state pool containing d and sigma
 * @param state_ids Indices of states to update
 * @param batch_size Number of states
 * @param d_S Symplectic matrix on GPU (2M x 2M)
 * @param d_dg Displacement vector on GPU (2M)
 */
void apply_batched_symplectic_update(
    GaussianStatePool* pool,
    const int* state_ids,
    int batch_size,
    const double* d_S,
    const double* d_dg,
    double* d_old_buffer = nullptr,
    double* d_temp_buffer = nullptr,
    cudaStream_t stream = nullptr,
    bool synchronize = true
);
