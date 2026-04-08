#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cv_state_pool.h"

/**
 * Dense Fock Baseline — 消融实验用
 *
 * 将所有 Fock 门退化为 generic dense D×D matrix-vector multiply，
 * 以对比 L0(diagonal O(D))、L1(ladder O(D))、L3(subspace O(Σk³))
 * 等专用 kernel 的性能增益。
 */

// Generic dense D×D matrix-vector multiply applied to batched states.
// d_dense_matrix must be D×D row-major on device.
void apply_dense_single_mode_gate_gpu(
    CVStatePool* state_pool,
    const int* d_target_indices,
    int batch_size,
    const cuDoubleComplex* d_dense_matrix,
    int matrix_dim,
    int target_qumode,
    int num_qumodes,
    cudaStream_t stream = nullptr,
    bool synchronize = true);

// Generic dense D²×D² tensor contraction for two-mode gates (beam splitter, TMS).
// d_tensor is cutoff^4 row-major on device: T[m1,m2,n1,n2].
void apply_dense_two_mode_gate_gpu(
    CVStatePool* state_pool,
    const int* d_target_indices,
    int batch_size,
    const cuDoubleComplex* d_tensor,
    int cutoff,
    int target_qumode1,
    int target_qumode2,
    int num_qumodes,
    cudaStream_t stream = nullptr,
    bool synchronize = true);
