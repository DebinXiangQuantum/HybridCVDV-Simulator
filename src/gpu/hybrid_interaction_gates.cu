/**
 * @file hybrid_interaction_gates.cu
 * @brief Jaynes-Cummings, Anti-Jaynes-Cummings, SQR, and other
 *        hybrid interaction gate CUDA kernels.
 *
 * Split from hybrid_gates.cu for maintainability.
 * Uses shared helpers from hybrid_gates_internal.h.
 */

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
#include "hybrid_gates_internal.h"

using namespace hybridcvdv_internal;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)     do {         cudaError_t err = call;         if (err != cudaSuccess) {             throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));         }     } while (0)
#endif

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
