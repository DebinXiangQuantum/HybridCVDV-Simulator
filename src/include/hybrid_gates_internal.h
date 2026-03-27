/**
 * @file hybrid_gates_internal.h
 * @brief Shared helpers for hybrid gate CUDA implementations.
 *
 * Contains validation and upload utilities used by both hybrid_gates.cu
 * and hybrid_interaction_gates.cu. Kept in an internal header to avoid
 * polluting public API.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include "cv_state_pool.h"

// Helper for CUDA error checking
#ifndef HYBRID_CHECK_CUDA
#define HYBRID_CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + \
                std::string(cudaGetErrorString(err))); \
        } \
    } while (0)
#endif

namespace hybridcvdv_internal {

inline bool hybrid_debug_logging_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("HYBRIDCVDV_FALLBACK_DEBUG");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

inline void validate_paired_state_lists(const char* gate_name,
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

inline int compute_mode_right_stride(int trunc_dim, int target_qumode, int num_qumodes) {
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

inline void upload_host_bytes(CVStatePool* state_pool,
                       void* device_ptr,
                       const void* host_ptr,
                       size_t bytes) {
    if (bytes == 0) return;
    void* staged = state_pool->host_transfer_staging.ensure(bytes);
    std::memcpy(staged, host_ptr, bytes);
    HYBRID_CHECK_CUDA(cudaMemcpy(device_ptr, staged, bytes, cudaMemcpyHostToDevice));
}

inline void upload_int_pair_vectors(CVStatePool* state_pool,
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
    HYBRID_CHECK_CUDA(cudaMemcpy(device_ptr, staged, total_bytes, cudaMemcpyHostToDevice));
}

}  // namespace hybridcvdv_internal

// Forward declarations for functions defined in hybrid_gates.cu
void apply_controlled_displacement(CVStatePool* state_pool,
                                   const std::vector<int>& controlled_states,
                                   cuDoubleComplex alpha);
void apply_controlled_displacement_on_mode(CVStatePool* state_pool,
                                           const std::vector<int>& controlled_states,
                                           cuDoubleComplex alpha,
                                           int target_qumode,
                                           int num_qumodes);
