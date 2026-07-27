#include "state_checksum.h"

#include "cv_state_pool.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__global__ void reduce_states_kernel(
    cuDoubleComplex* const* states,
    const int64_t* dimensions,
    const int* state_ids,
    int state_count,
    double* norm_squared,
    double* checksum) {
    const int state_index = blockIdx.x;
    if (state_index >= state_count) {
        return;
    }

    double local_norm = 0.0;
    double local_checksum = 0.0;
    const cuDoubleComplex* state = states[state_index];
    const int64_t dimension = dimensions[state_index];
    const double state_weight =
        131.0 * (static_cast<double>(state_ids[state_index]) + 1.0);
    for (int64_t index = threadIdx.x; index < dimension; index += blockDim.x) {
        const cuDoubleComplex value = state[index];
        local_norm += value.x * value.x + value.y * value.y;
        local_checksum +=
            (state_weight + static_cast<double>(index) + 1.0) *
            (value.x + 3.0 * value.y);
    }

    __shared__ double norm_partial[256];
    __shared__ double checksum_partial[256];
    norm_partial[threadIdx.x] = local_norm;
    checksum_partial[threadIdx.x] = local_checksum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
            checksum_partial[threadIdx.x] += checksum_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(norm_squared, norm_partial[0]);
        atomicAdd(checksum, checksum_partial[0]);
    }
}

}  // namespace

StateChecksumResult reduce_state_pool_checksum(CVStatePool& state_pool) {
    double norm_squared = 0.0;
    double checksum = 0.0;
    const std::vector<int> active_states = state_pool.get_active_state_ids();
    for (const auto& [device_id, state_ids] :
         state_pool.bucket_state_ids_by_device(active_states)) {
        if (state_ids.empty()) {
            continue;
        }
        check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

        std::vector<cuDoubleComplex*> pointers;
        std::vector<int64_t> dimensions;
        pointers.reserve(state_ids.size());
        dimensions.reserve(state_ids.size());
        for (const int state_id : state_ids) {
            pointers.push_back(state_pool.get_state_ptr(state_id));
            dimensions.push_back(state_pool.get_state_dim(state_id));
        }

        cuDoubleComplex** device_pointers = nullptr;
        int64_t* device_dimensions = nullptr;
        int* device_state_ids = nullptr;
        double* device_output = nullptr;
        check_cuda(
            cudaMalloc(&device_pointers, pointers.size() * sizeof(*device_pointers)),
            "cudaMalloc state pointers");
        check_cuda(
            cudaMalloc(&device_dimensions, dimensions.size() * sizeof(*device_dimensions)),
            "cudaMalloc state dimensions");
        check_cuda(
            cudaMalloc(&device_state_ids, state_ids.size() * sizeof(*device_state_ids)),
            "cudaMalloc state ids");
        check_cuda(cudaMalloc(&device_output, 2 * sizeof(double)), "cudaMalloc checksum output");
        check_cuda(
            cudaMemcpy(
                device_pointers,
                pointers.data(),
                pointers.size() * sizeof(*device_pointers),
                cudaMemcpyHostToDevice),
            "cudaMemcpy state pointers");
        check_cuda(
            cudaMemcpy(
                device_dimensions,
                dimensions.data(),
                dimensions.size() * sizeof(*device_dimensions),
                cudaMemcpyHostToDevice),
            "cudaMemcpy state dimensions");
        check_cuda(
            cudaMemcpy(
                device_state_ids,
                state_ids.data(),
                state_ids.size() * sizeof(*device_state_ids),
                cudaMemcpyHostToDevice),
            "cudaMemcpy state ids");
        check_cuda(cudaMemset(device_output, 0, 2 * sizeof(double)), "cudaMemset checksum output");

        reduce_states_kernel<<<static_cast<unsigned int>(state_ids.size()), 256>>>(
            device_pointers,
            device_dimensions,
            device_state_ids,
            static_cast<int>(state_ids.size()),
            device_output,
            device_output + 1);
        check_cuda(cudaGetLastError(), "reduce_states_kernel launch");
        double host_output[2] = {};
        check_cuda(
            cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost),
            "cudaMemcpy checksum output");
        norm_squared += host_output[0];
        checksum += host_output[1];

        cudaFree(device_output);
        cudaFree(device_state_ids);
        cudaFree(device_dimensions);
        cudaFree(device_pointers);
    }
    return {std::sqrt(norm_squared), checksum};
}
