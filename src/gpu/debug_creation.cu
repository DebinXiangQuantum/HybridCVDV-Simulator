#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include "cv_state_pool.h"

// 简单的调试内核，只检查内存访问
__global__ void debug_memory_access(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;
    
    int state_idx = target_indices[batch_id];
    printf("Kernel: batch_id=%d, state_idx=%d, d_trunc=%d, capacity=%d\n", 
           batch_id, state_idx, state_pool->d_trunc, state_pool->capacity);
    
    // 检查指针是否有效
    if (state_pool->data == nullptr) {
        printf("ERROR: state_pool->data is null\n");
        return;
    }
    
    // 检查索引是否在范围内
    if (state_idx < 0 || state_idx >= state_pool->capacity) {
        printf("ERROR: state_idx %d out of range [0, %d)\n", state_idx, state_pool->capacity);
        return;
    }
    
    // 尝试访问第一个元素
    cuDoubleComplex* psi = &state_pool->data[state_idx * state_pool->d_trunc];
    cuDoubleComplex first_val = psi[0];
    printf("SUCCESS: Accessed psi[0] = (%f, %f)\n", cuCreal(first_val), cuCimag(first_val));
}

extern "C" void debug_creation_operator(CVStatePool* state_pool, const int* target_indices, int batch_size) {
    std::cout << "Host: Calling debug kernel with batch_size=" << batch_size << std::endl;
    
    dim3 block_dim(1);
    dim3 grid_dim(1, batch_size);
    
    debug_memory_access<<<grid_dim, block_dim>>>(state_pool, target_indices, batch_size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Debug kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Debug kernel execution failed: " + std::string(cudaGetErrorString(err)));
    }
    
    std::cout << "Host: Debug kernel completed successfully" << std::endl;
}
