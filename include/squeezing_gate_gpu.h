#pragma once

#include "cv_state_pool.h"

/**
 * GPU版本的挤压门。
 * 参数变化时在 device 侧生成并缓存单模 squeezing 矩阵，
 * 再直接对多模 Fock 态目标 mode 做 GPU 应用。
 */
void apply_squeezing_gate_gpu(
    CVStatePool* pool,
    const int* target_indices,
    int batch_size,
    double r,
    double theta,
    int target_qumode = 0,
    int num_qumodes = 1,
    cudaStream_t stream = nullptr,
    bool synchronize = true
);

/**
 * 清理 device 侧挤压门缓存。
 */
void clear_squeezing_cache();
