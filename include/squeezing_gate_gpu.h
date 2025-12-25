#pragma once

#include "cv_state_pool.h"

/**
 * GPU版本的挤压门（带缓存优化）
 * 
 * @param pool 状态池
 * @param target_indices 设备端指针，指向目标状态ID数组
 * @param batch_size 批大小
 * @param r 挤压幅度
 * @param theta 挤压角度
 */
void apply_squeezing_gate_gpu(
    CVStatePool* pool,
    const int* target_indices,
    int batch_size,
    double r,
    double theta
);

/**
 * 清理挤压门缓存
 * 在程序结束时调用以释放GPU内存
 */
void clear_squeezing_cache();
