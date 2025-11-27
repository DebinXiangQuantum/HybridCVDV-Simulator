#include "cv_state_pool.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstring>

/**
 * CVStatePool 构造函数
 * 初始化GPU内存池和相关元数据
 */
CVStatePool::CVStatePool(int trunc_dim, int max_states)
    : d_trunc(trunc_dim), capacity(max_states), active_count(0) {

    if (d_trunc <= 0 || capacity <= 0) {
        throw std::invalid_argument("截断维度和容量必须为正数");
    }

    // 分配GPU内存用于状态数据
    size_t data_size = static_cast<size_t>(capacity) * d_trunc * sizeof(cuDoubleComplex);
    cudaError_t err = cudaMalloc(&data, data_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法分配GPU内存用于状态池: " + std::string(cudaGetErrorString(err)));
    }

    // 初始化GPU内存为零
    err = cudaMemset(data, 0, data_size);
    if (err != cudaSuccess) {
        cudaFree(data);
        throw std::runtime_error("无法初始化GPU内存: " + std::string(cudaGetErrorString(err)));
    }

    // 分配GPU内存用于空闲列表
    err = cudaMalloc(&free_list, capacity * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(data);
        throw std::runtime_error("无法分配GPU内存用于空闲列表: " + std::string(cudaGetErrorString(err)));
    }

    // 初始化空闲列表：0, 1, 2, ..., capacity-1
    std::vector<int> host_free_list(capacity);
    for (int i = 0; i < capacity; ++i) {
        host_free_list[i] = i;
    }

    err = cudaMemcpy(free_list, host_free_list.data(),
                     capacity * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(data);
        cudaFree(free_list);
        throw std::runtime_error("无法初始化空闲列表: " + std::string(cudaGetErrorString(err)));
    }

    // 初始化主机端数据副本
    host_data.resize(static_cast<size_t>(capacity) * d_trunc, make_cuDoubleComplex(0.0, 0.0));

    std::cout << "CVStatePool 初始化完成: 截断维度=" << d_trunc
              << ", 容量=" << capacity << std::endl;
}

/**
 * CVStatePool 析构函数
 * 释放所有GPU内存
 */
CVStatePool::~CVStatePool() {
    if (data) {
        cudaFree(data);
        data = nullptr;
    }
    if (free_list) {
        cudaFree(free_list);
        free_list = nullptr;
    }

    std::cout << "CVStatePool 销毁完成" << std::endl;
}

/**
 * 分配新的状态ID
 * 从空闲列表中获取一个可用的状态ID
 */
int CVStatePool::allocate_state() {
    if (active_count >= capacity) {
        std::cerr << "警告：状态池已满，无法分配新状态" << std::endl;
        return -1;
    }

    // 从空闲列表获取下一个可用的ID
    int new_state_id;
    cudaError_t err = cudaMemcpy(&new_state_id, free_list + active_count,
                                 sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "无法从空闲列表获取状态ID: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    active_count++;
    return new_state_id;
}

/**
 * 释放状态ID
 * 将状态ID返回到空闲列表中
 */
void CVStatePool::free_state(int state_id) {
    if (!is_valid_state(state_id)) {
        std::cerr << "警告：尝试释放无效的状态ID: " << state_id << std::endl;
        return;
    }

    if (active_count <= 0) {
        std::cerr << "警告：状态池为空，无法释放状态" << std::endl;
        return;
    }

    active_count--;

    // 将释放的状态ID放到空闲列表的末尾
    cudaError_t err = cudaMemcpy(free_list + active_count, &state_id,
                                 sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "无法释放状态ID: " << cudaGetErrorString(err) << std::endl;
        active_count++; // 恢复计数
    }
}

/**
 * 复制状态数据到GPU
 */
void CVStatePool::upload_state(int state_id, const std::vector<cuDoubleComplex>& host_state) {
    if (!is_valid_state(state_id)) {
        throw std::invalid_argument("无效的状态ID: " + std::to_string(state_id));
    }

    if (host_state.size() != static_cast<size_t>(d_trunc)) {
        throw std::invalid_argument("状态向量长度不匹配，期望: " + std::to_string(d_trunc) +
                                  ", 实际: " + std::to_string(host_state.size()));
    }

    cudaError_t err = cudaMemcpy(data + state_id * d_trunc, host_state.data(),
                                 d_trunc * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法上传状态到GPU: " + std::string(cudaGetErrorString(err)));
    }

    // 更新主机端副本
    std::copy(host_state.begin(), host_state.end(),
              host_data.begin() + state_id * d_trunc);
}

/**
 * 从GPU复制状态数据到主机
 */
void CVStatePool::download_state(int state_id, std::vector<cuDoubleComplex>& host_state) const {
    if (!is_valid_state(state_id)) {
        throw std::invalid_argument("无效的状态ID: " + std::to_string(state_id));
    }

    host_state.resize(d_trunc);

    cudaError_t err = cudaMemcpy(host_state.data(), data + state_id * d_trunc,
                                 d_trunc * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法从GPU下载状态: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * 获取状态向量的GPU指针
 */
cuDoubleComplex* CVStatePool::get_state_ptr(int state_id) {
    if (!is_valid_state(state_id)) {
        return nullptr;
    }
    return data + state_id * d_trunc;
}

/**
 * 获取状态向量的GPU指针 (const版本)
 */
const cuDoubleComplex* CVStatePool::get_state_ptr(int state_id) const {
    if (!is_valid_state(state_id)) {
        return nullptr;
    }
    return data + state_id * d_trunc;
}

/**
 * 检查状态ID是否有效
 */
bool CVStatePool::is_valid_state(int state_id) const {
    return state_id >= 0 && state_id < capacity;
}

/**
 * 获取所有活跃的状态ID
 */
std::vector<int> CVStatePool::get_active_state_ids() const {
    std::vector<int> active_ids;
    if (active_count == 0 || !free_list) {
        return active_ids;
    }

    // 从GPU的free_list中读取活跃的状态ID
    // free_list[0] 到 free_list[active_count-1] 是已分配的状态ID
    std::vector<int> host_free_list(active_count);
    cudaError_t err = cudaMemcpy(host_free_list.data(), free_list,
                                 active_count * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "无法从GPU读取活跃状态ID: " << cudaGetErrorString(err) << std::endl;
        return active_ids;
    }

    active_ids = host_free_list;
    return active_ids;
}

/**
 * 重置状态池
 */
void CVStatePool::reset() {
    // 同步所有GPU操作，确保在重置前所有操作完成
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess && sync_err != cudaErrorNotReady) {
        // 如果之前的操作有错误，尝试清除错误状态
        std::cerr << "警告：重置状态池前检测到GPU错误: " << cudaGetErrorString(sync_err) << std::endl;
        // 清除CUDA错误状态，允许后续操作继续
        cudaGetLastError(); // 清除错误标志
    }

    active_count = 0;

    // 重置空闲列表：0, 1, 2, ..., capacity-1
    std::vector<int> host_free_list(capacity);
    for (int i = 0; i < capacity; ++i) {
        host_free_list[i] = i;
    }

    cudaError_t err = cudaMemcpy(free_list, host_free_list.data(),
                                 capacity * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "重置状态池失败: " << cudaGetErrorString(err) << std::endl;
    }

    // 可选：重置数据内存为0
    // cudaMemset(data, 0, static_cast<size_t>(capacity) * d_trunc * sizeof(cuDoubleComplex));
    
    std::cout << "CVStatePool 已重置" << std::endl;
}
