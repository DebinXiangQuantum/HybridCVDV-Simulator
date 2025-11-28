#include "cv_state_pool.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstring>

/**
 * CVStatePool 构造函数 - 动态张量积版本
 * 初始化GPU内存池，支持动态内存分配
 */
CVStatePool::CVStatePool(int trunc_dim, int max_states, int num_qumodes, size_t max_memory_mb)
    : d_trunc(trunc_dim), capacity(max_states), active_count(0),
      max_total_dim(1), total_dim(1), total_memory_size(0), max_memory_size(max_memory_mb * 1024ULL * 1024ULL) {

    // 检查CUDA设备是否可用
    cudaError_t device_check = cudaGetDeviceCount(nullptr);
    if (device_check != cudaSuccess) {
        throw std::runtime_error("CUDA设备不可用: " + std::string(cudaGetErrorString(device_check)));
    }
    
    // 清除之前的CUDA错误状态
    cudaGetLastError();

    // 计算总维度：D^num_qumodes
    for (int i = 0; i < num_qumodes; ++i) {
        max_total_dim *= d_trunc;
    }
    total_dim = max_total_dim; // 向后兼容

    if (d_trunc <= 0 || capacity <= 0) {
        throw std::invalid_argument("截断维度和容量必须为正数");
    }

    // 分配GPU内存用于状态数据 - 初始分配全尺寸以支持QuantumCircuit的期望
    size_t initial_data_size = static_cast<size_t>(capacity) * max_total_dim * sizeof(cuDoubleComplex);
    cudaError_t err = cudaMalloc(&data, initial_data_size);
    if (err != cudaSuccess) {
        // 清除错误状态
        cudaGetLastError();
        throw std::runtime_error("无法分配GPU内存用于状态池: " + std::string(cudaGetErrorString(err)));
    }
    total_memory_size = initial_data_size;

    // 初始化GPU内存为零
    err = cudaMemset(data, 0, initial_data_size);
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

    // 分配GPU内存用于状态维度信息
    err = cudaMalloc(&state_dims, capacity * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(data);
        cudaFree(free_list);
        throw std::runtime_error("无法分配GPU内存用于状态维度: " + std::string(cudaGetErrorString(err)));
    }

    // 分配GPU内存用于状态偏移量
    err = cudaMalloc(&state_offsets, capacity * sizeof(size_t));
    if (err != cudaSuccess) {
        cudaFree(data);
        cudaFree(free_list);
        cudaFree(state_dims);
        throw std::runtime_error("无法分配GPU内存用于状态偏移量: " + std::string(cudaGetErrorString(err)));
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
        cudaFree(state_dims);
        cudaFree(state_offsets);
        throw std::runtime_error("无法初始化空闲列表: " + std::string(cudaGetErrorString(err)));
    }

    // 初始化状态维度和偏移量 - 初始每个状态为max_total_dim (full size)
    std::vector<int> host_state_dims(capacity, max_total_dim);
    std::vector<size_t> host_state_offsets(capacity);
    for (int i = 0; i < capacity; ++i) {
        host_state_offsets[i] = static_cast<size_t>(i) * max_total_dim;
    }

    err = cudaMemcpy(state_dims, host_state_dims.data(),
                     capacity * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(data);
        cudaFree(free_list);
        cudaFree(state_dims);
        cudaFree(state_offsets);
        throw std::runtime_error("无法初始化状态维度: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(state_offsets, host_state_offsets.data(),
                     capacity * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(data);
        cudaFree(free_list);
        cudaFree(state_dims);
        cudaFree(state_offsets);
        throw std::runtime_error("无法初始化状态偏移量: " + std::string(cudaGetErrorString(err)));
    }

    // 初始化主机端数据副本
    host_data.resize(static_cast<size_t>(capacity) * max_total_dim, make_cuDoubleComplex(0.0, 0.0));

    std::cout << "CVStatePool 初始化完成: 单个qumode截断维度=" << d_trunc
              << ", 初始总维度=" << max_total_dim
              << ", 容量=" << capacity
              << ", 初始内存=" << (initial_data_size / (1024.0 * 1024.0)) << " MB" << std::endl;
}

/**
 * CVStatePool 析构函数
 * 释放所有GPU内存
 */
CVStatePool::~CVStatePool() {
    // 清除之前的CUDA错误状态，避免影响释放操作
    cudaGetLastError();
    
    // 安全释放GPU内存，忽略错误（设备可能已不可用）
    if (data) {
        cudaError_t err = cudaFree(data);
        if (err != cudaSuccess && err != cudaErrorCudaNotInitialized) {
            // 只在非初始化错误时输出警告
            std::cerr << "警告：释放状态池数据内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        data = nullptr;
    }
    if (free_list) {
        cudaError_t err = cudaFree(free_list);
        if (err != cudaSuccess && err != cudaErrorCudaNotInitialized) {
            std::cerr << "警告：释放空闲列表内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        free_list = nullptr;
    }
    if (state_dims) {
        cudaError_t err = cudaFree(state_dims);
        if (err != cudaSuccess && err != cudaErrorCudaNotInitialized) {
            std::cerr << "警告：释放状态维度内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        state_dims = nullptr;
    }
    if (state_offsets) {
        cudaError_t err = cudaFree(state_offsets);
        if (err != cudaSuccess && err != cudaErrorCudaNotInitialized) {
            std::cerr << "警告：释放状态偏移量内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        state_offsets = nullptr;
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

    int state_dim = get_state_dim(state_id);
    if (host_state.size() != static_cast<size_t>(state_dim)) {
        throw std::invalid_argument("状态向量长度不匹配，期望: " + std::to_string(state_dim) +
                                  ", 实际: " + std::to_string(host_state.size()));
    }

    // 获取状态偏移量
    size_t offset;
    cudaError_t err = cudaMemcpy(&offset, state_offsets + state_id,
                                 sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法读取状态偏移量: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(data + offset, host_state.data(),
                     state_dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法上传状态到GPU: " + std::string(cudaGetErrorString(err)));
    }

    // 更新主机端副本（简化版，实际应该动态管理）
    if (host_data.size() < static_cast<size_t>(state_id + 1) * state_dim) {
        host_data.resize(static_cast<size_t>(capacity) * d_trunc, make_cuDoubleComplex(0.0, 0.0));
    }
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

    int state_dim = get_state_dim(state_id);
    host_state.resize(state_dim);

    // 获取状态偏移量
    size_t offset;
    cudaError_t err = cudaMemcpy(&offset, state_offsets + state_id,
                                 sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法读取状态偏移量: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(host_state.data(), data + offset,
                     state_dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
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

    size_t offset;
    cudaError_t err = cudaMemcpy(&offset, state_offsets + state_id,
                                 sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return data + offset;
}

/**
 * 获取状态向量的GPU指针 (const版本)
 */
const cuDoubleComplex* CVStatePool::get_state_ptr(int state_id) const {
    if (!is_valid_state(state_id)) {
        return nullptr;
    }

    size_t offset;
    cudaError_t err = cudaMemcpy(&offset, state_offsets + state_id,
                                 sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return data + offset;
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
    // cudaMemset(data, 0, static_cast<size_t>(capacity) * total_dim * sizeof(cuDoubleComplex));
    
    std::cout << "CVStatePool 已重置" << std::endl;
}

/**
 * 获取状态的当前维度
 */
int CVStatePool::get_state_dim(int state_id) const {
    if (!is_valid_state(state_id)) {
        return 0;
    }

    int dim;
    cudaError_t err = cudaMemcpy(&dim, state_dims + state_id,
                                 sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "无法读取状态维度: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return dim;
}

/**
 * 创建两个状态的张量积
 */
int CVStatePool::tensor_product(int state1_id, int state2_id) {
    if (!is_valid_state(state1_id) || !is_valid_state(state2_id)) {
        std::cerr << "无效的状态ID: " << state1_id << ", " << state2_id << std::endl;
        return -1;
    }

    int dim1 = get_state_dim(state1_id);
    int dim2 = get_state_dim(state2_id);
    int new_dim = dim1 * dim2;

    // 检查内存限制
    size_t required_memory = static_cast<size_t>(new_dim) * sizeof(cuDoubleComplex);
    if (max_memory_size > 0 && total_memory_size + required_memory > max_memory_size) {
        std::cerr << "内存不足，无法创建张量积。需要: "
                  << (required_memory / (1024.0 * 1024.0)) << " MB, "
                  << "可用: " << ((max_memory_size - total_memory_size) / (1024.0 * 1024.0)) << " MB" << std::endl;
        return -1;
    }

    // 分配新的状态ID
    int new_state_id = allocate_state();
    if (new_state_id == -1) {
        std::cerr << "无法分配新状态用于张量积" << std::endl;
        return -1;
    }

    // 扩展内存以容纳新的张量积状态
    size_t new_offset = total_memory_size / sizeof(cuDoubleComplex); // 元素偏移量
    size_t new_total_memory = total_memory_size + required_memory;

    cuDoubleComplex* new_data = nullptr;
    cudaError_t err = cudaMalloc(&new_data, new_total_memory);
    if (err != cudaSuccess) {
        free_state(new_state_id);
        std::cerr << "无法分配内存用于张量积: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 复制现有数据到新内存
    err = cudaMemcpy(new_data, data, total_memory_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaError_t free_err = cudaFree(new_data);
        if (free_err != cudaSuccess) {
            std::cerr << "警告：释放新分配的内存失败: " << cudaGetErrorString(free_err) << std::endl;
        }
        free_state(new_state_id);
        std::cerr << "无法复制现有数据: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 释放旧内存，更新指针
    cudaError_t free_err = cudaFree(data);
    if (free_err != cudaSuccess) {
        std::cerr << "警告：释放旧内存失败: " << cudaGetErrorString(free_err) << std::endl;
        // 继续执行，因为新内存已分配成功
    }
    data = new_data;
    total_memory_size = new_total_memory;

    // 更新状态维度和偏移量
    err = cudaMemcpy(state_dims + new_state_id, &new_dim,
                     sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "无法更新状态维度: " << cudaGetErrorString(err) << std::endl;
        free_state(new_state_id);
        return -1;
    }

    size_t new_offset_size_t = static_cast<size_t>(new_offset);
    err = cudaMemcpy(state_offsets + new_state_id, &new_offset,
                     sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "无法更新状态偏移量: " << cudaGetErrorString(err) << std::endl;
        free_state(new_state_id);
        return -1;
    }

    // 执行张量积计算
    // 这里需要实现张量积的GPU内核，但暂时用简单的CPU实现作为占位符
    // 实际实现需要一个专门的GPU内核来计算张量积

    std::cout << "创建张量积: 状态" << state1_id << " (dim=" << dim1 << ") ⊗ 状态"
              << state2_id << " (dim=" << dim2 << ") -> 状态" << new_state_id
              << " (dim=" << new_dim << ")" << std::endl;

    return new_state_id;
}
