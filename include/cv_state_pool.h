#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vector>
#include <memory>
#include <atomic>

/**
 * 连续变量状态池 (CV State Pool)
 *
 * 该结构体管理GPU上的连续变量量子态存储，支持动态张量积管理：
 * - 初始状态：每个状态对应单个qumode，大小为D
 * - 纠缠时：动态创建张量积，扩展状态空间
 * - 内存管理：按需分配，避免浪费
 */
struct CVStatePool {
    // 物理存储：动态分配的GPU内存
    cuDoubleComplex* data = nullptr;

    // 状态元数据 - 动态管理
    int d_trunc = 0;        // 截断维数 D (单个qumode的Fock空间维度)
    int capacity = 0;       // 最大支持的独立状态数
    int active_count = 0;   // 当前活跃状态数
    int* free_list = nullptr; // 垃圾回收链表，存储可用状态ID

    // 向后兼容：最大可能维度 (用于GPU内核的静态分配)
    int max_total_dim = 0;  // D^max_qumodes 或内存限制下的最大维度
    int total_dim = 0;      // 别名，用于向后兼容

    // 动态张量积管理
    int* state_dims = nullptr;    // 每个状态的当前维度 [capacity]
    int* state_offsets = nullptr; // 每个状态在data中的偏移量 [capacity]

    // 内存管理
    size_t total_memory_size = 0;  // 已分配的总内存大小
    size_t max_memory_size = 0;    // 最大允许内存大小

    // 主机端副本，用于CPU操作
    std::vector<cuDoubleComplex> host_data;

    /**
     * 构造函数 - 动态张量积版本
     * @param trunc_dim 单个qumode的截断维度D
     * @param max_states 最大状态数量
     * @param num_qumodes 初始qumode数量（用于计算初始状态维度D^num_qumodes）
     * @param max_memory_mb 最大内存限制（MB），0表示无限制
     */
    CVStatePool(int trunc_dim, int max_states, int num_qumodes = 1, size_t max_memory_mb = 0);

    /**
     * 析构函数
     */
    ~CVStatePool();

    /**
     * 分配新的状态ID
     * @return 新分配的状态ID，失败返回-1
     */
    int allocate_state();

    /**
     * 释放状态ID
     * @param state_id 要释放的状态ID
     */
    void free_state(int state_id);

    /**
     * 复制状态数据到GPU
     * @param state_id 目标状态ID
     * @param host_state 主机端状态向量
     */
    void upload_state(int state_id, const std::vector<cuDoubleComplex>& host_state);

    /**
     * 从GPU复制状态数据到主机
     * @param state_id 源状态ID
     * @param host_state 输出：主机端状态向量
     */
    void download_state(int state_id, std::vector<cuDoubleComplex>& host_state) const;

    /**
     * 获取状态向量的GPU指针
     * @param state_id 状态ID
     * @return GPU上的状态向量指针
     */
    cuDoubleComplex* get_state_ptr(int state_id);

    /**
     * 获取状态向量的GPU指针 (const版本)
     * @param state_id 状态ID
     * @return GPU上的状态向量指针
     */
    const cuDoubleComplex* get_state_ptr(int state_id) const;

    /**
     * 检查状态ID是否有效
     * @param state_id 状态ID
     * @return 是否有效
     */
    bool is_valid_state(int state_id) const;

    /**
     * 重置状态池
     * 重置所有分配，恢复为空闲状态
     */
    void reset();

    /**
     * 获取状态的当前维度
     * @param state_id 状态ID
     * @return 状态的当前维度
     */
    int get_state_dim(int state_id) const;

    /**
     * 获取最大总维度 (向后兼容)
     * @return 最大可能的张量积维度
     */
    int get_max_total_dim() const { return max_total_dim; }

    /**
     * 复制状态 (Deep Copy)
     * @param state_id 源状态ID
     * @return 新状态ID
     */
    int duplicate_state(int state_id);

    /**
     * 创建两个状态的张量积
     * @param state1_id 第一个状态ID
     * @param state2_id 第二个状态ID
     * @return 新状态ID，失败返回-1
     */
    int tensor_product(int state1_id, int state2_id);

    /**
     * 检查内存使用情况
     * @return 当前内存使用量（字节）
     */
    size_t get_memory_usage() const { return total_memory_size; }

    /**
     * 获取所有活跃的状态ID
     * @return 活跃状态ID的向量
     */
    std::vector<int> get_active_state_ids() const;

private:
    // 禁用拷贝构造和赋值
    CVStatePool(const CVStatePool&) = delete;
    CVStatePool& operator=(const CVStatePool&) = delete;
};
