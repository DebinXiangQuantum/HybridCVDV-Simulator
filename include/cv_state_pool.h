#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuComplex.h>
#else
// CPU-only build: define minimal CUDA types
typedef struct {
    double x, y;
} cuDoubleComplex;

inline cuDoubleComplex make_cuDoubleComplex(double x, double y) {
    return {x, y};
}

inline double cuCreal(cuDoubleComplex c) { return c.x; }
inline double cuCimag(cuDoubleComplex c) { return c.y; }
inline cuDoubleComplex cuCmul(cuDoubleComplex a, cuDoubleComplex b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
inline cuDoubleComplex cuCadd(cuDoubleComplex a, cuDoubleComplex b) {
    return {a.x + b.x, a.y + b.y};
}
inline cuDoubleComplex cuConj(cuDoubleComplex c) {
    return {c.x, -c.y};
}
#endif

#include <vector>
#include <memory>
#include <atomic>

/**
 * 连续变量状态池 (CV State Pool)
 *
 * 该结构体管理GPU上的连续变量量子态存储，采用Structure of Arrays (SoA)布局
 * 以优化内存访问模式和GPU计算性能。
 */
struct CVStatePool {
    // 物理存储：扁平化的大数组
    // 维度: [capacity * d_trunc]
    // d_trunc = d^M (单个CV分支的维度)
    cuDoubleComplex* data = nullptr;

    // 状态元数据
    int d_trunc = 0;        // 截断维数 D (单个qumode的Fock空间维度)
    int capacity = 0;       // 最大支持的独立状态数
    int active_count = 0;   // 当前活跃状态数
    int* free_list = nullptr; // 垃圾回收链表，存储可用状态ID

    // 主机端副本，用于CPU操作
    std::vector<cuDoubleComplex> host_data;

    /**
     * 构造函数
     * @param trunc_dim 截断维度D
     * @param max_states 最大状态数量
     */
    CVStatePool(int trunc_dim, int max_states);

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
     * 获取所有活跃的状态ID
     * @return 活跃状态ID的向量
     */
    std::vector<int> get_active_state_ids() const;

private:
    // 禁用拷贝构造和赋值
    CVStatePool(const CVStatePool&) = delete;
    CVStatePool& operator=(const CVStatePool&) = delete;
};
