#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <vector>
#include <queue>
#include <unordered_map>
#include <memory>
#include <atomic>

// 前向声明
class CVStatePool;
class HDDNodeManager;
class HDDNode;

/**
 * GPU内存块结构体
 */
struct GPUMemoryBlock {
    void* ptr = nullptr;              // GPU内存指针
    size_t size = 0;                  // 块大小 (字节)
    bool is_free = true;              // 是否空闲
    size_t allocation_id = 0;         // 分配ID (用于调试)

    GPUMemoryBlock(void* p = nullptr, size_t s = 0, size_t id = 0)
        : ptr(p), size(s), allocation_id(id) {}
};

/**
 * GPU内存池
 * 管理GPU内存分配，避免频繁的cudaMalloc/cudaFree调用
 */
class GPUMemoryPool {
private:
    std::vector<GPUMemoryBlock> blocks_;        // 内存块列表
    std::queue<size_t> free_blocks_;            // 空闲块索引队列
    std::unordered_map<size_t, size_t> alloc_map_; // 分配ID到块索引的映射

    size_t total_size_ = 0;                     // 总内存大小
    size_t block_size_ = 0;                     // 块大小
    size_t max_blocks_ = 0;                     // 最大块数量

    std::atomic<size_t> next_allocation_id_{1}; // 下一个分配ID

    // 统计信息
    size_t total_allocations_ = 0;
    size_t total_deallocations_ = 0;
    size_t peak_memory_usage_ = 0;

public:
    /**
     * 构造函数
     * @param total_size 总内存大小 (字节)
     * @param block_size 块大小 (字节)
     */
    GPUMemoryPool(size_t total_size, size_t block_size);

    /**
     * 析构函数
     */
    ~GPUMemoryPool();

    /**
     * 分配内存
     * @param size 请求的内存大小
     * @return 分配的GPU内存指针，失败返回nullptr
     */
    void* allocate(size_t size);

    /**
     * 释放内存
     * @param ptr 要释放的内存指针
     */
    void deallocate(void* ptr);

    /**
     * 获取内存使用统计
     */
    struct MemoryStats {
        size_t total_size;
        size_t used_size;
        size_t free_size;
        size_t num_blocks;
        size_t num_free_blocks;
        size_t peak_usage;
        double utilization;  // 利用率 (0.0-1.0)
    };
    MemoryStats get_stats() const;

    /**
     * 整理内存 (合并相邻空闲块)
     */
    void defragment();

    /**
     * 重置内存池
     */
    void reset();

private:
    /**
     * 查找合适的空闲块
     * 使用首次适应算法
     */
    size_t find_free_block(size_t required_size) const;

    /**
     * 合并相邻空闲块
     */
    void coalesce_blocks();

    /**
     * 分割内存块
     */
    bool split_block(size_t block_idx, size_t required_size);
};

/**
 * 垃圾回收器
 * 负责CV状态池的垃圾回收和HDD节点清理
 */
class GarbageCollector {
private:
    CVStatePool* state_pool_ = nullptr;
    HDDNodeManager* node_manager_ = nullptr;

    double fidelity_threshold_ = 0.999;  // 保真度阈值
    size_t max_states_ = 1000;           // 最大状态数量

    // 统计信息
    size_t total_collections_ = 0;
    size_t states_freed_ = 0;
    size_t nodes_cleaned_ = 0;

public:
    /**
     * 设置关联的池管理器
     */
    void set_state_pool(CVStatePool* pool) { state_pool_ = pool; }
    void set_node_manager(HDDNodeManager* manager) { node_manager_ = manager; }

    /**
     * 执行垃圾回收
     * @param force 是否强制执行 (忽略阈值)
     */
    void collect(bool force = false);

    /**
     * 检查是否需要垃圾回收
     */
    bool should_collect() const;

    /**
     * 去重相似状态
     * 将保真度足够高的状态合并
     */
    void deduplicate_states();

    /**
     * 清理未使用的HDD节点
     */
    void clean_unused_nodes();

    /**
     * 获取垃圾回收统计
     */
    struct GCStats {
        size_t total_collections;
        size_t states_freed;
        size_t nodes_cleaned;
        size_t current_active_states;
        double average_fidelity;
    };
    GCStats get_stats() const;

    /**
     * 配置参数
     */
    void set_fidelity_threshold(double threshold) { fidelity_threshold_ = threshold; }
    void set_max_states(size_t max_states) { max_states_ = max_states; }

private:
    /**
     * 计算两个状态之间的保真度
     * |<ψ₁|ψ₂>|²
     */
    double calculate_fidelity(int state_id1, int state_id2) const;

    /**
     * 合并两个相似状态
     */
    bool merge_similar_states(int state_id1, int state_id2);

    /**
     * 深度优先遍历HDD，更新状态引用
     */
    void update_hdd_references(HDDNode* node, int old_state_id, int new_state_id);
};

/**
 * 内存管理器
 * 统一管理所有内存资源
 */
class MemoryManager {
private:
    GPUMemoryPool gpu_memory_pool_;
    GarbageCollector garbage_collector_;

    CVStatePool* state_pool_ = nullptr;
    HDDNodeManager* node_manager_ = nullptr;

    bool auto_gc_enabled_ = true;
    size_t gc_threshold_ = 80;  // 当利用率超过80%时触发GC

public:
    /**
     * 构造函数
     * @param gpu_memory_size GPU内存池大小
     * @param block_size 内存块大小
     */
    MemoryManager(size_t gpu_memory_size = 1024 * 1024 * 1024, // 1GB
                  size_t block_size = 64 * 1024 * 1024);       // 64MB

    /**
     * 初始化内存管理器
     */
    void initialize(CVStatePool* state_pool, HDDNodeManager* node_manager);

    /**
     * 分配GPU内存
     */
    void* allocate_gpu_memory(size_t size);

    /**
     * 释放GPU内存
     */
    void deallocate_gpu_memory(void* ptr);

    /**
     * 执行完整的内存管理周期
     */
    void memory_management_cycle();

    /**
     * 强制垃圾回收
     */
    void force_garbage_collection();

    /**
     * 获取内存统计信息
     */
    struct OverallStats {
        GPUMemoryPool::MemoryStats gpu_stats;
        GarbageCollector::GCStats gc_stats;
        bool auto_gc_enabled;
        size_t gc_threshold;
    };
    OverallStats get_overall_stats() const;

    /**
     * 配置参数
     */
    void enable_auto_gc(bool enable) { auto_gc_enabled_ = enable; }
    void set_gc_threshold(size_t threshold) { gc_threshold_ = threshold; }
};
