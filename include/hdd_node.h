#pragma once

#include <complex>
#include <memory>
#include <atomic>
#include <unordered_map>

/**
 * 混合决策图节点 (HDD Node)
 *
 * HDD的叶子节点充当指向GPU显存的指针，用于压缩N个Qubit的离散希尔伯特空间
 */
struct HDDNode {
    // 唯一标识 (用于Hash Table去重)
    size_t unique_id;

    // Qubit层级 (0 to N-1). -1表示终端节点
    int16_t qubit_level;

    // 引用计数 (用于GPU显存垃圾回收)
    std::atomic<int> ref_count;

    // 子节点指针与权重 (Qubit分支)
    HDDNode* low;                 // |0> 分支
    HDDNode* high;                // |1> 分支
    std::complex<double> w_low;   // 权重 alpha (|0>分支权重)
    std::complex<double> w_high;  // 权重 beta (|1>分支权重)

    // GPU句柄 (仅Terminal Node有效)
    // 指向CVStatePool中的索引。若为-1，表示该分支概率为0
    int32_t tensor_id;

    /**
     * 构造函数 - 内部节点
     * @param level Qubit层级
     * @param low_child |0>分支子节点
     * @param high_child |1>分支子节点
     * @param weight_low |0>分支权重
     * @param weight_high |1>分支权重
     */
    HDDNode(int16_t level, HDDNode* low_child, HDDNode* high_child,
            std::complex<double> weight_low = 1.0, std::complex<double> weight_high = 1.0);

    /**
     * 构造函数 - 终端节点
     * @param cv_state_id GPU上的CV状态ID
     */
    explicit HDDNode(int32_t cv_state_id);

    /**
     * 析构函数
     */
    ~HDDNode();

    /**
     * 检查是否为终端节点
     */
    bool is_terminal() const { return qubit_level == -1; }

    /**
     * 获取节点的唯一标识
     * 用于Hash Table去重
     */
    size_t get_unique_id() const;

    /**
     * 增加引用计数
     */
    void increment_ref();

    /**
     * 减少引用计数
     * @return 新的引用计数
     */
    int decrement_ref();

    /**
     * 获取当前引用计数
     */
    int get_ref_count() const { return ref_count.load(); }

    /**
     * 深度复制节点 (用于Copy-on-Write)
     */
    HDDNode* deep_copy() const;

    /**
     * 计算节点哈希值
     * 用于唯一标识计算
     */
    size_t compute_hash() const;

private:
    // 禁用拷贝构造和赋值
    HDDNode(const HDDNode&) = delete;
    HDDNode& operator=(const HDDNode&) = delete;

    /**
     * 计算子节点的哈希值
     */
    size_t hash_combine(size_t lhs, size_t rhs) const;
};

/**
 * HDD节点哈希函数
 * 用于unordered_map等容器
 */
struct HDDNodeHash {
    size_t operator()(const HDDNode* node) const {
        return node ? node->get_unique_id() : 0;
    }
};

/**
 * HDD节点相等比较函数
 * 用于unordered_map等容器
 */
struct HDDNodeEqual {
    bool operator()(const HDDNode* lhs, const HDDNode* rhs) const {
        if (!lhs || !rhs) return lhs == rhs;
        return lhs->get_unique_id() == rhs->get_unique_id();
    }
};

/**
 * HDD节点管理器
 * 负责节点的创建、缓存和垃圾回收
 */
class HDDNodeManager {
private:
    std::unordered_map<size_t, HDDNode*> node_cache_;
    std::atomic<size_t> next_unique_id_{0};

public:
    /**
     * 创建或获取内部节点
     * 如果相同节点已存在，返回缓存的节点
     */
    HDDNode* get_or_create_node(int16_t level, HDDNode* low, HDDNode* high,
                               std::complex<double> w_low = 1.0,
                               std::complex<double> w_high = 1.0);

    /**
     * 创建终端节点
     */
    HDDNode* create_terminal_node(int32_t cv_state_id);

    /**
     * 释放节点 (减少引用计数，如果为0则删除)
     */
    void release_node(HDDNode* node);

    /**
     * 清理未使用的节点
     */
    void garbage_collect();

    /**
     * 获取缓存中的节点数量
     */
    size_t get_cache_size() const { return node_cache_.size(); }

    /**
     * 清空所有节点
     */
    void clear();
};
