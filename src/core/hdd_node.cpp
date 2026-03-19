#include "hdd_node.h"
#include <algorithm>
#include <iostream>
#include <functional>
#include <vector>

/**
 * HDDNode 构造函数 - 内部节点
 */
HDDNode::HDDNode(int16_t level, HDDNode* low_child, HDDNode* high_child,
                 std::complex<double> weight_low, std::complex<double> weight_high)
    : qubit_level(level), low(low_child), high(high_child),
      w_low(weight_low), w_high(weight_high), tensor_id(-1), ref_count(1) {

    if (level < 0) {
        throw std::invalid_argument("内部节点层级必须 >= 0");
    }

    unique_id = compute_hash();

    // 增加子节点的引用计数
    if (low) low->increment_ref();
    if (high) high->increment_ref();
}

/**
 * HDDNode 构造函数 - 终端节点
 */
HDDNode::HDDNode(int32_t cv_state_id)
    : qubit_level(-1), low(nullptr), high(nullptr),
      w_low(1.0, 0.0), w_high(1.0, 0.0), tensor_id(cv_state_id), ref_count(1) {

    unique_id = compute_hash();
}

/**
 * HDDNode 析构函数
 */
HDDNode::~HDDNode() {
    // 释放子节点引用
    if (low) low->decrement_ref();
    if (high) high->decrement_ref();
}

/**
 * 获取节点的唯一标识
 */
size_t HDDNode::get_unique_id() const {
    return unique_id;
}

/**
 * 增加引用计数
 */
void HDDNode::increment_ref() {
    ref_count.fetch_add(1, std::memory_order_relaxed);
}

/**
 * 减少引用计数
 */
int HDDNode::decrement_ref() {
    int new_count = ref_count.fetch_sub(1, std::memory_order_relaxed) - 1;
    return new_count;
}

/**
 * 深度复制节点 (用于Copy-on-Write)
 */
HDDNode* HDDNode::deep_copy() const {
    HDDNode* copy = nullptr;
    if (is_terminal()) {
        copy = new HDDNode(tensor_id);
    } else {
        copy = new HDDNode(qubit_level, low, high, w_low, w_high);
    }

    // Generate a new unique ID for the copy to make it truly independent
    // Use a static counter to ensure each copy gets a unique ID
    static std::atomic<size_t> copy_id_counter{1000000000}; // Start from high number to avoid collision
    copy->unique_id = copy_id_counter.fetch_add(1, std::memory_order_relaxed);

    return copy;
}

/**
 * 计算节点哈希值
 */
size_t HDDNode::compute_hash() const {
    size_t hash = 0;

    // 结合层级信息
    hash = hash_combine(hash, static_cast<size_t>(qubit_level));

    if (is_terminal()) {
        // 终端节点：结合tensor_id
        hash = hash_combine(hash, static_cast<size_t>(tensor_id));
    } else {
        // 内部节点：结合子节点ID和权重
        if (low) {
            hash = hash_combine(hash, low->unique_id);
            hash = hash_combine(hash, std::hash<double>()(w_low.real()));
            hash = hash_combine(hash, std::hash<double>()(w_low.imag()));
        }
        if (high) {
            hash = hash_combine(hash, high->unique_id);
            hash = hash_combine(hash, std::hash<double>()(w_high.real()));
            hash = hash_combine(hash, std::hash<double>()(w_high.imag()));
        }
    }

    return hash;
}

/**
 * 计算子节点的哈希值
 */
size_t HDDNode::hash_combine(size_t lhs, size_t rhs) const {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
}

// ===== HDDNodeManager 实现 =====

bool HDDNodeManager::nodes_equivalent(const HDDNode* lhs, const HDDNode* rhs) const {
    if (lhs == rhs) {
        return true;
    }
    if (!lhs || !rhs) {
        return false;
    }
    if (lhs->qubit_level != rhs->qubit_level) {
        return false;
    }
    if (lhs->is_terminal()) {
        return lhs->tensor_id == rhs->tensor_id;
    }
    return lhs->low == rhs->low &&
           lhs->high == rhs->high &&
           lhs->w_low == rhs->w_low &&
           lhs->w_high == rhs->w_high;
}

/**
 * 创建或获取内部节点
 */
HDDNode* HDDNodeManager::get_or_create_node(int16_t level, HDDNode* low, HDDNode* high,
                                           std::complex<double> w_low, std::complex<double> w_high) {
    HDDNode* new_node = new HDDNode(level, low, high, w_low, w_high);
    size_t hash_key = new_node->unique_id;

    auto& bucket = node_cache_[hash_key];
    for (HDDNode* cached_node : bucket) {
        if (nodes_equivalent(cached_node, new_node)) {
            delete new_node;
            cached_node->increment_ref();
            return cached_node;
        }
    }

    bucket.push_back(new_node);
    ++cached_node_count_;
    return new_node;
}

/**
 * 创建终端节点
 */
HDDNode* HDDNodeManager::create_terminal_node(int32_t cv_state_id) {
    HDDNode* new_node = new HDDNode(cv_state_id);
    size_t hash_key = new_node->unique_id;

    auto& bucket = node_cache_[hash_key];
    for (HDDNode* cached_node : bucket) {
        if (nodes_equivalent(cached_node, new_node)) {
            delete new_node;
            cached_node->increment_ref();
            return cached_node;
        }
    }

    bucket.push_back(new_node);
    ++cached_node_count_;
    return new_node;
}

/**
 * 释放节点 (减少引用计数，如果为0则删除)
 */
void HDDNodeManager::release_node(HDDNode* node) {
    if (!node) return;

    int new_count = node->decrement_ref();
    if (new_count <= 0) {
        auto bucket_it = node_cache_.find(node->unique_id);
        if (bucket_it != node_cache_.end()) {
            auto& bucket = bucket_it->second;
            bucket.erase(std::remove(bucket.begin(), bucket.end(), node), bucket.end());
            if (bucket.empty()) {
                node_cache_.erase(bucket_it);
            }
        }
        if (cached_node_count_ > 0) {
            --cached_node_count_;
        }
        delete node;
    }
}

/**
 * 清理未使用的节点
 */
void HDDNodeManager::garbage_collect() {
    size_t removed_count = 0;
    for (auto it = node_cache_.begin(); it != node_cache_.end(); ) {
        auto& bucket = it->second;
        for (auto bucket_it = bucket.begin(); bucket_it != bucket.end(); ) {
            HDDNode* node = *bucket_it;
            if (node->get_ref_count() <= 0) {
                delete node;
                bucket_it = bucket.erase(bucket_it);
                ++removed_count;
                if (cached_node_count_ > 0) {
                    --cached_node_count_;
                }
            } else {
                ++bucket_it;
            }
        }

        if (bucket.empty()) {
            it = node_cache_.erase(it);
        } else {
            ++it;
        }
    }

    if (removed_count > 0) {
        std::cout << "HDD垃圾回收：清理了 " << removed_count << " 个未使用节点" << std::endl;
    }
}

/**
 * 清空所有节点
 */
void HDDNodeManager::clear() {
    for (auto& pair : node_cache_) {
        for (HDDNode* node : pair.second) {
            delete node;
        }
    }
    node_cache_.clear();
    cached_node_count_ = 0;
    next_unique_id_.store(0);

    std::cout << "HDD节点管理器已清空" << std::endl;
}
