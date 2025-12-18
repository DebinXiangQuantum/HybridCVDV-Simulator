#include "hdd_node.h"
#include <iostream>
#include <functional>

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
    if (is_terminal()) {
        return new HDDNode(tensor_id);
    } else {
        return new HDDNode(qubit_level, low, high, w_low, w_high);
    }
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

/**
 * 创建或获取内部节点
 */
HDDNode* HDDNodeManager::get_or_create_node(int16_t level, HDDNode* low, HDDNode* high,
                                           std::complex<double> w_low, std::complex<double> w_high) {
    // 创建新节点（不增加子节点引用）
    HDDNode* new_node = new HDDNode(level, low, high, w_low, w_high);
    size_t hash_key = new_node->unique_id;

    // 查找是否已存在相同哈希的节点
    auto it = node_cache_.find(hash_key);
    if (it != node_cache_.end()) {
        // 找到现有节点，删除新创建的临时节点，返回现有节点
        delete new_node;
        it->second->increment_ref();
        return it->second;
    }

    // 没有找到，添加新节点到缓存
    node_cache_[hash_key] = new_node;

    return new_node;
}

/**
 * 创建终端节点
 */
HDDNode* HDDNodeManager::create_terminal_node(int32_t cv_state_id) {
    // 创建新终端节点
    HDDNode* new_node = new HDDNode(cv_state_id);
    size_t hash_key = new_node->unique_id;

    // 查找是否已存在相同哈希的节点
    auto it = node_cache_.find(hash_key);
    if (it != node_cache_.end()) {
        // 找到现有节点，删除新创建的临时节点，返回现有节点
        delete new_node;
        it->second->increment_ref();
        return it->second;
    }

    // 没有找到，添加新节点到缓存
    node_cache_[hash_key] = new_node;

    return new_node;
}

/**
 * 释放节点 (减少引用计数，如果为0则删除)
 */
void HDDNodeManager::release_node(HDDNode* node) {
    if (!node) return;

    int new_count = node->decrement_ref();
    if (new_count <= 0) {
        // 从缓存中移除
        node_cache_.erase(node->unique_id);
        delete node;
    }
}

/**
 * 清理未使用的节点
 */
void HDDNodeManager::garbage_collect() {
    std::vector<size_t> to_remove;

    for (const auto& pair : node_cache_) {
        HDDNode* node = pair.second;
        if (node->get_ref_count() <= 0) {
            to_remove.push_back(pair.first);
        }
    }

    for (size_t id : to_remove) {
        delete node_cache_[id];
        node_cache_.erase(id);
    }

    if (!to_remove.empty()) {
        std::cout << "HDD垃圾回收：清理了 " << to_remove.size() << " 个未使用节点" << std::endl;
    }
}

/**
 * 清空所有节点
 */
void HDDNodeManager::clear() {
    for (auto& pair : node_cache_) {
        delete pair.second;
    }
    node_cache_.clear();
    next_unique_id_.store(0);

    std::cout << "HDD节点管理器已清空" << std::endl;
}
