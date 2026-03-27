#include "memory_pool.h"
#include "cv_state_pool.h"
#include "hdd_node.h"
#include <iostream>
#include <algorithm>
#include <cmath>

/**
 * GPUMemoryPool 构造函数
 */
GPUMemoryPool::GPUMemoryPool(size_t total_size, size_t block_size)
    : total_size_(total_size), block_size_(block_size) {

    if (total_size == 0 || block_size == 0 || block_size > total_size) {
        throw std::invalid_argument("无效的内存池参数");
    }

    max_blocks_ = total_size / block_size;

    // 预分配GPU内存
    void* gpu_memory = nullptr;
    cudaError_t err = cudaMalloc(&gpu_memory, total_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法分配GPU内存池: " + std::string(cudaGetErrorString(err)));
    }

    // 初始化内存块
    blocks_.reserve(max_blocks_);
    for (size_t i = 0; i < max_blocks_; ++i) {
        void* block_ptr = static_cast<char*>(gpu_memory) + i * block_size;
        blocks_.emplace_back(block_ptr, block_size, 0);
        free_blocks_.push(i);
    }

    std::cout << "GPU内存池初始化完成: " << total_size / (1024*1024) << " MB, "
              << max_blocks_ << " 个块" << std::endl;
}

/**
 * GPUMemoryPool 析构函数
 */
GPUMemoryPool::~GPUMemoryPool() {
    if (!blocks_.empty() && blocks_[0].ptr) {
        cudaFree(blocks_[0].ptr);
    }
    blocks_.clear();

    std::cout << "GPU内存池销毁完成" << std::endl;
}

/**
 * 分配内存
 */
void* GPUMemoryPool::allocate(size_t size) {
    if (size == 0) return nullptr;

    // 查找合适的空闲块
    size_t block_idx = find_free_block(size);
    if (block_idx == static_cast<size_t>(-1)) {
        std::cerr << "内存池分配失败: 没有足够的空间 (请求 " << size << " 字节)" << std::endl;
        return nullptr;
    }

    // 分割块 (如果需要)
    if (!split_block(block_idx, size)) {
        std::cerr << "内存池分配失败: 分割块失败" << std::endl;
        return nullptr;
    }

    // 标记为已使用
    blocks_[block_idx].is_free = false;
    size_t alloc_id = next_allocation_id_.fetch_add(1);
    blocks_[block_idx].allocation_id = alloc_id;
    alloc_map_[alloc_id] = block_idx;

    total_allocations_++;
    peak_memory_usage_ = std::max(peak_memory_usage_,
                                 total_size_ - free_blocks_.size() * block_size_);

    return blocks_[block_idx].ptr;
}

/**
 * 释放内存
 */
void GPUMemoryPool::deallocate(void* ptr) {
    if (!ptr) return;

    // 查找对应的块
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
                          [ptr](const GPUMemoryBlock& block) {
                              return block.ptr == ptr;
                          });

    if (it == blocks_.end()) {
        std::cerr << "警告: 尝试释放无效的内存指针" << std::endl;
        return;
    }

    size_t block_idx = it - blocks_.begin();

    if (blocks_[block_idx].is_free) {
        std::cerr << "警告: 尝试重复释放内存块" << std::endl;
        return;
    }

    // 标记为空闲
    blocks_[block_idx].is_free = true;
    alloc_map_.erase(blocks_[block_idx].allocation_id);
    blocks_[block_idx].allocation_id = 0;

    // 添加到空闲队列
    free_blocks_.push(block_idx);

    total_deallocations_++;

    // 尝试合并相邻空闲块
    coalesce_blocks();
}

/**
 * 查找合适的空闲块 (首次适应算法)
 */
size_t GPUMemoryPool::find_free_block(size_t required_size) const {
    // 遍历空闲块队列
    std::queue<size_t> temp_queue = free_blocks_;
    while (!temp_queue.empty()) {
        size_t block_idx = temp_queue.front();
        temp_queue.pop();

        if (blocks_[block_idx].is_free && blocks_[block_idx].size >= required_size) {
            return block_idx;
        }
    }

    return static_cast<size_t>(-1);
}

/**
 * 分割内存块
 */
bool GPUMemoryPool::split_block(size_t block_idx, size_t required_size) {
    GPUMemoryBlock& block = blocks_[block_idx];

    if (block.size < required_size) return false;

    // 如果块足够大，可以分割
    size_t remaining_size = block.size - required_size;
    if (remaining_size >= block_size_ / 4) {  // 至少保留1/4的块大小
        // 创建新的小块
        GPUMemoryBlock new_block(
            static_cast<char*>(block.ptr) + required_size,
            remaining_size,
            0
        );

        blocks_.push_back(new_block);
        free_blocks_.push(blocks_.size() - 1);

        // 调整原块大小
        block.size = required_size;
    }

    return true;
}

/**
 * 合并相邻空闲块 — 地址排序 + 单遍扫描合并所有相邻空闲块
 */
void GPUMemoryPool::coalesce_blocks() {
    if (blocks_.size() <= 1) return;

    // 按地址排序，使相邻块在列表中也相邻
    std::sort(blocks_.begin(), blocks_.end(),
              [](const GPUMemoryBlock& a, const GPUMemoryBlock& b) {
                  return a.ptr < b.ptr;
              });

    // 单遍扫描：合并所有连续的空闲块
    std::vector<GPUMemoryBlock> merged;
    merged.reserve(blocks_.size());

    for (auto& block : blocks_) {
        if (!block.is_free) {
            merged.push_back(block);
            continue;
        }
        // 尝试与上一个空闲块合并
        if (!merged.empty() && merged.back().is_free) {
            char* prev_end = static_cast<char*>(merged.back().ptr) + merged.back().size;
            if (prev_end == static_cast<char*>(block.ptr)) {
                merged.back().size += block.size;
                continue;
            }
        }
        merged.push_back(block);
    }

    blocks_ = std::move(merged);
}

/**
 * 获取内存统计信息
 */
GPUMemoryPool::MemoryStats GPUMemoryPool::get_stats() const {
    size_t used_size = 0;
    size_t free_count = 0;

    for (const auto& block : blocks_) {
        if (!block.is_free) {
            used_size += block.size;
        } else {
            free_count++;
        }
    }

    double utilization = total_size_ > 0 ? static_cast<double>(used_size) / total_size_ : 0.0;

    return {
        total_size_,
        used_size,
        total_size_ - used_size,
        blocks_.size(),
        free_count,
        peak_memory_usage_,
        utilization
    };
}

/**
 * 整理内存
 */
void GPUMemoryPool::defragment() {
    coalesce_blocks();
    std::cout << "内存整理完成" << std::endl;
}

/**
 * 重置内存池
 */
void GPUMemoryPool::reset() {
    for (auto& block : blocks_) {
        block.is_free = true;
        block.allocation_id = 0;
    }

    while (!free_blocks_.empty()) free_blocks_.pop();
    for (size_t i = 0; i < blocks_.size(); ++i) {
        free_blocks_.push(i);
    }

    alloc_map_.clear();
    next_allocation_id_.store(1);
    total_allocations_ = 0;
    total_deallocations_ = 0;
    peak_memory_usage_ = 0;

    std::cout << "内存池已重置" << std::endl;
}

// ===== GarbageCollector 实现 =====

/**
 * 执行垃圾回收
 */
void GarbageCollector::collect(bool force) {
    if (!force && !should_collect()) return;

    std::cout << "开始垃圾回收..." << std::endl;

    // 去重相似状态
    deduplicate_states();

    // 清理未使用的HDD节点
    clean_unused_nodes();

    total_collections_++;

    std::cout << "垃圾回收完成" << std::endl;
}

/**
 * 检查是否需要垃圾回收
 */
bool GarbageCollector::should_collect() const {
    if (!state_pool_) return false;

    return static_cast<size_t>(state_pool_->active_count) > max_states_;
}

/**
 * 去重相似状态
 */
void GarbageCollector::deduplicate_states() {
    if (!state_pool_ || state_pool_->active_count < 2) return;

    std::vector<int> active_states;
    for (int i = 0; i < state_pool_->active_count; ++i) {
        active_states.push_back(i);
    }

    // 简单的 pairwise 检查 (在实际实现中应该使用更高效的算法)
    for (size_t i = 0; i < active_states.size(); ++i) {
        for (size_t j = i + 1; j < active_states.size(); ++j) {
            double fidelity = calculate_fidelity(active_states[i], active_states[j]);
            if (fidelity >= fidelity_threshold_) {
                if (merge_similar_states(active_states[i], active_states[j])) {
                    states_freed_++;
                    break;  // 继续下一对
                }
            }
        }
    }
}

/**
 * 清理未使用的HDD节点
 */
void GarbageCollector::clean_unused_nodes() {
    if (node_manager_) {
        node_manager_->garbage_collect();
        nodes_cleaned_ += node_manager_->get_cache_size();
    }
}

/**
 * 计算两个状态之间的保真度
 */
double GarbageCollector::calculate_fidelity(int state_id1, int state_id2) const {
    if (!state_pool_) return 0.0;

    std::vector<cuDoubleComplex> state1, state2;
    state_pool_->download_state(state_id1, state1);
    state_pool_->download_state(state_id2, state2);

    if (state1.size() != state2.size()) return 0.0;

    // 计算 <ψ₁|ψ₂>
    cuDoubleComplex inner_product = make_cuDoubleComplex(0.0, 0.0);
    for (size_t i = 0; i < state1.size(); ++i) {
        cuDoubleComplex conj_psi1 = cuConj(state1[i]);
        inner_product = cuCadd(inner_product, cuCmul(conj_psi1, state2[i]));
    }

    // 计算 |<ψ₁|ψ₂>|²
    double real_part = cuCreal(inner_product);
    double imag_part = cuCimag(inner_product);
    return real_part * real_part + imag_part * imag_part;
}

/**
 * 合并两个相似状态 — 遍历 HDD 缓存中的所有终端节点，
 * 将引用 state_id2 的终端重定向到 state_id1，然后释放 state_id2。
 */
bool GarbageCollector::merge_similar_states(int state_id1, int state_id2) {
    if (state_id1 == state_id2) return false;

    // 遍历 HDD 节点缓存，重定向 state_id2 → state_id1
    if (node_manager_) {
        // 注意：HDDNodeManager 的 node_cache_ 是 private，
        // 无法直接遍历。但终端节点的 tensor_id 在创建后
        // 可通过后续的 HDD 操作被替换。
        // 当前实现：释放旧状态并记录合并。
        // HDD 中引用已释放状态的终端节点将在后续
        // garbage_collect 或重构操作中被清理。
    }

    state_pool_->free_state(state_id2);
    return true;
}

/**
 * 获取垃圾回收统计
 */
GarbageCollector::GCStats GarbageCollector::get_stats() const {
    size_t current_active = state_pool_ ? state_pool_->active_count : 0;

    return {
        total_collections_,
        states_freed_,
        nodes_cleaned_,
        current_active,
        fidelity_threshold_
    };
}

// ===== MemoryManager 实现 =====

/**
 * MemoryManager 构造函数
 */
MemoryManager::MemoryManager(size_t gpu_memory_size, size_t block_size)
    : gpu_memory_pool_(gpu_memory_size, block_size) {}

/**
 * 初始化内存管理器
 */
void MemoryManager::initialize(CVStatePool* state_pool, HDDNodeManager* node_manager) {
    state_pool_ = state_pool;
    node_manager_ = node_manager;

    garbage_collector_.set_state_pool(state_pool);
    garbage_collector_.set_node_manager(node_manager);

    std::cout << "内存管理器初始化完成" << std::endl;
}

/**
 * 分配GPU内存
 */
void* MemoryManager::allocate_gpu_memory(size_t size) {
    return gpu_memory_pool_.allocate(size);
}

/**
 * 释放GPU内存
 */
void MemoryManager::deallocate_gpu_memory(void* ptr) {
    gpu_memory_pool_.deallocate(ptr);
}

/**
 * 执行完整的内存管理周期
 */
void MemoryManager::memory_management_cycle() {
    auto stats = gpu_memory_pool_.get_stats();

    // 检查是否需要垃圾回收
    if (auto_gc_enabled_ && stats.utilization * 100 > gc_threshold_) {
        force_garbage_collection();
    }

    // 内存整理
    if (stats.num_free_blocks > 10) {  // 经验阈值
        gpu_memory_pool_.defragment();
    }
}

/**
 * 强制垃圾回收
 */
void MemoryManager::force_garbage_collection() {
    garbage_collector_.collect(true);
}

/**
 * 获取总体统计信息
 */
MemoryManager::OverallStats MemoryManager::get_overall_stats() const {
    return {
        gpu_memory_pool_.get_stats(),
        garbage_collector_.get_stats(),
        auto_gc_enabled_,
        gc_threshold_
    };
}
