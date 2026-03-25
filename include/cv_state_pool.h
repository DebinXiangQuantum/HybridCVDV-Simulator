#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

/**
 * High-water-mark GPU scratch buffer.
 * Grows on demand, never shrinks during a circuit execution.
 * Eliminates per-gate cudaMalloc/cudaFree overhead.
 */
struct GPUScratchBuffer {
    void* ptr = nullptr;
    size_t capacity_bytes = 0;

    void* ensure(size_t required_bytes) {
        if (required_bytes <= capacity_bytes) return ptr;
        if (ptr) { cudaFree(ptr); ptr = nullptr; capacity_bytes = 0; }
        cudaError_t err = cudaMalloc(&ptr, required_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "GPUScratchBuffer::ensure cudaMalloc failed (" +
                std::to_string(required_bytes) + " bytes): " +
                std::string(cudaGetErrorString(err)));
        }
        capacity_bytes = required_bytes;
        return ptr;
    }

    void release() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; capacity_bytes = 0; }
    }

    ~GPUScratchBuffer() { release(); }

    // Non-copyable
    GPUScratchBuffer() = default;
    GPUScratchBuffer(const GPUScratchBuffer&) = delete;
    GPUScratchBuffer& operator=(const GPUScratchBuffer&) = delete;
};

/**
 * Reusable pinned host staging buffer for faster H2D uploads.
 * This keeps small/medium parameter uploads off pageable memory.
 */
struct PinnedHostBuffer {
    void* ptr = nullptr;
    size_t capacity_bytes = 0;

    void* ensure(size_t required_bytes) {
        if (required_bytes <= capacity_bytes) return ptr;
        if (ptr) { cudaFreeHost(ptr); ptr = nullptr; capacity_bytes = 0; }
        cudaError_t err = cudaHostAlloc(&ptr, required_bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "PinnedHostBuffer::ensure cudaHostAlloc failed (" +
                std::to_string(required_bytes) + " bytes): " +
                std::string(cudaGetErrorString(err)));
        }
        capacity_bytes = required_bytes;
        return ptr;
    }

    void release() {
        if (ptr) { cudaFreeHost(ptr); ptr = nullptr; capacity_bytes = 0; }
    }

    ~PinnedHostBuffer() { release(); }

    PinnedHostBuffer() = default;
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
};

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
    int64_t max_total_dim = 0;  // D^max_qumodes 或内存限制下的最大维度
    int64_t total_dim = 0;      // 别名，用于向后兼容

    // 动态张量积管理
    int64_t* state_dims = nullptr;       // 每个状态的当前维度 [capacity]
    size_t* state_offsets = nullptr; // 每个状态在data中的偏移量 [capacity] (元素偏移量)

    // 内存管理
    size_t total_memory_size = 0;  // 已分配的总内存大小
    size_t max_memory_size = 0;    // 最大允许内存大小

    // 主机端状态跟踪
    std::vector<int> free_state_ids;
    std::vector<uint8_t> active_flags;
    std::vector<int64_t> host_state_dims;
    std::vector<size_t> host_state_offsets;
    std::vector<size_t> host_state_capacities;

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
    int64_t get_state_dim(int state_id) const;

    /**
     * 获取最大总维度 (向后兼容)
     * @return 最大可能的张量积维度
     */
    int64_t get_max_total_dim() const { return max_total_dim; }

    /**
     * 获取当前活跃状态实际占用的总元素数。
     */
    size_t get_active_storage_elements() const;

    /**
     * 预留至少指定总元素数的连续设备存储。
     */
    void reserve_total_storage_elements(size_t required_elements);

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
     * 为状态预留指定维度的存储空间。
     * 调用者负责后续写满该状态向量。
     */
    void reserve_state_storage(int state_id, int64_t state_dim);

    /**
     * 为临时输出预留独立存储块，但不立即绑定到任何状态ID。
     */
    size_t allocate_detached_storage(size_t required_elements);

    /**
     * 释放未绑定到状态ID的独立存储块。
     */
    void release_detached_storage(size_t offset, size_t reserved_elements);

    /**
     * 将状态ID重新绑定到一个已经写满的新存储块，并释放旧存储块。
     */
    void replace_state_storage(int state_id,
                               size_t new_offset,
                               size_t new_capacity,
                               int state_dim);

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

    // ── Scratch buffers (reused across gate executions) ──────────────
    GPUScratchBuffer scratch_target_ids;  // for d_target_ids (int arrays)
    GPUScratchBuffer scratch_temp;        // for gate temp buffers (cuDoubleComplex)
    GPUScratchBuffer scratch_aux;         // for small auxiliary allocations
    PinnedHostBuffer host_transfer_staging; // reusable pinned staging for H2D uploads

    template <typename T>
    T* upload_values_to_buffer(const T* host_values,
                               size_t count,
                               GPUScratchBuffer& scratch) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "upload_values_to_buffer requires trivially copyable elements");
        if (count == 0) {
            return nullptr;
        }
        if (!host_values) {
            throw std::invalid_argument("upload_values_to_buffer host_values must not be null");
        }

        const size_t bytes = count * sizeof(T);
        T* staged = static_cast<T*>(host_transfer_staging.ensure(bytes));
        std::memcpy(staged, host_values, bytes);

        T* device_ptr = static_cast<T*>(scratch.ensure(bytes));
        const cudaError_t err = cudaMemcpy(device_ptr, staged, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "upload_values_to_buffer cudaMemcpy failed (" +
                std::to_string(bytes) + " bytes): " +
                std::string(cudaGetErrorString(err)));
        }
        return device_ptr;
    }

    template <typename T>
    T* upload_vector_to_buffer(const std::vector<T>& host_values,
                               GPUScratchBuffer& scratch) {
        return upload_values_to_buffer(host_values.data(), host_values.size(), scratch);
    }

private:
    struct FreeBlock {
        size_t offset = 0;
        size_t length = 0;
    };

    size_t data_capacity_elements_ = 0;
    size_t allocated_elements_ = 0;
    size_t metadata_memory_size_ = 0;
    std::vector<FreeBlock> free_blocks_;

    void release_device_scratch_buffers();
    size_t active_storage_elements() const;
    void refresh_total_memory_size();
    void sync_state_metadata_to_device(int state_id);
    void ensure_data_capacity(size_t required_elements);
    size_t acquire_storage_block(size_t required_elements);
    void release_storage_block(int state_id);
    void assign_state_storage(int state_id, size_t required_elements);
    void merge_free_blocks();

    // 禁用拷贝构造和赋值
    CVStatePool(const CVStatePool&) = delete;
    CVStatePool& operator=(const CVStatePool&) = delete;
};
