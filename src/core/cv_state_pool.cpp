#include "cv_state_pool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

size_t bytes_for_elements(size_t elements) {
    return elements * sizeof(cuDoubleComplex);
}

}  // namespace

CVStatePool::CVStatePool(int trunc_dim, int max_states, int num_qumodes, size_t max_memory_mb)
    : d_trunc(trunc_dim),
      capacity(max_states),
      active_count(0),
      max_total_dim(1),
      total_dim(1),
      total_memory_size(0),
      max_memory_size(max_memory_mb * 1024ULL * 1024ULL) {
    int device_count = 0;
    cudaError_t device_check = cudaGetDeviceCount(&device_count);
    std::cout << "CUDA设备检查: device_count=" << device_count
              << ", error=" << cudaGetErrorString(device_check) << std::endl;

    if (device_check != cudaSuccess || device_count == 0) {
        setenv("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH", 1);
        cudaError_t retry = cudaGetDeviceCount(&device_count);
        std::cout << "重新检查CUDA设备: device_count=" << device_count
                  << ", error=" << cudaGetErrorString(retry) << std::endl;
        if (retry != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA设备不可用: " + std::string(cudaGetErrorString(retry)));
        }
    }

    if (d_trunc <= 0 || capacity <= 0) {
        throw std::invalid_argument("截断维度和容量必须为正数");
    }

    const cudaError_t set_device_err = cudaSetDevice(0);
    if (set_device_err != cudaSuccess) {
        throw std::runtime_error("无法设置CUDA设备: " + std::string(cudaGetErrorString(set_device_err)));
    }
    cudaGetLastError();

    for (int i = 0; i < num_qumodes; ++i) {
        if (max_total_dim > std::numeric_limits<int64_t>::max() / d_trunc) {
            throw std::overflow_error("状态空间维度溢出");
        }
        max_total_dim *= d_trunc;
    }
    total_dim = max_total_dim;

    auto cleanup = [this]() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
        if (free_list) {
            cudaFree(free_list);
            free_list = nullptr;
        }
        if (state_dims) {
            cudaFree(state_dims);
            state_dims = nullptr;
        }
        if (state_offsets) {
            cudaFree(state_offsets);
            state_offsets = nullptr;
        }
    };

    cudaError_t err = cudaMalloc(&free_list, capacity * sizeof(int));
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法分配GPU内存用于空闲列表: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&state_dims, capacity * sizeof(int64_t));
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法分配GPU内存用于状态维度: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&state_offsets, capacity * sizeof(size_t));
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法分配GPU内存用于状态偏移量: " + std::string(cudaGetErrorString(err)));
    }

    metadata_memory_size_ =
        static_cast<size_t>(capacity) * sizeof(int) +
        static_cast<size_t>(capacity) * sizeof(int) +
        static_cast<size_t>(capacity) * sizeof(size_t);

    std::vector<int> host_free_list(capacity);
    for (int i = 0; i < capacity; ++i) {
        host_free_list[i] = i;
    }
    err = cudaMemcpy(free_list, host_free_list.data(), capacity * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法初始化空闲列表: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemset(state_dims, 0, capacity * sizeof(int64_t));
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法初始化状态维度: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemset(state_offsets, 0, capacity * sizeof(size_t));
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法初始化状态偏移量: " + std::string(cudaGetErrorString(err)));
    }

    free_state_ids.reserve(capacity);
    for (int i = capacity - 1; i >= 0; --i) {
        free_state_ids.push_back(i);
    }
    active_flags.assign(capacity, 0);
    host_state_dims.assign(capacity, 0);
    host_state_offsets.assign(capacity, 0);
    host_state_capacities.assign(capacity, 0);
    refresh_total_memory_size();

    std::cout << "CVStatePool 初始化完成: 单个qumode截断维度=" << d_trunc
              << ", 初始总维度=" << max_total_dim
              << ", 容量=" << capacity
              << ", 初始数据内存=" << (data_capacity_elements_ * sizeof(cuDoubleComplex) / (1024.0 * 1024.0))
              << " MB" << std::endl;
}

CVStatePool::~CVStatePool() {
    cudaGetLastError();

    if (data) {
        cudaError_t err = cudaFree(data);
        if (err != cudaSuccess && err != cudaErrorProfilerNotInitialized) {
            std::cerr << "警告：释放状态池数据内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        data = nullptr;
    }
    if (free_list) {
        cudaError_t err = cudaFree(free_list);
        if (err != cudaSuccess && err != cudaErrorProfilerNotInitialized) {
            std::cerr << "警告：释放空闲列表内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        free_list = nullptr;
    }
    if (state_dims) {
        cudaError_t err = cudaFree(state_dims);
        if (err != cudaSuccess && err != cudaErrorProfilerNotInitialized) {
            std::cerr << "警告：释放状态维度内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        state_dims = nullptr;
    }
    if (state_offsets) {
        cudaError_t err = cudaFree(state_offsets);
        if (err != cudaSuccess && err != cudaErrorProfilerNotInitialized) {
            std::cerr << "警告：释放状态偏移量内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        state_offsets = nullptr;
    }

    std::cout << "CVStatePool 销毁完成" << std::endl;
}

void CVStatePool::refresh_total_memory_size() {
    total_memory_size = metadata_memory_size_ + bytes_for_elements(data_capacity_elements_);
}

void CVStatePool::release_device_scratch_buffers() {
    scratch_target_ids.release();
    scratch_temp.release();
    scratch_aux.release();
}

size_t CVStatePool::active_storage_elements() const {
    size_t live_elements = 0;
    for (int state_id = 0; state_id < capacity; ++state_id) {
        if (!active_flags[state_id]) {
            continue;
        }
        live_elements += host_state_capacities[state_id];
    }
    return live_elements;
}

size_t CVStatePool::get_active_storage_elements() const {
    return active_storage_elements();
}

void CVStatePool::sync_state_metadata_to_device(int state_id) {
    cudaError_t err = cudaMemcpy(state_dims + state_id,
                                 &host_state_dims[state_id],
                                 sizeof(int64_t),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法同步状态维度: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(state_offsets + state_id,
                     &host_state_offsets[state_id],
                     sizeof(size_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法同步状态偏移量: " + std::string(cudaGetErrorString(err)));
    }
}

void CVStatePool::ensure_data_capacity(size_t required_elements) {
    if (required_elements <= data_capacity_elements_) {
        return;
    }

    auto try_repack_live_storage = [&](size_t target_capacity, cuDoubleComplex** out_data) -> bool {
        const size_t live_elements = active_storage_elements();
        if (!data || live_elements == 0 || live_elements >= data_capacity_elements_) {
            return false;
        }

        std::vector<int> active_state_ids = get_active_state_ids();
        std::vector<size_t> compact_offsets(static_cast<size_t>(capacity), 0);
        size_t compact_cursor = 0;

        cuDoubleComplex* compact_data = nullptr;
        cudaError_t compact_alloc_err = cudaMalloc(&compact_data, bytes_for_elements(live_elements));
        if (compact_alloc_err != cudaSuccess) {
            cudaGetLastError();
            return false;
        }

        auto cleanup_compact = [&]() {
            if (compact_data) {
                cudaFree(compact_data);
                compact_data = nullptr;
            }
        };

        try {
            for (int state_id : active_state_ids) {
                const size_t reserved = host_state_capacities[state_id];
                if (reserved == 0) {
                    continue;
                }

                compact_offsets[static_cast<size_t>(state_id)] = compact_cursor;
                const cudaError_t copy_err = cudaMemcpy(
                    compact_data + compact_cursor,
                    data + host_state_offsets[state_id],
                    bytes_for_elements(reserved),
                    cudaMemcpyDeviceToDevice);
                if (copy_err != cudaSuccess) {
                    throw std::runtime_error(
                        "无法压缩迁移活跃状态: " + std::string(cudaGetErrorString(copy_err)));
                }
                compact_cursor += reserved;
            }
        } catch (...) {
            cleanup_compact();
            cudaGetLastError();
            return false;
        }

        cudaError_t free_err = cudaFree(data);
        if (free_err != cudaSuccess) {
            cleanup_compact();
            cudaGetLastError();
            return false;
        }
        data = nullptr;
        data_capacity_elements_ = 0;

        cuDoubleComplex* rebuilt_data = nullptr;
        cudaError_t rebuilt_alloc_err = cudaMalloc(&rebuilt_data, bytes_for_elements(target_capacity));
        if (rebuilt_alloc_err != cudaSuccess) {
            cleanup_compact();
            cudaGetLastError();
            return false;
        }

        try {
            for (int state_id : active_state_ids) {
                const size_t reserved = host_state_capacities[state_id];
                if (reserved == 0) {
                    continue;
                }

                const size_t compact_offset = compact_offsets[static_cast<size_t>(state_id)];
                const cudaError_t copy_err = cudaMemcpy(
                    rebuilt_data + compact_offset,
                    compact_data + compact_offset,
                    bytes_for_elements(reserved),
                    cudaMemcpyDeviceToDevice);
                if (copy_err != cudaSuccess) {
                    throw std::runtime_error(
                        "无法恢复压缩后的活跃状态: " + std::string(cudaGetErrorString(copy_err)));
                }

                host_state_offsets[state_id] = compact_offset;
                sync_state_metadata_to_device(state_id);
            }
        } catch (...) {
            cleanup_compact();
            cudaFree(rebuilt_data);
            cudaGetLastError();
            return false;
        }

        cleanup_compact();
        allocated_elements_ = live_elements;
        free_blocks_.clear();
        *out_data = rebuilt_data;
        return true;
    };

    size_t new_capacity = required_elements;
    if (data_capacity_elements_ != 0) {
        size_t growth = required_elements - data_capacity_elements_;
        growth = std::max(growth, data_capacity_elements_ / 4);
        growth = std::max<size_t>(growth, 1);
        growth = std::min(growth, static_cast<size_t>(std::max(INT64_C(1), max_total_dim)));
        new_capacity += growth;

        const size_t state_dim = static_cast<size_t>(std::max(INT64_C(1), max_total_dim));
        if (required_elements - data_capacity_elements_ <= state_dim) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                constexpr size_t kSafetyBytes = 256ULL * 1024ULL * 1024ULL;
                if (free_bytes > kSafetyBytes) {
                    const size_t allocatable_now =
                        (free_bytes - kSafetyBytes) / sizeof(cuDoubleComplex);
                    if (allocatable_now > required_elements) {
                        const size_t max_extra_states =
                            (allocatable_now - required_elements) / state_dim;
                        const size_t extra_states = std::min<size_t>(4, max_extra_states);
                        if (extra_states > 0 &&
                            extra_states <= (std::numeric_limits<size_t>::max() - required_elements) /
                                                state_dim) {
                            const size_t proactive_capacity =
                                required_elements + extra_states * state_dim;
                            new_capacity = std::max(new_capacity, proactive_capacity);
                        }
                    }
                }
            }
        }
    }

    if (max_memory_size > 0) {
        if (max_memory_size <= metadata_memory_size_) {
            throw std::runtime_error("内存限制过小，无法为状态数据预留空间");
        }
        const size_t max_data_elements =
            (max_memory_size - metadata_memory_size_) / sizeof(cuDoubleComplex);
        if (required_elements > max_data_elements) {
            throw std::runtime_error("超出CVStatePool内存上限");
        }
        new_capacity = std::min(new_capacity, max_data_elements);
    }

    cuDoubleComplex* new_data = nullptr;
    bool repacked_live_storage = false;
    cudaError_t alloc_err = cudaMalloc(&new_data, bytes_for_elements(new_capacity));
    if (alloc_err != cudaSuccess) {
        const size_t live_elements = active_storage_elements();
        const size_t previous_capacity = data_capacity_elements_;
        const size_t scratch_target_bytes = scratch_target_ids.capacity_bytes;
        const size_t scratch_temp_bytes = scratch_temp.capacity_bytes;
        const size_t scratch_aux_bytes = scratch_aux.capacity_bytes;

        release_device_scratch_buffers();
        cudaGetLastError();

        new_capacity = required_elements;
        alloc_err = cudaMalloc(&new_data, bytes_for_elements(new_capacity));
        if (alloc_err != cudaSuccess) {
            repacked_live_storage = try_repack_live_storage(new_capacity, &new_data);
            if (repacked_live_storage) {
                cudaGetLastError();
                alloc_err = cudaSuccess;
            }
        }
        if (alloc_err != cudaSuccess && !repacked_live_storage) {
            throw std::runtime_error(
                "无法扩展GPU状态池: " + std::string(cudaGetErrorString(alloc_err)) +
                " (required_elements=" + std::to_string(required_elements) +
                ", previous_capacity=" + std::to_string(previous_capacity) +
                ", live_elements=" + std::to_string(live_elements) +
                ", scratch_target_bytes=" + std::to_string(scratch_target_bytes) +
                ", scratch_temp_bytes=" + std::to_string(scratch_temp_bytes) +
                ", scratch_aux_bytes=" + std::to_string(scratch_aux_bytes) + ")");
        }
    }

    if (!repacked_live_storage) {
        const size_t copy_elements = std::min(allocated_elements_, data_capacity_elements_);
        if (data && copy_elements > 0) {
            const cudaError_t copy_err = cudaMemcpy(new_data,
                                                    data,
                                                    bytes_for_elements(copy_elements),
                                                    cudaMemcpyDeviceToDevice);
            if (copy_err != cudaSuccess) {
                cudaFree(new_data);
                throw std::runtime_error("无法迁移状态池数据: " + std::string(cudaGetErrorString(copy_err)));
            }
        }

        if (data) {
            cudaError_t free_err = cudaFree(data);
            if (free_err != cudaSuccess) {
                std::cerr << "警告：释放旧状态池数据失败: " << cudaGetErrorString(free_err) << std::endl;
            }
        }
    }

    data = new_data;
    data_capacity_elements_ = new_capacity;
    refresh_total_memory_size();
}

void CVStatePool::reserve_total_storage_elements(size_t required_elements) {
    ensure_data_capacity(required_elements);
}

size_t CVStatePool::acquire_storage_block(size_t required_elements) {
    if (required_elements == 0) {
        return 0;
    }

    size_t best_index = free_blocks_.size();
    size_t best_length = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < free_blocks_.size(); ++i) {
        const FreeBlock& block = free_blocks_[i];
        if (block.length >= required_elements && block.length < best_length) {
            best_index = i;
            best_length = block.length;
        }
    }

    if (best_index != free_blocks_.size()) {
        const size_t offset = free_blocks_[best_index].offset;
        if (free_blocks_[best_index].length == required_elements) {
            free_blocks_.erase(free_blocks_.begin() + static_cast<std::ptrdiff_t>(best_index));
        } else {
            free_blocks_[best_index].offset += required_elements;
            free_blocks_[best_index].length -= required_elements;
        }
        return offset;
    }

    const size_t offset = allocated_elements_;
    allocated_elements_ += required_elements;
    try {
        ensure_data_capacity(allocated_elements_);
    } catch (...) {
        allocated_elements_ -= required_elements;
        throw;
    }
    return offset;
}

void CVStatePool::merge_free_blocks() {
    if (free_blocks_.empty()) {
        return;
    }

    std::sort(free_blocks_.begin(), free_blocks_.end(), [](const FreeBlock& lhs, const FreeBlock& rhs) {
        return lhs.offset < rhs.offset;
    });

    std::vector<FreeBlock> merged;
    merged.reserve(free_blocks_.size());
    for (const FreeBlock& block : free_blocks_) {
        if (!merged.empty() && merged.back().offset + merged.back().length == block.offset) {
            merged.back().length += block.length;
        } else {
            merged.push_back(block);
        }
    }
    free_blocks_.swap(merged);

    while (!free_blocks_.empty()) {
        const FreeBlock& tail = free_blocks_.back();
        if (tail.offset + tail.length != allocated_elements_) {
            break;
        }
        allocated_elements_ = tail.offset;
        free_blocks_.pop_back();
    }
}

void CVStatePool::release_storage_block(int state_id) {
    const size_t reserved = host_state_capacities[state_id];
    if (reserved == 0) {
        host_state_dims[state_id] = 0;
        host_state_offsets[state_id] = 0;
        sync_state_metadata_to_device(state_id);
        return;
    }

    free_blocks_.push_back({host_state_offsets[state_id], reserved});
    host_state_dims[state_id] = 0;
    host_state_offsets[state_id] = 0;
    host_state_capacities[state_id] = 0;
    sync_state_metadata_to_device(state_id);
    merge_free_blocks();
}

void CVStatePool::assign_state_storage(int state_id, size_t required_elements) {
    if (required_elements == 0) {
        release_storage_block(state_id);
        return;
    }

    if (host_state_capacities[state_id] >= required_elements && host_state_capacities[state_id] != 0) {
        host_state_dims[state_id] = static_cast<int64_t>(required_elements);
        sync_state_metadata_to_device(state_id);
        return;
    }

    if (host_state_capacities[state_id] != 0) {
        release_storage_block(state_id);
    }

    host_state_offsets[state_id] = acquire_storage_block(required_elements);
    host_state_capacities[state_id] = required_elements;
    host_state_dims[state_id] = static_cast<int64_t>(required_elements);
    sync_state_metadata_to_device(state_id);
}

int CVStatePool::allocate_state() {
    if (free_state_ids.empty()) {
        std::cerr << "警告：状态池已满，无法分配新状态" << std::endl;
        return -1;
    }

    const int new_state_id = free_state_ids.back();
    free_state_ids.pop_back();
    active_flags[new_state_id] = 1;
    host_state_dims[new_state_id] = 0;
    host_state_offsets[new_state_id] = 0;
    host_state_capacities[new_state_id] = 0;
    sync_state_metadata_to_device(new_state_id);
    ++active_count;
    return new_state_id;
}

void CVStatePool::free_state(int state_id) {
    if (state_id < 0 || state_id >= capacity) {
        std::cerr << "警告：尝试释放无效的状态ID: " << state_id << std::endl;
        return;
    }

    if (!active_flags[state_id]) {
        std::cerr << "警告：状态ID未处于活跃状态: " << state_id << std::endl;
        return;
    }

    try {
        release_storage_block(state_id);
    } catch (const std::exception& ex) {
        std::cerr << "警告：释放状态存储失败: " << ex.what() << std::endl;
    }

    active_flags[state_id] = 0;
    free_state_ids.push_back(state_id);
    --active_count;
}

void CVStatePool::reserve_state_storage(int state_id, int64_t state_dim) {
    if (!is_valid_state(state_id)) {
        throw std::invalid_argument("无效的状态ID: " + std::to_string(state_id));
    }
    if (state_dim < 0) {
        throw std::invalid_argument("状态维度不能为负数");
    }
    assign_state_storage(state_id, static_cast<size_t>(state_dim));
}

size_t CVStatePool::allocate_detached_storage(size_t required_elements) {
    return acquire_storage_block(required_elements);
}

void CVStatePool::release_detached_storage(size_t offset, size_t reserved_elements) {
    if (reserved_elements == 0) {
        return;
    }
    free_blocks_.push_back({offset, reserved_elements});
    merge_free_blocks();
}

void CVStatePool::replace_state_storage(int state_id,
                                        size_t new_offset,
                                        size_t new_capacity,
                                        int state_dim) {
    if (!is_valid_state(state_id)) {
        throw std::invalid_argument("无效的状态ID: " + std::to_string(state_id));
    }
    if (state_dim < 0) {
        throw std::invalid_argument("状态维度不能为负数");
    }
    if (new_capacity < static_cast<size_t>(state_dim)) {
        throw std::invalid_argument("新存储块容量小于状态维度");
    }

    const size_t old_offset = host_state_offsets[state_id];
    const size_t old_capacity = host_state_capacities[state_id];

    host_state_offsets[state_id] = new_offset;
    host_state_capacities[state_id] = new_capacity;
    host_state_dims[state_id] = state_dim;
    sync_state_metadata_to_device(state_id);

    if (old_capacity != 0) {
        free_blocks_.push_back({old_offset, old_capacity});
        merge_free_blocks();
    }
}

void CVStatePool::upload_state(int state_id, const std::vector<cuDoubleComplex>& host_state) {
    if (!is_valid_state(state_id)) {
        throw std::invalid_argument("无效的状态ID: " + std::to_string(state_id));
    }

    reserve_state_storage(state_id, static_cast<int>(host_state.size()));
    if (host_state.empty()) {
        return;
    }

    const size_t offset = host_state_offsets[state_id];
    const cudaError_t err = cudaMemcpy(data + offset,
                                       host_state.data(),
                                       bytes_for_elements(host_state.size()),
                                       cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法上传状态到GPU: " + std::string(cudaGetErrorString(err)));
    }
}

void CVStatePool::download_state(int state_id, std::vector<cuDoubleComplex>& host_state) const {
    if (!is_valid_state(state_id)) {
        throw std::invalid_argument("无效的状态ID: " + std::to_string(state_id));
    }

    const int64_t state_dim = get_state_dim(state_id);
    host_state.resize(static_cast<size_t>(state_dim));
    if (state_dim == 0) {
        return;
    }

    const size_t offset = host_state_offsets[state_id];
    const cudaError_t err = cudaMemcpy(host_state.data(),
                                       data + offset,
                                       bytes_for_elements(static_cast<size_t>(state_dim)),
                                       cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法从GPU下载状态: " + std::string(cudaGetErrorString(err)));
    }
}

cuDoubleComplex* CVStatePool::get_state_ptr(int state_id) {
    if (!is_valid_state(state_id) || !data || host_state_capacities[state_id] == 0) {
        return nullptr;
    }
    return data + host_state_offsets[state_id];
}

const cuDoubleComplex* CVStatePool::get_state_ptr(int state_id) const {
    if (!is_valid_state(state_id) || !data || host_state_capacities[state_id] == 0) {
        return nullptr;
    }
    return data + host_state_offsets[state_id];
}

bool CVStatePool::is_valid_state(int state_id) const {
    return state_id >= 0 && state_id < capacity &&
           state_id < static_cast<int>(active_flags.size()) &&
           active_flags[state_id] != 0;
}

std::vector<int> CVStatePool::get_active_state_ids() const {
    std::vector<int> active_ids;
    if (active_count == 0) {
        return active_ids;
    }

    active_ids.reserve(active_count);
    for (int state_id = 0; state_id < capacity; ++state_id) {
        if (active_flags[state_id]) {
            active_ids.push_back(state_id);
        }
    }
    return active_ids;
}

void CVStatePool::reset() {
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess && sync_err != cudaErrorNotReady) {
        std::cerr << "警告：重置状态池前检测到GPU错误: " << cudaGetErrorString(sync_err) << std::endl;
        cudaGetLastError();
    }

    active_count = 0;
    active_flags.assign(capacity, 0);
    free_state_ids.clear();
    free_state_ids.reserve(capacity);
    for (int i = capacity - 1; i >= 0; --i) {
        free_state_ids.push_back(i);
    }

    host_state_dims.assign(capacity, 0);
    host_state_offsets.assign(capacity, 0);
    host_state_capacities.assign(capacity, 0);
    free_blocks_.clear();
    allocated_elements_ = 0;

    if (free_list) {
        std::vector<int> host_free_list(capacity);
        for (int i = 0; i < capacity; ++i) {
            host_free_list[i] = i;
        }
        cudaError_t err = cudaMemcpy(free_list,
                                     host_free_list.data(),
                                     capacity * sizeof(int),
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "重置空闲列表失败: " << cudaGetErrorString(err) << std::endl;
        }
    }

    if (state_dims) {
        cudaError_t err = cudaMemset(state_dims, 0, capacity * sizeof(int64_t));
        if (err != cudaSuccess) {
            std::cerr << "重置状态维度失败: " << cudaGetErrorString(err) << std::endl;
        }
    }
    if (state_offsets) {
        cudaError_t err = cudaMemset(state_offsets, 0, capacity * sizeof(size_t));
        if (err != cudaSuccess) {
            std::cerr << "重置状态偏移量失败: " << cudaGetErrorString(err) << std::endl;
        }
    }

    if (data) {
        cudaError_t err = cudaFree(data);
        if (err != cudaSuccess) {
            std::cerr << "警告：释放状态池数据失败: " << cudaGetErrorString(err) << std::endl;
        }
        data = nullptr;
    }
    data_capacity_elements_ = 0;
    scratch_target_ids.release();
    scratch_temp.release();
    scratch_aux.release();
    host_transfer_staging.release();
    refresh_total_memory_size();

    std::cout << "CVStatePool 已重置" << std::endl;
}

int CVStatePool::duplicate_state(int state_id) {
    if (!is_valid_state(state_id)) {
        std::cerr << "无效的状态ID: " << state_id << std::endl;
        return -1;
    }

    const int new_state_id = allocate_state();
    if (new_state_id == -1) {
        std::cerr << "无法分配新状态用于复制" << std::endl;
        return -1;
    }

    const int64_t state_dim = get_state_dim(state_id);
    try {
        reserve_state_storage(new_state_id, state_dim);
        if (state_dim > 0) {
            const cudaError_t err = cudaMemcpy(data + host_state_offsets[new_state_id],
                                               data + host_state_offsets[state_id],
                                               bytes_for_elements(static_cast<size_t>(state_dim)),
                                               cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("无法复制状态数据: " + std::string(cudaGetErrorString(err)));
            }
        }
    } catch (const std::exception& ex) {
        free_state(new_state_id);
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return new_state_id;
}

int64_t CVStatePool::get_state_dim(int state_id) const {
    if (!is_valid_state(state_id)) {
        return 0;
    }
    return host_state_dims[state_id];
}

int CVStatePool::tensor_product(int state1_id, int state2_id) {
    if (!is_valid_state(state1_id) || !is_valid_state(state2_id)) {
        std::cerr << "无效的状态ID: " << state1_id << ", " << state2_id << std::endl;
        return -1;
    }

    const int64_t dim1 = get_state_dim(state1_id);
    const int64_t dim2 = get_state_dim(state2_id);
    const size_t new_dim_size_t = static_cast<size_t>(dim1) * static_cast<size_t>(dim2);
    if (new_dim_size_t > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        std::cerr << "张量积维度过大" << std::endl;
        return -1;
    }
    const int64_t new_dim = static_cast<int64_t>(new_dim_size_t);

    const int new_state_id = allocate_state();
    if (new_state_id == -1) {
        std::cerr << "无法分配新状态用于张量积" << std::endl;
        return -1;
    }

    std::vector<cuDoubleComplex> state1_host;
    std::vector<cuDoubleComplex> state2_host;
    download_state(state1_id, state1_host);
    download_state(state2_id, state2_host);

    std::vector<cuDoubleComplex> product_state(new_dim_size_t, make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            product_state[static_cast<size_t>(i) * static_cast<size_t>(dim2) + static_cast<size_t>(j)] =
                cuCmul(state1_host[i], state2_host[j]);
        }
    }

    try {
        upload_state(new_state_id, product_state);
    } catch (const std::exception& ex) {
        free_state(new_state_id);
        std::cerr << "无法写入张量积状态: " << ex.what() << std::endl;
        return -1;
    }

    if (new_dim > max_total_dim) {
        max_total_dim = new_dim;
        total_dim = new_dim;
    }

    std::cout << "创建张量积: 状态" << state1_id << " (dim=" << dim1 << ") ⊗ 状态"
              << state2_id << " (dim=" << dim2 << ") -> 状态" << new_state_id
              << " (dim=" << new_dim << ")" << std::endl;

    return new_state_id;
}
