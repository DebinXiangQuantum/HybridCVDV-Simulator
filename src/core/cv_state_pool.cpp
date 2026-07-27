#include "cv_state_pool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

size_t bytes_for_elements(size_t elements) {
    return elements * sizeof(cuDoubleComplex);
}

size_t metadata_bytes_for_capacity(int capacity) {
    return static_cast<size_t>(capacity) * sizeof(int) +
           static_cast<size_t>(capacity) * sizeof(int64_t) +
           static_cast<size_t>(capacity) * sizeof(size_t);
}

void check_cuda(cudaError_t err, const std::string& message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(message + ": " + std::string(cudaGetErrorString(err)));
    }
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

    int current_device = 0;
    cudaError_t current_device_err = cudaGetDevice(&current_device);
    if (current_device_err != cudaSuccess) {
        current_device = 0;
        cudaGetLastError();
    }

    for (int i = 0; i < num_qumodes; ++i) {
        if (max_total_dim > std::numeric_limits<int64_t>::max() / d_trunc) {
            throw std::overflow_error("状态空间维度溢出");
        }
        max_total_dim *= d_trunc;
    }
    total_dim = max_total_dim;

    device_ids_.reserve(device_count);
    device_views_.resize(device_count);
    for (int device_id = 0; device_id < device_count; ++device_id) {
        device_ids_.push_back(device_id);
        initialize_device_metadata(device_views_[device_id], device_id);
    }

    free_state_ids.reserve(capacity);
    for (int i = capacity - 1; i >= 0; --i) {
        free_state_ids.push_back(i);
    }
    active_flags.assign(static_cast<size_t>(capacity), 0);
    host_state_dims.assign(static_cast<size_t>(capacity), 0);
    host_state_offsets.assign(static_cast<size_t>(capacity), 0);
    host_state_capacities.assign(static_cast<size_t>(capacity), 0);
    host_state_devices_.assign(static_cast<size_t>(capacity), -1);

    active_device_id_ = -1;
    activate_device_view(current_device);
    refresh_total_memory_size();

    std::cout << "CVStatePool 初始化完成: 单个qumode截断维度=" << d_trunc
              << ", 初始总维度=" << max_total_dim
              << ", 容量=" << capacity
              << ", 设备数=" << device_ids_.size()
              << std::endl;
}

CVStatePool::~CVStatePool() {
    cudaGetLastError();

    const int previous_active = active_device_id_;
    for (int device_id : device_ids_) {
        cudaSetDevice(device_id);
        cudaDeviceSynchronize();
        cudaGetLastError();

        if (active_device_id_ != device_id) {
            activate_device_view(device_id);
        }

        release_active_data_and_scratch();

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
        metadata_memory_size_ = 0;
    }

    if (previous_active >= 0 && previous_active < static_cast<int>(device_ids_.size())) {
        cudaSetDevice(previous_active);
    }

    std::cout << "CVStatePool 销毁完成" << std::endl;
}

void CVStatePool::initialize_device_metadata(DeviceStorage& storage, int device_id) {
    storage.device_id = device_id;
    storage.data = nullptr;
    storage.free_list = nullptr;
    storage.state_dims = nullptr;
    storage.state_offsets = nullptr;
    storage.data_capacity_elements = 0;
    storage.allocated_elements = 0;
    storage.metadata_memory_size = metadata_bytes_for_capacity(capacity);
    storage.free_blocks.clear();

    check_cuda(cudaSetDevice(device_id), "无法设置CUDA设备");

    auto cleanup = [&]() {
        if (storage.free_list) {
            cudaFree(storage.free_list);
            storage.free_list = nullptr;
        }
        if (storage.state_dims) {
            cudaFree(storage.state_dims);
            storage.state_dims = nullptr;
        }
        if (storage.state_offsets) {
            cudaFree(storage.state_offsets);
            storage.state_offsets = nullptr;
        }
    };

    cudaError_t err = cudaMalloc(&storage.free_list, static_cast<size_t>(capacity) * sizeof(int));
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法分配GPU内存用于空闲列表: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&storage.state_dims, static_cast<size_t>(capacity) * sizeof(int64_t));
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法分配GPU内存用于状态维度: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&storage.state_offsets, static_cast<size_t>(capacity) * sizeof(size_t));
    if (err != cudaSuccess) {
        cleanup();
        throw std::runtime_error("无法分配GPU内存用于状态偏移量: " +
                                 std::string(cudaGetErrorString(err)));
    }

    std::vector<int> host_free_list(static_cast<size_t>(capacity));
    for (int i = 0; i < capacity; ++i) {
        host_free_list[static_cast<size_t>(i)] = i;
    }

    check_cuda(cudaMemcpy(storage.free_list,
                          host_free_list.data(),
                          static_cast<size_t>(capacity) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "无法初始化空闲列表");
    check_cuda(cudaMemset(storage.state_dims, 0, static_cast<size_t>(capacity) * sizeof(int64_t)),
               "无法初始化状态维度");
    check_cuda(cudaMemset(storage.state_offsets, 0, static_cast<size_t>(capacity) * sizeof(size_t)),
               "无法初始化状态偏移量");
}

void CVStatePool::swap_active_view(DeviceStorage& storage) {
    std::swap(data, storage.data);
    std::swap(free_list, storage.free_list);
    std::swap(state_dims, storage.state_dims);
    std::swap(state_offsets, storage.state_offsets);
    std::swap(scratch_target_ids, storage.scratch_target_ids);
    std::swap(scratch_temp, storage.scratch_temp);
    std::swap(scratch_aux, storage.scratch_aux);
    std::swap(host_transfer_staging, storage.host_transfer_staging);
    std::swap(data_capacity_elements_, storage.data_capacity_elements);
    std::swap(allocated_elements_, storage.allocated_elements);
    std::swap(metadata_memory_size_, storage.metadata_memory_size);
    std::swap(free_blocks_, storage.free_blocks);
}

void CVStatePool::activate_device_view(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(device_views_.size())) {
        throw std::out_of_range("无效的CVStatePool device_id: " + std::to_string(device_id));
    }
    if (active_device_id_ == device_id) {
        return;
    }

    if (active_device_id_ >= 0) {
        swap_active_view(device_views_[active_device_id_]);
    }
    swap_active_view(device_views_[device_id]);
    active_device_id_ = device_id;
}

void CVStatePool::refresh_total_memory_size() {
    size_t total = metadata_memory_size_ + bytes_for_elements(data_capacity_elements_);
    for (const DeviceStorage& storage : device_views_) {
        total += storage.metadata_memory_size + bytes_for_elements(storage.data_capacity_elements);
    }
    total_memory_size = total;
}

void CVStatePool::release_device_scratch_buffers() {
    scratch_target_ids.release();
    scratch_temp.release();
    scratch_aux.release();
}

void CVStatePool::release_active_data_and_scratch() {
    if (data) {
        cudaError_t err = cudaFree(data);
        if (err != cudaSuccess && err != cudaErrorProfilerNotInitialized) {
            std::cerr << "警告：释放状态池数据内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        data = nullptr;
    }
    data_capacity_elements_ = 0;
    allocated_elements_ = 0;
    free_blocks_.clear();
    scratch_target_ids.release();
    scratch_temp.release();
    scratch_aux.release();
    host_transfer_staging.release();
}

size_t CVStatePool::active_storage_elements() const {
    size_t live_elements = 0;
    for (int state_id = 0; state_id < capacity; ++state_id) {
        if (!active_flags[static_cast<size_t>(state_id)]) {
            continue;
        }
        live_elements += host_state_capacities[static_cast<size_t>(state_id)];
    }
    return live_elements;
}

size_t CVStatePool::active_storage_elements_on_device(int device_id) const {
    size_t live_elements = 0;
    for (int state_id = 0; state_id < capacity; ++state_id) {
        if (!active_flags[static_cast<size_t>(state_id)] ||
            host_state_devices_[static_cast<size_t>(state_id)] != device_id) {
            continue;
        }
        live_elements += host_state_capacities[static_cast<size_t>(state_id)];
    }
    return live_elements;
}

size_t CVStatePool::get_active_storage_elements() const {
    return active_storage_elements();
}

size_t CVStatePool::get_active_storage_elements_on_device(int device_id) const {
    return active_storage_elements_on_device(device_id);
}

void CVStatePool::sync_state_metadata_to_device(int state_id) {
    const int previous_active = active_device_id_;
    const int owner_device =
        (state_id >= 0 && state_id < static_cast<int>(host_state_devices_.size()) &&
         active_flags[static_cast<size_t>(state_id)] != 0)
            ? host_state_devices_[static_cast<size_t>(state_id)]
            : -1;

    for (int device_id : device_ids_) {
        check_cuda(cudaSetDevice(device_id), "无法设置CUDA设备同步状态元数据");
        activate_device_view(device_id);

        const int64_t dim = (device_id == owner_device)
                                ? host_state_dims[static_cast<size_t>(state_id)]
                                : 0;
        const size_t offset = (device_id == owner_device)
                                  ? host_state_offsets[static_cast<size_t>(state_id)]
                                  : 0;

        check_cuda(cudaMemcpy(state_dims + state_id,
                              &dim,
                              sizeof(int64_t),
                              cudaMemcpyHostToDevice),
                   "无法同步状态维度");
        check_cuda(cudaMemcpy(state_offsets + state_id,
                              &offset,
                              sizeof(size_t),
                              cudaMemcpyHostToDevice),
                   "无法同步状态偏移量");
    }

    if (previous_active >= 0) {
        check_cuda(cudaSetDevice(previous_active), "无法恢复CUDA设备");
        activate_device_view(previous_active);
    }
}

void CVStatePool::grow_state_capacity(int min_capacity) {
    if (min_capacity <= capacity) {
        return;
    }

    const int old_capacity = capacity;
    const int new_capacity = std::max(min_capacity, std::max(capacity * 2, capacity + 1));
    const size_t new_metadata_bytes = metadata_bytes_for_capacity(new_capacity);
    if (max_memory_size > 0 && new_metadata_bytes >= max_memory_size) {
        throw std::runtime_error("状态池元数据扩容后将超出内存上限");
    }

    const int previous_active = active_device_id_;
    std::vector<int> host_free_list(static_cast<size_t>(new_capacity));
    for (int i = 0; i < new_capacity; ++i) {
        host_free_list[static_cast<size_t>(i)] = i;
    }

    for (int device_id : device_ids_) {
        check_cuda(cudaSetDevice(device_id), "无法设置CUDA设备扩容元数据");
        check_cuda(cudaDeviceSynchronize(), "无法在扩容元数据前同步CUDA设备");
        activate_device_view(device_id);

        int* new_free_list = nullptr;
        int64_t* new_state_dims = nullptr;
        size_t* new_state_offsets = nullptr;

        auto cleanup = [&]() {
            if (new_free_list) {
                cudaFree(new_free_list);
                new_free_list = nullptr;
            }
            if (new_state_dims) {
                cudaFree(new_state_dims);
                new_state_dims = nullptr;
            }
            if (new_state_offsets) {
                cudaFree(new_state_offsets);
                new_state_offsets = nullptr;
            }
        };

        cudaError_t err = cudaMalloc(&new_free_list, static_cast<size_t>(new_capacity) * sizeof(int));
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("无法扩展GPU空闲列表元数据: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&new_state_dims, static_cast<size_t>(new_capacity) * sizeof(int64_t));
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("无法扩展GPU状态维度元数据: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&new_state_offsets, static_cast<size_t>(new_capacity) * sizeof(size_t));
        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("无法扩展GPU状态偏移元数据: " +
                                     std::string(cudaGetErrorString(err)));
        }

        check_cuda(cudaMemcpy(new_free_list,
                              host_free_list.data(),
                              static_cast<size_t>(new_capacity) * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "无法初始化扩容后的空闲列表");

        if (state_dims && old_capacity > 0) {
            check_cuda(cudaMemcpy(new_state_dims,
                                  state_dims,
                                  static_cast<size_t>(old_capacity) * sizeof(int64_t),
                                  cudaMemcpyDeviceToDevice),
                       "无法复制扩容前的状态维度元数据");
        }
        if (new_capacity > old_capacity) {
            check_cuda(cudaMemset(new_state_dims + old_capacity,
                                  0,
                                  static_cast<size_t>(new_capacity - old_capacity) * sizeof(int64_t)),
                       "无法清零新增状态维度元数据");
        }

        if (state_offsets && old_capacity > 0) {
            check_cuda(cudaMemcpy(new_state_offsets,
                                  state_offsets,
                                  static_cast<size_t>(old_capacity) * sizeof(size_t),
                                  cudaMemcpyDeviceToDevice),
                       "无法复制扩容前的状态偏移元数据");
        }
        if (new_capacity > old_capacity) {
            check_cuda(cudaMemset(new_state_offsets + old_capacity,
                                  0,
                                  static_cast<size_t>(new_capacity - old_capacity) * sizeof(size_t)),
                       "无法清零新增状态偏移元数据");
        }

        if (free_list) {
            cudaFree(free_list);
        }
        if (state_dims) {
            cudaFree(state_dims);
        }
        if (state_offsets) {
            cudaFree(state_offsets);
        }

        free_list = new_free_list;
        state_dims = new_state_dims;
        state_offsets = new_state_offsets;
        metadata_memory_size_ = new_metadata_bytes;
    }

    if (previous_active >= 0) {
        check_cuda(cudaSetDevice(previous_active), "无法恢复CUDA设备");
        activate_device_view(previous_active);
    }

    capacity = new_capacity;
    active_flags.resize(static_cast<size_t>(new_capacity), 0);
    host_state_dims.resize(static_cast<size_t>(new_capacity), 0);
    host_state_offsets.resize(static_cast<size_t>(new_capacity), 0);
    host_state_capacities.resize(static_cast<size_t>(new_capacity), 0);
    host_state_devices_.resize(static_cast<size_t>(new_capacity), -1);

    free_state_ids.reserve(static_cast<size_t>(new_capacity));
    for (int state_id = new_capacity - 1; state_id >= old_capacity; --state_id) {
        free_state_ids.push_back(state_id);
    }

    refresh_total_memory_size();
}

void CVStatePool::ensure_data_capacity(size_t required_elements) {
    if (required_elements <= data_capacity_elements_) {
        return;
    }

    check_cuda(cudaDeviceSynchronize(), "无法在扩容状态池数据前同步CUDA设备");

    auto try_repack_live_storage = [&](size_t target_capacity, cuDoubleComplex** out_data) -> bool {
        const size_t live_elements = active_storage_elements_on_device(active_device_id_);
        if (!data || live_elements == 0 || live_elements >= data_capacity_elements_) {
            return false;
        }

        std::vector<int> active_state_ids;
        active_state_ids.reserve(active_count);
        for (int state_id = 0; state_id < capacity; ++state_id) {
            if (!active_flags[static_cast<size_t>(state_id)] ||
                host_state_devices_[static_cast<size_t>(state_id)] != active_device_id_) {
                continue;
            }
            active_state_ids.push_back(state_id);
        }

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
                const size_t reserved = host_state_capacities[static_cast<size_t>(state_id)];
                if (reserved == 0) {
                    continue;
                }

                compact_offsets[static_cast<size_t>(state_id)] = compact_cursor;
                const cudaError_t copy_err = cudaMemcpy(
                    compact_data + compact_cursor,
                    data + host_state_offsets[static_cast<size_t>(state_id)],
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
                const size_t reserved = host_state_capacities[static_cast<size_t>(state_id)];
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

                host_state_offsets[static_cast<size_t>(state_id)] = compact_offset;
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
        const size_t live_elements = active_storage_elements_on_device(active_device_id_);
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
                " (device=" + std::to_string(active_device_id_) +
                ", required_elements=" + std::to_string(required_elements) +
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
                throw std::runtime_error("无法迁移状态池数据: " +
                                         std::string(cudaGetErrorString(copy_err)));
            }
        }

        if (data) {
            cudaError_t free_err = cudaFree(data);
            if (free_err != cudaSuccess) {
                std::cerr << "警告：释放旧状态池数据失败: "
                          << cudaGetErrorString(free_err) << std::endl;
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

void CVStatePool::reserve_total_storage_elements_on_device(int device_id,
                                                           size_t required_elements) {
    if (std::find(device_ids_.begin(), device_ids_.end(), device_id) == device_ids_.end()) {
        throw std::out_of_range("无效的CVStatePool device_id: " + std::to_string(device_id));
    }

    const int previous_active = active_device_id_;
    check_cuda(cudaSetDevice(device_id), "无法设置CUDA设备预留状态池容量");
    activate_device_view(device_id);
    try {
        ensure_data_capacity(required_elements);
    } catch (...) {
        if (previous_active >= 0 && previous_active != device_id) {
            check_cuda(cudaSetDevice(previous_active), "无法恢复CUDA设备");
            activate_device_view(previous_active);
        }
        throw;
    }

    if (previous_active >= 0 && previous_active != device_id) {
        check_cuda(cudaSetDevice(previous_active), "无法恢复CUDA设备");
        activate_device_view(previous_active);
    }
}

void CVStatePool::synchronize_all_devices() {
    int current_device = active_device_id_;
    if (current_device < 0) {
        cudaError_t err = cudaGetDevice(&current_device);
        if (err != cudaSuccess) {
            current_device = device_ids_.empty() ? 0 : device_ids_.front();
            cudaGetLastError();
        }
    }

    for (int device_id : device_ids_) {
        check_cuda(cudaSetDevice(device_id), "无法设置CUDA设备进行全局同步");
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess && err != cudaErrorNotReady) {
            throw std::runtime_error(
                "多GPU同步失败(device=" + std::to_string(device_id) + "): " +
                std::string(cudaGetErrorString(err)));
        }
    }

    if (current_device >= 0) {
        check_cuda(cudaSetDevice(current_device), "无法恢复CUDA设备");
        if (std::find(device_ids_.begin(), device_ids_.end(), current_device) != device_ids_.end()) {
            activate_device_view(current_device);
        }
    }
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

    std::sort(free_blocks_.begin(), free_blocks_.end(),
              [](const FreeBlock& lhs, const FreeBlock& rhs) {
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

int CVStatePool::choose_device_for_storage(size_t required_elements, int preferred_device) {
    if (device_ids_.empty()) {
        return 0;
    }
    if (device_ids_.size() == 1) {
        return device_ids_.front();
    }

    const size_t required_bytes = bytes_for_elements(required_elements);
    constexpr size_t kSafetyBytes = 256ULL * 1024ULL * 1024ULL;
    const size_t rr_start =
        static_cast<size_t>(next_round_robin_device_index_ % static_cast<int>(device_ids_.size()));

    int current_device = 0;
    cudaError_t current_device_err = cudaGetDevice(&current_device);
    if (current_device_err != cudaSuccess) {
        current_device = active_device_id_ >= 0 ? active_device_id_ : device_ids_.front();
        cudaGetLastError();
    }

    int fallback_device = device_ids_[rr_start];
    size_t fallback_free_bytes = 0;
    size_t fallback_rr_rank = device_ids_.size();
    int best_fit_device = -1;
    size_t best_fit_reserved_bytes = std::numeric_limits<size_t>::max();
    size_t best_fit_rr_rank = device_ids_.size();
    size_t preferred_free_bytes = 0;

    for (size_t index = 0; index < device_ids_.size(); ++index) {
        const int device_id = device_ids_[index];
        const size_t rr_rank =
            (index + device_ids_.size() - rr_start) % device_ids_.size();
        cudaSetDevice(device_id);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        if (free_bytes > fallback_free_bytes ||
            (free_bytes == fallback_free_bytes && rr_rank < fallback_rr_rank)) {
            fallback_free_bytes = free_bytes;
            fallback_device = device_id;
            fallback_rr_rank = rr_rank;
        }
        size_t reserved_bytes = 0;
        for (size_t state_id = 0; state_id < host_state_devices_.size(); ++state_id) {
            if (host_state_devices_[state_id] == device_id) {
                reserved_bytes += bytes_for_elements(host_state_capacities[state_id]);
            }
        }
        if (free_bytes > required_bytes + kSafetyBytes &&
            (reserved_bytes < best_fit_reserved_bytes ||
             (reserved_bytes == best_fit_reserved_bytes && rr_rank < best_fit_rr_rank))) {
            best_fit_reserved_bytes = reserved_bytes;
            best_fit_device = device_id;
            best_fit_rr_rank = rr_rank;
        }
        if (device_id == preferred_device) {
            preferred_free_bytes = free_bytes;
        }
    }

    cudaSetDevice(current_device);

    int selected_device = fallback_device;
    if (preferred_device >= 0 && preferred_free_bytes > required_bytes + (kSafetyBytes / 2)) {
        selected_device = preferred_device;
    } else if (best_fit_device >= 0) {
        selected_device = best_fit_device;
    }

    const auto selected_it =
        std::find(device_ids_.begin(), device_ids_.end(), selected_device);
    if (selected_it != device_ids_.end()) {
        next_round_robin_device_index_ =
            static_cast<int>((std::distance(device_ids_.begin(), selected_it) + 1) %
                             static_cast<std::ptrdiff_t>(device_ids_.size()));
    }
    return selected_device;
}

int CVStatePool::recommend_device_for_storage(size_t required_elements, int preferred_device) {
    return choose_device_for_storage(required_elements, preferred_device);
}

void CVStatePool::ensure_state_device_assigned(int state_id, size_t required_elements) {
    if (host_state_devices_[static_cast<size_t>(state_id)] >= 0) {
        const int owner = host_state_devices_[static_cast<size_t>(state_id)];
        check_cuda(cudaSetDevice(owner), "无法设置状态所属CUDA设备");
        activate_device_view(owner);
        return;
    }

    const int owner = choose_device_for_storage(required_elements);
    host_state_devices_[static_cast<size_t>(state_id)] = owner;
    check_cuda(cudaSetDevice(owner), "无法设置新状态所属CUDA设备");
    activate_device_view(owner);
    sync_state_metadata_to_device(state_id);
}

void CVStatePool::release_storage_block(int state_id) {
    const size_t reserved = host_state_capacities[static_cast<size_t>(state_id)];
    if (reserved == 0) {
        host_state_dims[static_cast<size_t>(state_id)] = 0;
        host_state_offsets[static_cast<size_t>(state_id)] = 0;
        sync_state_metadata_to_device(state_id);
        return;
    }

    free_blocks_.push_back({host_state_offsets[static_cast<size_t>(state_id)], reserved});
    host_state_dims[static_cast<size_t>(state_id)] = 0;
    host_state_offsets[static_cast<size_t>(state_id)] = 0;
    host_state_capacities[static_cast<size_t>(state_id)] = 0;
    sync_state_metadata_to_device(state_id);
    merge_free_blocks();
}

void CVStatePool::assign_state_storage(int state_id, size_t required_elements) {
    if (required_elements == 0) {
        release_storage_block(state_id);
        return;
    }

    ensure_state_device_assigned(state_id, required_elements);

    if (host_state_capacities[static_cast<size_t>(state_id)] >= required_elements &&
        host_state_capacities[static_cast<size_t>(state_id)] != 0) {
        host_state_dims[static_cast<size_t>(state_id)] = static_cast<int64_t>(required_elements);
        sync_state_metadata_to_device(state_id);
        return;
    }

    if (host_state_capacities[static_cast<size_t>(state_id)] != 0) {
        release_storage_block(state_id);
    }

    host_state_offsets[static_cast<size_t>(state_id)] = acquire_storage_block(required_elements);
    host_state_capacities[static_cast<size_t>(state_id)] = required_elements;
    host_state_dims[static_cast<size_t>(state_id)] = static_cast<int64_t>(required_elements);
    sync_state_metadata_to_device(state_id);
}

int CVStatePool::allocate_state(int preferred_device) {
    if (free_state_ids.empty()) {
        try {
            grow_state_capacity(capacity + 1);
        } catch (const std::exception& ex) {
            std::cerr << "警告：状态池已满，且自动扩容失败: " << ex.what() << std::endl;
            return -1;
        }
    }

    const int new_state_id = free_state_ids.back();
    free_state_ids.pop_back();
    active_flags[static_cast<size_t>(new_state_id)] = 1;
    host_state_dims[static_cast<size_t>(new_state_id)] = 0;
    host_state_offsets[static_cast<size_t>(new_state_id)] = 0;
    host_state_capacities[static_cast<size_t>(new_state_id)] = 0;
    host_state_devices_[static_cast<size_t>(new_state_id)] =
        (preferred_device >= 0 && preferred_device < static_cast<int>(device_ids_.size()))
            ? preferred_device
            : -1;
    sync_state_metadata_to_device(new_state_id);
    ++active_count;
    return new_state_id;
}

void CVStatePool::free_state(int state_id) {
    if (state_id < 0 || state_id >= capacity) {
        std::cerr << "警告：尝试释放无效的状态ID: " << state_id << std::endl;
        return;
    }

    if (!active_flags[static_cast<size_t>(state_id)]) {
        std::cerr << "警告：状态ID未处于活跃状态: " << state_id << std::endl;
        return;
    }

    try {
        const int owner = host_state_devices_[static_cast<size_t>(state_id)];
        if (owner >= 0) {
            check_cuda(cudaSetDevice(owner), "无法设置CUDA设备释放状态");
            activate_device_view(owner);
        }
        release_storage_block(state_id);
    } catch (const std::exception& ex) {
        std::cerr << "警告：释放状态存储失败: " << ex.what() << std::endl;
    }

    host_state_devices_[static_cast<size_t>(state_id)] = -1;
    active_flags[static_cast<size_t>(state_id)] = 0;
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

    if (host_state_devices_[static_cast<size_t>(state_id)] < 0) {
        host_state_devices_[static_cast<size_t>(state_id)] = active_device_id_;
    }
    if (host_state_devices_[static_cast<size_t>(state_id)] != active_device_id_) {
        throw std::runtime_error("replace_state_storage 必须在状态所属device view上调用");
    }

    const size_t old_offset = host_state_offsets[static_cast<size_t>(state_id)];
    const size_t old_capacity = host_state_capacities[static_cast<size_t>(state_id)];

    host_state_offsets[static_cast<size_t>(state_id)] = new_offset;
    host_state_capacities[static_cast<size_t>(state_id)] = new_capacity;
    host_state_dims[static_cast<size_t>(state_id)] = state_dim;
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

    reserve_state_storage(state_id, static_cast<int64_t>(host_state.size()));
    if (host_state.empty()) {
        return;
    }

    const size_t offset = host_state_offsets[static_cast<size_t>(state_id)];
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

    const int owner = host_state_devices_[static_cast<size_t>(state_id)];
    CVStatePool* self = const_cast<CVStatePool*>(this);
    check_cuda(cudaSetDevice(owner), "无法设置CUDA设备下载状态");
    self->activate_device_view(owner);

    const size_t offset = host_state_offsets[static_cast<size_t>(state_id)];
    const cudaError_t err = cudaMemcpy(host_state.data(),
                                       data + offset,
                                       bytes_for_elements(static_cast<size_t>(state_dim)),
                                       cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法从GPU下载状态: " + std::string(cudaGetErrorString(err)));
    }
}

cuDoubleComplex* CVStatePool::get_state_ptr(int state_id) {
    if (!is_valid_state(state_id) || host_state_capacities[static_cast<size_t>(state_id)] == 0) {
        return nullptr;
    }
    const int owner = host_state_devices_[static_cast<size_t>(state_id)];
    if (owner < 0) {
        return nullptr;
    }
    if (active_device_id_ != owner) {
        check_cuda(cudaSetDevice(owner), "无法设置状态所属CUDA设备");
        activate_device_view(owner);
    }
    if (!data) {
        return nullptr;
    }
    return data + host_state_offsets[static_cast<size_t>(state_id)];
}

const cuDoubleComplex* CVStatePool::get_state_ptr(int state_id) const {
    return const_cast<CVStatePool*>(this)->get_state_ptr(state_id);
}

bool CVStatePool::is_valid_state(int state_id) const {
    return state_id >= 0 && state_id < capacity &&
           state_id < static_cast<int>(active_flags.size()) &&
           active_flags[static_cast<size_t>(state_id)] != 0;
}

std::vector<int> CVStatePool::get_active_state_ids() const {
    std::vector<int> active_ids;
    if (active_count == 0) {
        return active_ids;
    }

    active_ids.reserve(active_count);
    for (int state_id = 0; state_id < capacity; ++state_id) {
        if (active_flags[static_cast<size_t>(state_id)]) {
            active_ids.push_back(state_id);
        }
    }
    return active_ids;
}

int CVStatePool::get_state_device_id(int state_id) const {
    if (!is_valid_state(state_id)) {
        return -1;
    }
    return host_state_devices_[static_cast<size_t>(state_id)];
}

void CVStatePool::migrate_state_to_device(int state_id, int target_device) {
    if (!is_valid_state(state_id)) {
        throw std::invalid_argument("无法迁移无效状态ID: " + std::to_string(state_id));
    }
    if (std::find(device_ids_.begin(), device_ids_.end(), target_device) == device_ids_.end()) {
        throw std::invalid_argument("无法迁移到无效CUDA设备: " + std::to_string(target_device));
    }

    const int source_device = host_state_devices_[static_cast<size_t>(state_id)];
    if (source_device == target_device) {
        return;
    }
    const size_t elements = static_cast<size_t>(host_state_dims[static_cast<size_t>(state_id)]);
    if (source_device < 0 || elements == 0) {
        host_state_devices_[static_cast<size_t>(state_id)] = target_device;
        activate_device_view(target_device);
        sync_state_metadata_to_device(state_id);
        return;
    }

    activate_device_view(source_device);
    const cuDoubleComplex* source_ptr =
        data + host_state_offsets[static_cast<size_t>(state_id)];
    release_storage_block(state_id);

    host_state_devices_[static_cast<size_t>(state_id)] = target_device;
    activate_device_view(target_device);
    assign_state_storage(state_id, elements);
    cuDoubleComplex* target_ptr =
        data + host_state_offsets[static_cast<size_t>(state_id)];
    copy_state_between_devices(
        source_ptr, source_device, target_ptr, target_device, elements);
    sync_state_metadata_to_device(state_id);
}

std::vector<std::pair<int, std::vector<int>>> CVStatePool::bucket_state_ids_by_device(
    const std::vector<int>& state_ids) const {
    std::vector<std::pair<int, std::vector<int>>> buckets;
    buckets.reserve(device_ids_.size());

    for (int device_id : device_ids_) {
        std::vector<int> local_ids;
        local_ids.reserve(state_ids.size());
        for (int state_id : state_ids) {
            if (is_valid_state(state_id) &&
                host_state_devices_[static_cast<size_t>(state_id)] == device_id) {
                local_ids.push_back(state_id);
            }
        }
        if (!local_ids.empty()) {
            buckets.emplace_back(device_id, std::move(local_ids));
        }
    }
    return buckets;
}

bool CVStatePool::spans_multiple_devices(const std::vector<int>& state_ids) const {
    int seen_device = -1;
    for (int state_id : state_ids) {
        if (!is_valid_state(state_id)) {
            continue;
        }
        const int device_id = host_state_devices_[static_cast<size_t>(state_id)];
        if (device_id < 0) {
            continue;
        }
        if (seen_device < 0) {
            seen_device = device_id;
            continue;
        }
        if (seen_device != device_id) {
            return true;
        }
    }
    return false;
}

std::vector<CVStatePool::DeviceMemoryStats> CVStatePool::get_device_memory_stats() const {
    std::vector<DeviceMemoryStats> stats;
    stats.reserve(device_ids_.size());
    for (int device_id : device_ids_) {
        DeviceMemoryStats item;
        item.device_id = device_id;
        for (int state_id = 0; state_id < capacity; ++state_id) {
            if (active_flags[static_cast<size_t>(state_id)] &&
                host_state_devices_[static_cast<size_t>(state_id)] == device_id) {
                ++item.active_state_count;
                item.active_bytes +=
                    bytes_for_elements(host_state_capacities[static_cast<size_t>(state_id)]);
            }
        }

        if (device_id == active_device_id_) {
            item.reserved_bytes = bytes_for_elements(data_capacity_elements_);
            item.metadata_bytes = metadata_memory_size_;
            item.scratch_bytes = scratch_target_ids.capacity_bytes +
                                 scratch_temp.capacity_bytes +
                                 scratch_aux.capacity_bytes;
        } else {
            const DeviceStorage& storage = device_views_[static_cast<size_t>(device_id)];
            item.reserved_bytes = bytes_for_elements(storage.data_capacity_elements);
            item.metadata_bytes = storage.metadata_memory_size;
            item.scratch_bytes = storage.scratch_target_ids.capacity_bytes +
                                 storage.scratch_temp.capacity_bytes +
                                 storage.scratch_aux.capacity_bytes;
        }
        stats.push_back(item);
    }
    return stats;
}

void CVStatePool::reset() {
    active_count = 0;
    active_flags.assign(static_cast<size_t>(capacity), 0);
    free_state_ids.clear();
    free_state_ids.reserve(static_cast<size_t>(capacity));
    for (int i = capacity - 1; i >= 0; --i) {
        free_state_ids.push_back(i);
    }

    host_state_dims.assign(static_cast<size_t>(capacity), 0);
    host_state_offsets.assign(static_cast<size_t>(capacity), 0);
    host_state_capacities.assign(static_cast<size_t>(capacity), 0);
    host_state_devices_.assign(static_cast<size_t>(capacity), -1);

    const int previous_active = active_device_id_;
    std::vector<int> host_free_list(static_cast<size_t>(capacity));
    for (int i = 0; i < capacity; ++i) {
        host_free_list[static_cast<size_t>(i)] = i;
    }

    for (int device_id : device_ids_) {
        check_cuda(cudaSetDevice(device_id), "无法设置CUDA设备重置状态池");
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess && sync_err != cudaErrorNotReady) {
            std::cerr << "警告：重置状态池前检测到GPU错误(device=" << device_id
                      << "): " << cudaGetErrorString(sync_err) << std::endl;
            cudaGetLastError();
        }
        activate_device_view(device_id);

        free_blocks_.clear();
        allocated_elements_ = 0;

        if (free_list) {
            cudaError_t err = cudaMemcpy(free_list,
                                         host_free_list.data(),
                                         static_cast<size_t>(capacity) * sizeof(int),
                                         cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "重置空闲列表失败: " << cudaGetErrorString(err) << std::endl;
            }
        }
        if (state_dims) {
            cudaError_t err = cudaMemset(state_dims, 0, static_cast<size_t>(capacity) * sizeof(int64_t));
            if (err != cudaSuccess) {
                std::cerr << "重置状态维度失败: " << cudaGetErrorString(err) << std::endl;
            }
        }
        if (state_offsets) {
            cudaError_t err = cudaMemset(state_offsets, 0, static_cast<size_t>(capacity) * sizeof(size_t));
            if (err != cudaSuccess) {
                std::cerr << "重置状态偏移量失败: " << cudaGetErrorString(err) << std::endl;
            }
        }

        release_active_data_and_scratch();
    }

    if (previous_active >= 0) {
        check_cuda(cudaSetDevice(previous_active), "无法恢复CUDA设备");
        activate_device_view(previous_active);
    }

    refresh_total_memory_size();
    std::cout << "CVStatePool 已重置" << std::endl;
}

void CVStatePool::copy_state_between_devices(const cuDoubleComplex* src_ptr,
                                             int src_device,
                                             cuDoubleComplex* dst_ptr,
                                             int dst_device,
                                             size_t elements) const {
    if (elements == 0) {
        return;
    }

    constexpr size_t kChunkBytes = 1ULL << 30;
    const size_t total_bytes = bytes_for_elements(elements);
    const char* src_bytes = reinterpret_cast<const char*>(src_ptr);
    char* dst_bytes = reinterpret_cast<char*>(dst_ptr);

    if (src_device == dst_device) {
        check_cuda(cudaSetDevice(src_device), "无法设置同设备拷贝CUDA设备");
        for (size_t offset = 0; offset < total_bytes; offset += kChunkBytes) {
            const size_t chunk = std::min(kChunkBytes, total_bytes - offset);
            check_cuda(cudaMemcpy(dst_bytes + offset,
                                  src_bytes + offset,
                                  chunk,
                                  cudaMemcpyDeviceToDevice),
                       "无法执行同设备状态复制");
        }
        return;
    }

    const auto started_at = std::chrono::steady_clock::now();
    int can_access_peer = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access_peer, dst_device, src_device),
               "无法查询CUDA peer access");
    if (can_access_peer) {
        for (size_t offset = 0; offset < total_bytes; offset += kChunkBytes) {
            const size_t chunk = std::min(kChunkBytes, total_bytes - offset);
            check_cuda(cudaMemcpyPeer(dst_bytes + offset,
                                      dst_device,
                                      src_bytes + offset,
                                      src_device,
                                      chunk),
                       "无法执行跨GPU状态复制");
        }
        const auto ended_at = std::chrono::steady_clock::now();
        transfer_counters_.p2p_bytes += total_bytes;
        transfer_counters_.p2p_count += 1;
        transfer_counters_.state_migrations += 1;
        transfer_counters_.p2p_time_ms +=
            std::chrono::duration<double, std::milli>(ended_at - started_at).count();
        return;
    }

    constexpr size_t kStagingBytes = 64ULL * 1024ULL * 1024ULL;
    void* staging = nullptr;
    check_cuda(cudaHostAlloc(&staging, std::min(kStagingBytes, total_bytes), cudaHostAllocPortable),
               "无法分配跨GPU pinned host staging");
    try {
        for (size_t offset = 0; offset < total_bytes; offset += kStagingBytes) {
            const size_t chunk = std::min(kStagingBytes, total_bytes - offset);
            check_cuda(cudaSetDevice(src_device), "无法设置源CUDA设备执行host-staged复制");
            check_cuda(cudaMemcpy(staging,
                                  src_bytes + offset,
                                  chunk,
                                  cudaMemcpyDeviceToHost),
                       "无法执行跨GPU D2H staging");
            check_cuda(cudaSetDevice(dst_device), "无法设置目标CUDA设备执行host-staged复制");
            check_cuda(cudaMemcpy(dst_bytes + offset,
                                  staging,
                                  chunk,
                                  cudaMemcpyHostToDevice),
                       "无法执行跨GPU H2D staging");
        }
    } catch (...) {
        cudaFreeHost(staging);
        throw;
    }
    check_cuda(cudaFreeHost(staging), "无法释放跨GPU pinned host staging");
    const auto ended_at = std::chrono::steady_clock::now();
    transfer_counters_.host_staged_bytes += total_bytes;
    transfer_counters_.host_staged_count += 1;
    transfer_counters_.state_migrations += 1;
    transfer_counters_.host_staged_time_ms +=
        std::chrono::duration<double, std::milli>(ended_at - started_at).count();
}

int CVStatePool::duplicate_state(int state_id) {
    if (!is_valid_state(state_id)) {
        std::cerr << "无效的状态ID: " << state_id << std::endl;
        return -1;
    }

    const int src_device = host_state_devices_[static_cast<size_t>(state_id)];
    const int64_t state_dim = get_state_dim(state_id);
    const int preferred_device =
        (state_dim > 0) ? choose_device_for_storage(static_cast<size_t>(state_dim), -1) : src_device;

    const int new_state_id = allocate_state(preferred_device);
    if (new_state_id == -1) {
        std::cerr << "无法分配新状态用于复制" << std::endl;
        return -1;
    }

    try {
        reserve_state_storage(new_state_id, state_dim);
        if (state_dim > 0) {
            const int dst_device = host_state_devices_[static_cast<size_t>(new_state_id)];
            check_cuda(cudaSetDevice(dst_device), "无法设置目标CUDA设备复制状态");
            activate_device_view(dst_device);
            cuDoubleComplex* dst_ptr = data + host_state_offsets[static_cast<size_t>(new_state_id)];

            check_cuda(cudaSetDevice(src_device), "无法设置源CUDA设备复制状态");
            activate_device_view(src_device);
            const cuDoubleComplex* src_ptr = data + host_state_offsets[static_cast<size_t>(state_id)];

            copy_state_between_devices(src_ptr,
                                       src_device,
                                       dst_ptr,
                                       dst_device,
                                       static_cast<size_t>(state_dim));

            check_cuda(cudaSetDevice(dst_device), "无法恢复目标CUDA设备复制状态");
            activate_device_view(dst_device);
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
    return host_state_dims[static_cast<size_t>(state_id)];
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
                cuCmul(state1_host[static_cast<size_t>(i)], state2_host[static_cast<size_t>(j)]);
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
