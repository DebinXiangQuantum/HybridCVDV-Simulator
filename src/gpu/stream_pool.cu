/**
 * @file stream_pool.cu
 * @brief Implementation of per-device CUDA stream and event management.
 *
 * Creates and manages CUDA streams for compute and transfer operations,
 * enabling overlap of kernel execution with data transfers. Uses
 * high-priority streams for transfers to avoid blocking on compute work.
 */

#include "../include/stream_pool.h"
#include "../include/gpu_context.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

namespace hybridcvdv {

// ─── Static Members ──────────────────────────────────────────────────────────

std::mutex StreamPool::init_mutex_;
StreamPool* StreamPool::instance_ = nullptr;
bool StreamPool::initialized_ = false;

// ─── Lifecycle ───────────────────────────────────────────────────────────────

bool StreamPool::initialize(int num_devices,
                            int compute_per_dev,
                            int transfer_per_dev) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (initialized_) return true;

    if (num_devices <= 0) {
        fprintf(stderr, "[StreamPool] Invalid num_devices: %d\n", num_devices);
        return false;
    }

    instance_ = new StreamPool();
    instance_->per_device_.resize(num_devices);

    for (int d = 0; d < num_devices; ++d) {
        MGPU_CHECK_CUDA(cudaSetDevice(d));

        auto& ds = instance_->per_device_[d];
        ds.device_id = d;

        // Query stream priority range for this device
        int low_priority, high_priority;
        MGPU_CHECK_CUDA(
            cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority));

        // Create compute streams (normal priority)
        ds.compute_streams.resize(compute_per_dev);
        for (int i = 0; i < compute_per_dev; ++i) {
            MGPU_CHECK_CUDA(cudaStreamCreateWithPriority(
                &ds.compute_streams[i],
                cudaStreamNonBlocking,
                low_priority));
        }

        // Create transfer streams (high priority for low latency)
        ds.transfer_streams.resize(transfer_per_dev);
        for (int i = 0; i < transfer_per_dev; ++i) {
            MGPU_CHECK_CUDA(cudaStreamCreateWithPriority(
                &ds.transfer_streams[i],
                cudaStreamNonBlocking,
                high_priority));
        }
    }

    initialized_ = true;
    printf("[StreamPool] Initialized: %d devices × (%d compute + %d transfer) "
           "streams\n", num_devices, compute_per_dev, transfer_per_dev);
    return true;
}

void StreamPool::shutdown() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (!initialized_ || !instance_) return;

    for (auto& ds : instance_->per_device_) {
        MGPU_CHECK_CUDA(cudaSetDevice(ds.device_id));

        // Synchronize before destroying
        for (auto& s : ds.compute_streams) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
        for (auto& s : ds.transfer_streams) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
        ds.compute_streams.clear();
        ds.transfer_streams.clear();
    }

    delete instance_;
    instance_ = nullptr;
    initialized_ = false;
}

StreamPool& StreamPool::instance() {
    if (!initialized_ || !instance_) {
        throw std::runtime_error(
            "[StreamPool] Not initialized. Call StreamPool::initialize() first.");
    }
    return *instance_;
}

bool StreamPool::is_initialized() {
    return initialized_;
}

// ─── Stream Access ───────────────────────────────────────────────────────────

cudaStream_t StreamPool::get_compute_stream(int device_id) const {
    if (device_id < 0 ||
        device_id >= static_cast<int>(per_device_.size())) {
        return nullptr;  // Fallback to default stream
    }
    return per_device_[device_id].next_compute_stream();
}

cudaStream_t StreamPool::get_compute_stream(int device_id, int index) const {
    if (device_id < 0 ||
        device_id >= static_cast<int>(per_device_.size())) {
        return nullptr;
    }
    return per_device_[device_id].compute_stream(index);
}

cudaStream_t StreamPool::get_transfer_stream(int device_id) const {
    if (device_id < 0 ||
        device_id >= static_cast<int>(per_device_.size())) {
        return nullptr;
    }
    return per_device_[device_id].next_transfer_stream();
}

const DeviceStreams& StreamPool::device_streams(int device_id) const {
    if (device_id < 0 ||
        device_id >= static_cast<int>(per_device_.size())) {
        throw std::out_of_range("[StreamPool] Invalid device_id: " +
                                std::to_string(device_id));
    }
    return per_device_[device_id];
}

// ─── Synchronization ─────────────────────────────────────────────────────────

void StreamPool::sync_device(int device_id) const {
    if (device_id < 0 ||
        device_id >= static_cast<int>(per_device_.size())) {
        return;
    }

    const auto& ds = per_device_[device_id];
    MGPU_CHECK_CUDA(cudaSetDevice(ds.device_id));

    for (auto s : ds.compute_streams) {
        MGPU_CHECK_CUDA(cudaStreamSynchronize(s));
    }
    for (auto s : ds.transfer_streams) {
        MGPU_CHECK_CUDA(cudaStreamSynchronize(s));
    }
}

void StreamPool::sync_all() const {
    for (int d = 0; d < static_cast<int>(per_device_.size()); ++d) {
        sync_device(d);
    }
}

void StreamPool::sync_stream(cudaStream_t stream) const {
    if (stream) {
        MGPU_CHECK_CUDA(cudaStreamSynchronize(stream));
    } else {
        MGPU_CHECK_CUDA(cudaDeviceSynchronize());
    }
}

// ─── Events ──────────────────────────────────────────────────────────────────

int StreamPool::find_device_for_stream(cudaStream_t stream) const {
    for (const auto& ds : per_device_) {
        for (auto s : ds.compute_streams) {
            if (s == stream) return ds.device_id;
        }
        for (auto s : ds.transfer_streams) {
            if (s == stream) return ds.device_id;
        }
    }
    return -1;  // Unknown stream (e.g. default stream)
}

cudaEvent_t StreamPool::record_event(cudaStream_t stream) const {
    // Set the correct device context for event creation.
    // Events must be created on the same device as the stream.
    int target_dev = find_device_for_stream(stream);
    if (target_dev >= 0) {
        MGPU_CHECK_CUDA(cudaSetDevice(target_dev));
    }

    cudaEvent_t event;
    // Use disable-timing flag for lower overhead when timing isn't needed
    MGPU_CHECK_CUDA(cudaEventCreateWithFlags(&event,
                                              cudaEventDisableTiming));
    MGPU_CHECK_CUDA(cudaEventRecord(event, stream));
    return event;
}

void StreamPool::wait_event(cudaStream_t stream, cudaEvent_t event) const {
    MGPU_CHECK_CUDA(cudaStreamWaitEvent(stream, event, 0));
}

void StreamPool::cross_device_sync(cudaStream_t src_stream,
                                   cudaStream_t dst_stream) const {
    // Record event on source stream's device, then make destination wait.
    // cudaStreamWaitEvent works across devices.
    cudaEvent_t event = record_event(src_stream);
    wait_event(dst_stream, event);
    MGPU_CHECK_CUDA(cudaEventDestroy(event));
}

} // namespace hybridcvdv
