/**
 * @file gpu_context.cu
 * @brief Implementation of GPU device context management.
 *
 * Handles GPU detection, P2P setup, and device selection for
 * both single-GPU and multi-GPU execution paths.
 */

#include "gpu_context.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <numeric>

// Include path for src/include
#include "../include/gpu_context.h"

namespace hybridcvdv {

// ─── Static Members ──────────────────────────────────────────────────────────

std::mutex GPUContext::init_mutex_;
GPUContext* GPUContext::instance_ = nullptr;
bool GPUContext::initialized_ = false;

// ─── Lifecycle ───────────────────────────────────────────────────────────────

bool GPUContext::initialize(bool enable_p2p) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (initialized_) {
        return true;  // Already initialized
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "[GPUContext] No CUDA devices found.\n");
        return false;
    }

    instance_ = new GPUContext();

    // Probe each device
    instance_->devices_.resize(device_count);
    for (int i = 0; i < device_count; ++i) {
        instance_->probe_device(i, instance_->devices_[i]);
    }

    // Setup P2P if multi-GPU and requested
    if (device_count > 1 && enable_p2p) {
        instance_->setup_p2p(true);
    }

    initialized_ = true;
    printf("[GPUContext] Initialized with %d device(s)%s\n",
           device_count,
           device_count > 1 ? " (multi-GPU enabled)" : "");
    return true;
}

void GPUContext::shutdown() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (!initialized_ || !instance_) return;

    // Disable P2P access
    for (auto& link : instance_->p2p_links_) {
        if (link.access_enabled) {
            cudaSetDevice(link.src_device);
            cudaDeviceDisablePeerAccess(link.dst_device);
            link.access_enabled = false;
        }
    }

    delete instance_;
    instance_ = nullptr;
    initialized_ = false;
}

GPUContext& GPUContext::instance() {
    if (!initialized_ || !instance_) {
        throw std::runtime_error(
            "[GPUContext] Not initialized. Call GPUContext::initialize() first.");
    }
    return *instance_;
}

bool GPUContext::is_initialized() {
    return initialized_;
}

// ─── Device Probing ──────────────────────────────────────────────────────────

void GPUContext::probe_device(int device_id, DeviceInfo& info) {
    cudaDeviceProp prop;
    MGPU_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    info.device_id = device_id;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_memory_bytes = prop.totalGlobalMem;
    info.sm_count = prop.multiProcessorCount;
    info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    info.max_shared_memory_per_block = static_cast<int>(prop.sharedMemPerBlock);
    info.max_shared_memory_per_sm = static_cast<int>(prop.sharedMemPerMultiprocessor);
    info.unified_addressing = prop.unifiedAddressing != 0;
    info.managed_memory = prop.managedMemory != 0;
    info.pci_bus_id = prop.pciBusID;
    info.pci_device_id = prop.pciDeviceID;

    // Query current free memory
    MGPU_CHECK_CUDA(cudaSetDevice(device_id));
    size_t free_bytes = 0, total_bytes = 0;
    MGPU_CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    info.free_memory_bytes = free_bytes;
}

// ─── P2P Setup ───────────────────────────────────────────────────────────────

void GPUContext::setup_p2p(bool enable) {
    int n = static_cast<int>(devices_.size());
    p2p_links_.clear();
    p2p_links_.reserve(n * (n - 1));

    for (int src = 0; src < n; ++src) {
        for (int dst = 0; dst < n; ++dst) {
            if (src == dst) continue;

            P2PLink link;
            link.src_device = src;
            link.dst_device = dst;
            link.access_enabled = false;
            link.is_nvlink = false;

            // Check if P2P access is possible
            int can_access = 0;
            MGPU_CHECK_CUDA(
                cudaDeviceCanAccessPeer(&can_access, src, dst));
            link.access_supported = (can_access != 0);

            if (link.access_supported && enable) {
                MGPU_CHECK_CUDA(cudaSetDevice(src));
                cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
                if (err == cudaSuccess) {
                    link.access_enabled = true;
                } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
                    link.access_enabled = true;
                    cudaGetLastError(); // Clear the error
                } else {
                    fprintf(stderr,
                        "[GPUContext] Warning: P2P access %d→%d supported "
                        "but enable failed: %s\n",
                        src, dst, cudaGetErrorString(err));
                    cudaGetLastError(); // Clear the error
                }
            }

            // Detect NVLink by checking if devices share a NUMA node
            // or have a high-bandwidth link. Use P2P performance attributes
            // if available (CUDA 11.4+).
#if CUDART_VERSION >= 11040
            if (link.access_supported) {
                int perf_rank = 0;
                cudaError_t attr_err = cudaDeviceGetP2PAttribute(
                    &perf_rank,
                    cudaDevP2PAttrPerformanceRank, src, dst);
                // Performance rank > 0 typically indicates NVLink
                if (attr_err == cudaSuccess && perf_rank > 0) {
                    link.is_nvlink = true;
                }
            }
#endif

            p2p_links_.push_back(link);
        }
    }

    // Summary
    int enabled_count = 0;
    int nvlink_count = 0;
    for (const auto& link : p2p_links_) {
        if (link.access_enabled) ++enabled_count;
        if (link.is_nvlink) ++nvlink_count;
    }
    printf("[GPUContext] P2P: %d links enabled (%d NVLink)\n",
           enabled_count, nvlink_count);
}

// ─── Device Queries ──────────────────────────────────────────────────────────

const DeviceInfo& GPUContext::device_info(int device_id) const {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        throw std::out_of_range("[GPUContext] Invalid device_id: " +
                                std::to_string(device_id));
    }
    return devices_[device_id];
}

void GPUContext::refresh_memory_info() {
    int prev_device;
    MGPU_CHECK_CUDA(cudaGetDevice(&prev_device));

    for (auto& info : devices_) {
        MGPU_CHECK_CUDA(cudaSetDevice(info.device_id));
        size_t free_bytes = 0, total_bytes = 0;
        MGPU_CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        info.free_memory_bytes = free_bytes;
    }

    MGPU_CHECK_CUDA(cudaSetDevice(prev_device));
}

int GPUContext::current_device() const {
    int dev;
    MGPU_CHECK_CUDA(cudaGetDevice(&dev));
    return dev;
}

// ─── Device Selection ────────────────────────────────────────────────────────

int GPUContext::select_device(DeviceSelectionPolicy policy,
                              int affinity_device) const {
    int n = num_devices();
    if (n == 0) {
        throw std::runtime_error("[GPUContext] No devices available.");
    }
    if (n == 1) {
        return devices_[0].device_id;  // Single GPU fast path
    }

    switch (policy) {
    case DeviceSelectionPolicy::ROUND_ROBIN: {
        int idx = round_robin_counter_.fetch_add(1,
                      std::memory_order_relaxed) % n;
        return devices_[idx].device_id;
    }

    case DeviceSelectionPolicy::MOST_FREE_MEM: {
        // Refresh memory info for accurate selection
        const_cast<GPUContext*>(this)->refresh_memory_info();
        int best = 0;
        size_t best_free = 0;
        for (int i = 0; i < n; ++i) {
            if (devices_[i].free_memory_bytes > best_free) {
                best_free = devices_[i].free_memory_bytes;
                best = i;
            }
        }
        return devices_[best].device_id;
    }

    case DeviceSelectionPolicy::AFFINITY: {
        if (affinity_device >= 0 && affinity_device < n) {
            return devices_[affinity_device].device_id;
        }
        // Fall back to round-robin if affinity is invalid
        return select_device(DeviceSelectionPolicy::ROUND_ROBIN);
    }

    case DeviceSelectionPolicy::EXPLICIT:
        // Caller must handle explicit selection externally
        return current_device();

    default:
        return devices_[0].device_id;
    }
}

void GPUContext::set_device(int device_id) const {
    MGPU_CHECK_CUDA(cudaSetDevice(device_id));
}

// ─── P2P Queries ─────────────────────────────────────────────────────────────

bool GPUContext::is_p2p_enabled(int src_device, int dst_device) const {
    for (const auto& link : p2p_links_) {
        if (link.src_device == src_device &&
            link.dst_device == dst_device) {
            return link.access_enabled;
        }
    }
    return false;
}

bool GPUContext::is_nvlink(int src_device, int dst_device) const {
    for (const auto& link : p2p_links_) {
        if (link.src_device == src_device &&
            link.dst_device == dst_device) {
            return link.is_nvlink;
        }
    }
    return false;
}

// ─── Utility ─────────────────────────────────────────────────────────────────

void GPUContext::print_device_summary() const {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║          GPU Context Summary                    ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    for (const auto& dev : devices_) {
        printf("║ GPU %d: %-40s ║\n", dev.device_id, dev.name.c_str());
        printf("║   Compute: SM %d.%d  |  SMs: %-4d  |  Warp: %d   ║\n",
               dev.compute_major, dev.compute_minor,
               dev.sm_count, dev.warp_size);
        printf("║   Memory:  %5zu MiB total / %5zu MiB free    ║\n",
               dev.total_memory_mib(), dev.free_memory_mib());
        printf("║   SharedMem/Block: %5d KB  /SM: %5d KB    ║\n",
               dev.max_shared_memory_per_block / 1024,
               dev.max_shared_memory_per_sm / 1024);
    }

    if (!p2p_links_.empty()) {
        printf("╠──────────────────────────────────────────────────╣\n");
        printf("║ P2P Topology:                                    ║\n");
        for (const auto& link : p2p_links_) {
            if (link.access_enabled) {
                printf("║   %d → %d  [%s]%s                            ║\n",
                       link.src_device, link.dst_device,
                       link.access_enabled ? "ENABLED" : "disabled",
                       link.is_nvlink ? " NVLink" : "");
            }
        }
    }
    printf("╚══════════════════════════════════════════════════╝\n");
}

size_t GPUContext::total_free_memory() const {
    size_t total = 0;
    for (const auto& dev : devices_) {
        total += dev.free_memory_bytes;
    }
    return total;
}

} // namespace hybridcvdv
