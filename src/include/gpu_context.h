/**
 * @file gpu_context.h
 * @brief GPU device context management for single and multi-GPU execution.
 *
 * Provides a unified interface for:
 * - Detecting available GPU devices and their capabilities
 * - Enabling peer-to-peer (P2P) access between GPU pairs
 * - Device selection policies (memory-aware, round-robin)
 * - Querying device memory and compute properties
 *
 * Design: Singleton GPUContext that initializes once and provides
 * device information throughout the simulator's lifetime. When only
 * one GPU is present, all operations degenerate to the single-GPU
 * fast path with zero overhead.
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>
#include <mutex>
#include <atomic>

namespace hybridcvdv {

// ─── Error handling ──────────────────────────────────────────────────────────

#define MGPU_CHECK_CUDA(call)                                                   \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "[MGPU] CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(err));                   \
        }                                                                        \
    } while (0)

// ─── Device Information ──────────────────────────────────────────────────────

/**
 * @brief Capabilities and properties of a single GPU device.
 */
struct DeviceInfo {
    int device_id;                     ///< CUDA device ordinal
    std::string name;                  ///< Device name (e.g., "NVIDIA A40")
    int compute_major;                 ///< Compute capability major
    int compute_minor;                 ///< Compute capability minor
    size_t total_memory_bytes;         ///< Total global memory
    size_t free_memory_bytes;          ///< Currently free global memory
    int sm_count;                      ///< Number of streaming multiprocessors
    int max_threads_per_sm;            ///< Max resident threads per SM
    int max_threads_per_block;         ///< Max threads per block
    int warp_size;                     ///< Warp size (typically 32)
    int max_shared_memory_per_block;   ///< Max shared memory per block (bytes)
    int max_shared_memory_per_sm;      ///< Max shared memory per SM (bytes)
    bool unified_addressing;           ///< Supports unified virtual addressing
    bool managed_memory;               ///< Supports managed memory
    int pci_bus_id;                    ///< PCI bus ID
    int pci_device_id;                 ///< PCI device ID

    /** @brief Total memory in MiB. */
    size_t total_memory_mib() const { return total_memory_bytes >> 20; }

    /** @brief Free memory in MiB. */
    size_t free_memory_mib() const { return free_memory_bytes >> 20; }

    /** @brief Utilization ratio (0.0 = empty, 1.0 = full). */
    double memory_utilization() const {
        if (total_memory_bytes == 0) return 1.0;
        return 1.0 - static_cast<double>(free_memory_bytes) /
                      static_cast<double>(total_memory_bytes);
    }
};

// ─── Device Selection Policy ─────────────────────────────────────────────────

/**
 * @brief Strategy for choosing which GPU to allocate new states on.
 */
enum class DeviceSelectionPolicy {
    ROUND_ROBIN,    ///< Cycle through devices evenly
    MOST_FREE_MEM,  ///< Pick device with most free memory
    AFFINITY,       ///< Prefer same device as related states
    EXPLICIT        ///< Caller specifies device explicitly
};

// ─── P2P Topology ────────────────────────────────────────────────────────────

/**
 * @brief Peer-to-peer connectivity between two devices.
 */
struct P2PLink {
    int src_device;
    int dst_device;
    bool access_supported;    ///< cudaDeviceCanAccessPeer returned true
    bool access_enabled;      ///< cudaDeviceEnablePeerAccess succeeded
    bool is_nvlink;           ///< Connected via NVLink (vs PCIe)
};

// ─── GPU Context (Singleton) ─────────────────────────────────────────────────

/**
 * @brief Central GPU device manager for the simulator.
 *
 * Detects all available GPUs, enables P2P access, and provides
 * device selection and query APIs. Thread-safe singleton.
 *
 * Usage:
 *   GPUContext::initialize();
 *   int dev = GPUContext::instance().select_device(DeviceSelectionPolicy::MOST_FREE_MEM);
 *   GPUContext::instance().set_device(dev);
 *   // ... do work ...
 *   GPUContext::shutdown();
 */
class GPUContext {
public:
    // ── Lifecycle ────────────────────────────────────────────────────────

    /**
     * @brief Initialize the GPU context. Must be called before any GPU work.
     * @param enable_p2p  If true, enable P2P access between all capable pairs.
     * @return true on success.
     */
    static bool initialize(bool enable_p2p = true);

    /** @brief Shutdown and release all GPU resources. */
    static void shutdown();

    /** @brief Get the singleton instance (must call initialize() first). */
    static GPUContext& instance();

    /** @brief Check if the context has been initialized. */
    static bool is_initialized();

    // ── Device Queries ───────────────────────────────────────────────────

    /** @brief Number of available GPU devices. */
    int num_devices() const { return static_cast<int>(devices_.size()); }

    /** @brief Whether multi-GPU is active (more than 1 device). */
    bool is_multi_gpu() const { return devices_.size() > 1; }

    /** @brief Get info for a specific device. */
    const DeviceInfo& device_info(int device_id) const;

    /** @brief Get info for all devices. */
    const std::vector<DeviceInfo>& all_devices() const { return devices_; }

    /** @brief Refresh free memory info for all devices. */
    void refresh_memory_info();

    /** @brief Get the currently active CUDA device for this thread. */
    int current_device() const;

    // ── Device Selection ─────────────────────────────────────────────────

    /**
     * @brief Select the best device based on the given policy.
     * @param policy  Selection strategy.
     * @param affinity_device  Preferred device when policy is AFFINITY.
     * @return Selected device ID.
     */
    int select_device(DeviceSelectionPolicy policy,
                      int affinity_device = -1) const;

    /**
     * @brief Set the active CUDA device for the calling thread.
     * @param device_id  Target device ordinal.
     */
    void set_device(int device_id) const;

    // ── P2P Access ───────────────────────────────────────────────────────

    /**
     * @brief Check if P2P access is enabled from src to dst.
     * @return true if direct peer access is available.
     */
    bool is_p2p_enabled(int src_device, int dst_device) const;

    /**
     * @brief Check if two devices are connected via NVLink.
     */
    bool is_nvlink(int src_device, int dst_device) const;

    /** @brief Get all P2P links. */
    const std::vector<P2PLink>& p2p_links() const { return p2p_links_; }

    // ── Utility ──────────────────────────────────────────────────────────

    /** @brief Print device info summary to stdout. */
    void print_device_summary() const;

    /** @brief Get total free memory across all devices. */
    size_t total_free_memory() const;

    // ── Utility Types ──────────────────────────────────────────────────

    /** @brief Check if device_id is valid. */
    bool is_valid_device(int device_id) const {
        return device_id >= 0 && device_id < num_devices();
    }

private:
    GPUContext() = default;
    ~GPUContext() = default;
    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    /** @brief Probe device properties and fill DeviceInfo. */
    void probe_device(int device_id, DeviceInfo& info);

    /** @brief Detect and enable P2P between all device pairs. */
    void setup_p2p(bool enable);

    std::vector<DeviceInfo> devices_;
    std::vector<P2PLink> p2p_links_;
    mutable std::atomic<int> round_robin_counter_{0};

    static std::mutex init_mutex_;
    static GPUContext* instance_;
    static bool initialized_;
};

} // namespace hybridcvdv

// ─── DeviceGuard (RAII) ──────────────────────────────────────────────────────

/**
 * @brief RAII guard that sets the CUDA device on construction and
 *        restores the previous device on destruction.
 *
 * Use at the top of any function that needs to operate on a specific GPU:
 *   void my_gpu_function(int device_id) {
 *       DeviceGuard guard(device_id);
 *       // All CUDA calls now target device_id
 *   } // Previous device is restored here
 */
struct DeviceGuard {
    int previous_device;

    explicit DeviceGuard(int target_device) {
        cudaGetDevice(&previous_device);
        if (previous_device != target_device) {
            cudaSetDevice(target_device);
        }
    }

    ~DeviceGuard() {
        int current;
        cudaGetDevice(&current);
        if (current != previous_device) {
            cudaSetDevice(previous_device);
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};
