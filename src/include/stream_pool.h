/**
 * @file stream_pool.h
 * @brief Per-device CUDA stream and event management.
 *
 * Provides a pool of CUDA streams for each GPU device, enabling:
 * - Concurrent kernel execution on the same device
 * - Overlap of compute and data transfer
 * - Event-based synchronization between streams and devices
 *
 * Each device gets dedicated compute streams and transfer streams.
 * Transfer streams are used for H2D, D2H, and D2D (P2P) copies,
 * allowing them to overlap with kernel execution on compute streams.
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <cstdint>

namespace hybridcvdv {

// ─── Configuration ───────────────────────────────────────────────────────────

/** @brief Default number of compute streams per device. */
constexpr int kDefaultComputeStreams = 4;

/** @brief Default number of transfer streams per device. */
constexpr int kDefaultTransferStreams = 2;

// ─── Stream Types ────────────────────────────────────────────────────────────

/** @brief Type of stream in the pool. */
enum class StreamType {
    COMPUTE,   ///< For kernel launches
    TRANSFER   ///< For memory transfers (H2D, D2H, P2P)
};

// ─── Per-Device Stream Set ───────────────────────────────────────────────────

/**
 * @brief Manages CUDA streams and events for one GPU device.
 */
struct DeviceStreams {
    int device_id;
    std::vector<cudaStream_t> compute_streams;
    std::vector<cudaStream_t> transfer_streams;

    /** @brief Round-robin counters for stream selection. */
    mutable int next_compute = 0;
    mutable int next_transfer = 0;

    /** @brief Get the next compute stream (round-robin). */
    cudaStream_t next_compute_stream() const {
        if (compute_streams.empty()) return nullptr;
        int idx = next_compute++ % static_cast<int>(compute_streams.size());
        return compute_streams[idx];
    }

    /** @brief Get the next transfer stream (round-robin). */
    cudaStream_t next_transfer_stream() const {
        if (transfer_streams.empty()) return nullptr;
        int idx = next_transfer++ % static_cast<int>(transfer_streams.size());
        return transfer_streams[idx];
    }

    /** @brief Get a specific compute stream by index. */
    cudaStream_t compute_stream(int index) const {
        return compute_streams[index % compute_streams.size()];
    }

    /** @brief Get a specific transfer stream by index. */
    cudaStream_t transfer_stream(int index) const {
        return transfer_streams[index % transfer_streams.size()];
    }
};

// ─── Stream Pool ─────────────────────────────────────────────────────────────

/**
 * @brief Global stream pool managing per-device CUDA streams.
 *
 * Thread-safe stream allocation and synchronization. Streams are
 * created at initialization and reused throughout the simulation.
 *
 * Usage:
 *   StreamPool::initialize(num_devices);
 *   cudaStream_t s = StreamPool::instance().get_compute_stream(device_id);
 *   // launch kernel on s
 *   StreamPool::instance().sync_device(device_id);
 *   StreamPool::shutdown();
 */
class StreamPool {
public:
    // ── Lifecycle ────────────────────────────────────────────────────────

    /**
     * @brief Initialize stream pool for all devices.
     * @param num_devices       Number of GPU devices.
     * @param compute_per_dev   Compute streams per device.
     * @param transfer_per_dev  Transfer streams per device.
     */
    static bool initialize(int num_devices,
                           int compute_per_dev = kDefaultComputeStreams,
                           int transfer_per_dev = kDefaultTransferStreams);

    /** @brief Destroy all streams and release resources. */
    static void shutdown();

    /** @brief Get the singleton instance. */
    static StreamPool& instance();

    /** @brief Check if initialized. */
    static bool is_initialized();

    // ── Stream Access ────────────────────────────────────────────────────

    /** @brief Get the next compute stream for a device (round-robin). */
    cudaStream_t get_compute_stream(int device_id) const;

    /** @brief Get a specific compute stream for a device. */
    cudaStream_t get_compute_stream(int device_id, int index) const;

    /** @brief Get the next transfer stream for a device (round-robin). */
    cudaStream_t get_transfer_stream(int device_id) const;

    /** @brief Get all streams for a device. */
    const DeviceStreams& device_streams(int device_id) const;

    // ── Synchronization ──────────────────────────────────────────────────

    /** @brief Synchronize all streams on a device. */
    void sync_device(int device_id) const;

    /** @brief Synchronize all streams on all devices. */
    void sync_all() const;

    /** @brief Synchronize a specific stream. */
    void sync_stream(cudaStream_t stream) const;

    // ── Events ───────────────────────────────────────────────────────────

    /**
     * @brief Record an event on a stream.
     * @param stream  Stream to record event on.
     * @return Newly created event (caller must destroy with cudaEventDestroy).
     */
    cudaEvent_t record_event(cudaStream_t stream) const;

    /**
     * @brief Make a stream wait for an event.
     * @param stream  Stream that should wait.
     * @param event   Event to wait for.
     */
    void wait_event(cudaStream_t stream, cudaEvent_t event) const;

    /**
     * @brief Cross-device synchronization: make dst_stream wait until
     *        src_stream completes its current work.
     * @param src_stream  Source stream (may be on a different device).
     * @param dst_stream  Destination stream that should wait.
     */
    void cross_device_sync(cudaStream_t src_stream,
                           cudaStream_t dst_stream) const;

    /**
     * @brief Look up which device owns a given stream.
     * @return Device ID, or -1 if not found in the pool.
     */
    int find_device_for_stream(cudaStream_t stream) const;

private:
    StreamPool() = default;
    ~StreamPool() = default;
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

    std::vector<DeviceStreams> per_device_;

    static std::mutex init_mutex_;
    static StreamPool* instance_;
    static bool initialized_;
};

} // namespace hybridcvdv
