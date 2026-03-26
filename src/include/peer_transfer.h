/**
 * @file peer_transfer.h
 * @brief Cross-device data transfer engine for multi-GPU state migration.
 *
 * Provides optimized transfer primitives:
 * - P2P direct copy via cudaMemcpyPeerAsync (NVLink or PCIe)
 * - Host-staged fallback using pinned buffers
 * - Batch transfer of multiple state vectors
 * - Async transfer with stream-based overlap
 *
 * Transfer strategy is selected automatically based on P2P availability
 * and transfer size. Small transfers use direct P2P; large transfers
 * may benefit from chunked pipelining via host staging.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cstdint>

namespace hybridcvdv {

// ─── Transfer Strategy ───────────────────────────────────────────────────────

/** @brief Method for cross-device data transfer. */
enum class TransferStrategy {
    P2P_DIRECT,     ///< Direct GPU-to-GPU via cudaMemcpyPeerAsync
    HOST_STAGED,    ///< GPU → pinned host → GPU
    AUTO            ///< Auto-select based on P2P availability
};

// ─── Transfer Statistics ─────────────────────────────────────────────────────

/** @brief Statistics for a completed transfer operation. */
struct TransferStats {
    size_t bytes_transferred;
    double elapsed_ms;           ///< Wall-clock time in milliseconds
    double bandwidth_gbps;       ///< Effective bandwidth in GB/s
    TransferStrategy strategy_used;
    int src_device;
    int dst_device;
};

// ─── Peer Transfer Engine ────────────────────────────────────────────────────

/**
 * @brief Engine for transferring state vectors between GPU devices.
 *
 * Usage:
 *   PeerTransfer xfer;
 *   xfer.initialize(2);  // 2 GPUs
 *   xfer.transfer_state(src_ptr, 0, dst_ptr, 1, num_elements, stream);
 *   xfer.shutdown();
 */
class PeerTransfer {
public:
    PeerTransfer() = default;
    ~PeerTransfer();

    /**
     * @brief Initialize transfer engine with pinned staging buffers.
     * @param num_devices     Number of GPU devices.
     * @param staging_bytes   Size of per-device pinned staging buffer.
     */
    void initialize(int num_devices, size_t staging_bytes = 256 * 1024 * 1024);

    /** @brief Release all staging buffers. */
    void shutdown();

    // ── Single Transfer ──────────────────────────────────────────────────

    /**
     * @brief Transfer a contiguous block of cuDoubleComplex data between devices.
     * @param src_ptr        Source pointer on src_device.
     * @param src_device     Source GPU device ID.
     * @param dst_ptr        Destination pointer on dst_device.
     * @param dst_device     Destination GPU device ID.
     * @param num_elements   Number of cuDoubleComplex elements to transfer.
     * @param stream         CUDA stream for async transfer (nullptr = sync).
     * @param strategy       Transfer method (AUTO = pick best).
     * @return Transfer statistics.
     */
    TransferStats transfer_state(
        const cuDoubleComplex* src_ptr, int src_device,
        cuDoubleComplex* dst_ptr, int dst_device,
        int64_t num_elements,
        cudaStream_t stream = nullptr,
        TransferStrategy strategy = TransferStrategy::AUTO);

    // ── Batch Transfer ───────────────────────────────────────────────────

    /**
     * @brief Transfer multiple state vectors in a single batch.
     *
     * Groups transfers by (src, dst) device pair and pipelines them.
     * Each entry: {src_ptr, src_device, dst_ptr, dst_device, num_elements}.
     */
    struct BatchEntry {
        const cuDoubleComplex* src_ptr;
        int src_device;
        cuDoubleComplex* dst_ptr;
        int dst_device;
        int64_t num_elements;
    };

    /**
     * @brief Execute a batch of transfers with optimal pipelining.
     * @param entries  Vector of transfer entries.
     * @param stream   CUDA stream (nullptr = default).
     * @return Total bytes transferred.
     */
    size_t transfer_batch(const std::vector<BatchEntry>& entries,
                          cudaStream_t stream = nullptr);

    // ── Utility ──────────────────────────────────────────────────────────

    /** @brief Get cumulative transfer statistics. */
    TransferStats cumulative_stats() const { return cumulative_stats_; }

    /** @brief Reset cumulative statistics. */
    void reset_stats() { cumulative_stats_ = {}; }

private:
    /**
     * @brief P2P direct transfer (assumes P2P access is enabled).
     */
    TransferStats transfer_p2p(
        const cuDoubleComplex* src_ptr, int src_device,
        cuDoubleComplex* dst_ptr, int dst_device,
        int64_t num_elements, cudaStream_t stream);

    /**
     * @brief Host-staged transfer through pinned buffer.
     * Chunks the transfer if data exceeds staging buffer size.
     */
    TransferStats transfer_host_staged(
        const cuDoubleComplex* src_ptr, int src_device,
        cuDoubleComplex* dst_ptr, int dst_device,
        int64_t num_elements, cudaStream_t stream);

    /** @brief Determine best strategy for a transfer. */
    TransferStrategy choose_strategy(int src_device, int dst_device) const;

    // Per-device pinned staging buffers
    std::vector<void*> staging_buffers_;
    size_t staging_capacity_ = 0;
    int num_devices_ = 0;
    bool initialized_ = false;
    TransferStats cumulative_stats_ = {};
};

} // namespace hybridcvdv
