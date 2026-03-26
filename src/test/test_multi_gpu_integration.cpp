/**
 * @file test_multi_gpu_integration.cpp
 * @brief Integration tests for multi-GPU quantum simulation workflow.
 *
 * Tests end-to-end scenarios combining GPUContext, StreamPool,
 * PeerTransfer, and CVStatePool with multiple GPU devices.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cmath>
#include <numeric>
#include "gpu_context.h"
#include "stream_pool.h"
#include "peer_transfer.h"
#include "cv_state_pool.h"

using namespace hybridcvdv;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class MultiGPUIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!GPUContext::is_initialized()) {
            GPUContext::initialize(true);
        }
        num_devices_ = GPUContext::instance().num_devices();
        if (!StreamPool::is_initialized()) {
            StreamPool::initialize(num_devices_);
        }
    }

    int num_devices_ = 0;
};

static bool complex_near(cuDoubleComplex a, cuDoubleComplex b, double tol = 1e-12) {
    return std::abs(cuCreal(a) - cuCreal(b)) < tol &&
           std::abs(cuCimag(a) - cuCimag(b)) < tol;
}

// ─── Full Stack Initialization ───────────────────────────────────────────────

TEST_F(MultiGPUIntegrationTest, FullStackInit) {
    EXPECT_TRUE(GPUContext::is_initialized());
    EXPECT_TRUE(StreamPool::is_initialized());
    EXPECT_GE(num_devices_, 1);
}

// ─── Multi-Pool Round Trip ───────────────────────────────────────────────────

TEST_F(MultiGPUIntegrationTest, MultiPoolRoundTrip) {
    if (num_devices_ < 2) GTEST_SKIP() << "Need 2+ GPUs";

    const int D = 20;
    CVStatePool pool0(D, 8, 1, 0);
    CVStatePool pool1(D, 8, 1, 0);

    // Create a quantum state on device 0 (coherent state approximation)
    int sid0 = pool0.allocate_state();
    std::vector<cuDoubleComplex> coherent(D);
    double alpha = 2.0;
    double norm_sq = 0;
    for (int n = 0; n < D; ++n) {
        double amp = std::exp(-alpha * alpha / 2.0);
        for (int k = 1; k <= n; ++k) amp *= alpha / std::sqrt(k);
        coherent[n] = make_cuDoubleComplex(amp, 0);
        norm_sq += amp * amp;
    }
    // Normalize
    double norm = std::sqrt(norm_sq);
    for (int n = 0; n < D; ++n) {
        coherent[n] = make_cuDoubleComplex(cuCreal(coherent[n]) / norm, 0);
    }

    pool0.upload_state(sid0, coherent);

    // Transfer to pool1 via host
    std::vector<cuDoubleComplex> h_transfer;
    pool0.download_state(sid0, h_transfer);
    int sid1 = pool1.allocate_state();
    ASSERT_GE(sid1, 0);
    pool1.upload_state(sid1, h_transfer);

    // Download from pool1 and verify
    std::vector<cuDoubleComplex> result;
    pool1.download_state(sid1, result);
    ASSERT_EQ(result.size(), static_cast<size_t>(D));

    double result_norm_sq = 0;
    for (int n = 0; n < D; ++n) {
        EXPECT_TRUE(complex_near(coherent[n], result[n], 1e-10))
            << "Fock level " << n;
        result_norm_sq += cuCreal(result[n]) * cuCreal(result[n]) +
                          cuCimag(result[n]) * cuCimag(result[n]);
    }
    EXPECT_NEAR(result_norm_sq, 1.0, 1e-8);

    pool0.free_state(sid0);
    pool1.free_state(sid1);
}

// ─── Bidirectional Migration ─────────────────────────────────────────────────

TEST_F(MultiGPUIntegrationTest, BidirectionalTransfer) {
    if (num_devices_ < 2) GTEST_SKIP() << "Need 2+ GPUs";

    const int D = 15;
    CVStatePool pool0(D, 8, 1, 0);
    CVStatePool pool1(D, 8, 1, 0);

    // Create state on pool0
    int sid0 = pool0.allocate_state();
    std::vector<cuDoubleComplex> original(D);
    for (int i = 0; i < D; ++i) {
        original[i] = make_cuDoubleComplex(std::sin(i), std::cos(i));
    }
    pool0.upload_state(sid0, original);

    // Pool0 → Pool1 via host
    std::vector<cuDoubleComplex> h_transfer;
    pool0.download_state(sid0, h_transfer);
    int sid1 = pool1.allocate_state();
    ASSERT_GE(sid1, 0);
    pool1.upload_state(sid1, h_transfer);

    // Pool1 → Pool0 via host (new state)
    std::vector<cuDoubleComplex> h_transfer2;
    pool1.download_state(sid1, h_transfer2);
    int sid0_back = pool0.allocate_state();
    ASSERT_GE(sid0_back, 0);
    pool0.upload_state(sid0_back, h_transfer2);

    // Verify round-trip fidelity
    std::vector<cuDoubleComplex> roundtrip;
    pool0.download_state(sid0_back, roundtrip);
    ASSERT_EQ(roundtrip.size(), static_cast<size_t>(D));
    for (int i = 0; i < D; ++i) {
        EXPECT_TRUE(complex_near(original[i], roundtrip[i], 1e-10))
            << "Round-trip mismatch at index " << i;
    }

    pool0.free_state(sid0);
    pool0.free_state(sid0_back);
    pool1.free_state(sid1);
}

// ─── Stream-Based Async Transfer ─────────────────────────────────────────────

TEST_F(MultiGPUIntegrationTest, AsyncTransferWithStreams) {
    if (num_devices_ < 2) GTEST_SKIP() << "Need 2+ GPUs";

    PeerTransfer xfer;
    xfer.initialize(num_devices_);

    const int N = 4096;

    cudaSetDevice(0);
    cuDoubleComplex* d_src;
    cudaMalloc(&d_src, N * sizeof(cuDoubleComplex));

    cudaSetDevice(1);
    cuDoubleComplex* d_dst;
    cudaMalloc(&d_dst, N * sizeof(cuDoubleComplex));

    // Init
    std::vector<cuDoubleComplex> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = make_cuDoubleComplex(i * 0.001, -i * 0.002);
    }
    cudaSetDevice(0);
    cudaMemcpy(d_src, h_data.data(), N * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    // Use transfer stream
    auto& spool = StreamPool::instance();
    cudaStream_t xfer_stream = spool.get_transfer_stream(0);

    auto stats = xfer.transfer_state(d_src, 0, d_dst, 1, N, xfer_stream);
    spool.sync_stream(xfer_stream);

    EXPECT_EQ(stats.bytes_transferred, static_cast<size_t>(N) * sizeof(cuDoubleComplex));

    // Verify
    cudaSetDevice(1);
    cuDoubleComplex h_check;
    cudaMemcpy(&h_check, d_dst + N / 2, sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);
    EXPECT_TRUE(complex_near(h_check, h_data[N / 2], 1e-10));

    cudaSetDevice(0);
    cudaFree(d_src);
    cudaSetDevice(1);
    cudaFree(d_dst);
    xfer.shutdown();
}

// ─── DeviceGuard with Pool Operations ────────────────────────────────────────

TEST_F(MultiGPUIntegrationTest, DeviceGuardWithPools) {
    if (num_devices_ < 2) GTEST_SKIP() << "Need 2+ GPUs";

    cudaSetDevice(0);

    {
        DeviceGuard guard(1);
        CVStatePool pool(10, 4, 1, 0);
        int sid = pool.allocate_state();
        EXPECT_GE(sid, 0);

        std::vector<cuDoubleComplex> data(10, make_cuDoubleComplex(1.0, 0));
        pool.upload_state(sid, data);
        pool.free_state(sid);
    }

    // Device should be restored to 0
    int current;
    cudaGetDevice(&current);
    EXPECT_EQ(current, 0);
}

// ─── Memory Pressure Distribution ────────────────────────────────────────────

TEST_F(MultiGPUIntegrationTest, DistributeStatesAcrossDevices) {
    if (num_devices_ < 2) GTEST_SKIP() << "Need 2+ GPUs";

    const int D = 50;
    const int STATES_PER_DEVICE = 4;

    CVStatePool pool0(D, STATES_PER_DEVICE * 2, 1, 0);
    CVStatePool pool1(D, STATES_PER_DEVICE * 2, 1, 0);

    auto& ctx = GPUContext::instance();

    // Allocate states using memory-aware selection
    std::vector<std::pair<int, int>> allocations;  // (device_id, state_id)
    for (int i = 0; i < STATES_PER_DEVICE * 2; ++i) {
        int dev = ctx.select_device(DeviceSelectionPolicy::ROUND_ROBIN);
        CVStatePool& pool = (dev == 0) ? pool0 : pool1;
        int sid = pool.allocate_state();
        ASSERT_GE(sid, 0) << "Failed to allocate state " << i << " on device " << dev;

        // Upload a simple state
        std::vector<cuDoubleComplex> state(D, make_cuDoubleComplex(0, 0));
        state[i % D] = make_cuDoubleComplex(1, 0);
        pool.upload_state(sid, state);

        allocations.emplace_back(dev, sid);
    }

    // Verify states are distributed across both devices
    int count0 = 0, count1 = 0;
    for (auto [dev, sid] : allocations) {
        if (dev == 0) ++count0;
        else ++count1;
    }
    EXPECT_GT(count0, 0);
    EXPECT_GT(count1, 0);

    // Cleanup
    for (auto [dev, sid] : allocations) {
        CVStatePool& pool = (dev == 0) ? pool0 : pool1;
        pool.free_state(sid);
    }
}
