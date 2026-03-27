/**
 * @file test_peer_transfer.cpp
 * @brief Unit tests for PeerTransfer - cross-device data transfer.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cmath>
#include "gpu_context.h"
#include "peer_transfer.h"

using namespace hybridcvdv;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class PeerTransferTest : public ::testing::Test {
protected:
    PeerTransfer xfer;

    void SetUp() override {
        if (!GPUContext::is_initialized()) {
            GPUContext::initialize(true);
        }
        int n = GPUContext::instance().num_devices();
        xfer.initialize(n, 64 * 1024 * 1024);  // 64 MiB staging
    }

    void TearDown() override {
        xfer.shutdown();
    }
};

// ─── Helper ──────────────────────────────────────────────────────────────────

static bool complex_near(cuDoubleComplex a, cuDoubleComplex b, double tol = 1e-12) {
    return std::abs(cuCreal(a) - cuCreal(b)) < tol &&
           std::abs(cuCimag(a) - cuCimag(b)) < tol;
}

// ─── Same Device Transfer ────────────────────────────────────────────────────

TEST_F(PeerTransferTest, SameDeviceCopy) {
    const int N = 1024;
    cudaSetDevice(0);

    cuDoubleComplex* d_src;
    cuDoubleComplex* d_dst;
    cudaMalloc(&d_src, N * sizeof(cuDoubleComplex));
    cudaMalloc(&d_dst, N * sizeof(cuDoubleComplex));

    // Initialize source with known pattern
    std::vector<cuDoubleComplex> h_src(N);
    for (int i = 0; i < N; ++i) {
        h_src[i] = make_cuDoubleComplex(i * 1.0, -i * 0.5);
    }
    cudaMemcpy(d_src, h_src.data(), N * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    auto stats = xfer.transfer_state(d_src, 0, d_dst, 0, N);
    EXPECT_EQ(stats.bytes_transferred, N * sizeof(cuDoubleComplex));
    EXPECT_EQ(stats.src_device, 0);
    EXPECT_EQ(stats.dst_device, 0);

    // Verify
    std::vector<cuDoubleComplex> h_dst(N);
    cudaMemcpy(h_dst.data(), d_dst, N * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(complex_near(h_src[i], h_dst[i]))
            << "Mismatch at index " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

// ─── Cross-Device Transfer ───────────────────────────────────────────────────

TEST_F(PeerTransferTest, CrossDeviceTransfer) {
    int n = GPUContext::instance().num_devices();
    if (n < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for cross-device test";
    }

    const int N = 2048;

    cudaSetDevice(0);
    cuDoubleComplex* d_src;
    cudaMalloc(&d_src, N * sizeof(cuDoubleComplex));

    cudaSetDevice(1);
    cuDoubleComplex* d_dst;
    cudaMalloc(&d_dst, N * sizeof(cuDoubleComplex));

    // Initialize source
    std::vector<cuDoubleComplex> h_src(N);
    for (int i = 0; i < N; ++i) {
        h_src[i] = make_cuDoubleComplex(std::sin(i * 0.1), std::cos(i * 0.1));
    }
    cudaSetDevice(0);
    cudaMemcpy(d_src, h_src.data(), N * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    // Transfer 0→1
    auto stats = xfer.transfer_state(d_src, 0, d_dst, 1, N);
    EXPECT_EQ(stats.bytes_transferred, N * sizeof(cuDoubleComplex));

    // Verify on device 1
    cudaSetDevice(1);
    std::vector<cuDoubleComplex> h_dst(N);
    cudaMemcpy(h_dst.data(), d_dst, N * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(complex_near(h_src[i], h_dst[i]))
            << "Mismatch at index " << i;
    }

    cudaSetDevice(0);
    cudaFree(d_src);
    cudaSetDevice(1);
    cudaFree(d_dst);
}

// ─── Large Transfer ──────────────────────────────────────────────────────────

TEST_F(PeerTransferTest, LargeTransfer) {
    int n = GPUContext::instance().num_devices();
    if (n < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for large transfer test";
    }

    // Transfer larger than staging buffer chunk to test pipelining
    const int N = 4 * 1024 * 1024;  // 4M elements = 64 MiB

    cudaSetDevice(0);
    cuDoubleComplex* d_src;
    cudaMalloc(&d_src, N * sizeof(cuDoubleComplex));

    cudaSetDevice(1);
    cuDoubleComplex* d_dst;
    cudaMalloc(&d_dst, N * sizeof(cuDoubleComplex));

    // Initialize with simple pattern
    std::vector<cuDoubleComplex> h_src(N);
    for (int i = 0; i < N; ++i) {
        h_src[i] = make_cuDoubleComplex(i % 100, -(i % 50));
    }
    cudaSetDevice(0);
    cudaMemcpy(d_src, h_src.data(), N * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    auto stats = xfer.transfer_state(d_src, 0, d_dst, 1, N);
    EXPECT_EQ(stats.bytes_transferred, static_cast<size_t>(N) * sizeof(cuDoubleComplex));
    EXPECT_GT(stats.bandwidth_gbps, 0.0);

    // Spot-check first and last elements
    cudaSetDevice(1);
    cuDoubleComplex h_first, h_last;
    cudaMemcpy(&h_first, d_dst, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last, d_dst + N - 1, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(complex_near(h_first, h_src[0]));
    EXPECT_TRUE(complex_near(h_last, h_src[N - 1]));

    cudaSetDevice(0);
    cudaFree(d_src);
    cudaSetDevice(1);
    cudaFree(d_dst);
}

// ─── Batch Transfer ──────────────────────────────────────────────────────────

TEST_F(PeerTransferTest, BatchTransfer) {
    const int N = 512;
    cudaSetDevice(0);

    cuDoubleComplex* d_src1;
    cuDoubleComplex* d_dst1;
    cuDoubleComplex* d_src2;
    cuDoubleComplex* d_dst2;
    cudaMalloc(&d_src1, N * sizeof(cuDoubleComplex));
    cudaMalloc(&d_dst1, N * sizeof(cuDoubleComplex));
    cudaMalloc(&d_src2, N * sizeof(cuDoubleComplex));
    cudaMalloc(&d_dst2, N * sizeof(cuDoubleComplex));

    // Initialize
    std::vector<cuDoubleComplex> h(N);
    for (int i = 0; i < N; ++i) {
        h[i] = make_cuDoubleComplex(i, 0);
    }
    cudaMemcpy(d_src1, h.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src2, h.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    std::vector<PeerTransfer::BatchEntry> entries = {
        {d_src1, 0, d_dst1, 0, N},
        {d_src2, 0, d_dst2, 0, N}
    };

    size_t total = xfer.transfer_batch(entries);
    EXPECT_EQ(total, 2u * N * sizeof(cuDoubleComplex));

    cudaFree(d_src1);
    cudaFree(d_dst1);
    cudaFree(d_src2);
    cudaFree(d_dst2);
}

// ─── Zero-Length Transfer ────────────────────────────────────────────────────

TEST_F(PeerTransferTest, ZeroLengthTransfer) {
    auto stats = xfer.transfer_state(nullptr, 0, nullptr, 0, 0);
    EXPECT_EQ(stats.bytes_transferred, 0u);
}

// ─── Statistics ──────────────────────────────────────────────────────────────

TEST_F(PeerTransferTest, CumulativeStats) {
    xfer.reset_stats();
    auto stats = xfer.cumulative_stats();
    EXPECT_EQ(stats.bytes_transferred, 0u);

    // Do a transfer
    const int N = 128;
    cudaSetDevice(0);
    cuDoubleComplex* d_src;
    cuDoubleComplex* d_dst;
    cudaMalloc(&d_src, N * sizeof(cuDoubleComplex));
    cudaMalloc(&d_dst, N * sizeof(cuDoubleComplex));
    xfer.transfer_state(d_src, 0, d_dst, 0, N);

    stats = xfer.cumulative_stats();
    EXPECT_EQ(stats.bytes_transferred, N * sizeof(cuDoubleComplex));

    cudaFree(d_src);
    cudaFree(d_dst);
}
