/**
 * @file test_stream_pool.cpp
 * @brief Unit tests for StreamPool - CUDA stream management.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "gpu_context.h"
#include "stream_pool.h"

using namespace hybridcvdv;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class StreamPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!GPUContext::is_initialized()) {
            GPUContext::initialize(true);
        }
        int n = GPUContext::instance().num_devices();
        if (!StreamPool::is_initialized()) {
            ASSERT_TRUE(StreamPool::initialize(n, 4, 2));
        }
    }
};

// ─── Basic Stream Access ─────────────────────────────────────────────────────

TEST_F(StreamPoolTest, ComputeStreamNotNull) {
    auto& pool = StreamPool::instance();
    cudaStream_t s = pool.get_compute_stream(0);
    EXPECT_NE(s, nullptr);
}

TEST_F(StreamPoolTest, TransferStreamNotNull) {
    auto& pool = StreamPool::instance();
    cudaStream_t s = pool.get_transfer_stream(0);
    EXPECT_NE(s, nullptr);
}

TEST_F(StreamPoolTest, InvalidDeviceReturnsNull) {
    auto& pool = StreamPool::instance();
    cudaStream_t s = pool.get_compute_stream(-1);
    EXPECT_EQ(s, nullptr);
    s = pool.get_compute_stream(999);
    EXPECT_EQ(s, nullptr);
}

TEST_F(StreamPoolTest, RoundRobinStreams) {
    auto& pool = StreamPool::instance();
    cudaStream_t s0 = pool.get_compute_stream(0);
    cudaStream_t s1 = pool.get_compute_stream(0);
    cudaStream_t s2 = pool.get_compute_stream(0);
    cudaStream_t s3 = pool.get_compute_stream(0);
    // After 4 streams, should wrap around
    cudaStream_t s4 = pool.get_compute_stream(0);
    EXPECT_EQ(s0, s4);  // Round-robin wraps
}

TEST_F(StreamPoolTest, DeviceStreamsAccessor) {
    auto& pool = StreamPool::instance();
    const auto& ds = pool.device_streams(0);
    EXPECT_EQ(ds.device_id, 0);
    EXPECT_EQ(ds.compute_streams.size(), 4u);
    EXPECT_EQ(ds.transfer_streams.size(), 2u);
}

TEST_F(StreamPoolTest, InvalidDeviceStreamsThrows) {
    auto& pool = StreamPool::instance();
    EXPECT_THROW(pool.device_streams(-1), std::out_of_range);
}

// ─── Synchronization ─────────────────────────────────────────────────────────

TEST_F(StreamPoolTest, SyncDeviceDoesNotThrow) {
    auto& pool = StreamPool::instance();
    EXPECT_NO_THROW(pool.sync_device(0));
}

TEST_F(StreamPoolTest, SyncAllDoesNotThrow) {
    auto& pool = StreamPool::instance();
    EXPECT_NO_THROW(pool.sync_all());
}

TEST_F(StreamPoolTest, SyncNullStreamDoesNotThrow) {
    auto& pool = StreamPool::instance();
    EXPECT_NO_THROW(pool.sync_stream(nullptr));
}

// ─── Events ──────────────────────────────────────────────────────────────────

TEST_F(StreamPoolTest, RecordAndWaitEvent) {
    auto& pool = StreamPool::instance();
    cudaStream_t s = pool.get_compute_stream(0);

    cudaEvent_t event = pool.record_event(s);
    EXPECT_NE(event, nullptr);

    // Another stream waits for the event
    cudaStream_t s2 = pool.get_compute_stream(0);
    EXPECT_NO_THROW(pool.wait_event(s2, event));

    cudaEventDestroy(event);
}

TEST_F(StreamPoolTest, CrossDeviceSync) {
    auto& pool = StreamPool::instance();
    int n = GPUContext::instance().num_devices();
    if (n < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for cross-device sync test";
    }

    cudaStream_t src = pool.get_compute_stream(0);
    cudaStream_t dst = pool.get_compute_stream(1);
    EXPECT_NO_THROW(pool.cross_device_sync(src, dst));
}

// ─── Kernel Launch on Stream ─────────────────────────────────────────────────

TEST_F(StreamPoolTest, KernelLaunchOnComputeStream) {
    auto& pool = StreamPool::instance();
    cudaStream_t s = pool.get_compute_stream(0);

    // Allocate a small buffer and do a memset async
    void* d_buf;
    cudaMalloc(&d_buf, 256);
    cudaMemsetAsync(d_buf, 0, 256, s);
    pool.sync_stream(s);

    // Verify by downloading
    char h_buf[256];
    cudaMemcpy(h_buf, d_buf, 256, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(h_buf[i], 0);
    }
    cudaFree(d_buf);
}
