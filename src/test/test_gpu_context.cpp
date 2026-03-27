/**
 * @file test_gpu_context.cpp
 * @brief Unit tests for GPUContext - device detection, P2P, selection.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "gpu_context.h"

using namespace hybridcvdv;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class GPUContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure context is initialized
        if (!GPUContext::is_initialized()) {
            ASSERT_TRUE(GPUContext::initialize(true));
        }
    }
};

// ─── Device Detection ────────────────────────────────────────────────────────

TEST_F(GPUContextTest, HasAtLeastOneDevice) {
    const auto& ctx = GPUContext::instance();
    EXPECT_GE(ctx.num_devices(), 1);
}

TEST_F(GPUContextTest, DeviceInfoIsValid) {
    const auto& ctx = GPUContext::instance();
    for (int d = 0; d < ctx.num_devices(); ++d) {
        const auto& info = ctx.device_info(d);
        EXPECT_EQ(info.device_id, d);
        EXPECT_GT(info.total_memory_bytes, 0u);
        EXPECT_GT(info.sm_count, 0);
        EXPECT_EQ(info.warp_size, 32);
        EXPECT_FALSE(info.name.empty());
        EXPECT_GE(info.compute_major, 3);
    }
}

TEST_F(GPUContextTest, InvalidDeviceIdThrows) {
    const auto& ctx = GPUContext::instance();
    EXPECT_THROW(ctx.device_info(-1), std::out_of_range);
    EXPECT_THROW(ctx.device_info(999), std::out_of_range);
}

TEST_F(GPUContextTest, MemoryUtilization) {
    const auto& ctx = GPUContext::instance();
    const auto& info = ctx.device_info(0);
    double util = info.memory_utilization();
    EXPECT_GE(util, 0.0);
    EXPECT_LE(util, 1.0);
}

// ─── Memory Refresh ──────────────────────────────────────────────────────────

TEST_F(GPUContextTest, RefreshMemoryInfo) {
    auto& ctx = GPUContext::instance();
    ctx.refresh_memory_info();
    const auto& info = ctx.device_info(0);
    EXPECT_GT(info.free_memory_bytes, 0u);
    EXPECT_LE(info.free_memory_bytes, info.total_memory_bytes);
}

TEST_F(GPUContextTest, TotalFreeMemory) {
    const auto& ctx = GPUContext::instance();
    size_t total = ctx.total_free_memory();
    EXPECT_GT(total, 0u);
}

// ─── Device Selection ────────────────────────────────────────────────────────

TEST_F(GPUContextTest, RoundRobinSelection) {
    const auto& ctx = GPUContext::instance();
    if (ctx.num_devices() < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for round-robin test";
    }

    int d0 = ctx.select_device(DeviceSelectionPolicy::ROUND_ROBIN);
    int d1 = ctx.select_device(DeviceSelectionPolicy::ROUND_ROBIN);
    // With 2 devices, round-robin should alternate
    EXPECT_NE(d0, d1);
}

TEST_F(GPUContextTest, MostFreeMemSelection) {
    const auto& ctx = GPUContext::instance();
    int dev = ctx.select_device(DeviceSelectionPolicy::MOST_FREE_MEM);
    EXPECT_GE(dev, 0);
    EXPECT_LT(dev, ctx.num_devices());
}

TEST_F(GPUContextTest, AffinitySelection) {
    const auto& ctx = GPUContext::instance();
    int dev = ctx.select_device(DeviceSelectionPolicy::AFFINITY, 0);
    EXPECT_EQ(dev, 0);
}

TEST_F(GPUContextTest, SetDevice) {
    const auto& ctx = GPUContext::instance();
    ctx.set_device(0);
    EXPECT_EQ(ctx.current_device(), 0);
}

// ─── P2P Access ──────────────────────────────────────────────────────────────

TEST_F(GPUContextTest, P2PQueries) {
    const auto& ctx = GPUContext::instance();
    if (ctx.num_devices() < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for P2P test";
    }

    // P2P with self makes no sense (not in link list)
    EXPECT_FALSE(ctx.is_p2p_enabled(0, 0));

    // Check 0→1 link status (might be true or false depending on hardware)
    bool p2p_01 = ctx.is_p2p_enabled(0, 1);
    // Just verify it doesn't crash; actual P2P depends on hardware
    (void)p2p_01;
}

TEST_F(GPUContextTest, P2PLinksConsistency) {
    const auto& ctx = GPUContext::instance();
    const auto& links = ctx.p2p_links();
    for (const auto& link : links) {
        EXPECT_NE(link.src_device, link.dst_device);
        EXPECT_GE(link.src_device, 0);
        EXPECT_GE(link.dst_device, 0);
        EXPECT_LT(link.src_device, ctx.num_devices());
        EXPECT_LT(link.dst_device, ctx.num_devices());
        if (link.access_enabled) {
            EXPECT_TRUE(link.access_supported);
        }
    }
}

// ─── DeviceGuard ─────────────────────────────────────────────────────────────

TEST_F(GPUContextTest, DeviceGuardRestoresDevice) {
    const auto& ctx = GPUContext::instance();
    if (ctx.num_devices() < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for DeviceGuard test";
    }

    ctx.set_device(0);
    EXPECT_EQ(ctx.current_device(), 0);

    {
        DeviceGuard guard(1);
        EXPECT_EQ(ctx.current_device(), 1);
    }
    // Should be restored to 0
    EXPECT_EQ(ctx.current_device(), 0);
}

TEST_F(GPUContextTest, DeviceGuardSameDevice) {
    const auto& ctx = GPUContext::instance();
    ctx.set_device(0);
    {
        DeviceGuard guard(0);
        EXPECT_EQ(ctx.current_device(), 0);
    }
    EXPECT_EQ(ctx.current_device(), 0);
}

// ─── Print Summary ───────────────────────────────────────────────────────────

TEST_F(GPUContextTest, PrintSummaryDoesNotCrash) {
    const auto& ctx = GPUContext::instance();
    EXPECT_NO_THROW(ctx.print_device_summary());
}

// ─── Multi-GPU Detection ─────────────────────────────────────────────────────

TEST_F(GPUContextTest, IsMultiGPU) {
    const auto& ctx = GPUContext::instance();
    if (ctx.num_devices() >= 2) {
        EXPECT_TRUE(ctx.is_multi_gpu());
    } else {
        EXPECT_FALSE(ctx.is_multi_gpu());
    }
}

TEST_F(GPUContextTest, ValidDeviceCheck) {
    const auto& ctx = GPUContext::instance();
    EXPECT_TRUE(ctx.is_valid_device(0));
    EXPECT_FALSE(ctx.is_valid_device(-1));
    EXPECT_FALSE(ctx.is_valid_device(999));
}
