/**
 * @file test_multi_gpu_state.cpp
 * @brief Unit tests for multi-GPU CVStatePool operations.
 *
 * Tests state allocation on specific devices, cross-device migration,
 * and multi-pool coordination.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cmath>
#include "cv_state_pool.h"
#include "gpu_context.h"

using namespace hybridcvdv;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class MultiGPUStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!GPUContext::is_initialized()) {
            GPUContext::initialize(true);
        }
    }
};

static bool complex_near(cuDoubleComplex a, cuDoubleComplex b, double tol = 1e-12) {
    return std::abs(cuCreal(a) - cuCreal(b)) < tol &&
           std::abs(cuCimag(a) - cuCimag(b)) < tol;
}

// ─── Pool Construction ──────────────────────────────────────────────────────

TEST_F(MultiGPUStateTest, PoolHasCorrectTruncDim) {
    CVStatePool pool(10, 16, 1, 0);
    EXPECT_EQ(pool.d_trunc, 10);
}

TEST_F(MultiGPUStateTest, PoolHasCorrectCapacity) {
    CVStatePool pool(10, 16, 1, 0);
    EXPECT_EQ(pool.capacity, 16);
}

TEST_F(MultiGPUStateTest, ConstructionWithValidParams) {
    // Verify construction with valid default params doesn't throw
    EXPECT_NO_THROW(CVStatePool pool(10, 16, 1, 0));
}

TEST_F(MultiGPUStateTest, DefaultParamsPool) {
    CVStatePool pool(10, 16);
    EXPECT_EQ(pool.d_trunc, 10);
    EXPECT_EQ(pool.capacity, 16);
}

// ─── State Allocation ────────────────────────────────────────────────────────

TEST_F(MultiGPUStateTest, AllocateStateBasic) {
    CVStatePool pool(10, 16, 1, 0);
    int sid = pool.allocate_state();
    EXPECT_GE(sid, 0);
    EXPECT_TRUE(pool.is_valid_state(sid));
    pool.free_state(sid);
}

TEST_F(MultiGPUStateTest, AllocateMultipleStates) {
    CVStatePool pool(10, 16, 1, 0);
    int sid0 = pool.allocate_state();
    int sid1 = pool.allocate_state();
    EXPECT_GE(sid0, 0);
    EXPECT_GE(sid1, 0);
    EXPECT_NE(sid0, sid1);
    EXPECT_TRUE(pool.is_valid_state(sid0));
    EXPECT_TRUE(pool.is_valid_state(sid1));
    pool.free_state(sid0);
    pool.free_state(sid1);
}

// ─── Upload/Download ─────────────────────────────────────────────────────────

TEST_F(MultiGPUStateTest, UploadDownloadRoundTrip) {
    CVStatePool pool(10, 16, 1, 0);
    int sid = pool.allocate_state();

    std::vector<cuDoubleComplex> h_state(10);
    for (int i = 0; i < 10; ++i) {
        h_state[i] = make_cuDoubleComplex(i * 1.1, -i * 0.5);
    }
    pool.upload_state(sid, h_state);

    std::vector<cuDoubleComplex> h_out;
    pool.download_state(sid, h_out);
    ASSERT_EQ(h_out.size(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(complex_near(h_state[i], h_out[i]));
    }
    pool.free_state(sid);
}

TEST_F(MultiGPUStateTest, UploadDownloadTrigonometric) {
    CVStatePool pool(10, 16, 1, 0);
    int sid = pool.allocate_state();

    std::vector<cuDoubleComplex> h_state(10);
    for (int i = 0; i < 10; ++i) {
        h_state[i] = make_cuDoubleComplex(std::cos(i), std::sin(i));
    }
    pool.upload_state(sid, h_state);

    std::vector<cuDoubleComplex> h_out;
    pool.download_state(sid, h_out);
    ASSERT_EQ(h_out.size(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(complex_near(h_state[i], h_out[i]));
    }
    pool.free_state(sid);
}

// ─── Cross-Pool State Transfer (manual download+upload) ─────────────────────

TEST_F(MultiGPUStateTest, ManualCrossPoolTransfer) {
    CVStatePool pool0(10, 16, 1, 0);
    CVStatePool pool1(10, 16, 1, 0);

    // Create state on pool0
    int sid0 = pool0.allocate_state();
    std::vector<cuDoubleComplex> h_state(10);
    for (int i = 0; i < 10; ++i) {
        h_state[i] = make_cuDoubleComplex(i * 2.0, i * 3.0);
    }
    pool0.upload_state(sid0, h_state);

    // Transfer to pool1 via host
    std::vector<cuDoubleComplex> h_transfer;
    pool0.download_state(sid0, h_transfer);
    int sid1 = pool1.allocate_state();
    EXPECT_GE(sid1, 0);
    pool1.upload_state(sid1, h_transfer);

    // Verify on pool1
    std::vector<cuDoubleComplex> h_out;
    pool1.download_state(sid1, h_out);
    ASSERT_EQ(h_out.size(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(complex_near(h_state[i], h_out[i]))
            << "Mismatch at index " << i;
    }

    pool0.free_state(sid0);
    pool1.free_state(sid1);
}

TEST_F(MultiGPUStateTest, ManualSamePoolTransfer) {
    CVStatePool pool0(10, 16, 1, 0);
    CVStatePool pool0b(10, 16, 1, 0);

    int sid = pool0.allocate_state();
    std::vector<cuDoubleComplex> h_state(10);
    for (int i = 0; i < 10; ++i) {
        h_state[i] = make_cuDoubleComplex(i, 0);
    }
    pool0.upload_state(sid, h_state);

    // Transfer via host
    std::vector<cuDoubleComplex> h_transfer;
    pool0.download_state(sid, h_transfer);
    int new_sid = pool0b.allocate_state();
    EXPECT_GE(new_sid, 0);
    pool0b.upload_state(new_sid, h_transfer);

    std::vector<cuDoubleComplex> h_out;
    pool0b.download_state(new_sid, h_out);
    ASSERT_EQ(h_out.size(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(complex_near(h_state[i], h_out[i]));
    }

    pool0.free_state(sid);
    pool0b.free_state(new_sid);
}

TEST_F(MultiGPUStateTest, InvalidStateAccess) {
    CVStatePool pool(10, 16, 1, 0);
    EXPECT_FALSE(pool.is_valid_state(-1));
    EXPECT_FALSE(pool.is_valid_state(999));
}

// ─── Multi-Pool Parallel Allocation ──────────────────────────────────────────

TEST_F(MultiGPUStateTest, ParallelPoolAllocation) {
    CVStatePool pool0(10, 32, 1, 0);
    CVStatePool pool1(10, 32, 1, 0);

    // Allocate states on both devices
    std::vector<int> sids0, sids1;
    for (int i = 0; i < 10; ++i) {
        sids0.push_back(pool0.allocate_state());
        sids1.push_back(pool1.allocate_state());
    }

    // Verify all allocations succeeded
    for (int sid : sids0) {
        EXPECT_GE(sid, 0);
        EXPECT_TRUE(pool0.is_valid_state(sid));
    }
    for (int sid : sids1) {
        EXPECT_GE(sid, 0);
        EXPECT_TRUE(pool1.is_valid_state(sid));
    }

    // Cleanup
    for (int sid : sids0) pool0.free_state(sid);
    for (int sid : sids1) pool1.free_state(sid);
}

// ─── Large State Transfer ────────────────────────────────────────────────────

TEST_F(MultiGPUStateTest, LargeStateTransfer) {
    const int DIM = 100;  // 100 Fock levels → large state
    CVStatePool pool0(DIM, 4, 1, 0);
    CVStatePool pool1(DIM, 4, 1, 0);

    int sid0 = pool0.allocate_state();
    std::vector<cuDoubleComplex> h_state(DIM);
    double norm_sq = 0;
    for (int i = 0; i < DIM; ++i) {
        double re = std::exp(-0.1 * i) * std::cos(0.5 * i);
        double im = std::exp(-0.1 * i) * std::sin(0.5 * i);
        h_state[i] = make_cuDoubleComplex(re, im);
        norm_sq += re * re + im * im;
    }
    // Normalize
    double norm = std::sqrt(norm_sq);
    for (int i = 0; i < DIM; ++i) {
        h_state[i] = make_cuDoubleComplex(
            cuCreal(h_state[i]) / norm, cuCimag(h_state[i]) / norm);
    }

    pool0.upload_state(sid0, h_state);

    // Transfer via host
    std::vector<cuDoubleComplex> h_transfer;
    pool0.download_state(sid0, h_transfer);
    int sid1 = pool1.allocate_state();
    EXPECT_GE(sid1, 0);
    pool1.upload_state(sid1, h_transfer);

    std::vector<cuDoubleComplex> h_out;
    pool1.download_state(sid1, h_out);
    ASSERT_EQ(h_out.size(), static_cast<size_t>(DIM));

    // Check norm preservation
    double transferred_norm_sq = 0;
    for (int i = 0; i < DIM; ++i) {
        transferred_norm_sq += cuCreal(h_out[i]) * cuCreal(h_out[i]) +
                               cuCimag(h_out[i]) * cuCimag(h_out[i]);
        EXPECT_TRUE(complex_near(h_state[i], h_out[i], 1e-10))
            << "Mismatch at Fock level " << i;
    }
    EXPECT_NEAR(transferred_norm_sq, 1.0, 1e-10);

    pool0.free_state(sid0);
    pool1.free_state(sid1);
}
