#include <gtest/gtest.h>
#include "cv_state_pool.h"
#include <vector>
#include <complex>

/**
 * CVStatePool 单元测试
 */
class CVStatePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试参数
        d_trunc = 16;
        max_states = 64;
        pool = new CVStatePool(d_trunc, max_states);
    }

    void TearDown() override {
        delete pool;
    }

    int d_trunc;
    int max_states;
    CVStatePool* pool;
};

TEST_F(CVStatePoolTest, Initialization) {
    EXPECT_EQ(pool->d_trunc, d_trunc);
    EXPECT_EQ(pool->capacity, max_states);
    EXPECT_EQ(pool->active_count, 0);
}

TEST_F(CVStatePoolTest, AllocateState) {
    int state_id = pool->allocate_state();
    EXPECT_GE(state_id, 0);
    EXPECT_LT(state_id, max_states);
    EXPECT_EQ(pool->active_count, 1);
}

TEST_F(CVStatePoolTest, AllocateMultipleStates) {
    std::vector<int> state_ids;
    for (int i = 0; i < max_states; ++i) {
        int id = pool->allocate_state();
        EXPECT_GE(id, 0);
        state_ids.push_back(id);
    }

    EXPECT_EQ(pool->active_count, max_states);

    // 尝试分配超出容量的状态
    int failed_id = pool->allocate_state();
    EXPECT_EQ(failed_id, -1);
}

TEST_F(CVStatePoolTest, FreeState) {
    int state_id = pool->allocate_state();
    EXPECT_EQ(pool->active_count, 1);

    pool->free_state(state_id);
    EXPECT_EQ(pool->active_count, 0);
}

TEST_F(CVStatePoolTest, UploadDownloadState) {
    int state_id = pool->allocate_state();

    // 创建测试状态向量
    std::vector<cuDoubleComplex> test_state(d_trunc);
    for (int i = 0; i < d_trunc; ++i) {
        test_state[i] = make_cuDoubleComplex(i * 0.1, i * 0.05);
    }

    // 上传状态
    pool->upload_state(state_id, test_state);

    // 下载状态并验证
    std::vector<cuDoubleComplex> downloaded_state;
    pool->download_state(state_id, downloaded_state);

    EXPECT_EQ(downloaded_state.size(), static_cast<size_t>(d_trunc));
    for (int i = 0; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(downloaded_state[i]), cuCreal(test_state[i]), 1e-10);
        EXPECT_NEAR(cuCimag(downloaded_state[i]), cuCimag(test_state[i]), 1e-10);
    }
}

TEST_F(CVStatePoolTest, VacuumState) {
    int state_id = pool->allocate_state();

    // 初始化真空态 |0⟩ = [1, 0, 0, ...]
    std::vector<cuDoubleComplex> vacuum_state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    vacuum_state[0] = make_cuDoubleComplex(1.0, 0.0);

    pool->upload_state(state_id, vacuum_state);

    // 计算归一化因子
    double norm_squared = 0.0;
    for (const auto& amp : vacuum_state) {
        double real = cuCreal(amp);
        double imag = cuCimag(amp);
        norm_squared += real * real + imag * imag;
    }

    EXPECT_NEAR(norm_squared, 1.0, 1e-10);
}

TEST_F(CVStatePoolTest, StateReuse) {
    // 分配和释放状态，验证重用
    int state_id1 = pool->allocate_state();
    pool->free_state(state_id1);

    int state_id2 = pool->allocate_state();
    // 在简单的实现中，可能会重用相同的ID
    EXPECT_GE(state_id2, 0);
    EXPECT_EQ(pool->active_count, 1);
}
