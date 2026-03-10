#include <gtest/gtest.h>
#include "cv_state_pool.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// 声明外部函数
extern void apply_pauli_x(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_pauli_y(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_pauli_z(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_sigma_plus(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_sigma_minus(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_projector_0(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_projector_1(CVStatePool* pool, const int* targets, int batch_size);

// 辅助宏
#define ALLOC_DEVICE_TARGETS(d_ptr, host_array, size) \
    do { \
        cudaMalloc(&(d_ptr), (size) * sizeof(int)); \
        cudaMemcpy((d_ptr), (host_array), (size) * sizeof(int), cudaMemcpyHostToDevice); \
    } while(0)

#define FREE_DEVICE_TARGETS(d_ptr) \
    do { \
        cudaFree(d_ptr); \
        (d_ptr) = nullptr; \
    } while(0)

/**
 * 量子比特门操作单元测试
 */
class QubitGatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        d_trunc = 2;  // 量子比特维度为 2
        max_states = 4;
        pool = new CVStatePool(d_trunc, max_states, 1);

        // 初始化测试状态
        state_id = pool->allocate_state();
    }

    void TearDown() override {
        delete pool;
    }

    int d_trunc;
    int max_states;
    CVStatePool* pool;
    int state_id;

    // 辅助函数：设置 |0⟩ 态
    void set_state_0() {
        std::vector<cuDoubleComplex> state(2);
        state[0] = make_cuDoubleComplex(1.0, 0.0);
        state[1] = make_cuDoubleComplex(0.0, 0.0);
        pool->upload_state(state_id, state);
    }

    // 辅助函数：设置 |1⟩ 态
    void set_state_1() {
        std::vector<cuDoubleComplex> state(2);
        state[0] = make_cuDoubleComplex(0.0, 0.0);
        state[1] = make_cuDoubleComplex(1.0, 0.0);
        pool->upload_state(state_id, state);
    }

    // 辅助函数：设置 |+⟩ = (|0⟩ + |1⟩)/√2 态
    void set_state_plus() {
        std::vector<cuDoubleComplex> state(2);
        double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        state[0] = make_cuDoubleComplex(inv_sqrt2, 0.0);
        state[1] = make_cuDoubleComplex(inv_sqrt2, 0.0);
        pool->upload_state(state_id, state);
    }
};

TEST_F(QubitGatesTest, PauliX_State0) {
    // X|0⟩ = |1⟩
    set_state_0();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_pauli_x(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, PauliX_State1) {
    // X|1⟩ = |0⟩
    set_state_1();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_pauli_x(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, PauliY_State0) {
    // Y|0⟩ = i|1⟩
    set_state_0();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_pauli_y(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 1.0, 1e-10);
}

TEST_F(QubitGatesTest, PauliY_State1) {
    // Y|1⟩ = -i|0⟩
    set_state_1();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_pauli_y(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), -1.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, PauliZ_State0) {
    // Z|0⟩ = |0⟩
    set_state_0();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_pauli_z(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, PauliZ_State1) {
    // Z|1⟩ = -|1⟩
    set_state_1();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_pauli_z(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), -1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, SigmaPlus_State0) {
    // σ+|0⟩ = |1⟩
    set_state_0();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_sigma_plus(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, SigmaPlus_State1) {
    // σ+|1⟩ = 0
    set_state_1();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_sigma_plus(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, SigmaMinus_State0) {
    // σ-|0⟩ = 0
    set_state_0();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_sigma_minus(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, SigmaMinus_State1) {
    // σ-|1⟩ = |0⟩
    set_state_1();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_sigma_minus(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, Projector0_State0) {
    // P0|0⟩ = |0⟩
    set_state_0();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_projector_0(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, Projector0_State1) {
    // P0|1⟩ = 0
    set_state_1();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_projector_0(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, Projector1_State0) {
    // P1|0⟩ = 0
    set_state_0();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_projector_1(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, Projector1_State1) {
    // P1|1⟩ = |1⟩
    set_state_1();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_projector_1(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
}

TEST_F(QubitGatesTest, PauliCommutationRelations) {
    // 测试 Pauli 矩阵的对易关系
    // XY = iZ, YZ = iX, ZX = iY
    
    set_state_plus();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    // 应用 X 然后 Y
    apply_pauli_x(pool, d_targets, 1);
    apply_pauli_y(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result_xy;
    pool->download_state(state_id, result_xy);

    // 重置状态并应用 Y 然后 X
    set_state_plus();
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_pauli_y(pool, d_targets, 1);
    apply_pauli_x(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result_yx;
    pool->download_state(state_id, result_yx);

    // XY - YX 应该等于 2iZ
    // 这里只验证它们不相等（因为不对易）
    bool different = false;
    for (int i = 0; i < 2; ++i) {
        if (std::abs(cuCreal(result_xy[i]) - cuCreal(result_yx[i])) > 1e-10 ||
            std::abs(cuCimag(result_xy[i]) - cuCimag(result_yx[i])) > 1e-10) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

TEST_F(QubitGatesTest, BatchProcessing) {
    // 测试批处理
    int state_id2 = pool->allocate_state();
    set_state_0();
    
    std::vector<cuDoubleComplex> state2(2);
    state2[0] = make_cuDoubleComplex(0.0, 0.0);
    state2[1] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id2, state2);

    int host_targets[] = {state_id, state_id2};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 2);

    apply_pauli_x(pool, d_targets, 2);

    FREE_DEVICE_TARGETS(d_targets);

    // 验证第一个状态: X|0⟩ = |1⟩
    std::vector<cuDoubleComplex> result1;
    pool->download_state(state_id, result1);
    EXPECT_NEAR(cuCreal(result1[1]), 1.0, 1e-10);

    // 验证第二个状态: X|1⟩ = |0⟩
    std::vector<cuDoubleComplex> result2;
    pool->download_state(state_id2, result2);
    EXPECT_NEAR(cuCreal(result2[0]), 1.0, 1e-10);

    pool->free_state(state_id2);
}
