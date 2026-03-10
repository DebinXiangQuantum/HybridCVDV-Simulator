#include <gtest/gtest.h>
#include "cv_state_pool.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// 声明外部函数
extern void apply_xgate(CVStatePool* pool, const int* targets, int batch_size, double x);
extern void apply_zgate(CVStatePool* pool, const int* targets, int batch_size, double p);
extern void apply_pgate(CVStatePool* pool, const int* targets, int batch_size, double s);
extern void apply_vgate(CVStatePool* pool, const int* targets, int batch_size, double gamma);
extern void apply_fouriergate(CVStatePool* pool, const int* targets, int batch_size);

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
 * Strawberry Fields 单模门单元测试
 */
class SFSingleModeGatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        d_trunc = 8;
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

    // 辅助函数：设置 Fock 态 |n⟩
    void set_fock_state(int n) {
        std::vector<cuDoubleComplex> state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
        if (n < d_trunc) {
            state[n] = make_cuDoubleComplex(1.0, 0.0);
        }
        pool->upload_state(state_id, state);
    }

    // 辅助函数：设置叠加态
    void set_superposition_state() {
        std::vector<cuDoubleComplex> state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
        double norm = 1.0 / sqrt(2.0);
        state[0] = make_cuDoubleComplex(norm, 0.0);
        state[1] = make_cuDoubleComplex(norm, 0.0);
        pool->upload_state(state_id, state);
    }

    // 辅助函数：验证归一化
    void expect_normalized(const std::vector<cuDoubleComplex>& state, double tolerance = 1e-6) {
        double norm_sq = 0.0;
        for (const auto& amp : state) {
            double real = cuCreal(amp);
            double imag = cuCimag(amp);
            norm_sq += real * real + imag * imag;
        }
        EXPECT_NEAR(sqrt(norm_sq), 1.0, tolerance);
    }
};

// ==================== Xgate 测试 ====================

TEST_F(SFSingleModeGatesTest, Xgate_VacuumState) {
    // 测试 Xgate 作用在真空态上
    // X(x)|0⟩ = D(x/√2)|0⟩
    set_fock_state(0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_xgate(pool, d_targets, 1, 1.0);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 验证归一化
    expect_normalized(result);

    // Xgate 应该产生位移相干态
    // 主要分量应该在低 Fock 态
    double total_prob = 0.0;
    for (int i = 0; i < 4; ++i) {
        double prob = cuCreal(result[i]) * cuCreal(result[i]) + 
                     cuCimag(result[i]) * cuCimag(result[i]);
        total_prob += prob;
    }
    EXPECT_GT(total_prob, 0.8);  // 大部分概率在前4个态
}

TEST_F(SFSingleModeGatesTest, Xgate_Identity) {
    // 测试 X(0) 应该是恒等操作
    set_fock_state(1);

    std::vector<cuDoubleComplex> original;
    pool->download_state(state_id, original);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_xgate(pool, d_targets, 1, 0.0);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 状态应该基本保持不变
    for (int i = 0; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), cuCreal(original[i]), 1e-6);
        EXPECT_NEAR(cuCimag(result[i]), cuCimag(original[i]), 1e-6);
    }
}

// ==================== Zgate 测试 ====================

TEST_F(SFSingleModeGatesTest, Zgate_VacuumState) {
    // 测试 Zgate 作用在真空态上
    // Z(p)|0⟩ = D(ip/√2)|0⟩
    set_fock_state(0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_zgate(pool, d_targets, 1, 1.0);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 验证归一化
    expect_normalized(result);

    // Zgate 应该产生位移相干态（虚位移）
    double total_prob = 0.0;
    for (int i = 0; i < 4; ++i) {
        double prob = cuCreal(result[i]) * cuCreal(result[i]) + 
                     cuCimag(result[i]) * cuCimag(result[i]);
        total_prob += prob;
    }
    EXPECT_GT(total_prob, 0.8);
}

TEST_F(SFSingleModeGatesTest, Zgate_Identity) {
    // 测试 Z(0) 应该是恒等操作
    set_fock_state(2);

    std::vector<cuDoubleComplex> original;
    pool->download_state(state_id, original);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_zgate(pool, d_targets, 1, 0.0);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 状态应该基本保持不变
    for (int i = 0; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), cuCreal(original[i]), 1e-6);
        EXPECT_NEAR(cuCimag(result[i]), cuCimag(original[i]), 1e-6);
    }
}

// ==================== Pgate 测试 ====================

TEST_F(SFSingleModeGatesTest, Pgate_VacuumState) {
    // 测试 Pgate 作用在真空态上
    // P(s)|0⟩ = exp(i s/4)|0⟩
    set_fock_state(0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    double s = M_PI;
    apply_pgate(pool, d_targets, 1, s);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // P(s)|0⟩ = exp(i s (0 + 1/2)/2)|0⟩ = exp(i s/4)|0⟩
    double phase = s / 4.0;
    double expected_real = cos(phase);
    double expected_imag = sin(phase);

    EXPECT_NEAR(cuCreal(result[0]), expected_real, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), expected_imag, 1e-10);

    // 其他分量应该为 0
    for (int i = 1; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), 0.0, 1e-10);
        EXPECT_NEAR(cuCimag(result[i]), 0.0, 1e-10);
    }

    expect_normalized(result);
}

TEST_F(SFSingleModeGatesTest, Pgate_FockState) {
    // 测试 Pgate 作用在 Fock 态上
    // P(s)|n⟩ = exp(i s (n + 1/2)/2)|n⟩
    set_fock_state(2);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    double s = M_PI / 2.0;
    apply_pgate(pool, d_targets, 1, s);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // P(π/2)|2⟩ = exp(i π/2 (2 + 1/2)/2)|2⟩ = exp(i 5π/8)|2⟩
    double phase = s * (2.0 + 0.5) / 2.0;
    double expected_real = cos(phase);
    double expected_imag = sin(phase);

    EXPECT_NEAR(cuCreal(result[2]), expected_real, 1e-10);
    EXPECT_NEAR(cuCimag(result[2]), expected_imag, 1e-10);

    expect_normalized(result);
}

TEST_F(SFSingleModeGatesTest, Pgate_Identity) {
    // 测试 P(0) 应该是恒等操作
    set_superposition_state();

    std::vector<cuDoubleComplex> original;
    pool->download_state(state_id, original);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_pgate(pool, d_targets, 1, 0.0);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 状态应该保持不变
    for (int i = 0; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), cuCreal(original[i]), 1e-10);
        EXPECT_NEAR(cuCimag(result[i]), cuCimag(original[i]), 1e-10);
    }
}

TEST_F(SFSingleModeGatesTest, Pgate_Superposition) {
    // 测试 Pgate 作用在叠加态上
    set_superposition_state();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    double s = M_PI;
    apply_pgate(pool, d_targets, 1, s);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 验证每个 Fock 态获得正确的相位
    double norm = 1.0 / sqrt(2.0);
    
    // |0⟩ 分量
    double phase0 = s * 0.5 / 2.0;
    EXPECT_NEAR(cuCreal(result[0]), norm * cos(phase0), 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), norm * sin(phase0), 1e-10);

    // |1⟩ 分量
    double phase1 = s * 1.5 / 2.0;
    EXPECT_NEAR(cuCreal(result[1]), norm * cos(phase1), 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), norm * sin(phase1), 1e-10);

    expect_normalized(result);
}

// ==================== Vgate 测试 ====================

TEST_F(SFSingleModeGatesTest, Vgate_VacuumState) {
    // 测试 Vgate 作用在真空态上
    set_fock_state(0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_vgate(pool, d_targets, 1, 0.1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // Vgate 是三次相位门，实现较复杂
    // 这里只验证归一化和基本性质
    expect_normalized(result, 1e-5);
}

TEST_F(SFSingleModeGatesTest, Vgate_Identity) {
    // 测试 V(0) 应该接近恒等操作
    set_fock_state(1);

    std::vector<cuDoubleComplex> original;
    pool->download_state(state_id, original);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_vgate(pool, d_targets, 1, 0.0);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 状态应该基本保持不变
    for (int i = 0; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), cuCreal(original[i]), 1e-5);
        EXPECT_NEAR(cuCimag(result[i]), cuCimag(original[i]), 1e-5);
    }
}

// ==================== Fouriergate 测试 ====================

TEST_F(SFSingleModeGatesTest, Fouriergate_VacuumState) {
    // 测试 Fouriergate 作用在真空态上
    // F|0⟩ = exp(i π/4)|0⟩
    set_fock_state(0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_fouriergate(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // F|0⟩ = exp(i π/2 (0 + 1/2))|0⟩ = exp(i π/4)|0⟩
    double phase = M_PI / 4.0;
    double expected_real = cos(phase);
    double expected_imag = sin(phase);

    EXPECT_NEAR(cuCreal(result[0]), expected_real, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), expected_imag, 1e-10);

    // 其他分量应该为 0
    for (int i = 1; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), 0.0, 1e-10);
        EXPECT_NEAR(cuCimag(result[i]), 0.0, 1e-10);
    }

    expect_normalized(result);
}

TEST_F(SFSingleModeGatesTest, Fouriergate_FockState) {
    // 测试 Fouriergate 作用在 Fock 态上
    // F|n⟩ = exp(i π/2 (n + 1/2))|n⟩ = i^(n+1/2)|n⟩
    set_fock_state(1);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_fouriergate(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // F|1⟩ = exp(i π/2 (1 + 1/2))|1⟩ = exp(i 3π/4)|1⟩
    double phase = M_PI / 2.0 * 1.5;
    double expected_real = cos(phase);
    double expected_imag = sin(phase);

    EXPECT_NEAR(cuCreal(result[1]), expected_real, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), expected_imag, 1e-10);

    expect_normalized(result);
}

TEST_F(SFSingleModeGatesTest, Fouriergate_Superposition) {
    // 测试 Fouriergate 作用在叠加态上
    set_superposition_state();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_fouriergate(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    double norm = 1.0 / sqrt(2.0);

    // |0⟩ 分量: exp(i π/4)
    double phase0 = M_PI / 4.0;
    EXPECT_NEAR(cuCreal(result[0]), norm * cos(phase0), 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), norm * sin(phase0), 1e-10);

    // |1⟩ 分量: exp(i 3π/4)
    double phase1 = 3.0 * M_PI / 4.0;
    EXPECT_NEAR(cuCreal(result[1]), norm * cos(phase1), 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), norm * sin(phase1), 1e-10);

    expect_normalized(result);
}

TEST_F(SFSingleModeGatesTest, Fouriergate_FourthPower) {
    // 测试 F^4 = I (Fourier 门的四次方是恒等)
    set_superposition_state();

    std::vector<cuDoubleComplex> original;
    pool->download_state(state_id, original);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    // 应用 F 四次
    for (int i = 0; i < 4; ++i) {
        apply_fouriergate(pool, d_targets, 1);
    }

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 状态应该回到原始状态（可能有全局相位）
    // 验证概率分布相同
    for (int i = 0; i < d_trunc; ++i) {
        double orig_prob = cuCreal(original[i]) * cuCreal(original[i]) + 
                          cuCimag(original[i]) * cuCimag(original[i]);
        double result_prob = cuCreal(result[i]) * cuCreal(result[i]) + 
                            cuCimag(result[i]) * cuCimag(result[i]);
        EXPECT_NEAR(orig_prob, result_prob, 1e-10);
    }
}

// ==================== 批处理测试 ====================

TEST_F(SFSingleModeGatesTest, BatchProcessing_Pgate) {
    // 测试批处理多个状态
    int state_id2 = pool->allocate_state();
    
    set_fock_state(0);
    
    std::vector<cuDoubleComplex> state2(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    state2[1] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id2, state2);

    int host_targets[] = {state_id, state_id2};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 2);

    double s = M_PI;
    apply_pgate(pool, d_targets, 2, s);

    FREE_DEVICE_TARGETS(d_targets);

    // 验证第一个状态
    std::vector<cuDoubleComplex> result1;
    pool->download_state(state_id, result1);
    double phase0 = s * 0.5 / 2.0;
    EXPECT_NEAR(cuCreal(result1[0]), cos(phase0), 1e-10);
    EXPECT_NEAR(cuCimag(result1[0]), sin(phase0), 1e-10);

    // 验证第二个状态
    std::vector<cuDoubleComplex> result2;
    pool->download_state(state_id2, result2);
    double phase1 = s * 1.5 / 2.0;
    EXPECT_NEAR(cuCreal(result2[1]), cos(phase1), 1e-10);
    EXPECT_NEAR(cuCimag(result2[1]), sin(phase1), 1e-10);

    pool->free_state(state_id2);
}

TEST_F(SFSingleModeGatesTest, BatchProcessing_Fouriergate) {
    // 测试批处理 Fouriergate
    int state_id2 = pool->allocate_state();
    
    set_fock_state(0);
    
    std::vector<cuDoubleComplex> state2(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    state2[2] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id2, state2);

    int host_targets[] = {state_id, state_id2};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 2);

    apply_fouriergate(pool, d_targets, 2);

    FREE_DEVICE_TARGETS(d_targets);

    // 验证两个状态都被正确处理
    std::vector<cuDoubleComplex> result1;
    pool->download_state(state_id, result1);
    expect_normalized(result1);

    std::vector<cuDoubleComplex> result2;
    pool->download_state(state_id2, result2);
    expect_normalized(result2);

    pool->free_state(state_id2);
}
