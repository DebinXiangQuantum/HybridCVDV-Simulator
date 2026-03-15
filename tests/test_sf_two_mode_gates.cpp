#include <gtest/gtest.h>
#include "cv_state_pool.h"
#include "two_mode_gates.h"
#include "reference_gates.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

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
 * Strawberry Fields 双模门单元测试
 */
class SFTwoModeGatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        cutoff_a = 4;
        cutoff_b = 4;
        d_trunc = cutoff_a * cutoff_b;  // 双模系统维度
        max_states = 4;
        pool = new CVStatePool(d_trunc, max_states, 1);

        // 初始化测试状态
        state_id = pool->allocate_state();
    }

    void TearDown() override {
        delete pool;
    }

    int cutoff_a;
    int cutoff_b;
    int d_trunc;
    int max_states;
    CVStatePool* pool;
    int state_id;

    // 辅助函数：设置双模 Fock 态 |m,n⟩
    void set_two_mode_fock_state(int m, int n) {
        std::vector<cuDoubleComplex> state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
        if (m < cutoff_a && n < cutoff_b) {
            int idx = m * cutoff_b + n;
            state[idx] = make_cuDoubleComplex(1.0, 0.0);
        }
        pool->upload_state(state_id, state);
    }

    // 辅助函数：设置双模叠加态
    void set_two_mode_superposition() {
        std::vector<cuDoubleComplex> state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
        double norm = 1.0 / sqrt(2.0);
        state[0] = make_cuDoubleComplex(norm, 0.0);  // |0,0⟩
        state[cutoff_b + 1] = make_cuDoubleComplex(norm, 0.0);  // |1,1⟩
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

    // 辅助函数：获取双模 Fock 态的概率
    double get_fock_probability(const std::vector<cuDoubleComplex>& state, int m, int n) {
        if (m >= cutoff_a || n >= cutoff_b) return 0.0;
        int idx = m * cutoff_b + n;
        double real = cuCreal(state[idx]);
        double imag = cuCimag(state[idx]);
        return real * real + imag * imag;
    }

    Reference::Vector to_reference(const std::vector<cuDoubleComplex>& state) {
        Reference::Vector result(state.size(), Reference::Complex(0.0, 0.0));
        for (size_t i = 0; i < state.size(); ++i) {
            result[i] = Reference::Complex(cuCreal(state[i]), cuCimag(state[i]));
        }
        return result;
    }

    double compute_error(const std::vector<cuDoubleComplex>& gpu_state,
                         const Reference::Vector& reference_state) {
        double error_sq = 0.0;
        for (size_t i = 0; i < gpu_state.size(); ++i) {
            const Reference::Complex gpu(cuCreal(gpu_state[i]), cuCimag(gpu_state[i]));
            error_sq += std::norm(gpu - reference_state[i]);
        }
        return std::sqrt(error_sq);
    }
};

// ==================== MZgate 测试 ====================

TEST_F(SFTwoModeGatesTest, MZgate_VacuumState) {
    // 测试 MZ 门作用在双真空态上
    // MZ(φ_in, φ_ex)|0,0⟩ = |0,0⟩
    set_two_mode_fock_state(0, 0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_mzgate(pool, d_targets, 1, 0.0, 0.0, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |0,0⟩ 应该保持不变
    EXPECT_NEAR(cuCreal(result[0]), 1.0, 1e-6);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-6);

    expect_normalized(result);
}

TEST_F(SFTwoModeGatesTest, MZgate_SinglePhoton) {
    // 测试 MZ 门作用在单光子态上
    // MZ 门应该在两个模式之间分配光子
    set_two_mode_fock_state(1, 0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_mzgate(pool, d_targets, 1, 0.0, M_PI, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 验证归一化
    expect_normalized(result);

    // 光子应该在两个模式之间分布
    double prob_10 = get_fock_probability(result, 1, 0);
    double prob_01 = get_fock_probability(result, 0, 1);
    
    // 总概率应该接近 1
    EXPECT_NEAR(prob_10 + prob_01, 1.0, 1e-5);
}

TEST_F(SFTwoModeGatesTest, MZgate_Superposition) {
    // 测试 MZ 门作用在叠加态上
    set_two_mode_superposition();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_mzgate(pool, d_targets, 1, M_PI / 4.0, M_PI / 4.0, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 验证归一化
    expect_normalized(result);
}

// ==================== CZgate 测试 ====================

TEST_F(SFTwoModeGatesTest, CZgate_VacuumState) {
    // 测试 CZ 门作用在双真空态上
    // CZ(s)|0,0⟩ = |0,0⟩
    set_two_mode_fock_state(0, 0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_czgate(pool, d_targets, 1, 1.0, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |0,0⟩ 应该保持不变
    EXPECT_NEAR(cuCreal(result[0]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);

    expect_normalized(result);
}

TEST_F(SFTwoModeGatesTest, CZgate_FockState) {
    // 测试 CZ 门作用在 Fock 态上
    // CZ(s)|m,n⟩ = exp(i s m n)|m,n⟩
    set_two_mode_fock_state(1, 1);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    double s = M_PI;
    apply_czgate(pool, d_targets, 1, s, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // CZ(π)|1,1⟩ = exp(i π · 1 · 1)|1,1⟩ = -|1,1⟩
    int idx = 1 * cutoff_b + 1;
    EXPECT_NEAR(cuCreal(result[idx]), -1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[idx]), 0.0, 1e-10);

    expect_normalized(result);
}

TEST_F(SFTwoModeGatesTest, CZgate_Identity) {
    // 测试 CZ(0) 应该是恒等操作
    set_two_mode_superposition();

    std::vector<cuDoubleComplex> original;
    pool->download_state(state_id, original);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_czgate(pool, d_targets, 1, 0.0, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 状态应该保持不变
    for (int i = 0; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), cuCreal(original[i]), 1e-10);
        EXPECT_NEAR(cuCimag(result[i]), cuCimag(original[i]), 1e-10);
    }
}

TEST_F(SFTwoModeGatesTest, CZgate_HigherFockState) {
    // 测试 CZ 门作用在更高的 Fock 态上
    // CZ(s)|2,2⟩ = exp(i s · 4)|2,2⟩
    set_two_mode_fock_state(2, 2);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    double s = M_PI / 4.0;
    apply_czgate(pool, d_targets, 1, s, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // CZ(π/4)|2,2⟩ = exp(i π)|2,2⟩ = -|2,2⟩
    int idx = 2 * cutoff_b + 2;
    double phase = s * 4.0;  // s * m * n = π/4 * 2 * 2 = π
    EXPECT_NEAR(cuCreal(result[idx]), cos(phase), 1e-10);
    EXPECT_NEAR(cuCimag(result[idx]), sin(phase), 1e-10);

    expect_normalized(result);
}

// ==================== CKgate 测试 ====================

TEST_F(SFTwoModeGatesTest, CKgate_VacuumState) {
    // 测试 CK 门作用在双真空态上
    // CK(κ)|0,0⟩ = |0,0⟩
    set_two_mode_fock_state(0, 0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_ckgate(pool, d_targets, 1, 1.0, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |0,0⟩ 应该保持不变
    EXPECT_NEAR(cuCreal(result[0]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);

    expect_normalized(result);
}

TEST_F(SFTwoModeGatesTest, CKgate_FockState) {
    // 测试 CK 门作用在 Fock 态上
    // CK(κ)|m,n⟩ = exp(i κ m n²)|m,n⟩
    set_two_mode_fock_state(1, 1);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    double kappa = M_PI;
    apply_ckgate(pool, d_targets, 1, kappa, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // CK(π)|1,1⟩ = exp(i π · 1 · 1)|1,1⟩ = -|1,1⟩
    int idx = 1 * cutoff_b + 1;
    double phase = kappa * 1.0 * 1.0;  // κ * m * n = π
    EXPECT_NEAR(cuCreal(result[idx]), cos(phase), 1e-10);
    EXPECT_NEAR(cuCimag(result[idx]), sin(phase), 1e-10);

    expect_normalized(result);
}

TEST_F(SFTwoModeGatesTest, CKgate_Identity) {
    // 测试 CK(0) 应该是恒等操作
    set_two_mode_superposition();

    std::vector<cuDoubleComplex> original;
    pool->download_state(state_id, original);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_ckgate(pool, d_targets, 1, 0.0, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 状态应该保持不变
    for (int i = 0; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), cuCreal(original[i]), 1e-10);
        EXPECT_NEAR(cuCimag(result[i]), cuCimag(original[i]), 1e-10);
    }
}

TEST_F(SFTwoModeGatesTest, CKgate_AsymmetricFockState) {
    // 测试 CK 门作用在非对称 Fock 态上
    // CK(κ)|2,1⟩ = exp(i κ · 2 · 1)|2,1⟩
    set_two_mode_fock_state(2, 1);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    double kappa = M_PI / 2.0;
    apply_ckgate(pool, d_targets, 1, kappa, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // CK(π/2)|2,1⟩ = exp(i π)|2,1⟩ = -|2,1⟩
    int idx = 2 * cutoff_b + 1;
    double phase = kappa * 2.0 * 1.0;  // π/2 * 2 * 1 = π
    EXPECT_NEAR(cuCreal(result[idx]), cos(phase), 1e-10);
    EXPECT_NEAR(cuCimag(result[idx]), sin(phase), 1e-10);

    expect_normalized(result);
}

// ==================== 批处理测试 ====================

TEST_F(SFTwoModeGatesTest, BatchProcessing_CZgate) {
    // 测试批处理多个状态
    int state_id2 = pool->allocate_state();
    
    set_two_mode_fock_state(0, 0);
    
    std::vector<cuDoubleComplex> state2(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    state2[cutoff_b + 1] = make_cuDoubleComplex(1.0, 0.0);  // |1,1⟩
    pool->upload_state(state_id2, state2);

    int host_targets[] = {state_id, state_id2};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 2);

    double s = M_PI;
    apply_czgate(pool, d_targets, 2, s, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    // 验证第一个状态: CZ(π)|0,0⟩ = |0,0⟩
    std::vector<cuDoubleComplex> result1;
    pool->download_state(state_id, result1);
    EXPECT_NEAR(cuCreal(result1[0]), 1.0, 1e-10);

    // 验证第二个状态: CZ(π)|1,1⟩ = -|1,1⟩
    std::vector<cuDoubleComplex> result2;
    pool->download_state(state_id2, result2);
    int idx = cutoff_b + 1;
    EXPECT_NEAR(cuCreal(result2[idx]), -1.0, 1e-10);

    pool->free_state(state_id2);
}

TEST_F(SFTwoModeGatesTest, BatchProcessing_CKgate) {
    // 测试批处理 CKgate
    int state_id2 = pool->allocate_state();
    
    set_two_mode_fock_state(1, 0);
    
    std::vector<cuDoubleComplex> state2(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    state2[2 * cutoff_b + 2] = make_cuDoubleComplex(1.0, 0.0);  // |2,2⟩
    pool->upload_state(state_id2, state2);

    int host_targets[] = {state_id, state_id2};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 2);

    double kappa = M_PI / 4.0;
    apply_ckgate(pool, d_targets, 2, kappa, cutoff_a, cutoff_b);

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

// ==================== 组合门测试 ====================

TEST_F(SFTwoModeGatesTest, CombinedGates_CZ_CK) {
    // 测试组合应用 CZ 和 CK 门
    set_two_mode_fock_state(1, 1);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    // 先应用 CZ
    apply_czgate(pool, d_targets, 1, M_PI / 2.0, cutoff_a, cutoff_b);
    
    // 再应用 CK
    apply_ckgate(pool, d_targets, 1, M_PI / 2.0, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 验证归一化
    expect_normalized(result);

    // 验证 |1,1⟩ 获得了正确的总相位
    int idx = cutoff_b + 1;
    double total_phase = M_PI / 2.0 + M_PI / 2.0;  // CZ + CK
    EXPECT_NEAR(cuCreal(result[idx]), cos(total_phase), 1e-10);
    EXPECT_NEAR(cuCimag(result[idx]), sin(total_phase), 1e-10);
}

TEST_F(SFTwoModeGatesTest, TMSGate_MatchesReference) {
    set_two_mode_fock_state(0, 0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    const double r = 0.2;
    const double theta = 0.1;
    apply_two_mode_squeezing_recursive(pool, d_targets, 1, r, theta);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);
    expect_normalized(result);

    Reference::Vector input(d_trunc, Reference::Complex(0.0, 0.0));
    input[0] = Reference::Complex(1.0, 0.0);
    const auto reference =
        Reference::TwoModeGatesExtended::apply_two_mode_squeezing(
            input, Reference::Complex(r * std::cos(theta), r * std::sin(theta)));

    EXPECT_LT(compute_error(result, reference), 1e-8);
    EXPECT_GT(get_fock_probability(result, 1, 1), 1e-4);
    EXPECT_NEAR(get_fock_probability(result, 1, 0), 0.0, 1e-8);
}

TEST_F(SFTwoModeGatesTest, SUMGate_MatchesReference) {
    set_two_mode_fock_state(1, 0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    const double scale = 0.15;
    apply_sum_gate(pool, d_targets, 1, scale, cutoff_a, cutoff_b);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);
    expect_normalized(result, 1e-5);

    Reference::Vector input(d_trunc, Reference::Complex(0.0, 0.0));
    input[cutoff_b] = Reference::Complex(1.0, 0.0);
    const auto reference = Reference::TwoModeGatesExtended::apply_sum_gate(input, scale, 0.0);

    EXPECT_LT(compute_error(result, reference), 1e-8);
    EXPECT_GT(get_fock_probability(result, 0, 1), 1e-4);
}
