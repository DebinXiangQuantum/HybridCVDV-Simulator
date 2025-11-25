#include <gtest/gtest.h>
#include "cv_state_pool.h"
#include "reference_gates.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// 声明外部函数
extern void apply_phase_rotation(CVStatePool& pool, const int* targets, int batch_size, double theta);
extern void apply_kerr_gate(CVStatePool& pool, const int* targets, int batch_size, double chi);
extern void apply_conditional_parity(CVStatePool& pool, const int* targets, int batch_size, double parity);

/**
 * 对角门操作单元测试
 */
class DiagonalGatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        d_trunc = 8;
        max_states = 4;
        pool = new CVStatePool(d_trunc, max_states);

        // 初始化测试状态
        state_id = pool->allocate_state();
        std::vector<cuDoubleComplex> initial_state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
        initial_state[0] = make_cuDoubleComplex(1.0, 0.0);  // |0⟩状态
        pool->upload_state(state_id, initial_state);
    }

    void TearDown() override {
        delete pool;
    }

    int d_trunc;
    int max_states;
    CVStatePool* pool;
    int state_id;

    /**
     * 辅助函数：计算状态归一化因子
     */
    double calculate_norm(const std::vector<cuDoubleComplex>& state) {
        double norm_sq = 0.0;
        for (const auto& amp : state) {
            double real = cuCreal(amp);
            double imag = cuCimag(amp);
            norm_sq += real * real + imag * imag;
        }
        return std::sqrt(norm_sq);
    }

    /**
     * 辅助函数：验证状态是否归一化
     */
    void expect_normalized(const std::vector<cuDoubleComplex>& state, double tolerance = 1e-10) {
        double norm = calculate_norm(state);
        EXPECT_NEAR(norm, 1.0, tolerance);
    }

    /**
     * 辅助函数：将cuDoubleComplex向量转换为std::complex<double>向量
     */
    Reference::Vector to_reference_vector(const std::vector<cuDoubleComplex>& cuda_vec) {
        Reference::Vector ref_vec;
        for (const auto& val : cuda_vec) {
            ref_vec.emplace_back(cuCreal(val), cuCimag(val));
        }
        return ref_vec;
    }

    /**
     * 辅助函数：对比GPU实现与参考实现的误差
     */
    void compare_with_reference(const std::vector<cuDoubleComplex>& gpu_result,
                               const Reference::Vector& ref_result,
                               double max_allowed_error = 1e-10) {
        auto gpu_as_ref = to_reference_vector(gpu_result);
        auto error_metrics = Reference::compute_error_metrics(ref_result, gpu_as_ref);

        std::cout << "误差指标:" << std::endl;
        std::cout << "  L2误差: " << error_metrics.l2_error << std::endl;
        std::cout << "  最大误差: " << error_metrics.max_error << std::endl;
        std::cout << "  相对误差: " << error_metrics.relative_error << std::endl;
        std::cout << "  保真度偏差: " << error_metrics.fidelity_deviation << std::endl;

        EXPECT_LT(error_metrics.l2_error, max_allowed_error);
        EXPECT_LT(error_metrics.max_error, max_allowed_error);
        EXPECT_LT(error_metrics.relative_error, max_allowed_error);
        EXPECT_LT(error_metrics.fidelity_deviation, max_allowed_error);
    }
};

TEST_F(DiagonalGatesTest, PhaseRotationIdentity) {
    // 测试恒等旋转 (θ = 0)
    std::vector<cuDoubleComplex> original_state;
    pool->download_state(state_id, original_state);

    // 参考实现
    auto ref_input = to_reference_vector(original_state);
    auto ref_result = Reference::DiagonalGates::apply_phase_rotation(ref_input, 0.0);

    // GPU实现
    int target_ids[] = {state_id};
    apply_phase_rotation(*pool, target_ids, 1, 0.0);

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    compare_with_reference(gpu_result, ref_result);
    expect_normalized(gpu_result);
}

TEST_F(DiagonalGatesTest, PhaseRotationPi) {
    // 测试π旋转: R(π) |0⟩ = -|0⟩
    std::vector<cuDoubleComplex> input_state;
    pool->download_state(state_id, input_state);

    // 参考实现
    auto ref_input = to_reference_vector(input_state);
    auto ref_result = Reference::DiagonalGates::apply_phase_rotation(ref_input, M_PI);

    // GPU实现
    int target_ids[] = {state_id};
    apply_phase_rotation(*pool, target_ids, 1, M_PI);

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    compare_with_reference(gpu_result, ref_result);
    expect_normalized(gpu_result);
}

TEST_F(DiagonalGatesTest, PhaseRotationPiHalf) {
    // 测试π/2旋转: R(π/2) |0⟩ = i|0⟩
    std::vector<cuDoubleComplex> input_state;
    pool->download_state(state_id, input_state);

    // 参考实现
    auto ref_input = to_reference_vector(input_state);
    auto ref_result = Reference::DiagonalGates::apply_phase_rotation(ref_input, M_PI / 2.0);

    // GPU实现
    int target_ids[] = {state_id};
    apply_phase_rotation(*pool, target_ids, 1, M_PI / 2.0);

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    compare_with_reference(gpu_result, ref_result);
    expect_normalized(gpu_result);
}

TEST_F(DiagonalGatesTest, KerrGateVacuum) {
    // 测试Kerr门作用在真空态上: K(χ) |0⟩ = e^0 |0⟩ = |0⟩
    std::vector<cuDoubleComplex> input_state;
    pool->download_state(state_id, input_state);

    // 参考实现
    auto ref_input = to_reference_vector(input_state);
    auto ref_result = Reference::DiagonalGates::apply_kerr_gate(ref_input, 1.0);

    // GPU实现
    int target_ids[] = {state_id};
    apply_kerr_gate(*pool, target_ids, 1, 1.0);

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    compare_with_reference(gpu_result, ref_result);
    expect_normalized(gpu_result);
}

TEST_F(DiagonalGatesTest, KerrGateFockState) {
    // 创建Fock态 |2⟩
    std::vector<cuDoubleComplex> fock_state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    fock_state[2] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id, fock_state);

    // 应用Kerr门: K(χ) |2⟩ = e^{-iχ·2²} |2⟩ = e^{-iχ·4} |2⟩
    double chi = 0.5;
    int target_ids[] = {state_id};
    apply_kerr_gate(*pool, target_ids, 1, chi);

    std::vector<cuDoubleComplex> result_state;
    pool->download_state(state_id, result_state);

    // |2⟩分量应该变成 e^{-iχ·4}
    double expected_phase = -chi * 4.0;
    double expected_real = cos(expected_phase);
    double expected_imag = -sin(expected_phase);  // 注意Kerr门公式中的负号

    EXPECT_NEAR(cuCreal(result_state[2]), expected_real, 1e-10);
    EXPECT_NEAR(cuCimag(result_state[2]), expected_imag, 1e-10);

    // 其他分量应该为0
    result_state[2] = make_cuDoubleComplex(0.0, 0.0);
    expect_normalized(result_state);
}

TEST_F(DiagonalGatesTest, ConditionalParityEven) {
    // 测试条件奇偶校验门 (偶数模式)
    int target_ids[] = {state_id};
    apply_conditional_parity(*pool, target_ids, 1, 0.0);  // 偶数

    std::vector<cuDoubleComplex> result_state;
    pool->download_state(state_id, result_state);

    // |0⟩是偶数模式，应该得到 +|0⟩
    EXPECT_NEAR(cuCreal(result_state[0]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result_state[0]), 0.0, 1e-10);

    expect_normalized(result_state);
}

TEST_F(DiagonalGatesTest, ConditionalParityOdd) {
    // 测试条件奇偶校验门 (奇数模式)
    int target_ids[] = {state_id};
    apply_conditional_parity(*pool, target_ids, 1, 1.0);  // 奇数

    std::vector<cuDoubleComplex> result_state;
    pool->download_state(state_id, result_state);

    // |0⟩是偶数模式，应该得到 +|0⟩ (不受奇数控制影响)
    EXPECT_NEAR(cuCreal(result_state[0]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result_state[0]), 0.0, 1e-10);

    expect_normalized(result_state);
}

TEST_F(DiagonalGatesTest, BatchProcessing) {
    // 测试批处理多个状态
    int state_id2 = pool->allocate_state();
    std::vector<cuDoubleComplex> state2(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    state2[1] = make_cuDoubleComplex(1.0, 0.0);  // |1⟩状态
    pool->upload_state(state_id2, state2);

    int target_ids[] = {state_id, state_id2};
    apply_phase_rotation(*pool, target_ids, 2, M_PI);

    // 验证第一个状态
    std::vector<cuDoubleComplex> result1;
    pool->download_state(state_id, result1);
    EXPECT_NEAR(cuCreal(result1[0]), -1.0, 1e-10);

    // 验证第二个状态: R(π) |1⟩ = e^{-iπ·1} |1⟩ = -|1⟩
    std::vector<cuDoubleComplex> result2;
    pool->download_state(state_id2, result2);
    EXPECT_NEAR(cuCreal(result2[1]), -1.0, 1e-10);

    pool->free_state(state_id2);
}

TEST_F(DiagonalGatesTest, ComplexSuperposition) {
    // 创建叠加态 |ψ⟩ = (|0⟩ + |1⟩)/√2
    std::vector<cuDoubleComplex> superposition(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    double norm_factor = 1.0 / std::sqrt(2.0);
    superposition[0] = make_cuDoubleComplex(norm_factor, 0.0);
    superposition[1] = make_cuDoubleComplex(norm_factor, 0.0);
    pool->upload_state(state_id, superposition);

    auto ref_input = to_reference_vector(superposition);

    // 参考实现
    auto ref_result = Reference::DiagonalGates::apply_phase_rotation(ref_input, M_PI / 4.0);

    // GPU实现
    int target_ids[] = {state_id};
    apply_phase_rotation(*pool, target_ids, 1, M_PI / 4.0);

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    compare_with_reference(gpu_result, ref_result);
    expect_normalized(gpu_result);
}
