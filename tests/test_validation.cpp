#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include "reference_gates.h"

/**
 * 验证测试
 * 验证参考实现的正确性
 */
class ValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 16;
        
        // 创建真空态 |0⟩
        vacuum_state = Reference::Vector(dim, Reference::Complex(0.0, 0.0));
        vacuum_state[0] = Reference::Complex(1.0, 0.0);
        
        // 创建Fock态 |1⟩
        fock1_state = Reference::Vector(dim, Reference::Complex(0.0, 0.0));
        fock1_state[1] = Reference::Complex(1.0, 0.0);
        
        // 创建Fock态 |2⟩
        fock2_state = Reference::Vector(dim, Reference::Complex(0.0, 0.0));
        fock2_state[2] = Reference::Complex(1.0, 0.0);
        
        // 创建相干态 |α=0.5⟩
        coherent_state = Reference::Vector(dim, Reference::Complex(0.0, 0.0));
        double alpha_coherent = 0.5;
        double norm_factor = std::exp(-alpha_coherent * alpha_coherent / 2.0);
        for (int n = 0; n < dim; ++n) {
            double coeff = norm_factor * std::pow(alpha_coherent, n) / std::sqrt(std::tgamma(n + 1));
            coherent_state[n] = Reference::Complex(coeff, 0.0);
        }
        
        // 创建叠加态 (|0⟩ + |1⟩)/√2
        superposition_state = Reference::Vector(dim, Reference::Complex(0.0, 0.0));
        double sqrt_half = 1.0 / std::sqrt(2.0);
        superposition_state[0] = Reference::Complex(sqrt_half, 0.0);
        superposition_state[1] = Reference::Complex(sqrt_half, 0.0);
        
        // 添加到测试状态集合
        test_states.push_back(vacuum_state);
        test_states.push_back(fock1_state);
        test_states.push_back(fock2_state);
        test_states.push_back(coherent_state);
        test_states.push_back(superposition_state);
    }
    
    int dim;
    Reference::Vector vacuum_state;
    Reference::Vector fock1_state;
    Reference::Vector fock2_state;
    Reference::Vector coherent_state;
    Reference::Vector superposition_state;
    std::vector<Reference::Vector> test_states;
};

// 测试对角门
TEST_F(ValidationTest, DiagonalGates) {
    for (const auto& state : test_states) {
        // 测试相位旋转门
        auto phase_result = Reference::DiagonalGates::apply_phase_rotation(state, M_PI / 4.0);
        double phase_norm = Reference::vector_norm(phase_result);
        EXPECT_NEAR(phase_norm, 1.0, 1e-10);
        
        // 测试Kerr门
        auto kerr_result = Reference::DiagonalGates::apply_kerr_gate(state, 0.5);
        double kerr_norm = Reference::vector_norm(kerr_result);
        EXPECT_NEAR(kerr_norm, 1.0, 1e-10);
    }
}

// 测试梯算符门
TEST_F(ValidationTest, LadderGates) {
    for (const auto& state : test_states) {
        // 测试创建算符
        auto creation_result = Reference::LadderGates::apply_creation_operator(state);
        // 注意：创建算符不是幺正算符，不保持归一化
        // 只验证结果向量不为空且维度正确
        EXPECT_EQ(creation_result.size(), state.size());
        
        // 测试湮灭算符
        auto annihilation_result = Reference::LadderGates::apply_annihilation_operator(state);
        // 注意：湮灭算符不是幺正算符，不保持归一化
        // 只验证结果向量不为空且维度正确
        EXPECT_EQ(annihilation_result.size(), state.size());
    }
}

// 测试位移门
TEST_F(ValidationTest, DisplacementGate) {
    std::vector<Reference::Complex> alphas = {
        Reference::Complex(0.0, 0.0),  // 恒等算符测试
        Reference::Complex(0.1, 0.0),  // 小位移
        Reference::Complex(0.0, 0.1),  // 纯虚位移
        Reference::Complex(0.1, 0.1)   // 复位移
    };
    
    for (const auto& state : test_states) {
        for (const auto& alpha : alphas) {
            auto displacement_result = Reference::SingleModeGates::apply_displacement_gate(state, alpha);
            double norm = Reference::vector_norm(displacement_result);
            EXPECT_NEAR(norm, 1.0, 2e-7);
        }
    }
}

// 验证对易关系 [a, a†] = 1
TEST_F(ValidationTest, CommutationRelation) {
    // 计算 a†a |2⟩ = 2|2⟩
    auto a_dagger_a = Reference::LadderGates::apply_annihilation_operator(fock2_state);
    a_dagger_a = Reference::LadderGates::apply_creation_operator(a_dagger_a);
    EXPECT_NEAR(a_dagger_a[2].real(), 2.0, 1e-10);
    EXPECT_NEAR(a_dagger_a[2].imag(), 0.0, 1e-10);
    
    // 计算 aa† |2⟩ = 3|2⟩
    auto a_a_dagger = Reference::LadderGates::apply_creation_operator(fock2_state);
    a_a_dagger = Reference::LadderGates::apply_annihilation_operator(a_a_dagger);
    EXPECT_NEAR(a_a_dagger[2].real(), 3.0, 1e-10);
    EXPECT_NEAR(a_a_dagger[2].imag(), 0.0, 1e-10);
    
    // 验证对易关系 [a, a†] = aa† - a†a = 1
    Reference::Complex commutator = a_a_dagger[2] - a_dagger_a[2];
    EXPECT_NEAR(commutator.real(), 1.0, 1e-10);
    EXPECT_NEAR(commutator.imag(), 0.0, 1e-10);
}

// 测试混合控制门
TEST_F(ValidationTest, HybridControlGates) {
    Reference::Complex alpha_cd(0.1, 0.05);
    Reference::Complex xi_cs(0.1, 0.0);
    
    // 测试控制状态为0时，状态不应改变
    for (const auto& state : test_states) {
        auto cd_result = Reference::HybridControlGates::apply_controlled_displacement(0, state, alpha_cd);
        double cd_fidelity = Reference::fidelity(state, cd_result);
        EXPECT_NEAR(cd_fidelity, 1.0, 1e-10);
        
        auto cs_result = Reference::HybridControlGates::apply_controlled_squeezing(0, state, xi_cs);
        double cs_fidelity = Reference::fidelity(state, cs_result);
        EXPECT_NEAR(cs_fidelity, 1.0, 1e-10);
    }
    
    // 测试控制状态为1时，状态应该改变
    for (const auto& state : test_states) {
        auto cd_result = Reference::HybridControlGates::apply_controlled_displacement(1, state, alpha_cd);
        double cd_norm = Reference::vector_norm(cd_result);
        EXPECT_NEAR(cd_norm, 1.0, 1e-7);
        
        auto cs_result = Reference::HybridControlGates::apply_controlled_squeezing(1, state, xi_cs);
        double cs_norm = Reference::vector_norm(cs_result);
        EXPECT_NEAR(cs_norm, 1.0, 1e-7);
    }
}
