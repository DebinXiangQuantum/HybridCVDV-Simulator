#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "reference_gates.h"

/**
 * 简化的验证程序
 * 验证参考实现的正确性
 */
int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   HybridCVDV-Simulator 验证程序" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "验证传统矩阵向量乘法实现的正确性" << std::endl << std::endl;

    const int dim = 16;  // 更大的截断维度

    // 创建多个测试状态
    std::vector<Reference::Vector> test_states;

    // 1. 真空态 |0⟩
    Reference::Vector vacuum_state(dim, Reference::Complex(0.0, 0.0));
    vacuum_state[0] = Reference::Complex(1.0, 0.0);
    test_states.push_back(vacuum_state);

    // 2. Fock态 |1⟩
    Reference::Vector fock1_state(dim, Reference::Complex(0.0, 0.0));
    fock1_state[1] = Reference::Complex(1.0, 0.0);
    test_states.push_back(fock1_state);

    // 3. Fock态 |2⟩
    Reference::Vector fock2(dim, Reference::Complex(0.0, 0.0));
    fock2[2] = Reference::Complex(1.0, 0.0);
    test_states.push_back(fock2);

    // 4. 相干态 |α⟩ (α = 0.5)
    Reference::Vector coherent_state(dim, Reference::Complex(0.0, 0.0));
    double alpha_coherent = 0.5;
    double norm_factor = std::exp(-alpha_coherent * alpha_coherent / 2.0);
    for (int n = 0; n < dim; ++n) {
        double coeff = norm_factor * std::pow(alpha_coherent, n) / std::sqrt(std::tgamma(n + 1));
        coherent_state[n] = Reference::Complex(coeff, 0.0);
    }
    test_states.push_back(coherent_state);

    // 5. 叠加态 (|0⟩ + |1⟩)/√2
    Reference::Vector superposition_state(dim, Reference::Complex(0.0, 0.0));
    double sqrt_half = 1.0 / std::sqrt(2.0);
    superposition_state[0] = Reference::Complex(sqrt_half, 0.0);
    superposition_state[1] = Reference::Complex(sqrt_half, 0.0);
    test_states.push_back(superposition_state);

    std::vector<std::string> state_names = {"真空态 |0⟩", "Fock态 |1⟩", "Fock态 |2⟩",
                                           "相干态 |α=0.5⟩", "叠加态 (|0⟩+|1⟩)/√2"};

    std::cout << "测试状态集合 (截断维度 = " << dim << "):" << std::endl;
    for (size_t i = 0; i < test_states.size(); ++i) {
        std::cout << "  " << (i+1) << ". " << state_names[i] << std::endl;
        const auto& state = test_states[i];
        for (int j = 0; j < dim; ++j) {
            if (std::abs(state[j]) > 1e-10) {
                std::cout << "     [" << j << "]: " << state[j] << std::endl;
            }
        }
    }
    std::cout << std::endl;

    // ===== 测试对角门 =====
    std::cout << "1. 测试对角门 (Diagonal Gates)" << std::endl;

    for (size_t state_idx = 0; state_idx < test_states.size(); ++state_idx) {
        const auto& state = test_states[state_idx];
        std::cout << "   测试状态: " << state_names[state_idx] << std::endl;

        // 相位旋转门
        auto phase_result = Reference::DiagonalGates::apply_phase_rotation(state, M_PI / 4.0);
        std::cout << "     R(π/4): 最大分量 = " << *std::max_element(phase_result.begin(), phase_result.end(),
            [](const Reference::Complex& a, const Reference::Complex& b) {
                return std::abs(a) < std::abs(b);
            }) << std::endl;

        // Kerr门
        auto kerr_result = Reference::DiagonalGates::apply_kerr_gate(state, 0.5);
        double kerr_norm = Reference::vector_norm(kerr_result);
        std::cout << "     K(0.5): 归一化误差 = " << std::abs(kerr_norm - 1.0) << std::endl;
    }
    std::cout << std::endl;

    // ===== 测试梯算符门 =====
    std::cout << "2. 测试梯算符门 (Ladder Gates)" << std::endl;

    for (size_t state_idx = 0; state_idx < test_states.size(); ++state_idx) {
        const auto& state = test_states[state_idx];
        std::cout << "   测试状态: " << state_names[state_idx] << std::endl;

        // 创建算符
        auto creation_result = Reference::LadderGates::apply_creation_operator(state);
        double creation_max = 0.0;
        int creation_idx = -1;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(creation_result[i]) > creation_max) {
                creation_max = std::abs(creation_result[i]);
                creation_idx = i;
            }
        }
        std::cout << "     a†: 最大分量在 n=" << creation_idx << ", 幅值=" << creation_max << std::endl;

        // 湮灭算符
        auto annihilation_result = Reference::LadderGates::apply_annihilation_operator(state);
        double annihilation_max = 0.0;
        int annihilation_idx = -1;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(annihilation_result[i]) > annihilation_max) {
                annihilation_max = std::abs(annihilation_result[i]);
                annihilation_idx = i;
            }
        }
        std::cout << "     a: 最大分量在 n=" << annihilation_idx << ", 幅值=" << annihilation_max << std::endl;
    }
    std::cout << std::endl;

    // ===== 测试位移门 =====
    std::cout << "3. 测试位移门 (Displacement Gate)" << std::endl;

    std::vector<Reference::Complex> alphas = {
        Reference::Complex(0.0, 0.0),  // 恒等算符测试
        Reference::Complex(0.1, 0.0),  // 小位移
        Reference::Complex(0.0, 0.1),  // 纯虚位移
        Reference::Complex(0.1, 0.1)   // 复位移
    };

    for (size_t state_idx = 0; state_idx < test_states.size(); ++state_idx) {
        const auto& state = test_states[state_idx];
        std::cout << "   测试状态: " << state_names[state_idx] << std::endl;

        for (const auto& alpha : alphas) {
            auto displacement_result = Reference::SingleModeGates::apply_displacement_gate(state, alpha);

            // 计算归一化
            double norm = Reference::vector_norm(displacement_result);
            double normalization_error = std::abs(norm - 1.0);

            // 计算与理论值的差异（对于真空态）
            double theoretical_error = 0.0;
            if (state_idx == 0) {  // 真空态
                double alpha_norm_sq = std::norm(alpha);
                double exp_factor = std::exp(-alpha_norm_sq / 2.0);
                Reference::Complex theoretical(exp_factor, 0.0);
                theoretical_error = std::abs(displacement_result[0] - theoretical);
            }

            std::cout << "     D(" << alpha << "): 归一化误差=" << normalization_error;
            if (state_idx == 0) {
                std::cout << ", 理论误差=" << theoretical_error;
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    // ===== 验证对易关系 =====
    std::cout << std::endl << "4. 验证对易关系 [a, a†] = 1" << std::endl;

    // 创建 |2⟩ 状态
    Reference::Vector fock2_state(dim, Reference::Complex(0.0, 0.0));
    fock2_state[2] = Reference::Complex(1.0, 0.0);

    // 计算 a†a |2⟩ = 2|2⟩
    auto a_dagger_a = Reference::LadderGates::apply_annihilation_operator(fock2_state);
    a_dagger_a = Reference::LadderGates::apply_creation_operator(a_dagger_a);

    // 计算 aa† |2⟩ = 3|2⟩
    auto a_a_dagger = Reference::LadderGates::apply_creation_operator(fock2_state);
    a_a_dagger = Reference::LadderGates::apply_annihilation_operator(a_a_dagger);

    std::cout << "   a†a |2⟩ = " << a_dagger_a[2] << " (期望: 2.0)" << std::endl;
    std::cout << "   aa† |2⟩ = " << a_a_dagger[2] << " (期望: 3.0)" << std::endl;
    std::cout << "   [a, a†] = " << (a_a_dagger[2] - a_dagger_a[2]) << " (期望: 1.0)" << std::endl;

    // ===== 误差分析 =====
    std::cout << std::endl << "5. 误差分析" << std::endl;

    // 计算两个向量之间的误差
    auto error_metrics = Reference::compute_error_metrics(a_a_dagger, a_dagger_a);
    std::cout << "   对易关系验证误差:" << std::endl;
    std::cout << "   L2误差: " << error_metrics.l2_error << std::endl;
    std::cout << "   最大误差: " << error_metrics.max_error << std::endl;
    std::cout << "   相对误差: " << error_metrics.relative_error << std::endl;
    std::cout << "   保真度偏差: " << error_metrics.fidelity_deviation << std::endl;

    // ===== 测试混合控制门 =====
    std::cout << "4. 测试混合控制门 (Hybrid Control Gates)" << std::endl;

    Reference::Complex alpha_cd(0.1, 0.05);
    Reference::Complex xi_cs(0.1, 0.0);

    for (int control_state : {0, 1}) {
        std::cout << "   控制状态: |" << control_state << "⟩" << std::endl;

        for (size_t state_idx = 0; state_idx < test_states.size(); ++state_idx) {
            const auto& state = test_states[state_idx];
            std::cout << "     测试状态: " << state_names[state_idx] << std::endl;

            // 受控位移门
            // Q=0: D(alpha), Q=1: D(-alpha)
            Reference::Vector cd_result;
            if (control_state == 0) {
                cd_result = Reference::SingleModeGates::apply_displacement_gate(state, alpha_cd);
            } else {
                cd_result = Reference::SingleModeGates::apply_displacement_gate(state, -alpha_cd);
            }
            double cd_norm = Reference::vector_norm(cd_result);
            double cd_error = std::abs(cd_norm - 1.0);

            // 受控挤压门
            // Q=0: S(xi), Q=1: S(-xi)
            Reference::Vector cs_result;
            if (control_state == 0) {
                cs_result = Reference::SingleModeGates::apply_squeezing_gate(state, xi_cs);
            } else {
                cs_result = Reference::SingleModeGates::apply_squeezing_gate(state, -xi_cs);
            }
            double cs_norm = Reference::vector_norm(cs_result);
            double cs_error = std::abs(cs_norm - 1.0);

            // 计算状态变化
            double cd_fidelity = Reference::fidelity(state, cd_result);
            double cs_fidelity = Reference::fidelity(state, cs_result);

            std::cout << "       CD(α): 归一化误差=" << cd_error << ", 保真度=" << cd_fidelity << std::endl;
            std::cout << "       CS(ξ): 归一化误差=" << cs_error << ", 保真度=" << cs_fidelity << std::endl;

            // 验证控制逻辑：CD/CS在Hybrid定义下总是改变状态 (除alpha=0)
            // 原始测试期望Control=0时不变，这与Hybrid门定义(sigma_z)不符，故移除该检查

        }
    }
    std::cout << std::endl;

    // ===== 验证对易关系 =====
    std::cout << "5. 扩展对易关系验证" << std::endl;

    // 测试多个状态的对易关系
    for (size_t state_idx = 0; state_idx < test_states.size(); ++state_idx) {
        const auto& state = test_states[state_idx];
        std::cout << "   测试状态: " << state_names[state_idx] << std::endl;

        // 计算 a†a |ψ⟩ 和 aa† |ψ⟩
        auto a_dagger_a = Reference::LadderGates::apply_annihilation_operator(state);
        a_dagger_a = Reference::LadderGates::apply_creation_operator(a_dagger_a);

        auto a_a_dagger = Reference::LadderGates::apply_creation_operator(state);
        a_a_dagger = Reference::LadderGates::apply_annihilation_operator(a_a_dagger);

        // 计算 [a, a†] = aa† - a†a
        Reference::Vector commutator_vec(dim, Reference::Complex(0.0, 0.0));
        for (int i = 0; i < dim; ++i) {
            commutator_vec[i] = a_a_dagger[i] - a_dagger_a[i];
        }

        // 对于Fock态 |n⟩，有 aa†|n⟩ = (n+1)|n⟩，a†a|n⟩ = n|n⟩，所以 [a,a†]|n⟩ = |n⟩
        double commutator_norm = Reference::vector_norm(commutator_vec);
        double expected_norm = 1.0;  // 对于归一化状态，期望值为1

        std::cout << "     [a,a†] 范数: " << commutator_norm << " (期望: " << expected_norm << ")" << std::endl;

        auto error_metrics = Reference::compute_error_metrics(a_a_dagger, a_dagger_a);
        std::cout << "     对易关系误差: L2=" << error_metrics.l2_error
                  << ", 保真度偏差=" << error_metrics.fidelity_deviation << std::endl;
    }
    std::cout << std::endl;

    // ===== 性能测试 =====
    std::cout << "6. 性能测试" << std::endl;

    const int num_iterations = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        auto temp = Reference::DiagonalGates::apply_phase_rotation(vacuum_state, M_PI / 4.0);
        temp = Reference::LadderGates::apply_creation_operator(temp);
        temp = Reference::SingleModeGates::apply_displacement_gate(temp, alpha_coherent);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "   " << num_iterations << " 次迭代总时间: " << total_time << " 秒" << std::endl;
    std::cout << "   平均每次迭代时间: " << (total_time / num_iterations) * 1000 << " ms" << std::endl;

    std::cout << std::endl << "=========================================" << std::endl;
    std::cout << "   验证程序运行完成！" << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}
