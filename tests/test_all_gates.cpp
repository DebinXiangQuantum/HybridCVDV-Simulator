#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include "quantum_circuit.h"
#include "reference_gates.h"

// 测试工具函数
class GateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化测试环境
    }

    // 计算两个向量之间的误差
    double compute_error(const std::vector<std::complex<double>>& v1,
                        const std::vector<std::complex<double>>& v2) {
        if (v1.size() != v2.size()) return 1e9;
        double error = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            std::complex<double> diff = v1[i] - v2[i];
            error += std::norm(diff);
        }
        return std::sqrt(error);
    }

    // 归一化向量
    void normalize_vector(std::vector<std::complex<double>>& v) {
        double norm = 0.0;
        for (const auto& elem : v) {
            norm += std::norm(elem);
        }
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (auto& elem : v) {
                elem /= norm;
            }
        }
    }
};

// ===== 测试CPU端Qubit门 =====

TEST_F(GateTest, TestPauliX) {
    // 测试Pauli-X门
    Reference::Vector input = {{1.0, 0.0}, {0.0, 0.0}};  // |0⟩
    Reference::Vector expected = {{0.0, 0.0}, {1.0, 0.0}}; // |1⟩

    Reference::Vector result = Reference::QubitGates::apply_pauli_x(input);

    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "Pauli-X门误差过大";
}

TEST_F(GateTest, TestPauliY) {
    // 测试Pauli-Y门
    Reference::Vector input = {{1.0, 0.0}, {0.0, 0.0}};  // |0⟩
    Reference::Vector expected = {{0.0, 0.0}, {0.0, 1.0}}; // i|1⟩

    Reference::Vector result = Reference::QubitGates::apply_pauli_y(input);

    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "Pauli-Y门误差过大";
}

TEST_F(GateTest, TestPauliZ) {
    // 测试Pauli-Z门
    Reference::Vector input = {{0.0, 0.0}, {1.0, 0.0}};  // |1⟩
    Reference::Vector expected = {{0.0, 0.0}, {-1.0, 0.0}}; // -|1⟩

    Reference::Vector result = Reference::QubitGates::apply_pauli_z(input);

    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "Pauli-Z门误差过大";
}

TEST_F(GateTest, TestHadamard) {
    // 测试Hadamard门
    Reference::Vector input = {{1.0, 0.0}, {0.0, 0.0}};  // |0⟩
    Reference::Vector expected = {{1.0/std::sqrt(2.0), 0.0}, {1.0/std::sqrt(2.0), 0.0}}; // (|0⟩ + |1⟩)/√2

    Reference::Vector result = Reference::QubitGates::apply_hadamard(input);

    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "Hadamard门误差过大";
}

TEST_F(GateTest, TestRotationX) {
    // 测试Rx(π/2)门
    Reference::Vector input = {{1.0, 0.0}, {0.0, 0.0}};  // |0⟩
    double theta = M_PI / 2.0;
    double cos_half = std::cos(theta / 2.0);
    double sin_half = std::sin(theta / 2.0);
    Reference::Vector expected = {{cos_half, 0.0}, {0.0, -sin_half}}; // cos(θ/2)|0⟩ - i*sin(θ/2)|1⟩

    Reference::Vector result = Reference::QubitGates::apply_rotation_x(input, theta);

    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "Rx门误差过大";
}

TEST_F(GateTest, TestPhaseS) {
    // 测试S门
    Reference::Vector input = {{0.0, 0.0}, {1.0, 0.0}};  // |1⟩
    Reference::Vector expected = {{0.0, 0.0}, {0.0, 1.0}}; // i|1⟩

    Reference::Vector result = Reference::QubitGates::apply_phase_s(input);

    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "S门误差过大";
}

TEST_F(GateTest, TestPhaseT) {
    // 测试T门
    Reference::Vector input = {{0.0, 0.0}, {1.0, 0.0}};  // |1⟩
    Reference::Complex phase = std::exp(Reference::Complex(0.0, M_PI / 4.0));
    Reference::Vector expected = {{0.0, 0.0}, phase}; // e^(iπ/4)|1⟩

    Reference::Vector result = Reference::QubitGates::apply_phase_t(input);

    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "T门误差过大";
}

// ===== 测试GPU端Qumode门 =====

TEST_F(GateTest, TestPhaseRotation) {
    // 测试相位旋转门 R(θ)
    const int dim = 4;
    Reference::Vector input(dim, Reference::Complex(0.0, 0.0));
    input[0] = Reference::Complex(1.0, 0.0);  // |0⟩
    input[1] = Reference::Complex(0.0, 1.0);  // i|1⟩

    double theta = M_PI / 4.0;
    Reference::Vector result = Reference::DiagonalGates::apply_phase_rotation(input, theta);

    // 验证相位因子
    EXPECT_NEAR(std::arg(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(std::arg(result[1]), M_PI/4.0 + M_PI/2.0, 1e-10); // 原相位 + θ
    EXPECT_NEAR(std::arg(result[2]), 2.0 * M_PI/4.0, 1e-10);
}

TEST_F(GateTest, TestDisplacement) {
    // 测试位移门 D(α) - 简化测试
    const int dim = 4;
    Reference::Vector input(dim, Reference::Complex(0.0, 0.0));
    input[0] = Reference::Complex(1.0, 0.0);  // 真空态

    Reference::Complex alpha(0.1, 0.0);  // 小位移参数
    Reference::Vector result = Reference::SingleModeGates::apply_displacement_gate(input, alpha);

    // 验证结果非零且归一化
    double norm = 0.0;
    for (const auto& elem : result) {
        norm += std::norm(elem);
    }
    EXPECT_NEAR(norm, 1.0, 1e-6) << "位移门结果未归一化";
}

// ===== 测试混合门 =====

TEST_F(GateTest, TestControlledDisplacement) {
    // 测试受控位移门
    const int dim = 4;
    Reference::Vector target_state(dim, Reference::Complex(0.0, 0.0));
    target_state[0] = Reference::Complex(1.0, 0.0);  // 真空态

    Reference::Complex alpha(0.1, 0.0);

    // 控制位为0：不应用位移
    Reference::Vector result0 = Reference::HybridControlGates::apply_controlled_displacement(0, target_state, alpha);
    double error0 = compute_error(result0, target_state);
    EXPECT_LT(error0, 1e-10) << "控制位为0时CD门应保持不变";

    // 控制位为1：应用位移
    Reference::Vector result1 = Reference::HybridControlGates::apply_controlled_displacement(1, target_state, alpha);
    Reference::Vector expected1 = Reference::SingleModeGates::apply_displacement_gate(target_state, alpha);
    double error1 = compute_error(result1, expected1);
    EXPECT_LT(error1, 1e-6) << "控制位为1时CD门误差过大";
}

// ===== 集成测试 =====

TEST_F(GateTest, TestQuantumCircuitBasic) {
    // 基本量子电路测试
    try {
        QuantumCircuit circuit(2, 2, 4, 1024);  // 2 qubits, 2 qumodes, truncation=4

        // 添加一些门
        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::PauliX(1));
        circuit.add_gate(Gates::CNOT(0, 1));

        // 构建和执行
        circuit.build();
        circuit.execute();

        // 验证电路执行成功（没有抛出异常）
        SUCCEED() << "量子电路基本测试通过";

    } catch (const std::exception& e) {
        FAIL() << "量子电路测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestErrorMetrics) {
    // 测试误差计算函数
    Reference::Vector v1 = {{1.0, 0.0}, {0.0, 0.0}};
    Reference::Vector v2 = {{0.999, 0.001}, {0.001, 0.001}};

    Reference::ErrorMetrics metrics = Reference::compute_error_metrics(v1, v2);

    EXPECT_GT(metrics.l2_error, 0.0);
    EXPECT_LT(metrics.l2_error, 1.0);
    EXPECT_GT(metrics.fidelity_deviation, 0.0);
    EXPECT_LT(metrics.fidelity_deviation, 1.0);
}

// ===== 性能测试 =====

TEST_F(GateTest, DISABLED_TestPerformance) {
    // 性能测试（默认禁用）
    const int dim = 16;
    Reference::Vector state(dim, Reference::Complex(0.0, 0.0));
    state[0] = Reference::Complex(1.0, 0.0);

    // 测试相位旋转门的性能
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        Reference::Vector result = Reference::DiagonalGates::apply_phase_rotation(state, M_PI * i / 1000.0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "1000次相位旋转耗时: " << duration.count() << " ms" << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}