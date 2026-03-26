#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "quantum_circuit.h"
#include "reference_gates.h"
#include "cv_state_pool.h"

// 声明外部GPU函数
extern void apply_phase_rotation(CVStatePool* pool, const int* targets, int batch_size, double theta);
extern void apply_kerr_gate(CVStatePool* pool, const int* targets, int batch_size, double chi);
extern void apply_creation_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_annihilation_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_displacement_gate(CVStatePool* pool, const int* targets, int batch_size, cuDoubleComplex alpha, cudaStream_t stream = nullptr, bool synchronize = true);

namespace {

size_t integer_power(size_t base, int exponent) {
    size_t result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

size_t count_occurrences(const std::string& text, const std::string& needle) {
    if (needle.empty()) {
        return 0;
    }
    size_t count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

template <typename Transform>
Reference::Vector apply_single_mode_transform(const Reference::Vector& state,
                                              int cutoff,
                                              int num_qumodes,
                                              int target_qumode,
                                              Transform&& transform) {
    const size_t stride = integer_power(static_cast<size_t>(cutoff), num_qumodes - target_qumode - 1);
    const size_t prefix_count = integer_power(static_cast<size_t>(cutoff), target_qumode);
    Reference::Vector result(state.size(), Reference::Complex(0.0, 0.0));

    for (size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const size_t block_base = prefix * static_cast<size_t>(cutoff) * stride;
        for (size_t suffix = 0; suffix < stride; ++suffix) {
            Reference::Vector slice(static_cast<size_t>(cutoff), Reference::Complex(0.0, 0.0));
            for (int photon = 0; photon < cutoff; ++photon) {
                slice[static_cast<size_t>(photon)] =
                    state[block_base + static_cast<size_t>(photon) * stride + suffix];
            }

            const Reference::Vector transformed = transform(slice);
            for (int photon = 0; photon < cutoff; ++photon) {
                result[block_base + static_cast<size_t>(photon) * stride + suffix] =
                    transformed[static_cast<size_t>(photon)];
            }
        }
    }

    return result;
}

template <typename Transform>
Reference::Vector apply_two_mode_transform(const Reference::Vector& state,
                                           int cutoff,
                                           int num_qumodes,
                                           int first_qumode,
                                           int second_qumode,
                                           Transform&& transform) {
    std::vector<int> other_modes;
    std::vector<size_t> strides(static_cast<size_t>(num_qumodes), 1);
    for (int mode = num_qumodes - 2; mode >= 0; --mode) {
        strides[static_cast<size_t>(mode)] = strides[static_cast<size_t>(mode + 1)] * static_cast<size_t>(cutoff);
    }
    for (int mode = 0; mode < num_qumodes; ++mode) {
        if (mode != first_qumode && mode != second_qumode) {
            other_modes.push_back(mode);
        }
    }

    const size_t outer_count = integer_power(static_cast<size_t>(cutoff), num_qumodes - 2);
    Reference::Vector result(state.size(), Reference::Complex(0.0, 0.0));

    for (size_t outer_index = 0; outer_index < outer_count; ++outer_index) {
        size_t residual = outer_index;
        size_t base_index = 0;
        for (int idx = static_cast<int>(other_modes.size()) - 1; idx >= 0; --idx) {
            const int digit = static_cast<int>(residual % static_cast<size_t>(cutoff));
            residual /= static_cast<size_t>(cutoff);
            base_index += static_cast<size_t>(digit) *
                          strides[static_cast<size_t>(other_modes[static_cast<size_t>(idx)])];
        }

        Reference::Vector local_state(static_cast<size_t>(cutoff * cutoff), Reference::Complex(0.0, 0.0));
        for (int photon_a = 0; photon_a < cutoff; ++photon_a) {
            for (int photon_b = 0; photon_b < cutoff; ++photon_b) {
                const size_t linear_index =
                    base_index +
                    static_cast<size_t>(photon_a) * strides[static_cast<size_t>(first_qumode)] +
                    static_cast<size_t>(photon_b) * strides[static_cast<size_t>(second_qumode)];
                local_state[static_cast<size_t>(photon_a) * cutoff + static_cast<size_t>(photon_b)] =
                    state[linear_index];
            }
        }

        const Reference::Vector transformed = transform(local_state);
        for (int photon_a = 0; photon_a < cutoff; ++photon_a) {
            for (int photon_b = 0; photon_b < cutoff; ++photon_b) {
                const size_t linear_index =
                    base_index +
                    static_cast<size_t>(photon_a) * strides[static_cast<size_t>(first_qumode)] +
                    static_cast<size_t>(photon_b) * strides[static_cast<size_t>(second_qumode)];
                result[linear_index] =
                    transformed[static_cast<size_t>(photon_a) * cutoff + static_cast<size_t>(photon_b)];
            }
        }
    }

    return result;
}

}  // namespace

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

    // 将cuDoubleComplex向量转换为std::complex<double>向量
    std::vector<std::complex<double>> cuComplex_to_std(const std::vector<cuDoubleComplex>& v) {
        std::vector<std::complex<double>> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = std::complex<double>(cuCreal(v[i]), cuCimag(v[i]));
        }
        return result;
    }

    // 将std::complex<double>向量转换为cuDoubleComplex向量
    std::vector<cuDoubleComplex> std_to_cuComplex(const std::vector<std::complex<double>>& v) {
        std::vector<cuDoubleComplex> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = make_cuDoubleComplex(v[i].real(), v[i].imag());
        }
        return result;
    }
};

// GPU门测试类
class GPUGateTest : public ::testing::Test {
protected:
    void SetUp() override {
        d_trunc = 8;
        max_states = 4;
        pool = new CVStatePool(d_trunc, max_states, 1);  // 无内存限制

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

    // 辅助函数：计算状态归一化因子
    double calculate_norm(const std::vector<cuDoubleComplex>& state) {
        double norm_sq = 0.0;
        for (const auto& amp : state) {
            double real = cuCreal(amp);
            double imag = cuCimag(amp);
            norm_sq += real * real + imag * imag;
        }
        return std::sqrt(norm_sq);
    }

    // 比较GPU状态与参考实现的误差
    double compare_with_reference(const std::vector<cuDoubleComplex>& gpu_state,
                                  const std::vector<std::complex<double>>& ref_state) {
        if (gpu_state.size() != ref_state.size()) return 1e9;
        double error = 0.0;
        for (size_t i = 0; i < gpu_state.size(); ++i) {
            std::complex<double> gpu = std::complex<double>(cuCreal(gpu_state[i]), cuCimag(gpu_state[i]));
            std::complex<double> ref = ref_state[i];
            error += std::norm(gpu - ref);
        }
        return std::sqrt(error);
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
    // 测试相位旋转门 R(θ) - Reference实现
    const int dim = 4;
    Reference::Vector input(dim, Reference::Complex(0.0, 0.0));
    input[0] = Reference::Complex(1.0, 0.0);  // |0⟩
    input[1] = Reference::Complex(0.0, 1.0);  // i|1⟩
    input[2] = Reference::Complex(1.0, 0.0);  // |2⟩

    double theta = M_PI / 4.0;
    Reference::Vector result = Reference::DiagonalGates::apply_phase_rotation(input, theta);

    // 验证相位因子：R(θ) 应用相位因子 e^{-iθn}
    EXPECT_NEAR(std::arg(result[0]), 0.0, 1e-10);                           // e^{-iθ*0} * 1 = 1
    EXPECT_NEAR(std::arg(result[1]), M_PI/2.0 - M_PI/4.0, 1e-10);         // e^{-iθ*1} * i = i * e^{-iθ}，相位 = π/2 - θ
    EXPECT_NEAR(std::arg(result[2]), -2.0 * M_PI/4.0, 1e-10);             // e^{-iθ*2} * 1，相位 = -2θ
}

TEST_F(GPUGateTest, TestGPUPhaseRotation) {
    // 测试GPU相位旋转门 R(θ)
    double theta = M_PI / 4.0;

    // 创建初始状态
    std::vector<std::complex<double>> initial_state(d_trunc, std::complex<double>(0.0, 0.0));
    initial_state[0] = std::complex<double>(1.0, 0.0);
    initial_state[1] = std::complex<double>(0.0, 1.0);
    initial_state[2] = std::complex<double>(1.0, 0.0);

    // 上传初始状态到GPU
    std::vector<cuDoubleComplex> gpu_initial(d_trunc);
    for (size_t i = 0; i < d_trunc; ++i) {
        gpu_initial[i] = make_cuDoubleComplex(initial_state[i].real(), initial_state[i].imag());
    }
    pool->upload_state(state_id, gpu_initial);

    // 获取参考结果
    auto ref_result = Reference::DiagonalGates::apply_phase_rotation(initial_state, theta);

    // 应用GPU门
    int* d_target_ids = nullptr;
    cudaMalloc(&d_target_ids, sizeof(int));
    cudaMemcpy(d_target_ids, &state_id, sizeof(int), cudaMemcpyHostToDevice);

    apply_phase_rotation(pool, d_target_ids, 1, theta);

    cudaFree(d_target_ids);

    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(d_trunc);
    pool->download_state(state_id, gpu_result);

    // 比较误差
    double error = compare_with_reference(gpu_result, ref_result);
    EXPECT_LT(error, 1e-10) << "GPU相位旋转门误差过大";
}

TEST_F(GateTest, TestDisplacement) {
    // 测试位移门 D(α) - Reference实现
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

TEST_F(GPUGateTest, TestGPUDisplacement) {
    // 测试GPU位移门 D(α)
    std::complex<double> alpha(0.1, 0.0);

    // 获取参考结果
    std::vector<std::complex<double>> initial_state(d_trunc, std::complex<double>(0.0, 0.0));
    initial_state[0] = std::complex<double>(1.0, 0.0);
    auto ref_result = Reference::SingleModeGates::apply_displacement_gate(initial_state, alpha);

    // 应用GPU门
    cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.real(), alpha.imag());
    int* d_target_ids = nullptr;
    cudaMalloc(&d_target_ids, sizeof(int));
    cudaMemcpy(d_target_ids, &state_id, sizeof(int), cudaMemcpyHostToDevice);

    apply_displacement_gate(pool, d_target_ids, 1, alpha_cu);
    
    cudaFree(d_target_ids);

    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(d_trunc);
    pool->download_state(state_id, gpu_result);

    // 比较误差
    double error = compare_with_reference(gpu_result, ref_result);
    EXPECT_LT(error, 1e-6) << "GPU位移门误差过大: " << error;
}

TEST_F(GPUGateTest, TestGPUCreationOperator) {
    // 测试GPU湮灭算符 a†
    // 获取参考结果（创建算符）
    std::vector<std::complex<double>> initial_state(d_trunc, std::complex<double>(0.0, 0.0));
    initial_state[0] = std::complex<double>(1.0, 0.0);  // |0⟩
    auto ref_result = Reference::LadderGates::apply_creation_operator(initial_state);

    // 应用GPU门
    int* d_target_ids = nullptr;
    cudaMalloc(&d_target_ids, sizeof(int));
    cudaMemcpy(d_target_ids, &state_id, sizeof(int), cudaMemcpyHostToDevice);

    apply_creation_operator(pool, d_target_ids, 1);
    
    cudaFree(d_target_ids);

    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(d_trunc);
    pool->download_state(state_id, gpu_result);

    // 比较误差
    double error = compare_with_reference(gpu_result, ref_result);
    EXPECT_LT(error, 1e-10) << "GPU创建算符误差过大: " << error;
}

TEST_F(GPUGateTest, TestGPUAnnihilationOperator) {
    // 测试GPU湮灭算符 a
    // 先创建 |1⟩ 状态
    int new_state = pool->allocate_state();
    std::vector<cuDoubleComplex> one_state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    one_state[1] = make_cuDoubleComplex(1.0, 0.0);  // |1⟩
    pool->upload_state(new_state, one_state);

    // 获取参考结果
    std::vector<std::complex<double>> initial_state(d_trunc, std::complex<double>(0.0, 0.0));
    initial_state[1] = std::complex<double>(1.0, 0.0);
    auto ref_result = Reference::LadderGates::apply_annihilation_operator(initial_state);

    // 应用GPU门
    int* d_target_ids = nullptr;
    cudaMalloc(&d_target_ids, sizeof(int));
    cudaMemcpy(d_target_ids, &new_state, sizeof(int), cudaMemcpyHostToDevice);

    apply_annihilation_operator(pool, d_target_ids, 1);
    
    cudaFree(d_target_ids);

    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(d_trunc);
    pool->download_state(new_state, gpu_result);

    // 比较误差
    double error = compare_with_reference(gpu_result, ref_result);
    EXPECT_LT(error, 1e-10) << "GPU湮灭算符误差过大: " << error;

    pool->free_state(new_state);
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

TEST_F(GateTest, TestControlledTwoModeSqueezing) {
    const int cutoff = 4;
    const int dim = cutoff * cutoff;
    Reference::Vector target_state(dim, Reference::Complex(0.0, 0.0));
    target_state[0] = Reference::Complex(1.0, 0.0);

    const Reference::Complex xi(0.15, 0.05);

    Reference::Vector result0 =
        Reference::HybridControlGates::apply_controlled_two_mode_squeezing(0, target_state, xi);
    Reference::Vector expected0 =
        Reference::TwoModeGatesExtended::apply_two_mode_squeezing(target_state, xi);
    EXPECT_LT(compute_error(result0, expected0), 1e-10);

    Reference::Vector result1 =
        Reference::HybridControlGates::apply_controlled_two_mode_squeezing(1, target_state, xi);
    Reference::Vector expected1 =
        Reference::TwoModeGatesExtended::apply_two_mode_squeezing(target_state, -xi);
    EXPECT_LT(compute_error(result1, expected1), 1e-10);
}

TEST_F(GateTest, TestControlledSum) {
    const int cutoff = 4;
    const int dim = cutoff * cutoff;
    Reference::Vector target_state(dim, Reference::Complex(0.0, 0.0));
    target_state[cutoff] = Reference::Complex(1.0, 0.0);  // |1,0>

    const double theta = 0.12;

    Reference::Vector result0 =
        Reference::HybridControlGates::apply_controlled_sum(0, target_state, theta, 0.0);
    Reference::Vector expected0 =
        Reference::TwoModeGatesExtended::apply_sum_gate(target_state, theta, 0.0);
    EXPECT_LT(compute_error(result0, expected0), 1e-10);

    Reference::Vector result1 =
        Reference::HybridControlGates::apply_controlled_sum(1, target_state, theta, 0.0);
    Reference::Vector expected1 =
        Reference::TwoModeGatesExtended::apply_sum_gate(target_state, -theta, 0.0);
    EXPECT_LT(compute_error(result1, expected1), 1e-10);
}

TEST_F(GateTest, TestRabiInteraction) {
    Reference::Vector qubit_state = {
        {1.0 / std::sqrt(2.0), 0.0},
        {1.0 / std::sqrt(2.0), 0.0}
    };
    const int dim = 4;
    Reference::Vector qumode_state(dim, Reference::Complex(0.0, 0.0));
    qumode_state[0] = Reference::Complex(1.0, 0.0);  // 真空态

    double theta = M_PI / 4.0;

    Reference::Vector result = Reference::HybridControlGates::apply_rabi_interaction(qubit_state, qumode_state, theta);
    Reference::Vector expected = Reference::tensor_product(
        qubit_state,
        Reference::SingleModeGates::apply_displacement_gate(qumode_state, Reference::Complex(0.0, -theta)));
    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "Rabi相互作用参考实现测试失败";
}

TEST_F(GateTest, TestJaynesCummings) {
    Reference::Vector qubit_state = {{1.0, 0.0}, {0.0, 0.0}};  // |0⟩
    const int dim = 4;
    Reference::Vector qumode_state(dim, Reference::Complex(0.0, 0.0));
    qumode_state[1] = Reference::Complex(1.0, 0.0);  // |1⟩

    double theta = M_PI / 4.0;
    double phi = 0.0;

    Reference::Vector result = Reference::HybridControlGates::apply_jaynes_cummings(qubit_state, qumode_state, theta, phi);
    Reference::Vector expected(2 * dim, Reference::Complex(0.0, 0.0));
    expected[1] = std::cos(theta);
    expected[dim] = Reference::Complex(0.0, -std::sin(theta));
    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "Jaynes-Cummings相互作用测试失败";
}

TEST_F(GateTest, TestAntiJaynesCummings) {
    Reference::Vector qubit_state = {{1.0, 0.0}, {0.0, 0.0}};  // |0⟩
    const int dim = 4;
    Reference::Vector qumode_state(dim, Reference::Complex(0.0, 0.0));
    qumode_state[0] = Reference::Complex(1.0, 0.0);  // |0⟩

    double theta = M_PI / 4.0;
    double phi = 0.0;

    Reference::Vector result =
        Reference::HybridControlGates::apply_anti_jaynes_cummings(qubit_state, qumode_state, theta, phi);
    Reference::Vector expected(2 * dim, Reference::Complex(0.0, 0.0));
    expected[0] = std::cos(theta);
    expected[dim + 1] = Reference::Complex(0.0, -std::sin(theta));
    double error = compute_error(result, expected);
    EXPECT_LT(error, 1e-10) << "Anti-Jaynes-Cummings相互作用测试失败";
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

TEST_F(GateTest, TestQuantumCircuitWithHybridGates) {
    // 测试包含混合门的量子电路
    try {
        QuantumCircuit circuit(2, 2, 4, 1024);  // 2 qubits, 2 qumodes, truncation=4

        // 添加混合门
        circuit.add_gate(Gates::ConditionalDisplacement(0, 0, std::complex<double>(0.1, 0.0)));
        circuit.add_gate(Gates::ConditionalSqueezing(1, 1, std::complex<double>(0.05, 0.0)));
        circuit.add_gate(Gates::RabiInteraction(0, 0, M_PI/4.0));

        // 构建和执行
        circuit.build();
        circuit.execute();

        // 验证电路执行成功
        SUCCEED() << "包含混合门的量子电路测试通过";

    } catch (const std::exception& e) {
        FAIL() << "混合门电路测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestAllGatesConstruction) {
    // 测试所有门的构建和基本执行
    try {
        QuantumCircuit circuit(3, 2, 8, 1024);  // 3 qubits, 2 qumodes

        // 添加所有类型的门
        // Qubit门
        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::PauliX(1));
        circuit.add_gate(Gates::PauliY(2));
        circuit.add_gate(Gates::PauliZ(0));
        circuit.add_gate(Gates::RotationX(1, M_PI/4.0));
        circuit.add_gate(Gates::RotationY(2, M_PI/4.0));
        circuit.add_gate(Gates::RotationZ(0, M_PI/4.0));
        circuit.add_gate(Gates::PhaseGateS(1));
        circuit.add_gate(Gates::PhaseGateT(2));
        circuit.add_gate(Gates::CNOT(0, 1));
        circuit.add_gate(Gates::CZ(1, 2));

        // Qumode门
        circuit.add_gate(Gates::PhaseRotation(0, M_PI/4.0));
        circuit.add_gate(Gates::Displacement(1, std::complex<double>(0.1, 0.0)));
        circuit.add_gate(Gates::Squeezing(0, std::complex<double>(0.05, 0.0)));

        // 混合门
        circuit.add_gate(Gates::ConditionalDisplacement(0, 1, std::complex<double>(0.1, 0.0)));
        circuit.add_gate(Gates::ConditionalSqueezing(1, 0, std::complex<double>(0.05, 0.0)));
        // 注意：RabiInteraction等混合型门可能需要特殊处理，暂时跳过

        // 构建和执行
        circuit.build();
        circuit.execute();

        // 验证电路执行成功
        SUCCEED() << "所有门构建和执行测试通过";

    } catch (const std::exception& e) {
        FAIL() << "门构建和执行测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestQuantumCircuitWithTwoModeHybridGates) {
    try {
        QuantumCircuit circuit(1, 2, 8, 1024);

        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::ConditionalTwoModeSqueezing(0, 0, 1, std::complex<double>(0.12, 0.03)));
        circuit.add_gate(Gates::ConditionalSUM(0, 0, 1, 0.08, 0.0));

        circuit.build();
        circuit.execute();

        SUCCEED() << "双模混合门线路测试通过";
    } catch (const std::exception& e) {
        FAIL() << "双模混合门线路测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestMultiQumodeSingleModeAddressing) {
    try {
        QuantumCircuit circuit(1, 3, 4, 1024);
        circuit.add_gate(Gates::CreationOperator(2));
        circuit.add_gate(Gates::PhaseRotation(2, M_PI / 2.0));

        circuit.build();
        circuit.execute();

        std::vector<std::complex<double>> vacuum(4, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};
        std::vector<std::complex<double>> one(4, {0.0, 0.0});
        one[1] = {1.0, 0.0};

        const auto amplitude = circuit.get_amplitude({0}, {vacuum, vacuum, one});
        EXPECT_NEAR(amplitude.real(), 0.0, 1e-8);
        EXPECT_NEAR(amplitude.imag(), -1.0, 1e-8);
    } catch (const std::exception& e) {
        FAIL() << "多qumode单模寻址测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestMultiQumodeHybridAndTwoModeSmoke) {
    try {
        QuantumCircuit circuit(1, 3, 4, 1024);

        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::Squeezing(2, std::complex<double>(0.08, 0.0)));
        circuit.add_gate(Gates::BeamSplitter(0, 2, M_PI / 6.0, 0.0));
        circuit.add_gate(Gates::ConditionalTwoModeSqueezing(0, 1, 2, std::complex<double>(0.06, 0.02)));
        circuit.add_gate(Gates::ConditionalSUM(0, 0, 2, 0.04, 0.0));
        circuit.add_gate(Gates::RabiInteraction(0, 2, 0.1));
        circuit.add_gate(Gates::JaynesCummings(0, 1, 0.08, 0.0));
        circuit.add_gate(Gates::AntiJaynesCummings(0, 0, 0.05, 0.0));

        circuit.build();
        circuit.execute();

        SUCCEED() << "多qumode双模/混合门线路测试通过";
    } catch (const std::exception& e) {
        FAIL() << "多qumode双模/混合门线路测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestGaussianPrefixEDEProducesCorrectState) {
    try {
        constexpr int cutoff = 6;
        QuantumCircuit circuit(1, 2, cutoff, 1024);

        const std::complex<double> alpha(0.15, 0.02);
        const std::complex<double> xi(0.08, 0.0);
        const double theta = 0.17;

        circuit.add_gate(Gates::Displacement(0, alpha));
        circuit.add_gate(Gates::Squeezing(1, xi));
        circuit.add_gate(Gates::PhaseRotation(1, theta));
        circuit.add_gate(Gates::CreationOperator(0));

        circuit.build();
        circuit.execute();

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector ref_mode0 =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
        ref_mode0 = Reference::LadderGates::apply_creation_operator(ref_mode0);

        Reference::Vector ref_mode1 =
            Reference::SingleModeGates::apply_squeezing_gate(vacuum, xi);
        ref_mode1 = Reference::DiagonalGates::apply_phase_rotation(ref_mode1, theta);

        const Reference::Vector expected = Reference::tensor_product(ref_mode0, ref_mode1);

        Reference::Vector actual(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        for (int n0 = 0; n0 < cutoff; ++n0) {
            std::vector<std::complex<double>> basis0(cutoff, {0.0, 0.0});
            basis0[n0] = {1.0, 0.0};
            for (int n1 = 0; n1 < cutoff; ++n1) {
                std::vector<std::complex<double>> basis1(cutoff, {0.0, 0.0});
                basis1[n1] = {1.0, 0.0};
                actual[static_cast<size_t>(n0 * cutoff + n1)] =
                    circuit.get_amplitude({0}, {basis0, basis1});
            }
        }

        EXPECT_LT(compute_error(expected, actual), 1e-4);
    } catch (const std::exception& e) {
        FAIL() << "Gaussian前缀EDE正确性测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestGaussianPrefixEDEHandlesCorrelatedState) {
    try {
        constexpr int cutoff = 6;
        QuantumCircuit circuit(1, 2, cutoff, 1024);

        circuit.add_gate(Gates::Squeezing(0, std::complex<double>(0.25, 0.0)));
        circuit.add_gate(Gates::BeamSplitter(0, 1, 0.31, 0.0));
        circuit.add_gate(Gates::CreationOperator(0));

        circuit.build();
        circuit.execute();

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected = Reference::tensor_product(
            Reference::SingleModeGates::apply_squeezing_gate(vacuum, Reference::Complex(0.25, 0.0)),
            vacuum);
        expected = Reference::TwoModeGates::apply_beam_splitter(expected, 0.31, 0.0);
        expected = apply_single_mode_transform(
            expected,
            cutoff,
            2,
            0,
            [](const Reference::Vector& local_state) {
                return Reference::LadderGates::apply_creation_operator(local_state);
            });

        Reference::Vector actual(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        for (int n0 = 0; n0 < cutoff; ++n0) {
            std::vector<std::complex<double>> basis0(cutoff, {0.0, 0.0});
            basis0[n0] = {1.0, 0.0};
            for (int n1 = 0; n1 < cutoff; ++n1) {
                std::vector<std::complex<double>> basis1(cutoff, {0.0, 0.0});
                basis1[n1] = {1.0, 0.0};
                actual[static_cast<size_t>(n0 * cutoff + n1)] =
                    circuit.get_amplitude({0}, {basis0, basis1});
            }
        }

        EXPECT_LT(compute_error(expected, actual), 1e-4);
    } catch (const std::exception& e) {
        FAIL() << "相关Gaussian前缀EDE投影测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestExecutionBlocksCanonicalizeCommutingDiagonalWindow) {
    try {
        constexpr int cutoff = 6;
        QuantumCircuit circuit(1, 2, cutoff, 1024);

        const double theta0 = 0.11;
        const double theta1 = -0.19;
        const double chi = 0.07;
        const std::complex<double> alpha(0.18, -0.04);

        circuit.add_gate(Gates::PhaseRotation(0, theta0));
        circuit.add_gate(Gates::KerrGate(1, chi));
        circuit.add_gate(Gates::PhaseRotation(1, theta1));
        circuit.add_gate(Gates::Displacement(0, alpha));
        circuit.add_gate(Gates::CreationOperator(1));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("Gaussian EDE块级加速已启用，块门数=2"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector ref_mode0 =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
        Reference::Vector ref_mode1 = Reference::LadderGates::apply_creation_operator(vacuum);

        const Reference::Vector expected = Reference::tensor_product(ref_mode0, ref_mode1);

        Reference::Vector actual(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        for (int n0 = 0; n0 < cutoff; ++n0) {
            std::vector<std::complex<double>> basis0(cutoff, {0.0, 0.0});
            basis0[n0] = {1.0, 0.0};
            for (int n1 = 0; n1 < cutoff; ++n1) {
                std::vector<std::complex<double>> basis1(cutoff, {0.0, 0.0});
                basis1[n1] = {1.0, 0.0};
                actual[static_cast<size_t>(n0 * cutoff + n1)] =
                    circuit.get_amplitude({0}, {basis0, basis1});
            }
        }

        EXPECT_LT(compute_error(expected, actual), 1e-4);
    } catch (const std::exception& e) {
        FAIL() << "块级执行规范化测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestGaussianBlockSupportsControlledGaussianTrack) {
    try {
        constexpr int cutoff = 10;
        QuantumCircuit circuit(1, 2, cutoff, 2048);

        const std::complex<double> alpha(0.22, -0.07);
        const std::complex<double> xi(0.18, 0.04);
        const double theta = 0.29;
        const double phi = -0.21;

        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::ConditionalDisplacement(0, 0, alpha));
        circuit.add_gate(Gates::ConditionalSqueezing(0, 1, xi));
        circuit.add_gate(Gates::ConditionalBeamSplitter(0, 0, 1, theta, phi));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("Gaussian EDE块级加速已启用，块门数=3"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected_low = Reference::tensor_product(
            vacuum,
            Reference::SingleModeGates::apply_squeezing_gate(vacuum, xi));
        expected_low = apply_two_mode_transform(
            expected_low,
            cutoff,
            2,
            0,
            1,
            [theta, phi](const Reference::Vector& local_state) {
                return Reference::TwoModeGates::apply_beam_splitter(local_state, theta, phi);
            });

        Reference::Vector expected_high = Reference::tensor_product(
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha),
            Reference::SingleModeGates::apply_squeezing_gate(vacuum, -xi));
        expected_high = apply_two_mode_transform(
            expected_high,
            cutoff,
            2,
            0,
            1,
            [theta, phi](const Reference::Vector& local_state) {
                return Reference::TwoModeGates::apply_beam_splitter(local_state, -theta, phi);
            });

        const double branch_scale = 1.0 / std::sqrt(2.0);
        for (std::complex<double>& amplitude : expected_low) {
            amplitude *= branch_scale;
        }
        for (std::complex<double>& amplitude : expected_high) {
            amplitude *= branch_scale;
        }

        Reference::Vector actual_low(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        Reference::Vector actual_high(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        for (int n0 = 0; n0 < cutoff; ++n0) {
            std::vector<std::complex<double>> basis0(cutoff, {0.0, 0.0});
            basis0[n0] = {1.0, 0.0};
            for (int n1 = 0; n1 < cutoff; ++n1) {
                std::vector<std::complex<double>> basis1(cutoff, {0.0, 0.0});
                basis1[n1] = {1.0, 0.0};
                const size_t linear_index = static_cast<size_t>(n0 * cutoff + n1);
                actual_low[linear_index] = circuit.get_amplitude({0}, {basis0, basis1});
                actual_high[linear_index] = circuit.get_amplitude({1}, {basis0, basis1});
            }
        }

        EXPECT_LT(compute_error(expected_low, actual_low), 1e-4);
        EXPECT_LT(compute_error(expected_high, actual_high), 1e-4);
    } catch (const std::exception& e) {
        FAIL() << "受控Gaussian block测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestGaussianBlockSupportsControlledTwoModeGaussianTrack) {
    try {
        constexpr int cutoff = 10;
        QuantumCircuit circuit(1, 2, cutoff, 2048);

        const std::complex<double> xi(0.16, 0.05);
        const double theta = 0.11;
        const double rotation = -0.17;

        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::ConditionalTwoModeSqueezing(0, 0, 1, xi));
        circuit.add_gate(Gates::ConditionalSUM(0, 0, 1, theta, 0.0));
        circuit.add_gate(Gates::PhaseRotation(1, rotation));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("Gaussian EDE块级加速已启用，块门数=3"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected_low = Reference::tensor_product(vacuum, vacuum);
        expected_low = apply_two_mode_transform(
            expected_low,
            cutoff,
            2,
            0,
            1,
            [xi](const Reference::Vector& local_state) {
                return Reference::TwoModeGatesExtended::apply_two_mode_squeezing(local_state, xi);
            });
        expected_low = apply_two_mode_transform(
            expected_low,
            cutoff,
            2,
            0,
            1,
            [theta](const Reference::Vector& local_state) {
                return Reference::TwoModeGatesExtended::apply_sum_gate(local_state, theta, 0.0);
            });
        expected_low = apply_single_mode_transform(
            expected_low,
            cutoff,
            2,
            1,
            [rotation](const Reference::Vector& local_state) {
                return Reference::DiagonalGates::apply_phase_rotation(local_state, rotation);
            });

        Reference::Vector expected_high = Reference::tensor_product(vacuum, vacuum);
        expected_high = apply_two_mode_transform(
            expected_high,
            cutoff,
            2,
            0,
            1,
            [xi](const Reference::Vector& local_state) {
                return Reference::TwoModeGatesExtended::apply_two_mode_squeezing(local_state, -xi);
            });
        expected_high = apply_two_mode_transform(
            expected_high,
            cutoff,
            2,
            0,
            1,
            [theta](const Reference::Vector& local_state) {
                return Reference::TwoModeGatesExtended::apply_sum_gate(local_state, -theta, 0.0);
            });
        expected_high = apply_single_mode_transform(
            expected_high,
            cutoff,
            2,
            1,
            [rotation](const Reference::Vector& local_state) {
                return Reference::DiagonalGates::apply_phase_rotation(local_state, rotation);
            });

        const double branch_scale = 1.0 / std::sqrt(2.0);
        for (std::complex<double>& amplitude : expected_low) {
            amplitude *= branch_scale;
        }
        for (std::complex<double>& amplitude : expected_high) {
            amplitude *= branch_scale;
        }

        Reference::Vector actual_low(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        Reference::Vector actual_high(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        for (int n0 = 0; n0 < cutoff; ++n0) {
            std::vector<std::complex<double>> basis0(cutoff, {0.0, 0.0});
            basis0[n0] = {1.0, 0.0};
            for (int n1 = 0; n1 < cutoff; ++n1) {
                std::vector<std::complex<double>> basis1(cutoff, {0.0, 0.0});
                basis1[n1] = {1.0, 0.0};
                const size_t linear_index = static_cast<size_t>(n0 * cutoff + n1);
                actual_low[linear_index] = circuit.get_amplitude({0}, {basis0, basis1});
                actual_high[linear_index] = circuit.get_amplitude({1}, {basis0, basis1});
            }
        }

        EXPECT_LT(compute_error(expected_low, actual_low), 1e-4);
        EXPECT_LT(compute_error(expected_high, actual_high), 1e-4);
    } catch (const std::exception& e) {
        FAIL() << "受控双模Gaussian block测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestPureQubitBlockPreservesGaussianSymbolicTrack) {
    try {
        constexpr int cutoff = 10;
        QuantumCircuit circuit(1, 1, cutoff, 2048);

        const std::complex<double> alpha(0.18, -0.04);
        const std::complex<double> xi(0.13, 0.02);
        const double rotation = 0.27;

        circuit.add_gate(Gates::Displacement(0, alpha));
        circuit.add_gate(Gates::Squeezing(0, xi));
        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::PhaseRotation(0, rotation));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("Gaussian EDE块级加速已启用，块门数=2"), std::string::npos);
        EXPECT_NE(execution_log.find("Gaussian EDE块级加速已启用，块门数=1"), std::string::npos);
        EXPECT_EQ(execution_log.find("Gaussian EDE块回退到全量Fock执行"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
        expected = Reference::SingleModeGates::apply_squeezing_gate(expected, xi);
        expected = Reference::DiagonalGates::apply_phase_rotation(expected, rotation);

        const double branch_scale = 1.0 / std::sqrt(2.0);
        for (std::complex<double>& amplitude : expected) {
            amplitude *= branch_scale;
        }

        Reference::Vector actual_low(static_cast<size_t>(cutoff), {0.0, 0.0});
        Reference::Vector actual_high(static_cast<size_t>(cutoff), {0.0, 0.0});
        for (int n = 0; n < cutoff; ++n) {
            std::vector<std::complex<double>> basis(cutoff, {0.0, 0.0});
            basis[n] = {1.0, 0.0};
            actual_low[static_cast<size_t>(n)] = circuit.get_amplitude({0}, {basis});
            actual_high[static_cast<size_t>(n)] = circuit.get_amplitude({1}, {basis});
        }

        EXPECT_LT(compute_error(expected, actual_low), 1e-4);
        EXPECT_LT(compute_error(expected, actual_high), 1e-4);
    } catch (const std::exception& e) {
        FAIL() << "纯qubit block保留Gaussian symbolic track测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestDiagonalNonGaussianBlockUsesGaussianMixture) {
    try {
        constexpr int cutoff = 16;
        QuantumCircuit circuit(1, 1, cutoff, 1024);

        const std::complex<double> alpha(0.30, -0.05);
        const double chi = 5e-4;
        const double parity = 1.0;

        circuit.add_gate(Gates::Displacement(0, alpha));
        circuit.add_gate(Gates::KerrGate(0, chi));
        circuit.add_gate(Gates::ConditionalParity(0, parity));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("对角非高斯块Gaussian Mixture已启用"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
        expected = Reference::DiagonalGates::apply_kerr_gate(expected, chi);
        expected = Reference::DiagonalGates::apply_conditional_parity(expected, parity);

        Reference::Vector actual(static_cast<size_t>(cutoff), {0.0, 0.0});
        for (int n = 0; n < cutoff; ++n) {
            std::vector<std::complex<double>> basis(cutoff, {0.0, 0.0});
            basis[n] = {1.0, 0.0};
            actual[static_cast<size_t>(n)] = circuit.get_amplitude({0}, {basis});
        }

        const Reference::ErrorMetrics metrics = Reference::compute_error_metrics(expected, actual);
        EXPECT_LT(metrics.fidelity_deviation, 1e-4);
        EXPECT_LT(metrics.l2_error, 2e-2);
    } catch (const std::exception& e) {
        FAIL() << "对角非高斯块Gaussian Mixture集成测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestSnapBlockUsesGaussianMixture) {
    try {
        constexpr int cutoff = 16;
        QuantumCircuit circuit(1, 1, cutoff, 1024);

        const std::complex<double> alpha(0.18, -0.07);
        const double theta = 0.05;
        const int target_fock_state = 2;

        circuit.add_gate(Gates::Displacement(0, alpha));
        circuit.add_gate(Gates::Snap(0, theta, target_fock_state));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("对角非高斯块Gaussian Mixture已启用"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
        expected[static_cast<size_t>(target_fock_state)] *=
            std::exp(std::complex<double>(0.0, theta));

        Reference::Vector actual(static_cast<size_t>(cutoff), {0.0, 0.0});
        for (int n = 0; n < cutoff; ++n) {
            std::vector<std::complex<double>> basis(cutoff, {0.0, 0.0});
            basis[n] = {1.0, 0.0};
            actual[static_cast<size_t>(n)] = circuit.get_amplitude({0}, {basis});
        }

        const Reference::ErrorMetrics metrics = Reference::compute_error_metrics(expected, actual);
        EXPECT_LT(metrics.fidelity_deviation, 1e-4);
        EXPECT_LT(metrics.l2_error, 2e-2);
    } catch (const std::exception& e) {
        FAIL() << "SNAP块Gaussian Mixture集成测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestMultiSnapBlockUsesGaussianMixture) {
    try {
        constexpr int cutoff = 16;
        QuantumCircuit circuit(1, 1, cutoff, 1024);

        const std::complex<double> alpha(0.20, 0.03);
        const std::vector<double> phase_map = {0.0, 0.01, 0.0, -0.008};

        circuit.add_gate(Gates::Displacement(0, alpha));
        circuit.add_gate(Gates::MultiSNAP(0, phase_map));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("对角非高斯块Gaussian Mixture已启用"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
        for (size_t idx = 0; idx < phase_map.size(); ++idx) {
            expected[idx] *= std::exp(std::complex<double>(0.0, phase_map[idx]));
        }

        Reference::Vector actual(static_cast<size_t>(cutoff), {0.0, 0.0});
        for (int n = 0; n < cutoff; ++n) {
            std::vector<std::complex<double>> basis(cutoff, {0.0, 0.0});
            basis[n] = {1.0, 0.0};
            actual[static_cast<size_t>(n)] = circuit.get_amplitude({0}, {basis});
        }

        const Reference::ErrorMetrics metrics = Reference::compute_error_metrics(expected, actual);
        EXPECT_LT(metrics.fidelity_deviation, 1e-4);
        EXPECT_LT(metrics.l2_error, 2e-2);
    } catch (const std::exception& e) {
        FAIL() << "Multi-SNAP块Gaussian Mixture集成测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestCrossKerrBlockUsesGaussianMixture) {
    try {
        constexpr int cutoff = 8;
        QuantumCircuit circuit(1, 2, cutoff, 1024);

        const std::complex<double> alpha0(0.20, -0.05);
        const std::complex<double> alpha1(-0.15, 0.08);
        const double kappa = 1e-3;

        circuit.add_gate(Gates::Displacement(0, alpha0));
        circuit.add_gate(Gates::Displacement(1, alpha1));
        circuit.add_gate(Gates::CrossKerr(0, 1, kappa));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("对角非高斯块Gaussian Mixture已启用"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        const Reference::Vector mode0 =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha0);
        const Reference::Vector mode1 =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha1);
        Reference::Vector expected = Reference::tensor_product(mode0, mode1);
        expected = apply_two_mode_transform(
            expected,
            cutoff,
            2,
            0,
            1,
            [cutoff, kappa](const Reference::Vector& local_state) {
                Reference::Vector transformed = local_state;
                for (int first = 0; first < cutoff; ++first) {
                    for (int second = 0; second < cutoff; ++second) {
                        const size_t index =
                            static_cast<size_t>(first) * cutoff + static_cast<size_t>(second);
                        transformed[index] *=
                            std::exp(std::complex<double>(0.0, kappa * static_cast<double>(first * second)));
                    }
                }
                return transformed;
            });

        Reference::Vector actual(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        for (int n0 = 0; n0 < cutoff; ++n0) {
            std::vector<std::complex<double>> basis0(cutoff, {0.0, 0.0});
            basis0[n0] = {1.0, 0.0};
            for (int n1 = 0; n1 < cutoff; ++n1) {
                std::vector<std::complex<double>> basis1(cutoff, {0.0, 0.0});
                basis1[n1] = {1.0, 0.0};
                actual[static_cast<size_t>(n0) * cutoff + static_cast<size_t>(n1)] =
                    circuit.get_amplitude({0}, {basis0, basis1});
            }
        }

        const Reference::ErrorMetrics metrics = Reference::compute_error_metrics(expected, actual);
        EXPECT_LT(metrics.fidelity_deviation, 1e-4);
        EXPECT_LT(metrics.l2_error, 2e-2);
    } catch (const std::exception& e) {
        FAIL() << "Cross-Kerr块Gaussian Mixture集成测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestGaussianMixtureBranchStaysSymbolicForFollowingGaussianBlock) {
    try {
        constexpr int cutoff = 16;
        QuantumCircuit circuit(1, 1, cutoff, 1024);

        const std::complex<double> alpha(0.22, -0.04);
        const double chi = 5e-4;
        const std::complex<double> xi(0.08, -0.03);

        circuit.add_gate(Gates::Displacement(0, alpha));
        circuit.add_gate(Gates::KerrGate(0, chi));
        circuit.add_gate(Gates::Squeezing(0, xi));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("对角非高斯块Gaussian Mixture已启用"), std::string::npos);
        EXPECT_EQ(count_occurrences(execution_log, "Gaussian EDE块级加速已启用"), 2u);
        EXPECT_EQ(execution_log.find("Gaussian EDE块回退到全量Fock执行"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
        expected = Reference::DiagonalGates::apply_kerr_gate(expected, chi);
        expected = Reference::SingleModeGates::apply_squeezing_gate(expected, xi);

        Reference::Vector actual(static_cast<size_t>(cutoff), {0.0, 0.0});
        for (int n = 0; n < cutoff; ++n) {
            std::vector<std::complex<double>> basis(cutoff, {0.0, 0.0});
            basis[n] = {1.0, 0.0};
            actual[static_cast<size_t>(n)] = circuit.get_amplitude({0}, {basis});
        }

        const Reference::ErrorMetrics metrics = Reference::compute_error_metrics(expected, actual);
        EXPECT_LT(metrics.fidelity_deviation, 1e-4);
        EXPECT_LT(metrics.l2_error, 2e-2);
    } catch (const std::exception& e) {
        FAIL() << "GaussianMixture后续Gaussian block符号路径测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestGaussianMixtureFallsBackToFockForUnsupportedFollowingBlock) {
    try {
        constexpr int cutoff = 16;
        QuantumCircuit circuit(1, 1, cutoff, 1024);

        const std::complex<double> alpha(0.18, 0.02);
        const double chi = 5e-4;

        circuit.add_gate(Gates::Displacement(0, alpha));
        circuit.add_gate(Gates::KerrGate(0, chi));
        circuit.add_gate(Gates::CreationOperator(0));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_NE(execution_log.find("对角非高斯块Gaussian Mixture已启用"), std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        Reference::Vector expected =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
        expected = Reference::DiagonalGates::apply_kerr_gate(expected, chi);
        expected = Reference::LadderGates::apply_creation_operator(expected);

        Reference::Vector actual(static_cast<size_t>(cutoff), {0.0, 0.0});
        for (int n = 0; n < cutoff; ++n) {
            std::vector<std::complex<double>> basis(cutoff, {0.0, 0.0});
            basis[n] = {1.0, 0.0};
            actual[static_cast<size_t>(n)] = circuit.get_amplitude({0}, {basis});
        }

        const Reference::ErrorMetrics metrics = Reference::compute_error_metrics(expected, actual);
        EXPECT_LT(metrics.fidelity_deviation, 1e-4);
        EXPECT_LT(metrics.l2_error, 2e-2);
    } catch (const std::exception& e) {
        FAIL() << "GaussianMixture向Fock回退测试失败: " << e.what();
    }
}

TEST_F(GateTest, TestLargeCrossKerrBlockFallsBackToExactFockForFidelityTarget) {
    try {
        constexpr int cutoff = 16;
        QuantumCircuit circuit(1, 2, cutoff, 1024);

        const std::complex<double> alpha0(0.18, -0.03);
        const std::complex<double> alpha1(-0.11, 0.04);
        const double kappa = 0.02;

        circuit.add_gate(Gates::Displacement(0, alpha0));
        circuit.add_gate(Gates::Displacement(1, alpha1));
        circuit.add_gate(Gates::CrossKerr(0, 1, kappa));

        circuit.build();

        testing::internal::CaptureStdout();
        circuit.execute();
        const std::string execution_log = testing::internal::GetCapturedStdout();

        EXPECT_EQ(execution_log.find("对角非高斯块Gaussian Mixture已启用"), std::string::npos);
        EXPECT_NE(execution_log.find("对角非高斯块Mixture预编译失败，回退到精确Fock执行"),
                  std::string::npos);

        Reference::Vector vacuum(cutoff, {0.0, 0.0});
        vacuum[0] = {1.0, 0.0};

        const Reference::Vector mode0 =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha0);
        const Reference::Vector mode1 =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha1);
        Reference::Vector expected = Reference::tensor_product(mode0, mode1);
        expected = apply_two_mode_transform(
            expected,
            cutoff,
            2,
            0,
            1,
            [cutoff, kappa](const Reference::Vector& local_state) {
                Reference::Vector transformed = local_state;
                for (int first = 0; first < cutoff; ++first) {
                    for (int second = 0; second < cutoff; ++second) {
                        const size_t index =
                            static_cast<size_t>(first) * cutoff + static_cast<size_t>(second);
                        transformed[index] *= std::exp(std::complex<double>(
                            0.0, kappa * static_cast<double>(first * second)));
                    }
                }
                return transformed;
            });

        Reference::Vector actual(static_cast<size_t>(cutoff * cutoff), {0.0, 0.0});
        for (int n0 = 0; n0 < cutoff; ++n0) {
            std::vector<std::complex<double>> basis0(cutoff, {0.0, 0.0});
            basis0[n0] = {1.0, 0.0};
            for (int n1 = 0; n1 < cutoff; ++n1) {
                std::vector<std::complex<double>> basis1(cutoff, {0.0, 0.0});
                basis1[n1] = {1.0, 0.0};
                actual[static_cast<size_t>(n0) * cutoff + static_cast<size_t>(n1)] =
                    circuit.get_amplitude({0}, {basis0, basis1});
            }
        }

        const Reference::ErrorMetrics metrics = Reference::compute_error_metrics(expected, actual);
        EXPECT_LT(metrics.fidelity_deviation, 1e-10);
        EXPECT_LT(metrics.l2_error, 1e-10);
    } catch (const std::exception& e) {
        FAIL() << "大Cross-Kerr门精确Fock回退测试失败: " << e.what();
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
