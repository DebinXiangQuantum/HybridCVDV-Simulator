#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <memory>

/**
 * 传统矩阵向量乘法参考实现
 *
 * 用于验证GPU实现的正确性，通过对比误差来确保实现正确
 */

namespace Reference {

// 复数类型别名
using Complex = std::complex<double>;
using Vector = std::vector<Complex>;
using Matrix = std::vector<std::vector<Complex>>;

/**
 * 工具函数：创建单位矩阵
 */
Matrix create_identity_matrix(int dim);

/**
 * 工具函数：矩阵向量乘法
 */
Vector matrix_vector_multiply(const Matrix& matrix, const Vector& vector);

/**
 * 工具函数：计算两个向量的内积
 */
Complex inner_product(const Vector& v1, const Vector& v2);

/**
 * 工具函数：计算向量范数
 */
double vector_norm(const Vector& v);

/**
 * 工具函数：归一化向量
 */
Vector normalize_vector(const Vector& v);

/**
 * 工具函数：计算两个向量之间的保真度
 */
double fidelity(const Vector& v1, const Vector& v2);

/**
 * 工具函数：计算两个实现之间的误差
 */
struct ErrorMetrics {
    double l2_error;           // L2范数误差
    double max_error;          // 最大元素误差
    double relative_error;     // 相对误差
    double fidelity_deviation; // 保真度偏差
};

ErrorMetrics compute_error_metrics(const Vector& reference, const Vector& implementation);

/**
 * Level 0: 对角门参考实现
 */
class DiagonalGates {
public:
    /**
     * 相位旋转门 R(θ)
     * ψ_out[n] = e^(-i θ n) ψ_in[n]
     */
    static Vector apply_phase_rotation(const Vector& input, double theta);

    /**
     * Kerr门 K(χ)
     * ψ_out[n] = e^(-i χ n²) ψ_in[n]
     */
    static Vector apply_kerr_gate(const Vector& input, double chi);

    /**
     * 条件奇偶校验门 CP
     * ψ_out[n] = e^(-i parity π n mod 2) ψ_in[n]
     */
    static Vector apply_conditional_parity(const Vector& input, double parity);
};

/**
 * Level 1: 梯算符门参考实现
 */
class LadderGates {
public:
    /**
     * 光子创建算符 a†
     * ψ_out[n] = √n ψ_in[n-1] (n > 0), ψ_out[0] = 0
     */
    static Vector apply_creation_operator(const Vector& input);

    /**
     * 光子湮灭算符 a
     * ψ_out[n] = √(n+1) ψ_in[n+1] (n < D-1), ψ_out[D-1] = 0
     */
    static Vector apply_annihilation_operator(const Vector& input);
};

/**
 * Level 2: 单模门参考实现
 */
class SingleModeGates {
public:
    /**
     * 位移门 D(α)
     * 使用完整的矩阵表示计算
     */
    static Vector apply_displacement_gate(const Vector& input, Complex alpha);

    /**
     * 挤压门 S(ξ)
     * 使用完整的矩阵表示计算
     */
    static Vector apply_squeezing_gate(const Vector& input, Complex xi);
};

/**
 * Level 3: 双模门参考实现
 */
class TwoModeGates {
public:
    /**
     * 光束分裂器 BS(θ,φ)
     * 使用光子数守恒的分块矩阵计算
     */
    static Vector apply_beam_splitter(const Vector& input, double theta, double phi, int max_photon = 4);
};

/**
 * 工具函数：创建位移门矩阵
 */
Matrix create_displacement_matrix(int dim, Complex alpha);

/**
 * 工具函数：创建挤压门矩阵
 */
Matrix create_squeezing_matrix(int dim, Complex xi);

/**
 * 工具函数：创建光束分裂器矩阵
 */
Matrix create_beam_splitter_matrix(int dim1, int dim2, double theta, double phi);

/**
 * CPU端纯Qubit门参考实现
 */
class QubitGates {
public:
    /**
     * 泡利-X门 (NOT门)
     */
    static Vector apply_pauli_x(const Vector& input);

    /**
     * 泡利-Y门
     */
    static Vector apply_pauli_y(const Vector& input);

    /**
     * 泡利-Z门
     */
    static Vector apply_pauli_z(const Vector& input);

    /**
     * 阿达马门 (Hadamard)
     */
    static Vector apply_hadamard(const Vector& input);

    /**
     * 绕X轴旋转 Rx(θ)
     */
    static Vector apply_rotation_x(const Vector& input, double theta);

    /**
     * 绕Y轴旋转 Ry(θ)
     */
    static Vector apply_rotation_y(const Vector& input, double theta);

    /**
     * 绕Z轴旋转 Rz(θ)
     */
    static Vector apply_rotation_z(const Vector& input, double theta);

    /**
     * S门 (Z^{1/2})
     */
    static Vector apply_phase_s(const Vector& input);

    /**
     * T门 (Z^{1/4})
     */
    static Vector apply_phase_t(const Vector& input);
};

/**
 * Level 4: 混合控制门参考实现
 */
class HybridControlGates {
public:
    /**
     * 受控位移门 CD(α)
     * 当控制qubit为|1⟩时，应用位移门D(α)到目标qumode
     *
     * @param control_state 控制qubit状态 (0或1)
     * @param target_state 目标qumode状态向量
     * @param alpha 位移参数
     * @return 结果状态向量
     */
    static Vector apply_controlled_displacement(int control_state, const Vector& target_state,
                                               Reference::Complex alpha);

    /**
     * 受控挤压门 CS(ξ)
     * 当控制qubit为|1⟩时，应用挤压门S(ξ)到目标qumode
     */
    static Vector apply_controlled_squeezing(int control_state, const Vector& target_state,
                                            Reference::Complex xi);

    /**
     * 受控光束分裂器 CBS(θ,φ)
     */
    static Vector apply_controlled_beam_splitter(int control_state, const Vector& target_state,
                                                 double theta, double phi);

    /**
     * 受控双模挤压 CTMS(ξ)
     */
    static Vector apply_controlled_two_mode_squeezing(int control_state, const Vector& target_state,
                                                     Reference::Complex xi);

    /**
     * 受控SUM门
     */
    static Vector apply_controlled_sum(int control_state, const Vector& target_state,
                                      double theta, double phi);

    /**
     * Rabi振荡门 RB(θ)
     * Qubit-Qumode相互作用: σx ⊗ (a + a†)
     */
    static Vector apply_rabi_interaction(const Vector& qubit_state, const Vector& qumode_state, double theta);

    /**
     * Jaynes-Cummings相互作用 JC(θ,φ)
     * |1⟩⟨0| ⊗ a + |0⟩⟨1| ⊗ a†
     */
    static Vector apply_jaynes_cummings(const Vector& qubit_state, const Vector& qumode_state, double theta, double phi);

    /**
     * Anti-Jaynes-Cummings相互作用 AJC(θ,φ)
     * |0⟩⟨1| ⊗ a + |1⟩⟨0| ⊗ a†
     */
    static Vector apply_anti_jaynes_cummings(const Vector& qubit_state, const Vector& qumode_state, double theta, double phi);

    /**
     * 选择性Qubit旋转 SQR(θ,φ)
     * 根据Qumode光子数选择性旋转Qubit
     */
    static Vector apply_selective_qubit_rotation(const Vector& qubit_state, const Vector& qumode_state,
                                                 const std::vector<double>& theta_vec, const std::vector<double>& phi_vec);

    /**
     * 通用混合控制门
     * 支持任意控制条件和目标操作
     */
    static Vector apply_hybrid_control_gate(int control_state, const Vector& target_state,
                                           const std::string& gate_type,
                                           const std::vector<Reference::Complex>& params);
};

/**
 * 双模门扩展
 */
class TwoModeGatesExtended : public TwoModeGates {
public:
    /**
     * 双模挤压门 TMS(ξ)
     */
    static Vector apply_two_mode_squeezing(const Vector& input, Reference::Complex xi);

    /**
     * SUM门 (求和门)
     */
    static Vector apply_sum_gate(const Vector& input, double theta, double phi);
};

} // namespace Reference
