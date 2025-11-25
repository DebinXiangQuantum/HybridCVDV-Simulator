#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <memory>
#include <string>

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
    static Vector apply_beam_splitter(const Vector& input, double theta, double phi);
};

/**
 * Level 4: 混合控制门参考实现 (Separable)
 */
class HybridControlGates {
public:
    /**
     * 受控位移门 CD(α)
     * CD(alpha) = exp[sigma_z (alpha a^dag - alpha^* a)]
     * Q=0 (sigma_z=1): D(alpha)
     * Q=1 (sigma_z=-1): D(-alpha)
     */
    static void apply_conditional_displacement(Vector& v0, Vector& v1, Reference::Complex alpha);

    /**
     * 受控挤压门 CS(ζ)
     * CS(zeta) = exp[1/2 sigma_z (zeta^* a^2 - zeta (a^dag)^2)]
     * Q=0: S(zeta)
     * Q=1: S(-zeta)
     */
    static void apply_conditional_squeezing(Vector& v0, Vector& v1, Reference::Complex zeta);
    
    /**
     * 受控旋转门 CR(theta)
     * CR(theta) = exp[-i theta/2 sigma_z n]
     * Q=0: exp[-i theta/2 n]
     * Q=1: exp[+i theta/2 n]
     */
    static void apply_conditional_rotation(Vector& v0, Vector& v1, double theta);

    /**
     * 受控分束器 CBS(theta, phi)
     * CBS = exp[-i theta/2 sigma_z (BS_generator)]
     * Q=0: BS(theta, phi)
     * Q=1: BS(-theta, phi) = BS(theta, phi)^dagger
     */
    static void apply_conditional_beam_splitter(Vector& v0, Vector& v1, double theta, double phi);
};

/**
 * 相互作用门 (Mixing Gates)
 */
class InteractionGates {
public:
    /**
     * Rabi Interaction
     * RB(theta) = exp[-i sigma_x (theta a^dag + theta^* a)]
     */
    static void apply_rabi(Vector& v0, Vector& v1, double theta);

    /**
     * Jaynes-Cummings
     * JC(theta, phi) = exp[-i theta (e^{i\phi} \sigma_- a^\dagger + e^{-i\phi} \sigma_+ a)]
     */
    static void apply_jaynes_cummings(Vector& v0, Vector& v1, double theta, double phi);

    /**
     * Anti-Jaynes-Cummings
     * AJC(theta, phi) = exp[-i theta (e^{i\phi} \sigma_+ a^\dagger + e^{-i\phi} \sigma_- a)]
     */
    static void apply_anti_jaynes_cummings(Vector& v0, Vector& v1, double theta, double phi);
};

/**
 * 特殊映射门
 */
class SpecialGates {
public:
    /**
     * SQR (Selective Qubit Rotation)
     * SQR(vec_theta, vec_phi) = sum_n R_{phi_n}(theta_n) tensor |n><n|
     */
    static void apply_sqr(Vector& v0, Vector& v1, const std::vector<double>& thetas, const std::vector<double>& phis);
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

} // namespace Reference
