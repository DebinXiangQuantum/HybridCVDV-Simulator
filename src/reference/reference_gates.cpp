#include "reference_gates.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace Reference {

// ===== 工具函数实现 =====
// 添加缺失的工具函数
Matrix transpose(const Matrix& mat) {
    if (mat.empty()) return {};
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(cols, Vector(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

Matrix kronecker_product(const Matrix& A, const Matrix& B) {
    int a_rows = A.size();
    int a_cols = A[0].size();
    int b_rows = B.size();
    int b_cols = B[0].size();

    Matrix result(a_rows * b_rows, Vector(a_cols * b_cols, Complex(0.0, 0.0)));

    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < a_cols; ++j) {
            for (int k = 0; k < b_rows; ++k) {
                for (int l = 0; l < b_cols; ++l) {
                    result[i * b_rows + k][j * b_cols + l] = A[i][j] * B[k][l];
                }
            }
        }
    }
    return result;
}

Vector tensor_product(const Vector& a, const Vector& b) {
    Vector result(a.size() * b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i * b.size() + j] = a[i] * b[j];
        }
    }
    return result;
}

// ===== 辅助矩阵函数 =====

// 矩阵乘法
Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    int dim = A.size();
    Matrix result(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// 矩阵乘以标量
Matrix matrix_multiply_scalar(const Matrix& matrix, Complex scalar) {
    Matrix result = matrix;
    for (auto& row : result) {
        for (auto& elem : row) {
            elem *= scalar;
        }
    }
    return result;
}

// 矩阵加法 (原地修改)
void matrix_add_to(Matrix& target, const Matrix& source) {
    for (size_t i = 0; i < target.size(); ++i) {
        for (size_t j = 0; j < target[i].size(); ++j) {
            target[i][j] += source[i][j];
        }
    }
}

Matrix create_identity_matrix(int dim) {
    Matrix identity(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    for (int i = 0; i < dim; ++i) {
        identity[i][i] = Complex(1.0, 0.0);
    }
    return identity;
}

Vector matrix_vector_multiply(const Matrix& matrix, const Vector& vector) {
    int dim = matrix.size();
    Vector result(dim, Complex(0.0, 0.0));

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

Complex inner_product(const Vector& v1, const Vector& v2) {
    Complex result(0.0, 0.0);
    for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
        result += std::conj(v1[i]) * v2[i];
    }
    return result;
}

double vector_norm(const Vector& v) {
    double norm_sq = 0.0;
    for (const auto& elem : v) {
        norm_sq += std::norm(elem);
    }
    return std::sqrt(norm_sq);
}

Vector normalize_vector(const Vector& v) {
    double norm = vector_norm(v);
    if (norm < 1e-12) return v;

    Vector result = v;
    for (auto& elem : result) {
        elem /= norm;
    }
    return result;
}

double fidelity(const Vector& v1, const Vector& v2) {
    double norm1 = vector_norm(v1);
    double norm2 = vector_norm(v2);

    // 如果两个向量都接近零，认为它们是相同的（保真度为1）
    if (norm1 < 1e-12 && norm2 < 1e-12) {
        return 1.0;
    }

    // 如果只有一个向量接近零，保真度为0
    if (norm1 < 1e-12 || norm2 < 1e-12) {
        return 0.0;
    }

    Vector n1 = normalize_vector(v1);
    Vector n2 = normalize_vector(v2);

    Complex ip = inner_product(n1, n2);
    return std::norm(ip);
}

ErrorMetrics compute_error_metrics(const Vector& reference, const Vector& implementation) {
    if (reference.size() != implementation.size()) {
        throw std::invalid_argument("向量长度不匹配");
    }

    double l2_error = 0.0;
    double max_error = 0.0;
    double ref_norm_sq = 0.0;

    for (size_t i = 0; i < reference.size(); ++i) {
        Complex diff = reference[i] - implementation[i];
        double abs_error = std::abs(diff);

        l2_error += abs_error * abs_error;
        max_error = std::max(max_error, abs_error);
        ref_norm_sq += std::norm(reference[i]);
    }

    l2_error = std::sqrt(l2_error);
    double relative_error = (ref_norm_sq > 0) ? l2_error / std::sqrt(ref_norm_sq) : 0.0;
    double fid = fidelity(reference, implementation);
    double fidelity_deviation = 1.0 - fid;

    return {l2_error, max_error, relative_error, fidelity_deviation};
}

// ===== Level 0: 对角门实现 =====

Vector DiagonalGates::apply_phase_rotation(const Vector& input, double theta) {
    Vector result = input;
    for (size_t n = 0; n < result.size(); ++n) {
        double phase = -theta * static_cast<double>(n);  // 注意负号
        Complex phase_factor(std::cos(phase), std::sin(phase));
        result[n] *= phase_factor;
    }
    return result;
}

Vector DiagonalGates::apply_kerr_gate(const Vector& input, double chi) {
    Vector result = input;
    for (size_t n = 0; n < result.size(); ++n) {
        double phase = -chi * static_cast<double>(n * n);  // 注意负号
        Complex phase_factor(std::cos(phase), std::sin(phase));
        result[n] *= phase_factor;
    }
    return result;
}

Vector DiagonalGates::apply_conditional_parity(const Vector& input, double parity) {
    Vector result = input;
    for (size_t n = 0; n < result.size(); ++n) {
        double phase = -parity * M_PI * (static_cast<double>(n) - std::floor(static_cast<double>(n) / 2.0) * 2.0);
        Complex phase_factor(std::cos(phase), std::sin(phase));
        result[n] *= phase_factor;
    }
    return result;
}

// ===== Level 1: 梯算符门实现 =====

Vector LadderGates::apply_creation_operator(const Vector& input) {
    Vector result(input.size(), Complex(0.0, 0.0));
    for (size_t n = 1; n < input.size(); ++n) {
        double coeff = std::sqrt(static_cast<double>(n));
        result[n] = coeff * input[n - 1];
    }
    // result[0] 保持为 0
    return result;
}

Vector LadderGates::apply_annihilation_operator(const Vector& input) {
    Vector result(input.size(), Complex(0.0, 0.0));
    for (size_t n = 0; n < input.size() - 1; ++n) {
        double coeff = std::sqrt(static_cast<double>(n + 1));
        result[n] = coeff * input[n + 1];
    }
    // result.back() 保持为 0
    return result;
}

// ===== Level 2: 单模门实现 =====

Vector SingleModeGates::apply_displacement_gate(const Vector& input, Complex alpha) {
    int dim = input.size();
    Matrix displacement_matrix = create_displacement_matrix(dim, alpha);
    return matrix_vector_multiply(displacement_matrix, input);
}

Vector SingleModeGates::apply_squeezing_gate(const Vector& input, Complex xi) {
    int dim = input.size();
    Matrix squeezing_matrix = create_squeezing_matrix(dim, xi);
    return matrix_vector_multiply(squeezing_matrix, input);
}

// ===== Level 3: 双模门实现 =====

Vector TwoModeGates::apply_beam_splitter(const Vector& input, double theta, double phi, int max_photon) {
    int total_dim = input.size();
    // 简化的实现：假设输入是单模状态，扩展到双模
    // 在实际实现中，需要根据具体的状态表示方式处理

    // 这里提供一个简化的实现
    Vector result = input;

    // 计算光束分裂器的简单版本
    // BS = [[cosθ, -sinθ*e^(iφ)], [sinθ*e^(iφ), cosθ]]

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    Complex phase_factor(std::cos(phi), std::sin(phi));

    // 对于真空输入，简单的处理
    if (input.size() >= 2) {
        Complex input0 = input[0];
        Complex input1 = (input.size() > 1) ? input[1] : Complex(0.0, 0.0);

        result[0] = cos_theta * input0 - sin_theta * phase_factor * input1;
        if (result.size() > 1) {
            result[1] = sin_theta * phase_factor * input0 + cos_theta * input1;
        }
    }

    return result;
}

// ===== 矩阵创建函数 =====

// 广义Laguerre多项式计算
double generalized_laguerre(int n, int alpha, double x) {
    if (n < 0 || alpha < 0) return 0.0;
    if (n == 0) return 1.0;

    // 对于测试目的，使用简化的实现
    // L_n^alpha(x) 的近似值
    if (n == 1) return alpha + 1.0 - x;

    // 使用级数展开或近似
    // 对于小的x，我们可以使用前几项
    if (std::abs(x) < 1.0) {
        if (n == 0) return 1.0;
        if (n == 1) return alpha + 1.0 - x;
        if (n == 2) return (alpha + 1.0) * (alpha + 2.0) / 2.0 - (alpha + 3.0) * x / 2.0 + x * x / 2.0;
    }

    // 对于更大的x，使用渐进行为
    return std::exp(x / 2.0) * std::pow(-x, n) / std::tgamma(n + alpha + 1) * std::tgamma(alpha + n + 1);
}

Matrix create_displacement_matrix(int dim, Complex alpha) {
    // 使用矩阵指数方法计算位移算符
    // D(α) = exp(α*a† - conj(α)*a)

    // 创建a和a†矩阵
    Matrix a_matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    Matrix a_dagger_matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));

    for (int i = 0; i < dim - 1; ++i) {
        double coeff = std::sqrt(i + 1.0);
        a_matrix[i][i + 1] = Complex(coeff, 0.0);        // a |i+1⟩ = √(i+1) |i⟩
        a_dagger_matrix[i + 1][i] = Complex(coeff, 0.0);  // a† |i⟩ = √(i+1) |i+1⟩
    }

    // 计算生成元: α*a† - conj(α)*a
    Matrix generator = matrix_multiply_scalar(a_dagger_matrix, alpha);
    Matrix temp = matrix_multiply_scalar(a_matrix, -std::conj(alpha));
    matrix_add_to(generator, temp);

    // 计算矩阵指数: 使用更高的展开阶数以获得更好的精度
    // exp(generator) ≈ I + A + A²/2! + A³/3! + A⁴/4! + A⁵/5! + A⁶/6!
    Matrix result = create_identity_matrix(dim);  // I

    Matrix A_power = generator;  // A^1
    matrix_add_to(result, A_power);  // + A

    A_power = matrix_multiply(A_power, generator);  // A^2
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/2.0));  // + A²/2

    A_power = matrix_multiply(A_power, generator);  // A^3
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/6.0));  // + A³/6

    A_power = matrix_multiply(A_power, generator);  // A^4
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/24.0)); // + A⁴/24

    A_power = matrix_multiply(A_power, generator);  // A^5
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/120.0)); // + A⁵/5!

    A_power = matrix_multiply(A_power, generator);  // A^6
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/720.0)); // + A⁶/6!

    return result;
}

Matrix create_squeezing_matrix(int dim, Complex xi) {
    Matrix matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));

    // 简化的挤压矩阵实现
    // S(ξ) = exp(ξ*a²/2 - ξ*(a†)²/2)

    for (int n = 0; n < dim; ++n) {
        for (int m = 0; m < dim; ++m) {
            // 这里应该计算完整的挤压矩阵元素
            // 暂时使用单位矩阵作为占位符
            if (n == m) {
                matrix[n][m] = Complex(1.0, 0.0);
            }
        }
    }

    return matrix;
}

Matrix create_beam_splitter_matrix(int dim1, int dim2, double theta, double phi) {
    int total_dim = dim1 * dim2;
    Matrix matrix(total_dim, std::vector<Complex>(total_dim, Complex(0.0, 0.0)));

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    Complex phase_factor(std::cos(phi), std::sin(phi));

    // 简化的双模光束分裂器矩阵
    // 在实际实现中，这应该是一个更复杂的块对角矩阵

    for (int n1 = 0; n1 < dim1; ++n1) {
        for (int n2 = 0; n2 < dim2; ++n2) {
            int idx = n1 * dim2 + n2;

            // 对角元素 (n1,n2) -> (n1,n2)
            matrix[idx][idx] = cos_theta;

            // 交叉项 (n1,n2) -> (n1+1,n2-1) 等
            // 这里需要根据光子数守恒实现完整的逻辑
        }
    }

    return matrix;
}

// ===== Level 4: 混合控制门实现 =====

Vector HybridControlGates::apply_controlled_displacement(int control_state, const Vector& target_state,
                                                       Reference::Complex alpha) {
    if (control_state == 0) {
        // 控制位为|0⟩，不应用任何操作
        return target_state;
    } else if (control_state == 1) {
        // 控制位为|1⟩，应用位移门
        return SingleModeGates::apply_displacement_gate(target_state, alpha);
    } else {
        throw std::invalid_argument("控制状态必须是0或1");
    }
}

Vector HybridControlGates::apply_controlled_squeezing(int control_state, const Vector& target_state,
                                                    Reference::Complex xi) {
    if (control_state == 0) {
        // 控制位为|0⟩，不应用任何操作
        return target_state;
    } else if (control_state == 1) {
        // 控制位为|1⟩，应用挤压门
        return SingleModeGates::apply_squeezing_gate(target_state, xi);
    } else {
        throw std::invalid_argument("控制状态必须是0或1");
    }
}

Vector HybridControlGates::apply_hybrid_control_gate(int control_state, const Vector& target_state,
                                                   const std::string& gate_type,
                                                   const std::vector<Reference::Complex>& params) {
    if (gate_type == "controlled_displacement") {
        if (params.size() < 1) {
            throw std::invalid_argument("受控位移门需要1个参数");
        }
        return apply_controlled_displacement(control_state, target_state, params[0]);
    } else if (gate_type == "controlled_squeezing") {
        if (params.size() < 1) {
            throw std::invalid_argument("受控挤压门需要1个参数");
        }
        return apply_controlled_squeezing(control_state, target_state, params[0]);
    } else {
        throw std::invalid_argument("不支持的混合控制门类型: " + gate_type);
    }
}

// ===== Qubit门参考实现 =====

Vector QubitGates::apply_pauli_x(const Vector& input) {
    // X = [[0, 1], [1, 0]]
    if (input.size() != 2) {
        throw std::invalid_argument("Pauli-X门需要2维状态向量");
    }
    return {input[1], input[0]};
}

Vector QubitGates::apply_pauli_y(const Vector& input) {
    // Y = [[0, -i], [i, 0]]
    if (input.size() != 2) {
        throw std::invalid_argument("Pauli-Y门需要2维状态向量");
    }
    return {Complex(0.0, -1.0) * input[1], Complex(0.0, 1.0) * input[0]};
}

Vector QubitGates::apply_pauli_z(const Vector& input) {
    // Z = [[1, 0], [0, -1]]
    if (input.size() != 2) {
        throw std::invalid_argument("Pauli-Z门需要2维状态向量");
    }
    return {input[0], -input[1]};
}

Vector QubitGates::apply_hadamard(const Vector& input) {
    // H = 1/√2 * [[1, 1], [1, -1]]
    if (input.size() != 2) {
        throw std::invalid_argument("Hadamard门需要2维状态向量");
    }
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    return {inv_sqrt2 * (input[0] + input[1]), inv_sqrt2 * (input[0] - input[1])};
}

Vector QubitGates::apply_rotation_x(const Vector& input, double theta) {
    // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
    if (input.size() != 2) {
        throw std::invalid_argument("Rx门需要2维状态向量");
    }
    double cos_half = std::cos(theta / 2.0);
    double sin_half = std::sin(theta / 2.0);
    Complex i_sin = Complex(0.0, -sin_half);
    return {
        cos_half * input[0] + i_sin * input[1],
        i_sin * input[0] + cos_half * input[1]
    };
}

Vector QubitGates::apply_rotation_y(const Vector& input, double theta) {
    // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    if (input.size() != 2) {
        throw std::invalid_argument("Ry门需要2维状态向量");
    }
    double cos_half = std::cos(theta / 2.0);
    double sin_half = std::sin(theta / 2.0);
    return {
        cos_half * input[0] - sin_half * input[1],
        sin_half * input[0] + cos_half * input[1]
    };
}

Vector QubitGates::apply_rotation_z(const Vector& input, double theta) {
    // Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
    if (input.size() != 2) {
        throw std::invalid_argument("Rz门需要2维状态向量");
    }
    Complex phase_neg = std::exp(Complex(0.0, -theta / 2.0));
    Complex phase_pos = std::exp(Complex(0.0, theta / 2.0));
    return {phase_neg * input[0], phase_pos * input[1]};
}

Vector QubitGates::apply_phase_s(const Vector& input) {
    // S = [[1, 0], [0, i]]
    if (input.size() != 2) {
        throw std::invalid_argument("S门需要2维状态向量");
    }
    return {input[0], Complex(0.0, 1.0) * input[1]};
}

Vector QubitGates::apply_phase_t(const Vector& input) {
    // T = [[1, 0], [0, e^(iπ/4)]]
    if (input.size() != 2) {
        throw std::invalid_argument("T门需要2维状态向量");
    }
    Complex phase = std::exp(Complex(0.0, M_PI / 4.0));
    return {input[0], phase * input[1]};
}

// ===== 扩展的混合控制门实现 =====

Vector HybridControlGates::apply_controlled_beam_splitter(int control_state, const Vector& target_state,
                                                         double theta, double phi) {
    // CBS(θ,φ) = exp[-i θ/2 σ_z ⊗ (e^{iφ} a† b + e^{-iφ} a b†)]
    // 这是一个复杂的双模门，暂时使用简化的实现
    return target_state; // 占位符
}

Vector HybridControlGates::apply_controlled_two_mode_squeezing(int control_state, const Vector& target_state,
                                                             Complex xi) {
    // CTMS(ξ) = exp[1/2 σ_z ⊗ (ξ* a† b† - ξ* a b)]
    // 这是一个复杂的双模门，暂时使用简化的实现
    return target_state; // 占位符
}

Vector HybridControlGates::apply_controlled_sum(int control_state, const Vector& target_state,
                                              double theta, double phi) {
    // SUM门定义较为复杂，暂时使用简化的实现
    return target_state; // 占位符
}

Vector HybridControlGates::apply_rabi_interaction(const Vector& qubit_state, const Vector& qumode_state, double theta) {
    // RB(θ) = exp[-i σ_x ⊗ (θ a† + θ* a)]
    // 这是一个复杂的混合门，暂时使用简化的实现
    return tensor_product(qubit_state, qumode_state); // 占位符
}

Vector HybridControlGates::apply_jaynes_cummings(const Vector& qubit_state, const Vector& qumode_state, double theta, double phi) {
    // JC(θ,φ) = exp[-iθ(e^{iφ} σ- a† + e^{-iφ} σ+ a)]
    // 这是一个复杂的混合门，暂时使用简化的实现
    return tensor_product(qubit_state, qumode_state); // 占位符
}

Vector HybridControlGates::apply_anti_jaynes_cummings(const Vector& qubit_state, const Vector& qumode_state, double theta, double phi) {
    // AJC(θ,φ) = exp[-iθ(e^{iφ} σ+ a† + e^{-iφ} σ- a)]
    // 这是一个复杂的混合门，暂时使用简化的实现
    return tensor_product(qubit_state, qumode_state); // 占位符
}

Vector HybridControlGates::apply_selective_qubit_rotation(const Vector& qubit_state, const Vector& qumode_state,
                                                         const std::vector<double>& theta_vec, const std::vector<double>& phi_vec) {
    // SQR根据光子数选择性旋转Qubit
    // 这是一个非常复杂的门，暂时使用简化的实现
    return tensor_product(qubit_state, qumode_state); // 占位符
}

// ===== 双模门扩展实现 =====

Vector TwoModeGatesExtended::apply_two_mode_squeezing(const Vector& input, Complex xi) {
    // TMS(ξ) = exp[1/2 (ξ* a† b† - ξ a b)]
    // 暂时使用简化的实现
    return input; // 占位符
}

Vector TwoModeGatesExtended::apply_sum_gate(const Vector& input, double theta, double phi) {
    // SUM门定义较为复杂，暂时使用简化的实现
    return input; // 占位符
}

} // namespace Reference
