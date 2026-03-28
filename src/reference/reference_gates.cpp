#include "reference_gates.h"
#include "squeezing_matrix.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace Reference {

namespace {

void build_beam_splitter_tensor(std::vector<Complex>& tensor,
                                int cutoff,
                                double theta,
                                double phi) {
    tensor.assign(static_cast<size_t>(cutoff) * cutoff * cutoff * cutoff, Complex(0.0, 0.0));

    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const Complex phase(std::cos(phi), std::sin(phi));

    std::vector<double> sqrt_table(cutoff);
    for (int i = 0; i < cutoff; ++i) {
        sqrt_table[i] = std::sqrt(static_cast<double>(i));
    }

    tensor[0] = Complex(1.0, 0.0);

    for (int m = 0; m < cutoff; ++m) {
        for (int n = 0; n < cutoff - m; ++n) {
            const int p = m + n;
            if (p <= 0 || p >= cutoff) {
                continue;
            }

            const size_t idx =
                static_cast<size_t>(m) * cutoff * cutoff * cutoff +
                static_cast<size_t>(n) * cutoff * cutoff +
                static_cast<size_t>(p) * cutoff;
            Complex sum(0.0, 0.0);

            if (m > 0) {
                const size_t idx1 =
                    static_cast<size_t>(m - 1) * cutoff * cutoff * cutoff +
                    static_cast<size_t>(n) * cutoff * cutoff +
                    static_cast<size_t>(p - 1) * cutoff;
                sum += (ct * sqrt_table[m] / sqrt_table[p]) * tensor[idx1];
            }

            if (n > 0) {
                const size_t idx2 =
                    static_cast<size_t>(m) * cutoff * cutoff * cutoff +
                    static_cast<size_t>(n - 1) * cutoff * cutoff +
                    static_cast<size_t>(p - 1) * cutoff;
                sum += phase * (st * sqrt_table[n] / sqrt_table[p]) * tensor[idx2];
            }

            tensor[idx] = sum;
        }
    }

    for (int m = 0; m < cutoff; ++m) {
        for (int n = 0; n < cutoff; ++n) {
            for (int p = 0; p < cutoff; ++p) {
                const int q = m + n - p;
                if (q <= 0 || q >= cutoff) {
                    continue;
                }

                const size_t idx =
                    static_cast<size_t>(m) * cutoff * cutoff * cutoff +
                    static_cast<size_t>(n) * cutoff * cutoff +
                    static_cast<size_t>(p) * cutoff +
                    static_cast<size_t>(q);
                Complex sum(0.0, 0.0);

                if (m > 0) {
                    const size_t idx1 =
                        static_cast<size_t>(m - 1) * cutoff * cutoff * cutoff +
                        static_cast<size_t>(n) * cutoff * cutoff +
                        static_cast<size_t>(p) * cutoff +
                        static_cast<size_t>(q - 1);
                    sum += (-st * sqrt_table[m] / sqrt_table[q]) * std::conj(phase) * tensor[idx1];
                }

                if (n > 0) {
                    const size_t idx2 =
                        static_cast<size_t>(m) * cutoff * cutoff * cutoff +
                        static_cast<size_t>(n - 1) * cutoff * cutoff +
                        static_cast<size_t>(p) * cutoff +
                        static_cast<size_t>(q - 1);
                    sum += (ct * sqrt_table[n] / sqrt_table[q]) * tensor[idx2];
                }

                tensor[idx] = sum;
            }
        }
    }
}

double matrix_max_abs(const Matrix& matrix) {
    double max_abs = 0.0;
    for (const auto& row : matrix) {
        for (const auto& value : row) {
            max_abs = std::max(max_abs, std::abs(value));
        }
    }
    return max_abs;
}

double matrix_one_norm(const Matrix& matrix) {
    if (matrix.empty()) {
        return 0.0;
    }

    const int dim = static_cast<int>(matrix.size());
    double max_sum = 0.0;
    for (int col = 0; col < dim; ++col) {
        double sum = 0.0;
        for (int row = 0; row < dim; ++row) {
            sum += std::abs(matrix[row][col]);
        }
        max_sum = std::max(max_sum, sum);
    }
    return max_sum;
}

Matrix scale_matrix(const Matrix& matrix, double scale) {
    Matrix result = matrix;
    for (auto& row : result) {
        for (auto& value : row) {
            value *= scale;
        }
    }
    return result;
}

Matrix multiply_matrices(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.empty()) {
        return {};
    }

    const int dim = static_cast<int>(lhs.size());
    Matrix result(dim, Vector(dim, Complex(0.0, 0.0)));
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            Complex sum(0.0, 0.0);
            for (int k = 0; k < dim; ++k) {
                sum += lhs[row][k] * rhs[k][col];
            }
            result[row][col] = sum;
        }
    }
    return result;
}

void add_matrix_inplace(Matrix& target, const Matrix& source) {
    for (size_t row = 0; row < target.size(); ++row) {
        for (size_t col = 0; col < target[row].size(); ++col) {
            target[row][col] += source[row][col];
        }
    }
}

Matrix matrix_exponential(const Matrix& matrix) {
    if (matrix.empty()) {
        return {};
    }

    const int dim = static_cast<int>(matrix.size());
    const double norm = matrix_one_norm(matrix);
    const int scaling_power = norm > 1.0 ? static_cast<int>(std::ceil(std::log2(norm))) : 0;
    const double scale = std::ldexp(1.0, scaling_power);
    const Matrix scaled_matrix = scale_matrix(matrix, 1.0 / scale);

    Matrix result(dim, Vector(dim, Complex(0.0, 0.0)));
    Matrix term(dim, Vector(dim, Complex(0.0, 0.0)));
    for (int i = 0; i < dim; ++i) {
        result[i][i] = Complex(1.0, 0.0);
        term[i][i] = Complex(1.0, 0.0);
    }

    for (int order = 1; order <= 80; ++order) {
        term = multiply_matrices(term, scaled_matrix);
        const double inv_order = 1.0 / static_cast<double>(order);
        for (auto& row : term) {
            for (auto& value : row) {
                value *= inv_order;
            }
        }
        add_matrix_inplace(result, term);
        if (matrix_max_abs(term) < 1e-14) {
            break;
        }
    }

    for (int i = 0; i < scaling_power; ++i) {
        result = multiply_matrices(result, result);
    }
    return result;
}

int infer_two_mode_cutoff(const Vector& input) {
    const int cutoff = static_cast<int>(std::lround(std::sqrt(static_cast<double>(input.size()))));
    if (cutoff <= 0 || cutoff * cutoff != static_cast<int>(input.size())) {
        throw std::invalid_argument("双模门需要长度为 D^2 的状态向量");
    }
    return cutoff;
}

Matrix build_two_mode_squeezing_matrix(int cutoff, Complex xi) {
    const int dim = cutoff * cutoff;
    Matrix generator(dim, Vector(dim, Complex(0.0, 0.0)));

    for (int p = 0; p < cutoff; ++p) {
        for (int q = 0; q < cutoff; ++q) {
            const int input_idx = p * cutoff + q;

            if (p + 1 < cutoff && q + 1 < cutoff) {
                const int output_idx = (p + 1) * cutoff + (q + 1);
                const double coeff = 0.5 * std::sqrt(static_cast<double>((p + 1) * (q + 1)));
                generator[output_idx][input_idx] += std::conj(xi) * coeff;
            }

            if (p > 0 && q > 0) {
                const int output_idx = (p - 1) * cutoff + (q - 1);
                const double coeff = -0.5 * std::sqrt(static_cast<double>(p * q));
                generator[output_idx][input_idx] += xi * coeff;
            }
        }
    }

    return matrix_exponential(generator);
}

Matrix build_sum_matrix(int cutoff, double theta) {
    const int dim = cutoff * cutoff;
    Matrix generator(dim, Vector(dim, Complex(0.0, 0.0)));
    const double prefactor = 0.5 * theta;

    for (int p = 0; p < cutoff; ++p) {
        for (int q = 0; q < cutoff; ++q) {
            const int input_idx = p * cutoff + q;

            if (p > 0 && q + 1 < cutoff) {
                const int output_idx = (p - 1) * cutoff + (q + 1);
                generator[output_idx][input_idx] +=
                    prefactor * std::sqrt(static_cast<double>(p * (q + 1)));
            }

            if (p > 0 && q > 0) {
                const int output_idx = (p - 1) * cutoff + (q - 1);
                generator[output_idx][input_idx] +=
                    -prefactor * std::sqrt(static_cast<double>(p * q));
            }

            if (p + 1 < cutoff && q + 1 < cutoff) {
                const int output_idx = (p + 1) * cutoff + (q + 1);
                generator[output_idx][input_idx] +=
                    prefactor * std::sqrt(static_cast<double>((p + 1) * (q + 1)));
            }

            if (p + 1 < cutoff && q > 0) {
                const int output_idx = (p + 1) * cutoff + (q - 1);
                generator[output_idx][input_idx] +=
                    -prefactor * std::sqrt(static_cast<double>((p + 1) * q));
            }
        }
    }

    return matrix_exponential(generator);
}

}  // namespace

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
        double phase = chi * static_cast<double>(n * n);
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
    (void)max_photon;
    const int total_dim = static_cast<int>(input.size());
    const int single_mode_dim = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total_dim))));
    if (single_mode_dim * single_mode_dim != total_dim) {
        throw std::invalid_argument("Beam splitter reference expects a square two-mode state dimension");
    }

    Matrix beam_splitter_matrix = create_beam_splitter_matrix(single_mode_dim, single_mode_dim, theta, phi);
    return matrix_vector_multiply(beam_splitter_matrix, input);
}

// ===== 矩阵创建函数 =====

// 广义Laguerre多项式计算 — 标准三项递推关系
// L_0^α(x) = 1
// L_1^α(x) = α + 1 - x
// n·L_n^α(x) = (2n + α - 1 - x)·L_{n-1}^α(x) - (n + α - 1)·L_{n-2}^α(x)
double generalized_laguerre(int n, int alpha, double x) {
    if (n < 0) return 0.0;
    if (n == 0) return 1.0;

    double L_prev2 = 1.0;                              // L_0^α(x)
    double L_prev1 = static_cast<double>(alpha) + 1.0 - x;  // L_1^α(x)
    if (n == 1) return L_prev1;

    for (int k = 2; k <= n; ++k) {
        double L_curr = ((2.0 * k + alpha - 1.0 - x) * L_prev1
                         - (k + alpha - 1.0) * L_prev2) / static_cast<double>(k);
        L_prev2 = L_prev1;
        L_prev1 = L_curr;
    }
    return L_prev1;
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
    // exp(generator) ≈ I + A + A²/2! + A³/3! + A⁴/4! + A⁵/5! + A⁶/6! + A⁷/7! + A⁸/8! + A⁹/9! + A¹⁰/10!
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

    A_power = matrix_multiply(A_power, generator);  // A^7
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/5040.0)); // + A⁷/7!

    A_power = matrix_multiply(A_power, generator);  // A^8
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/40320.0)); // + A⁸/8!

    A_power = matrix_multiply(A_power, generator);  // A^9
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/362880.0)); // + A⁹/9!

    A_power = matrix_multiply(A_power, generator);  // A^10
    matrix_add_to(result, matrix_multiply_scalar(A_power, 1.0/3628800.0)); // + A¹⁰/10!

    return result;
}

Matrix create_squeezing_matrix(int dim, Complex xi) {
    Matrix matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    const std::vector<std::complex<double>> dense =
        generate_squeezing_matrix(std::abs(xi), std::arg(xi), dim);

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            matrix[row][col] = dense[static_cast<size_t>(row) * dim + static_cast<size_t>(col)];
        }
    }

    return matrix;
}

Matrix create_beam_splitter_matrix(int dim1, int dim2, double theta, double phi) {
    const int total_dim = dim1 * dim2;
    const int cutoff = std::max(dim1, dim2);
    Matrix matrix(total_dim, std::vector<Complex>(total_dim, Complex(0.0, 0.0)));
    std::vector<Complex> tensor;
    build_beam_splitter_tensor(tensor, cutoff, theta, phi);

    for (int m = 0; m < dim1; ++m) {
        for (int n = 0; n < dim2; ++n) {
            const int row = m * dim2 + n;
            for (int p = 0; p < dim1; ++p) {
                for (int q = 0; q < dim2; ++q) {
                    const int col = p * dim2 + q;
                    const size_t tensor_idx =
                        static_cast<size_t>(m) * cutoff * cutoff * cutoff +
                        static_cast<size_t>(n) * cutoff * cutoff +
                        static_cast<size_t>(p) * cutoff +
                        static_cast<size_t>(q);
                    matrix[row][col] = tensor[tensor_idx];
                }
            }
        }
    }

    return matrix;
}

// ===== Level 4: 混合控制门实现 =====

Vector HybridControlGates::apply_controlled_displacement(int control_state, const Vector& target_state,
                                                       Reference::Complex alpha) {
    if (control_state == 0) {
        return SingleModeGates::apply_displacement_gate(target_state, alpha);
    } else if (control_state == 1) {
        return SingleModeGates::apply_displacement_gate(target_state, -alpha);
    } else {
        throw std::invalid_argument("控制状态必须是0或1");
    }
}

Vector HybridControlGates::apply_controlled_squeezing(int control_state, const Vector& target_state,
                                                    Reference::Complex xi) {
    if (control_state == 0) {
        return SingleModeGates::apply_squeezing_gate(target_state, xi);
    } else if (control_state == 1) {
        return SingleModeGates::apply_squeezing_gate(target_state, -xi);
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

namespace {

std::pair<Vector, Vector> split_hybrid_state(const Vector& hybrid_state) {
    if (hybrid_state.size() % 2 != 0) {
        throw std::invalid_argument("混合态向量长度必须为偶数");
    }

    const size_t cutoff = hybrid_state.size() / 2;
    Vector branch0(cutoff, Complex(0.0, 0.0));
    Vector branch1(cutoff, Complex(0.0, 0.0));
    for (size_t n = 0; n < cutoff; ++n) {
        branch0[n] = hybrid_state[n];
        branch1[n] = hybrid_state[cutoff + n];
    }
    return {branch0, branch1};
}

Vector combine_hybrid_state(const Vector& branch0, const Vector& branch1) {
    if (branch0.size() != branch1.size()) {
        throw std::invalid_argument("混合态分支长度不匹配");
    }

    Vector hybrid_state(branch0.size() * 2, Complex(0.0, 0.0));
    for (size_t n = 0; n < branch0.size(); ++n) {
        hybrid_state[n] = branch0[n];
        hybrid_state[branch0.size() + n] = branch1[n];
    }
    return hybrid_state;
}

Vector make_hybrid_product_state(const Vector& qubit_state, const Vector& qumode_state) {
    if (qubit_state.size() != 2) {
        throw std::invalid_argument("混合门参考实现要求2维qubit状态");
    }
    return tensor_product(qubit_state, qumode_state);
}

}  // namespace

Vector HybridControlGates::apply_controlled_beam_splitter(int control_state, const Vector& target_state,
                                                         double theta, double phi) {
    return TwoModeGates::apply_beam_splitter(
        target_state, control_state == 0 ? theta : -theta, phi);
}

Vector HybridControlGates::apply_controlled_two_mode_squeezing(int control_state, const Vector& target_state,
                                                             Complex xi) {
    if (control_state == 0) {
        return TwoModeGatesExtended::apply_two_mode_squeezing(target_state, xi);
    }
    return TwoModeGatesExtended::apply_two_mode_squeezing(target_state, -xi);
}

Vector HybridControlGates::apply_controlled_sum(int control_state, const Vector& target_state,
                                              double theta, double phi) {
    if (control_state == 0) {
        return TwoModeGatesExtended::apply_sum_gate(target_state, theta, phi);
    }
    return TwoModeGatesExtended::apply_sum_gate(target_state, -theta, phi);
}

Vector HybridControlGates::apply_rabi_interaction(const Vector& qubit_state, const Vector& qumode_state, double theta) {
    Vector hybrid_state = make_hybrid_product_state(qubit_state, qumode_state);
    auto branches = split_hybrid_state(hybrid_state);
    Vector plus(branches.first.size(), Complex(0.0, 0.0));
    Vector minus(branches.first.size(), Complex(0.0, 0.0));
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (size_t n = 0; n < branches.first.size(); ++n) {
        plus[n] = (branches.first[n] + branches.second[n]) * inv_sqrt2;
        minus[n] = (branches.first[n] - branches.second[n]) * inv_sqrt2;
    }

    plus = SingleModeGates::apply_displacement_gate(plus, Complex(0.0, -theta));
    minus = SingleModeGates::apply_displacement_gate(minus, Complex(0.0, theta));

    Vector out0(branches.first.size(), Complex(0.0, 0.0));
    Vector out1(branches.first.size(), Complex(0.0, 0.0));
    for (size_t n = 0; n < branches.first.size(); ++n) {
        out0[n] = (plus[n] + minus[n]) * inv_sqrt2;
        out1[n] = (plus[n] - minus[n]) * inv_sqrt2;
    }

    return combine_hybrid_state(out0, out1);
}

Vector HybridControlGates::apply_jaynes_cummings(const Vector& qubit_state, const Vector& qumode_state, double theta, double phi) {
    Vector hybrid_state = make_hybrid_product_state(qubit_state, qumode_state);
    auto branches = split_hybrid_state(hybrid_state);
    Vector out0 = branches.first;
    Vector out1 = branches.second;

    for (size_t n = 0; n + 1 < branches.first.size(); ++n) {
        const double omega = theta * std::sqrt(static_cast<double>(n + 1));
        const double cos_w = std::cos(omega);
        const double sin_w = std::sin(omega);
        const Complex factor01(-std::sin(phi) * sin_w, -std::cos(phi) * sin_w);
        const Complex factor10(std::sin(phi) * sin_w, -std::cos(phi) * sin_w);

        const Complex c0 = branches.first[n + 1];
        const Complex c1 = branches.second[n];
        out0[n + 1] = cos_w * c0 + factor01 * c1;
        out1[n] = factor10 * c0 + cos_w * c1;
    }

    return combine_hybrid_state(out0, out1);
}

Vector HybridControlGates::apply_anti_jaynes_cummings(const Vector& qubit_state, const Vector& qumode_state, double theta, double phi) {
    Vector hybrid_state = make_hybrid_product_state(qubit_state, qumode_state);
    auto branches = split_hybrid_state(hybrid_state);
    Vector out0 = branches.first;
    Vector out1 = branches.second;

    for (size_t n = 0; n + 1 < branches.first.size(); ++n) {
        const double omega = theta * std::sqrt(static_cast<double>(n + 1));
        const double cos_w = std::cos(omega);
        const double sin_w = std::sin(omega);
        const Complex factor01(-std::sin(phi) * sin_w, -std::cos(phi) * sin_w);
        const Complex factor10(std::sin(phi) * sin_w, -std::cos(phi) * sin_w);

        const Complex c0 = branches.first[n];
        const Complex c1 = branches.second[n + 1];
        out0[n] = cos_w * c0 + factor01 * c1;
        out1[n + 1] = factor10 * c0 + cos_w * c1;
    }

    return combine_hybrid_state(out0, out1);
}

Vector HybridControlGates::apply_selective_qubit_rotation(const Vector& qubit_state, const Vector& qumode_state,
                                                         const std::vector<double>& theta_vec, const std::vector<double>& phi_vec) {
    Vector hybrid_state = make_hybrid_product_state(qubit_state, qumode_state);
    auto branches = split_hybrid_state(hybrid_state);
    Vector out0 = branches.first;
    Vector out1 = branches.second;

    for (size_t n = 0; n < branches.first.size(); ++n) {
        const double theta = n < theta_vec.size() ? theta_vec[n] : 0.0;
        const double phi = n < phi_vec.size() ? phi_vec[n] : 0.0;
        const double cos_t = std::cos(theta / 2.0);
        const double sin_t = std::sin(theta / 2.0);
        const Complex alpha(cos_t, 0.0);
        const Complex beta(-std::cos(phi) * sin_t, std::sin(phi) * sin_t);
        const Complex c0 = branches.first[n];
        const Complex c1 = branches.second[n];

        out0[n] = alpha * c0 + beta * c1;
        out1[n] = -std::conj(beta) * c0 + std::conj(alpha) * c1;
    }

    return combine_hybrid_state(out0, out1);
}

// ===== 双模门扩展实现 =====

Vector TwoModeGatesExtended::apply_two_mode_squeezing(const Vector& input, Complex xi) {
    const int cutoff = infer_two_mode_cutoff(input);
    return matrix_vector_multiply(build_two_mode_squeezing_matrix(cutoff, xi), input);
}

Vector TwoModeGatesExtended::apply_sum_gate(const Vector& input, double theta, double phi) {
    if (std::abs(phi) > 1e-14) {
        throw std::invalid_argument("当前SUM参考实现仅支持 phi = 0");
    }
    const int cutoff = infer_two_mode_cutoff(input);
    return matrix_vector_multiply(build_sum_matrix(cutoff, theta), input);
}

} // namespace Reference
