#include "reference_gates.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace Reference {

// ===== 工具函数实现 =====

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

// ===== 矩阵指数辅助函数 =====
Matrix matrix_exponentiate(const Matrix& generator, int order = 8) {
    int dim = generator.size();
    Matrix result = create_identity_matrix(dim);
    Matrix power = result; // A^0 = I
    
    // exp(A) = sum A^k / k!
    double factorial = 1.0;
    for (int k = 1; k <= order; ++k) {
        factorial *= k;
        power = matrix_multiply(power, generator);
        Matrix term = matrix_multiply_scalar(power, 1.0 / factorial);
        matrix_add_to(result, term);
    }
    return result;
}

// ===== Level 0: 对角门实现 =====

Vector DiagonalGates::apply_phase_rotation(const Vector& input, double theta) {
    Vector result = input;
    for (size_t n = 0; n < result.size(); ++n) {
        // R(theta) = exp[-i theta n]
        double phase = -theta * static_cast<double>(n);
        Complex phase_factor(std::cos(phase), std::sin(phase));
        result[n] *= phase_factor;
    }
    return result;
}

Vector DiagonalGates::apply_kerr_gate(const Vector& input, double chi) {
    Vector result = input;
    for (size_t n = 0; n < result.size(); ++n) {
        double phase = -chi * static_cast<double>(n * n);
        Complex phase_factor(std::cos(phase), std::sin(phase));
        result[n] *= phase_factor;
    }
    return result;
}

Vector DiagonalGates::apply_conditional_parity(const Vector& input, double parity) {
    Vector result = input;
    for (size_t n = 0; n < result.size(); ++n) {
        // CP = exp[-i pi n] for parity=1?
        // definition in gatesmath: CP = CR(pi) = exp[-i pi/2 n] for sigma_z terms...
        // But stand-alone CP usually means Parity (-1)^n.
        // gatesmath: "CP = CR(pi)" where CR(theta) = exp[-i theta/2 sigma_z n].
        // If this is stand-alone P (not conditional on qubit but just Parity op): P = exp[i pi n] = (-1)^n.
        // The existing implementation used some complex formula. I will stick to P = (-1)^n.
        double phase = parity * M_PI * static_cast<double>(n); 
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
    return result;
}

Vector LadderGates::apply_annihilation_operator(const Vector& input) {
    Vector result(input.size(), Complex(0.0, 0.0));
    for (size_t n = 0; n < input.size() - 1; ++n) {
        double coeff = std::sqrt(static_cast<double>(n + 1));
        result[n] = coeff * input[n + 1];
    }
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

Vector TwoModeGates::apply_beam_splitter(const Vector& input, double theta, double phi) {
    // Determine D from input size (assumed D^2)
    int total_dim = input.size();
    int dim = static_cast<int>(std::sqrt(total_dim));
    if (dim * dim != total_dim) {
        throw std::invalid_argument("Input vector size must be square number (D^2) for two-mode gate");
    }
    
    Matrix bs_matrix = create_beam_splitter_matrix(dim, dim, theta, phi);
    return matrix_vector_multiply(bs_matrix, input);
}

// ===== Level 4: 混合控制门实现 =====

void HybridControlGates::apply_conditional_displacement(Vector& v0, Vector& v1, Reference::Complex alpha) {
    // Branch 0: D(+alpha)
    v0 = SingleModeGates::apply_displacement_gate(v0, alpha);
    
    // Branch 1: D(-alpha)
    v1 = SingleModeGates::apply_displacement_gate(v1, -alpha);
}

void HybridControlGates::apply_conditional_squeezing(Vector& v0, Vector& v1, Reference::Complex zeta) {
    // Branch 0: S(+zeta)
    v0 = SingleModeGates::apply_squeezing_gate(v0, zeta);
    
    // Branch 1: S(-zeta)
    v1 = SingleModeGates::apply_squeezing_gate(v1, -zeta);
}

void HybridControlGates::apply_conditional_rotation(Vector& v0, Vector& v1, double theta) {
    // CR(theta) = exp[-i theta/2 sigma_z n]
    // Q=0: sigma_z = 1 => exp[-i theta/2 n]
    v0 = DiagonalGates::apply_phase_rotation(v0, theta / 2.0);
    
    // Q=1: sigma_z = -1 => exp[+i theta/2 n] = exp[-i (-theta/2) n]
    v1 = DiagonalGates::apply_phase_rotation(v1, -theta / 2.0);
}

void HybridControlGates::apply_conditional_beam_splitter(Vector& v0, Vector& v1, double theta, double phi) {
    // CBS(theta, phi)
    // Q=0: BS(theta, phi)
    v0 = TwoModeGates::apply_beam_splitter(v0, theta, phi);
    
    // Q=1: BS(-theta, phi)
    v1 = TwoModeGates::apply_beam_splitter(v1, -theta, phi);
}

// ===== Interaction Gates =====

void InteractionGates::apply_rabi(Vector& v0, Vector& v1, double theta) {
    // RB(theta) = exp[-i sigma_x (theta a^dag + theta^* a)]
    // Let H_disp = theta a^dag - (-theta^*) a. 
    // Wait, displacement generator is alpha a^dag - alpha^* a.
    // Here generator is G = theta a^dag + theta^* a.
    // If we set alpha = theta, D(theta) has generator theta a^dag - theta^* a.
    // If theta is real, G = theta(a^dag + a). D(i theta) = exp[i theta a^dag + i theta a] = exp[i G].
    // So G = -i ln(D(i theta)).
    // We need exp[-i sigma_x G] = cos(G) I - i sin(G) sigma_x.
    
    // Since Reference implementation can be inefficient, let's just compute G explicitly.
    int dim = v0.size();
    Matrix a_matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    Matrix a_dagger_matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    for (int i = 0; i < dim - 1; ++i) {
        double coeff = std::sqrt(i + 1.0);
        a_matrix[i][i + 1] = Complex(coeff, 0.0);
        a_dagger_matrix[i + 1][i] = Complex(coeff, 0.0);
    }
    
    Matrix G = matrix_multiply_scalar(a_dagger_matrix, theta);
    Matrix temp = matrix_multiply_scalar(a_matrix, std::conj(theta)); // + theta^* a
    matrix_add_to(G, temp); // G is Hermitian if theta is complex? No.
    // theta a^dag + theta^* a is Hermitian.
    
    // Compute cos(G) and sin(G)
    // cos(G) = I - G^2/2! + ...
    // sin(G) = G - G^3/3! + ...
    // We can use matrix_exponentiate for exp(iG) = cos(G) + i sin(G).
    Matrix iG = matrix_multiply_scalar(G, Complex(0.0, 1.0));
    Matrix exp_iG = matrix_exponentiate(iG);
    
    Matrix minus_iG = matrix_multiply_scalar(G, Complex(0.0, -1.0));
    Matrix exp_minus_iG = matrix_exponentiate(minus_iG);
    
    // cos(G) = (exp(iG) + exp(-iG)) / 2
    Matrix cosG = matrix_multiply_scalar(exp_iG, 0.5);
    matrix_add_to(cosG, matrix_multiply_scalar(exp_minus_iG, 0.5));
    
    // sin(G) = (exp(iG) - exp(-iG)) / 2i
    Matrix sinG = matrix_multiply_scalar(exp_iG, Complex(0.0, -0.5)); // 1/2i = -i/2
    matrix_add_to(sinG, matrix_multiply_scalar(exp_minus_iG, Complex(0.0, 0.5))); // -(-i/2) = i/2
    
    Vector new_v0 = matrix_vector_multiply(cosG, v0);
    Vector temp_v1 = matrix_vector_multiply(sinG, v1);
    // v0' = cosG v0 - i sinG v1
    for(int i=0; i<dim; ++i) new_v0[i] -= Complex(0.0, 1.0) * temp_v1[i];
    
    Vector new_v1 = matrix_vector_multiply(sinG, v0);
    // v1' = -i sinG v0 + cosG v1
    for(int i=0; i<dim; ++i) new_v1[i] *= Complex(0.0, -1.0);
    Vector temp_cos_v1 = matrix_vector_multiply(cosG, v1);
    for(int i=0; i<dim; ++i) new_v1[i] += temp_cos_v1[i];
    
    v0 = new_v0;
    v1 = new_v1;
}

void InteractionGates::apply_jaynes_cummings(Vector& v0, Vector& v1, double theta, double phi) {
    // JC acts on subspaces |1, n> and |0, n+1>.
    // v0 corresponds to |0>, v1 corresponds to |1>.
    // Couple v1[n] and v0[n+1].
    
    int dim = v0.size();
    Vector new_v0 = v0;
    Vector new_v1 = v1;
    
    // |0,0> is decoupled and unchanged (annihilated by a).
    
    for (int n = 0; n < dim - 1; ++n) {
        // Subspace: |1, n> (in v1 at index n) and |0, n+1> (in v0 at index n+1)
        // Coupling strength Omega_n = theta * sqrt(n+1)
        double omega = theta * std::sqrt(n + 1.0);
        
        Complex c1 = v1[n];      // |1, n>
        Complex c0 = v0[n+1];    // |0, n+1>
        
        // From gatesmath.md:
        // |0, n+1> -> cos(Omega) |0, n+1> - i e^{-i phi} sin(Omega) |1, n>
        // |1, n>   -> -i e^{i phi} sin(Omega) |0, n+1> + cos(Omega) |1, n>
        
        double cos_w = std::cos(omega);
        double sin_w = std::sin(omega);
        
        Complex term_from_1 = Complex(0.0, -1.0) * std::exp(Complex(0.0, -phi)) * sin_w * c1;
        new_v0[n+1] = cos_w * c0 + term_from_1;
        
        Complex term_from_0 = Complex(0.0, -1.0) * std::exp(Complex(0.0, phi)) * sin_w * c0;
        new_v1[n] = term_from_0 + cos_w * c1;
    }
    // v0[0] unchanged.
    // v1[dim-1] decoupled (if we truncate) or coupled to |0, dim> (which is out of bounds).
    // So v1[dim-1] effectively just rotates by cos? Or if we assume hard truncation, a|dim>=0?
    // If a|dim>=0, then |1, dim-1> -> |0, dim> is 0. So it stays |1, dim-1>.
    // cos(theta*sqrt(dim))? No, a|dim>=0 means sqrt(dim)|dim-1>.
    // Actually if we truncate, we assume higher states don't exist.
    
    v0 = new_v0;
    v1 = new_v1;
}

void InteractionGates::apply_anti_jaynes_cummings(Vector& v0, Vector& v1, double theta, double phi) {
    // AJC: |0, n> <-> |1, n+1>
    int dim = v0.size();
    Vector new_v0 = v0;
    Vector new_v1 = v1;
    
    for (int n = 0; n < dim - 1; ++n) {
        // Subspace: |0, n> and |1, n+1>
        // Omega = theta * sqrt(n+1)
        double omega = theta * std::sqrt(n + 1.0);
        
        Complex c0 = v0[n];      // |0, n>
        Complex c1 = v1[n+1];    // |1, n+1>
        
        // Logic derived similar to JC but flipped
        // |0, n> -> cos(Omega) |0, n> - i e^{-i phi} sin(Omega) |1, n+1>
        // |1, n+1> -> -i e^{i phi} sin(Omega) |0, n> + cos(Omega) |1, n+1>
        
        double cos_w = std::cos(omega);
        double sin_w = std::sin(omega);
        
        Complex term_from_1 = Complex(0.0, -1.0) * std::exp(Complex(0.0, -phi)) * sin_w * c1;
        new_v0[n] = cos_w * c0 + term_from_1;
        
        Complex term_from_0 = Complex(0.0, -1.0) * std::exp(Complex(0.0, phi)) * sin_w * c0;
        new_v1[n+1] = term_from_0 + cos_w * c1;
    }
    
    v0 = new_v0;
    v1 = new_v1;
}

// ===== Special Gates =====

void SpecialGates::apply_sqr(Vector& v0, Vector& v1, const std::vector<double>& thetas, const std::vector<double>& phis) {
    int dim = v0.size();
    for (int n = 0; n < dim; ++n) {
        double theta = (n < thetas.size()) ? thetas[n] : 0.0;
        double phi = (n < phis.size()) ? phis[n] : 0.0;
        
        // R(theta, phi) on qubit basis {|0>, |1>} for mode n
        // Matrix: [[alpha, beta], [-beta*, alpha*]]
        // alpha = cos(theta/2)
        // beta = -e^{-i phi} sin(theta/2)
        
        double cos_t = std::cos(theta / 2.0);
        double sin_t = std::sin(theta / 2.0);
        Complex alpha(cos_t, 0.0);
        Complex beta = -std::exp(Complex(0.0, -phi)) * sin_t;
        
        Complex c0 = v0[n];
        Complex c1 = v1[n];
        
        v0[n] = alpha * c0 + beta * c1;
        v1[n] = -std::conj(beta) * c0 + std::conj(alpha) * c1;
    }
}

// ===== 矩阵创建函数实现 =====

Matrix create_displacement_matrix(int dim, Complex alpha) {
    // D(α) = exp(α*a† - conj(α)*a)
    Matrix a_matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    Matrix a_dagger_matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));

    for (int i = 0; i < dim - 1; ++i) {
        double coeff = std::sqrt(i + 1.0);
        a_matrix[i][i + 1] = Complex(coeff, 0.0);
        a_dagger_matrix[i + 1][i] = Complex(coeff, 0.0);
    }

    Matrix generator = matrix_multiply_scalar(a_dagger_matrix, alpha);
    Matrix temp = matrix_multiply_scalar(a_matrix, -std::conj(alpha));
    matrix_add_to(generator, temp);

    return matrix_exponentiate(generator, 10); // Use helper
}

Matrix create_squeezing_matrix(int dim, Complex xi) {
    // S(ξ) = exp(1/2 * (ξ*a†^2 - ξ*a^2)) ? No, S(xi) = exp[1/2 (xi^* a^2 - xi (a^dag)^2)]
    // Check definition in gatesmath: S(xi) = exp[1/2 (xi^* a^2 - xi (a^dag)^2)]
    // Wait, typically S(z) = exp(1/2 (z* a^2 - z a^dag^2)) where z = r e^{i theta}.
    // Here xi is used.
    
    Matrix a_matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    Matrix a_dagger_matrix(dim, std::vector<Complex>(dim, Complex(0.0, 0.0)));
    
    for (int i = 0; i < dim - 1; ++i) {
        double coeff = std::sqrt(i + 1.0);
        a_matrix[i][i + 1] = Complex(coeff, 0.0);
        a_dagger_matrix[i + 1][i] = Complex(coeff, 0.0);
    }
    
    Matrix a2 = matrix_multiply(a_matrix, a_matrix);
    Matrix adag2 = matrix_multiply(a_dagger_matrix, a_dagger_matrix);
    
    // G = 1/2 (xi^* a^2 - xi (a^dag)^2)
    Matrix term1 = matrix_multiply_scalar(a2, 0.5 * std::conj(xi));
    Matrix term2 = matrix_multiply_scalar(adag2, -0.5 * xi);
    
    Matrix generator = term1;
    matrix_add_to(generator, term2);
    
    return matrix_exponentiate(generator, 10);
}

Matrix create_beam_splitter_matrix(int dim1, int dim2, double theta, double phi) {
    // BS(theta, phi) = exp[-i theta/2 (e^{i phi} a^dag b + e^{-i phi} a b^dag)]
    int total_dim = dim1 * dim2;
    Matrix generator(total_dim, std::vector<Complex>(total_dim, Complex(0.0, 0.0)));
    
    // a^dag b connects |n1, n2> to |n1+1, n2-1>
    // a b^dag connects |n1, n2> to |n1-1, n2+1>
    
    for (int n1 = 0; n1 < dim1; ++n1) {
        for (int n2 = 0; n2 < dim2; ++n2) {
            int idx_src = n1 * dim2 + n2;
            
            // Term 1: e^{i phi} a^dag b |n1, n2> = e^{i phi} sqrt(n1+1) sqrt(n2) |n1+1, n2-1>
            if (n1 + 1 < dim1 && n2 - 1 >= 0) {
                int idx_dst = (n1 + 1) * dim2 + (n2 - 1);
                Complex val = std::exp(Complex(0.0, phi)) * std::sqrt(n1 + 1.0) * std::sqrt(static_cast<double>(n2));
                // G[dst][src] += -i theta/2 * val
                generator[idx_dst][idx_src] += Complex(0.0, -theta/2.0) * val;
            }
            
            // Term 2: e^{-i phi} a b^dag |n1, n2> = e^{-i phi} sqrt(n1) sqrt(n2+1) |n1-1, n2+1>
            if (n1 - 1 >= 0 && n2 + 1 < dim2) {
                int idx_dst = (n1 - 1) * dim2 + (n2 + 1);
                Complex val = std::exp(Complex(0.0, -phi)) * std::sqrt(static_cast<double>(n1)) * std::sqrt(n2 + 1.0);
                generator[idx_dst][idx_src] += Complex(0.0, -theta/2.0) * val;
            }
        }
    }
    
    return matrix_exponentiate(generator, 10);
}

} // namespace Reference
