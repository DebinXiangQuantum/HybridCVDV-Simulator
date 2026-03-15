#include "gaussian_circuit.h"

#include "gaussian_kernels.h"
#include "reference_gates.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kGaussianTolerance = 1e-9;
constexpr double kEigenTolerance = 1e-12;
constexpr int kJacobiMaxSweeps = 128;

using Complex = std::complex<double>;
using ComplexMatrix = std::vector<std::vector<Complex>>;

struct SymmetricEigenDecomposition {
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;  // row-major, eigenvectors stored by column
};

struct PassiveBeamSplitter {
    int mode1 = 0;
    int mode2 = 0;
    double theta = 0.0;
    double phi = 0.0;
};

struct PassiveUnitaryDecomposition {
    std::vector<double> phase_rotation_thetas;
    std::vector<PassiveBeamSplitter> replay_beam_splitters;
};

bool approx_equal(double lhs, double rhs, double tolerance = kGaussianTolerance) {
    return std::abs(lhs - rhs) <= tolerance;
}

size_t integer_power(size_t base, int exponent) {
    size_t result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

std::vector<double> multiply_real_matrices(const std::vector<double>& lhs,
                                           const std::vector<double>& rhs,
                                           int dim) {
    std::vector<double> result(static_cast<size_t>(dim) * dim, 0.0);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double sum = 0.0;
            for (int k = 0; k < dim; ++k) {
                sum += lhs[static_cast<size_t>(row) * dim + static_cast<size_t>(k)] *
                       rhs[static_cast<size_t>(k) * dim + static_cast<size_t>(col)];
            }
            result[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] = sum;
        }
    }
    return result;
}

double vector_norm(const std::vector<double>& values) {
    double norm_sq = 0.0;
    for (double value : values) {
        norm_sq += value * value;
    }
    return std::sqrt(norm_sq);
}

std::vector<double> extract_column(const std::vector<double>& matrix, int dim, int col) {
    std::vector<double> column(static_cast<size_t>(dim), 0.0);
    for (int row = 0; row < dim; ++row) {
        column[static_cast<size_t>(row)] =
            matrix[static_cast<size_t>(row) * dim + static_cast<size_t>(col)];
    }
    return column;
}

void set_column(std::vector<double>& matrix, int dim, int col, const std::vector<double>& values) {
    for (int row = 0; row < dim; ++row) {
        matrix[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
            values[static_cast<size_t>(row)];
    }
}

std::vector<double> apply_negative_omega(const std::vector<double>& values) {
    if (values.size() % 2 != 0) {
        throw std::invalid_argument("phase-space vector length must be even");
    }

    std::vector<double> transformed(values.size(), 0.0);
    for (size_t mode = 0; mode < values.size() / 2; ++mode) {
        transformed[2 * mode] = -values[2 * mode + 1];
        transformed[2 * mode + 1] = values[2 * mode];
    }
    return transformed;
}

void canonicalize_vector_sign(std::vector<double>& values) {
    for (double value : values) {
        if (std::abs(value) > kEigenTolerance) {
            if (value < 0.0) {
                for (double& entry : values) {
                    entry = -entry;
                }
            }
            return;
        }
    }
}

bool is_pure_gaussian_covariance(const std::vector<double>& sigma, int num_qumodes) {
    const int dim = 2 * num_qumodes;
    std::vector<double> omega(static_cast<size_t>(dim) * dim, 0.0);
    for (int mode = 0; mode < num_qumodes; ++mode) {
        const int q = 2 * mode;
        omega[static_cast<size_t>(q) * dim + static_cast<size_t>(q + 1)] = 1.0;
        omega[static_cast<size_t>(q + 1) * dim + static_cast<size_t>(q)] = -1.0;
    }

    const std::vector<double> sigma_omega = multiply_real_matrices(sigma, omega, dim);
    const std::vector<double> purity_lhs = multiply_real_matrices(sigma_omega, sigma, dim);

    double max_error = 0.0;
    for (size_t idx = 0; idx < purity_lhs.size(); ++idx) {
        max_error = std::max(max_error, std::abs(purity_lhs[idx] - omega[idx] * 0.25));
    }
    return max_error < 1e-6;
}

SymmetricEigenDecomposition jacobi_eigendecompose_symmetric(const std::vector<double>& matrix, int dim) {
    std::vector<double> a = matrix;
    std::vector<double> eigenvectors(static_cast<size_t>(dim) * dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        eigenvectors[static_cast<size_t>(i) * dim + static_cast<size_t>(i)] = 1.0;
    }

    const int max_iterations = kJacobiMaxSweeps * dim * dim;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        int pivot_row = 0;
        int pivot_col = 1;
        double max_offdiag = 0.0;

        for (int row = 0; row < dim; ++row) {
            for (int col = row + 1; col < dim; ++col) {
                const double value = std::abs(a[static_cast<size_t>(row) * dim + static_cast<size_t>(col)]);
                if (value > max_offdiag) {
                    max_offdiag = value;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }

        if (max_offdiag < kEigenTolerance) {
            break;
        }

        const double app = a[static_cast<size_t>(pivot_row) * dim + static_cast<size_t>(pivot_row)];
        const double aqq = a[static_cast<size_t>(pivot_col) * dim + static_cast<size_t>(pivot_col)];
        const double apq = a[static_cast<size_t>(pivot_row) * dim + static_cast<size_t>(pivot_col)];

        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi);
        const double s = std::sin(phi);

        for (int k = 0; k < dim; ++k) {
            if (k == pivot_row || k == pivot_col) {
                continue;
            }

            const double aik = a[static_cast<size_t>(k) * dim + static_cast<size_t>(pivot_row)];
            const double akq = a[static_cast<size_t>(k) * dim + static_cast<size_t>(pivot_col)];
            const double new_aik = c * aik - s * akq;
            const double new_akq = s * aik + c * akq;

            a[static_cast<size_t>(k) * dim + static_cast<size_t>(pivot_row)] = new_aik;
            a[static_cast<size_t>(pivot_row) * dim + static_cast<size_t>(k)] = new_aik;
            a[static_cast<size_t>(k) * dim + static_cast<size_t>(pivot_col)] = new_akq;
            a[static_cast<size_t>(pivot_col) * dim + static_cast<size_t>(k)] = new_akq;
        }

        a[static_cast<size_t>(pivot_row) * dim + static_cast<size_t>(pivot_row)] =
            c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[static_cast<size_t>(pivot_col) * dim + static_cast<size_t>(pivot_col)] =
            s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[static_cast<size_t>(pivot_row) * dim + static_cast<size_t>(pivot_col)] = 0.0;
        a[static_cast<size_t>(pivot_col) * dim + static_cast<size_t>(pivot_row)] = 0.0;

        for (int row = 0; row < dim; ++row) {
            const double vip = eigenvectors[static_cast<size_t>(row) * dim + static_cast<size_t>(pivot_row)];
            const double viq = eigenvectors[static_cast<size_t>(row) * dim + static_cast<size_t>(pivot_col)];
            eigenvectors[static_cast<size_t>(row) * dim + static_cast<size_t>(pivot_row)] =
                c * vip - s * viq;
            eigenvectors[static_cast<size_t>(row) * dim + static_cast<size_t>(pivot_col)] =
                s * vip + c * viq;
        }
    }

    std::vector<int> ordering(static_cast<size_t>(dim), 0);
    for (int idx = 0; idx < dim; ++idx) {
        ordering[static_cast<size_t>(idx)] = idx;
    }
    std::sort(ordering.begin(), ordering.end(), [&a, dim](int lhs, int rhs) {
        return a[static_cast<size_t>(lhs) * dim + static_cast<size_t>(lhs)] <
               a[static_cast<size_t>(rhs) * dim + static_cast<size_t>(rhs)];
    });

    SymmetricEigenDecomposition decomposition;
    decomposition.eigenvalues.resize(static_cast<size_t>(dim), 0.0);
    decomposition.eigenvectors.assign(static_cast<size_t>(dim) * dim, 0.0);

    for (int new_col = 0; new_col < dim; ++new_col) {
        const int old_col = ordering[static_cast<size_t>(new_col)];
        decomposition.eigenvalues[static_cast<size_t>(new_col)] =
            a[static_cast<size_t>(old_col) * dim + static_cast<size_t>(old_col)];
        for (int row = 0; row < dim; ++row) {
            decomposition.eigenvectors[static_cast<size_t>(row) * dim + static_cast<size_t>(new_col)] =
                eigenvectors[static_cast<size_t>(row) * dim + static_cast<size_t>(old_col)];
        }
    }

    return decomposition;
}

ComplexMatrix passive_symplectic_to_unitary(const std::vector<double>& orthogonal_symplectic,
                                            int num_qumodes) {
    ComplexMatrix unitary(static_cast<size_t>(num_qumodes),
                          std::vector<Complex>(static_cast<size_t>(num_qumodes), Complex(0.0, 0.0)));
    const int dim = 2 * num_qumodes;

    for (int row_mode = 0; row_mode < num_qumodes; ++row_mode) {
        for (int col_mode = 0; col_mode < num_qumodes; ++col_mode) {
            const int row = 2 * row_mode;
            const int col = 2 * col_mode;
            const double block_xx = orthogonal_symplectic[static_cast<size_t>(row) * dim + static_cast<size_t>(col)];
            const double block_xp = orthogonal_symplectic[static_cast<size_t>(row) * dim + static_cast<size_t>(col + 1)];
            const double block_px = orthogonal_symplectic[static_cast<size_t>(row + 1) * dim + static_cast<size_t>(col)];
            const double block_pp = orthogonal_symplectic[static_cast<size_t>(row + 1) * dim + static_cast<size_t>(col + 1)];

            const Complex from_first_column(block_xx, block_px);
            const Complex from_second_column(block_pp, -block_xp);
            unitary[static_cast<size_t>(row_mode)][static_cast<size_t>(col_mode)] =
                0.5 * (from_first_column + from_second_column);
        }
    }

    return unitary;
}

void apply_left_beam_splitter(ComplexMatrix& unitary,
                              int mode1,
                              int mode2,
                              double theta,
                              double phi) {
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    const Complex upper_phase = std::polar(1.0, -phi);
    const Complex lower_phase = std::polar(1.0, phi);

    for (size_t col = 0; col < unitary[static_cast<size_t>(mode1)].size(); ++col) {
        const Complex upper = unitary[static_cast<size_t>(mode1)][col];
        const Complex lower = unitary[static_cast<size_t>(mode2)][col];

        unitary[static_cast<size_t>(mode1)][col] = cos_theta * upper + sin_theta * upper_phase * lower;
        unitary[static_cast<size_t>(mode2)][col] = -sin_theta * lower_phase * upper + cos_theta * lower;
    }
}

PassiveUnitaryDecomposition decompose_passive_unitary_reck(const ComplexMatrix& passive_unitary) {
    const int num_qumodes = static_cast<int>(passive_unitary.size());
    ComplexMatrix working = passive_unitary;
    std::vector<PassiveBeamSplitter> elimination_steps;

    for (int col = 0; col < num_qumodes - 1; ++col) {
        for (int row = num_qumodes - 1; row > col; --row) {
            const Complex upper = working[static_cast<size_t>(row - 1)][static_cast<size_t>(col)];
            const Complex lower = working[static_cast<size_t>(row)][static_cast<size_t>(col)];
            if (std::abs(lower) < kEigenTolerance) {
                continue;
            }

            const double upper_abs = std::abs(upper);
            const double lower_abs = std::abs(lower);
            const double theta = std::atan2(lower_abs, upper_abs);
            double phi = 0.0;
            if (upper_abs > kEigenTolerance) {
                phi = std::arg(lower) - std::arg(upper);
            } else {
                phi = std::arg(lower);
            }

            apply_left_beam_splitter(working, row - 1, row, theta, phi);
            working[static_cast<size_t>(row)][static_cast<size_t>(col)] = Complex(0.0, 0.0);

            elimination_steps.push_back({row - 1, row, theta, phi});
        }
    }

    PassiveUnitaryDecomposition decomposition;
    decomposition.phase_rotation_thetas.resize(static_cast<size_t>(num_qumodes), 0.0);
    for (int mode = 0; mode < num_qumodes; ++mode) {
        decomposition.phase_rotation_thetas[static_cast<size_t>(mode)] =
            -std::arg(working[static_cast<size_t>(mode)][static_cast<size_t>(mode)]);
    }

    decomposition.replay_beam_splitters.reserve(elimination_steps.size());
    for (auto it = elimination_steps.rbegin(); it != elimination_steps.rend(); ++it) {
        decomposition.replay_beam_splitters.push_back(
            {it->mode1, it->mode2, -it->theta, it->phi});
    }
    return decomposition;
}

std::complex<double> displacement_to_alpha(const std::vector<double>& d) {
    if (d.size() != 2) {
        throw std::invalid_argument("Single-mode displacement vector must have size 2");
    }
    return std::complex<double>(d[0], d[1]) / std::sqrt(2.0);
}

std::pair<double, double> covariance_to_squeezing(const std::vector<double>& sigma) {
    if (sigma.size() != 4) {
        throw std::invalid_argument("Single-mode covariance matrix must have size 4");
    }

    const double s_xx = sigma[0];
    const double s_xp = sigma[1];
    const double s_pp = sigma[3];

    const double cosh_2r = std::max(1.0, s_xx + s_pp);
    const double sinh_2r = std::sqrt(std::max(0.0, cosh_2r * cosh_2r - 1.0));
    if (sinh_2r < 1e-12) {
        return {0.0, 0.0};
    }

    const double theta = std::atan2(-2.0 * s_xp, s_pp - s_xx);
    const double r = 0.5 * std::asinh(sinh_2r);
    return {r, theta};
}

std::vector<std::complex<double>> normalize_fock_state(
    const std::vector<std::complex<double>>& state) {
    double norm_sq = 0.0;
    for (const auto& amplitude : state) {
        norm_sq += std::norm(amplitude);
    }

    if (norm_sq < 1e-16) {
        throw std::runtime_error("Projected Fock state has zero norm");
    }

    std::vector<std::complex<double>> normalized = state;
    const double inv_norm = 1.0 / std::sqrt(norm_sq);
    for (auto& amplitude : normalized) {
        amplitude *= inv_norm;
    }
    return normalized;
}

std::vector<std::complex<double>> tensor_product(
    const std::vector<std::complex<double>>& lhs,
    const std::vector<std::complex<double>>& rhs) {
    std::vector<std::complex<double>> result(lhs.size() * rhs.size(),
                                             std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < lhs.size(); ++i) {
        for (size_t j = 0; j < rhs.size(); ++j) {
            result[i * rhs.size() + j] = lhs[i] * rhs[j];
        }
    }
    return result;
}

bool is_identity_symplectic(const SymplecticGate& gate) {
    const int dim = 2 * gate.num_qumodes;
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            const double expected = row == col ? 1.0 : 0.0;
            if (!approx_equal(gate.S[row * dim + col], expected)) {
                return false;
            }
        }
    }
    return true;
}

bool has_nonzero_displacement(const SymplecticGate& gate) {
    for (double value : gate.d) {
        if (std::abs(value) > kGaussianTolerance) {
            return true;
        }
    }
    return false;
}

std::vector<int> find_active_modes(const SymplecticGate& gate) {
    const int dim = 2 * gate.num_qumodes;
    std::vector<int> active_modes;

    for (int mode = 0; mode < gate.num_qumodes; ++mode) {
        bool is_active = false;
        for (int local_row = 0; local_row < 2 && !is_active; ++local_row) {
            const int row = 2 * mode + local_row;
            for (int col = 0; col < dim; ++col) {
                const double expected = row == col ? 1.0 : 0.0;
                if (!approx_equal(gate.S[row * dim + col], expected)) {
                    is_active = true;
                    break;
                }
            }
        }

        if (!is_active) {
            for (int local_col = 0; local_col < 2 && !is_active; ++local_col) {
                const int col = 2 * mode + local_col;
                for (int row = 0; row < dim; ++row) {
                    const double expected = row == col ? 1.0 : 0.0;
                    if (!approx_equal(gate.S[row * dim + col], expected)) {
                        is_active = true;
                        break;
                    }
                }
            }
        }

        if (is_active) {
            active_modes.push_back(mode);
        }
    }

    return active_modes;
}

std::vector<double> extract_local_block(const SymplecticGate& gate, int mode) {
    const int dim = 2 * gate.num_qumodes;
    const int q = 2 * mode;
    return {
        gate.S[q * dim + q],
        gate.S[q * dim + q + 1],
        gate.S[(q + 1) * dim + q],
        gate.S[(q + 1) * dim + q + 1]
    };
}

std::vector<double> extract_cross_block(const SymplecticGate& gate, int row_mode, int col_mode) {
    const int dim = 2 * gate.num_qumodes;
    const int row = 2 * row_mode;
    const int col = 2 * col_mode;
    return {
        gate.S[row * dim + col],
        gate.S[row * dim + col + 1],
        gate.S[(row + 1) * dim + col],
        gate.S[(row + 1) * dim + col + 1]
    };
}

template <typename Transform>
std::vector<std::complex<double>> apply_single_mode_transform(
    const std::vector<std::complex<double>>& state,
    int cutoff,
    int num_qumodes,
    int target_qumode,
    Transform&& transform) {
    if (target_qumode < 0 || target_qumode >= num_qumodes) {
        throw std::out_of_range("single-mode transform target qumode out of range");
    }

    const size_t stride = integer_power(static_cast<size_t>(cutoff), num_qumodes - target_qumode - 1);
    const size_t prefix_count = integer_power(static_cast<size_t>(cutoff), target_qumode);
    std::vector<std::complex<double>> result(state.size(), std::complex<double>(0.0, 0.0));

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
std::vector<std::complex<double>> apply_two_mode_transform(
    const std::vector<std::complex<double>>& state,
    int cutoff,
    int num_qumodes,
    int first_qumode,
    int second_qumode,
    Transform&& transform) {
    if (first_qumode < 0 || first_qumode >= num_qumodes ||
        second_qumode < 0 || second_qumode >= num_qumodes ||
        first_qumode == second_qumode) {
        throw std::out_of_range("two-mode transform qumode indices are invalid");
    }

    std::vector<size_t> strides(static_cast<size_t>(num_qumodes), 1);
    for (int mode = 0; mode < num_qumodes; ++mode) {
        strides[static_cast<size_t>(mode)] =
            integer_power(static_cast<size_t>(cutoff), num_qumodes - mode - 1);
    }

    std::vector<int> other_modes;
    for (int mode = 0; mode < num_qumodes; ++mode) {
        if (mode != first_qumode && mode != second_qumode) {
            other_modes.push_back(mode);
        }
    }

    const size_t outer_count = integer_power(static_cast<size_t>(cutoff), num_qumodes - 2);
    const size_t local_dim = static_cast<size_t>(cutoff) * static_cast<size_t>(cutoff);
    std::vector<std::complex<double>> result(state.size(), std::complex<double>(0.0, 0.0));

    for (size_t outer_index = 0; outer_index < outer_count; ++outer_index) {
        size_t residual = outer_index;
        size_t base_index = 0;
        for (int idx = static_cast<int>(other_modes.size()) - 1; idx >= 0; --idx) {
            const int digit = static_cast<int>(residual % static_cast<size_t>(cutoff));
            residual /= static_cast<size_t>(cutoff);
            base_index += static_cast<size_t>(digit) * strides[static_cast<size_t>(other_modes[static_cast<size_t>(idx)])];
        }

        Reference::Vector local_state(local_dim, Reference::Complex(0.0, 0.0));
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

GaussianCircuit::GaussianCircuit(int num_qumodes, int capacity)
    : num_qumodes_(num_qumodes),
      symbolic_mode_limit_(4),
      non_gaussian_threshold_(1e-8),
      current_track_(ExecutionTrack::Symbolic),
      executed_gate_count_(0) {
    pool_ = std::make_unique<GaussianStatePool>(num_qumodes, capacity);
    root_state_id_ = pool_->allocate_state();
    if (root_state_id_ < 0) {
        throw std::runtime_error("Failed to allocate Gaussian root state");
    }

    const int dim = 2 * num_qumodes;
    std::vector<double> d(dim, 0.0);
    std::vector<double> sigma(dim * dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        sigma[i * dim + i] = 0.5;
    }
    pool_->upload_state(root_state_id_, d, sigma);
}

GaussianCircuit::~GaussianCircuit() = default;

void GaussianCircuit::add_gate(const SymplecticGate& gate) {
    if (gate.num_qumodes != num_qumodes_) {
        throw std::invalid_argument("Symplectic gate qumode count does not match circuit");
    }
    recorded_gate_history_.push_back(record_gate(gate));
    gate_sequence_.push_back(gate);
}

void GaussianCircuit::execute_symbolic_sequence() {
    if (gate_sequence_.empty()) {
        current_track_ = ExecutionTrack::Symbolic;
        projected_fock_state_.clear();
        return;
    }

    for (const auto& gate : gate_sequence_) {
        double* d_S = nullptr;
        double* d_dg = nullptr;
        int* d_state_ids = nullptr;
        const int state_ids[] = {root_state_id_};

        cudaError_t err = cudaMalloc(&d_S, gate.S.size() * sizeof(double));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate symplectic matrix buffer");
        }
        err = cudaMalloc(&d_dg, gate.d.size() * sizeof(double));
        if (err != cudaSuccess) {
            cudaFree(d_S);
            throw std::runtime_error("Failed to allocate displacement buffer");
        }
        err = cudaMalloc(&d_state_ids, sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_S);
            cudaFree(d_dg);
            throw std::runtime_error("Failed to allocate Gaussian state ID buffer");
        }

        err = cudaMemcpy(d_S, gate.S.data(), gate.S.size() * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_S);
            cudaFree(d_dg);
            cudaFree(d_state_ids);
            throw std::runtime_error("Failed to upload symplectic matrix");
        }
        err = cudaMemcpy(d_dg, gate.d.data(), gate.d.size() * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_S);
            cudaFree(d_dg);
            cudaFree(d_state_ids);
            throw std::runtime_error("Failed to upload displacement vector");
        }
        err = cudaMemcpy(d_state_ids, state_ids, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_S);
            cudaFree(d_dg);
            cudaFree(d_state_ids);
            throw std::runtime_error("Failed to upload Gaussian state IDs");
        }

        apply_batched_symplectic_update(pool_.get(), d_state_ids, 1, d_S, d_dg);

        cudaFree(d_S);
        cudaFree(d_dg);
        cudaFree(d_state_ids);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Gaussian symbolic execution failed during synchronization");
    }

    current_track_ = ExecutionTrack::Symbolic;
    projected_fock_state_.clear();
    executed_gate_count_ += gate_sequence_.size();
    gate_sequence_.clear();
}

void GaussianCircuit::execute() {
    execute_symbolic_sequence();
}

int GaussianCircuit::estimate_active_modes() const {
    std::vector<double> d;
    std::vector<double> sigma;
    pool_->download_state(root_state_id_, d, sigma);

    const int dim = 2 * num_qumodes_;
    std::vector<bool> active_modes(static_cast<size_t>(num_qumodes_), false);

    for (int mode = 0; mode < num_qumodes_; ++mode) {
        const int q = 2 * mode;
        const double dx = d[q];
        const double dp = d[q + 1];
        const double s_xx = sigma[q * dim + q];
        const double s_xp = sigma[q * dim + q + 1];
        const double s_px = sigma[(q + 1) * dim + q];
        const double s_pp = sigma[(q + 1) * dim + q + 1];

        if (std::abs(dx) > kGaussianTolerance ||
            std::abs(dp) > kGaussianTolerance ||
            std::abs(s_xx - 0.5) > kGaussianTolerance ||
            std::abs(s_pp - 0.5) > kGaussianTolerance ||
            std::abs(s_xp) > kGaussianTolerance ||
            std::abs(s_px) > kGaussianTolerance) {
            active_modes[static_cast<size_t>(mode)] = true;
        }
    }

    for (int mode_i = 0; mode_i < num_qumodes_; ++mode_i) {
        for (int mode_j = mode_i + 1; mode_j < num_qumodes_; ++mode_j) {
            bool correlated = false;
            for (int local_i = 0; local_i < 2 && !correlated; ++local_i) {
                for (int local_j = 0; local_j < 2; ++local_j) {
                    const int row = 2 * mode_i + local_i;
                    const int col = 2 * mode_j + local_j;
                    if (std::abs(sigma[row * dim + col]) > kGaussianTolerance ||
                        std::abs(sigma[col * dim + row]) > kGaussianTolerance) {
                        correlated = true;
                        break;
                    }
                }
            }
            if (correlated) {
                active_modes[static_cast<size_t>(mode_i)] = true;
                active_modes[static_cast<size_t>(mode_j)] = true;
            }
        }
    }

    return static_cast<int>(std::count(active_modes.begin(), active_modes.end(), true));
}

ExecutionDecision GaussianCircuit::decide_execution_track(double non_gaussianity) const {
    const int active_modes = estimate_active_modes();
    const bool requires_tensor =
        non_gaussianity > non_gaussian_threshold_ ||
        active_modes > symbolic_mode_limit_;

    return {
        requires_tensor ? ExecutionTrack::Tensor : ExecutionTrack::Symbolic,
        active_modes,
        non_gaussianity,
        requires_tensor
    };
}

GaussianCircuit::RecordedGate GaussianCircuit::record_gate(const SymplecticGate& gate) const {
    RecordedGate recorded;

    if (has_nonzero_displacement(gate)) {
        if (!is_identity_symplectic(gate)) {
            recorded.type = RecordedGateType::Unsupported;
            return recorded;
        }

        int target_mode = -1;
        for (int mode = 0; mode < gate.num_qumodes; ++mode) {
            const double dx = gate.d[static_cast<size_t>(2 * mode)];
            const double dp = gate.d[static_cast<size_t>(2 * mode + 1)];
            if (std::abs(dx) > kGaussianTolerance || std::abs(dp) > kGaussianTolerance) {
                if (target_mode != -1) {
                    recorded.type = RecordedGateType::Unsupported;
                    return recorded;
                }
                target_mode = mode;
                recorded.complex_params = {
                    displacement_to_alpha({dx, dp})
                };
            }
        }

        if (target_mode == -1) {
            recorded.type = RecordedGateType::NoOp;
            return recorded;
        }

        recorded.type = RecordedGateType::Displacement;
        recorded.target_qumodes = {target_mode};
        return recorded;
    }

    const std::vector<int> active_modes = find_active_modes(gate);
    if (active_modes.empty()) {
        recorded.type = RecordedGateType::NoOp;
        return recorded;
    }

    if (active_modes.size() == 1) {
        const int target_mode = active_modes[0];
        const std::vector<double> block = extract_local_block(gate, target_mode);

        recorded.target_qumodes = {target_mode};
        if (approx_equal(block[0], block[3]) && approx_equal(block[1], -block[2])) {
            recorded.type = RecordedGateType::PhaseRotation;
            recorded.real_params = {std::atan2(block[1], block[0])};
            return recorded;
        }

        if (approx_equal(block[1], block[2])) {
            const double x = 0.5 * (block[3] - block[0]);
            const double y = -block[1];
            const double sinh_r = std::hypot(x, y);
            const double r = std::asinh(sinh_r);
            const double theta = sinh_r < kGaussianTolerance ? 0.0 : std::atan2(y, x);

            recorded.type = RecordedGateType::Squeezing;
            recorded.complex_params = {std::polar(r, theta)};
            return recorded;
        }

        recorded.type = RecordedGateType::Unsupported;
        return recorded;
    }

    if (active_modes.size() == 2) {
        const int first_mode = active_modes[0];
        const int second_mode = active_modes[1];
        const std::vector<double> cross_block = extract_cross_block(gate, first_mode, second_mode);
        const int dim = 2 * gate.num_qumodes;
        const double cos_theta = gate.S[static_cast<size_t>(2 * first_mode) * dim + static_cast<size_t>(2 * first_mode)];
        const double sin_theta = std::hypot(cross_block[0], cross_block[1]);
        const double phi = sin_theta < kGaussianTolerance ? 0.0 : std::atan2(cross_block[1], cross_block[0]);

        recorded.type = RecordedGateType::BeamSplitter;
        recorded.target_qumodes = {first_mode, second_mode};
        recorded.real_params = {std::atan2(sin_theta, cos_theta), phi};
        return recorded;
    }

    recorded.type = RecordedGateType::Unsupported;
    return recorded;
}

std::vector<std::complex<double>> GaussianCircuit::project_executed_gaussian_sequence_to_fock(
    int cutoff) const {
    if (cutoff <= 0) {
        throw std::invalid_argument("Fock cutoff must be positive");
    }

    Reference::Vector projected_state(
        integer_power(static_cast<size_t>(cutoff), num_qumodes_),
        Reference::Complex(0.0, 0.0));
    projected_state[0] = Reference::Complex(1.0, 0.0);

    for (size_t gate_index = 0; gate_index < executed_gate_count_; ++gate_index) {
        const RecordedGate& gate = recorded_gate_history_[gate_index];
        switch (gate.type) {
            case RecordedGateType::NoOp:
                break;
            case RecordedGateType::PhaseRotation: {
                const double theta = gate.real_params[0];
                projected_state = apply_single_mode_transform(
                    projected_state,
                    cutoff,
                    num_qumodes_,
                    gate.target_qumodes[0],
                    [theta](const Reference::Vector& local_state) {
                        return Reference::DiagonalGates::apply_phase_rotation(local_state, theta);
                    });
                break;
            }
            case RecordedGateType::Displacement: {
                const Reference::Complex alpha = gate.complex_params[0];
                projected_state = apply_single_mode_transform(
                    projected_state,
                    cutoff,
                    num_qumodes_,
                    gate.target_qumodes[0],
                    [alpha](const Reference::Vector& local_state) {
                        return Reference::SingleModeGates::apply_displacement_gate(local_state, alpha);
                    });
                break;
            }
            case RecordedGateType::Squeezing: {
                const Reference::Complex xi = gate.complex_params[0];
                projected_state = apply_single_mode_transform(
                    projected_state,
                    cutoff,
                    num_qumodes_,
                    gate.target_qumodes[0],
                    [xi](const Reference::Vector& local_state) {
                        return Reference::SingleModeGates::apply_squeezing_gate(local_state, xi);
                    });
                break;
            }
            case RecordedGateType::BeamSplitter: {
                const double theta = gate.real_params[0];
                const double phi = gate.real_params[1];
                projected_state = apply_two_mode_transform(
                    projected_state,
                    cutoff,
                    num_qumodes_,
                    gate.target_qumodes[0],
                    gate.target_qumodes[1],
                    [theta, phi](const Reference::Vector& local_state) {
                        return Reference::TwoModeGates::apply_beam_splitter(local_state, theta, phi);
                    });
                break;
            }
            case RecordedGateType::Unsupported:
                throw std::runtime_error(
                    "Gaussian-to-Fock projection encountered an unsupported symplectic gate during replay");
        }
    }

    return normalize_fock_state(projected_state);
}

std::vector<std::complex<double>> GaussianCircuit::project_via_bloch_messiah_to_fock(
    const std::vector<double>& d,
    const std::vector<double>& sigma,
    int cutoff) const {
    if (cutoff <= 0) {
        throw std::invalid_argument("Fock cutoff must be positive");
    }
    if (!is_pure_gaussian_covariance(sigma, num_qumodes_)) {
        throw std::runtime_error("Bloch-Messiah projection currently supports pure Gaussian states only");
    }

    const int dim = 2 * num_qumodes_;
    const SymmetricEigenDecomposition decomposition =
        jacobi_eigendecompose_symmetric(sigma, dim);

    std::vector<double> orthogonal_symplectic(static_cast<size_t>(dim) * dim, 0.0);
    std::vector<double> squeezing_parameters(static_cast<size_t>(num_qumodes_), 0.0);

    for (int mode = 0; mode < num_qumodes_; ++mode) {
        const double lambda = std::max(decomposition.eigenvalues[static_cast<size_t>(mode)], 1e-15);
        squeezing_parameters[static_cast<size_t>(mode)] =
            std::max(0.0, -0.5 * std::log(2.0 * lambda));

        std::vector<double> eigenvector = extract_column(decomposition.eigenvectors, dim, mode);
        canonicalize_vector_sign(eigenvector);
        std::vector<double> partner = apply_negative_omega(eigenvector);

        const double partner_norm = vector_norm(partner);
        if (partner_norm < kEigenTolerance) {
            throw std::runtime_error("Bloch-Messiah projection failed to construct symplectic partner vector");
        }
        for (double& entry : partner) {
            entry /= partner_norm;
        }

        set_column(orthogonal_symplectic, dim, 2 * mode, eigenvector);
        set_column(orthogonal_symplectic, dim, 2 * mode + 1, partner);
    }

    const ComplexMatrix passive_unitary =
        passive_symplectic_to_unitary(orthogonal_symplectic, num_qumodes_);
    const PassiveUnitaryDecomposition passive_decomposition =
        decompose_passive_unitary_reck(passive_unitary);

    Reference::Vector projected_state(
        integer_power(static_cast<size_t>(cutoff), num_qumodes_),
        Reference::Complex(0.0, 0.0));
    projected_state[0] = Reference::Complex(1.0, 0.0);

    for (int mode = 0; mode < num_qumodes_; ++mode) {
        const double r = squeezing_parameters[static_cast<size_t>(mode)];
        if (std::abs(r) < 1e-12) {
            continue;
        }

        projected_state = apply_single_mode_transform(
            projected_state,
            cutoff,
            num_qumodes_,
            mode,
            [r](const Reference::Vector& local_state) {
                return Reference::SingleModeGates::apply_squeezing_gate(local_state, Reference::Complex(r, 0.0));
            });
    }

    for (int mode = 0; mode < num_qumodes_; ++mode) {
        const double theta = passive_decomposition.phase_rotation_thetas[static_cast<size_t>(mode)];
        if (std::abs(theta) < 1e-12) {
            continue;
        }

        projected_state = apply_single_mode_transform(
            projected_state,
            cutoff,
            num_qumodes_,
            mode,
            [theta](const Reference::Vector& local_state) {
                return Reference::DiagonalGates::apply_phase_rotation(local_state, theta);
            });
    }

    for (const PassiveBeamSplitter& beam_splitter : passive_decomposition.replay_beam_splitters) {
        if (std::abs(beam_splitter.theta) < 1e-12) {
            continue;
        }

        projected_state = apply_two_mode_transform(
            projected_state,
            cutoff,
            num_qumodes_,
            beam_splitter.mode1,
            beam_splitter.mode2,
            [&beam_splitter](const Reference::Vector& local_state) {
                return Reference::TwoModeGates::apply_beam_splitter(
                    local_state, beam_splitter.theta, beam_splitter.phi);
            });
    }

    for (int mode = 0; mode < num_qumodes_; ++mode) {
        const Reference::Complex alpha =
            displacement_to_alpha({d[static_cast<size_t>(2 * mode)], d[static_cast<size_t>(2 * mode + 1)]});
        if (std::abs(alpha) < 1e-12) {
            continue;
        }

        projected_state = apply_single_mode_transform(
            projected_state,
            cutoff,
            num_qumodes_,
            mode,
            [alpha](const Reference::Vector& local_state) {
                return Reference::SingleModeGates::apply_displacement_gate(local_state, alpha);
            });
    }

    return normalize_fock_state(projected_state);
}

std::vector<std::complex<double>> GaussianCircuit::project_single_mode_pure_gaussian_to_fock(
    const std::vector<double>& d,
    const std::vector<double>& sigma,
    int cutoff) const {
    if (cutoff <= 0) {
        throw std::invalid_argument("Fock cutoff must be positive");
    }

    const double determinant = sigma[0] * sigma[3] - sigma[1] * sigma[2];
    if (std::abs(determinant - 0.25) > 1e-6) {
        throw std::runtime_error("Gaussian-to-Fock projection currently supports pure single-mode states only");
    }

    const auto alpha = displacement_to_alpha(d);
    const auto [r, theta] = covariance_to_squeezing(sigma);
    const std::complex<double> xi = std::polar(r, theta);

    Reference::Vector vacuum(static_cast<size_t>(cutoff), Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);

    const auto squeezed = Reference::SingleModeGates::apply_squeezing_gate(vacuum, xi);
    const auto displaced = Reference::SingleModeGates::apply_displacement_gate(squeezed, alpha);

    std::vector<std::complex<double>> projected(displaced.begin(), displaced.end());
    return normalize_fock_state(projected);
}

std::vector<std::complex<double>> GaussianCircuit::project_root_state_to_fock(int cutoff) const {
    std::vector<double> d;
    std::vector<double> sigma;
    pool_->download_state(root_state_id_, d, sigma);

    if (executed_gate_count_ > 0) {
        try {
            return project_executed_gaussian_sequence_to_fock(cutoff);
        } catch (const std::exception&) {
            // Fall through to the Bloch-Messiah fallback for handwritten/general symplectic gates.
        }
    }

    if (num_qumodes_ == 1) {
        return project_single_mode_pure_gaussian_to_fock(d, sigma, cutoff);
    }

    try {
        return project_via_bloch_messiah_to_fock(d, sigma, cutoff);
    } catch (const std::exception&) {
        // Preserve the older separable fallback for manual state injection / debugging cases.
    }

    const int dim = 2 * num_qumodes_;
    std::vector<std::complex<double>> projected_state = {std::complex<double>(1.0, 0.0)};

    for (int mode_i = 0; mode_i < num_qumodes_; ++mode_i) {
        for (int mode_j = mode_i + 1; mode_j < num_qumodes_; ++mode_j) {
            for (int local_i = 0; local_i < 2; ++local_i) {
                for (int local_j = 0; local_j < 2; ++local_j) {
                    const int row = 2 * mode_i + local_i;
                    const int col = 2 * mode_j + local_j;
                    if (std::abs(sigma[row * dim + col]) > 1e-8 ||
                        std::abs(sigma[col * dim + row]) > 1e-8) {
                        throw std::runtime_error(
                            "Gaussian-to-Fock projection without gate history only supports separable multi-mode Gaussian states");
                    }
                }
            }
        }
    }

    for (int mode = 0; mode < num_qumodes_; ++mode) {
        const int q = 2 * mode;
        std::vector<double> local_d = {d[q], d[q + 1]};
        std::vector<double> local_sigma = {
            sigma[q * dim + q],
            sigma[q * dim + q + 1],
            sigma[(q + 1) * dim + q],
            sigma[(q + 1) * dim + q + 1]
        };

        projected_state = tensor_product(
            projected_state,
            project_single_mode_pure_gaussian_to_fock(local_d, local_sigma, cutoff));
    }

    return normalize_fock_state(projected_state);
}

ExecutionDecision GaussianCircuit::execute_with_ede(int fock_cutoff, double non_gaussianity) {
    execute_symbolic_sequence();

    const ExecutionDecision decision = decide_execution_track(non_gaussianity);
    current_track_ = decision.track;

    if (decision.track == ExecutionTrack::Tensor) {
        if (fock_cutoff <= 0) {
            throw std::invalid_argument("Tensor track execution requires a positive Fock cutoff");
        }
        projected_fock_state_ = project_root_state_to_fock(fock_cutoff);
    } else {
        projected_fock_state_.clear();
    }

    return decision;
}
