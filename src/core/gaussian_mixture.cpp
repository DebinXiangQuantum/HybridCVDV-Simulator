#include "gaussian_mixture.h"

#include "reference_gates.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<double>;

constexpr double kMixtureTolerance = 1e-12;

size_t integer_power(size_t base, int exponent) {
    size_t result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

double conservative_fidelity_lower_bound_from_max_error(double max_error) {
    if (max_error <= 0.0) {
        return 1.0;
    }
    if (max_error >= 1.0) {
        return 0.0;
    }

    const double overlap_lower_bound = (1.0 - max_error) / (1.0 + max_error);
    return overlap_lower_bound * overlap_lower_bound;
}

void validate_common_inputs(int total_qumodes, int cutoff, int max_branches) {
    if (total_qumodes <= 0) {
        throw std::invalid_argument("total_qumodes must be positive");
    }
    if (cutoff <= 0) {
        throw std::invalid_argument("cutoff must be positive");
    }
    if (max_branches <= 0) {
        throw std::invalid_argument("max_branches must be positive");
    }
}

void validate_target_qumode(int target_qumode, int total_qumodes) {
    if (target_qumode < 0 || target_qumode >= total_qumodes) {
        throw std::out_of_range("target qumode out of range");
    }
}

SymplecticGate embed_local_phase_rotations(int total_qumodes,
                                           const std::vector<int>& target_qumodes,
                                           const std::vector<double>& phase_rotation_thetas) {
    if (target_qumodes.size() != phase_rotation_thetas.size()) {
        throw std::invalid_argument("target_qumodes and phase_rotation_thetas size mismatch");
    }

    SymplecticGate embedded(total_qumodes);
    const int dim = 2 * total_qumodes;
    std::vector<bool> used_modes(static_cast<size_t>(total_qumodes), false);

    for (size_t idx = 0; idx < target_qumodes.size(); ++idx) {
        const int target_qumode = target_qumodes[idx];
        validate_target_qumode(target_qumode, total_qumodes);
        if (used_modes[static_cast<size_t>(target_qumode)]) {
            throw std::invalid_argument("duplicate target qumode in Gaussian mixture branch");
        }
        used_modes[static_cast<size_t>(target_qumode)] = true;

        const SymplecticGate local_gate = SymplecticFactory::Rotation(phase_rotation_thetas[idx]);
        const int row = 2 * target_qumode;
        embedded.S[row * dim + row] = local_gate.S[0];
        embedded.S[row * dim + row + 1] = local_gate.S[1];
        embedded.S[(row + 1) * dim + row] = local_gate.S[2];
        embedded.S[(row + 1) * dim + row + 1] = local_gate.S[3];
    }

    return embedded;
}

GaussianMixtureBranch build_phase_rotation_branch(Complex weight,
                                                  int total_qumodes,
                                                  const std::vector<int>& target_qumodes,
                                                  const std::vector<double>& phase_rotation_thetas) {
    std::vector<int> effective_targets;
    std::vector<double> effective_thetas;
    effective_targets.reserve(target_qumodes.size());
    effective_thetas.reserve(phase_rotation_thetas.size());

    for (size_t idx = 0; idx < target_qumodes.size(); ++idx) {
        if (std::abs(phase_rotation_thetas[idx]) < kMixtureTolerance) {
            continue;
        }
        effective_targets.push_back(target_qumodes[idx]);
        effective_thetas.push_back(phase_rotation_thetas[idx]);
    }

    return {
        weight,
        embed_local_phase_rotations(total_qumodes, effective_targets, effective_thetas),
        effective_targets,
        effective_thetas
    };
}

std::vector<Complex> build_exact_kerr_diagonal(double chi, int cutoff) {
    std::vector<Complex> diagonal(static_cast<size_t>(cutoff), Complex(0.0, 0.0));
    for (int n = 0; n < cutoff; ++n) {
        diagonal[static_cast<size_t>(n)] = std::exp(Complex(0.0, -chi * n * n));
    }
    return diagonal;
}

std::vector<Complex> build_exact_snap_diagonal(double theta, int target_fock_state, int cutoff) {
    if (target_fock_state < 0 || target_fock_state >= cutoff) {
        throw std::out_of_range("target_fock_state out of range");
    }

    std::vector<Complex> diagonal(static_cast<size_t>(cutoff), Complex(1.0, 0.0));
    diagonal[static_cast<size_t>(target_fock_state)] = std::exp(Complex(0.0, theta));
    return diagonal;
}

std::vector<Complex> build_exact_multisnap_diagonal(const std::vector<double>& phase_map, int cutoff) {
    std::vector<Complex> diagonal(static_cast<size_t>(cutoff), Complex(1.0, 0.0));
    const int applied_size = std::min(cutoff, static_cast<int>(phase_map.size()));
    for (int n = 0; n < applied_size; ++n) {
        diagonal[static_cast<size_t>(n)] = std::exp(Complex(0.0, phase_map[static_cast<size_t>(n)]));
    }
    return diagonal;
}

std::vector<Complex> build_exact_conditional_parity_diagonal(double parity, int cutoff) {
    std::vector<Complex> diagonal(static_cast<size_t>(cutoff), Complex(1.0, 0.0));
    for (int n = 0; n < cutoff; ++n) {
        diagonal[static_cast<size_t>(n)] =
            std::exp(Complex(0.0, -parity * M_PI * static_cast<double>(n % 2)));
    }
    return diagonal;
}

std::vector<Complex> build_exact_cross_kerr_diagonal(double kappa, int cutoff) {
    std::vector<Complex> diagonal(static_cast<size_t>(cutoff) * cutoff, Complex(0.0, 0.0));
    for (int first = 0; first < cutoff; ++first) {
        for (int second = 0; second < cutoff; ++second) {
            const size_t index = static_cast<size_t>(first) * cutoff + static_cast<size_t>(second);
            diagonal[index] = std::exp(Complex(0.0, kappa * static_cast<double>(first * second)));
        }
    }
    return diagonal;
}

std::vector<Complex> compute_dft_coefficients(const std::vector<Complex>& diagonal) {
    const int cutoff = static_cast<int>(diagonal.size());
    std::vector<Complex> coefficients(static_cast<size_t>(cutoff), Complex(0.0, 0.0));

    for (int freq = 0; freq < cutoff; ++freq) {
        Complex sum(0.0, 0.0);
        for (int n = 0; n < cutoff; ++n) {
            const double theta =
                2.0 * M_PI * static_cast<double>(freq * n) / static_cast<double>(cutoff);
            sum += diagonal[static_cast<size_t>(n)] * std::exp(Complex(0.0, theta));
        }
        coefficients[static_cast<size_t>(freq)] = sum / static_cast<double>(cutoff);
    }

    return coefficients;
}

std::vector<Complex> reconstruct_diagonal(const std::vector<Complex>& coefficients, int cutoff) {
    std::vector<Complex> diagonal(static_cast<size_t>(cutoff), Complex(0.0, 0.0));
    for (int n = 0; n < cutoff; ++n) {
        Complex value(0.0, 0.0);
        for (int freq = 0; freq < cutoff; ++freq) {
            if (std::abs(coefficients[static_cast<size_t>(freq)]) < kMixtureTolerance) {
                continue;
            }
            const double theta =
                2.0 * M_PI * static_cast<double>(freq * n) / static_cast<double>(cutoff);
            value += coefficients[static_cast<size_t>(freq)] * std::exp(Complex(0.0, -theta));
        }
        diagonal[static_cast<size_t>(n)] = value;
    }
    return diagonal;
}

std::vector<Complex> compute_two_mode_dft_coefficients(const std::vector<Complex>& diagonal, int cutoff) {
    const size_t expected_size = static_cast<size_t>(cutoff) * cutoff;
    if (diagonal.size() != expected_size) {
        throw std::invalid_argument("two-mode diagonal size must be cutoff^2");
    }

    std::vector<Complex> coefficients(expected_size, Complex(0.0, 0.0));
    const double norm = static_cast<double>(cutoff * cutoff);

    for (int freq_first = 0; freq_first < cutoff; ++freq_first) {
        for (int freq_second = 0; freq_second < cutoff; ++freq_second) {
            Complex sum(0.0, 0.0);
            for (int first = 0; first < cutoff; ++first) {
                for (int second = 0; second < cutoff; ++second) {
                    const size_t index =
                        static_cast<size_t>(first) * cutoff + static_cast<size_t>(second);
                    const double theta =
                        2.0 * M_PI *
                        static_cast<double>(freq_first * first + freq_second * second) /
                        static_cast<double>(cutoff);
                    sum += diagonal[index] * std::exp(Complex(0.0, theta));
                }
            }
            coefficients[static_cast<size_t>(freq_first) * cutoff + static_cast<size_t>(freq_second)] =
                sum / norm;
        }
    }

    return coefficients;
}

std::vector<Complex> reconstruct_two_mode_diagonal(const std::vector<Complex>& coefficients, int cutoff) {
    const size_t expected_size = static_cast<size_t>(cutoff) * cutoff;
    if (coefficients.size() != expected_size) {
        throw std::invalid_argument("two-mode coefficient size must be cutoff^2");
    }

    std::vector<Complex> diagonal(expected_size, Complex(0.0, 0.0));
    for (int first = 0; first < cutoff; ++first) {
        for (int second = 0; second < cutoff; ++second) {
            Complex value(0.0, 0.0);
            for (int freq_first = 0; freq_first < cutoff; ++freq_first) {
                for (int freq_second = 0; freq_second < cutoff; ++freq_second) {
                    const size_t coefficient_index =
                        static_cast<size_t>(freq_first) * cutoff + static_cast<size_t>(freq_second);
                    if (std::abs(coefficients[coefficient_index]) < kMixtureTolerance) {
                        continue;
                    }
                    const double theta =
                        2.0 * M_PI *
                        static_cast<double>(freq_first * first + freq_second * second) /
                        static_cast<double>(cutoff);
                    value += coefficients[coefficient_index] * std::exp(Complex(0.0, -theta));
                }
            }
            diagonal[static_cast<size_t>(first) * cutoff + static_cast<size_t>(second)] = value;
        }
    }
    return diagonal;
}

void finalize_approximation(GaussianMixtureApproximation* approximation) {
    double l2_error_sq = 0.0;
    double max_error = 0.0;
    for (size_t idx = 0; idx < approximation->target_diagonal.size(); ++idx) {
        const double error =
            std::abs(approximation->target_diagonal[idx] - approximation->approximated_diagonal[idx]);
        l2_error_sq += error * error;
        max_error = std::max(max_error, error);
    }

    approximation->l2_diagonal_error = std::sqrt(l2_error_sq);
    approximation->max_diagonal_error = max_error;
    approximation->conservative_fidelity_lower_bound =
        conservative_fidelity_lower_bound_from_max_error(max_error);
}

template <typename Transform>
std::vector<Complex> apply_single_mode_transform(const std::vector<Complex>& state,
                                                 int cutoff,
                                                 int num_qumodes,
                                                 int target_qumode,
                                                 Transform&& transform) {
    if (target_qumode < 0 || target_qumode >= num_qumodes) {
        throw std::out_of_range("single-mode transform target qumode out of range");
    }

    const size_t stride = integer_power(static_cast<size_t>(cutoff), num_qumodes - target_qumode - 1);
    const size_t prefix_count = integer_power(static_cast<size_t>(cutoff), target_qumode);
    std::vector<Complex> result(state.size(), Complex(0.0, 0.0));

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

}  // namespace

GaussianMixtureApproximation GaussianMixtureDecomposer::approximate_single_mode_diagonal(
    int total_qumodes,
    int target_qumode,
    const std::vector<std::complex<double>>& diagonal,
    int max_branches) {
    validate_common_inputs(total_qumodes, static_cast<int>(diagonal.size()), max_branches);
    validate_target_qumode(target_qumode, total_qumodes);

    const int cutoff = static_cast<int>(diagonal.size());
    GaussianMixtureApproximation approximation;
    approximation.target_diagonal = diagonal;

    const std::vector<Complex> coefficients = compute_dft_coefficients(approximation.target_diagonal);
    std::vector<int> ordering(static_cast<size_t>(cutoff), 0);
    std::iota(ordering.begin(), ordering.end(), 0);
    std::sort(ordering.begin(), ordering.end(), [&coefficients](int lhs, int rhs) {
        return std::abs(coefficients[static_cast<size_t>(lhs)]) >
               std::abs(coefficients[static_cast<size_t>(rhs)]);
    });

    const int kept_branches = std::min(cutoff, max_branches);
    std::vector<Complex> truncated_coefficients(static_cast<size_t>(cutoff), Complex(0.0, 0.0));
    approximation.branches.reserve(static_cast<size_t>(kept_branches));

    for (int idx = 0; idx < kept_branches; ++idx) {
        const int freq = ordering[static_cast<size_t>(idx)];
        const Complex weight = coefficients[static_cast<size_t>(freq)];
        if (std::abs(weight) < kMixtureTolerance) {
            continue;
        }

        const double theta = 2.0 * M_PI * static_cast<double>(freq) / static_cast<double>(cutoff);
        truncated_coefficients[static_cast<size_t>(freq)] = weight;
        approximation.branches.push_back(
            build_phase_rotation_branch(weight, total_qumodes, {target_qumode}, {theta}));
    }

    approximation.approximated_diagonal = reconstruct_diagonal(truncated_coefficients, cutoff);
    finalize_approximation(&approximation);
    return approximation;
}

GaussianMixtureApproximation GaussianMixtureDecomposer::approximate_kerr_gate(
    int total_qumodes,
    int target_qumode,
    double chi,
    int cutoff,
    int max_branches) {
    return approximate_single_mode_diagonal(
        total_qumodes,
        target_qumode,
        build_exact_kerr_diagonal(chi, cutoff),
        max_branches);
}

GaussianMixtureApproximation GaussianMixtureDecomposer::approximate_snap_gate(
    int total_qumodes,
    int target_qumode,
    double theta,
    int target_fock_state,
    int cutoff,
    int max_branches) {
    validate_common_inputs(total_qumodes, cutoff, max_branches);
    validate_target_qumode(target_qumode, total_qumodes);
    return approximate_single_mode_diagonal(
        total_qumodes,
        target_qumode,
        build_exact_snap_diagonal(theta, target_fock_state, cutoff),
        max_branches);
}

GaussianMixtureApproximation GaussianMixtureDecomposer::approximate_multisnap_gate(
    int total_qumodes,
    int target_qumode,
    const std::vector<double>& phase_map,
    int cutoff,
    int max_branches) {
    validate_common_inputs(total_qumodes, cutoff, max_branches);
    validate_target_qumode(target_qumode, total_qumodes);
    return approximate_single_mode_diagonal(
        total_qumodes,
        target_qumode,
        build_exact_multisnap_diagonal(phase_map, cutoff),
        max_branches);
}

GaussianMixtureApproximation GaussianMixtureDecomposer::approximate_conditional_parity_gate(
    int total_qumodes,
    int target_qumode,
    double parity,
    int cutoff,
    int max_branches) {
    validate_common_inputs(total_qumodes, cutoff, max_branches);
    validate_target_qumode(target_qumode, total_qumodes);
    return approximate_single_mode_diagonal(
        total_qumodes,
        target_qumode,
        build_exact_conditional_parity_diagonal(parity, cutoff),
        max_branches);
}

GaussianMixtureApproximation GaussianMixtureDecomposer::approximate_two_mode_diagonal(
    int total_qumodes,
    int first_target_qumode,
    int second_target_qumode,
    const std::vector<std::complex<double>>& diagonal,
    int cutoff,
    int max_branches) {
    validate_common_inputs(total_qumodes, cutoff, max_branches);
    validate_target_qumode(first_target_qumode, total_qumodes);
    validate_target_qumode(second_target_qumode, total_qumodes);
    if (first_target_qumode == second_target_qumode) {
        throw std::invalid_argument("two-mode diagonal requires distinct target qumodes");
    }
    if (diagonal.size() != static_cast<size_t>(cutoff) * cutoff) {
        throw std::invalid_argument("two-mode diagonal size must equal cutoff^2");
    }

    GaussianMixtureApproximation approximation;
    approximation.target_diagonal = diagonal;

    const std::vector<Complex> coefficients =
        compute_two_mode_dft_coefficients(approximation.target_diagonal, cutoff);
    std::vector<int> ordering(coefficients.size(), 0);
    std::iota(ordering.begin(), ordering.end(), 0);
    std::sort(ordering.begin(), ordering.end(), [&coefficients](int lhs, int rhs) {
        return std::abs(coefficients[static_cast<size_t>(lhs)]) >
               std::abs(coefficients[static_cast<size_t>(rhs)]);
    });

    const int kept_branches = std::min(static_cast<int>(coefficients.size()), max_branches);
    std::vector<Complex> truncated_coefficients(coefficients.size(), Complex(0.0, 0.0));
    approximation.branches.reserve(static_cast<size_t>(kept_branches));

    for (int idx = 0; idx < kept_branches; ++idx) {
        const int flat_frequency = ordering[static_cast<size_t>(idx)];
        const int first_frequency = flat_frequency / cutoff;
        const int second_frequency = flat_frequency % cutoff;
        const Complex weight = coefficients[static_cast<size_t>(flat_frequency)];
        if (std::abs(weight) < kMixtureTolerance) {
            continue;
        }

        const double theta_first =
            2.0 * M_PI * static_cast<double>(first_frequency) / static_cast<double>(cutoff);
        const double theta_second =
            2.0 * M_PI * static_cast<double>(second_frequency) / static_cast<double>(cutoff);
        truncated_coefficients[static_cast<size_t>(flat_frequency)] = weight;
        approximation.branches.push_back(build_phase_rotation_branch(
            weight,
            total_qumodes,
            {first_target_qumode, second_target_qumode},
            {theta_first, theta_second}));
    }

    approximation.approximated_diagonal =
        reconstruct_two_mode_diagonal(truncated_coefficients, cutoff);
    finalize_approximation(&approximation);
    return approximation;
}

GaussianMixtureApproximation GaussianMixtureDecomposer::approximate_cross_kerr_gate(
    int total_qumodes,
    int first_target_qumode,
    int second_target_qumode,
    double kappa,
    int cutoff,
    int max_branches) {
    return approximate_two_mode_diagonal(
        total_qumodes,
        first_target_qumode,
        second_target_qumode,
        build_exact_cross_kerr_diagonal(kappa, cutoff),
        cutoff,
        max_branches);
}

GaussianMixtureApproximation GaussianMixtureDecomposer::approximate_cz_gate(
    int total_qumodes,
    int first_target_qumode,
    int second_target_qumode,
    double s,
    int cutoff,
    int max_branches) {
    return approximate_cross_kerr_gate(
        total_qumodes, first_target_qumode, second_target_qumode, s, cutoff, max_branches);
}

std::vector<std::complex<double>> GaussianMixtureDecomposer::apply_to_fock_state(
    const std::vector<std::complex<double>>& input_state,
    int cutoff,
    int num_qumodes,
    const GaussianMixtureApproximation& approximation) {
    const size_t expected_dim = integer_power(static_cast<size_t>(cutoff), num_qumodes);
    if (input_state.size() != expected_dim) {
        throw std::invalid_argument("input_state size does not match cutoff^num_qumodes");
    }

    std::vector<Complex> output(input_state.size(), Complex(0.0, 0.0));
    for (const GaussianMixtureBranch& branch : approximation.branches) {
        if (branch.gaussian_gate.num_qumodes != num_qumodes) {
            throw std::invalid_argument("mixture branch qumode count does not match state");
        }
        if (branch.target_qumodes.size() != branch.phase_rotation_thetas.size()) {
            throw std::invalid_argument("mixture branch target/theta size mismatch");
        }

        std::vector<Complex> transformed = input_state;
        for (size_t idx = 0; idx < branch.target_qumodes.size(); ++idx) {
            const double theta = branch.phase_rotation_thetas[idx];
            if (std::abs(theta) < kMixtureTolerance) {
                continue;
            }
            transformed = apply_single_mode_transform(
                transformed,
                cutoff,
                num_qumodes,
                branch.target_qumodes[idx],
                [theta](const Reference::Vector& local_state) {
                    return Reference::DiagonalGates::apply_phase_rotation(local_state, theta);
                });
        }

        for (size_t idx = 0; idx < output.size(); ++idx) {
            output[idx] += branch.weight * transformed[idx];
        }
    }

    return output;
}
