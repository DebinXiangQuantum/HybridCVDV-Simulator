#include <gtest/gtest.h>

#include "gaussian_mixture.h"
#include "reference_gates.h"

#include <algorithm>
#include <complex>
#include <vector>

namespace {

using Complex = std::complex<double>;

size_t integer_power(size_t base, int exponent) {
    size_t result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

template <typename Transform>
std::vector<Complex> apply_single_mode_transform(const std::vector<Complex>& state,
                                                 int cutoff,
                                                 int num_qumodes,
                                                 int target_qumode,
                                                 Transform&& transform) {
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

template <typename Transform>
std::vector<Complex> apply_two_mode_transform(const std::vector<Complex>& state,
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
    std::vector<Complex> result(state.size(), Complex(0.0, 0.0));

    for (size_t outer_index = 0; outer_index < outer_count; ++outer_index) {
        size_t residual = outer_index;
        size_t base_index = 0;
        for (int idx = static_cast<int>(other_modes.size()) - 1; idx >= 0; --idx) {
            const int digit = static_cast<int>(residual % static_cast<size_t>(cutoff));
            residual /= static_cast<size_t>(cutoff);
            base_index += static_cast<size_t>(digit) *
                          strides[static_cast<size_t>(other_modes[static_cast<size_t>(idx)])];
        }

        std::vector<Complex> local_state(static_cast<size_t>(cutoff * cutoff), Complex(0.0, 0.0));
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

        const std::vector<Complex> transformed = transform(local_state);
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

std::vector<Complex> build_coherent_state(int cutoff, Complex alpha) {
    Reference::Vector vacuum(static_cast<size_t>(cutoff), Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);
    const Reference::Vector coherent = Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
    return std::vector<Complex>(coherent.begin(), coherent.end());
}

std::vector<Complex> build_product_state(const std::vector<Reference::Vector>& modes) {
    Reference::Vector state = modes.front();
    for (size_t idx = 1; idx < modes.size(); ++idx) {
        state = Reference::tensor_product(state, modes[idx]);
    }
    return std::vector<Complex>(state.begin(), state.end());
}

std::vector<Complex> exact_kerr_on_mode(const std::vector<Complex>& state,
                                        int cutoff,
                                        int num_qumodes,
                                        int target_qumode,
                                        double chi) {
    return apply_single_mode_transform(
        state,
        cutoff,
        num_qumodes,
        target_qumode,
        [chi](const Reference::Vector& local_state) {
            return Reference::DiagonalGates::apply_kerr_gate(local_state, chi);
        });
}

std::vector<Complex> exact_snap_on_mode(const std::vector<Complex>& state,
                                        int cutoff,
                                        int num_qumodes,
                                        int target_qumode,
                                        double theta,
                                        int target_fock_state) {
    return apply_single_mode_transform(
        state,
        cutoff,
        num_qumodes,
        target_qumode,
        [theta, target_fock_state](const Reference::Vector& local_state) {
            Reference::Vector transformed = local_state;
            transformed[static_cast<size_t>(target_fock_state)] *= std::exp(Complex(0.0, theta));
            return transformed;
        });
}

std::vector<Complex> exact_multisnap_on_mode(const std::vector<Complex>& state,
                                             int cutoff,
                                             int num_qumodes,
                                             int target_qumode,
                                             const std::vector<double>& phase_map) {
    return apply_single_mode_transform(
        state,
        cutoff,
        num_qumodes,
        target_qumode,
        [&phase_map](const Reference::Vector& local_state) {
            Reference::Vector transformed = local_state;
            const int applied_size = std::min(
                static_cast<int>(local_state.size()),
                static_cast<int>(phase_map.size()));
            for (int photon = 0; photon < applied_size; ++photon) {
                transformed[static_cast<size_t>(photon)] *=
                    std::exp(Complex(0.0, phase_map[static_cast<size_t>(photon)]));
            }
            return transformed;
        });
}

std::vector<Complex> exact_conditional_parity_on_mode(const std::vector<Complex>& state,
                                                      int cutoff,
                                                      int num_qumodes,
                                                      int target_qumode,
                                                      double parity) {
    return apply_single_mode_transform(
        state,
        cutoff,
        num_qumodes,
        target_qumode,
        [parity](const Reference::Vector& local_state) {
            return Reference::DiagonalGates::apply_conditional_parity(local_state, parity);
        });
}

std::vector<Complex> exact_cross_kerr_on_modes(const std::vector<Complex>& state,
                                               int cutoff,
                                               int num_qumodes,
                                               int first_qumode,
                                               int second_qumode,
                                               double kappa) {
    return apply_two_mode_transform(
        state,
        cutoff,
        num_qumodes,
        first_qumode,
        second_qumode,
        [cutoff, kappa](const std::vector<Complex>& local_state) {
            std::vector<Complex> transformed = local_state;
            for (int first = 0; first < cutoff; ++first) {
                for (int second = 0; second < cutoff; ++second) {
                    const size_t index =
                        static_cast<size_t>(first) * cutoff + static_cast<size_t>(second);
                    transformed[index] *=
                        std::exp(Complex(0.0, kappa * static_cast<double>(first * second)));
                }
            }
            return transformed;
        });
}

double l2_error(const std::vector<Complex>& lhs, const std::vector<Complex>& rhs) {
    double error_sq = 0.0;
    for (size_t idx = 0; idx < lhs.size(); ++idx) {
        error_sq += std::norm(lhs[idx] - rhs[idx]);
    }
    return std::sqrt(error_sq);
}

}  // namespace

TEST(GaussianMixtureTest, KerrMixtureIsExactAtFullCutoffRank) {
    constexpr int cutoff = 12;
    const double chi = 0.17;

    const GaussianMixtureApproximation approximation =
        GaussianMixtureDecomposer::approximate_kerr_gate(1, 0, chi, cutoff, cutoff);

    EXPECT_LT(approximation.l2_diagonal_error, 1e-10);
    EXPECT_LT(approximation.max_diagonal_error, 1e-10);
    EXPECT_GT(approximation.conservative_fidelity_lower_bound, 1.0 - 1e-10);

    const std::vector<Complex> input_state = build_coherent_state(cutoff, Complex(0.7, -0.2));
    const std::vector<Complex> exact_state = exact_kerr_on_mode(input_state, cutoff, 1, 0, chi);
    const std::vector<Complex> approx_state =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 1, approximation);

    EXPECT_LT(l2_error(exact_state, approx_state), 1e-10);
}

TEST(GaussianMixtureTest, SnapMixtureIsExactAtFullCutoffRank) {
    constexpr int cutoff = 12;
    const std::vector<Complex> input_state = build_coherent_state(cutoff, Complex(0.6, 0.3));

    const GaussianMixtureApproximation approximation =
        GaussianMixtureDecomposer::approximate_snap_gate(1, 0, M_PI / 3.0, 5, cutoff, cutoff);

    EXPECT_LT(approximation.l2_diagonal_error, 1e-10);
    EXPECT_LT(approximation.max_diagonal_error, 1e-10);
    EXPECT_GT(approximation.conservative_fidelity_lower_bound, 1.0 - 1e-10);

    const std::vector<Complex> exact_state =
        exact_snap_on_mode(input_state, cutoff, 1, 0, M_PI / 3.0, 5);
    const std::vector<Complex> approx_state =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 1, approximation);

    EXPECT_LT(l2_error(exact_state, approx_state), 1e-10);
}

TEST(GaussianMixtureTest, MultiSnapMixtureAccuracyImprovesWithBranchCount) {
    constexpr int cutoff = 16;
    const std::vector<double> phase_map = {
        0.0, M_PI, M_PI / 2.0, 0.0, 0.2, -0.35, 0.0, 0.1,
        0.0, 0.0, -0.4, 0.0, 0.25, 0.0, 0.0, -0.15
    };
    const std::vector<Complex> input_state = build_coherent_state(cutoff, Complex(0.85, -0.1));
    const std::vector<Complex> exact_state =
        exact_multisnap_on_mode(input_state, cutoff, 1, 0, phase_map);

    const GaussianMixtureApproximation approximation_k4 =
        GaussianMixtureDecomposer::approximate_multisnap_gate(1, 0, phase_map, cutoff, 4);
    const GaussianMixtureApproximation approximation_k8 =
        GaussianMixtureDecomposer::approximate_multisnap_gate(1, 0, phase_map, cutoff, 8);
    const GaussianMixtureApproximation approximation_k12 =
        GaussianMixtureDecomposer::approximate_multisnap_gate(1, 0, phase_map, cutoff, 12);

    const std::vector<Complex> approx_state_k4 =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 1, approximation_k4);
    const std::vector<Complex> approx_state_k8 =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 1, approximation_k8);
    const std::vector<Complex> approx_state_k12 =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 1, approximation_k12);

    EXPECT_GT(approximation_k4.l2_diagonal_error, approximation_k8.l2_diagonal_error);
    EXPECT_GT(approximation_k8.l2_diagonal_error, approximation_k12.l2_diagonal_error);
    EXPECT_LT(approximation_k4.conservative_fidelity_lower_bound,
              approximation_k12.conservative_fidelity_lower_bound);

    const double fidelity_k4 = Reference::fidelity(exact_state, approx_state_k4);
    const double fidelity_k12 = Reference::fidelity(exact_state, approx_state_k12);
    EXPECT_GT(fidelity_k12, fidelity_k4);
    EXPECT_GT(fidelity_k12, 0.96);
}

TEST(GaussianMixtureTest, ConditionalParityMixtureIsExactWithTwoBranches) {
    constexpr int cutoff = 12;
    const double parity = 0.37;
    const std::vector<Complex> input_state = build_coherent_state(cutoff, Complex(0.5, 0.2));

    const GaussianMixtureApproximation approximation =
        GaussianMixtureDecomposer::approximate_conditional_parity_gate(1, 0, parity, cutoff, 2);

    EXPECT_LT(approximation.l2_diagonal_error, 1e-10);
    EXPECT_LT(approximation.max_diagonal_error, 1e-10);
    EXPECT_GT(approximation.conservative_fidelity_lower_bound, 1.0 - 1e-10);

    const std::vector<Complex> exact_state =
        exact_conditional_parity_on_mode(input_state, cutoff, 1, 0, parity);
    const std::vector<Complex> approx_state =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 1, approximation);

    EXPECT_LT(l2_error(exact_state, approx_state), 1e-10);
}

TEST(GaussianMixtureTest, CrossKerrMixtureIsExactAtFullCutoffRank) {
    constexpr int cutoff = 8;
    const double kappa = 0.23;

    Reference::Vector vacuum(static_cast<size_t>(cutoff), Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);
    const Reference::Vector mode0 = Reference::SingleModeGates::apply_displacement_gate(vacuum, {0.45, 0.1});
    const Reference::Vector mode1 = Reference::SingleModeGates::apply_squeezing_gate(vacuum, {0.16, -0.02});
    const std::vector<Complex> input_state = build_product_state({mode0, mode1});

    const GaussianMixtureApproximation approximation =
        GaussianMixtureDecomposer::approximate_cross_kerr_gate(2, 0, 1, kappa, cutoff, cutoff * cutoff);

    EXPECT_LT(approximation.l2_diagonal_error, 1e-10);
    EXPECT_LT(approximation.max_diagonal_error, 1e-10);
    EXPECT_GT(approximation.conservative_fidelity_lower_bound, 1.0 - 1e-10);

    const std::vector<Complex> exact_state =
        exact_cross_kerr_on_modes(input_state, cutoff, 2, 0, 1, kappa);
    const std::vector<Complex> approx_state =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 2, approximation);

    EXPECT_LT(l2_error(exact_state, approx_state), 1e-10);
}

TEST(GaussianMixtureTest, CrossKerrMixtureAccuracyImprovesWithBranchCount) {
    constexpr int cutoff = 8;
    const double kappa = 0.08;

    Reference::Vector vacuum(static_cast<size_t>(cutoff), Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);
    const Reference::Vector mode0 = Reference::SingleModeGates::apply_displacement_gate(vacuum, {0.55, -0.15});
    const Reference::Vector mode1 = Reference::SingleModeGates::apply_displacement_gate(vacuum, {0.35, 0.05});
    const std::vector<Complex> input_state = build_product_state({mode0, mode1});
    const std::vector<Complex> exact_state =
        exact_cross_kerr_on_modes(input_state, cutoff, 2, 0, 1, kappa);

    const GaussianMixtureApproximation approximation_k4 =
        GaussianMixtureDecomposer::approximate_cross_kerr_gate(2, 0, 1, kappa, cutoff, 4);
    const GaussianMixtureApproximation approximation_k16 =
        GaussianMixtureDecomposer::approximate_cross_kerr_gate(2, 0, 1, kappa, cutoff, 16);
    const GaussianMixtureApproximation approximation_k32 =
        GaussianMixtureDecomposer::approximate_cross_kerr_gate(2, 0, 1, kappa, cutoff, 32);

    const std::vector<Complex> approx_state_k4 =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 2, approximation_k4);
    const std::vector<Complex> approx_state_k32 =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 2, approximation_k32);

    EXPECT_GT(approximation_k4.l2_diagonal_error, approximation_k16.l2_diagonal_error);
    EXPECT_GT(approximation_k16.l2_diagonal_error, approximation_k32.l2_diagonal_error);
    EXPECT_LT(approximation_k4.conservative_fidelity_lower_bound,
              approximation_k32.conservative_fidelity_lower_bound);

    const double fidelity_k4 = Reference::fidelity(exact_state, approx_state_k4);
    const double fidelity_k32 = Reference::fidelity(exact_state, approx_state_k32);
    EXPECT_GT(fidelity_k32, fidelity_k4);
    EXPECT_GT(fidelity_k32, 0.99);
}

TEST(GaussianMixtureTest, KerrMixtureTargetsSelectedModeInMultiModeState) {
    constexpr int cutoff = 10;
    const double chi = 0.05;

    Reference::Vector vacuum(static_cast<size_t>(cutoff), Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);
    const Reference::Vector mode0 = Reference::SingleModeGates::apply_displacement_gate(vacuum, {0.6, 0.0});
    const Reference::Vector mode1 = Reference::SingleModeGates::apply_squeezing_gate(vacuum, {0.12, 0.0});
    const Reference::Vector product_state = Reference::tensor_product(mode0, mode1);
    const std::vector<Complex> input_state(product_state.begin(), product_state.end());

    const GaussianMixtureApproximation approximation =
        GaussianMixtureDecomposer::approximate_kerr_gate(2, 1, chi, cutoff, cutoff);

    const std::vector<Complex> exact_state = exact_kerr_on_mode(input_state, cutoff, 2, 1, chi);
    const std::vector<Complex> approx_state =
        GaussianMixtureDecomposer::apply_to_fock_state(input_state, cutoff, 2, approximation);

    EXPECT_LT(l2_error(exact_state, approx_state), 1e-10);
}
