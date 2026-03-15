#include "gaussian_mixture.h"
#include "reference_gates.h"

#include <complex>
#include <iostream>
#include <string>
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

Reference::Vector vacuum_state(int cutoff) {
    Reference::Vector vacuum(static_cast<size_t>(cutoff), Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);
    return vacuum;
}

std::vector<Complex> build_single_mode_state(const std::string& state_name, int cutoff) {
    const Reference::Vector vacuum = vacuum_state(cutoff);

    if (state_name == "coherent") {
        const Reference::Vector state =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, Reference::Complex(0.9, 0.1));
        return std::vector<Complex>(state.begin(), state.end());
    }
    if (state_name == "squeezed") {
        const Reference::Vector state =
            Reference::SingleModeGates::apply_squeezing_gate(vacuum, Reference::Complex(0.18, 0.0));
        return std::vector<Complex>(state.begin(), state.end());
    }

    Reference::Vector state =
        Reference::SingleModeGates::apply_squeezing_gate(vacuum, Reference::Complex(0.15, 0.05));
    state = Reference::SingleModeGates::apply_displacement_gate(state, Reference::Complex(0.6, -0.2));
    return std::vector<Complex>(state.begin(), state.end());
}

std::vector<Complex> build_two_mode_state(const std::string& state_name, int cutoff) {
    const Reference::Vector vacuum = vacuum_state(cutoff);

    if (state_name == "coherent_x_coherent") {
        const Reference::Vector first =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, Reference::Complex(0.65, 0.0));
        const Reference::Vector second =
            Reference::SingleModeGates::apply_displacement_gate(vacuum, Reference::Complex(0.35, -0.1));
        const Reference::Vector state = Reference::tensor_product(first, second);
        return std::vector<Complex>(state.begin(), state.end());
    }

    const Reference::Vector first =
        Reference::SingleModeGates::apply_squeezing_gate(vacuum, Reference::Complex(0.14, 0.02));
    const Reference::Vector second =
        Reference::SingleModeGates::apply_displacement_gate(vacuum, Reference::Complex(0.45, 0.15));
    const Reference::Vector state = Reference::tensor_product(first, second);
    return std::vector<Complex>(state.begin(), state.end());
}

std::vector<Complex> exact_kerr(const std::vector<Complex>& state, int cutoff, double chi) {
    return apply_single_mode_transform(
        state,
        cutoff,
        1,
        0,
        [chi](const Reference::Vector& local_state) {
            return Reference::DiagonalGates::apply_kerr_gate(local_state, chi);
        });
}

std::vector<Complex> exact_snap(const std::vector<Complex>& state, int cutoff, double theta, int target_fock_state) {
    return apply_single_mode_transform(
        state,
        cutoff,
        1,
        0,
        [theta, target_fock_state](const Reference::Vector& local_state) {
            Reference::Vector transformed = local_state;
            transformed[static_cast<size_t>(target_fock_state)] *= std::exp(Complex(0.0, theta));
            return transformed;
        });
}

std::vector<Complex> exact_multisnap(const std::vector<Complex>& state,
                                     int cutoff,
                                     const std::vector<double>& phase_map) {
    return apply_single_mode_transform(
        state,
        cutoff,
        1,
        0,
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

std::vector<Complex> exact_conditional_parity(const std::vector<Complex>& state,
                                              int cutoff,
                                              double parity) {
    return apply_single_mode_transform(
        state,
        cutoff,
        1,
        0,
        [parity](const Reference::Vector& local_state) {
            return Reference::DiagonalGates::apply_conditional_parity(local_state, parity);
        });
}

std::vector<Complex> exact_cross_kerr(const std::vector<Complex>& state, int cutoff, double kappa) {
    return apply_two_mode_transform(
        state,
        cutoff,
        2,
        0,
        1,
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

void emit_row(const std::string& gate,
              const std::string& scenario,
              int k,
              const GaussianMixtureApproximation& approximation,
              const std::vector<Complex>& exact_state,
              const std::vector<Complex>& approx_state) {
    std::cout << gate << ','
              << scenario << ','
              << k << ','
              << approximation.l2_diagonal_error << ','
              << approximation.max_diagonal_error << ','
              << l2_error(exact_state, approx_state) << ','
              << Reference::fidelity(exact_state, approx_state) << '\n';
}

}  // namespace

int main() {
    const int single_mode_cutoff = 16;
    const int two_mode_cutoff = 8;
    const std::vector<int> single_mode_branch_counts = {1, 2, 4, 6, 8, 12, 16};
    const std::vector<int> two_mode_branch_counts = {1, 4, 8, 16, 32, 64};
    const std::vector<std::string> single_mode_states = {"coherent", "squeezed", "displaced_squeezed"};
    const std::vector<std::string> two_mode_states = {"coherent_x_coherent", "squeezed_x_coherent"};

    const std::vector<double> kerr_chis = {0.01, 0.03, 0.05, 0.1};
    const std::vector<double> cross_kerr_kappas = {0.03, 0.08};
    const std::vector<double> multisnap_phase_map = {
        0.0, M_PI, M_PI / 2.0, 0.0, 0.2, -0.35, 0.0, 0.1,
        0.0, 0.0, -0.4, 0.0, 0.25, 0.0, 0.0, -0.15
    };

    std::cout << "gate,scenario,k,diag_l2,diag_max,state_l2,fidelity\n";

    for (const std::string& state_name : single_mode_states) {
        const std::vector<Complex> input_state = build_single_mode_state(state_name, single_mode_cutoff);

        for (double chi : kerr_chis) {
            const std::string scenario = state_name + ":chi=" + std::to_string(chi);
            const std::vector<Complex> exact_state = exact_kerr(input_state, single_mode_cutoff, chi);
            for (int k : single_mode_branch_counts) {
                const GaussianMixtureApproximation approximation =
                    GaussianMixtureDecomposer::approximate_kerr_gate(1, 0, chi, single_mode_cutoff, k);
                const std::vector<Complex> approx_state =
                    GaussianMixtureDecomposer::apply_to_fock_state(input_state, single_mode_cutoff, 1, approximation);
                emit_row("kerr", scenario, k, approximation, exact_state, approx_state);
            }
        }

        {
            const std::string scenario = state_name + ":theta=1.570796,target=5";
            const std::vector<Complex> exact_state =
                exact_snap(input_state, single_mode_cutoff, M_PI / 2.0, 5);
            for (int k : single_mode_branch_counts) {
                const GaussianMixtureApproximation approximation =
                    GaussianMixtureDecomposer::approximate_snap_gate(
                        1, 0, M_PI / 2.0, 5, single_mode_cutoff, k);
                const std::vector<Complex> approx_state =
                    GaussianMixtureDecomposer::apply_to_fock_state(input_state, single_mode_cutoff, 1, approximation);
                emit_row("snap", scenario, k, approximation, exact_state, approx_state);
            }
        }

        {
            const std::string scenario = state_name + ":custom_phase_map";
            const std::vector<Complex> exact_state =
                exact_multisnap(input_state, single_mode_cutoff, multisnap_phase_map);
            for (int k : single_mode_branch_counts) {
                const GaussianMixtureApproximation approximation =
                    GaussianMixtureDecomposer::approximate_multisnap_gate(
                        1, 0, multisnap_phase_map, single_mode_cutoff, k);
                const std::vector<Complex> approx_state =
                    GaussianMixtureDecomposer::apply_to_fock_state(input_state, single_mode_cutoff, 1, approximation);
                emit_row("multisnap", scenario, k, approximation, exact_state, approx_state);
            }
        }

        {
            const std::string scenario = state_name + ":parity=0.37";
            const std::vector<Complex> exact_state =
                exact_conditional_parity(input_state, single_mode_cutoff, 0.37);
            for (int k : {1, 2, 4}) {
                const GaussianMixtureApproximation approximation =
                    GaussianMixtureDecomposer::approximate_conditional_parity_gate(
                        1, 0, 0.37, single_mode_cutoff, k);
                const std::vector<Complex> approx_state =
                    GaussianMixtureDecomposer::apply_to_fock_state(input_state, single_mode_cutoff, 1, approximation);
                emit_row("conditional_parity", scenario, k, approximation, exact_state, approx_state);
            }
        }
    }

    for (const std::string& state_name : two_mode_states) {
        const std::vector<Complex> input_state = build_two_mode_state(state_name, two_mode_cutoff);
        for (double kappa : cross_kerr_kappas) {
            const std::string scenario = state_name + ":kappa=" + std::to_string(kappa);
            const std::vector<Complex> exact_state = exact_cross_kerr(input_state, two_mode_cutoff, kappa);
            for (int k : two_mode_branch_counts) {
                const GaussianMixtureApproximation approximation =
                    GaussianMixtureDecomposer::approximate_cross_kerr_gate(2, 0, 1, kappa, two_mode_cutoff, k);
                const std::vector<Complex> approx_state =
                    GaussianMixtureDecomposer::apply_to_fock_state(input_state, two_mode_cutoff, 2, approximation);
                emit_row("cross_kerr", scenario, k, approximation, exact_state, approx_state);
            }
        }
    }

    return 0;
}
