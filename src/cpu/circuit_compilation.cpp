// circuit_compilation.cpp — Gate classification, canonicalization, compilation

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

namespace circuit_internal {

void trim_trailing_zero_phases(std::vector<double>* phase_map) {
    while (!phase_map->empty() && !is_nontrivial_phase(phase_map->back())) {
        phase_map->pop_back();
    }
}

GateParams make_two_mode_gate(GateType type,
                              int first_target_qumode,
                              int second_target_qumode,
                              std::complex<double> param) {
    return GateParams(type, {}, {first_target_qumode, second_target_qumode}, {param});
}

bool is_unconditional_gaussian_gate(const GateParams& gate) {
    if (!gate.target_qubits.empty()) {
        return false;
    }

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 1;
        case GateType::DISPLACEMENT:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 1;
        case GateType::SQUEEZING:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 1;
        case GateType::BEAM_SPLITTER:
            return gate.target_qumodes.size() == 2 && gate.params.size() >= 1;
        default:
            return false;
    }
}

bool is_supported_controlled_gaussian_gate(const GateParams& gate) {
    switch (gate.type) {
        case GateType::CONDITIONAL_DISPLACEMENT:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 1 &&
                   gate.params.size() >= 1;
        case GateType::CONDITIONAL_SQUEEZING:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 1 &&
                   gate.params.size() >= 1;
        case GateType::CONDITIONAL_BEAM_SPLITTER:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 2 &&
                   gate.params.size() >= 1;
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 2 &&
                   gate.params.size() >= 1;
        case GateType::CONDITIONAL_SUM:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 2 &&
                   gate.params.size() >= 1 &&
                   (gate.params.size() < 2 || std::abs(gate.params[1]) < 1e-14);
        default:
            return false;
    }
}

bool is_gaussian_track_gate(const GateParams& gate) {
    return is_unconditional_gaussian_gate(gate) ||
           is_supported_controlled_gaussian_gate(gate);
}

bool is_pure_qubit_gate(const GateParams& gate) {
    if (!gate.target_qumodes.empty()) {
        return false;
    }

    switch (gate.type) {
        case GateType::HADAMARD:
        case GateType::PAULI_X:
        case GateType::PAULI_Y:
        case GateType::PAULI_Z:
        case GateType::ROTATION_X:
        case GateType::ROTATION_Y:
        case GateType::ROTATION_Z:
        case GateType::PHASE_GATE_S:
        case GateType::PHASE_GATE_T:
        case GateType::CNOT:
        case GateType::CZ:
            return !gate.target_qubits.empty();
        default:
            return false;
    }
}

const char* gate_type_name(GateType type) {
    switch (type) {
        case GateType::HADAMARD: return "Hadamard";
        case GateType::PAULI_X: return "PauliX";
        case GateType::PAULI_Y: return "PauliY";
        case GateType::PAULI_Z: return "PauliZ";
        case GateType::ROTATION_X: return "RotationX";
        case GateType::ROTATION_Y: return "RotationY";
        case GateType::ROTATION_Z: return "RotationZ";
        case GateType::PHASE_GATE_S: return "PhaseS";
        case GateType::PHASE_GATE_T: return "PhaseT";
        case GateType::CNOT: return "CNOT";
        case GateType::CZ: return "CZ";
        case GateType::PHASE_ROTATION: return "PhaseRotation";
        case GateType::KERR_GATE: return "Kerr";
        case GateType::CONDITIONAL_PARITY: return "ConditionalParity";
        case GateType::SNAP_GATE: return "SNAP";
        case GateType::MULTI_SNAP_GATE: return "MultiSNAP";
        case GateType::CROSS_KERR_GATE: return "CrossKerr";
        case GateType::CREATION_OPERATOR: return "Creation";
        case GateType::ANNIHILATION_OPERATOR: return "Annihilation";
        case GateType::DISPLACEMENT: return "Displacement";
        case GateType::SQUEEZING: return "Squeezing";
        case GateType::BEAM_SPLITTER: return "BeamSplitter";
        case GateType::CONDITIONAL_DISPLACEMENT: return "ConditionalDisplacement";
        case GateType::CONDITIONAL_SQUEEZING: return "ConditionalSqueezing";
        case GateType::CONDITIONAL_BEAM_SPLITTER: return "ConditionalBeamSplitter";
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING: return "ConditionalTwoModeSqueezing";
        case GateType::CONDITIONAL_SUM: return "ConditionalSUM";
        case GateType::RABI_INTERACTION: return "RabiInteraction";
        case GateType::JAYNES_CUMMINGS: return "JaynesCummings";
        case GateType::ANTI_JAYNES_CUMMINGS: return "AntiJaynesCummings";
        case GateType::SELECTIVE_QUBIT_ROTATION: return "SelectiveQubitRotation";
        default: return "UnknownGate";
    }
}

bool validate_gaussian_track_gate(const GateParams& gate,
                                  int total_qubits,
                                  int total_qumodes,
                                  std::string* error_message) {
    auto fail = [error_message](const std::string& message) {
        if (error_message) {
            *error_message = message;
        }
        return false;
    };

    auto validate_qumode = [&](int qumode) {
        return qumode >= 0 && qumode < total_qumodes;
    };

    auto validate_single_qumode_gate =
        [&](const char* gate_name, bool requires_control, bool require_two_modes) {
            const size_t expected_qumodes = require_two_modes ? 2 : 1;
            if (gate.target_qumodes.size() != expected_qumodes) {
                return fail(std::string(gate_name) + " 目标qumode数量不正确");
            }
            for (int qumode : gate.target_qumodes) {
                if (!validate_qumode(qumode)) {
                    return fail(std::string(gate_name) + " 目标qumode越界");
                }
            }
            if (require_two_modes && gate.target_qumodes[0] == gate.target_qumodes[1]) {
                return fail(std::string(gate_name) + " 需要两个不同的目标qumode");
            }
            if (gate.params.empty()) {
                return fail(std::string(gate_name) + " 缺少参数");
            }
            if (requires_control) {
                if (gate.target_qubits.size() != 1) {
                    return fail(std::string(gate_name) + " 需要且仅需要一个控制qubit");
                }
                const int control_qubit = gate.target_qubits[0];
                if (control_qubit < 0 || control_qubit >= total_qubits) {
                    return fail(std::string(gate_name) + " 控制qubit越界");
                }
            } else if (!gate.target_qubits.empty()) {
                return fail(std::string(gate_name) + " 不应携带qubit目标");
            }
            return true;
        };

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            return validate_single_qumode_gate("PhaseRotation", false, false);
        case GateType::DISPLACEMENT:
            return validate_single_qumode_gate("Displacement", false, false);
        case GateType::SQUEEZING:
            return validate_single_qumode_gate("Squeezing", false, false);
        case GateType::BEAM_SPLITTER:
            return validate_single_qumode_gate("BeamSplitter", false, true);
        case GateType::CONDITIONAL_DISPLACEMENT:
            return validate_single_qumode_gate("ConditionalDisplacement", true, false);
        case GateType::CONDITIONAL_SQUEEZING:
            return validate_single_qumode_gate("ConditionalSqueezing", true, false);
        case GateType::CONDITIONAL_BEAM_SPLITTER:
            return validate_single_qumode_gate("ConditionalBeamSplitter", true, true);
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
            return validate_single_qumode_gate("ConditionalTwoModeSqueezing", true, true);
        case GateType::CONDITIONAL_SUM:
            if (!validate_single_qumode_gate("ConditionalSUM", true, true)) {
                return false;
            }
            if (gate.params.size() >= 2 && std::abs(gate.params[1]) > 1e-14) {
                return fail("ConditionalSUM 当前仅支持 phi = 0 进入Gaussian track");
            }
            return true;
        default:
            return fail("该门当前不能进入Gaussian track");
    }
}

std::vector<int> collect_gaussian_control_qubits(const std::vector<GateParams>& gates) {
    std::unordered_set<int> seen_controls;
    std::vector<int> control_qubits;
    for (const GateParams& gate : gates) {
        if (is_supported_controlled_gaussian_gate(gate) &&
            seen_controls.insert(gate.target_qubits[0]).second) {
            control_qubits.push_back(gate.target_qubits[0]);
        }
    }
    std::sort(control_qubits.begin(), control_qubits.end());
    return control_qubits;
}

std::string make_control_assignment_key(const std::vector<int>& control_qubits,
                                        const std::vector<int>& qubit_assignment) {
    std::string key;
    key.reserve(control_qubits.size());
    for (int control_qubit : control_qubits) {
        key.push_back(qubit_assignment[control_qubit] == 0 ? '0' : '1');
    }
    return key;
}

std::vector<int> assignment_from_key(const std::vector<int>& control_qubits,
                                     const std::string& key,
                                     int total_qubits) {
    std::vector<int> assignment(static_cast<size_t>(total_qubits), 0);
    for (size_t idx = 0; idx < control_qubits.size(); ++idx) {
        assignment[control_qubits[idx]] = key[idx] == '1' ? 1 : 0;
    }
    return assignment;
}

std::optional<GateParams> resolve_gaussian_gate_for_assignment(
    const GateParams& gate,
    const std::vector<int>& qubit_assignment) {
    if (is_unconditional_gaussian_gate(gate)) {
        return gate;
    }

    switch (gate.type) {
        case GateType::CONDITIONAL_DISPLACEMENT: {
            const int control_qubit = gate.target_qubits[0];
            if (qubit_assignment[control_qubit] == 0) {
                return std::nullopt;
            }
            return GateParams(
                GateType::DISPLACEMENT,
                {},
                {gate.target_qumodes[0]},
                {gate.params[0]});
        }
        case GateType::CONDITIONAL_SQUEEZING: {
            const int control_qubit = gate.target_qubits[0];
            const std::complex<double> xi =
                qubit_assignment[control_qubit] == 0 ? gate.params[0] : -gate.params[0];
            return GateParams(
                GateType::SQUEEZING,
                {},
                {gate.target_qumodes[0]},
                {xi});
        }
        case GateType::CONDITIONAL_BEAM_SPLITTER: {
            const int control_qubit = gate.target_qubits[0];
            const double theta =
                qubit_assignment[control_qubit] == 0 ?
                gate.params[0].real() :
                -gate.params[0].real();
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return GateParams(
                GateType::BEAM_SPLITTER,
                {},
                {gate.target_qumodes[0], gate.target_qumodes[1]},
                {
                    std::complex<double>(theta, 0.0),
                    std::complex<double>(phi, 0.0)
                });
        }
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING: {
            const int control_qubit = gate.target_qubits[0];
            const std::complex<double> xi =
                qubit_assignment[control_qubit] == 0 ? gate.params[0] : -gate.params[0];
            return GateParams(
                GateType::CONDITIONAL_TWO_MODE_SQUEEZING,
                {},
                {gate.target_qumodes[0], gate.target_qumodes[1]},
                {xi});
        }
        case GateType::CONDITIONAL_SUM: {
            const int control_qubit = gate.target_qubits[0];
            const double theta =
                qubit_assignment[control_qubit] == 0 ?
                gate.params[0].real() :
                -gate.params[0].real();
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return GateParams(
                GateType::CONDITIONAL_SUM,
                {},
                {gate.target_qumodes[0], gate.target_qumodes[1]},
                {
                    std::complex<double>(theta, 0.0),
                    std::complex<double>(phi, 0.0)
                });
        }
        default:
            return std::nullopt;
    }
}

void scale_cuda_state(std::vector<cuDoubleComplex>* state,
                      std::complex<double> scale) {
    const cuDoubleComplex scale_cu = make_cuDoubleComplex(scale.real(), scale.imag());
    for (cuDoubleComplex& amplitude : *state) {
        amplitude = cuCmul(scale_cu, amplitude);
    }
}

void collect_symbolic_terminal_ids_recursive(
    HDDNode* node,
    std::unordered_set<size_t>& visited_nodes,
    std::unordered_set<int>& symbolic_ids) {
    if (!node) {
        return;
    }
    if (!visited_nodes.insert(node->get_unique_id()).second) {
        return;
    }
    if (node->is_terminal()) {
        if (node->tensor_id < -1) {
            symbolic_ids.insert(node->tensor_id);
        }
        return;
    }
    collect_symbolic_terminal_ids_recursive(node->low, visited_nodes, symbolic_ids);
    collect_symbolic_terminal_ids_recursive(node->high, visited_nodes, symbolic_ids);
}

int choose_snap_mixture_branch_cap(const GateParams& gate, int cutoff) {
    if (cutoff <= 0) {
        return 0;
    }

    if (gate.type == GateType::SNAP_GATE) {
        return std::min(cutoff, kMaxSnapMixtureBranches);
    }

    if (gate.type == GateType::MULTI_SNAP_GATE) {
        return std::min(cutoff, kMaxSnapMixtureBranches);
    }

    return 0;
}

int choose_cross_kerr_mixture_branch_cap(int cutoff) {
    if (cutoff <= 0) {
        return 0;
    }
    return std::min(cutoff * cutoff, kMaxCrossKerrMixtureBranches);
}

template <typename PhaseFactorFn>
void apply_single_mode_diagonal_exact(std::vector<std::complex<double>>* state,
                                      int cutoff,
                                      int num_qumodes,
                                      int target_qumode,
                                      PhaseFactorFn&& phase_factor) {
    if (target_qumode < 0 || target_qumode >= num_qumodes) {
        throw std::out_of_range("single-mode diagonal target qumode out of range");
    }

    const size_t expected_dim = integer_power(static_cast<size_t>(cutoff), num_qumodes);
    if (state->size() != expected_dim) {
        throw std::invalid_argument("state size does not match cutoff^num_qumodes");
    }

    const size_t stride = integer_power(static_cast<size_t>(cutoff), num_qumodes - target_qumode - 1);
    const size_t prefix_count = integer_power(static_cast<size_t>(cutoff), target_qumode);

    for (size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const size_t block_base = prefix * static_cast<size_t>(cutoff) * stride;
        for (int photon = 0; photon < cutoff; ++photon) {
            const std::complex<double> factor = phase_factor(photon);
            if (std::abs(factor - std::complex<double>(1.0, 0.0)) < kDiagonalCanonicalizationTolerance) {
                continue;
            }
            for (size_t suffix = 0; suffix < stride; ++suffix) {
                (*state)[block_base + static_cast<size_t>(photon) * stride + suffix] *= factor;
            }
        }
    }
}

template <typename PhaseFactorFn>
void apply_two_mode_diagonal_exact(std::vector<std::complex<double>>* state,
                                   int cutoff,
                                   int num_qumodes,
                                   int first_target_qumode,
                                   int second_target_qumode,
                                   PhaseFactorFn&& phase_factor) {
    if (first_target_qumode < 0 || first_target_qumode >= num_qumodes ||
        second_target_qumode < 0 || second_target_qumode >= num_qumodes ||
        first_target_qumode == second_target_qumode) {
        throw std::out_of_range("two-mode diagonal target qumode out of range");
    }

    const size_t expected_dim = integer_power(static_cast<size_t>(cutoff), num_qumodes);
    if (state->size() != expected_dim) {
        throw std::invalid_argument("state size does not match cutoff^num_qumodes");
    }

    const size_t first_stride =
        integer_power(static_cast<size_t>(cutoff), num_qumodes - first_target_qumode - 1);
    const size_t second_stride =
        integer_power(static_cast<size_t>(cutoff), num_qumodes - second_target_qumode - 1);

    for (size_t flat_index = 0; flat_index < state->size(); ++flat_index) {
        const int first_photon = static_cast<int>((flat_index / first_stride) % static_cast<size_t>(cutoff));
        const int second_photon = static_cast<int>((flat_index / second_stride) % static_cast<size_t>(cutoff));
        const std::complex<double> factor = phase_factor(first_photon, second_photon);
        if (std::abs(factor - std::complex<double>(1.0, 0.0)) < kDiagonalCanonicalizationTolerance) {
            continue;
        }
        (*state)[flat_index] *= factor;
    }
}

void apply_exact_diagonal_gate_host(std::vector<std::complex<double>>* state,
                                    const GateParams& gate,
                                    int cutoff,
                                    int num_qumodes) {
    switch (gate.type) {
        case GateType::SNAP_GATE: {
            const double theta = gate.params[0].real();
            const int target_fock_state = static_cast<int>(std::llround(gate.params[1].real()));
            apply_single_mode_diagonal_exact(
                state,
                cutoff,
                num_qumodes,
                gate.target_qumodes[0],
                [theta, target_fock_state](int photon) {
                    if (photon != target_fock_state) {
                        return std::complex<double>(1.0, 0.0);
                    }
                    return std::exp(std::complex<double>(0.0, theta));
                });
            break;
        }
        case GateType::MULTI_SNAP_GATE: {
            std::vector<double> phase_map;
            phase_map.reserve(gate.params.size());
            for (const auto& phase : gate.params) {
                phase_map.push_back(phase.real());
            }
            apply_single_mode_diagonal_exact(
                state,
                cutoff,
                num_qumodes,
                gate.target_qumodes[0],
                [phase_map = std::move(phase_map)](int photon) {
                    if (photon >= static_cast<int>(phase_map.size()) ||
                        !is_nontrivial_phase(phase_map[static_cast<size_t>(photon)])) {
                        return std::complex<double>(1.0, 0.0);
                    }
                    return std::exp(std::complex<double>(0.0, phase_map[static_cast<size_t>(photon)]));
                });
            break;
        }
        case GateType::CROSS_KERR_GATE: {
            const double kappa = gate.params[0].real();
            apply_two_mode_diagonal_exact(
                state,
                cutoff,
                num_qumodes,
                gate.target_qumodes[0],
                gate.target_qumodes[1],
                [kappa](int first_photon, int second_photon) {
                    return std::exp(std::complex<double>(
                        0.0, kappa * static_cast<double>(first_photon * second_photon)));
                });
            break;
        }
        default:
            throw std::invalid_argument("unsupported host exact diagonal gate");
    }
}

bool is_pure_cv_diagonal_gate(const GateParams& gate) {
    if (!gate.target_qubits.empty()) {
        return false;
    }

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
            return gate.target_qumodes.size() == 1 && !gate.params.empty();
        case GateType::SNAP_GATE:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 2;
        case GateType::MULTI_SNAP_GATE:
            return gate.target_qumodes.size() == 1;
        case GateType::CROSS_KERR_GATE:
            return gate.target_qumodes.size() == 2 &&
                   gate.target_qumodes[0] != gate.target_qumodes[1] &&
                   !gate.params.empty();
        default:
            return false;
    }
}

bool is_pure_cv_diagonal_non_gaussian_gate(const GateParams& gate) {
    if (!gate.target_qubits.empty()) {
        return false;
    }

    switch (gate.type) {
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
            return gate.target_qumodes.size() == 1 && !gate.params.empty();
        case GateType::SNAP_GATE:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 2;
        case GateType::MULTI_SNAP_GATE:
            return gate.target_qumodes.size() == 1;
        case GateType::CROSS_KERR_GATE:
            return gate.target_qumodes.size() == 2 &&
                   gate.target_qumodes[0] != gate.target_qumodes[1] &&
                   !gate.params.empty();
        default:
            return false;
    }
}

GateParams make_single_mode_gate(GateType type, int target_qumode, std::complex<double> param) {
    return GateParams(type, {}, {target_qumode}, {param});
}

int choose_kerr_mixture_branch_cap(int cutoff) {
    if (cutoff <= 0) {
        return 0;
    }
    return std::min(cutoff, kMaxKerrMixtureBranches);
}

int required_exact_branch_count_for_gate(const GateParams& gate, int cutoff) {
    switch (gate.type) {
        case GateType::KERR_GATE:
        case GateType::SNAP_GATE:
        case GateType::MULTI_SNAP_GATE:
            return cutoff;
        case GateType::CONDITIONAL_PARITY:
            return std::min(cutoff, 2);
        case GateType::CROSS_KERR_GATE:
            return cutoff * cutoff;
        default:
            return 0;
    }
}

bool compile_diagonal_gate_mixture_approximation(
    const GateParams& gate,
    int total_qumodes,
    int cutoff,
    double target_fidelity,
    GaussianMixtureApproximation* approximation,
    std::string* error_message) {
    try {
        int max_branch_cap = 0;
        switch (gate.type) {
            case GateType::KERR_GATE:
                max_branch_cap = choose_kerr_mixture_branch_cap(cutoff);
                break;
            case GateType::SNAP_GATE:
            case GateType::MULTI_SNAP_GATE:
                max_branch_cap = choose_snap_mixture_branch_cap(gate, cutoff);
                break;
            case GateType::CONDITIONAL_PARITY:
                max_branch_cap = std::min(cutoff, 2);
                break;
            case GateType::CROSS_KERR_GATE:
                max_branch_cap = choose_cross_kerr_mixture_branch_cap(cutoff);
                break;
            default:
                if (error_message) {
                    *error_message = "未实现该对角非高斯门的Gaussian Mixture编译";
                }
                return false;
        }

        if (max_branch_cap <= 0) {
            if (error_message) {
                *error_message = "当前门的Gaussian Mixture分支上限为0";
            }
            return false;
        }

        const int required_branch_count = required_exact_branch_count_for_gate(gate, cutoff);
        if (required_branch_count <= 0) {
            if (error_message) {
                *error_message = "无法为该门推导保证目标精度的先验K";
            }
            return false;
        }

        if (required_branch_count > max_branch_cap) {
            if (error_message) {
                *error_message =
                    "达到fidelity>=" + std::to_string(target_fidelity) +
                    " 需要先验K=" + std::to_string(required_branch_count) +
                    "，超过上限 " + std::to_string(max_branch_cap) +
                    "，回退到精确Fock执行";
            }
            return false;
        }

        auto build_approximation = [&](int branch_cap) -> GaussianMixtureApproximation {
            switch (gate.type) {
                case GateType::KERR_GATE:
                    return GaussianMixtureDecomposer::approximate_kerr_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        gate.params[0].real(),
                        cutoff,
                        branch_cap);
                case GateType::SNAP_GATE: {
                    const int target_fock_state =
                        static_cast<int>(std::llround(gate.params[1].real()));
                    return GaussianMixtureDecomposer::approximate_snap_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        gate.params[0].real(),
                        target_fock_state,
                        cutoff,
                        branch_cap);
                }
                case GateType::MULTI_SNAP_GATE: {
                    std::vector<double> phase_map;
                    phase_map.reserve(gate.params.size());
                    for (const auto& phase : gate.params) {
                        phase_map.push_back(phase.real());
                    }
                    return GaussianMixtureDecomposer::approximate_multisnap_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        phase_map,
                        cutoff,
                        branch_cap);
                }
                case GateType::CONDITIONAL_PARITY:
                    return GaussianMixtureDecomposer::approximate_conditional_parity_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        gate.params[0].real(),
                        cutoff,
                        branch_cap);
                case GateType::CROSS_KERR_GATE:
                    return GaussianMixtureDecomposer::approximate_cross_kerr_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        gate.target_qumodes[1],
                        gate.params[0].real(),
                        cutoff,
                        branch_cap);
                default:
                    throw std::invalid_argument("unsupported diagonal non-Gaussian gate");
            }
        };

        *approximation = build_approximation(required_branch_count);
        return true;
    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = e.what();
        }
        return false;
    }
}

std::vector<GateParams> canonicalize_pure_cv_diagonal_window(const std::vector<GateParams>& window) {
    std::map<int, double> phase_rotations;
    std::map<int, double> kerr_gates;
    std::map<int, double> conditional_parity_gates;
    std::map<int, std::vector<double>> multisnap_gates;
    std::map<std::pair<int, int>, double> cross_kerr_gates;

    for (const GateParams& gate : window) {
        switch (gate.type) {
            case GateType::PHASE_ROTATION:
                phase_rotations[gate.target_qumodes[0]] += gate.params[0].real();
                break;
            case GateType::KERR_GATE:
                kerr_gates[gate.target_qumodes[0]] += gate.params[0].real();
                break;
            case GateType::CONDITIONAL_PARITY:
                conditional_parity_gates[gate.target_qumodes[0]] += gate.params[0].real();
                break;
            case GateType::SNAP_GATE: {
                const double theta = gate.params[0].real();
                if (!is_nontrivial_phase(theta)) {
                    break;
                }
                const int target_qumode = gate.target_qumodes[0];
                const int target_fock_state =
                    static_cast<int>(std::llround(gate.params[1].real()));
                auto& phase_map = multisnap_gates[target_qumode];
                if (target_fock_state >= 0) {
                    if (phase_map.size() <= static_cast<size_t>(target_fock_state)) {
                        phase_map.resize(static_cast<size_t>(target_fock_state) + 1, 0.0);
                    }
                    phase_map[static_cast<size_t>(target_fock_state)] += theta;
                }
                break;
            }
            case GateType::MULTI_SNAP_GATE: {
                const int target_qumode = gate.target_qumodes[0];
                auto& phase_map = multisnap_gates[target_qumode];
                if (phase_map.size() < gate.params.size()) {
                    phase_map.resize(gate.params.size(), 0.0);
                }
                for (size_t idx = 0; idx < gate.params.size(); ++idx) {
                    phase_map[idx] += gate.params[idx].real();
                }
                break;
            }
            case GateType::CROSS_KERR_GATE: {
                int first_target_qumode = gate.target_qumodes[0];
                int second_target_qumode = gate.target_qumodes[1];
                if (first_target_qumode > second_target_qumode) {
                    std::swap(first_target_qumode, second_target_qumode);
                }
                cross_kerr_gates[{first_target_qumode, second_target_qumode}] += gate.params[0].real();
                break;
            }
            default:
                throw std::invalid_argument("non-diagonal gate encountered in diagonal canonicalization window");
        }
    }

    std::vector<GateParams> canonicalized;
    canonicalized.reserve(
        phase_rotations.size() +
        kerr_gates.size() +
        conditional_parity_gates.size() +
        multisnap_gates.size() +
        cross_kerr_gates.size());

    for (const auto& [target_qumode, theta] : phase_rotations) {
        if (is_nontrivial_phase(theta)) {
            canonicalized.push_back(make_single_mode_gate(
                GateType::PHASE_ROTATION, target_qumode, std::complex<double>(theta, 0.0)));
        }
    }
    for (const auto& [target_qumode, chi] : kerr_gates) {
        if (is_nontrivial_phase(chi)) {
            canonicalized.push_back(make_single_mode_gate(
                GateType::KERR_GATE, target_qumode, std::complex<double>(chi, 0.0)));
        }
    }
    for (const auto& [target_qumode, parity] : conditional_parity_gates) {
        if (is_nontrivial_phase(parity)) {
            canonicalized.push_back(make_single_mode_gate(
                GateType::CONDITIONAL_PARITY, target_qumode, std::complex<double>(parity, 0.0)));
        }
    }
    for (auto& [target_qumode, phase_map] : multisnap_gates) {
        trim_trailing_zero_phases(&phase_map);
        if (phase_map.empty()) {
            continue;
        }
        std::vector<std::complex<double>> params;
        params.reserve(phase_map.size());
        for (double phase : phase_map) {
            params.emplace_back(phase, 0.0);
        }
        canonicalized.emplace_back(
            GateType::MULTI_SNAP_GATE,
            std::vector<int>{},
            std::vector<int>{target_qumode},
            params);
    }
    for (const auto& [targets, kappa] : cross_kerr_gates) {
        if (is_nontrivial_phase(kappa)) {
            canonicalized.push_back(make_two_mode_gate(
                GateType::CROSS_KERR_GATE,
                targets.first,
                targets.second,
                std::complex<double>(kappa, 0.0)));
        }
    }

    return canonicalized;
}

SymplecticGate embed_single_mode_gate(const SymplecticGate& local_gate,
                                      int total_qumodes,
                                      int target_qumode) {
    if (local_gate.num_qumodes != 1) {
        throw std::invalid_argument("expected a single-mode symplectic gate");
    }
    if (target_qumode < 0 || target_qumode >= total_qumodes) {
        throw std::out_of_range("target qumode out of range for symplectic embedding");
    }

    SymplecticGate embedded(total_qumodes);
    const int dim = 2 * total_qumodes;
    const int target_row = 2 * target_qumode;

    embedded.S[target_row * dim + target_row] = local_gate.S[0];
    embedded.S[target_row * dim + target_row + 1] = local_gate.S[1];
    embedded.S[(target_row + 1) * dim + target_row] = local_gate.S[2];
    embedded.S[(target_row + 1) * dim + target_row + 1] = local_gate.S[3];

    embedded.d[target_row] = local_gate.d[0];
    embedded.d[target_row + 1] = local_gate.d[1];
    return embedded;
}

SymplecticGate gate_to_symplectic(const GateParams& gate, int total_qumodes) {
    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            return embed_single_mode_gate(
                SymplecticFactory::Rotation(gate.params[0].real()),
                total_qumodes,
                gate.target_qumodes[0]);
        case GateType::DISPLACEMENT:
            return embed_single_mode_gate(
                SymplecticFactory::Displacement(gate.params[0]),
                total_qumodes,
                gate.target_qumodes[0]);
        case GateType::SQUEEZING:
            return embed_single_mode_gate(
                SymplecticFactory::Squeezing(std::abs(gate.params[0]), std::arg(gate.params[0])),
                total_qumodes,
                gate.target_qumodes[0]);
        case GateType::BEAM_SPLITTER: {
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return SymplecticFactory::BeamSplitter(
                gate.params[0].real(),
                phi,
                total_qumodes,
                gate.target_qumodes[0],
                gate.target_qumodes[1]);
        }
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
            return SymplecticFactory::TwoModeSqueezing(
                gate.params[0],
                total_qumodes,
                gate.target_qumodes[0],
                gate.target_qumodes[1]);
        case GateType::CONDITIONAL_SUM: {
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return SymplecticFactory::SUM(
                gate.params[0].real(),
                phi,
                total_qumodes,
                gate.target_qumodes[0],
                gate.target_qumodes[1]);
        }
        default:
            throw std::invalid_argument("gate cannot be represented as a symplectic update");
    }
}

void collect_terminal_state_ids_recursive(HDDNode* node,
                                          std::unordered_set<size_t>& visited_nodes,
                                          std::unordered_set<int>& state_ids) {
    if (!node) {
        return;
    }

    if (!visited_nodes.insert(node->get_unique_id()).second) {
        return;
    }

    if (node->is_terminal()) {
        if (node->tensor_id >= 0) {
            state_ids.insert(node->tensor_id);
        }
        return;
    }

    collect_terminal_state_ids_recursive(node->low, visited_nodes, state_ids);
    collect_terminal_state_ids_recursive(node->high, visited_nodes, state_ids);
}

std::vector<int> collect_terminal_state_ids(HDDNode* root) {
    std::unordered_set<size_t> visited_nodes;
    std::unordered_set<int> state_ids;
    collect_terminal_state_ids_recursive(root, visited_nodes, state_ids);

    std::vector<int> ordered_ids(state_ids.begin(), state_ids.end());
    std::sort(ordered_ids.begin(), ordered_ids.end());
    return ordered_ids;
}

size_t count_reachable_hdd_nodes_recursive(HDDNode* node, std::unordered_set<size_t>& visited_nodes) {
    if (!node) {
        return 0;
    }

    if (!visited_nodes.insert(node->get_unique_id()).second) {
        return 0;
    }

    if (node->is_terminal()) {
        return 1;
    }

    return 1 + count_reachable_hdd_nodes_recursive(node->low, visited_nodes) +
           count_reachable_hdd_nodes_recursive(node->high, visited_nodes);
}

size_t count_reachable_hdd_nodes(HDDNode* root) {
    std::unordered_set<size_t> visited_nodes;
    return count_reachable_hdd_nodes_recursive(root, visited_nodes);
}


std::complex<double> compute_qumode_overlap(
    const std::vector<cuDoubleComplex>& state_data,
    int d_trunc,
    int num_qumodes,
    const std::vector<std::vector<std::complex<double>>>& qumode_states) {
    if (state_data.empty()) {
        return std::complex<double>(0.0, 0.0);
    }

    if (qumode_states.empty()) {
        return std::complex<double>(cuCreal(state_data[0]), cuCimag(state_data[0]));
    }

    std::vector<std::vector<std::complex<double>>> mode_vectors(num_qumodes);
    for (int mode = 0; mode < num_qumodes; ++mode) {
        if (mode < static_cast<int>(qumode_states.size()) && !qumode_states[mode].empty()) {
            if (qumode_states[mode].size() != static_cast<size_t>(d_trunc)) {
                throw std::invalid_argument("Qumode状态向量长度与截断维度不匹配");
            }
            mode_vectors[mode] = qumode_states[mode];
        } else {
            mode_vectors[mode].assign(d_trunc, std::complex<double>(0.0, 0.0));
            mode_vectors[mode][0] = std::complex<double>(1.0, 0.0);
        }
    }

    std::complex<double> overlap(0.0, 0.0);
    for (size_t linear_index = 0; linear_index < state_data.size(); ++linear_index) {
        size_t index = linear_index;
        std::complex<double> basis_coeff(1.0, 0.0);
        for (int mode = num_qumodes - 1; mode >= 0; --mode) {
            const int fock_index = static_cast<int>(index % static_cast<size_t>(d_trunc));
            index /= static_cast<size_t>(d_trunc);
            basis_coeff *= mode_vectors[mode][fock_index];
        }

        const std::complex<double> amplitude(cuCreal(state_data[linear_index]),
                                             cuCimag(state_data[linear_index]));
        overlap += std::conj(basis_coeff) * amplitude;
    }

    return overlap;
}


}  // namespace circuit_internal

std::vector<GateParams> QuantumCircuit::canonicalize_gate_sequence_for_execution() const {
    std::vector<GateParams> canonicalized;
    canonicalized.reserve(gate_sequence_.size());

    size_t gate_index = 0;
    while (gate_index < gate_sequence_.size()) {
        if (!is_pure_cv_diagonal_gate(gate_sequence_[gate_index])) {
            canonicalized.push_back(gate_sequence_[gate_index]);
            ++gate_index;
            continue;
        }

        std::vector<GateParams> diagonal_window;
        while (gate_index < gate_sequence_.size() &&
               is_pure_cv_diagonal_gate(gate_sequence_[gate_index])) {
            diagonal_window.push_back(gate_sequence_[gate_index]);
            ++gate_index;
        }

        const std::vector<GateParams> window_canonicalized =
            canonicalize_pure_cv_diagonal_window(diagonal_window);
        canonicalized.insert(canonicalized.end(),
                             window_canonicalized.begin(),
                             window_canonicalized.end());
    }

    return canonicalized;
}

std::vector<QuantumCircuit::ExecutionBlock> QuantumCircuit::partition_execution_blocks(
    const std::vector<GateParams>& execution_sequence) const {
    std::vector<ExecutionBlock> blocks;
    size_t gate_index = 0;
    while (gate_index < execution_sequence.size()) {
        ExecutionBlockKind kind = ExecutionBlockKind::Other;
        if (is_pure_qubit_gate(execution_sequence[gate_index])) {
            kind = ExecutionBlockKind::QubitOnly;
        } else if (is_gaussian_track_gate(execution_sequence[gate_index])) {
            kind = ExecutionBlockKind::Gaussian;
        } else if (is_pure_cv_diagonal_non_gaussian_gate(execution_sequence[gate_index])) {
            kind = ExecutionBlockKind::DiagonalNonGaussian;
        }

        size_t block_end = gate_index + 1;
        while (block_end < execution_sequence.size()) {
            ExecutionBlockKind next_kind = ExecutionBlockKind::Other;
            if (is_pure_qubit_gate(execution_sequence[block_end])) {
                next_kind = ExecutionBlockKind::QubitOnly;
            } else if (is_gaussian_track_gate(execution_sequence[block_end])) {
                next_kind = ExecutionBlockKind::Gaussian;
            } else if (is_pure_cv_diagonal_non_gaussian_gate(execution_sequence[block_end])) {
                next_kind = ExecutionBlockKind::DiagonalNonGaussian;
            }

            if (next_kind != kind) {
                break;
            }
            ++block_end;
        }

        blocks.push_back({kind, gate_index, block_end});
        gate_index = block_end;
    }

    return blocks;
}

QuantumCircuit::CompiledExecutionBlock QuantumCircuit::compile_execution_block(
    const std::vector<GateParams>& execution_sequence,
    const std::vector<ExecutionBlock>& execution_blocks,
    size_t block_index) const {
    if (block_index >= execution_blocks.size()) {
        throw std::out_of_range("execution block index out of range");
    }

    const auto compile_start = std::chrono::high_resolution_clock::now();

    const ExecutionBlock& block = execution_blocks[block_index];
    CompiledExecutionBlock compiled_block;
    compiled_block.kind = block.kind;
    compiled_block.begin = block.begin;
    compiled_block.end = block.end;
    compiled_block.gates.assign(
        execution_sequence.begin() + static_cast<std::ptrdiff_t>(block.begin),
        execution_sequence.begin() + static_cast<std::ptrdiff_t>(block.end));

    for (size_t next_block_index = block_index + 1;
         next_block_index < execution_blocks.size();
         ++next_block_index) {
        const ExecutionBlockKind next_kind = execution_blocks[next_block_index].kind;
        if (next_kind != ExecutionBlockKind::Gaussian &&
            next_kind != ExecutionBlockKind::QubitOnly) {
            compiled_block.downstream_non_gaussianity = 1.0;
            break;
        }
    }

    if (block.kind == ExecutionBlockKind::Gaussian) {
        try {
            compiled_block.gaussian_updates.reserve(compiled_block.gates.size());
            for (const GateParams& gate : compiled_block.gates) {
                std::string gaussian_error;
                if (!validate_gaussian_track_gate(
                        gate, num_qubits_, num_qumodes_, &gaussian_error)) {
                    throw std::invalid_argument(gaussian_error);
                }
                if (is_unconditional_gaussian_gate(gate)) {
                    compiled_block.gaussian_updates.push_back(gate_to_symplectic(gate, num_qumodes_));
                }
            }
            compiled_block.gaussian_ready = true;
        } catch (const std::exception& e) {
            compiled_block.gaussian_ready = false;
            compiled_block.compile_error = e.what();
            compiled_block.gaussian_updates.clear();
        }
    } else if (block.kind == ExecutionBlockKind::DiagonalNonGaussian) {
        compiled_block.diagonal_mixture_updates.reserve(compiled_block.gates.size());
        compiled_block.diagonal_mixture_ready = true;
        const double per_gate_fidelity_target =
            std::pow(kTargetMixtureFidelity,
                     1.0 / static_cast<double>(std::max<size_t>(1, compiled_block.gates.size())));

        for (const GateParams& gate : compiled_block.gates) {
            GaussianMixtureApproximation approximation;
            std::string mixture_error;
            if (!compile_diagonal_gate_mixture_approximation(
                    gate,
                    num_qumodes_,
                    cv_truncation_,
                    per_gate_fidelity_target,
                    &approximation,
                    &mixture_error)) {
                compiled_block.diagonal_mixture_ready = false;
                compiled_block.compile_error = mixture_error;
                compiled_block.diagonal_mixture_updates.clear();
                compiled_block.estimated_diagonal_l2_error = 0.0;
                compiled_block.diagonal_fidelity_lower_bound = 0.0;
                compiled_block.mixture_branch_count = 0;
                break;
            }

            compiled_block.estimated_diagonal_l2_error += approximation.l2_diagonal_error;
            compiled_block.diagonal_fidelity_lower_bound *=
                approximation.conservative_fidelity_lower_bound;
            compiled_block.mixture_branch_count += approximation.branches.size();
            compiled_block.diagonal_mixture_updates.push_back(std::move(approximation));
        }
    }

    const auto compile_end = std::chrono::high_resolution_clock::now();
    compiled_block.compile_time_ms =
        std::chrono::duration<double, std::milli>(compile_end - compile_start).count();
    return compiled_block;
}

