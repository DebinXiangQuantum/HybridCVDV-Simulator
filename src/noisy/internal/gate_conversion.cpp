#include "gate_conversion.h"

#include <cmath>
#include <stdexcept>

namespace hybridcvdv::noisy::internal {

namespace {

SymplecticGate embed_single_mode_gate(
    const SymplecticGate& local_gate,
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

    embedded.S[static_cast<std::size_t>(target_row) * dim + target_row] = local_gate.S[0];
    embedded.S[static_cast<std::size_t>(target_row) * dim + target_row + 1] = local_gate.S[1];
    embedded.S[static_cast<std::size_t>(target_row + 1) * dim + target_row] = local_gate.S[2];
    embedded.S[static_cast<std::size_t>(target_row + 1) * dim + target_row + 1] = local_gate.S[3];

    embedded.d[static_cast<std::size_t>(target_row)] = local_gate.d[0];
    embedded.d[static_cast<std::size_t>(target_row + 1)] = local_gate.d[1];
    return embedded;
}

}  // namespace

bool is_unconditional_gaussian_gate(const GateParams& gate) {
    switch (gate.type) {
        case GateType::PHASE_ROTATION:
        case GateType::DISPLACEMENT:
        case GateType::SQUEEZING:
        case GateType::BEAM_SPLITTER:
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
        case GateType::CONDITIONAL_SUM:
            return true;
        default:
            return false;
    }
}

SymplecticGate gate_to_symplectic(const GateParams& gate, int total_qumodes) {
    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            return embed_single_mode_gate(
                SymplecticFactory::Rotation(gate.params.at(0).real()),
                total_qumodes,
                gate.target_qumodes.at(0));
        case GateType::DISPLACEMENT:
            return embed_single_mode_gate(
                SymplecticFactory::Displacement(gate.params.at(0)),
                total_qumodes,
                gate.target_qumodes.at(0));
        case GateType::SQUEEZING:
            return embed_single_mode_gate(
                SymplecticFactory::Squeezing(std::abs(gate.params.at(0)), std::arg(gate.params.at(0))),
                total_qumodes,
                gate.target_qumodes.at(0));
        case GateType::BEAM_SPLITTER: {
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return SymplecticFactory::BeamSplitter(
                gate.params.at(0).real(),
                phi,
                total_qumodes,
                gate.target_qumodes.at(0),
                gate.target_qumodes.at(1));
        }
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
            return SymplecticFactory::TwoModeSqueezing(
                gate.params.at(0),
                total_qumodes,
                gate.target_qumodes.at(0),
                gate.target_qumodes.at(1));
        case GateType::CONDITIONAL_SUM: {
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return SymplecticFactory::SUM(
                gate.params.at(0).real(),
                phi,
                total_qumodes,
                gate.target_qumodes.at(0),
                gate.target_qumodes.at(1));
        }
        default:
            throw std::invalid_argument("gate cannot be represented as a symplectic update");
    }
}

}  // namespace hybridcvdv::noisy::internal
