#pragma once

#include "../../../include/quantum_circuit.h"
#include "../../../include/symplectic_math.h"

namespace hybridcvdv::noisy::internal {

bool is_unconditional_gaussian_gate(const GateParams& gate);
SymplecticGate gate_to_symplectic(const GateParams& gate, int total_qumodes);

}  // namespace hybridcvdv::noisy::internal
