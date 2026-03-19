#pragma once

#include <variant>
#include <vector>

#include "../../../include/quantum_circuit.h"
#include "types.h"

namespace hybridcvdv::noisy {

class NoisyProgram {
public:
    using Instruction = std::variant<GateParams, NoiseInstruction>;

    void add_gate(const GateParams& gate);
    void add_noise(const NoiseInstruction& noise);
    void add_observable(const ObservableSpec& observable);

    const std::vector<Instruction>& instructions() const { return instructions_; }
    const std::vector<ObservableSpec>& observables() const { return observables_; }

    bool empty() const { return instructions_.empty(); }

private:
    std::vector<Instruction> instructions_;
    std::vector<ObservableSpec> observables_;
};

}  // namespace hybridcvdv::noisy
