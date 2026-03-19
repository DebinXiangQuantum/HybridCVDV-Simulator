#include "noisy_program.h"

namespace hybridcvdv::noisy {

void NoisyProgram::add_gate(const GateParams& gate) {
    instructions_.emplace_back(gate);
}

void NoisyProgram::add_noise(const NoiseInstruction& noise) {
    instructions_.emplace_back(noise);
}

void NoisyProgram::add_observable(const ObservableSpec& observable) {
    observables_.push_back(observable);
}

}  // namespace hybridcvdv::noisy
