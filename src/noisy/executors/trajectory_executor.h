#pragma once

#include "../core/noisy_program.h"
#include "../core/types.h"

namespace hybridcvdv::noisy {

class TrajectoryExecutor {
public:
    NoisyRunResult run(
        const NoisyProgram& program,
        const NoisyRunConfig& config,
        int num_qubits,
        int num_qumodes,
        int cv_truncation,
        int max_states) const;
};

}  // namespace hybridcvdv::noisy
