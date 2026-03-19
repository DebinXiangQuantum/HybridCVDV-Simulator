#pragma once

#include "../core/noisy_program.h"
#include "../executors/gaussian_channel_executor.h"
#include "../executors/trajectory_executor.h"

namespace hybridcvdv::noisy {

class NoisyRuntime {
public:
    NoisyRunResult execute(
        const NoisyProgram& program,
        const NoisyRunConfig& config,
        int num_qubits,
        int num_qumodes,
        int cv_truncation,
        int max_states) const;

    NoisyExecutionMode select_mode(
        const NoisyProgram& program,
        const NoisyRunConfig& config) const;
};

}  // namespace hybridcvdv::noisy
