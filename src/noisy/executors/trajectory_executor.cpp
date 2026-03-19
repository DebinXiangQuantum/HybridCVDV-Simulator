#include "trajectory_executor.h"

#include <stdexcept>
#include <variant>

#include "../../../include/quantum_circuit.h"

namespace hybridcvdv::noisy {

NoisyRunResult TrajectoryExecutor::run(
    const NoisyProgram& program,
    const NoisyRunConfig& config,
    int num_qubits,
    int num_qumodes,
    int cv_truncation,
    int max_states) const {
    if (config.trajectory.num_trajectories == 0) {
        throw std::invalid_argument("trajectory mode requires num_trajectories > 0");
    }
    if (!program.observables().empty()) {
        throw std::logic_error(
            "trajectory observable evaluation is not implemented yet in src/noisy");
    }

    for (std::size_t shot = 0; shot < config.trajectory.num_trajectories; ++shot) {
        QuantumCircuit circuit(num_qubits, num_qumodes, cv_truncation, max_states);
        for (const NoisyProgram::Instruction& instruction : program.instructions()) {
            if (const GateParams* gate = std::get_if<GateParams>(&instruction)) {
                circuit.add_gate(*gate);
                continue;
            }

            const NoiseInstruction& noise = std::get<NoiseInstruction>(instruction);
            throw std::logic_error(
                "trajectory noise execution is not implemented yet for channel kind " +
                std::to_string(static_cast<int>(noise.kind)));
        }
        circuit.build();
        circuit.execute();
    }

    NoisyRunResult result;
    result.mode_used = NoisyExecutionMode::QuantumTrajectories;
    result.completed_trajectories = config.trajectory.num_trajectories;
    return result;
}

}  // namespace hybridcvdv::noisy
