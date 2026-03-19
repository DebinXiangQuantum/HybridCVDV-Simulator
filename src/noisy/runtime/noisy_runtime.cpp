#include "noisy_runtime.h"

#include <cmath>
#include <stdexcept>
#include <variant>

#include "../channels/gaussian_channels.h"
#include "../core/gaussian_moment_state.h"
#include "../internal/gate_conversion.h"
#include "../internal/matrix_utils.h"

namespace hybridcvdv::noisy {

namespace {

GaussianChannel noise_to_gaussian_channel(
    const NoiseInstruction& noise,
    int num_qumodes) {
    if (noise.target_qumodes.size() != 1) {
        throw std::invalid_argument("current Gaussian noise conversion requires exactly one target qumode");
    }

    const int target_qumode = noise.target_qumodes[0];
    switch (noise.kind) {
        case NoiseChannelKind::GaussianLoss:
            return GaussianChannelFactory::pure_loss(
                noise.real_params.at(0), num_qumodes, target_qumode);
        case NoiseChannelKind::ThermalLoss:
            return GaussianChannelFactory::thermal_loss(
                noise.real_params.at(0), noise.real_params.at(1), num_qumodes, target_qumode);
        case NoiseChannelKind::AdditiveGaussianNoise:
            return GaussianChannelFactory::additive_noise(
                noise.real_params.at(0), num_qumodes, target_qumode);
        case NoiseChannelKind::GaussianAmplifier:
            return GaussianChannelFactory::phase_insensitive_amplifier(
                noise.real_params.at(0), noise.real_params.at(1), num_qumodes, target_qumode);
        default:
            throw std::invalid_argument("noise instruction is not a Gaussian channel");
    }
}

bool is_gaussian_channel_kind(NoiseChannelKind kind) {
    switch (kind) {
        case NoiseChannelKind::GaussianLoss:
        case NoiseChannelKind::ThermalLoss:
        case NoiseChannelKind::AdditiveGaussianNoise:
        case NoiseChannelKind::GaussianAmplifier:
            return true;
        default:
            return false;
    }
}

bool is_gaussian_only_program(const NoisyProgram& program) {
    for (const NoisyProgram::Instruction& instruction : program.instructions()) {
        if (const GateParams* gate = std::get_if<GateParams>(&instruction)) {
            if (!internal::is_unconditional_gaussian_gate(*gate)) {
                return false;
            }
            continue;
        }

        const NoiseInstruction& noise = std::get<NoiseInstruction>(instruction);
        if (!is_gaussian_channel_kind(noise.kind)) {
            return false;
        }
    }
    return true;
}

double photon_number_mean(const GaussianMomentState& state, int target_qumode) {
    const int dim = state.dim();
    const int offset = 2 * target_qumode;
    const double x_mean = state.d[static_cast<std::size_t>(offset)];
    const double p_mean = state.d[static_cast<std::size_t>(offset + 1)];
    const double sigma_xx = state.sigma[static_cast<std::size_t>(offset) * dim + offset];
    const double sigma_pp = state.sigma[static_cast<std::size_t>(offset + 1) * dim + offset + 1];
    return 0.5 * (sigma_xx + sigma_pp + x_mean * x_mean + p_mean * p_mean - 1.0);
}

std::complex<double> parity_mean(const GaussianMomentState& state, int target_qumode) {
    const int dim = state.dim();
    const int offset = 2 * target_qumode;
    const double det = internal::determinant_2x2(state.sigma, offset, dim);
    if (det <= 0.0) {
        throw std::runtime_error("reduced covariance is not positive definite");
    }

    const std::vector<double> sigma_inv = internal::inverse_2x2(state.sigma, offset, dim);
    const double x_mean = state.d[static_cast<std::size_t>(offset)];
    const double p_mean = state.d[static_cast<std::size_t>(offset + 1)];
    const double quadratic =
        x_mean * (sigma_inv[0] * x_mean + sigma_inv[1] * p_mean) +
        p_mean * (sigma_inv[2] * x_mean + sigma_inv[3] * p_mean);
    const double parity = 0.5 / std::sqrt(det) * std::exp(-0.5 * quadratic);
    return {parity, 0.0};
}

ObservableEstimate evaluate_gaussian_observable(
    const GaussianMomentState& state,
    const ObservableSpec& observable) {
    ObservableEstimate estimate;
    estimate.label = observable.label;

    if (observable.target_qumodes.size() != 1) {
        throw std::invalid_argument("current Gaussian observable evaluation requires one target qumode");
    }

    const int target_qumode = observable.target_qumodes[0];
    if (target_qumode < 0 || target_qumode >= state.num_qumodes) {
        throw std::out_of_range("observable target qumode out of range");
    }

    switch (observable.kind) {
        case ObservableKind::PhotonNumber:
            estimate.mean = {photon_number_mean(state, target_qumode), 0.0};
            return estimate;
        case ObservableKind::Parity:
            estimate.mean = parity_mean(state, target_qumode);
            return estimate;
        default:
            throw std::logic_error(
                "current Gaussian-only runtime supports PhotonNumber and Parity observables only");
    }
}

NoisyRunResult execute_gaussian_only_program(
    const NoisyProgram& program,
    int num_qumodes) {
    GaussianMomentState state = GaussianMomentState::vacuum(num_qumodes);
    GaussianChannelExecutor executor;

    for (const NoisyProgram::Instruction& instruction : program.instructions()) {
        if (const GateParams* gate = std::get_if<GateParams>(&instruction)) {
            executor.apply_gate(&state, internal::gate_to_symplectic(*gate, num_qumodes));
            continue;
        }

        const NoiseInstruction& noise = std::get<NoiseInstruction>(instruction);
        executor.apply_channel(&state, noise_to_gaussian_channel(noise, num_qumodes));
    }

    NoisyRunResult result;
    result.mode_used = NoisyExecutionMode::GaussianChannelsOnly;
    result.completed_trajectories = 1;
    result.observables.reserve(program.observables().size());
    for (const ObservableSpec& observable : program.observables()) {
        result.observables.push_back(evaluate_gaussian_observable(state, observable));
    }
    return result;
}

}  // namespace

NoisyRunResult NoisyRuntime::execute(
    const NoisyProgram& program,
    const NoisyRunConfig& config,
    int num_qubits,
    int num_qumodes,
    int cv_truncation,
    int max_states) const {
    const NoisyExecutionMode mode = select_mode(program, config);
    switch (mode) {
        case NoisyExecutionMode::GaussianChannelsOnly:
            return execute_gaussian_only_program(program, num_qumodes);
        case NoisyExecutionMode::QuantumTrajectories: {
            TrajectoryExecutor executor;
            return executor.run(
                program,
                config,
                num_qubits,
                num_qumodes,
                cv_truncation,
                max_states);
        }
        case NoisyExecutionMode::ReferenceDensityMatrix:
            throw std::logic_error("reference density-matrix backend is not implemented yet in src/noisy");
    }

    throw std::logic_error("unknown noisy execution mode");
}

NoisyExecutionMode NoisyRuntime::select_mode(
    const NoisyProgram& program,
    const NoisyRunConfig& config) const {
    if (config.preferred_mode == NoisyExecutionMode::ReferenceDensityMatrix) {
        return NoisyExecutionMode::ReferenceDensityMatrix;
    }
    if (config.preferred_mode == NoisyExecutionMode::GaussianChannelsOnly &&
        !is_gaussian_only_program(program)) {
        throw std::invalid_argument(
            "preferred GaussianChannelsOnly mode is incompatible with the program");
    }

    if (is_gaussian_only_program(program)) {
        return NoisyExecutionMode::GaussianChannelsOnly;
    }
    return NoisyExecutionMode::QuantumTrajectories;
}

}  // namespace hybridcvdv::noisy
