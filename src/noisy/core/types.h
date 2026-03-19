#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hybridcvdv::noisy {

enum class NoiseChannelKind {
    GaussianLoss,
    ThermalLoss,
    AdditiveGaussianNoise,
    GaussianAmplifier,
    DVAmplitudeDamping,
    DVDephasing,
    CustomJumpOperator
};

enum class NoisyExecutionMode {
    GaussianChannelsOnly,
    QuantumTrajectories,
    ReferenceDensityMatrix
};

enum class ObservableKind {
    Expectation,
    Variance,
    Probability,
    PhotonNumber,
    Parity,
    Fidelity,
    SampleHistogram
};

struct GaussianMomentUpdate {
    std::vector<double> X;
    std::vector<double> Y;
    std::vector<double> c;
    int num_qumodes = 0;
    std::vector<int> target_qumodes;
};

struct NoiseInstruction {
    NoiseChannelKind kind = NoiseChannelKind::GaussianLoss;
    std::vector<int> target_qubits;
    std::vector<int> target_qumodes;
    std::vector<double> real_params;
    std::vector<std::complex<double>> complex_params;
    double rate = 0.0;
    std::string label;
};

struct TrajectoryConfig {
    std::size_t num_trajectories = 0;
    std::uint64_t seed = 0;
    std::size_t max_jump_count_per_shot = 0;
    bool store_samples = false;
};

struct ObservableSpec {
    ObservableKind kind = ObservableKind::Expectation;
    std::vector<int> target_qubits;
    std::vector<int> target_qumodes;
    std::string label;
};

struct ObservableEstimate {
    std::string label;
    std::complex<double> mean{0.0, 0.0};
    double variance = 0.0;
    double standard_error = 0.0;
};

struct NoisyRunConfig {
    NoisyExecutionMode preferred_mode = NoisyExecutionMode::QuantumTrajectories;
    TrajectoryConfig trajectory;
};

struct NoisyRunResult {
    NoisyExecutionMode mode_used = NoisyExecutionMode::QuantumTrajectories;
    std::vector<ObservableEstimate> observables;
    std::size_t completed_trajectories = 0;
};

}  // namespace hybridcvdv::noisy
