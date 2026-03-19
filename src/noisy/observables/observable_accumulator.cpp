#include "observable_accumulator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hybridcvdv::noisy {

void ObservableAccumulator::RunningStat::record(double value) {
    ++count;
    const double delta = value - mean;
    mean += delta / static_cast<double>(count);
    const double delta2 = value - mean;
    m2 += delta * delta2;
}

double ObservableAccumulator::RunningStat::variance() const {
    if (count < 2) {
        return 0.0;
    }
    return m2 / static_cast<double>(count - 1);
}

double ObservableAccumulator::RunningStat::standard_error() const {
    if (count == 0) {
        return 0.0;
    }
    return std::sqrt(variance() / static_cast<double>(count));
}

void ObservableAccumulator::begin_run(const std::vector<ObservableSpec>& observables) {
    observables_ = observables;
    real_stats_.assign(observables.size(), RunningStat{});
    imag_stats_.assign(observables.size(), RunningStat{});
}

void ObservableAccumulator::record_trajectory_value(
    std::size_t observable_index,
    double value) {
    if (observable_index >= real_stats_.size()) {
        throw std::out_of_range("observable_index out of range");
    }
    real_stats_[observable_index].record(value);
}

void ObservableAccumulator::record_trajectory_complex_value(
    std::size_t observable_index,
    std::complex<double> value) {
    if (observable_index >= real_stats_.size()) {
        throw std::out_of_range("observable_index out of range");
    }
    real_stats_[observable_index].record(value.real());
    imag_stats_[observable_index].record(value.imag());
}

NoisyRunResult ObservableAccumulator::finish() const {
    NoisyRunResult result;
    result.mode_used = NoisyExecutionMode::QuantumTrajectories;
    result.completed_trajectories = 0;
    result.observables.reserve(observables_.size());

    for (std::size_t i = 0; i < observables_.size(); ++i) {
        ObservableEstimate estimate;
        estimate.label = observables_[i].label;
        estimate.mean = {
            real_stats_[i].mean,
            imag_stats_[i].mean};
        estimate.variance = real_stats_[i].variance() + imag_stats_[i].variance();
        estimate.standard_error =
            std::sqrt(
                real_stats_[i].standard_error() * real_stats_[i].standard_error() +
                imag_stats_[i].standard_error() * imag_stats_[i].standard_error());
        result.completed_trajectories =
            std::max(result.completed_trajectories, real_stats_[i].count);
        result.observables.push_back(std::move(estimate));
    }
    return result;
}

}  // namespace hybridcvdv::noisy
