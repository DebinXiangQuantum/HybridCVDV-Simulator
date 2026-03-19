#pragma once

#include "../core/types.h"

namespace hybridcvdv::noisy {

class ObservableAccumulator {
public:
    void begin_run(const std::vector<ObservableSpec>& observables);
    void record_trajectory_value(std::size_t observable_index, double value);
    void record_trajectory_complex_value(std::size_t observable_index, std::complex<double> value);
    NoisyRunResult finish() const;

private:
    struct RunningStat {
        std::size_t count = 0;
        double mean = 0.0;
        double m2 = 0.0;

        void record(double value);
        double variance() const;
        double standard_error() const;
    };

    std::vector<ObservableSpec> observables_;
    std::vector<RunningStat> real_stats_;
    std::vector<RunningStat> imag_stats_;
};

}  // namespace hybridcvdv::noisy
