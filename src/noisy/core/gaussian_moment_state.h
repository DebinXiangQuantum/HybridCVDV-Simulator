#pragma once

#include <vector>

namespace hybridcvdv::noisy {

struct GaussianMomentState {
    int num_qumodes = 0;
    std::vector<double> d;
    std::vector<double> sigma;

    static GaussianMomentState vacuum(int num_qumodes);

    int dim() const { return 2 * num_qumodes; }
    bool is_valid() const;
};

}  // namespace hybridcvdv::noisy
