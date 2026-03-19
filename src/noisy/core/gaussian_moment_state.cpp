#include "gaussian_moment_state.h"

namespace hybridcvdv::noisy {

GaussianMomentState GaussianMomentState::vacuum(int num_qumodes) {
    GaussianMomentState state;
    state.num_qumodes = num_qumodes;
    const int dim = 2 * num_qumodes;
    state.d.assign(static_cast<std::size_t>(dim), 0.0);
    state.sigma.assign(static_cast<std::size_t>(dim) * dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        state.sigma[static_cast<std::size_t>(i) * dim + i] = 0.5;
    }
    return state;
}

bool GaussianMomentState::is_valid() const {
    const int dimension = dim();
    return num_qumodes >= 0 &&
           d.size() == static_cast<std::size_t>(dimension) &&
           sigma.size() == static_cast<std::size_t>(dimension) * dimension;
}

}  // namespace hybridcvdv::noisy
