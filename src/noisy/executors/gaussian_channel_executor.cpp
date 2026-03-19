#include "gaussian_channel_executor.h"

#include <stdexcept>

#include "../internal/matrix_utils.h"

namespace hybridcvdv::noisy {

bool GaussianChannelExecutor::can_remain_symbolic(const GaussianChannel& channel) const {
    return validate_gaussian_channel(channel);
}

void GaussianChannelExecutor::apply_gate(
    GaussianMomentState* state,
    const SymplecticGate& gate) const {
    if (!state || !state->is_valid()) {
        throw std::invalid_argument("apply_gate requires a valid GaussianMomentState");
    }
    if (gate.num_qumodes != state->num_qumodes) {
        throw std::invalid_argument("symplectic gate mode count does not match GaussianMomentState");
    }

    const int dim = state->dim();
    const std::vector<double> sigma_old = state->sigma;
    state->d = internal::multiply_matrix_vector(gate.S, state->d, dim);
    for (int i = 0; i < dim; ++i) {
        state->d[static_cast<std::size_t>(i)] += gate.d[static_cast<std::size_t>(i)];
    }
    state->sigma = internal::multiply_square(
        internal::multiply_square(gate.S, sigma_old, dim),
        internal::transpose_square(gate.S, dim),
        dim);
}

void GaussianChannelExecutor::apply_channel(
    GaussianMomentState* state,
    const GaussianChannel& channel) const {
    if (!state || !state->is_valid()) {
        throw std::invalid_argument("apply_channel requires a valid GaussianMomentState");
    }
    if (state->num_qumodes != channel.num_qumodes) {
        throw std::invalid_argument("Gaussian channel mode count does not match GaussianMomentState");
    }
    if (!validate_gaussian_channel(channel)) {
        throw std::invalid_argument("invalid Gaussian channel");
    }

    const int dim = state->dim();
    const std::vector<double> sigma_old = state->sigma;
    state->d = internal::multiply_matrix_vector(channel.X, state->d, dim);
    for (int i = 0; i < dim; ++i) {
        state->d[static_cast<std::size_t>(i)] += channel.c[static_cast<std::size_t>(i)];
    }
    state->sigma = internal::add_square(
        internal::multiply_square(
            internal::multiply_square(channel.X, sigma_old, dim),
            internal::transpose_square(channel.X, dim),
            dim),
        channel.Y,
        dim);
}

void GaussianChannelExecutor::apply_channel_sequence(
    GaussianMomentState* state,
    const std::vector<GaussianChannel>& channels) const {
    for (const GaussianChannel& channel : channels) {
        apply_channel(state, channel);
    }
}

}  // namespace hybridcvdv::noisy
