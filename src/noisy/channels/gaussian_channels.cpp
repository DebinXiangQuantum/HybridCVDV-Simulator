#include "gaussian_channels.h"

#include <cmath>
#include <stdexcept>

#include "../internal/matrix_utils.h"

namespace hybridcvdv::noisy {

namespace {

constexpr double kVacuumVariance = 0.5;
constexpr double kValidationTolerance = 1e-10;

GaussianChannel make_single_mode_channel(
    const std::vector<double>& local_x,
    const std::vector<double>& local_y,
    const std::vector<double>& local_c,
    int num_qumodes,
    int target_qumode) {
    GaussianChannel channel;
    channel.num_qumodes = num_qumodes;
    channel.target_qumodes = {target_qumode};
    channel.X = internal::embed_single_mode_matrix(local_x, num_qumodes, target_qumode);
    channel.Y = internal::embed_single_mode_matrix(local_y, num_qumodes, target_qumode);
    channel.c = internal::embed_single_mode_vector(local_c, num_qumodes, target_qumode);
    return channel;
}

bool has_valid_dimensions(const GaussianChannel& channel) {
    const int dim = 2 * channel.num_qumodes;
    return channel.num_qumodes > 0 &&
           channel.X.size() == static_cast<std::size_t>(dim) * dim &&
           channel.Y.size() == static_cast<std::size_t>(dim) * dim &&
           channel.c.size() == static_cast<std::size_t>(dim);
}

bool check_complete_positivity(const GaussianChannel& channel) {
    const int dim = 2 * channel.num_qumodes;
    const std::vector<double> omega = internal::symplectic_form(channel.num_qumodes);
    const std::vector<double> x_omega = internal::multiply_square(channel.X, omega, dim);
    const std::vector<double> x_omega_xt =
        internal::multiply_square(x_omega, internal::transpose_square(channel.X, dim), dim);
    const std::vector<double> skew_part =
        internal::scale_square(internal::subtract_square(omega, x_omega_xt, dim), dim, 0.5);

    const int real_dim = 2 * dim;
    std::vector<double> realified(static_cast<std::size_t>(real_dim) * real_dim, 0.0);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            const double y = channel.Y[static_cast<std::size_t>(row) * dim + col];
            const double k = skew_part[static_cast<std::size_t>(row) * dim + col];
            realified[static_cast<std::size_t>(row) * real_dim + col] = y;
            realified[static_cast<std::size_t>(row) * real_dim + dim + col] = -k;
            realified[static_cast<std::size_t>(dim + row) * real_dim + col] = k;
            realified[static_cast<std::size_t>(dim + row) * real_dim + dim + col] = y;
        }
    }

    const double min_eigenvalue =
        internal::jacobi_min_eigenvalue(std::move(realified), real_dim, 1e-12);
    return min_eigenvalue >= -kValidationTolerance;
}

}  // namespace

GaussianChannel GaussianChannelFactory::pure_loss(
    double eta,
    int num_qumodes,
    int target_qumode) {
    if (eta < 0.0 || eta > 1.0) {
        throw std::invalid_argument("pure_loss requires eta in [0, 1]");
    }

    const double attenuation = std::sqrt(eta);
    const double added_noise = (1.0 - eta) * kVacuumVariance;
    return make_single_mode_channel(
        {attenuation, 0.0, 0.0, attenuation},
        {added_noise, 0.0, 0.0, added_noise},
        {0.0, 0.0},
        num_qumodes,
        target_qumode);
}

GaussianChannel GaussianChannelFactory::thermal_loss(
    double eta,
    double n_th,
    int num_qumodes,
    int target_qumode) {
    if (eta < 0.0 || eta > 1.0) {
        throw std::invalid_argument("thermal_loss requires eta in [0, 1]");
    }
    if (n_th < 0.0) {
        throw std::invalid_argument("thermal_loss requires n_th >= 0");
    }

    const double attenuation = std::sqrt(eta);
    const double env_variance = (2.0 * n_th + 1.0) * kVacuumVariance;
    const double added_noise = (1.0 - eta) * env_variance;
    return make_single_mode_channel(
        {attenuation, 0.0, 0.0, attenuation},
        {added_noise, 0.0, 0.0, added_noise},
        {0.0, 0.0},
        num_qumodes,
        target_qumode);
}

GaussianChannel GaussianChannelFactory::additive_noise(
    double variance,
    int num_qumodes,
    int target_qumode) {
    if (variance < 0.0) {
        throw std::invalid_argument("additive_noise requires variance >= 0");
    }

    return make_single_mode_channel(
        {1.0, 0.0, 0.0, 1.0},
        {variance, 0.0, 0.0, variance},
        {0.0, 0.0},
        num_qumodes,
        target_qumode);
}

GaussianChannel GaussianChannelFactory::phase_insensitive_amplifier(
    double gain,
    double n_env,
    int num_qumodes,
    int target_qumode) {
    if (gain < 1.0) {
        throw std::invalid_argument("phase_insensitive_amplifier requires gain >= 1");
    }
    if (n_env < 0.0) {
        throw std::invalid_argument("phase_insensitive_amplifier requires n_env >= 0");
    }

    const double amplification = std::sqrt(gain);
    const double env_variance = (2.0 * n_env + 1.0) * kVacuumVariance;
    const double added_noise = (gain - 1.0) * env_variance;
    return make_single_mode_channel(
        {amplification, 0.0, 0.0, amplification},
        {added_noise, 0.0, 0.0, added_noise},
        {0.0, 0.0},
        num_qumodes,
        target_qumode);
}

bool validate_gaussian_channel(const GaussianChannel& channel) {
    if (!has_valid_dimensions(channel)) {
        return false;
    }

    if (!internal::all_finite(channel.X) ||
        !internal::all_finite(channel.Y) ||
        !internal::all_finite(channel.c)) {
        return false;
    }

    const int dim = 2 * channel.num_qumodes;
    if (!internal::is_symmetric(channel.Y, dim, 1e-10)) {
        return false;
    }

    for (int target_qumode : channel.target_qumodes) {
        if (target_qumode < 0 || target_qumode >= channel.num_qumodes) {
            return false;
        }
    }

    return check_complete_positivity(channel);
}

GaussianChannel compose_gaussian_channels(
    const GaussianChannel& first,
    const GaussianChannel& second) {
    if (first.num_qumodes != second.num_qumodes) {
        throw std::invalid_argument("cannot compose Gaussian channels with different mode counts");
    }

    const int dim = 2 * first.num_qumodes;
    GaussianChannel composed;
    composed.num_qumodes = first.num_qumodes;
    composed.target_qumodes = internal::merge_unique_targets(
        first.target_qumodes,
        second.target_qumodes);
    composed.X = internal::multiply_square(second.X, first.X, dim);
    composed.Y = internal::add_square(
        internal::multiply_square(
            internal::multiply_square(second.X, first.Y, dim),
            internal::transpose_square(second.X, dim),
            dim),
        second.Y,
        dim);
    composed.c = internal::multiply_matrix_vector(second.X, first.c, dim);
    for (int i = 0; i < dim; ++i) {
        composed.c[static_cast<std::size_t>(i)] += second.c[static_cast<std::size_t>(i)];
    }

    if (!validate_gaussian_channel(composed)) {
        throw std::runtime_error("composed Gaussian channel is invalid");
    }
    return composed;
}

GaussianMomentUpdate to_moment_update(const GaussianChannel& channel) {
    return GaussianMomentUpdate{
        channel.X,
        channel.Y,
        channel.c,
        channel.num_qumodes,
        channel.target_qumodes};
}

}  // namespace hybridcvdv::noisy
