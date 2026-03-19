#pragma once

#include "../core/types.h"

namespace hybridcvdv::noisy {

struct GaussianChannel {
    std::vector<double> X;
    std::vector<double> Y;
    std::vector<double> c;
    int num_qumodes = 0;
    std::vector<int> target_qumodes;
};

class GaussianChannelFactory {
public:
    static GaussianChannel pure_loss(double eta, int num_qumodes, int target_qumode);
    static GaussianChannel thermal_loss(double eta, double n_th, int num_qumodes, int target_qumode);
    static GaussianChannel additive_noise(double variance, int num_qumodes, int target_qumode);
    static GaussianChannel phase_insensitive_amplifier(
        double gain,
        double n_env,
        int num_qumodes,
        int target_qumode);
};

bool validate_gaussian_channel(const GaussianChannel& channel);

GaussianChannel compose_gaussian_channels(
    const GaussianChannel& first,
    const GaussianChannel& second);

GaussianMomentUpdate to_moment_update(const GaussianChannel& channel);

}  // namespace hybridcvdv::noisy
