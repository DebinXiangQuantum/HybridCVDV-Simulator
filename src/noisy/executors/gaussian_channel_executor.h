#pragma once

#include "../channels/gaussian_channels.h"
#include "../core/gaussian_moment_state.h"
#include "../../../include/symplectic_math.h"

namespace hybridcvdv::noisy {

class GaussianChannelExecutor {
public:
    bool can_remain_symbolic(const GaussianChannel& channel) const;
    void apply_gate(GaussianMomentState* state, const SymplecticGate& gate) const;
    void apply_channel(GaussianMomentState* state, const GaussianChannel& channel) const;
    void apply_channel_sequence(
        GaussianMomentState* state,
        const std::vector<GaussianChannel>& channels) const;
};

}  // namespace hybridcvdv::noisy
