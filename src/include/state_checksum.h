#pragma once

struct CVStatePool;

struct StateChecksumResult {
    double norm = 0.0;
    double checksum = 0.0;
};

// Reduces all active CV states on their owning GPUs and transfers only two
// scalars per device back to the host.
StateChecksumResult reduce_state_pool_checksum(CVStatePool& state_pool);
