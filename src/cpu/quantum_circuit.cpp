#include "quantum_circuit.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include <chrono>
#include <future>
#include <map>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_set>
#if defined(HYBRIDCVDV_HAS_NVTX) && HYBRIDCVDV_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

// CUDA错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA错误: " + std::string(cudaGetErrorString(err))); \
        } \
    } while (0)

// 包含GPU内核头文件
// Level 0
void apply_phase_rotation_on_mode(CVStatePool* pool, const int* targets, int batch_size, double theta,
                                  int target_qumode, int num_qumodes,
                                  cudaStream_t stream = nullptr, bool synchronize = true);
void apply_kerr_gate_on_mode(CVStatePool* pool, const int* targets, int batch_size, double chi,
                             int target_qumode, int num_qumodes,
                             cudaStream_t stream = nullptr, bool synchronize = true);
void apply_conditional_parity_on_mode(CVStatePool* pool, const int* targets, int batch_size, double parity,
                                      int target_qumode, int num_qumodes,
                                      cudaStream_t stream = nullptr, bool synchronize = true);
void apply_snap_on_mode(CVStatePool* pool, const int* targets, int batch_size, double theta,
                        int target_fock_state, int target_qumode, int num_qumodes);
void apply_multisnap_on_mode(CVStatePool* pool, const int* targets, int batch_size,
                             const std::vector<double>& phase_map,
                             int target_qumode, int num_qumodes);
struct FusedDiagonalOp {
    int right_stride;
    double theta;
    double chi;
    double parity;
};
void apply_fused_diagonal_gates(CVStatePool* state_pool, const int* target_indices,
                                int batch_size, const std::vector<FusedDiagonalOp>& ops_host,
                                int num_qumodes,
                                cudaStream_t stream = nullptr, bool synchronize = true);

void zero_state_device(CVStatePool* pool, int state_id,
                       cudaStream_t stream = nullptr, bool synchronize = true);
void initialize_vacuum_state_device(CVStatePool* pool, int state_id, int64_t state_dim,
                                    cudaStream_t stream = nullptr, bool synchronize = true);
void copy_state_device(CVStatePool* pool, int src_state_id, int dst_state_id,
                       cudaStream_t stream = nullptr, bool synchronize = true);
void copy_scale_state_device(CVStatePool* pool, int src_state_id, int dst_state_id, cuDoubleComplex weight,
                             cudaStream_t stream = nullptr, bool synchronize = true);
void scale_state_device(CVStatePool* pool, int state_id, cuDoubleComplex weight,
                        cudaStream_t stream = nullptr, bool synchronize = true);
void axpy_state_device(CVStatePool* pool, int src_state_id, int dst_state_id, cuDoubleComplex weight,
                       cudaStream_t stream = nullptr, bool synchronize = true);
void classify_vacuum_ray_device(CVStatePool* pool, int state_id, double tolerance,
                                int* is_zero, int* is_scaled_vacuum, cuDoubleComplex* scale);

// Level 1
void apply_creation_operator_on_mode(CVStatePool* pool, const int* targets, int batch_size,
                                     int target_qumode, int num_qumodes,
                                     cudaStream_t stream = nullptr, bool synchronize = true);
void apply_annihilation_operator_on_mode(CVStatePool* pool, const int* targets, int batch_size,
                                         int target_qumode, int num_qumodes,
                                         cudaStream_t stream = nullptr, bool synchronize = true);

// Level 2
void apply_single_mode_gate(CVStatePool* pool, FockELLOperator* ell_op,
                           const int* targets, int batch_size,
                           cudaStream_t stream = nullptr, bool synchronize = true);
void apply_displacement_gate(CVStatePool* pool, const int* targets, int batch_size,
                            cuDoubleComplex alpha,
                            cudaStream_t stream = nullptr, bool synchronize = true);

// Level 3
// Level 4
void apply_hybrid_control_gate(HDDNode* root_node, CVStatePool* state_pool,
                              HDDNodeManager& node_manager,
                              const std::string& gate_type,
                              cuDoubleComplex param);
void apply_controlled_displacement_on_mode(CVStatePool* state_pool,
                                           const std::vector<int>& controlled_states,
                                           cuDoubleComplex alpha,
                                           int target_qumode,
                                           int num_qumodes);
void apply_rabi_interaction(CVStatePool* state_pool,
                            const std::vector<int>& qubit0_states,
                            const std::vector<int>& qubit1_states,
                            double theta);
void apply_rabi_interaction_on_mode(CVStatePool* state_pool,
                                    const std::vector<int>& qubit0_states,
                                    const std::vector<int>& qubit1_states,
                                    double theta,
                                    int target_qumode,
                                    int num_qumodes);
void apply_jaynes_cummings(CVStatePool* state_pool,
                           const std::vector<int>& qubit0_states,
                           const std::vector<int>& qubit1_states,
                           double theta,
                           double phi);
void apply_jaynes_cummings_on_mode(CVStatePool* state_pool,
                                   const std::vector<int>& qubit0_states,
                                   const std::vector<int>& qubit1_states,
                                   double theta,
                                   double phi,
                                   int target_qumode,
                                   int num_qumodes);
void apply_anti_jaynes_cummings(CVStatePool* state_pool,
                                const std::vector<int>& qubit0_states,
                                const std::vector<int>& qubit1_states,
                                double theta,
                                double phi);
void apply_anti_jaynes_cummings_on_mode(CVStatePool* state_pool,
                                        const std::vector<int>& qubit0_states,
                                        const std::vector<int>& qubit1_states,
                                        double theta,
                                        double phi,
                                        int target_qumode,
                                        int num_qumodes);
void apply_sqr(CVStatePool* state_pool,
               const std::vector<int>& qubit0_states,
               const std::vector<int>& qubit1_states,
               const std::vector<double>& thetas,
               const std::vector<double>& phis);

// 状态加法函数
void add_states(CVStatePool* state_pool,
                const int* src1_indices,
                const cuDoubleComplex* weights1,
                const int* src2_indices,
                const cuDoubleComplex* weights2,
                const int* dst_indices,
                int batch_size);
void combine_states_device(CVStatePool* state_pool,
                           int src1_state_id,
                           cuDoubleComplex weight1,
                           int src2_state_id,
                           cuDoubleComplex weight2,
                           int dst_state_id,
                           cudaStream_t stream = nullptr,
                           bool synchronize = true);

namespace {

bool fallback_debug_logging_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("HYBRIDCVDV_FALLBACK_DEBUG");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

bool async_cv_pipeline_disabled() {
    static const bool disabled = []() {
        const char* env = std::getenv("HYBRIDCVDV_DISABLE_ASYNC_CV_PIPELINE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return disabled;
}

#define FALLBACK_DEBUG_LOG if (!fallback_debug_logging_enabled()) {} else std::cout

const char* block_progress_log_path() {
    static const char* path = std::getenv("HYBRIDCVDV_BLOCK_PROGRESS_LOG");
    return (path != nullptr && path[0] != '\0') ? path : nullptr;
}

size_t block_progress_log_interval() {
    static const size_t interval = []() -> size_t {
        const char* env = std::getenv("HYBRIDCVDV_BLOCK_PROGRESS_EVERY");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end == env || value <= 0) {
            return 0;
        }
        return static_cast<size_t>(value);
    }();
    return interval;
}

void log_block_progress_if_requested(size_t block_index, size_t total_blocks) {
    const char* path = block_progress_log_path();
    const size_t interval = block_progress_log_interval();
    if (path == nullptr || interval == 0 || total_blocks == 0) {
        return;
    }
    if ((block_index % interval) != 0 && (block_index + 1) != total_blocks) {
        return;
    }

    std::ofstream progress_log(path, std::ios::app);
    if (!progress_log) {
        return;
    }
    progress_log << "BLOCK " << (block_index + 1) << "/" << total_blocks << '\n';
}

std::vector<int> collect_terminal_state_ids(HDDNode* root);

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* name) {
#if defined(HYBRIDCVDV_HAS_NVTX) && HYBRIDCVDV_HAS_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~ScopedNvtxRange() {
#if defined(HYBRIDCVDV_HAS_NVTX) && HYBRIDCVDV_HAS_NVTX
        nvtxRangePop();
#endif
    }
};

int compute_qumode_right_stride(int trunc_dim, int target_qumode, int num_qumodes) {
    int right_stride = 1;
    for (int mode = target_qumode + 1; mode < num_qumodes; ++mode) {
        right_stride *= trunc_dim;
    }
    return right_stride;
}

size_t align_scratch_offset(size_t offset, size_t alignment) {
    const size_t mask = alignment - 1;
    return (offset + mask) & ~mask;
}

struct WeightedNodePairKey {
    HDDNode* low_node = nullptr;
    HDDNode* high_node = nullptr;
    std::complex<double> low_weight{1.0, 0.0};
    std::complex<double> high_weight{1.0, 0.0};

    bool operator==(const WeightedNodePairKey& other) const {
        return low_node == other.low_node &&
               high_node == other.high_node &&
               low_weight == other.low_weight &&
               high_weight == other.high_weight;
    }
};

struct ExactFockCheckpointHeader {
    char magic[8];
    uint32_t version;
    uint32_t num_qubits;
    uint32_t num_qumodes;
    uint32_t cv_truncation;
    uint32_t max_states;
    int32_t shared_zero_state_id;
    uint64_t next_block_index;
    uint64_t total_blocks;
    uint64_t state_count;
    uint64_t node_count;
    uint64_t root_index;
};

struct ExactFockCheckpointStateRecord {
    int32_t state_id;
    int32_t state_dim;
};

struct ExactFockCheckpointNodeRecord {
    uint8_t is_terminal;
    int16_t qubit_level;
    int32_t tensor_id;
    uint64_t low_index;
    uint64_t high_index;
    double w_low_real;
    double w_low_imag;
    double w_high_real;
    double w_high_imag;
};

struct WeightedNodePairKeyHash {
    size_t operator()(const WeightedNodePairKey& key) const {
        size_t hash = std::hash<HDDNode*>()(key.low_node);
        hash ^= std::hash<HDDNode*>()(key.high_node) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<double>()(key.low_weight.real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<double>()(key.low_weight.imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<double>()(key.high_weight.real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<double>()(key.high_weight.imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct PairwiseHybridStorageEstimate {
    size_t pair_count = 0;
    size_t duplicate_state_count = 0;
    size_t extra_elements = 0;
};

bool is_pairwise_hybrid_gate_type(GateType type) {
    switch (type) {
        case GateType::RABI_INTERACTION:
        case GateType::JAYNES_CUMMINGS:
        case GateType::ANTI_JAYNES_CUMMINGS:
        case GateType::SELECTIVE_QUBIT_ROTATION:
            return true;
        default:
            return false;
    }
}

PairwiseHybridStorageEstimate estimate_pairwise_hybrid_storage(HDDNode* root,
                                                               int control_qubit,
                                                               size_t state_dim) {
    PairwiseHybridStorageEstimate estimate;
    if (!root || state_dim == 0) {
        return estimate;
    }

    std::unordered_set<WeightedNodePairKey, WeightedNodePairKeyHash> visited_pairs;
    std::unordered_set<HDDNode*> visited_nodes;

    std::function<size_t(HDDNode*, std::complex<double>, HDDNode*, std::complex<double>)>
        count_pairs =
            [&](HDDNode* low_node,
                std::complex<double> low_weight,
                HDDNode* high_node,
                std::complex<double> high_weight) -> size_t {
                if (!low_node || !high_node) {
                    throw std::runtime_error(
                        "pairwise hybrid storage estimate encountered null HDD branch");
                }

                const WeightedNodePairKey key{low_node, high_node, low_weight, high_weight};
                if (!visited_pairs.insert(key).second) {
                    return 0;
                }

                if (low_node->is_terminal() && high_node->is_terminal()) {
                    return 1;
                }

                if (low_node->is_terminal() || high_node->is_terminal()) {
                    throw std::runtime_error(
                        "pairwise hybrid storage estimate encountered mismatched HDD structure");
                }
                if (low_node->qubit_level != high_node->qubit_level) {
                    throw std::runtime_error(
                        "pairwise hybrid storage estimate encountered mismatched HDD levels");
                }

                return count_pairs(low_node->low,
                                   low_weight * low_node->w_low,
                                   high_node->low,
                                   high_weight * high_node->w_low) +
                       count_pairs(low_node->high,
                                   low_weight * low_node->w_high,
                                   high_node->high,
                                   high_weight * high_node->w_high);
            };

    std::function<size_t(HDDNode*)> count_transform =
        [&](HDDNode* node) -> size_t {
            if (!node || node->is_terminal()) {
                return 0;
            }

            if (!visited_nodes.insert(node).second) {
                return 0;
            }

            if (node->qubit_level == control_qubit) {
                return count_pairs(node->low, node->w_low, node->high, node->w_high);
            }
            if (node->qubit_level > control_qubit) {
                return count_transform(node->low) + count_transform(node->high);
            }
            return 0;
        };

    estimate.pair_count = count_transform(root);
    if (estimate.pair_count == 0) {
        return estimate;
    }
    if (estimate.pair_count > std::numeric_limits<size_t>::max() / 2) {
        throw std::overflow_error("pairwise hybrid duplicate state estimate overflow");
    }

    estimate.duplicate_state_count = estimate.pair_count * 2;
    if (estimate.duplicate_state_count > std::numeric_limits<size_t>::max() / state_dim) {
        throw std::overflow_error("pairwise hybrid storage estimate overflow");
    }
    estimate.extra_elements = estimate.duplicate_state_count * state_dim;
    return estimate;
}

std::unordered_map<int, size_t> collect_pairwise_replaced_state_use_counts(
    HDDNode* root,
    int control_qubit) {
    std::unordered_map<int, size_t> state_use_counts;
    if (!root) {
        return state_use_counts;
    }

    std::unordered_set<WeightedNodePairKey, WeightedNodePairKeyHash> visited_pairs;
    std::unordered_set<HDDNode*> visited_nodes;

    std::function<void(HDDNode*, std::complex<double>, HDDNode*, std::complex<double>)> count_pairs =
        [&](HDDNode* low_node,
            std::complex<double> low_weight,
            HDDNode* high_node,
            std::complex<double> high_weight) {
            if (!low_node || !high_node) {
                throw std::runtime_error(
                    "pairwise replaced-state counting encountered null HDD branch");
            }

            const WeightedNodePairKey key{low_node, high_node, low_weight, high_weight};
            if (!visited_pairs.insert(key).second) {
                return;
            }

            if (low_node->is_terminal() && high_node->is_terminal()) {
                if (low_node->tensor_id >= 0) {
                    ++state_use_counts[low_node->tensor_id];
                }
                if (high_node->tensor_id >= 0) {
                    ++state_use_counts[high_node->tensor_id];
                }
                return;
            }

            if (low_node->is_terminal() || high_node->is_terminal()) {
                throw std::runtime_error(
                    "pairwise replaced-state counting encountered mismatched HDD structure");
            }
            if (low_node->qubit_level != high_node->qubit_level) {
                throw std::runtime_error(
                    "pairwise replaced-state counting encountered mismatched HDD levels");
            }

            count_pairs(low_node->low,
                        low_weight * low_node->w_low,
                        high_node->low,
                        high_weight * high_node->w_low);
            count_pairs(low_node->high,
                        low_weight * low_node->w_high,
                        high_node->high,
                        high_weight * high_node->w_high);
        };

    std::function<void(HDDNode*)> walk_transform_region =
        [&](HDDNode* node) {
            if (!node || node->is_terminal()) {
                return;
            }
            if (!visited_nodes.insert(node).second) {
                return;
            }
            if (node->qubit_level == control_qubit) {
                count_pairs(node->low, node->w_low, node->high, node->w_high);
                return;
            }
            if (node->qubit_level > control_qubit) {
                walk_transform_region(node->low);
                walk_transform_region(node->high);
            }
        };

    walk_transform_region(root);
    return state_use_counts;
}

std::unordered_set<int> collect_retained_state_ids_outside_pairwise_region(
    HDDNode* root,
    int control_qubit) {
    std::unordered_set<int> retained_state_ids;
    if (!root) {
        return retained_state_ids;
    }

    std::unordered_set<size_t> visited_transform_nodes;
    std::unordered_set<size_t> visited_retained_nodes;
    std::function<void(HDDNode*)> collect_retained_subtree =
        [&](HDDNode* node) {
            if (!node) {
                return;
            }
            if (!visited_retained_nodes.insert(node->get_unique_id()).second) {
                return;
            }
            if (node->is_terminal()) {
                if (node->tensor_id >= 0) {
                    retained_state_ids.insert(node->tensor_id);
                }
                return;
            }
            collect_retained_subtree(node->low);
            collect_retained_subtree(node->high);
        };

    std::function<void(HDDNode*)> walk =
        [&](HDDNode* node) {
            if (!node) {
                return;
            }
            if (node->is_terminal()) {
                return;
            }
            if (node->qubit_level < control_qubit) {
                collect_retained_subtree(node);
                return;
            }
            if (!visited_transform_nodes.insert(node->get_unique_id()).second) {
                return;
            }
            if (node->qubit_level > control_qubit) {
                walk(node->low);
                walk(node->high);
            }
        };

    walk(root);
    return retained_state_ids;
}

size_t pairwise_hybrid_working_extra_elements(
    HDDNode* root,
    int control_qubit,
    size_t state_dim,
    const PairwiseHybridStorageEstimate& full_estimate,
    bool* early_release_enabled = nullptr) {
    const bool allow_early_release =
        full_estimate.extra_elements != 0 &&
        collect_retained_state_ids_outside_pairwise_region(root, control_qubit).empty();
    if (early_release_enabled) {
        *early_release_enabled = allow_early_release;
    }
    if (!allow_early_release || full_estimate.extra_elements == 0) {
        return full_estimate.extra_elements;
    }
    if (state_dim > std::numeric_limits<size_t>::max() / 2) {
        throw std::overflow_error("pairwise hybrid working-set estimate overflow");
    }
    return std::min(full_estimate.extra_elements, static_cast<size_t>(2) * state_dim);
}

void release_pairwise_replaced_state_if_safe(
    int state_id,
    bool allow_early_release,
    std::unordered_map<int, size_t>& remaining_use_counts,
    CVStatePool& state_pool,
    int shared_zero_state_id) {
    if (!allow_early_release || state_id < 0 || state_id == shared_zero_state_id) {
        return;
    }

    const auto it = remaining_use_counts.find(state_id);
    if (it == remaining_use_counts.end()) {
        return;
    }

    if (it->second == 0) {
        remaining_use_counts.erase(it);
        return;
    }

    if (--(it->second) != 0) {
        return;
    }

    remaining_use_counts.erase(it);
    if (state_pool.is_valid_state(state_id)) {
        state_pool.free_state(state_id);
    }
}

void reserve_pairwise_hybrid_headroom(const char* gate_name,
                                      HDDNode* root,
                                      int control_qubit,
                                      CVStatePool& state_pool) {
    const size_t state_dim = static_cast<size_t>(state_pool.get_max_total_dim());
    if (state_dim == 0) {
        return;
    }

    const PairwiseHybridStorageEstimate estimate =
        estimate_pairwise_hybrid_storage(root, control_qubit, state_dim);
    if (estimate.extra_elements == 0) {
        return;
    }

    bool early_release_enabled = false;
    const size_t working_extra_elements =
        pairwise_hybrid_working_extra_elements(
            root, control_qubit, state_dim, estimate, &early_release_enabled);

    const size_t active_storage = state_pool.get_active_storage_elements();
    if (active_storage > std::numeric_limits<size_t>::max() - working_extra_elements) {
        throw std::overflow_error("pairwise hybrid headroom reservation overflow");
    }

    FALLBACK_DEBUG_LOG << "[fallback] " << gate_name
                       << " pre-reserving pairwise headroom"
                       << " pairs=" << estimate.pair_count
                       << " duplicate_states=" << estimate.duplicate_state_count
                       << " working_extra_elements=" << working_extra_elements
                       << " early_release=" << (early_release_enabled ? 1 : 0)
                       << " active_storage=" << active_storage
                       << std::endl;
    state_pool.reserve_total_storage_elements(active_storage + working_extra_elements);
}

void cleanup_duplicated_pairwise_states(CVStatePool& state_pool,
                                        const std::vector<int>& low_ids,
                                        const std::vector<int>& high_ids) {
    std::unordered_set<int> unique_state_ids;
    unique_state_ids.reserve(low_ids.size() + high_ids.size());
    for (int state_id : low_ids) {
        if (state_id >= 0) {
            unique_state_ids.insert(state_id);
        }
    }
    for (int state_id : high_ids) {
        if (state_id >= 0) {
            unique_state_ids.insert(state_id);
        }
    }

    for (int state_id : unique_state_ids) {
        if (state_pool.is_valid_state(state_id)) {
            state_pool.free_state(state_id);
        }
    }
}

void cleanup_pairwise_build_failure(
    HDDNodeManager& node_manager,
    CVStatePool& state_pool,
    const std::unordered_map<WeightedNodePairKey, std::pair<HDDNode*, HDDNode*>, WeightedNodePairKeyHash>& pair_memo,
    const std::unordered_map<HDDNode*, HDDNode*>& node_memo,
    const std::vector<int>& low_ids,
    const std::vector<int>& high_ids) {
    for (const auto& [original_node, transformed_node] : node_memo) {
        if (transformed_node && transformed_node != original_node) {
            node_manager.release_node(transformed_node);
        }
    }

    for (const auto& [pair_key, pair] : pair_memo) {
        (void)pair_key;
        if (pair.first) {
            node_manager.release_node(pair.first);
        }
        if (pair.second) {
            node_manager.release_node(pair.second);
        }
    }

    cleanup_duplicated_pairwise_states(state_pool, low_ids, high_ids);
}

void release_transient_pairwise_node(
    HDDNodeManager& node_manager,
    CVStatePool& state_pool,
    HDDNode* node,
    const std::vector<int>& low_ids,
    const std::vector<int>& high_ids) {
    if (!node) {
        return;
    }

    const int state_id = node->tensor_id;
    node_manager.release_node(node);

    const bool tracked_low =
        std::find(low_ids.begin(), low_ids.end(), state_id) != low_ids.end();
    const bool tracked_high =
        std::find(high_ids.begin(), high_ids.end(), state_id) != high_ids.end();
    if (state_id >= 0 && !tracked_low && !tracked_high && state_pool.is_valid_state(state_id)) {
        state_pool.free_state(state_id);
    }
}

std::vector<double> expand_selective_rotation_profile(
    const std::vector<double>& per_photon_values,
    int trunc_dim,
    int control_qumode,
    int num_qumodes,
    int64_t max_total_dim) {
    std::vector<double> expanded(static_cast<size_t>(max_total_dim), 0.0);
    if (trunc_dim <= 0 || max_total_dim <= 0 || control_qumode < 0 || control_qumode >= num_qumodes) {
        return expanded;
    }

    const int right_stride = compute_qumode_right_stride(trunc_dim, control_qumode, num_qumodes);
    for (int64_t flat_index = 0; flat_index < max_total_dim; ++flat_index) {
        const int photon_number = (flat_index / right_stride) % trunc_dim;
        if (photon_number < static_cast<int>(per_photon_values.size())) {
            expanded[static_cast<size_t>(flat_index)] = per_photon_values[static_cast<size_t>(photon_number)];
        }
    }
    return expanded;
}

constexpr double kVacuumTolerance = 1e-12;
constexpr double kSymbolicBranchPruneTolerance = 1e-14;
constexpr int kDefaultSymbolicBranchLimit = 64;
constexpr double kTargetMixtureFidelity = 0.9999;
constexpr int kMaxKerrMixtureBranches = 32;
constexpr int kMaxSnapMixtureBranches = 32;
constexpr int kMaxCrossKerrMixtureBranches = 64;
constexpr double kDiagonalCanonicalizationTolerance = 1e-14;

size_t integer_power(size_t base, int exponent) {
    size_t result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

bool is_nontrivial_phase(double theta) {
    return std::abs(theta) > kDiagonalCanonicalizationTolerance;
}

bool symbolic_mixture_would_exceed_branch_limit(
    const std::vector<std::complex<double>>& initial_weights,
    const std::vector<GaussianMixtureApproximation>& approximations,
    size_t branch_limit,
    size_t* projected_branch_count,
    size_t* failing_update_index) {
    std::vector<std::complex<double>> current_weights;
    current_weights.reserve(initial_weights.size());
    for (const std::complex<double>& weight : initial_weights) {
        if (std::abs(weight) >= kSymbolicBranchPruneTolerance) {
            current_weights.push_back(weight);
        }
    }

    if (projected_branch_count) {
        *projected_branch_count = current_weights.size();
    }
    if (failing_update_index) {
        *failing_update_index = approximations.size();
    }

    for (size_t approximation_index = 0; approximation_index < approximations.size();
         ++approximation_index) {
        const GaussianMixtureApproximation& approximation = approximations[approximation_index];
        std::vector<std::complex<double>> expanded_weights;
        expanded_weights.reserve(std::min(branch_limit + 1, current_weights.size() + approximation.branches.size()));

        for (const std::complex<double>& base_weight : current_weights) {
            for (const GaussianMixtureBranch& mixture_branch : approximation.branches) {
                const std::complex<double> new_weight = base_weight * mixture_branch.weight;
                if (std::abs(new_weight) < kSymbolicBranchPruneTolerance) {
                    continue;
                }

                expanded_weights.push_back(new_weight);
                if (expanded_weights.size() > branch_limit) {
                    if (projected_branch_count) {
                        *projected_branch_count = expanded_weights.size();
                    }
                    if (failing_update_index) {
                        *failing_update_index = approximation_index;
                    }
                    return true;
                }
            }
        }

        current_weights = std::move(expanded_weights);
        if (projected_branch_count) {
            *projected_branch_count = current_weights.size();
        }
        if (current_weights.empty()) {
            return false;
        }
    }

    return false;
}

double conservative_fidelity_lower_bound_from_operator_error(double operator_error) {
    if (operator_error <= 0.0) {
        return 1.0;
    }
    if (operator_error >= 1.0) {
        return 0.0;
    }

    const double overlap_lower_bound = (1.0 - operator_error) / (1.0 + operator_error);
    return overlap_lower_bound * overlap_lower_bound;
}

HDDNode* find_all_zero_qubit_terminal(HDDNode* node) {
    HDDNode* current = node;
    while (current && !current->is_terminal()) {
        current = current->low;
    }
    return current;
}

bool is_vacuum_fock_state(const std::vector<cuDoubleComplex>& state) {
    if (state.empty()) {
        return false;
    }

    if (std::abs(cuCreal(state[0]) - 1.0) > kVacuumTolerance ||
        std::abs(cuCimag(state[0])) > kVacuumTolerance) {
        return false;
    }

    for (size_t i = 1; i < state.size(); ++i) {
        if (std::abs(cuCreal(state[i])) > kVacuumTolerance ||
            std::abs(cuCimag(state[i])) > kVacuumTolerance) {
            return false;
        }
    }

    return true;
}

struct VacuumRayInfo {
    bool is_zero = false;
    bool is_scaled_vacuum = false;
    std::complex<double> scale{0.0, 0.0};
};

VacuumRayInfo classify_vacuum_ray(const std::vector<cuDoubleComplex>& state) {
    VacuumRayInfo info;
    if (state.empty()) {
        return info;
    }

    bool all_zero = true;
    for (const cuDoubleComplex amplitude : state) {
        if (std::abs(cuCreal(amplitude)) > kVacuumTolerance ||
            std::abs(cuCimag(amplitude)) > kVacuumTolerance) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        info.is_zero = true;
        info.is_scaled_vacuum = true;
        return info;
    }

    for (size_t idx = 1; idx < state.size(); ++idx) {
        if (std::abs(cuCreal(state[idx])) > kVacuumTolerance ||
            std::abs(cuCimag(state[idx])) > kVacuumTolerance) {
            return info;
        }
    }

    info.is_scaled_vacuum = true;
    info.scale = std::complex<double>(cuCreal(state[0]), cuCimag(state[0]));
    return info;
}

VacuumRayInfo classify_vacuum_ray_on_device(CVStatePool& state_pool, int state_id) {
    int is_zero = 0;
    int is_scaled_vacuum = 0;
    cuDoubleComplex scale = make_cuDoubleComplex(0.0, 0.0);
    classify_vacuum_ray_device(
        &state_pool,
        state_id,
        kVacuumTolerance,
        &is_zero,
        &is_scaled_vacuum,
        &scale);

    VacuumRayInfo info;
    info.is_zero = (is_zero != 0);
    info.is_scaled_vacuum = (is_scaled_vacuum != 0);
    info.scale = std::complex<double>(cuCreal(scale), cuCimag(scale));
    return info;
}

void clear_cuda_runtime_error_state() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess && err != cudaErrorNotReady) {
        cudaGetLastError();
    }

    err = cudaGetLastError();
    if (err != cudaSuccess && err != cudaErrorNotReady) {
        cudaGetLastError();
    }
}

std::vector<cuDoubleComplex> to_cuda_state(
    const std::vector<std::complex<double>>& state) {
    std::vector<cuDoubleComplex> cuda_state;
    cuda_state.reserve(state.size());
    for (const auto& amplitude : state) {
        cuda_state.push_back(make_cuDoubleComplex(amplitude.real(), amplitude.imag()));
    }
    return cuda_state;
}

std::vector<std::complex<double>> to_host_complex(
    const std::vector<cuDoubleComplex>& state) {
    std::vector<std::complex<double>> host_state;
    host_state.reserve(state.size());
    for (const cuDoubleComplex amplitude : state) {
        host_state.emplace_back(cuCreal(amplitude), cuCimag(amplitude));
    }
    return host_state;
}

}  // namespace

void QuantumCircuit::ensure_async_cv_pipeline() {
    if (compute_stream_ && upload_stream_) {
        return;
    }

    CHECK_CUDA(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking));
    for (TargetUploadSlot& slot : target_upload_slots_) {
        if (!slot.upload_ready) {
            CHECK_CUDA(cudaEventCreateWithFlags(&slot.upload_ready, cudaEventDisableTiming));
        }
        if (!slot.reusable) {
            CHECK_CUDA(cudaEventCreateWithFlags(&slot.reusable, cudaEventDisableTiming));
        }
        slot.reusable_recorded = false;
    }
    next_target_upload_slot_ = 0;
    async_cv_work_pending_ = false;
}

void QuantumCircuit::release_async_cv_pipeline() {
    synchronize_async_cv_pipeline();

    for (TargetUploadSlot& slot : target_upload_slots_) {
        if (slot.upload_ready) {
            cudaEventDestroy(slot.upload_ready);
            slot.upload_ready = nullptr;
        }
        if (slot.reusable) {
            cudaEventDestroy(slot.reusable);
            slot.reusable = nullptr;
        }
        slot.reusable_recorded = false;
        slot.device_buffer.release();
        slot.host_buffer.release();
    }

    if (upload_stream_) {
        cudaStreamDestroy(upload_stream_);
        upload_stream_ = nullptr;
    }
    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
        compute_stream_ = nullptr;
    }
    next_target_upload_slot_ = 0;
    async_cv_work_pending_ = false;
}

void QuantumCircuit::prewarm_async_target_upload_slots() {
    ensure_async_cv_pipeline();

    const size_t target_id_bytes = static_cast<size_t>(state_pool_.capacity) * sizeof(int);
    if (target_id_bytes == 0) {
        return;
    }

    for (TargetUploadSlot& slot : target_upload_slots_) {
        slot.device_buffer.ensure(target_id_bytes);
        slot.host_buffer.ensure(target_id_bytes);
    }
}

void QuantumCircuit::synchronize_async_cv_pipeline() {
    if (!async_cv_work_pending_) {
        return;
    }
    if (upload_stream_) {
        CHECK_CUDA(cudaStreamSynchronize(upload_stream_));
    }
    if (compute_stream_) {
        CHECK_CUDA(cudaStreamSynchronize(compute_stream_));
    }
    async_cv_work_pending_ = false;
}

std::pair<int*, size_t> QuantumCircuit::upload_target_states_for_compute(
    const std::vector<int>& target_states,
    size_t* slot_index) {
    if (target_states.empty()) {
        if (slot_index) {
            *slot_index = 0;
        }
        return {nullptr, 0};
    }

    if (!async_cv_pipeline_enabled_) {
        int* d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
        if (slot_index) {
            *slot_index = 0;
        }
        return {d_target_ids, target_states.size() * sizeof(int)};
    }

    ensure_async_cv_pipeline();
    const size_t ids_bytes = target_states.size() * sizeof(int);
    const size_t current_slot = next_target_upload_slot_;
    TargetUploadSlot& slot = target_upload_slots_[current_slot];

    if (slot.reusable_recorded) {
        CHECK_CUDA(cudaEventSynchronize(slot.reusable));
        slot.reusable_recorded = false;
    }

    int* staged_host = static_cast<int*>(slot.host_buffer.ensure(ids_bytes));
    std::memcpy(staged_host, target_states.data(), ids_bytes);
    int* device_target_ids = static_cast<int*>(slot.device_buffer.ensure(ids_bytes));

    CHECK_CUDA(cudaMemcpyAsync(device_target_ids,
                               staged_host,
                               ids_bytes,
                               cudaMemcpyHostToDevice,
                               upload_stream_));
    CHECK_CUDA(cudaEventRecord(slot.upload_ready, upload_stream_));
    CHECK_CUDA(cudaStreamWaitEvent(compute_stream_, slot.upload_ready, 0));

    next_target_upload_slot_ = (current_slot + 1) % target_upload_slots_.size();
    if (slot_index) {
        *slot_index = current_slot;
    }
    return {device_target_ids, ids_bytes};
}

void QuantumCircuit::mark_target_upload_slot_in_use(size_t slot_index) {
    if (!async_cv_pipeline_enabled_) {
        return;
    }
    if (slot_index >= target_upload_slots_.size()) {
        throw std::out_of_range("target upload slot index out of range");
    }
    TargetUploadSlot& slot = target_upload_slots_[slot_index];
    CHECK_CUDA(cudaEventRecord(slot.reusable, compute_stream_));
    slot.reusable_recorded = true;
    async_cv_work_pending_ = true;
}

void QuantumCircuit::invalidate_root_caches() {
    ++root_revision_;
    cached_target_state_revision_ = 0;
    cached_target_state_ids_.clear();
    cached_symbolic_terminal_revision_ = 0;
    cached_symbolic_terminal_ids_.clear();
}

const std::vector<int>& QuantumCircuit::get_cached_target_states() const {
    if (cached_target_state_revision_ == root_revision_) {
        return cached_target_state_ids_;
    }

    cached_target_state_ids_ = collect_terminal_state_ids(root_node_);
    cached_target_state_ids_.erase(
        std::remove(cached_target_state_ids_.begin(),
                    cached_target_state_ids_.end(),
                    shared_zero_state_id_),
        cached_target_state_ids_.end());
    cached_target_state_revision_ = root_revision_;
    return cached_target_state_ids_;
}

const std::vector<int>& QuantumCircuit::get_cached_symbolic_terminal_ids() const {
    if (cached_symbolic_terminal_revision_ == root_revision_) {
        return cached_symbolic_terminal_ids_;
    }

    cached_symbolic_terminal_ids_ = collect_symbolic_terminal_ids(root_node_);
    cached_symbolic_terminal_revision_ = root_revision_;
    return cached_symbolic_terminal_ids_;
}

namespace {

void trim_trailing_zero_phases(std::vector<double>* phase_map) {
    while (!phase_map->empty() && !is_nontrivial_phase(phase_map->back())) {
        phase_map->pop_back();
    }
}

GateParams make_two_mode_gate(GateType type,
                              int first_target_qumode,
                              int second_target_qumode,
                              std::complex<double> param) {
    return GateParams(type, {}, {first_target_qumode, second_target_qumode}, {param});
}

bool is_unconditional_gaussian_gate(const GateParams& gate) {
    if (!gate.target_qubits.empty()) {
        return false;
    }

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 1;
        case GateType::DISPLACEMENT:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 1;
        case GateType::SQUEEZING:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 1;
        case GateType::BEAM_SPLITTER:
            return gate.target_qumodes.size() == 2 && gate.params.size() >= 1;
        default:
            return false;
    }
}

bool is_supported_controlled_gaussian_gate(const GateParams& gate) {
    switch (gate.type) {
        case GateType::CONDITIONAL_DISPLACEMENT:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 1 &&
                   gate.params.size() >= 1;
        case GateType::CONDITIONAL_SQUEEZING:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 1 &&
                   gate.params.size() >= 1;
        case GateType::CONDITIONAL_BEAM_SPLITTER:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 2 &&
                   gate.params.size() >= 1;
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 2 &&
                   gate.params.size() >= 1;
        case GateType::CONDITIONAL_SUM:
            return gate.target_qubits.size() == 1 &&
                   gate.target_qumodes.size() == 2 &&
                   gate.params.size() >= 1 &&
                   (gate.params.size() < 2 || std::abs(gate.params[1]) < 1e-14);
        default:
            return false;
    }
}

bool is_gaussian_track_gate(const GateParams& gate) {
    return is_unconditional_gaussian_gate(gate) ||
           is_supported_controlled_gaussian_gate(gate);
}

bool is_pure_qubit_gate(const GateParams& gate) {
    if (!gate.target_qumodes.empty()) {
        return false;
    }

    switch (gate.type) {
        case GateType::HADAMARD:
        case GateType::PAULI_X:
        case GateType::PAULI_Y:
        case GateType::PAULI_Z:
        case GateType::ROTATION_X:
        case GateType::ROTATION_Y:
        case GateType::ROTATION_Z:
        case GateType::PHASE_GATE_S:
        case GateType::PHASE_GATE_T:
        case GateType::CNOT:
        case GateType::CZ:
            return !gate.target_qubits.empty();
        default:
            return false;
    }
}

const char* gate_type_name(GateType type) {
    switch (type) {
        case GateType::HADAMARD: return "Hadamard";
        case GateType::PAULI_X: return "PauliX";
        case GateType::PAULI_Y: return "PauliY";
        case GateType::PAULI_Z: return "PauliZ";
        case GateType::ROTATION_X: return "RotationX";
        case GateType::ROTATION_Y: return "RotationY";
        case GateType::ROTATION_Z: return "RotationZ";
        case GateType::PHASE_GATE_S: return "PhaseS";
        case GateType::PHASE_GATE_T: return "PhaseT";
        case GateType::CNOT: return "CNOT";
        case GateType::CZ: return "CZ";
        case GateType::PHASE_ROTATION: return "PhaseRotation";
        case GateType::KERR_GATE: return "Kerr";
        case GateType::CONDITIONAL_PARITY: return "ConditionalParity";
        case GateType::SNAP_GATE: return "SNAP";
        case GateType::MULTI_SNAP_GATE: return "MultiSNAP";
        case GateType::CROSS_KERR_GATE: return "CrossKerr";
        case GateType::CREATION_OPERATOR: return "Creation";
        case GateType::ANNIHILATION_OPERATOR: return "Annihilation";
        case GateType::DISPLACEMENT: return "Displacement";
        case GateType::SQUEEZING: return "Squeezing";
        case GateType::BEAM_SPLITTER: return "BeamSplitter";
        case GateType::CONDITIONAL_DISPLACEMENT: return "ConditionalDisplacement";
        case GateType::CONDITIONAL_SQUEEZING: return "ConditionalSqueezing";
        case GateType::CONDITIONAL_BEAM_SPLITTER: return "ConditionalBeamSplitter";
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING: return "ConditionalTwoModeSqueezing";
        case GateType::CONDITIONAL_SUM: return "ConditionalSUM";
        case GateType::RABI_INTERACTION: return "RabiInteraction";
        case GateType::JAYNES_CUMMINGS: return "JaynesCummings";
        case GateType::ANTI_JAYNES_CUMMINGS: return "AntiJaynesCummings";
        case GateType::SELECTIVE_QUBIT_ROTATION: return "SelectiveQubitRotation";
        default: return "UnknownGate";
    }
}

bool validate_gaussian_track_gate(const GateParams& gate,
                                  int total_qubits,
                                  int total_qumodes,
                                  std::string* error_message) {
    auto fail = [error_message](const std::string& message) {
        if (error_message) {
            *error_message = message;
        }
        return false;
    };

    auto validate_qumode = [&](int qumode) {
        return qumode >= 0 && qumode < total_qumodes;
    };

    auto validate_single_qumode_gate =
        [&](const char* gate_name, bool requires_control, bool require_two_modes) {
            const size_t expected_qumodes = require_two_modes ? 2 : 1;
            if (gate.target_qumodes.size() != expected_qumodes) {
                return fail(std::string(gate_name) + " 目标qumode数量不正确");
            }
            for (int qumode : gate.target_qumodes) {
                if (!validate_qumode(qumode)) {
                    return fail(std::string(gate_name) + " 目标qumode越界");
                }
            }
            if (require_two_modes && gate.target_qumodes[0] == gate.target_qumodes[1]) {
                return fail(std::string(gate_name) + " 需要两个不同的目标qumode");
            }
            if (gate.params.empty()) {
                return fail(std::string(gate_name) + " 缺少参数");
            }
            if (requires_control) {
                if (gate.target_qubits.size() != 1) {
                    return fail(std::string(gate_name) + " 需要且仅需要一个控制qubit");
                }
                const int control_qubit = gate.target_qubits[0];
                if (control_qubit < 0 || control_qubit >= total_qubits) {
                    return fail(std::string(gate_name) + " 控制qubit越界");
                }
            } else if (!gate.target_qubits.empty()) {
                return fail(std::string(gate_name) + " 不应携带qubit目标");
            }
            return true;
        };

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            return validate_single_qumode_gate("PhaseRotation", false, false);
        case GateType::DISPLACEMENT:
            return validate_single_qumode_gate("Displacement", false, false);
        case GateType::SQUEEZING:
            return validate_single_qumode_gate("Squeezing", false, false);
        case GateType::BEAM_SPLITTER:
            return validate_single_qumode_gate("BeamSplitter", false, true);
        case GateType::CONDITIONAL_DISPLACEMENT:
            return validate_single_qumode_gate("ConditionalDisplacement", true, false);
        case GateType::CONDITIONAL_SQUEEZING:
            return validate_single_qumode_gate("ConditionalSqueezing", true, false);
        case GateType::CONDITIONAL_BEAM_SPLITTER:
            return validate_single_qumode_gate("ConditionalBeamSplitter", true, true);
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
            return validate_single_qumode_gate("ConditionalTwoModeSqueezing", true, true);
        case GateType::CONDITIONAL_SUM:
            if (!validate_single_qumode_gate("ConditionalSUM", true, true)) {
                return false;
            }
            if (gate.params.size() >= 2 && std::abs(gate.params[1]) > 1e-14) {
                return fail("ConditionalSUM 当前仅支持 phi = 0 进入Gaussian track");
            }
            return true;
        default:
            return fail("该门当前不能进入Gaussian track");
    }
}

std::vector<int> collect_gaussian_control_qubits(const std::vector<GateParams>& gates) {
    std::unordered_set<int> seen_controls;
    std::vector<int> control_qubits;
    for (const GateParams& gate : gates) {
        if (is_supported_controlled_gaussian_gate(gate) &&
            seen_controls.insert(gate.target_qubits[0]).second) {
            control_qubits.push_back(gate.target_qubits[0]);
        }
    }
    std::sort(control_qubits.begin(), control_qubits.end());
    return control_qubits;
}

std::string make_control_assignment_key(const std::vector<int>& control_qubits,
                                        const std::vector<int>& qubit_assignment) {
    std::string key;
    key.reserve(control_qubits.size());
    for (int control_qubit : control_qubits) {
        key.push_back(qubit_assignment[control_qubit] == 0 ? '0' : '1');
    }
    return key;
}

std::vector<int> assignment_from_key(const std::vector<int>& control_qubits,
                                     const std::string& key,
                                     int total_qubits) {
    std::vector<int> assignment(static_cast<size_t>(total_qubits), 0);
    for (size_t idx = 0; idx < control_qubits.size(); ++idx) {
        assignment[control_qubits[idx]] = key[idx] == '1' ? 1 : 0;
    }
    return assignment;
}

std::optional<GateParams> resolve_gaussian_gate_for_assignment(
    const GateParams& gate,
    const std::vector<int>& qubit_assignment) {
    if (is_unconditional_gaussian_gate(gate)) {
        return gate;
    }

    switch (gate.type) {
        case GateType::CONDITIONAL_DISPLACEMENT: {
            const int control_qubit = gate.target_qubits[0];
            if (qubit_assignment[control_qubit] == 0) {
                return std::nullopt;
            }
            return GateParams(
                GateType::DISPLACEMENT,
                {},
                {gate.target_qumodes[0]},
                {gate.params[0]});
        }
        case GateType::CONDITIONAL_SQUEEZING: {
            const int control_qubit = gate.target_qubits[0];
            const std::complex<double> xi =
                qubit_assignment[control_qubit] == 0 ? gate.params[0] : -gate.params[0];
            return GateParams(
                GateType::SQUEEZING,
                {},
                {gate.target_qumodes[0]},
                {xi});
        }
        case GateType::CONDITIONAL_BEAM_SPLITTER: {
            const int control_qubit = gate.target_qubits[0];
            const double theta =
                qubit_assignment[control_qubit] == 0 ?
                gate.params[0].real() :
                -gate.params[0].real();
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return GateParams(
                GateType::BEAM_SPLITTER,
                {},
                {gate.target_qumodes[0], gate.target_qumodes[1]},
                {
                    std::complex<double>(theta, 0.0),
                    std::complex<double>(phi, 0.0)
                });
        }
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING: {
            const int control_qubit = gate.target_qubits[0];
            const std::complex<double> xi =
                qubit_assignment[control_qubit] == 0 ? gate.params[0] : -gate.params[0];
            return GateParams(
                GateType::CONDITIONAL_TWO_MODE_SQUEEZING,
                {},
                {gate.target_qumodes[0], gate.target_qumodes[1]},
                {xi});
        }
        case GateType::CONDITIONAL_SUM: {
            const int control_qubit = gate.target_qubits[0];
            const double theta =
                qubit_assignment[control_qubit] == 0 ?
                gate.params[0].real() :
                -gate.params[0].real();
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return GateParams(
                GateType::CONDITIONAL_SUM,
                {},
                {gate.target_qumodes[0], gate.target_qumodes[1]},
                {
                    std::complex<double>(theta, 0.0),
                    std::complex<double>(phi, 0.0)
                });
        }
        default:
            return std::nullopt;
    }
}

void scale_cuda_state(std::vector<cuDoubleComplex>* state,
                      std::complex<double> scale) {
    const cuDoubleComplex scale_cu = make_cuDoubleComplex(scale.real(), scale.imag());
    for (cuDoubleComplex& amplitude : *state) {
        amplitude = cuCmul(scale_cu, amplitude);
    }
}

void collect_symbolic_terminal_ids_recursive(
    HDDNode* node,
    std::unordered_set<size_t>& visited_nodes,
    std::unordered_set<int>& symbolic_ids) {
    if (!node) {
        return;
    }
    if (!visited_nodes.insert(node->get_unique_id()).second) {
        return;
    }
    if (node->is_terminal()) {
        if (node->tensor_id < -1) {
            symbolic_ids.insert(node->tensor_id);
        }
        return;
    }
    collect_symbolic_terminal_ids_recursive(node->low, visited_nodes, symbolic_ids);
    collect_symbolic_terminal_ids_recursive(node->high, visited_nodes, symbolic_ids);
}

int choose_snap_mixture_branch_cap(const GateParams& gate, int cutoff) {
    if (cutoff <= 0) {
        return 0;
    }

    if (gate.type == GateType::SNAP_GATE) {
        return std::min(cutoff, kMaxSnapMixtureBranches);
    }

    if (gate.type == GateType::MULTI_SNAP_GATE) {
        return std::min(cutoff, kMaxSnapMixtureBranches);
    }

    return 0;
}

int choose_cross_kerr_mixture_branch_cap(int cutoff) {
    if (cutoff <= 0) {
        return 0;
    }
    return std::min(cutoff * cutoff, kMaxCrossKerrMixtureBranches);
}

template <typename PhaseFactorFn>
void apply_single_mode_diagonal_exact(std::vector<std::complex<double>>* state,
                                      int cutoff,
                                      int num_qumodes,
                                      int target_qumode,
                                      PhaseFactorFn&& phase_factor) {
    if (target_qumode < 0 || target_qumode >= num_qumodes) {
        throw std::out_of_range("single-mode diagonal target qumode out of range");
    }

    const size_t expected_dim = integer_power(static_cast<size_t>(cutoff), num_qumodes);
    if (state->size() != expected_dim) {
        throw std::invalid_argument("state size does not match cutoff^num_qumodes");
    }

    const size_t stride = integer_power(static_cast<size_t>(cutoff), num_qumodes - target_qumode - 1);
    const size_t prefix_count = integer_power(static_cast<size_t>(cutoff), target_qumode);

    for (size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const size_t block_base = prefix * static_cast<size_t>(cutoff) * stride;
        for (int photon = 0; photon < cutoff; ++photon) {
            const std::complex<double> factor = phase_factor(photon);
            if (std::abs(factor - std::complex<double>(1.0, 0.0)) < kDiagonalCanonicalizationTolerance) {
                continue;
            }
            for (size_t suffix = 0; suffix < stride; ++suffix) {
                (*state)[block_base + static_cast<size_t>(photon) * stride + suffix] *= factor;
            }
        }
    }
}

template <typename PhaseFactorFn>
void apply_two_mode_diagonal_exact(std::vector<std::complex<double>>* state,
                                   int cutoff,
                                   int num_qumodes,
                                   int first_target_qumode,
                                   int second_target_qumode,
                                   PhaseFactorFn&& phase_factor) {
    if (first_target_qumode < 0 || first_target_qumode >= num_qumodes ||
        second_target_qumode < 0 || second_target_qumode >= num_qumodes ||
        first_target_qumode == second_target_qumode) {
        throw std::out_of_range("two-mode diagonal target qumode out of range");
    }

    const size_t expected_dim = integer_power(static_cast<size_t>(cutoff), num_qumodes);
    if (state->size() != expected_dim) {
        throw std::invalid_argument("state size does not match cutoff^num_qumodes");
    }

    const size_t first_stride =
        integer_power(static_cast<size_t>(cutoff), num_qumodes - first_target_qumode - 1);
    const size_t second_stride =
        integer_power(static_cast<size_t>(cutoff), num_qumodes - second_target_qumode - 1);

    for (size_t flat_index = 0; flat_index < state->size(); ++flat_index) {
        const int first_photon = static_cast<int>((flat_index / first_stride) % static_cast<size_t>(cutoff));
        const int second_photon = static_cast<int>((flat_index / second_stride) % static_cast<size_t>(cutoff));
        const std::complex<double> factor = phase_factor(first_photon, second_photon);
        if (std::abs(factor - std::complex<double>(1.0, 0.0)) < kDiagonalCanonicalizationTolerance) {
            continue;
        }
        (*state)[flat_index] *= factor;
    }
}

void apply_exact_diagonal_gate_host(std::vector<std::complex<double>>* state,
                                    const GateParams& gate,
                                    int cutoff,
                                    int num_qumodes) {
    switch (gate.type) {
        case GateType::SNAP_GATE: {
            const double theta = gate.params[0].real();
            const int target_fock_state = static_cast<int>(std::llround(gate.params[1].real()));
            apply_single_mode_diagonal_exact(
                state,
                cutoff,
                num_qumodes,
                gate.target_qumodes[0],
                [theta, target_fock_state](int photon) {
                    if (photon != target_fock_state) {
                        return std::complex<double>(1.0, 0.0);
                    }
                    return std::exp(std::complex<double>(0.0, theta));
                });
            break;
        }
        case GateType::MULTI_SNAP_GATE: {
            std::vector<double> phase_map;
            phase_map.reserve(gate.params.size());
            for (const auto& phase : gate.params) {
                phase_map.push_back(phase.real());
            }
            apply_single_mode_diagonal_exact(
                state,
                cutoff,
                num_qumodes,
                gate.target_qumodes[0],
                [phase_map = std::move(phase_map)](int photon) {
                    if (photon >= static_cast<int>(phase_map.size()) ||
                        !is_nontrivial_phase(phase_map[static_cast<size_t>(photon)])) {
                        return std::complex<double>(1.0, 0.0);
                    }
                    return std::exp(std::complex<double>(0.0, phase_map[static_cast<size_t>(photon)]));
                });
            break;
        }
        case GateType::CROSS_KERR_GATE: {
            const double kappa = gate.params[0].real();
            apply_two_mode_diagonal_exact(
                state,
                cutoff,
                num_qumodes,
                gate.target_qumodes[0],
                gate.target_qumodes[1],
                [kappa](int first_photon, int second_photon) {
                    return std::exp(std::complex<double>(
                        0.0, kappa * static_cast<double>(first_photon * second_photon)));
                });
            break;
        }
        default:
            throw std::invalid_argument("unsupported host exact diagonal gate");
    }
}

bool is_pure_cv_diagonal_gate(const GateParams& gate) {
    if (!gate.target_qubits.empty()) {
        return false;
    }

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
            return gate.target_qumodes.size() == 1 && !gate.params.empty();
        case GateType::SNAP_GATE:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 2;
        case GateType::MULTI_SNAP_GATE:
            return gate.target_qumodes.size() == 1;
        case GateType::CROSS_KERR_GATE:
            return gate.target_qumodes.size() == 2 &&
                   gate.target_qumodes[0] != gate.target_qumodes[1] &&
                   !gate.params.empty();
        default:
            return false;
    }
}

bool is_pure_cv_diagonal_non_gaussian_gate(const GateParams& gate) {
    if (!gate.target_qubits.empty()) {
        return false;
    }

    switch (gate.type) {
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
            return gate.target_qumodes.size() == 1 && !gate.params.empty();
        case GateType::SNAP_GATE:
            return gate.target_qumodes.size() == 1 && gate.params.size() >= 2;
        case GateType::MULTI_SNAP_GATE:
            return gate.target_qumodes.size() == 1;
        case GateType::CROSS_KERR_GATE:
            return gate.target_qumodes.size() == 2 &&
                   gate.target_qumodes[0] != gate.target_qumodes[1] &&
                   !gate.params.empty();
        default:
            return false;
    }
}

GateParams make_single_mode_gate(GateType type, int target_qumode, std::complex<double> param) {
    return GateParams(type, {}, {target_qumode}, {param});
}

int choose_kerr_mixture_branch_cap(int cutoff) {
    if (cutoff <= 0) {
        return 0;
    }
    return std::min(cutoff, kMaxKerrMixtureBranches);
}

int required_exact_branch_count_for_gate(const GateParams& gate, int cutoff) {
    switch (gate.type) {
        case GateType::KERR_GATE:
        case GateType::SNAP_GATE:
        case GateType::MULTI_SNAP_GATE:
            return cutoff;
        case GateType::CONDITIONAL_PARITY:
            return std::min(cutoff, 2);
        case GateType::CROSS_KERR_GATE:
            return cutoff * cutoff;
        default:
            return 0;
    }
}

bool compile_diagonal_gate_mixture_approximation(
    const GateParams& gate,
    int total_qumodes,
    int cutoff,
    double target_fidelity,
    GaussianMixtureApproximation* approximation,
    std::string* error_message) {
    try {
        int max_branch_cap = 0;
        switch (gate.type) {
            case GateType::KERR_GATE:
                max_branch_cap = choose_kerr_mixture_branch_cap(cutoff);
                break;
            case GateType::SNAP_GATE:
            case GateType::MULTI_SNAP_GATE:
                max_branch_cap = choose_snap_mixture_branch_cap(gate, cutoff);
                break;
            case GateType::CONDITIONAL_PARITY:
                max_branch_cap = std::min(cutoff, 2);
                break;
            case GateType::CROSS_KERR_GATE:
                max_branch_cap = choose_cross_kerr_mixture_branch_cap(cutoff);
                break;
            default:
                if (error_message) {
                    *error_message = "未实现该对角非高斯门的Gaussian Mixture编译";
                }
                return false;
        }

        if (max_branch_cap <= 0) {
            if (error_message) {
                *error_message = "当前门的Gaussian Mixture分支上限为0";
            }
            return false;
        }

        const int required_branch_count = required_exact_branch_count_for_gate(gate, cutoff);
        if (required_branch_count <= 0) {
            if (error_message) {
                *error_message = "无法为该门推导保证目标精度的先验K";
            }
            return false;
        }

        if (required_branch_count > max_branch_cap) {
            if (error_message) {
                *error_message =
                    "达到fidelity>=" + std::to_string(target_fidelity) +
                    " 需要先验K=" + std::to_string(required_branch_count) +
                    "，超过上限 " + std::to_string(max_branch_cap) +
                    "，回退到精确Fock执行";
            }
            return false;
        }

        auto build_approximation = [&](int branch_cap) -> GaussianMixtureApproximation {
            switch (gate.type) {
                case GateType::KERR_GATE:
                    return GaussianMixtureDecomposer::approximate_kerr_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        gate.params[0].real(),
                        cutoff,
                        branch_cap);
                case GateType::SNAP_GATE: {
                    const int target_fock_state =
                        static_cast<int>(std::llround(gate.params[1].real()));
                    return GaussianMixtureDecomposer::approximate_snap_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        gate.params[0].real(),
                        target_fock_state,
                        cutoff,
                        branch_cap);
                }
                case GateType::MULTI_SNAP_GATE: {
                    std::vector<double> phase_map;
                    phase_map.reserve(gate.params.size());
                    for (const auto& phase : gate.params) {
                        phase_map.push_back(phase.real());
                    }
                    return GaussianMixtureDecomposer::approximate_multisnap_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        phase_map,
                        cutoff,
                        branch_cap);
                }
                case GateType::CONDITIONAL_PARITY:
                    return GaussianMixtureDecomposer::approximate_conditional_parity_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        gate.params[0].real(),
                        cutoff,
                        branch_cap);
                case GateType::CROSS_KERR_GATE:
                    return GaussianMixtureDecomposer::approximate_cross_kerr_gate(
                        total_qumodes,
                        gate.target_qumodes[0],
                        gate.target_qumodes[1],
                        gate.params[0].real(),
                        cutoff,
                        branch_cap);
                default:
                    throw std::invalid_argument("unsupported diagonal non-Gaussian gate");
            }
        };

        *approximation = build_approximation(required_branch_count);
        return true;
    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = e.what();
        }
        return false;
    }
}

std::vector<GateParams> canonicalize_pure_cv_diagonal_window(const std::vector<GateParams>& window) {
    std::map<int, double> phase_rotations;
    std::map<int, double> kerr_gates;
    std::map<int, double> conditional_parity_gates;
    std::map<int, std::vector<double>> multisnap_gates;
    std::map<std::pair<int, int>, double> cross_kerr_gates;

    for (const GateParams& gate : window) {
        switch (gate.type) {
            case GateType::PHASE_ROTATION:
                phase_rotations[gate.target_qumodes[0]] += gate.params[0].real();
                break;
            case GateType::KERR_GATE:
                kerr_gates[gate.target_qumodes[0]] += gate.params[0].real();
                break;
            case GateType::CONDITIONAL_PARITY:
                conditional_parity_gates[gate.target_qumodes[0]] += gate.params[0].real();
                break;
            case GateType::SNAP_GATE: {
                const double theta = gate.params[0].real();
                if (!is_nontrivial_phase(theta)) {
                    break;
                }
                const int target_qumode = gate.target_qumodes[0];
                const int target_fock_state =
                    static_cast<int>(std::llround(gate.params[1].real()));
                auto& phase_map = multisnap_gates[target_qumode];
                if (target_fock_state >= 0) {
                    if (phase_map.size() <= static_cast<size_t>(target_fock_state)) {
                        phase_map.resize(static_cast<size_t>(target_fock_state) + 1, 0.0);
                    }
                    phase_map[static_cast<size_t>(target_fock_state)] += theta;
                }
                break;
            }
            case GateType::MULTI_SNAP_GATE: {
                const int target_qumode = gate.target_qumodes[0];
                auto& phase_map = multisnap_gates[target_qumode];
                if (phase_map.size() < gate.params.size()) {
                    phase_map.resize(gate.params.size(), 0.0);
                }
                for (size_t idx = 0; idx < gate.params.size(); ++idx) {
                    phase_map[idx] += gate.params[idx].real();
                }
                break;
            }
            case GateType::CROSS_KERR_GATE: {
                int first_target_qumode = gate.target_qumodes[0];
                int second_target_qumode = gate.target_qumodes[1];
                if (first_target_qumode > second_target_qumode) {
                    std::swap(first_target_qumode, second_target_qumode);
                }
                cross_kerr_gates[{first_target_qumode, second_target_qumode}] += gate.params[0].real();
                break;
            }
            default:
                throw std::invalid_argument("non-diagonal gate encountered in diagonal canonicalization window");
        }
    }

    std::vector<GateParams> canonicalized;
    canonicalized.reserve(
        phase_rotations.size() +
        kerr_gates.size() +
        conditional_parity_gates.size() +
        multisnap_gates.size() +
        cross_kerr_gates.size());

    for (const auto& [target_qumode, theta] : phase_rotations) {
        if (is_nontrivial_phase(theta)) {
            canonicalized.push_back(make_single_mode_gate(
                GateType::PHASE_ROTATION, target_qumode, std::complex<double>(theta, 0.0)));
        }
    }
    for (const auto& [target_qumode, chi] : kerr_gates) {
        if (is_nontrivial_phase(chi)) {
            canonicalized.push_back(make_single_mode_gate(
                GateType::KERR_GATE, target_qumode, std::complex<double>(chi, 0.0)));
        }
    }
    for (const auto& [target_qumode, parity] : conditional_parity_gates) {
        if (is_nontrivial_phase(parity)) {
            canonicalized.push_back(make_single_mode_gate(
                GateType::CONDITIONAL_PARITY, target_qumode, std::complex<double>(parity, 0.0)));
        }
    }
    for (auto& [target_qumode, phase_map] : multisnap_gates) {
        trim_trailing_zero_phases(&phase_map);
        if (phase_map.empty()) {
            continue;
        }
        std::vector<std::complex<double>> params;
        params.reserve(phase_map.size());
        for (double phase : phase_map) {
            params.emplace_back(phase, 0.0);
        }
        canonicalized.emplace_back(
            GateType::MULTI_SNAP_GATE,
            std::vector<int>{},
            std::vector<int>{target_qumode},
            params);
    }
    for (const auto& [targets, kappa] : cross_kerr_gates) {
        if (is_nontrivial_phase(kappa)) {
            canonicalized.push_back(make_two_mode_gate(
                GateType::CROSS_KERR_GATE,
                targets.first,
                targets.second,
                std::complex<double>(kappa, 0.0)));
        }
    }

    return canonicalized;
}

SymplecticGate embed_single_mode_gate(const SymplecticGate& local_gate,
                                      int total_qumodes,
                                      int target_qumode) {
    if (local_gate.num_qumodes != 1) {
        throw std::invalid_argument("expected a single-mode symplectic gate");
    }
    if (target_qumode < 0 || target_qumode >= total_qumodes) {
        throw std::out_of_range("target qumode out of range for symplectic embedding");
    }

    SymplecticGate embedded(total_qumodes);
    const int dim = 2 * total_qumodes;
    const int target_row = 2 * target_qumode;

    embedded.S[target_row * dim + target_row] = local_gate.S[0];
    embedded.S[target_row * dim + target_row + 1] = local_gate.S[1];
    embedded.S[(target_row + 1) * dim + target_row] = local_gate.S[2];
    embedded.S[(target_row + 1) * dim + target_row + 1] = local_gate.S[3];

    embedded.d[target_row] = local_gate.d[0];
    embedded.d[target_row + 1] = local_gate.d[1];
    return embedded;
}

SymplecticGate gate_to_symplectic(const GateParams& gate, int total_qumodes) {
    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            return embed_single_mode_gate(
                SymplecticFactory::Rotation(gate.params[0].real()),
                total_qumodes,
                gate.target_qumodes[0]);
        case GateType::DISPLACEMENT:
            return embed_single_mode_gate(
                SymplecticFactory::Displacement(gate.params[0]),
                total_qumodes,
                gate.target_qumodes[0]);
        case GateType::SQUEEZING:
            return embed_single_mode_gate(
                SymplecticFactory::Squeezing(std::abs(gate.params[0]), std::arg(gate.params[0])),
                total_qumodes,
                gate.target_qumodes[0]);
        case GateType::BEAM_SPLITTER: {
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return SymplecticFactory::BeamSplitter(
                gate.params[0].real(),
                phi,
                total_qumodes,
                gate.target_qumodes[0],
                gate.target_qumodes[1]);
        }
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
            return SymplecticFactory::TwoModeSqueezing(
                gate.params[0],
                total_qumodes,
                gate.target_qumodes[0],
                gate.target_qumodes[1]);
        case GateType::CONDITIONAL_SUM: {
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return SymplecticFactory::SUM(
                gate.params[0].real(),
                phi,
                total_qumodes,
                gate.target_qumodes[0],
                gate.target_qumodes[1]);
        }
        default:
            throw std::invalid_argument("gate cannot be represented as a symplectic update");
    }
}

void collect_terminal_state_ids_recursive(HDDNode* node,
                                          std::unordered_set<size_t>& visited_nodes,
                                          std::unordered_set<int>& state_ids) {
    if (!node) {
        return;
    }

    if (!visited_nodes.insert(node->get_unique_id()).second) {
        return;
    }

    if (node->is_terminal()) {
        if (node->tensor_id >= 0) {
            state_ids.insert(node->tensor_id);
        }
        return;
    }

    collect_terminal_state_ids_recursive(node->low, visited_nodes, state_ids);
    collect_terminal_state_ids_recursive(node->high, visited_nodes, state_ids);
}

std::vector<int> collect_terminal_state_ids(HDDNode* root) {
    std::unordered_set<size_t> visited_nodes;
    std::unordered_set<int> state_ids;
    collect_terminal_state_ids_recursive(root, visited_nodes, state_ids);

    std::vector<int> ordered_ids(state_ids.begin(), state_ids.end());
    std::sort(ordered_ids.begin(), ordered_ids.end());
    return ordered_ids;
}

size_t count_reachable_hdd_nodes_recursive(HDDNode* node, std::unordered_set<size_t>& visited_nodes) {
    if (!node) {
        return 0;
    }

    if (!visited_nodes.insert(node->get_unique_id()).second) {
        return 0;
    }

    if (node->is_terminal()) {
        return 1;
    }

    return 1 + count_reachable_hdd_nodes_recursive(node->low, visited_nodes) +
           count_reachable_hdd_nodes_recursive(node->high, visited_nodes);
}

size_t count_reachable_hdd_nodes(HDDNode* root) {
    std::unordered_set<size_t> visited_nodes;
    return count_reachable_hdd_nodes_recursive(root, visited_nodes);
}

struct NodeSignKey {
    HDDNode* node = nullptr;
    bool negated = false;

    bool operator==(const NodeSignKey& other) const {
        return node == other.node && negated == other.negated;
    }
};

struct NodeSignKeyHash {
    size_t operator()(const NodeSignKey& key) const {
        size_t seed = std::hash<HDDNode*>()(key.node);
        seed ^= std::hash<bool>()(key.negated) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

std::complex<double> compute_qumode_overlap(
    const std::vector<cuDoubleComplex>& state_data,
    int d_trunc,
    int num_qumodes,
    const std::vector<std::vector<std::complex<double>>>& qumode_states) {
    if (state_data.empty()) {
        return std::complex<double>(0.0, 0.0);
    }

    if (qumode_states.empty()) {
        return std::complex<double>(cuCreal(state_data[0]), cuCimag(state_data[0]));
    }

    std::vector<std::vector<std::complex<double>>> mode_vectors(num_qumodes);
    for (int mode = 0; mode < num_qumodes; ++mode) {
        if (mode < static_cast<int>(qumode_states.size()) && !qumode_states[mode].empty()) {
            if (qumode_states[mode].size() != static_cast<size_t>(d_trunc)) {
                throw std::invalid_argument("Qumode状态向量长度与截断维度不匹配");
            }
            mode_vectors[mode] = qumode_states[mode];
        } else {
            mode_vectors[mode].assign(d_trunc, std::complex<double>(0.0, 0.0));
            mode_vectors[mode][0] = std::complex<double>(1.0, 0.0);
        }
    }

    std::complex<double> overlap(0.0, 0.0);
    for (size_t linear_index = 0; linear_index < state_data.size(); ++linear_index) {
        size_t index = linear_index;
        std::complex<double> basis_coeff(1.0, 0.0);
        for (int mode = num_qumodes - 1; mode >= 0; --mode) {
            const int fock_index = static_cast<int>(index % static_cast<size_t>(d_trunc));
            index /= static_cast<size_t>(d_trunc);
            basis_coeff *= mode_vectors[mode][fock_index];
        }

        const std::complex<double> amplitude(cuCreal(state_data[linear_index]),
                                             cuCimag(state_data[linear_index]));
        overlap += std::conj(basis_coeff) * amplitude;
    }

    return overlap;
}

}  // namespace

/**
 * HDD节点数乘: result = weight * node
 */
HDDNode* QuantumCircuit::scale_hdd_node(HDDNode* node, std::complex<double> weight) {
    if (!node) {
        return nullptr;
    }

    if (std::abs(weight) < 1e-14) {
        if (shared_zero_state_id_ >= 0) {
            return node_manager_.create_terminal_node(shared_zero_state_id_);
        }
        const int zero_id = state_pool_.allocate_state();
        if (zero_id < 0) {
            throw std::runtime_error("HDD缩放失败：无法分配零状态");
        }

        int64_t zero_dim = shared_zero_state_id_ >= 0 ? state_pool_.get_state_dim(shared_zero_state_id_) : 0;
        if (node->is_terminal()) {
            zero_dim = state_pool_.get_state_dim(node->tensor_id);
        }
        if (zero_dim <= 0) {
            zero_dim = state_pool_.get_max_total_dim();
        }

        try {
            state_pool_.reserve_state_storage(zero_id, zero_dim);
            zero_state_device(&state_pool_, zero_id, nullptr, false);
        } catch (...) {
            state_pool_.free_state(zero_id);
            throw;
        }
        return node_manager_.create_terminal_node(zero_id);
    }

    if (std::abs(weight - std::complex<double>(1.0, 0.0)) < 1e-14) {
        return node;
    }

    if (node->is_terminal()) {
        if (node->tensor_id == shared_zero_state_id_) {
            return node_manager_.create_terminal_node(shared_zero_state_id_);
        }
        if (is_symbolic_terminal_id(node->tensor_id)) {
            const SymbolicTerminalState& symbolic_state =
                symbolic_terminal_states_.at(node->tensor_id);
            std::vector<SymbolicGaussianBranch> scaled_branches;
            scaled_branches.reserve(symbolic_state.branches.size());
            for (const SymbolicGaussianBranch& existing_branch : symbolic_state.branches) {
                SymbolicGaussianBranch scaled_branch = existing_branch;
                scaled_branch.gaussian_state_id =
                    duplicate_gaussian_state(existing_branch.gaussian_state_id);
                scaled_branch.weight *= weight;
                scaled_branches.push_back(std::move(scaled_branch));
            }

            if (scaled_branches.empty()) {
                return node_manager_.create_terminal_node(shared_zero_state_id_);
            }

            const int scaled_terminal_id = allocate_symbolic_terminal_id();
            symbolic_terminal_states_.emplace(
                scaled_terminal_id,
                SymbolicTerminalState{std::move(scaled_branches)});
            return node_manager_.create_terminal_node(scaled_terminal_id);
        }

        const int scaled_state_id = state_pool_.allocate_state();
        if (scaled_state_id < 0) {
            throw std::runtime_error("HDD缩放失败：无法分配目标终端状态");
        }

        const cuDoubleComplex weight_cu = make_cuDoubleComplex(weight.real(), weight.imag());
        try {
            copy_scale_state_device(&state_pool_, node->tensor_id, scaled_state_id, weight_cu, nullptr, false);
        } catch (...) {
            state_pool_.free_state(scaled_state_id);
            throw;
        }
        return node_manager_.create_terminal_node(scaled_state_id);
    }

    return node_manager_.get_or_create_node(node->qubit_level,
                                            node->low,
                                            node->high,
                                            weight * node->w_low,
                                            weight * node->w_high);
}

/**
 * 复制并缩放终端节点状态
 */
HDDNode* QuantumCircuit::duplicate_scaled_terminal_node(HDDNode* terminal_node,
                                                        std::complex<double> weight) {
    if (!terminal_node || !terminal_node->is_terminal()) {
        throw std::runtime_error("终端节点复制失败：输入节点不是终端节点");
    }

    const int scaled_state_id = state_pool_.allocate_state();
    if (scaled_state_id < 0) {
        throw std::runtime_error("终端节点复制失败：无法分配状态");
    }

    if (std::abs(weight) < 1e-14) {
        int64_t zero_dim = state_pool_.get_state_dim(scaled_state_id);
        if (zero_dim <= 0) {
            zero_dim = state_pool_.get_state_dim(terminal_node->tensor_id);
        }
        if (zero_dim <= 0) {
            zero_dim = state_pool_.get_max_total_dim();
        }

        try {
            state_pool_.reserve_state_storage(scaled_state_id, zero_dim);
            zero_state_device(&state_pool_, scaled_state_id, nullptr, false);
        } catch (...) {
            state_pool_.free_state(scaled_state_id);
            throw;
        }
        return node_manager_.create_terminal_node(scaled_state_id);
    }

    if (std::abs(weight - std::complex<double>(1.0, 0.0)) < 1e-14) {
        try {
            copy_state_device(&state_pool_, terminal_node->tensor_id, scaled_state_id, nullptr, false);
        } catch (...) {
            state_pool_.free_state(scaled_state_id);
            throw;
        }
        return node_manager_.create_terminal_node(scaled_state_id);
    }

    const cuDoubleComplex weight_cu = make_cuDoubleComplex(weight.real(), weight.imag());
    try {
        copy_scale_state_device(&state_pool_, terminal_node->tensor_id, scaled_state_id, weight_cu, nullptr, false);
    } catch (...) {
        state_pool_.free_state(scaled_state_id);
        throw;
    }
    return node_manager_.create_terminal_node(scaled_state_id);
}

/**
 * HDD节点加法: result = w1 * n1 + w2 * n2
 */
HDDNode* QuantumCircuit::hdd_add(HDDNode* n1, std::complex<double> w1, HDDNode* n2, std::complex<double> w2) {
    std::unordered_map<WeightedNodePairKey, HDDNode*, WeightedNodePairKeyHash> add_memo;
    std::unordered_map<int, int> projected_symbolic_terminal_cache;

    const auto project_symbolic_terminal =
        [&](int terminal_id) -> int {
            const auto cached = projected_symbolic_terminal_cache.find(terminal_id);
            if (cached != projected_symbolic_terminal_cache.end()) {
                return cached->second;
            }

            const int projected_id = project_symbolic_terminal_to_fock_state(terminal_id);
            projected_symbolic_terminal_cache.emplace(terminal_id, projected_id);
            return projected_id;
        };

    const auto free_temporary_symbolic_branches =
        [&](std::vector<SymbolicGaussianBranch>* branches) {
            if (!gaussian_state_pool_) {
                branches->clear();
                return;
            }
            for (const SymbolicGaussianBranch& branch : *branches) {
                if (branch.gaussian_state_id >= 0) {
                    gaussian_state_pool_->free_state(branch.gaussian_state_id);
                }
            }
            branches->clear();
        };

    std::function<HDDNode*(HDDNode*, std::complex<double>, HDDNode*, std::complex<double>)>
        add_recursive =
            [&](HDDNode* lhs,
                std::complex<double> lhs_weight,
                HDDNode* rhs,
                std::complex<double> rhs_weight) -> HDDNode* {
                if (!lhs || !rhs) {
                    return scale_hdd_node(lhs ? lhs : rhs, std::complex<double>(0.0, 0.0));
                }

                const WeightedNodePairKey key{lhs, rhs, lhs_weight, rhs_weight};
                const auto memo_it = add_memo.find(key);
                if (memo_it != add_memo.end()) {
                    return memo_it->second;
                }

                HDDNode* result = nullptr;

                if (std::abs(lhs_weight) < 1e-14) {
                    if (std::abs(rhs_weight) < 1e-14) {
                        result = scale_hdd_node(lhs, std::complex<double>(0.0, 0.0));
                    } else {
                        result = scale_hdd_node(rhs, rhs_weight);
                    }
                    add_memo.emplace(key, result);
                    return result;
                }
                if (std::abs(rhs_weight) < 1e-14) {
                    result = scale_hdd_node(lhs, lhs_weight);
                    add_memo.emplace(key, result);
                    return result;
                }

                if (lhs == rhs) {
                    result = scale_hdd_node(lhs, lhs_weight + rhs_weight);
                    add_memo.emplace(key, result);
                    return result;
                }

                if (lhs->is_terminal() && rhs->is_terminal()) {
                    int id1 = lhs->tensor_id;
                    int id2 = rhs->tensor_id;
                    const bool symbolic1 = is_symbolic_terminal_id(id1);
                    const bool symbolic2 = is_symbolic_terminal_id(id2);
                    const bool zero1 = id1 == shared_zero_state_id_;
                    const bool zero2 = id2 == shared_zero_state_id_;
                    bool force_project_symbolic_sum = false;

                    if ((symbolic1 || zero1) && (symbolic2 || zero2)) {
                        std::vector<SymbolicGaussianBranch> combined_branches;

                        auto append_scaled_branches =
                            [&](int terminal_id, std::complex<double> scale) {
                                if (std::abs(scale) < 1e-14 || !is_symbolic_terminal_id(terminal_id)) {
                                    return;
                                }
                                const SymbolicTerminalState& symbolic_state =
                                    symbolic_terminal_states_.at(terminal_id);
                                for (const SymbolicGaussianBranch& existing_branch : symbolic_state.branches) {
                                    SymbolicGaussianBranch new_branch = existing_branch;
                                    new_branch.gaussian_state_id =
                                        duplicate_gaussian_state(existing_branch.gaussian_state_id);
                                    new_branch.weight *= scale;
                                    combined_branches.push_back(std::move(new_branch));
                                }
                            };

                        append_scaled_branches(id1, lhs_weight);
                        append_scaled_branches(id2, rhs_weight);

                        if (combined_branches.empty()) {
                            result = node_manager_.create_terminal_node(shared_zero_state_id_);
                            add_memo.emplace(key, result);
                            return result;
                        }

                        if (combined_branches.size() <=
                            static_cast<size_t>(symbolic_branch_limit_)) {
                            const int symbolic_terminal_id = allocate_symbolic_terminal_id();
                            symbolic_terminal_states_.emplace(
                                symbolic_terminal_id,
                                SymbolicTerminalState{std::move(combined_branches)});
                            result = node_manager_.create_terminal_node(symbolic_terminal_id);
                            add_memo.emplace(key, result);
                            return result;
                        }

                        FALLBACK_DEBUG_LOG
                            << "[fallback] hdd_add materializing symbolic sum because combined branches would grow to "
                            << combined_branches.size() << std::endl;
                        free_temporary_symbolic_branches(&combined_branches);
                        force_project_symbolic_sum = true;
                    }

                    if (force_project_symbolic_sum) {
                        if (symbolic1) {
                            id1 = project_symbolic_terminal(id1);
                        }
                        if (symbolic2) {
                            id2 = project_symbolic_terminal(id2);
                        }
                    }

                    if (zero1) {
                        if (force_project_symbolic_sum) {
                            result = scale_hdd_node(
                                node_manager_.create_terminal_node(id2), rhs_weight);
                        } else {
                            result = scale_hdd_node(rhs, rhs_weight);
                        }
                        add_memo.emplace(key, result);
                        return result;
                    }
                    if (zero2) {
                        if (force_project_symbolic_sum) {
                            result = scale_hdd_node(
                                node_manager_.create_terminal_node(id1), lhs_weight);
                        } else {
                            result = scale_hdd_node(lhs, lhs_weight);
                        }
                        add_memo.emplace(key, result);
                        return result;
                    }

                    if (symbolic1 && !force_project_symbolic_sum) {
                        id1 = project_symbolic_terminal(id1);
                    }
                    if (symbolic2 && !force_project_symbolic_sum) {
                        id2 = project_symbolic_terminal(id2);
                    }

                    if (id1 == id2) {
                        result = scale_hdd_node(
                            node_manager_.create_terminal_node(id1), lhs_weight + rhs_weight);
                        add_memo.emplace(key, result);
                        return result;
                    }
                    if (id1 == shared_zero_state_id_) {
                        result = scale_hdd_node(
                            node_manager_.create_terminal_node(id2), rhs_weight);
                        add_memo.emplace(key, result);
                        return result;
                    }
                    if (id2 == shared_zero_state_id_) {
                        result = scale_hdd_node(
                            node_manager_.create_terminal_node(id1), lhs_weight);
                        add_memo.emplace(key, result);
                        return result;
                    }

                    const int new_id = state_pool_.allocate_state();

                    auto transfer_start = std::chrono::high_resolution_clock::now();
                    const cuDoubleComplex w1_cu =
                        make_cuDoubleComplex(lhs_weight.real(), lhs_weight.imag());
                    const cuDoubleComplex w2_cu =
                        make_cuDoubleComplex(rhs_weight.real(), rhs_weight.imag());
                    auto transfer_end = std::chrono::high_resolution_clock::now();
                    transfer_time_ += std::chrono::duration<double, std::milli>(
                        transfer_end - transfer_start).count();

                    auto compute_start = std::chrono::high_resolution_clock::now();
                    combine_states_device(&state_pool_, id1, w1_cu, id2, w2_cu, new_id, nullptr, false);
                    auto compute_end = std::chrono::high_resolution_clock::now();
                    computation_time_ += std::chrono::duration<double, std::milli>(
                        compute_end - compute_start).count();

                    result = node_manager_.create_terminal_node(new_id);
                    add_memo.emplace(key, result);
                    return result;
                }

                const int level1 = lhs->is_terminal() ? -1 : lhs->qubit_level;
                const int level2 = rhs->is_terminal() ? -1 : rhs->qubit_level;

                if (level1 != level2) {
                    if (level1 > level2) {
                        result = node_manager_.get_or_create_node(
                            level1,
                            add_recursive(lhs->low, lhs_weight * lhs->w_low, rhs, rhs_weight),
                            add_recursive(lhs->high, lhs_weight * lhs->w_high, rhs, rhs_weight),
                            1.0,
                            1.0);
                    } else {
                        result = node_manager_.get_or_create_node(
                            level2,
                            add_recursive(lhs, lhs_weight, rhs->low, rhs_weight * rhs->w_low),
                            add_recursive(lhs, lhs_weight, rhs->high, rhs_weight * rhs->w_high),
                            1.0,
                            1.0);
                    }
                    add_memo.emplace(key, result);
                    return result;
                }

                result = node_manager_.get_or_create_node(
                    level1,
                    add_recursive(lhs->low, lhs_weight * lhs->w_low, rhs->low, rhs_weight * rhs->w_low),
                    add_recursive(lhs->high, lhs_weight * lhs->w_high, rhs->high, rhs_weight * rhs->w_high),
                    1.0,
                    1.0);
                add_memo.emplace(key, result);
                return result;
            };

    return add_recursive(n1, w1, n2, w2);
}

HDDNode* QuantumCircuit::apply_single_qubit_gate_recursive(HDDNode* node, int target_qubit, const std::vector<std::complex<double>>& u) {
    if (node->is_terminal()) {
        return node;
    }

    if (node->qubit_level == target_qubit) {
        // 应用矩阵
        // u: [u00, u01, u10, u11]
        HDDNode* L = node->low;
        HDDNode* H = node->high;
        
        std::complex<double> wL = node->w_low;
        std::complex<double> wH = node->w_high;
        
        // New L = u00 * (wL * L) + u01 * (wH * H)
        HDDNode* new_L = hdd_add(L, u[0] * wL, H, u[1] * wH);
        
        // New H = u10 * (wL * L) + u11 * (wH * H)
        HDDNode* new_H = hdd_add(L, u[2] * wL, H, u[3] * wH);
        
        return node_manager_.get_or_create_node(node->qubit_level, new_L, new_H, 1.0, 1.0);
    } else if (node->qubit_level > target_qubit) {
        HDDNode* new_low = apply_single_qubit_gate_recursive(node->low, target_qubit, u);
        HDDNode* new_high = apply_single_qubit_gate_recursive(node->high, target_qubit, u);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    } else {
        return node;
    }
}

HDDNode* QuantumCircuit::apply_cnot_recursive(HDDNode* node, int control, int target) {
    if (node->is_terminal()) return node;
    
    if (node->qubit_level == control) {
        // 控制节点
        HDDNode* new_low = node->low; // 恒等
        
        // 高分支：对目标应用X
        std::vector<std::complex<double>> px = {0.0, 1.0, 1.0, 0.0};
        HDDNode* new_high = apply_single_qubit_gate_recursive(node->high, target, px);
        
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    } else if (node->qubit_level > control) {
        HDDNode* new_low = apply_cnot_recursive(node->low, control, target);
        HDDNode* new_high = apply_cnot_recursive(node->high, control, target);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    }
    
    return node;
}

/**
 * 递归应用CZ门
 */
HDDNode* QuantumCircuit::apply_cz_recursive(HDDNode* node, int control, int target) {
    if (node->is_terminal()) return node;

    if (node->qubit_level == control) {
        // 控制节点
        HDDNode* new_low = node->low; // 恒等

        // 高分支：对目标应用Z
        std::vector<std::complex<double>> pz = {1.0, 0.0, 0.0, -1.0};
        HDDNode* new_high = apply_single_qubit_gate_recursive(node->high, target, pz);

        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    } else if (node->qubit_level > control) {
        HDDNode* new_low = apply_cz_recursive(node->low, control, target);
        HDDNode* new_high = apply_cz_recursive(node->high, control, target);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    }

    return node;
}

/**
 * QuantumCircuit 构造函数
 */
QuantumCircuit::QuantumCircuit(int num_qubits, int num_qumodes, int cv_truncation, int max_states)
    : num_qubits_(num_qubits), num_qumodes_(num_qumodes), cv_truncation_(cv_truncation),
      root_node_(nullptr), state_pool_(cv_truncation, max_states, num_qumodes),
      gaussian_state_pool_(nullptr),
      is_built_(false), is_executed_(false), shared_zero_state_id_(-1),
      total_time_(0.0), transfer_time_(0.0), computation_time_(0.0), planning_time_(0.0),
      gaussian_symbolic_mode_limit_(4), symbolic_branch_limit_(kDefaultSymbolicBranchLimit),
      gaussian_state_pool_capacity_override_(0),
      next_symbolic_terminal_id_(-2),
      pending_gc_replacements_(0) {

    if (num_qubits < 0 || num_qumodes <= 0 || cv_truncation <= 0) {
        throw std::invalid_argument("Qubit数量不能为负数，Qumode数量和截断维度必须为正数");
    }

    std::cout << "创建量子电路: " << num_qubits << " qubits, "
              << num_qumodes << " qumodes, 截断维度=" << cv_truncation << std::endl;
}

/**
 * QuantumCircuit 析构函数
 */
QuantumCircuit::~QuantumCircuit() {
    reset();
}

bool QuantumCircuit::is_symbolic_terminal_id(int terminal_id) const {
    return symbolic_terminal_states_.find(terminal_id) != symbolic_terminal_states_.end();
}

std::vector<int> QuantumCircuit::collect_symbolic_terminal_ids(HDDNode* root) const {
    std::unordered_set<size_t> visited_nodes;
    std::unordered_set<int> symbolic_ids;
    collect_symbolic_terminal_ids_recursive(root, visited_nodes, symbolic_ids);
    std::vector<int> ordered_ids(symbolic_ids.begin(), symbolic_ids.end());
    std::sort(ordered_ids.begin(), ordered_ids.end());
    return ordered_ids;
}

bool QuantumCircuit::has_symbolic_terminals() const {
    return !get_cached_symbolic_terminal_ids().empty();
}

void QuantumCircuit::ensure_gaussian_state_pool() {
    if (gaussian_state_pool_) {
        return;
    }
    const int capacity = gaussian_state_pool_capacity_override_ > 0
        ? gaussian_state_pool_capacity_override_
        : std::max(4096, state_pool_.capacity * 16);
    gaussian_state_pool_ = std::make_unique<GaussianStatePool>(num_qumodes_, capacity);
}

int QuantumCircuit::allocate_symbolic_terminal_id() {
    return next_symbolic_terminal_id_--;
}

void QuantumCircuit::initialize_gaussian_vacuum_state(int gaussian_state_id) {
    ensure_gaussian_state_pool();

    const int dim = 2 * num_qumodes_;
    std::vector<double> d(static_cast<size_t>(dim), 0.0);
    std::vector<double> sigma(static_cast<size_t>(dim) * dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        sigma[static_cast<size_t>(i) * dim + i] = 0.5;
    }
    gaussian_state_pool_->upload_state(gaussian_state_id, d, sigma);
}

int QuantumCircuit::duplicate_gaussian_state(int gaussian_state_id) {
    ensure_gaussian_state_pool();

    const int duplicated_state_id = gaussian_state_pool_->allocate_state();
    if (duplicated_state_id < 0) {
        throw std::runtime_error("Gaussian状态池已满，无法复制symbolic branch");
    }

    const int dim = 2 * num_qumodes_;
    const size_t d_bytes = static_cast<size_t>(dim) * sizeof(double);
    const size_t sigma_bytes = static_cast<size_t>(dim) * dim * sizeof(double);

    CHECK_CUDA(cudaMemcpy(
        gaussian_state_pool_->get_displacement_ptr(duplicated_state_id),
        gaussian_state_pool_->get_displacement_ptr(gaussian_state_id),
        d_bytes,
        cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(
        gaussian_state_pool_->get_covariance_ptr(duplicated_state_id),
        gaussian_state_pool_->get_covariance_ptr(gaussian_state_id),
        sigma_bytes,
        cudaMemcpyDeviceToDevice));

    return duplicated_state_id;
}

void QuantumCircuit::apply_symplectic_update_to_gaussian_states(
    const std::vector<int>& gaussian_state_ids,
    const SymplecticGate& gate) {
    if (gaussian_state_ids.empty()) {
        return;
    }

    ensure_gaussian_state_pool();

    size_t offset = 0;
    offset = align_scratch_offset(offset, alignof(int));
    const size_t state_ids_offset = offset;
    offset += gaussian_state_ids.size() * sizeof(int);

    offset = align_scratch_offset(offset, alignof(double));
    const size_t s_offset = offset;
    offset += gate.S.size() * sizeof(double);

    offset = align_scratch_offset(offset, alignof(double));
    const size_t dg_offset = offset;
    offset += gate.d.size() * sizeof(double);

    const int dim = 2 * num_qumodes_;
    offset = align_scratch_offset(offset, alignof(double));
    const size_t old_offset = offset;
    offset += gaussian_state_ids.size() * static_cast<size_t>(dim) * sizeof(double);

    offset = align_scratch_offset(offset, alignof(double));
    const size_t temp_offset = offset;
    offset += gaussian_state_ids.size() * static_cast<size_t>(dim) * dim * sizeof(double);

    char* scratch = static_cast<char*>(state_pool_.scratch_aux.ensure(offset));
    int* d_state_ids = reinterpret_cast<int*>(scratch + state_ids_offset);
    double* d_S = reinterpret_cast<double*>(scratch + s_offset);
    double* d_dg = reinterpret_cast<double*>(scratch + dg_offset);
    double* d_old = reinterpret_cast<double*>(scratch + old_offset);
    double* d_temp = reinterpret_cast<double*>(scratch + temp_offset);

    char* staged = static_cast<char*>(state_pool_.host_transfer_staging.ensure(dg_offset + gate.d.size() * sizeof(double)));
    std::memcpy(staged + state_ids_offset,
                gaussian_state_ids.data(),
                gaussian_state_ids.size() * sizeof(int));
    std::memcpy(staged + s_offset,
                gate.S.data(),
                gate.S.size() * sizeof(double));
    std::memcpy(staged + dg_offset,
                gate.d.data(),
                gate.d.size() * sizeof(double));
    CHECK_CUDA(cudaMemcpy(scratch,
                          staged,
                          dg_offset + gate.d.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    apply_batched_symplectic_update(
        gaussian_state_pool_.get(),
        d_state_ids,
        static_cast<int>(gaussian_state_ids.size()),
        d_S,
        d_dg,
        d_old,
        d_temp,
        nullptr,
        true);
}

void QuantumCircuit::release_symbolic_terminal(int terminal_id) {
    const auto it = symbolic_terminal_states_.find(terminal_id);
    if (it == symbolic_terminal_states_.end()) {
        return;
    }
    if (gaussian_state_pool_) {
        for (const SymbolicGaussianBranch& branch : it->second.branches) {
            if (branch.gaussian_state_id >= 0) {
                gaussian_state_pool_->free_state(branch.gaussian_state_id);
            }
        }
    }
    symbolic_terminal_states_.erase(it);
}

void QuantumCircuit::clear_symbolic_terminals() {
    std::vector<int> symbolic_ids;
    symbolic_ids.reserve(symbolic_terminal_states_.size());
    for (const auto& entry : symbolic_terminal_states_) {
        symbolic_ids.push_back(entry.first);
    }
    for (int terminal_id : symbolic_ids) {
        release_symbolic_terminal(terminal_id);
    }
    gaussian_state_pool_.reset();
    next_symbolic_terminal_id_ = -2;
}

/**
 * 添加门操作到线路
 */
void QuantumCircuit::add_gate(const GateParams& gate) {
    if (is_built_) {
        throw std::runtime_error("不能在构建后添加门操作");
    }
    gate_sequence_.push_back(gate);
}

/**
 * 批量添加门操作
 */
void QuantumCircuit::add_gates(const std::vector<GateParams>& gates) {
    if (is_built_) {
        throw std::runtime_error("不能在构建后添加门操作");
    }
    gate_sequence_.insert(gate_sequence_.end(), gates.begin(), gates.end());
}

/**
 * 构建量子线路
 */
void QuantumCircuit::build() {
    if (is_built_) return;

    std::cout << "构建量子线路..." << std::endl;

    async_cv_pipeline_enabled_ = !async_cv_pipeline_disabled();
    if (async_cv_pipeline_enabled_) {
        prewarm_async_target_upload_slots();
    } else {
        std::cout << "异步CV流水线已通过环境变量禁用" << std::endl;
    }

    const size_t target_id_bytes = static_cast<size_t>(state_pool_.capacity) * sizeof(int);
    if (target_id_bytes > 0) {
        state_pool_.scratch_target_ids.ensure(target_id_bytes);
        state_pool_.host_transfer_staging.ensure(target_id_bytes);
    }

    // 初始化HDD结构
    initialize_hdd();

    is_built_ = true;
    std::cout << "量子线路构建完成" << std::endl;
}

std::vector<GateParams> QuantumCircuit::canonicalize_gate_sequence_for_execution() const {
    std::vector<GateParams> canonicalized;
    canonicalized.reserve(gate_sequence_.size());

    size_t gate_index = 0;
    while (gate_index < gate_sequence_.size()) {
        if (!is_pure_cv_diagonal_gate(gate_sequence_[gate_index])) {
            canonicalized.push_back(gate_sequence_[gate_index]);
            ++gate_index;
            continue;
        }

        std::vector<GateParams> diagonal_window;
        while (gate_index < gate_sequence_.size() &&
               is_pure_cv_diagonal_gate(gate_sequence_[gate_index])) {
            diagonal_window.push_back(gate_sequence_[gate_index]);
            ++gate_index;
        }

        const std::vector<GateParams> window_canonicalized =
            canonicalize_pure_cv_diagonal_window(diagonal_window);
        canonicalized.insert(canonicalized.end(),
                             window_canonicalized.begin(),
                             window_canonicalized.end());
    }

    return canonicalized;
}

std::vector<QuantumCircuit::ExecutionBlock> QuantumCircuit::partition_execution_blocks(
    const std::vector<GateParams>& execution_sequence) const {
    std::vector<ExecutionBlock> blocks;
    size_t gate_index = 0;
    while (gate_index < execution_sequence.size()) {
        ExecutionBlockKind kind = ExecutionBlockKind::Other;
        if (is_pure_qubit_gate(execution_sequence[gate_index])) {
            kind = ExecutionBlockKind::QubitOnly;
        } else if (is_gaussian_track_gate(execution_sequence[gate_index])) {
            kind = ExecutionBlockKind::Gaussian;
        } else if (is_pure_cv_diagonal_non_gaussian_gate(execution_sequence[gate_index])) {
            kind = ExecutionBlockKind::DiagonalNonGaussian;
        }

        size_t block_end = gate_index + 1;
        while (block_end < execution_sequence.size()) {
            ExecutionBlockKind next_kind = ExecutionBlockKind::Other;
            if (is_pure_qubit_gate(execution_sequence[block_end])) {
                next_kind = ExecutionBlockKind::QubitOnly;
            } else if (is_gaussian_track_gate(execution_sequence[block_end])) {
                next_kind = ExecutionBlockKind::Gaussian;
            } else if (is_pure_cv_diagonal_non_gaussian_gate(execution_sequence[block_end])) {
                next_kind = ExecutionBlockKind::DiagonalNonGaussian;
            }

            if (next_kind != kind) {
                break;
            }
            ++block_end;
        }

        blocks.push_back({kind, gate_index, block_end});
        gate_index = block_end;
    }

    return blocks;
}

QuantumCircuit::CompiledExecutionBlock QuantumCircuit::compile_execution_block(
    const std::vector<GateParams>& execution_sequence,
    const std::vector<ExecutionBlock>& execution_blocks,
    size_t block_index) const {
    if (block_index >= execution_blocks.size()) {
        throw std::out_of_range("execution block index out of range");
    }

    const auto compile_start = std::chrono::high_resolution_clock::now();

    const ExecutionBlock& block = execution_blocks[block_index];
    CompiledExecutionBlock compiled_block;
    compiled_block.kind = block.kind;
    compiled_block.begin = block.begin;
    compiled_block.end = block.end;
    compiled_block.gates.assign(
        execution_sequence.begin() + static_cast<std::ptrdiff_t>(block.begin),
        execution_sequence.begin() + static_cast<std::ptrdiff_t>(block.end));

    for (size_t next_block_index = block_index + 1;
         next_block_index < execution_blocks.size();
         ++next_block_index) {
        const ExecutionBlockKind next_kind = execution_blocks[next_block_index].kind;
        if (next_kind != ExecutionBlockKind::Gaussian &&
            next_kind != ExecutionBlockKind::QubitOnly) {
            compiled_block.downstream_non_gaussianity = 1.0;
            break;
        }
    }

    if (block.kind == ExecutionBlockKind::Gaussian) {
        try {
            compiled_block.gaussian_updates.reserve(compiled_block.gates.size());
            for (const GateParams& gate : compiled_block.gates) {
                std::string gaussian_error;
                if (!validate_gaussian_track_gate(
                        gate, num_qubits_, num_qumodes_, &gaussian_error)) {
                    throw std::invalid_argument(gaussian_error);
                }
                if (is_unconditional_gaussian_gate(gate)) {
                    compiled_block.gaussian_updates.push_back(gate_to_symplectic(gate, num_qumodes_));
                }
            }
            compiled_block.gaussian_ready = true;
        } catch (const std::exception& e) {
            compiled_block.gaussian_ready = false;
            compiled_block.compile_error = e.what();
            compiled_block.gaussian_updates.clear();
        }
    } else if (block.kind == ExecutionBlockKind::DiagonalNonGaussian) {
        compiled_block.diagonal_mixture_updates.reserve(compiled_block.gates.size());
        compiled_block.diagonal_mixture_ready = true;
        const double per_gate_fidelity_target =
            std::pow(kTargetMixtureFidelity,
                     1.0 / static_cast<double>(std::max<size_t>(1, compiled_block.gates.size())));

        for (const GateParams& gate : compiled_block.gates) {
            GaussianMixtureApproximation approximation;
            std::string mixture_error;
            if (!compile_diagonal_gate_mixture_approximation(
                    gate,
                    num_qumodes_,
                    cv_truncation_,
                    per_gate_fidelity_target,
                    &approximation,
                    &mixture_error)) {
                compiled_block.diagonal_mixture_ready = false;
                compiled_block.compile_error = mixture_error;
                compiled_block.diagonal_mixture_updates.clear();
                compiled_block.estimated_diagonal_l2_error = 0.0;
                compiled_block.diagonal_fidelity_lower_bound = 0.0;
                compiled_block.mixture_branch_count = 0;
                break;
            }

            compiled_block.estimated_diagonal_l2_error += approximation.l2_diagonal_error;
            compiled_block.diagonal_fidelity_lower_bound *=
                approximation.conservative_fidelity_lower_bound;
            compiled_block.mixture_branch_count += approximation.branches.size();
            compiled_block.diagonal_mixture_updates.push_back(std::move(approximation));
        }
    }

    const auto compile_end = std::chrono::high_resolution_clock::now();
    compiled_block.compile_time_ms =
        std::chrono::duration<double, std::milli>(compile_end - compile_start).count();
    return compiled_block;
}

bool QuantumCircuit::try_execute_gaussian_block_with_ede(
    const CompiledExecutionBlock& compiled_block) {
    ScopedNvtxRange nvtx_range("qc::gaussian_block_ede");
    if (compiled_block.kind != ExecutionBlockKind::Gaussian ||
        compiled_block.gates.empty() ||
        !root_node_) {
        return false;
    }

    if (!compiled_block.gaussian_ready) {
        if (!compiled_block.compile_error.empty()) {
            std::cout << "Gaussian块预编译失败，回退到全量Fock执行: "
                      << compiled_block.compile_error << std::endl;
        }
        return false;
    }

    try {
        ensure_gaussian_state_pool();
        const std::vector<int> control_qubits =
            collect_gaussian_control_qubits(compiled_block.gates);
        std::vector<int> qubit_assignment(static_cast<size_t>(num_qubits_), 0);
        std::unordered_map<int, VacuumRayInfo> terminal_state_cache;
        std::unordered_map<std::string, std::vector<GateParams>> resolved_gate_cache;
        std::unordered_map<std::string, HDDNode*> transformed_terminal_cache;

        auto resolved_gates_for_assignment =
            [&](const std::string& assignment_key) -> const std::vector<GateParams>& {
                auto it = resolved_gate_cache.find(assignment_key);
                if (it != resolved_gate_cache.end()) {
                    return it->second;
                }

                std::vector<GateParams> resolved_gates;
                const std::vector<int> branch_assignment =
                    assignment_from_key(control_qubits, assignment_key, num_qubits_);
                for (const GateParams& gate : compiled_block.gates) {
                    const std::optional<GateParams> resolved_gate =
                        resolve_gaussian_gate_for_assignment(gate, branch_assignment);
                    if (resolved_gate.has_value()) {
                        resolved_gates.push_back(*resolved_gate);
                    }
                }
                return resolved_gate_cache.emplace(assignment_key, std::move(resolved_gates)).first->second;
            };

        std::function<HDDNode*(HDDNode*)> transform_gaussian_block_recursive =
            [&](HDDNode* node) -> HDDNode* {
                if (!node) {
                    return nullptr;
                }
                if (node->is_terminal()) {
                    if (node->tensor_id == shared_zero_state_id_) {
                        return node_manager_.create_terminal_node(shared_zero_state_id_);
                    }

                    const std::string assignment_key =
                        make_control_assignment_key(control_qubits, qubit_assignment);
                    const std::string cache_key =
                        std::to_string(node->tensor_id) + "|" + assignment_key;
                    const auto transformed_it = transformed_terminal_cache.find(cache_key);
                    if (transformed_it != transformed_terminal_cache.end()) {
                        return transformed_it->second;
                    }

                    const std::vector<GateParams>& resolved_gates =
                        resolved_gates_for_assignment(assignment_key);
                    if (resolved_gates.empty()) {
                        HDDNode* unchanged_terminal =
                            node_manager_.create_terminal_node(node->tensor_id);
                        transformed_terminal_cache.emplace(cache_key, unchanged_terminal);
                        return unchanged_terminal;
                    }

                    std::vector<SymbolicGaussianBranch> transformed_branches;

                    if (is_symbolic_terminal_id(node->tensor_id)) {
                        const SymbolicTerminalState& symbolic_state =
                            symbolic_terminal_states_.at(node->tensor_id);
                        transformed_branches.reserve(symbolic_state.branches.size());
                        for (const SymbolicGaussianBranch& existing_branch : symbolic_state.branches) {
                            SymbolicGaussianBranch transformed_branch = existing_branch;
                            transformed_branch.gaussian_state_id =
                                duplicate_gaussian_state(existing_branch.gaussian_state_id);
                            transformed_branches.push_back(std::move(transformed_branch));
                        }
                    } else {
                        auto cache_it = terminal_state_cache.find(node->tensor_id);
                        if (cache_it == terminal_state_cache.end()) {
                            cache_it = terminal_state_cache.emplace(
                                node->tensor_id,
                                classify_vacuum_ray_on_device(state_pool_, node->tensor_id)).first;
                        }
                        const VacuumRayInfo& info = cache_it->second;
                        if (info.is_zero) {
                            HDDNode* zero_terminal =
                                node_manager_.create_terminal_node(shared_zero_state_id_);
                            transformed_terminal_cache.emplace(cache_key, zero_terminal);
                            return zero_terminal;
                        }
                        if (!info.is_scaled_vacuum) {
                            throw std::runtime_error(
                                "当前Gaussian block仅支持vacuum或已有Gaussian/GaussianMixture terminal，state_id=" +
                                std::to_string(node->tensor_id));
                        }

                        const int gaussian_state_id = gaussian_state_pool_->allocate_state();
                        if (gaussian_state_id < 0) {
                            throw std::runtime_error("Gaussian状态池已满，无法创建symbolic branch");
                        }
                        initialize_gaussian_vacuum_state(gaussian_state_id);
                        transformed_branches.push_back(
                            {gaussian_state_id, info.scale, {}});
                    }

                    std::vector<int> gaussian_state_ids;
                    gaussian_state_ids.reserve(transformed_branches.size());
                    for (const SymbolicGaussianBranch& branch : transformed_branches) {
                        gaussian_state_ids.push_back(branch.gaussian_state_id);
                    }

                    auto compute_start = std::chrono::high_resolution_clock::now();
                    for (const GateParams& gate : resolved_gates) {
                        apply_symplectic_update_to_gaussian_states(
                            gaussian_state_ids,
                            gate_to_symplectic(gate, num_qumodes_));
                        for (SymbolicGaussianBranch& branch : transformed_branches) {
                            branch.replay_gates.push_back(gate);
                        }
                    }
                    auto compute_end = std::chrono::high_resolution_clock::now();
                    computation_time_ +=
                        std::chrono::duration<double, std::milli>(compute_end - compute_start).count();

                    int symbolic_terminal_id = allocate_symbolic_terminal_id();
                    symbolic_terminal_states_.emplace(
                        symbolic_terminal_id,
                        SymbolicTerminalState{std::move(transformed_branches)});
                    HDDNode* transformed_terminal =
                        node_manager_.create_terminal_node(symbolic_terminal_id);
                    transformed_terminal_cache.emplace(cache_key, transformed_terminal);
                    return transformed_terminal;
                }

                const int level = node->qubit_level;
                const int saved_value = qubit_assignment[level];
                qubit_assignment[level] = 0;
                HDDNode* new_low = transform_gaussian_block_recursive(node->low);
                qubit_assignment[level] = 1;
                HDDNode* new_high = transform_gaussian_block_recursive(node->high);
                qubit_assignment[level] = saved_value;

                return node_manager_.get_or_create_node(
                    node->qubit_level,
                    new_low,
                    new_high,
                    node->w_low,
                    node->w_high);
            };

        replace_root_node(transform_gaussian_block_recursive(root_node_));

        FALLBACK_DEBUG_LOG << "Gaussian EDE块级加速已启用，块门数="
                           << compiled_block.gates.size() << std::endl;
        return true;
    } catch (const std::exception& e) {
        clear_cuda_runtime_error_state();
        FALLBACK_DEBUG_LOG << "Gaussian EDE块回退到全量Fock执行: " << e.what() << std::endl;
        return false;
    }
}

bool QuantumCircuit::apply_gaussian_mixture_approximation_on_gpu(
    int state_id,
    const GaussianMixtureApproximation& approximation) {
    if (approximation.branches.empty()) {
        return true;
    }

    const int64_t state_dim = state_pool_.get_state_dim(state_id);
    if (state_dim <= 0) {
        return false;
    }

    const int scratch_state_id = state_pool_.allocate_state();
    if (scratch_state_id < 0) {
        throw std::runtime_error("Gaussian Mixture失败：无法分配scratch状态");
    }

    const int accum_state_id = state_pool_.allocate_state();
    if (accum_state_id < 0) {
        state_pool_.free_state(scratch_state_id);
        throw std::runtime_error("Gaussian Mixture失败：无法分配accumulator状态");
    }

    int* d_single_target = nullptr;
    auto cleanup = [&]() {
        state_pool_.free_state(accum_state_id);
        state_pool_.free_state(scratch_state_id);
    };

    try {
        state_pool_.reserve_state_storage(scratch_state_id, state_dim);
        state_pool_.reserve_state_storage(accum_state_id, state_dim);
        zero_state_device(&state_pool_, accum_state_id, nullptr, false);

        d_single_target = state_pool_.upload_values_to_buffer(
            &scratch_state_id, 1, state_pool_.scratch_target_ids);

        size_t max_diagonal_ops = 0;
        for (const GaussianMixtureBranch& branch : approximation.branches) {
            max_diagonal_ops = std::max(max_diagonal_ops, branch.target_qumodes.size());
        }
        if (max_diagonal_ops > 0) {
            state_pool_.scratch_aux.ensure(max_diagonal_ops * sizeof(FusedDiagonalOp));
        }

        for (const GaussianMixtureBranch& branch : approximation.branches) {
            if (branch.gaussian_gate.num_qumodes != num_qumodes_) {
                throw std::runtime_error("Gaussian Mixture分支qumode数量不匹配");
            }
            if (branch.target_qumodes.size() != branch.phase_rotation_thetas.size()) {
                throw std::runtime_error("Gaussian Mixture分支target/theta长度不匹配");
            }

            copy_state_device(&state_pool_, state_id, scratch_state_id, nullptr, false);

            std::vector<FusedDiagonalOp> diagonal_ops;
            diagonal_ops.reserve(branch.target_qumodes.size());
            for (size_t idx = 0; idx < branch.target_qumodes.size(); ++idx) {
                const double theta = branch.phase_rotation_thetas[idx];
                if (std::abs(theta) < kDiagonalCanonicalizationTolerance) {
                    continue;
                }
                diagonal_ops.push_back(FusedDiagonalOp{
                    compute_qumode_right_stride(state_pool_.d_trunc,
                                                branch.target_qumodes[idx],
                                                num_qumodes_),
                    theta,
                    0.0,
                    0.0});
            }
            if (!diagonal_ops.empty()) {
                apply_fused_diagonal_gates(
                    &state_pool_,
                    d_single_target,
                    1,
                    diagonal_ops,
                    num_qumodes_,
                    nullptr,
                    false);
            }

            axpy_state_device(
                &state_pool_,
                scratch_state_id,
                accum_state_id,
                make_cuDoubleComplex(branch.weight.real(), branch.weight.imag()),
                nullptr,
                false);
        }

        copy_state_device(&state_pool_, accum_state_id, state_id, nullptr, false);
        CHECK_CUDA(cudaDeviceSynchronize());
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    return true;
}

void QuantumCircuit::apply_replayable_gaussian_gate_to_state(int state_id, const GateParams& gate) {
    switch (gate.type) {
        case GateType::PHASE_ROTATION: {
            int* d_state_id = state_pool_.upload_values_to_buffer(
                &state_id, 1, state_pool_.scratch_target_ids);
            apply_phase_rotation_on_mode(
                &state_pool_,
                d_state_id,
                1,
                gate.params[0].real(),
                gate.target_qumodes[0],
                num_qumodes_,
                nullptr,
                false);
            break;
        }
        case GateType::DISPLACEMENT:
            apply_displacement_to_state(state_id, gate.params[0], gate.target_qumodes[0]);
            break;
        case GateType::SQUEEZING:
            apply_squeezing_to_state(state_id, gate.params[0], gate.target_qumodes[0]);
            break;
        case GateType::BEAM_SPLITTER: {
            const double theta = gate.params[0].real();
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            apply_beam_splitter_to_state(
                state_id,
                theta,
                phi,
                gate.target_qumodes[0],
                gate.target_qumodes[1]);
            break;
        }
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
            apply_two_mode_squeezing_to_state(
                state_id,
                gate.target_qumodes[0],
                gate.target_qumodes[1],
                gate.params[0]);
            break;
        case GateType::CONDITIONAL_SUM: {
            const double theta = gate.params[0].real();
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            apply_sum_to_state(
                state_id,
                gate.target_qumodes[0],
                gate.target_qumodes[1],
                theta,
                phi);
            break;
        }
        default:
            throw std::runtime_error("symbolic->Fock replay encountered unsupported Gaussian gate");
    }
}

int QuantumCircuit::project_symbolic_terminal_to_fock_state(int terminal_id) {
    auto it = symbolic_terminal_states_.find(terminal_id);
    if (it == symbolic_terminal_states_.end()) {
        throw std::runtime_error("symbolic terminal sidecar missing during Fock materialization");
    }

    if (gaussian_state_pool_ && it->second.branches.size() > 1) {
        std::unordered_map<std::string, size_t> unique_branch_index;
        std::vector<SymbolicGaussianBranch> coalesced_branches;
        coalesced_branches.reserve(it->second.branches.size());

        std::vector<double> displacement;
        std::vector<double> covariance;
        for (SymbolicGaussianBranch& branch : it->second.branches) {
            if (branch.gaussian_state_id < 0 ||
                std::abs(branch.weight) < kSymbolicBranchPruneTolerance) {
                if (branch.gaussian_state_id >= 0) {
                    gaussian_state_pool_->free_state(branch.gaussian_state_id);
                }
                continue;
            }

            gaussian_state_pool_->download_state(
                branch.gaussian_state_id, displacement, covariance);

            std::string state_key;
            state_key.resize((displacement.size() + covariance.size()) * sizeof(double));
            std::memcpy(state_key.data(),
                        displacement.data(),
                        displacement.size() * sizeof(double));
            std::memcpy(state_key.data() + displacement.size() * sizeof(double),
                        covariance.data(),
                        covariance.size() * sizeof(double));

            const auto [dedup_it, inserted] =
                unique_branch_index.emplace(state_key, coalesced_branches.size());
            if (inserted) {
                coalesced_branches.push_back(std::move(branch));
                continue;
            }

            SymbolicGaussianBranch& canonical_branch = coalesced_branches[dedup_it->second];
            canonical_branch.weight += branch.weight;
            gaussian_state_pool_->free_state(branch.gaussian_state_id);
        }

        std::vector<SymbolicGaussianBranch> pruned_branches;
        pruned_branches.reserve(coalesced_branches.size());
        for (SymbolicGaussianBranch& branch : coalesced_branches) {
            if (std::abs(branch.weight) < kSymbolicBranchPruneTolerance) {
                if (branch.gaussian_state_id >= 0) {
                    gaussian_state_pool_->free_state(branch.gaussian_state_id);
                }
                continue;
            }
            pruned_branches.push_back(std::move(branch));
        }

        if (pruned_branches.size() != it->second.branches.size()) {
            FALLBACK_DEBUG_LOG << "[fallback] coalesced symbolic terminal " << terminal_id
                               << " branches: " << it->second.branches.size()
                               << " -> " << pruned_branches.size() << std::endl;
        }
        it->second.branches = std::move(pruned_branches);
    }

    if (it->second.branches.empty()) {
        return shared_zero_state_id_;
    }

    const int64_t state_dim = state_pool_.get_max_total_dim();
    const int accum_state_id = state_pool_.allocate_state();
    if (accum_state_id < 0) {
        throw std::runtime_error("symbolic->Fock materialization failed: unable to allocate accumulator");
    }
    const int scratch_state_id = state_pool_.allocate_state();
    if (scratch_state_id < 0) {
        state_pool_.free_state(accum_state_id);
        throw std::runtime_error("symbolic->Fock materialization failed: unable to allocate scratch");
    }

    FALLBACK_DEBUG_LOG << "[fallback] projecting symbolic terminal " << terminal_id
                       << " to Fock, branches=" << it->second.branches.size() << std::endl;

    try {
        state_pool_.reserve_state_storage(accum_state_id, state_dim);
        zero_state_device(&state_pool_, accum_state_id, nullptr, false);
        state_pool_.reserve_state_storage(scratch_state_id, state_dim);

        for (size_t branch_index = 0; branch_index < it->second.branches.size(); ++branch_index) {
            const SymbolicGaussianBranch& branch = it->second.branches[branch_index];
            if (std::abs(branch.weight) < kSymbolicBranchPruneTolerance) {
                continue;
            }

            if (branch_index == 0 ||
                branch_index + 1 == it->second.branches.size() ||
                ((branch_index + 1) % 32 == 0)) {
                FALLBACK_DEBUG_LOG << "[fallback] terminal " << terminal_id
                                   << " replay branch " << (branch_index + 1)
                                   << "/" << it->second.branches.size() << std::endl;
            }

            FALLBACK_DEBUG_LOG << "[fallback] terminal " << terminal_id
                               << " branch " << (branch_index + 1)
                               << " reset scratch state " << scratch_state_id << std::endl;
            initialize_vacuum_state_device(&state_pool_, scratch_state_id, state_dim, nullptr, false);
            if (fallback_debug_logging_enabled()) {
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            for (size_t replay_gate_index = 0; replay_gate_index < branch.replay_gates.size(); ++replay_gate_index) {
                const GateParams& replay_gate = branch.replay_gates[replay_gate_index];
                FALLBACK_DEBUG_LOG << "[fallback] terminal " << terminal_id
                                   << " branch " << (branch_index + 1)
                                   << " gate " << (replay_gate_index + 1)
                                   << "/" << branch.replay_gates.size()
                                   << " " << gate_type_name(replay_gate.type)
                                   << " target_qumodes=";
                for (size_t tq_index = 0; tq_index < replay_gate.target_qumodes.size(); ++tq_index) {
                    if (tq_index != 0) {
                        FALLBACK_DEBUG_LOG << ",";
                    }
                    FALLBACK_DEBUG_LOG << replay_gate.target_qumodes[tq_index];
                }
                if (!replay_gate.params.empty()) {
                    FALLBACK_DEBUG_LOG << " param0=" << replay_gate.params[0];
                }
                FALLBACK_DEBUG_LOG << std::endl;
                apply_replayable_gaussian_gate_to_state(scratch_state_id, replay_gate);
                if (fallback_debug_logging_enabled()) {
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
                FALLBACK_DEBUG_LOG << "[fallback] terminal " << terminal_id
                                   << " branch " << (branch_index + 1)
                                   << " gate " << (replay_gate_index + 1)
                                   << " complete" << std::endl;
            }

            FALLBACK_DEBUG_LOG << "[fallback] terminal " << terminal_id
                               << " branch " << (branch_index + 1)
                               << " accumulate into state " << accum_state_id << std::endl;
            if (fallback_debug_logging_enabled()) {
                FALLBACK_DEBUG_LOG << "[fallback] terminal " << terminal_id
                                   << " accum metadata: id=" << accum_state_id
                                   << " dim=" << state_pool_.host_state_dims[accum_state_id]
                                   << " offset=" << state_pool_.host_state_offsets[accum_state_id]
                                   << " capacity=" << state_pool_.host_state_capacities[accum_state_id]
                                   << " ptr="
                                   << static_cast<const void*>(state_pool_.get_state_ptr(accum_state_id))
                                   << " | scratch metadata: id=" << scratch_state_id
                                   << " dim=" << state_pool_.host_state_dims[scratch_state_id]
                                   << " offset=" << state_pool_.host_state_offsets[scratch_state_id]
                                   << " capacity=" << state_pool_.host_state_capacities[scratch_state_id]
                                   << " ptr="
                                   << static_cast<const void*>(state_pool_.get_state_ptr(scratch_state_id))
                                   << std::endl;
            }
            axpy_state_device(
                &state_pool_,
                scratch_state_id,
                accum_state_id,
                make_cuDoubleComplex(branch.weight.real(), branch.weight.imag()),
                nullptr,
                false);
            if (fallback_debug_logging_enabled()) {
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    } catch (...) {
        state_pool_.free_state(scratch_state_id);
        state_pool_.free_state(accum_state_id);
        throw;
    }

    state_pool_.free_state(scratch_state_id);
    FALLBACK_DEBUG_LOG << "[fallback] symbolic terminal " << terminal_id
                       << " projected to Fock state " << accum_state_id << std::endl;
    return accum_state_id;
}

bool QuantumCircuit::materialize_symbolic_terminals_to_fock() {
    ScopedNvtxRange nvtx_range("qc::symbolic_to_fock");
    if (!has_symbolic_terminals()) {
        return true;
    }

    const std::vector<int> symbolic_terminal_ids = collect_symbolic_terminal_ids(root_node_);
    if (!symbolic_terminal_ids.empty()) {
        const size_t state_dim = static_cast<size_t>(state_pool_.get_max_total_dim());
        if (state_dim > 0) {
            const size_t active_storage = state_pool_.get_active_storage_elements();
            const size_t projected_terminal_count = symbolic_terminal_ids.size();
            if (projected_terminal_count >
                (std::numeric_limits<size_t>::max() / state_dim) - 1) {
                throw std::overflow_error(
                    "symbolic->Fock materialization storage estimate overflow");
            }
            const size_t reserved_projection_elements =
                (projected_terminal_count + 1) * state_dim;
            state_pool_.reserve_total_storage_elements(
                active_storage + reserved_projection_elements);
        }
    }
    FALLBACK_DEBUG_LOG << "[fallback] materialize_symbolic_terminals_to_fock begin"
                       << ", symbolic terminals=" << symbolic_terminal_ids.size() << std::endl;

    std::unordered_map<int, HDDNode*> projected_terminal_cache;
    std::function<HDDNode*(HDDNode*)> transform_recursive =
        [&](HDDNode* node) -> HDDNode* {
            if (!node) {
                return nullptr;
            }
            if (node->is_terminal()) {
                if (!is_symbolic_terminal_id(node->tensor_id)) {
                    return node;
                }
                const auto cached = projected_terminal_cache.find(node->tensor_id);
                if (cached != projected_terminal_cache.end()) {
                    return cached->second;
                }

                const int fock_state_id = project_symbolic_terminal_to_fock_state(node->tensor_id);
                HDDNode* projected_terminal = node_manager_.create_terminal_node(fock_state_id);
                projected_terminal_cache.emplace(node->tensor_id, projected_terminal);
                return projected_terminal;
            }

            HDDNode* new_low = transform_recursive(node->low);
            HDDNode* new_high = transform_recursive(node->high);
            return node_manager_.get_or_create_node(
                node->qubit_level,
                new_low,
                new_high,
                node->w_low,
                node->w_high);
        };

    replace_root_node(transform_recursive(root_node_));
    FALLBACK_DEBUG_LOG << "[fallback] materialize_symbolic_terminals_to_fock complete"
                       << std::endl;
    return true;
}

bool QuantumCircuit::try_execute_diagonal_non_gaussian_block_with_mixture(
    const CompiledExecutionBlock& compiled_block) {
    ScopedNvtxRange nvtx_range("qc::diagonal_mixture");
    if (compiled_block.kind != ExecutionBlockKind::DiagonalNonGaussian ||
        compiled_block.gates.empty()) {
        return false;
    }

    if (!compiled_block.diagonal_mixture_ready ||
        compiled_block.diagonal_mixture_updates.size() != compiled_block.gates.size()) {
        if (!compiled_block.compile_error.empty()) {
            FALLBACK_DEBUG_LOG << "对角非高斯块Mixture预编译失败，回退到精确Fock执行: "
                               << compiled_block.compile_error << std::endl;
        }
        return false;
    }

    ensure_gaussian_state_pool();
    std::unordered_map<int, VacuumRayInfo> terminal_state_cache;
    std::unordered_map<int, HDDNode*> transformed_terminal_cache;
    std::vector<int> created_symbolic_terminal_ids;
    std::vector<HDDNode*> created_nodes;
    const int saved_next_symbolic_terminal_id = next_symbolic_terminal_id_;

    auto free_branch_states = [&](std::vector<SymbolicGaussianBranch>* branches) {
        if (!gaussian_state_pool_) {
            return;
        }
        for (const SymbolicGaussianBranch& branch : *branches) {
            if (branch.gaussian_state_id >= 0) {
                gaussian_state_pool_->free_state(branch.gaussian_state_id);
            }
        }
        branches->clear();
    };

    auto make_terminal_node = [&](int tensor_id) -> HDDNode* {
        HDDNode* node = node_manager_.create_terminal_node(tensor_id);
        created_nodes.push_back(node);
        return node;
    };

    auto make_internal_node = [&](int16_t level,
                                  HDDNode* low,
                                  HDDNode* high,
                                  std::complex<double> w_low,
                                  std::complex<double> w_high) -> HDDNode* {
        HDDNode* node = node_manager_.get_or_create_node(level, low, high, w_low, w_high);
        created_nodes.push_back(node);
        return node;
    };

    try {
        std::function<HDDNode*(HDDNode*)> transform_recursive =
            [&](HDDNode* node) -> HDDNode* {
                if (!node) {
                    return nullptr;
                }
                if (node->is_terminal()) {
                    if (node->tensor_id == shared_zero_state_id_) {
                        return make_terminal_node(shared_zero_state_id_);
                    }

                    const auto transformed_it = transformed_terminal_cache.find(node->tensor_id);
                    if (transformed_it != transformed_terminal_cache.end()) {
                        return transformed_it->second;
                    }

                    std::vector<SymbolicGaussianBranch> current_branches;

                    if (is_symbolic_terminal_id(node->tensor_id)) {
                        const SymbolicTerminalState& symbolic_state =
                            symbolic_terminal_states_.at(node->tensor_id);
                        std::vector<std::complex<double>> initial_branch_weights;
                        initial_branch_weights.reserve(symbolic_state.branches.size());
                        for (const SymbolicGaussianBranch& existing_branch : symbolic_state.branches) {
                            initial_branch_weights.push_back(existing_branch.weight);
                        }

                        size_t projected_branch_count = symbolic_state.branches.size();
                        size_t failing_update_index = compiled_block.diagonal_mixture_updates.size();
                        if (symbolic_mixture_would_exceed_branch_limit(
                                initial_branch_weights,
                                compiled_block.diagonal_mixture_updates,
                                static_cast<size_t>(symbolic_branch_limit_),
                                &projected_branch_count,
                                &failing_update_index)) {
                            throw std::runtime_error(
                                "symbolic mixture branch expansion would exceed limit before update " +
                                std::to_string(failing_update_index + 1) +
                                " (" + std::to_string(projected_branch_count) +
                                " > " + std::to_string(symbolic_branch_limit_) +
                                "), switching block to exact Fock");
                        }

                        current_branches.reserve(symbolic_state.branches.size());
                        for (const SymbolicGaussianBranch& existing_branch : symbolic_state.branches) {
                            SymbolicGaussianBranch duplicated_branch = existing_branch;
                            duplicated_branch.gaussian_state_id =
                                duplicate_gaussian_state(existing_branch.gaussian_state_id);
                            current_branches.push_back(std::move(duplicated_branch));
                        }
                    } else {
                        auto cache_it = terminal_state_cache.find(node->tensor_id);
                        if (cache_it == terminal_state_cache.end()) {
                            cache_it = terminal_state_cache.emplace(
                                node->tensor_id,
                                classify_vacuum_ray_on_device(state_pool_, node->tensor_id)).first;
                        }
                        const VacuumRayInfo& info = cache_it->second;
                        if (info.is_zero) {
                            HDDNode* zero_terminal = make_terminal_node(shared_zero_state_id_);
                            transformed_terminal_cache.emplace(node->tensor_id, zero_terminal);
                            return zero_terminal;
                        }
                        if (!info.is_scaled_vacuum) {
                            throw std::runtime_error(
                                "当前DiagonalNonGaussian symbolic路径仅支持vacuum或已有Gaussian/GaussianMixture terminal，state_id=" +
                                std::to_string(node->tensor_id));
                        }

                        const std::vector<std::complex<double>> initial_branch_weights{info.scale};
                        size_t projected_branch_count = 1;
                        size_t failing_update_index = compiled_block.diagonal_mixture_updates.size();
                        if (symbolic_mixture_would_exceed_branch_limit(
                                initial_branch_weights,
                                compiled_block.diagonal_mixture_updates,
                                static_cast<size_t>(symbolic_branch_limit_),
                                &projected_branch_count,
                                &failing_update_index)) {
                            throw std::runtime_error(
                                "symbolic mixture branch expansion would exceed limit before update " +
                                std::to_string(failing_update_index + 1) +
                                " (" + std::to_string(projected_branch_count) +
                                " > " + std::to_string(symbolic_branch_limit_) +
                                "), switching block to exact Fock");
                        }

                        const int gaussian_state_id = gaussian_state_pool_->allocate_state();
                        if (gaussian_state_id < 0) {
                            throw std::runtime_error("Gaussian状态池已满，无法创建mixture base branch");
                        }
                        initialize_gaussian_vacuum_state(gaussian_state_id);
                        current_branches.push_back({gaussian_state_id, info.scale, {}});
                    }

                    try {
                        auto compute_start = std::chrono::high_resolution_clock::now();
                        for (const GaussianMixtureApproximation& approximation :
                             compiled_block.diagonal_mixture_updates) {
                            std::vector<SymbolicGaussianBranch> expanded_branches;
                            expanded_branches.reserve(std::min(
                                current_branches.size() * std::max<size_t>(size_t{1}, approximation.branches.size()),
                                static_cast<size_t>(symbolic_branch_limit_)));
                            for (const SymbolicGaussianBranch& base_branch : current_branches) {
                                for (const GaussianMixtureBranch& mixture_branch : approximation.branches) {
                                    const std::complex<double> new_weight =
                                        base_branch.weight * mixture_branch.weight;
                                    if (std::abs(new_weight) < kSymbolicBranchPruneTolerance) {
                                        continue;
                                    }

                                    SymbolicGaussianBranch expanded_branch = base_branch;
                                    expanded_branch.weight = new_weight;
                                    expanded_branch.gaussian_state_id =
                                        duplicate_gaussian_state(base_branch.gaussian_state_id);

                                    for (size_t idx = 0; idx < mixture_branch.target_qumodes.size(); ++idx) {
                                        const double theta = mixture_branch.phase_rotation_thetas[idx];
                                        if (std::abs(theta) < kDiagonalCanonicalizationTolerance) {
                                            continue;
                                        }
                                        GateParams phase_gate(
                                            GateType::PHASE_ROTATION,
                                            {},
                                            {mixture_branch.target_qumodes[idx]},
                                            {std::complex<double>(theta, 0.0)});
                                        apply_symplectic_update_to_gaussian_states(
                                            {expanded_branch.gaussian_state_id},
                                            gate_to_symplectic(phase_gate, num_qumodes_));
                                        expanded_branch.replay_gates.push_back(phase_gate);
                                    }
                                    expanded_branches.push_back(std::move(expanded_branch));
                                }
                            }

                            free_branch_states(&current_branches);
                            current_branches = std::move(expanded_branches);
                            if (current_branches.size() >
                                static_cast<size_t>(symbolic_branch_limit_)) {
                                free_branch_states(&current_branches);
                                throw std::runtime_error("symbolic mixture branch count exceeded limit");
                            }
                        }
                        auto compute_end = std::chrono::high_resolution_clock::now();
                        computation_time_ +=
                            std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
                    } catch (...) {
                        free_branch_states(&current_branches);
                        throw;
                    }

                    if (current_branches.empty()) {
                        HDDNode* zero_terminal = make_terminal_node(shared_zero_state_id_);
                        transformed_terminal_cache.emplace(node->tensor_id, zero_terminal);
                        return zero_terminal;
                    }

                    int symbolic_terminal_id = allocate_symbolic_terminal_id();
                    created_symbolic_terminal_ids.push_back(symbolic_terminal_id);
                    symbolic_terminal_states_.emplace(
                        symbolic_terminal_id,
                        SymbolicTerminalState{std::move(current_branches)});
                    HDDNode* transformed_terminal = make_terminal_node(symbolic_terminal_id);
                    transformed_terminal_cache.emplace(node->tensor_id, transformed_terminal);
                    return transformed_terminal;
                }

                HDDNode* new_low = transform_recursive(node->low);
                HDDNode* new_high = transform_recursive(node->high);
                return make_internal_node(
                    node->qubit_level, new_low, new_high, node->w_low, node->w_high);
            };

        replace_root_node(transform_recursive(root_node_));
    } catch (const std::exception& e) {
        for (auto it = created_nodes.rbegin(); it != created_nodes.rend(); ++it) {
            node_manager_.release_node(*it);
        }
        for (int terminal_id : created_symbolic_terminal_ids) {
            release_symbolic_terminal(terminal_id);
        }
        next_symbolic_terminal_id_ = saved_next_symbolic_terminal_id;
        FALLBACK_DEBUG_LOG << "对角非高斯块Gaussian Mixture回退到精确Fock执行: "
                           << e.what() << std::endl;
        return false;
    }

    FALLBACK_DEBUG_LOG << "对角非高斯块Gaussian Mixture已启用，块门数="
                       << compiled_block.gates.size()
                       << "，总分支数=" << compiled_block.mixture_branch_count
                       << "，估计对角L2误差=" << compiled_block.estimated_diagonal_l2_error
                       << "，保守fidelity下界=" << compiled_block.diagonal_fidelity_lower_bound
                       << std::endl;
    return true;
}

/**
 * 执行量子线路
 */
void QuantumCircuit::execute() {
    (void)execute_range(0, std::numeric_limits<size_t>::max());
}

size_t QuantumCircuit::get_execution_block_count() const {
    const std::vector<GateParams> execution_sequence = canonicalize_gate_sequence_for_execution();
    return partition_execution_blocks(execution_sequence).size();
}

size_t QuantumCircuit::execute_range(size_t start_block, size_t max_blocks) {
    ScopedNvtxRange nvtx_range("qc::execute");
    if (!is_built_) {
        throw std::runtime_error("必须先构建量子线路");
    }

    if (max_blocks == 0) {
        return start_block;
    }

    if (is_executed_ && start_block == 0) {
        std::cout << "线路已执行，跳过重复执行" << std::endl;
        return get_execution_block_count();
    }

    std::cout << "执行量子线路..." << std::endl;

    // 重置时间统计
    total_time_ = 0.0;
    transfer_time_ = 0.0;
    computation_time_ = 0.0;
    planning_time_ = 0.0;

    // 记录总开始时间
    auto start_total = std::chrono::high_resolution_clock::now();

    auto planning_start = std::chrono::high_resolution_clock::now();
    std::vector<GateParams> execution_sequence;
    std::vector<ExecutionBlock> execution_blocks;
    {
        ScopedNvtxRange planning_range("qc::planning");
        execution_sequence = canonicalize_gate_sequence_for_execution();
        execution_blocks = partition_execution_blocks(execution_sequence);
    }
    auto planning_end = std::chrono::high_resolution_clock::now();
    planning_time_ +=
        std::chrono::duration<double, std::milli>(planning_end - planning_start).count();

    const size_t total_blocks = execution_blocks.size();
    if (execution_blocks.empty()) {
        collect_hdd_garbage_if_needed(true);
        is_executed_ = true;
        auto end_total = std::chrono::high_resolution_clock::now();
        total_time_ = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        return 0;
    }

    if (start_block > total_blocks) {
        throw std::out_of_range("start_block 超出执行块范围");
    }
    if (start_block == total_blocks) {
        is_executed_ = true;
        auto end_total = std::chrono::high_resolution_clock::now();
        total_time_ = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        return total_blocks;
    }

    CompiledExecutionBlock current_block =
        compile_execution_block(execution_sequence, execution_blocks, start_block);
    planning_time_ += current_block.compile_time_ms;

    std::future<CompiledExecutionBlock> next_block_future;

    if (total_blocks > 1) {
        std::cout << "块级编译-执行流水线已启用，块数=" << total_blocks << std::endl;
    }

    size_t executed_blocks = 0;
    for (size_t block_index = start_block;
         block_index < total_blocks && executed_blocks < max_blocks;
         ++block_index, ++executed_blocks) {
        log_block_progress_if_requested(block_index, total_blocks);

        if (block_index + 1 < total_blocks &&
            executed_blocks + 1 < max_blocks &&
            !next_block_future.valid()) {
            next_block_future = std::async(
                std::launch::async,
                [this, &execution_sequence, &execution_blocks, block_index]() {
                    return compile_execution_block(
                        execution_sequence, execution_blocks, block_index + 1);
                });
        }

        bool block_executed = false;
        if (current_block.kind == ExecutionBlockKind::Gaussian) {
            block_executed = try_execute_gaussian_block_with_ede(current_block);
        } else if (current_block.kind == ExecutionBlockKind::DiagonalNonGaussian) {
            block_executed = try_execute_diagonal_non_gaussian_block_with_mixture(current_block);
        }

        if (block_executed) {
            synchronize_async_cv_pipeline();
            collect_hdd_garbage_if_needed(false);
            if (block_index + 1 < total_blocks &&
                executed_blocks + 1 < max_blocks) {
                current_block = next_block_future.get();
                planning_time_ += current_block.compile_time_ms;
                next_block_future = std::future<CompiledExecutionBlock>();
            }
            continue;
        }

        if (current_block.kind != ExecutionBlockKind::QubitOnly &&
            has_symbolic_terminals()) {
            const auto symbolic_terminal_ids = collect_symbolic_terminal_ids(root_node_);
            if (!symbolic_terminal_ids.empty()) {
                const size_t state_dim = static_cast<size_t>(state_pool_.get_max_total_dim());
                if (state_dim > 0) {
                    const size_t active_storage = state_pool_.get_active_storage_elements();
                    const size_t projected_terminal_count = symbolic_terminal_ids.size();
                    if (projected_terminal_count >
                        (std::numeric_limits<size_t>::max() / state_dim) - 1) {
                        throw std::overflow_error(
                            "symbolic->Fock materialization storage estimate overflow");
                    }

                    size_t extra_pairwise_elements = 0;
                    PairwiseHybridStorageEstimate pairwise_estimate;
                    bool pairwise_early_release_enabled = false;
                    for (const GateParams& gate : current_block.gates) {
                        if (!is_pairwise_hybrid_gate_type(gate.type) ||
                            gate.target_qubits.empty()) {
                            continue;
                        }

                        const PairwiseHybridStorageEstimate candidate =
                            estimate_pairwise_hybrid_storage(
                                root_node_,
                                gate.target_qubits[0],
                                state_dim);
                        const size_t candidate_working_extra =
                            pairwise_hybrid_working_extra_elements(
                                root_node_,
                                gate.target_qubits[0],
                                state_dim,
                                candidate);
                        if (candidate_working_extra > extra_pairwise_elements) {
                            pairwise_estimate = candidate;
                            extra_pairwise_elements = candidate_working_extra;
                            pairwise_early_release_enabled =
                                candidate.extra_elements != 0 &&
                                candidate_working_extra < candidate.extra_elements;
                        }
                    }
                    if (extra_pairwise_elements != 0) {
                        FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                                           << " pre-reserving pairwise hybrid headroom"
                                           << " projected_terminals=" << projected_terminal_count
                                           << " pairwise_pairs=" << pairwise_estimate.pair_count
                                           << " duplicate_states="
                                           << pairwise_estimate.duplicate_state_count
                                           << " working_extra_elements=" << extra_pairwise_elements
                                           << " early_release="
                                           << (pairwise_early_release_enabled ? 1 : 0)
                                           << std::endl;
                    }

                    const size_t reserved_projection_elements =
                        (projected_terminal_count + 1) * state_dim;
                    size_t exact_phase_peak_elements = reserved_projection_elements;
                    if (projected_terminal_count >
                        std::numeric_limits<size_t>::max() / state_dim) {
                        throw std::overflow_error(
                            "pairwise hybrid active exact-state estimate overflow");
                    }
                    const size_t duplicated_gate_peak_elements =
                        projected_terminal_count * state_dim + extra_pairwise_elements;
                    exact_phase_peak_elements =
                        std::max(exact_phase_peak_elements, duplicated_gate_peak_elements);
                    state_pool_.reserve_total_storage_elements(
                        active_storage + exact_phase_peak_elements);
                }
            }
            FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                               << " materializing symbolic terminals to exact Fock"
                               << std::endl;
            materialize_symbolic_terminals_to_fock();
            FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                               << " symbolic materialization complete" << std::endl;
        }

        // ── Cross-mode fused diagonal optimization ──────────────────
        // Fuse multiple diagonal gates (PhaseRotation/Kerr/ConditionalParity)
        // into a single kernel pass. This also catches Gaussian blocks that
        // fell back from EDE (their kind is still Gaussian but they contain
        // fusable diagonal gates like PhaseRotation).
        if (current_block.kind != ExecutionBlockKind::QubitOnly) {
            // Partition gates: fusable simple diagonals vs. everything else
            std::vector<GateParams> fusable_gates;
            std::vector<GateParams> other_gates;
            for (const GateParams& gate : current_block.gates) {
                if (gate.target_qubits.empty() && !gate.target_qumodes.empty() &&
                    (gate.type == GateType::PHASE_ROTATION ||
                     gate.type == GateType::KERR_GATE ||
                     gate.type == GateType::CONDITIONAL_PARITY)) {
                    fusable_gates.push_back(gate);
                } else {
                    other_gates.push_back(gate);
                }
            }

            if (fusable_gates.size() >= 2) {
                // Build per-mode descriptor: accumulate params by target mode
                std::map<int, FusedDiagonalOp> mode_ops;
                for (const GateParams& gate : fusable_gates) {
                    int mode = gate.target_qumodes[0];
                    auto& op = mode_ops[mode];
                    if (op.right_stride == 0) {
                        // Compute right stride for this mode
                        int rs = 1;
                        for (int m = mode + 1; m < num_qumodes_; ++m) {
                            rs *= cv_truncation_;
                        }
                        op.right_stride = rs;
                    }
                    double param = gate.params[0].real();
                    switch (gate.type) {
                        case GateType::PHASE_ROTATION:   op.theta   += param; break;
                        case GateType::KERR_GATE:        op.chi     += param; break;
                        case GateType::CONDITIONAL_PARITY: op.parity += param; break;
                        default: break;
                    }
                }

                std::vector<FusedDiagonalOp> ops_vec;
                ops_vec.reserve(mode_ops.size());
                for (auto& [mode, op] : mode_ops) {
                    if (std::abs(op.theta) > 1e-14 ||
                        std::abs(op.chi)   > 1e-14 ||
                        std::abs(op.parity)> 1e-14) {
                        ops_vec.push_back(op);
                    }
                }

                if (!ops_vec.empty()) {
                    const auto& target_states = get_cached_target_states();
                    if (!target_states.empty()) {
                        auto transfer_start = std::chrono::high_resolution_clock::now();
                        size_t upload_slot = 0;
                        auto [d_target_ids, ids_bytes] = upload_target_states_for_compute(
                            target_states, &upload_slot);
                        auto transfer_end = std::chrono::high_resolution_clock::now();
                        transfer_time_ += std::chrono::duration<double, std::milli>(
                            transfer_end - transfer_start).count();

                        auto compute_start = std::chrono::high_resolution_clock::now();
                        apply_fused_diagonal_gates(&state_pool_, d_target_ids,
                                                   static_cast<int>(target_states.size()),
                                                   ops_vec, num_qumodes_,
                                                   async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                                   !async_cv_pipeline_enabled_);
                        if (async_cv_pipeline_enabled_) {
                            mark_target_upload_slot_in_use(upload_slot);
                        }
                        auto compute_end = std::chrono::high_resolution_clock::now();
                        computation_time_ += std::chrono::duration<double, std::milli>(
                            compute_end - compute_start).count();
                    }
                }

                // Execute remaining non-fusable gates normally
                FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                                   << " executing exact path after fused diagonal split"
                                   << " kind=" << static_cast<int>(current_block.kind)
                                   << " (remaining gates=" << other_gates.size() << ")"
                                   << std::endl;
                for (size_t gate_index = 0; gate_index < other_gates.size(); ++gate_index) {
                    const GateParams& gate = other_gates[gate_index];
                    FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                                       << " gate " << (gate_index + 1) << "/"
                                       << other_gates.size() << " "
                                       << gate_type_name(gate.type) << std::endl;
                    execute_gate(gate);
                }

                synchronize_async_cv_pipeline();
                collect_hdd_garbage_if_needed(false);
                FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                                   << " exact path complete after fused diagonal split"
                                   << std::endl;

                if (block_index + 1 < total_blocks &&
                    executed_blocks + 1 < max_blocks) {
                    current_block = next_block_future.get();
                    planning_time_ += current_block.compile_time_ms;
                    next_block_future = std::future<CompiledExecutionBlock>();
                }
                continue;
            }
        }
        // ── End fused diagonal ──────────────────────────────────────

        FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                           << " executing exact path with " << current_block.gates.size()
                           << " gates"
                           << " kind=" << static_cast<int>(current_block.kind) << std::endl;
        for (size_t gate_index = 0; gate_index < current_block.gates.size(); ++gate_index) {
            const GateParams& gate = current_block.gates[gate_index];
            FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                               << " gate " << (gate_index + 1) << "/"
                               << current_block.gates.size() << " "
                               << gate_type_name(gate.type) << std::endl;
            execute_gate(gate);
        }

        synchronize_async_cv_pipeline();
        collect_hdd_garbage_if_needed(false);
        FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                           << " exact path complete" << std::endl;

        if (block_index + 1 < total_blocks &&
            executed_blocks + 1 < max_blocks) {
            current_block = next_block_future.get();
            planning_time_ += current_block.compile_time_ms;
            next_block_future = std::future<CompiledExecutionBlock>();
        }
    }

    // 记录总结束时间
    auto end_total = std::chrono::high_resolution_clock::now();
    total_time_ = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    const size_t next_block_index =
        std::min(start_block + executed_blocks, total_blocks);
    is_executed_ = (next_block_index == total_blocks);
    if (is_executed_) {
        std::cout << "量子线路执行完成" << std::endl;
        std::cout << "执行时间: " << total_time_ << " ms" << std::endl;
        std::cout << "传输时延: " << transfer_time_ << " ms" << std::endl;
        std::cout << "计算时延: " << computation_time_ << " ms" << std::endl;
        std::cout << "规划时延: " << planning_time_ << " ms" << std::endl;
    }
    return next_block_index;
}

void QuantumCircuit::save_exact_fock_checkpoint(const std::string& path,
                                                size_t next_block_index,
                                                size_t total_blocks) const {
    if (!is_built_ || !root_node_) {
        throw std::runtime_error("无法保存checkpoint：线路未构建或根节点为空");
    }
    if (has_symbolic_terminals()) {
        throw std::runtime_error("当前checkpoint仅支持exact Fock状态，不支持symbolic terminals");
    }

    std::vector<int> state_ids = collect_terminal_state_ids(root_node_);
    if (shared_zero_state_id_ >= 0 &&
        std::find(state_ids.begin(), state_ids.end(), shared_zero_state_id_) == state_ids.end()) {
        state_ids.push_back(shared_zero_state_id_);
        std::sort(state_ids.begin(), state_ids.end());
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("无法打开checkpoint文件用于写入: " + path);
    }

    ExactFockCheckpointHeader header{};
    std::memcpy(header.magic, "HCVDVCK1", sizeof(header.magic));
    header.version = 1;
    header.num_qubits = static_cast<uint32_t>(num_qubits_);
    header.num_qumodes = static_cast<uint32_t>(num_qumodes_);
    header.cv_truncation = static_cast<uint32_t>(cv_truncation_);
    header.max_states = static_cast<uint32_t>(state_pool_.capacity);
    header.shared_zero_state_id = shared_zero_state_id_;
    header.next_block_index = static_cast<uint64_t>(next_block_index);
    header.total_blocks = static_cast<uint64_t>(total_blocks);
    header.state_count = static_cast<uint64_t>(state_ids.size());

    std::vector<ExactFockCheckpointStateRecord> state_records;
    state_records.reserve(state_ids.size());
    std::vector<std::vector<cuDoubleComplex>> state_payloads;
    state_payloads.reserve(state_ids.size());
    for (int state_id : state_ids) {
        std::vector<cuDoubleComplex> host_state;
        state_pool_.download_state(state_id, host_state);
        state_records.push_back(
            ExactFockCheckpointStateRecord{state_id, static_cast<int32_t>(host_state.size())});
        state_payloads.push_back(std::move(host_state));
    }

    std::vector<ExactFockCheckpointNodeRecord> node_records;
    std::unordered_map<HDDNode*, uint64_t> node_indices;
    std::function<uint64_t(HDDNode*)> serialize_node =
        [&](HDDNode* node) -> uint64_t {
            const auto cached = node_indices.find(node);
            if (cached != node_indices.end()) {
                return cached->second;
            }

            ExactFockCheckpointNodeRecord record{};
            if (node->is_terminal()) {
                record.is_terminal = 1;
                record.qubit_level = -1;
                record.tensor_id = node->tensor_id;
            } else {
                record.is_terminal = 0;
                record.qubit_level = node->qubit_level;
                record.low_index = serialize_node(node->low);
                record.high_index = serialize_node(node->high);
                record.w_low_real = node->w_low.real();
                record.w_low_imag = node->w_low.imag();
                record.w_high_real = node->w_high.real();
                record.w_high_imag = node->w_high.imag();
            }

            const uint64_t index = static_cast<uint64_t>(node_records.size());
            node_records.push_back(record);
            node_indices.emplace(node, index);
            return index;
        };
    header.root_index = serialize_node(root_node_);
    header.node_count = static_cast<uint64_t>(node_records.size());

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (size_t i = 0; i < state_records.size(); ++i) {
        out.write(reinterpret_cast<const char*>(&state_records[i]), sizeof(state_records[i]));
        const std::vector<cuDoubleComplex>& payload = state_payloads[i];
        if (!payload.empty()) {
            out.write(reinterpret_cast<const char*>(payload.data()),
                      static_cast<std::streamsize>(payload.size() * sizeof(cuDoubleComplex)));
        }
    }
    for (const ExactFockCheckpointNodeRecord& record : node_records) {
        out.write(reinterpret_cast<const char*>(&record), sizeof(record));
    }
    if (!out) {
        throw std::runtime_error("写入checkpoint失败: " + path);
    }
}

size_t QuantumCircuit::load_exact_fock_checkpoint(const std::string& path, size_t* total_blocks) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法打开checkpoint文件: " + path);
    }

    ExactFockCheckpointHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) {
        throw std::runtime_error("读取checkpoint头失败: " + path);
    }
    if (std::memcmp(header.magic, "HCVDVCK1", sizeof(header.magic)) != 0 || header.version != 1) {
        throw std::runtime_error("checkpoint格式不受支持: " + path);
    }
    if (header.num_qubits != static_cast<uint32_t>(num_qubits_) ||
        header.num_qumodes != static_cast<uint32_t>(num_qumodes_) ||
        header.cv_truncation != static_cast<uint32_t>(cv_truncation_) ||
        header.max_states != static_cast<uint32_t>(state_pool_.capacity)) {
        throw std::runtime_error("checkpoint配置与当前线路不匹配");
    }

    synchronize_async_cv_pipeline();
    if (root_node_) {
        node_manager_.release_node(root_node_);
        root_node_ = nullptr;
    }
    node_manager_.clear();
    clear_symbolic_terminals();
    state_pool_.reset();
    invalidate_root_caches();
    is_executed_ = false;
    pending_gc_replacements_ = 0;

    std::unordered_map<int32_t, int32_t> state_id_map;
    for (uint64_t state_index = 0; state_index < header.state_count; ++state_index) {
        ExactFockCheckpointStateRecord record{};
        in.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!in || record.state_dim < 0) {
            throw std::runtime_error("读取checkpoint状态元数据失败: " + path);
        }

        std::vector<cuDoubleComplex> payload(static_cast<size_t>(record.state_dim));
        if (!payload.empty()) {
            in.read(reinterpret_cast<char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size() * sizeof(cuDoubleComplex)));
            if (!in) {
                throw std::runtime_error("读取checkpoint状态数据失败: " + path);
            }
        }

        const int new_state_id = state_pool_.allocate_state();
        if (new_state_id < 0) {
            throw std::runtime_error("加载checkpoint失败：状态池容量不足");
        }
        state_pool_.upload_state(new_state_id, payload);
        state_id_map.emplace(record.state_id, new_state_id);
    }

    if (header.node_count == 0 || header.root_index >= header.node_count) {
        throw std::runtime_error("checkpoint节点图无效");
    }

    std::vector<HDDNode*> rebuilt_nodes(static_cast<size_t>(header.node_count), nullptr);
    for (uint64_t node_index = 0; node_index < header.node_count; ++node_index) {
        ExactFockCheckpointNodeRecord record{};
        in.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!in) {
            throw std::runtime_error("读取checkpoint节点失败: " + path);
        }

        if (record.is_terminal != 0) {
            const auto mapped = state_id_map.find(record.tensor_id);
            if (mapped == state_id_map.end()) {
                throw std::runtime_error("checkpoint引用了未知状态ID");
            }
            rebuilt_nodes[static_cast<size_t>(node_index)] =
                node_manager_.create_terminal_node(mapped->second);
        } else {
            if (record.low_index >= node_index || record.high_index >= node_index) {
                throw std::runtime_error("checkpoint节点拓扑顺序无效");
            }
            rebuilt_nodes[static_cast<size_t>(node_index)] =
                node_manager_.get_or_create_node(
                    record.qubit_level,
                    rebuilt_nodes[static_cast<size_t>(record.low_index)],
                    rebuilt_nodes[static_cast<size_t>(record.high_index)],
                    std::complex<double>(record.w_low_real, record.w_low_imag),
                    std::complex<double>(record.w_high_real, record.w_high_imag));
        }
    }

    root_node_ = rebuilt_nodes[static_cast<size_t>(header.root_index)];
    shared_zero_state_id_ = -1;
    if (header.shared_zero_state_id >= 0) {
        const auto mapped = state_id_map.find(header.shared_zero_state_id);
        if (mapped == state_id_map.end()) {
            throw std::runtime_error("checkpoint缺少shared zero state");
        }
        shared_zero_state_id_ = mapped->second;
    }
    if (total_blocks) {
        *total_blocks = static_cast<size_t>(header.total_blocks);
    }
    return static_cast<size_t>(header.next_block_index);
}

/**
 * 重置量子线路状态
 */
void QuantumCircuit::reset() {
    // 同步所有GPU操作，确保在重置前所有操作完成
    release_async_cv_pipeline();
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess && sync_err != cudaErrorNotReady) {
        // 如果之前的操作有错误，尝试清除错误状态
        std::cerr << "警告：重置前检测到GPU错误: " << cudaGetErrorString(sync_err) << std::endl;
        // 清除CUDA错误状态，允许后续操作继续
        cudaGetLastError(); // 清除错误标志
    }

    if (root_node_) {
        node_manager_.release_node(root_node_);
        root_node_ = nullptr;
    }

    node_manager_.clear();
    clear_symbolic_terminals();
    state_pool_.reset();  // 重置状态池，释放所有分配的状态
    gate_sequence_.clear();
    is_built_ = false;
    is_executed_ = false;
    shared_zero_state_id_ = -1;
    pending_gc_replacements_ = 0;
    async_cv_pipeline_enabled_ = false;
    invalidate_root_caches();
    
    // 重置时间统计
    total_time_ = 0.0;
    transfer_time_ = 0.0;
    computation_time_ = 0.0;
    planning_time_ = 0.0;
}

void QuantumCircuit::set_gaussian_symbolic_mode_limit(int limit) {
    if (limit <= 0) {
        throw std::invalid_argument("Gaussian symbolic mode limit must be positive");
    }
    gaussian_symbolic_mode_limit_ = limit;
}

void QuantumCircuit::set_symbolic_branch_limit(int limit) {
    if (limit <= 0) {
        throw std::invalid_argument("symbolic branch limit must be positive");
    }
    symbolic_branch_limit_ = limit;
}

void QuantumCircuit::set_gaussian_state_pool_capacity(int capacity) {
    if (capacity <= 0) {
        throw std::invalid_argument("Gaussian state pool capacity must be positive");
    }

    const int desired_capacity = capacity;
    if (gaussian_state_pool_ &&
        gaussian_state_pool_->get_capacity() != desired_capacity) {
        throw std::logic_error(
            "Gaussian state pool capacity must be configured before the pool is initialized");
    }

    gaussian_state_pool_capacity_override_ = desired_capacity;
}

/**
 * 初始化HDD结构
 */
void QuantumCircuit::initialize_hdd() {
    const int vacuum_state_id = state_pool_.allocate_state();
    if (vacuum_state_id < 0) {
        throw std::runtime_error("初始化HDD失败：无法分配初始状态");
    }

    const int64_t total_dim = state_pool_.get_max_total_dim();
    initialize_vacuum_state_device(&state_pool_, vacuum_state_id, total_dim);

    // Pure CV (nq=0): no qubit branching, so skip zero_state allocation to save
    // one full state_dim of GPU memory (critical for nm=8 where state = 68.7GB).
    if (num_qubits_ == 0) {
        shared_zero_state_id_ = -1;
        HDDNode* active_branch = node_manager_.create_terminal_node(vacuum_state_id);
        root_node_ = active_branch;
        invalidate_root_caches();
        return;
    }

    const int zero_state_id = state_pool_.allocate_state();
    if (zero_state_id < 0) {
        throw std::runtime_error("初始化HDD失败：无法分配零状态");
    }

    state_pool_.reserve_state_storage(zero_state_id, total_dim);
    zero_state_device(&state_pool_, zero_state_id);
    shared_zero_state_id_ = zero_state_id;

    // 共享零态仅用于HDD零分支占位，不计入面向用户的活跃态统计。
    if (state_pool_.active_count > 0) {
        --state_pool_.active_count;
    }

    HDDNode* active_branch = node_manager_.create_terminal_node(vacuum_state_id);
    HDDNode* zero_branch = node_manager_.create_terminal_node(zero_state_id);

    for (int level = 0; level < num_qubits_; ++level) {
        HDDNode* next_active = node_manager_.get_or_create_node(level, active_branch, zero_branch, 1.0, 1.0);
        HDDNode* next_zero = node_manager_.get_or_create_node(level, zero_branch, zero_branch, 1.0, 1.0);
        active_branch = next_active;
        zero_branch = next_zero;
    }

    root_node_ = active_branch;
    invalidate_root_caches();
}

void QuantumCircuit::replace_root_node(HDDNode* new_root) {
    if (new_root == root_node_) {
        return;
    }

    const std::vector<int> old_state_ids = collect_terminal_state_ids(root_node_);
    const std::vector<int> new_state_ids = collect_terminal_state_ids(new_root);
    const std::vector<int> old_symbolic_ids = collect_symbolic_terminal_ids(root_node_);
    const std::vector<int> new_symbolic_ids = collect_symbolic_terminal_ids(new_root);
    std::unordered_set<int> new_state_set(new_state_ids.begin(), new_state_ids.end());
    std::unordered_set<int> new_symbolic_set(new_symbolic_ids.begin(), new_symbolic_ids.end());

    HDDNode* old_root = root_node_;
    root_node_ = new_root;
    invalidate_root_caches();

    if (old_root) {
        node_manager_.release_node(old_root);
    }
    ++pending_gc_replacements_;

    for (int state_id : old_state_ids) {
        if (state_id == shared_zero_state_id_) {
            continue;
        }
        if (new_state_set.find(state_id) == new_state_set.end()) {
            state_pool_.free_state(state_id);
        }
    }

    for (int symbolic_id : old_symbolic_ids) {
        if (new_symbolic_set.find(symbolic_id) == new_symbolic_set.end()) {
            release_symbolic_terminal(symbolic_id);
        }
    }

    collect_hdd_garbage_if_needed(false);
}

void QuantumCircuit::replace_root_node_preserving_terminals(HDDNode* new_root) {
    if (new_root == root_node_) {
        return;
    }

    HDDNode* old_root = root_node_;
    root_node_ = new_root;
    invalidate_root_caches();

    if (old_root) {
        node_manager_.release_node(old_root);
    }
    ++pending_gc_replacements_;
    collect_hdd_garbage_if_needed(false);
}

void QuantumCircuit::collect_hdd_garbage_if_needed(bool force) {
    constexpr size_t kGarbageCollectReplacementInterval = 32;
    if (!force && pending_gc_replacements_ < kGarbageCollectReplacementInterval) {
        return;
    }

    size_t previous_cache_size = 0;
    do {
        previous_cache_size = node_manager_.get_cache_size();
        node_manager_.garbage_collect();
    } while (node_manager_.get_cache_size() < previous_cache_size);

    pending_gc_replacements_ = 0;
}

/**
 * 执行单个门操作
 */
void QuantumCircuit::execute_gate(const GateParams& gate) {
    switch (gate.type) {
        // CPU端纯Qubit门
        case GateType::HADAMARD:
        case GateType::PAULI_X:
        case GateType::PAULI_Y:
        case GateType::PAULI_Z:
        case GateType::ROTATION_X:
        case GateType::ROTATION_Y:
        case GateType::ROTATION_Z:
        case GateType::PHASE_GATE_S:
        case GateType::PHASE_GATE_T:
        case GateType::CNOT:
        case GateType::CZ:
            synchronize_async_cv_pipeline();
            execute_qubit_gate(gate);
            break;

        // GPU端纯Qumode门
        case GateType::PHASE_ROTATION:
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
        case GateType::SNAP_GATE:
        case GateType::MULTI_SNAP_GATE:
        case GateType::CROSS_KERR_GATE:
            execute_level0_gate(gate);
            break;

        case GateType::CREATION_OPERATOR:
        case GateType::ANNIHILATION_OPERATOR:
            execute_level1_gate(gate);
            break;

        case GateType::DISPLACEMENT:
        case GateType::SQUEEZING:
            execute_level2_gate(gate);
            break;

        case GateType::BEAM_SPLITTER:
            execute_level3_gate(gate);
            break;

        // CPU+GPU混合门
        case GateType::CONDITIONAL_DISPLACEMENT:
        case GateType::CONDITIONAL_SQUEEZING:
        case GateType::CONDITIONAL_BEAM_SPLITTER:
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
        case GateType::CONDITIONAL_SUM:
        case GateType::RABI_INTERACTION:
        case GateType::JAYNES_CUMMINGS:
        case GateType::ANTI_JAYNES_CUMMINGS:
        case GateType::SELECTIVE_QUBIT_ROTATION:
            synchronize_async_cv_pipeline();
            execute_hybrid_gate(gate);
            break;

        default:
            throw std::runtime_error("不支持的门类型");
    }
}

/**
 * 执行Level 0门 (对角门)
 */
void QuantumCircuit::execute_level0_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_level0_gate");
    const auto& target_states = get_cached_target_states();

    if (target_states.empty()) return;

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    const bool use_async_compute =
        async_cv_pipeline_enabled_ &&
        (gate.type == GateType::PHASE_ROTATION ||
         gate.type == GateType::KERR_GATE ||
         gate.type == GateType::CONDITIONAL_PARITY);
    if (!use_async_compute && async_cv_pipeline_enabled_) {
        synchronize_async_cv_pipeline();
    }

    size_t upload_slot = 0;
    int* d_target_ids = nullptr;
    if (use_async_compute) {
        std::tie(d_target_ids, std::ignore) =
            upload_target_states_for_compute(target_states, &upload_slot);
    } else {
        d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
    }

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    double param = gate.params.empty() ? 0.0 : gate.params[0].real();
    const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];

    // 统计计算时延
    auto compute_start = std::chrono::high_resolution_clock::now();

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            apply_phase_rotation_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                         target_qumode, num_qumodes_,
                                         use_async_compute ? compute_stream_ : nullptr,
                                         !use_async_compute);
            break;
        case GateType::KERR_GATE:
            apply_kerr_gate_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                    target_qumode, num_qumodes_,
                                    use_async_compute ? compute_stream_ : nullptr,
                                    !use_async_compute);
            break;
        case GateType::CONDITIONAL_PARITY:
            apply_conditional_parity_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                             target_qumode, num_qumodes_,
                                             use_async_compute ? compute_stream_ : nullptr,
                                             !use_async_compute);
            break;
        case GateType::SNAP_GATE: {
            if (gate.params.size() < 2) {
                throw std::runtime_error("SNAP门缺少目标Fock态参数");
            }
            const int target_fock_state = static_cast<int>(std::llround(gate.params[1].real()));
            apply_snap_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                               target_fock_state, target_qumode, num_qumodes_);
            break;
        }
        case GateType::MULTI_SNAP_GATE: {
            std::vector<double> phase_map;
            phase_map.reserve(gate.params.size());
            for (const auto& phase : gate.params) {
                phase_map.push_back(phase.real());
            }
            apply_multisnap_on_mode(&state_pool_, d_target_ids, target_states.size(), phase_map,
                                    target_qumode, num_qumodes_);
            break;
        }
        case GateType::CROSS_KERR_GATE: {
            if (gate.target_qumodes.size() < 2) {
                throw std::runtime_error("Cross-Kerr门缺少两个目标qumode");
            }
            apply_ckgate_on_modes(&state_pool_, d_target_ids, target_states.size(), param,
                                  gate.target_qumodes[0], gate.target_qumodes[1], num_qumodes_);
            break;
        }
        default:
            throw std::runtime_error("未实现的Level0门类型");
    }

    // 检查GPU内核执行错误
    CHECK_CUDA(cudaGetLastError());
    if (use_async_compute) {
        mark_target_upload_slot_in_use(upload_slot);
    }

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行Level 1门 (梯算符门)
 */
void QuantumCircuit::execute_level1_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_level1_gate");
    const auto& target_states = get_cached_target_states();

    if (target_states.empty()) return;

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    size_t upload_slot = 0;
    int* d_target_ids = nullptr;
    if (async_cv_pipeline_enabled_) {
        std::tie(d_target_ids, std::ignore) =
            upload_target_states_for_compute(target_states, &upload_slot);
    } else {
        d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
    }

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    // 统计计算时延
    auto compute_start = std::chrono::high_resolution_clock::now();
    const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];

    switch (gate.type) {
        case GateType::CREATION_OPERATOR:
            apply_creation_operator_on_mode(&state_pool_, d_target_ids, target_states.size(),
                                            target_qumode, num_qumodes_,
                                            async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                            !async_cv_pipeline_enabled_);
            break;
        case GateType::ANNIHILATION_OPERATOR:
            apply_annihilation_operator_on_mode(&state_pool_, d_target_ids, target_states.size(),
                                                target_qumode, num_qumodes_,
                                                async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                                !async_cv_pipeline_enabled_);
            break;
        default:
            break;
    }

    // 检查GPU内核执行错误
    CHECK_CUDA(cudaGetLastError());
    if (async_cv_pipeline_enabled_) {
        mark_target_upload_slot_in_use(upload_slot);
    }

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行Level 2门 (单模门)
 */
void QuantumCircuit::execute_level2_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_level2_gate");
    const auto& target_states = get_cached_target_states();

    if (target_states.empty()) return;

    const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];
    const bool displacement_uses_direct_kernel =
        gate.type == GateType::DISPLACEMENT &&
        num_qumodes_ == 1 &&
        target_qumode == 0;
    const bool use_async_compute =
        async_cv_pipeline_enabled_ &&
        ((gate.type == GateType::SQUEEZING && !gate.params.empty()) ||
         (gate.type == GateType::DISPLACEMENT &&
          !gate.params.empty() &&
          displacement_uses_direct_kernel));
    const bool needs_target_upload =
        gate.type != GateType::DISPLACEMENT || displacement_uses_direct_kernel;
    if (!use_async_compute && async_cv_pipeline_enabled_) {
        synchronize_async_cv_pipeline();
    }

    auto transfer_start = std::chrono::high_resolution_clock::now();
    size_t upload_slot = 0;
    int* d_target_ids = nullptr;
    if (use_async_compute) {
        std::tie(d_target_ids, std::ignore) =
            upload_target_states_for_compute(target_states, &upload_slot);
    } else if (needs_target_upload) {
        d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
    }

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    if (gate.type == GateType::DISPLACEMENT && !gate.params.empty()) {
        cuDoubleComplex alpha = make_cuDoubleComplex(gate.params[0].real(), gate.params[0].imag());
        auto compute_start = std::chrono::high_resolution_clock::now();

        if (use_async_compute) {
            apply_displacement_gate(&state_pool_,
                                    d_target_ids,
                                    target_states.size(),
                                    alpha,
                                    compute_stream_,
                                    false);
        } else {
            apply_controlled_displacement_on_mode(
                &state_pool_, target_states, alpha, target_qumode, num_qumodes_);
        }

        CHECK_CUDA(cudaGetLastError());
        if (use_async_compute) {
            mark_target_upload_slot_in_use(upload_slot);
        }

        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    } else if (gate.type == GateType::SQUEEZING && !gate.params.empty()) {
        auto compute_start = std::chrono::high_resolution_clock::now();

        apply_squeezing_gate_gpu(&state_pool_,
                                 d_target_ids,
                                 static_cast<int>(target_states.size()),
                                 std::abs(gate.params[0]),
                                 std::arg(gate.params[0]),
                                 target_qumode,
                                 num_qumodes_,
                                 use_async_compute ? compute_stream_ : nullptr,
                                 !use_async_compute);

        CHECK_CUDA(cudaGetLastError());
        if (use_async_compute) {
            mark_target_upload_slot_in_use(upload_slot);
        }

        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    } else {
        // 使用ELL格式的通用实现
        FockELLOperator* ell_op = prepare_ell_operator(gate);
        if (ell_op && ell_op->ell_val && ell_op->ell_col && ell_op->dim > 0) {
            // 确保ELL算符已上传到GPU
            ell_op->upload_to_gpu();
            
            // 清除之前的CUDA错误
            cudaGetLastError();
            
            // 统计计算时延
            auto compute_start = std::chrono::high_resolution_clock::now();
            
            apply_single_mode_gate(&state_pool_, ell_op, d_target_ids, target_states.size(),
                                   nullptr, true);
            
            // 检查GPU内核执行错误
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                delete ell_op;
                throw std::runtime_error("GPU单模门执行失败: " + std::string(cudaGetErrorString(err)));
            }

            auto compute_end = std::chrono::high_resolution_clock::now();
            computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
            
            delete ell_op;
        } else {
            // ELL算符为空或无效
            std::cerr << "警告：单模门ELL算符无效，跳过执行" << std::endl;
            if (ell_op) delete ell_op;
        }
    }
}

/**
 * 执行Level 3门 (双模门)
 */
void QuantumCircuit::execute_level3_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_level3_gate");
    const auto& target_states = get_cached_target_states();

    if (target_states.empty()) return;

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    size_t upload_slot = 0;
    int* d_target_ids = nullptr;
    if (async_cv_pipeline_enabled_) {
        std::tie(d_target_ids, std::ignore) =
            upload_target_states_for_compute(target_states, &upload_slot);
    } else {
        d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
    }

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    if (gate.type == GateType::BEAM_SPLITTER && gate.params.size() >= 2) {
        double theta = gate.params[0].real();
        double phi = gate.params[1].real();
        const int target_qumode1 = gate.target_qumodes[0];
        const int target_qumode2 = gate.target_qumodes[1];

        // 统计计算时延
        auto compute_start = std::chrono::high_resolution_clock::now();

        apply_beam_splitter_recursive(&state_pool_, d_target_ids, static_cast<int>(target_states.size()),
                                      theta, phi, target_qumode1, target_qumode2, num_qumodes_,
                                      async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                      !async_cv_pipeline_enabled_);

        // 检查GPU内核执行错误
        CHECK_CUDA(cudaGetLastError());
        if (async_cv_pipeline_enabled_) {
            mark_target_upload_slot_in_use(upload_slot);
        }

        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    }
}

/**
 * 执行Level 4门 (混合控制门)
 */
void QuantumCircuit::execute_level4_gate(const GateParams& gate) {
    execute_hybrid_gate(gate);
}

/**
 * 执行Qubit门操作
 */
void QuantumCircuit::execute_qubit_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_qubit_gate");
    if (gate.target_qubits.empty()) {
        throw std::runtime_error("Qubit门需要指定目标Qubit");
    }

    int target_qubit = gate.target_qubits[0];
    if (target_qubit >= num_qubits_) {
        throw std::runtime_error("目标Qubit索引超出范围");
    }

    auto build_single_qubit_transform =
        [&](HDDNode* root,
            int single_target,
            const std::vector<std::complex<double>>& matrix) -> HDDNode* {
            std::unordered_map<HDDNode*, HDDNode*> memo;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = memo.find(node);
                    if (memo_it != memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == single_target) {
                        HDDNode* low = node->low;
                        HDDNode* high = node->high;
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            hdd_add(low, matrix[0] * node->w_low, high, matrix[1] * node->w_high),
                            hdd_add(low, matrix[2] * node->w_low, high, matrix[3] * node->w_high),
                            1.0,
                            1.0);
                    } else if (node->qubit_level > single_target) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    memo.emplace(node, transformed);
                    return transformed;
                };
            return transform(root);
        };

    auto build_monomial_single_qubit_transform =
        [&](HDDNode* root,
            int single_target,
            const std::vector<std::complex<double>>& matrix) -> HDDNode* {
            constexpr double kMatrixTolerance = 1e-14;
            const bool diagonal =
                std::abs(matrix[1]) < kMatrixTolerance &&
                std::abs(matrix[2]) < kMatrixTolerance;
            const bool anti_diagonal =
                std::abs(matrix[0]) < kMatrixTolerance &&
                std::abs(matrix[3]) < kMatrixTolerance;
            if (!diagonal && !anti_diagonal) {
                throw std::invalid_argument("monomial qubit transform requires diagonal or anti-diagonal matrix");
            }

            std::unordered_map<HDDNode*, HDDNode*> memo;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = memo.find(node);
                    if (memo_it != memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == single_target) {
                        if (diagonal) {
                            transformed = node_manager_.get_or_create_node(
                                node->qubit_level,
                                node->low,
                                node->high,
                                matrix[0] * node->w_low,
                                matrix[3] * node->w_high);
                        } else {
                            transformed = node_manager_.get_or_create_node(
                                node->qubit_level,
                                node->high,
                                node->low,
                                matrix[1] * node->w_high,
                                matrix[2] * node->w_low);
                        }
                    } else if (node->qubit_level > single_target) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    memo.emplace(node, transformed);
                    return transformed;
                };
            return transform(root);
        };

    // 构造对应的单比特门矩阵U
    std::vector<std::complex<double>> u(4, std::complex<double>(0.0, 0.0));
    bool use_monomial_single_qubit_transform = false;

    switch (gate.type) {
        case GateType::PAULI_X: {
            // X = [[0, 1], [1, 0]]
            u[1] = 1.0;  // (0,1)
            u[2] = 1.0;  // (1,0)
            use_monomial_single_qubit_transform = true;
            break;
        }
        case GateType::PAULI_Y: {
            // Y = [[0, -i], [i, 0]]
            u[1] = std::complex<double>(0.0, -1.0);  // (0,1)
            u[2] = std::complex<double>(0.0, 1.0);   // (1,0)
            use_monomial_single_qubit_transform = true;
            break;
        }
        case GateType::PAULI_Z: {
            // Z = [[1, 0], [0, -1]]
            u[0] = 1.0;   // (0,0)
            u[3] = -1.0;  // (1,1)
            use_monomial_single_qubit_transform = true;
            break;
        }
        case GateType::HADAMARD: {
            // H = 1/√2 * [[1, 1], [1, -1]]
            double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            u[0] = inv_sqrt2;  // (0,0)
            u[1] = inv_sqrt2;  // (0,1)
            u[2] = inv_sqrt2;  // (1,0)
            u[3] = -inv_sqrt2; // (1,1)
            break;
        }
        case GateType::ROTATION_X: {
            if (gate.params.empty()) {
                throw std::runtime_error("Rx门需要角度参数");
            }
            double theta = gate.params[0].real();
            double cos_half = std::cos(theta / 2.0);
            double sin_half = std::sin(theta / 2.0);
            // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
            u[0] = cos_half;                           // (0,0)
            u[1] = std::complex<double>(0.0, -sin_half); // (0,1)
            u[2] = std::complex<double>(0.0, -sin_half); // (1,0)
            u[3] = cos_half;                           // (1,1)
            break;
        }
        case GateType::ROTATION_Y: {
            if (gate.params.empty()) {
                throw std::runtime_error("Ry门需要角度参数");
            }
            double theta = gate.params[0].real();
            double cos_half = std::cos(theta / 2.0);
            double sin_half = std::sin(theta / 2.0);
            // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
            u[0] = cos_half;     // (0,0)
            u[1] = -sin_half;    // (0,1)
            u[2] = sin_half;     // (1,0)
            u[3] = cos_half;     // (1,1)
            break;
        }
        case GateType::ROTATION_Z: {
            if (gate.params.empty()) {
                throw std::runtime_error("Rz门需要角度参数");
            }
            double theta = gate.params[0].real();
            double cos_half = std::cos(theta / 2.0);
            double sin_half = std::sin(theta / 2.0);
            // Rz(θ) = [[cos(θ/2)-i*sin(θ/2), 0], [0, cos(θ/2)+i*sin(θ/2)]]
            // = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
            u[0] = std::complex<double>(cos_half, -sin_half); // (0,0)
            u[3] = std::complex<double>(cos_half, sin_half);  // (1,1)
            use_monomial_single_qubit_transform = true;
            break;
        }
        case GateType::PHASE_GATE_S: {
            // S = [[1, 0], [0, i]]
            u[0] = 1.0;  // (0,0)
            u[3] = std::complex<double>(0.0, 1.0); // (1,1)
            use_monomial_single_qubit_transform = true;
            break;
        }
        case GateType::PHASE_GATE_T: {
            // T = [[1, 0], [0, e^(iπ/4)]]
            u[0] = 1.0;  // (0,0)
            u[3] = std::complex<double>(std::cos(M_PI/4.0), std::sin(M_PI/4.0)); // (1,1)
            use_monomial_single_qubit_transform = true;
            break;
        }
        case GateType::CNOT: {
            if (gate.target_qubits.size() < 2) {
                throw std::runtime_error("CNOT门需要控制位和目标位");
            }
            const int control = gate.target_qubits[0];
            const int target = gate.target_qubits[1];
            const std::vector<std::complex<double>> px = {0.0, 1.0, 1.0, 0.0};

            std::unordered_map<HDDNode*, HDDNode*> control_memo;
            HDDNode* new_root = nullptr;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = control_memo.find(node);
                    if (memo_it != control_memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == control) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            node->low,
                            build_monomial_single_qubit_transform(node->high, target, px),
                            node->w_low,
                            node->w_high);
                    } else if (node->qubit_level > control) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    control_memo.emplace(node, transformed);
                    return transformed;
                };

            new_root = transform(root_node_);
            replace_root_node_preserving_terminals(new_root);
            return;
        }
        case GateType::CZ: {
            if (gate.target_qubits.size() < 2) {
                throw std::runtime_error("CZ门需要控制位和目标位");
            }
            const int control = gate.target_qubits[0];
            const int target = gate.target_qubits[1];
            const std::vector<std::complex<double>> pz = {1.0, 0.0, 0.0, -1.0};

            std::unordered_map<HDDNode*, HDDNode*> control_memo;
            HDDNode* new_root = nullptr;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = control_memo.find(node);
                    if (memo_it != control_memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == control) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            node->low,
                            build_monomial_single_qubit_transform(node->high, target, pz),
                            node->w_low,
                            node->w_high);
                    } else if (node->qubit_level > control) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    control_memo.emplace(node, transformed);
                    return transformed;
                };

            new_root = transform(root_node_);
            replace_root_node_preserving_terminals(new_root);
            return;
        }
        default:
            throw std::runtime_error("不支持的Qubit门类型");
    }

    HDDNode* new_root = use_monomial_single_qubit_transform
        ? build_monomial_single_qubit_transform(root_node_, target_qubit, u)
        : build_single_qubit_transform(root_node_, target_qubit, u);
    if (use_monomial_single_qubit_transform) {
        replace_root_node_preserving_terminals(new_root);
        return;
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    replace_root_node(new_root);
}

/**
 * 执行混合门操作 (CPU+GPU)
 */
void QuantumCircuit::execute_hybrid_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_hybrid_gate");
    // 混合门需要同时操作Qubit和Qumode
    // 根据gatemath.md，这些门分为分离型和混合型

    switch (gate.type) {
        // 分离型受控门
        case GateType::CONDITIONAL_DISPLACEMENT: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("CD门需要控制Qubit和目标Qumode");
            }
            // 实现CD门：当Qubit为|1⟩时应用Displacement
            execute_conditional_displacement(gate);
            break;
        }
        case GateType::CONDITIONAL_SQUEEZING: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("CS门需要控制Qubit和目标Qumode");
            }
            execute_conditional_squeezing(gate);
            break;
        }
        case GateType::CONDITIONAL_BEAM_SPLITTER: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 2) {
                throw std::runtime_error("CBS门需要控制Qubit和两个目标Qumode");
            }
            execute_conditional_beam_splitter(gate);
            break;
        }
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 2) {
                throw std::runtime_error("CTMS门需要控制Qubit和两个目标Qumode");
            }
            execute_conditional_two_mode_squeezing(gate);
            break;
        }
        case GateType::CONDITIONAL_SUM: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 2) {
                throw std::runtime_error("SUM门需要控制Qubit和两个目标Qumode");
            }
            execute_conditional_sum(gate);
            break;
        }

        // 混合型相互作用门
        case GateType::RABI_INTERACTION: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("RB门需要控制Qubit和目标Qumode");
            }
            execute_rabi_interaction(gate);
            break;
        }
        case GateType::JAYNES_CUMMINGS: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("JC门需要控制Qubit和目标Qumode");
            }
            execute_jaynes_cummings(gate);
            break;
        }
        case GateType::ANTI_JAYNES_CUMMINGS: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("AJC门需要控制Qubit和目标Qumode");
            }
            execute_anti_jaynes_cummings(gate);
            break;
        }
        case GateType::SELECTIVE_QUBIT_ROTATION: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("SQR门需要目标Qubit和控制Qumode");
            }
            execute_selective_qubit_rotation(gate);
            break;
        }

        default:
            throw std::runtime_error("未知的混合门类型: " + std::to_string(static_cast<int>(gate.type)));
    }
}

/**
 * 执行受控位移门 CD(α)
 * 按 σ_z 语义分支：|0⟩ 分支应用 D(+α)，|1⟩ 分支应用 D(-α)
 */
void QuantumCircuit::execute_conditional_displacement(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    std::complex<double> alpha = gate.params.empty() ? std::complex<double>(0.0, 0.0) : gate.params[0];

    std::unordered_map<NodeSignKey, HDDNode*, NodeSignKeyHash> memo;
    std::function<HDDNode*(HDDNode*, bool)> transform =
        [&](HDDNode* node, bool negated) -> HDDNode* {
            if (!node) {
                return nullptr;
            }

            const NodeSignKey key{node, negated};
            const auto memo_it = memo.find(key);
            if (memo_it != memo.end()) {
                return memo_it->second;
            }

            HDDNode* result = nullptr;
            if (node->is_terminal()) {
                if (std::abs(alpha) < 1e-14) {
                    result = node;
                } else {
                    const std::complex<double> effective_alpha = negated ? -alpha : alpha;
                    const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
                    if (duplicated_state < 0) {
                        throw std::runtime_error("条件位移失败：无法复制终端状态");
                    }

                    apply_displacement_to_state(duplicated_state, effective_alpha, target_qumode);
                    result = node_manager_.create_terminal_node(duplicated_state);
                }
            } else if (node->qubit_level == control_qubit) {
                HDDNode* low_branch = transform(node->low, negated);
                HDDNode* high_branch = transform(node->high, !negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, low_branch, high_branch, node->w_low, node->w_high);
            } else if (node->qubit_level > control_qubit) {
                HDDNode* new_low = transform(node->low, negated);
                HDDNode* new_high = transform(node->high, negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, new_low, new_high, node->w_low, node->w_high);
            } else {
                result = node;
            }

            memo.emplace(key, result);
            return result;
        };

    replace_root_node(transform(root_node_, false));
}

/**
 * 递归应用条件位移门
 */
HDDNode* QuantumCircuit::apply_conditional_displacement_recursive(
    HDDNode* node, int control_qubit, int target_qumode, std::complex<double> alpha) {
    if (node->is_terminal()) {
        if (std::abs(alpha) < 1e-14) {
            return node;
        }

        int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
        if (duplicated_state < 0) {
            throw std::runtime_error("条件位移失败：无法复制终端状态");
        }

        apply_displacement_to_state(duplicated_state, alpha, target_qumode);
        return node_manager_.create_terminal_node(duplicated_state);
    }

    if (node->qubit_level == control_qubit) {
        HDDNode* low_branch = apply_conditional_displacement_recursive(
            node->low, control_qubit, target_qumode, alpha);
        HDDNode* high_branch = apply_conditional_displacement_recursive(
            node->high, control_qubit, target_qumode, -alpha);

        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        // 递归到更深层级
        HDDNode* new_low = apply_conditional_displacement_recursive(
            node->low, control_qubit, target_qumode, alpha);
        HDDNode* new_high = apply_conditional_displacement_recursive(
            node->high, control_qubit, target_qumode, alpha);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        // 不需要处理的层级
        return node;
    }
}

/**
 * 对单个状态应用位移门
 */
void QuantumCircuit::apply_displacement_to_state(
    int state_id, std::complex<double> alpha, int target_qumode) {
    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.real(), alpha.imag());
    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    // 统计计算时延
    auto compute_start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> target_states{state_id};
    apply_controlled_displacement_on_mode(
        &state_pool_, target_states, alpha_cu, target_qumode, num_qumodes_);

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行受控挤压门 CS(ξ)
 * CS(ξ) = exp[σ_z ⊗ (ξ* a²/2 - ξ* (a†)²/2)]
 * 分离型：控制qubit为|0⟩时应用S(+ξ)，为|1⟩时应用S(-ξ)
 */
void QuantumCircuit::execute_conditional_squeezing(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    std::complex<double> xi = gate.params.empty() ? std::complex<double>(0.0, 0.0) : gate.params[0];

    std::unordered_map<NodeSignKey, HDDNode*, NodeSignKeyHash> memo;
    std::function<HDDNode*(HDDNode*, bool)> transform =
        [&](HDDNode* node, bool negated) -> HDDNode* {
            if (!node) {
                return nullptr;
            }

            const NodeSignKey key{node, negated};
            const auto memo_it = memo.find(key);
            if (memo_it != memo.end()) {
                return memo_it->second;
            }

            HDDNode* result = nullptr;
            if (node->is_terminal()) {
                if (std::abs(xi) < 1e-14) {
                    result = node;
                } else {
                    const std::complex<double> effective_xi = negated ? -xi : xi;
                    const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
                    if (duplicated_state < 0) {
                        throw std::runtime_error("条件挤压失败：无法复制终端状态");
                    }

                    apply_squeezing_to_state(duplicated_state, effective_xi, target_qumode);
                    result = node_manager_.create_terminal_node(duplicated_state);
                }
            } else if (node->qubit_level == control_qubit) {
                HDDNode* low_branch = transform(node->low, negated);
                HDDNode* high_branch = transform(node->high, !negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, low_branch, high_branch, node->w_low, node->w_high);
            } else if (node->qubit_level > control_qubit) {
                HDDNode* new_low = transform(node->low, negated);
                HDDNode* new_high = transform(node->high, negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, new_low, new_high, node->w_low, node->w_high);
            } else {
                result = node;
            }

            memo.emplace(key, result);
            return result;
        };

    replace_root_node(transform(root_node_, false));
}

/**
 * 递归应用条件挤压门
 */
HDDNode* QuantumCircuit::apply_conditional_squeezing_recursive(
    HDDNode* node, int control_qubit, int target_qumode, std::complex<double> xi) {

    if (node->is_terminal()) {
        if (std::abs(xi) < 1e-14) {
            return node;
        }

        const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
        if (duplicated_state < 0) {
            throw std::runtime_error("条件挤压失败：无法复制终端状态");
        }

        apply_squeezing_to_state(duplicated_state, xi, target_qumode);
        return node_manager_.create_terminal_node(duplicated_state);
    }

    if (node->qubit_level == control_qubit) {
        // 分支0 (|0⟩): S(+ξ)
        HDDNode* low_branch = apply_conditional_squeezing_recursive(
            node->low, control_qubit, target_qumode, xi);

        // 分支1 (|1⟩): S(-ξ)
        HDDNode* high_branch = apply_conditional_squeezing_recursive(
            node->high, control_qubit, target_qumode, -xi);

        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_conditional_squeezing_recursive(
            node->low, control_qubit, target_qumode, xi);
        HDDNode* new_high = apply_conditional_squeezing_recursive(
            node->high, control_qubit, target_qumode, xi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用挤压门
 */
void QuantumCircuit::apply_squeezing_to_state(int state_id, std::complex<double> xi, int target_qumode) {
    auto transfer_start = std::chrono::high_resolution_clock::now();

    int* d_state_id = state_pool_.upload_values_to_buffer(
        &state_id, 1, state_pool_.scratch_target_ids);

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    cudaGetLastError();

    auto compute_start = std::chrono::high_resolution_clock::now();
    apply_squeezing_gate_gpu(&state_pool_, d_state_id, 1, std::abs(xi), std::arg(xi),
                             target_qumode, num_qumodes_);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("GPU挤压门执行失败: " + std::string(cudaGetErrorString(err)));
    }

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行受控光束分裂器 CBS(θ,φ)
 * CBS(θ,φ) = exp[-i θ/2 σ_z ⊗ (e^{iφ} a† b + e^{-iφ} a b†)]
 * 分离型：控制qubit为|0⟩时应用BS(+θ,φ)，为|1⟩时应用BS(-θ,φ)
 */
void QuantumCircuit::execute_conditional_beam_splitter(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode1 = gate.target_qumodes[0];
    int target_qumode2 = gate.target_qumodes[1];
    double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;

    std::unordered_map<NodeSignKey, HDDNode*, NodeSignKeyHash> memo;
    std::function<HDDNode*(HDDNode*, bool)> transform =
        [&](HDDNode* node, bool negated) -> HDDNode* {
            if (!node) {
                return nullptr;
            }

            const NodeSignKey key{node, negated};
            const auto memo_it = memo.find(key);
            if (memo_it != memo.end()) {
                return memo_it->second;
            }

            HDDNode* result = nullptr;
            if (node->is_terminal()) {
                if (std::abs(theta) < 1e-14) {
                    result = node;
                } else {
                    const double effective_theta = negated ? -theta : theta;
                    const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
                    if (duplicated_state < 0) {
                        throw std::runtime_error("条件光束分裂失败：无法复制终端状态");
                    }

                    apply_beam_splitter_to_state(
                        duplicated_state, effective_theta, phi, target_qumode1, target_qumode2);
                    result = node_manager_.create_terminal_node(duplicated_state);
                }
            } else if (node->qubit_level == control_qubit) {
                HDDNode* low_branch = transform(node->low, negated);
                HDDNode* high_branch = transform(node->high, !negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, low_branch, high_branch, node->w_low, node->w_high);
            } else if (node->qubit_level > control_qubit) {
                HDDNode* new_low = transform(node->low, negated);
                HDDNode* new_high = transform(node->high, negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, new_low, new_high, node->w_low, node->w_high);
            } else {
                result = node;
            }

            memo.emplace(key, result);
            return result;
        };

    replace_root_node(transform(root_node_, false));
}

/**
 * 递归应用条件光束分裂器
 */
HDDNode* QuantumCircuit::apply_conditional_beam_splitter_recursive(
    HDDNode* node, int control_qubit, int qumode1, int qumode2, double theta, double phi) {

    if (node->is_terminal()) {
        if (std::abs(theta) < 1e-14) {
            return node;
        }

        const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
        if (duplicated_state < 0) {
            throw std::runtime_error("条件光束分裂失败：无法复制终端状态");
        }

        apply_beam_splitter_to_state(duplicated_state, theta, phi, qumode1, qumode2);
        return node_manager_.create_terminal_node(duplicated_state);
    }

    if (node->qubit_level == control_qubit) {
        // 分支0 (|0⟩): BS(+θ, φ)
        HDDNode* low_branch = apply_conditional_beam_splitter_recursive(
            node->low, control_qubit, qumode1, qumode2, theta, phi);

        // 分支1 (|1⟩): BS(-θ, φ)
        HDDNode* high_branch = apply_conditional_beam_splitter_recursive(
            node->high, control_qubit, qumode1, qumode2, -theta, phi);

        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_conditional_beam_splitter_recursive(
            node->low, control_qubit, qumode1, qumode2, theta, phi);
        HDDNode* new_high = apply_conditional_beam_splitter_recursive(
            node->high, control_qubit, qumode1, qumode2, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用光束分裂器
 */
void QuantumCircuit::apply_beam_splitter_to_state(int state_id, double theta, double phi,
                                                  int qumode1, int qumode2) {
    // 调用GPU光束分裂器内核
    int* d_state_id = state_pool_.upload_values_to_buffer(
        &state_id, 1, state_pool_.scratch_target_ids);

    apply_beam_splitter_recursive(&state_pool_, d_state_id, 1, theta, phi,
                                  qumode1, qumode2, num_qumodes_);
}

/**
 * 执行受控双模挤压 CTMS(ξ)
 * CTMS(ξ) = exp[1/2 σ_z ⊗ (ξ* a† b† - ξ a b)]
 * 分离型：控制qubit为|0⟩时应用TMS(+ξ)，为|1⟩时应用TMS(-ξ)
 */
void QuantumCircuit::execute_conditional_two_mode_squeezing(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode1 = gate.target_qumodes[0];
    int target_qumode2 = gate.target_qumodes[1];
    std::complex<double> xi = gate.params.empty() ? std::complex<double>(0.0, 0.0) : gate.params[0];

    std::unordered_map<NodeSignKey, HDDNode*, NodeSignKeyHash> memo;
    std::function<HDDNode*(HDDNode*, bool)> transform =
        [&](HDDNode* node, bool negated) -> HDDNode* {
            if (!node) {
                return nullptr;
            }

            const NodeSignKey key{node, negated};
            const auto memo_it = memo.find(key);
            if (memo_it != memo.end()) {
                return memo_it->second;
            }

            HDDNode* result = nullptr;
            if (node->is_terminal()) {
                if (std::abs(xi) < 1e-14) {
                    result = node;
                } else {
                    const std::complex<double> effective_xi = negated ? -xi : xi;
                    const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
                    if (duplicated_state < 0) {
                        throw std::runtime_error("条件双模挤压失败：无法复制终端状态");
                    }

                    apply_two_mode_squeezing_to_state(
                        duplicated_state, target_qumode1, target_qumode2, effective_xi);
                    result = node_manager_.create_terminal_node(duplicated_state);
                }
            } else if (node->qubit_level == control_qubit) {
                HDDNode* low_branch = transform(node->low, negated);
                HDDNode* high_branch = transform(node->high, !negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, low_branch, high_branch, node->w_low, node->w_high);
            } else if (node->qubit_level > control_qubit) {
                HDDNode* new_low = transform(node->low, negated);
                HDDNode* new_high = transform(node->high, negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, new_low, new_high, node->w_low, node->w_high);
            } else {
                result = node;
            }

            memo.emplace(key, result);
            return result;
        };

    replace_root_node(transform(root_node_, false));
}

/**
 * 递归应用条件双模挤压
 */
HDDNode* QuantumCircuit::apply_conditional_two_mode_squeezing_recursive(
    HDDNode* node, int control_qubit, int qumode1, int qumode2, std::complex<double> xi) {

    if (node->is_terminal()) {
        if (std::abs(xi) < 1e-14) {
            return node;
        }

        const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
        if (duplicated_state < 0) {
            throw std::runtime_error("条件双模挤压失败：无法复制终端状态");
        }

        apply_two_mode_squeezing_to_state(duplicated_state, qumode1, qumode2, xi);
        return node_manager_.create_terminal_node(duplicated_state);
    }

    if (node->qubit_level == control_qubit) {
        HDDNode* low_branch = apply_conditional_two_mode_squeezing_recursive(
            node->low, control_qubit, qumode1, qumode2, xi);
        HDDNode* high_branch = apply_conditional_two_mode_squeezing_recursive(
            node->high, control_qubit, qumode1, qumode2, -xi);
        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_conditional_two_mode_squeezing_recursive(
            node->low, control_qubit, qumode1, qumode2, xi);
        HDDNode* new_high = apply_conditional_two_mode_squeezing_recursive(
            node->high, control_qubit, qumode1, qumode2, xi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用双模挤压门
 */
void QuantumCircuit::apply_two_mode_squeezing_to_state(
    int state_id,
    int qumode1,
    int qumode2,
    std::complex<double> xi) {
    if (qumode1 == qumode2) {
        throw std::runtime_error("TMS需要两个不同的目标qumode");
    }

    auto transfer_start = std::chrono::high_resolution_clock::now();

    int* d_state_id = state_pool_.upload_values_to_buffer(
        &state_id, 1, state_pool_.scratch_target_ids);

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    auto compute_start = std::chrono::high_resolution_clock::now();

    apply_two_mode_squeezing_recursive(&state_pool_, d_state_id, 1, std::abs(xi), std::arg(xi),
                                       qumode1, qumode2, num_qumodes_);

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行受控SUM门
 */
void QuantumCircuit::execute_conditional_sum(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode1 = gate.target_qumodes[0];
    int target_qumode2 = gate.target_qumodes[1];
    double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;

    std::unordered_map<NodeSignKey, HDDNode*, NodeSignKeyHash> memo;
    std::function<HDDNode*(HDDNode*, bool)> transform =
        [&](HDDNode* node, bool negated) -> HDDNode* {
            if (!node) {
                return nullptr;
            }

            const NodeSignKey key{node, negated};
            const auto memo_it = memo.find(key);
            if (memo_it != memo.end()) {
                return memo_it->second;
            }

            HDDNode* result = nullptr;
            if (node->is_terminal()) {
                if (std::abs(theta) < 1e-14) {
                    result = node;
                } else {
                    const double effective_theta = negated ? -theta : theta;
                    const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
                    if (duplicated_state < 0) {
                        throw std::runtime_error("条件SUM失败：无法复制终端状态");
                    }

                    apply_sum_to_state(
                        duplicated_state, target_qumode1, target_qumode2, effective_theta, phi);
                    result = node_manager_.create_terminal_node(duplicated_state);
                }
            } else if (node->qubit_level == control_qubit) {
                HDDNode* low_branch = transform(node->low, negated);
                HDDNode* high_branch = transform(node->high, !negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, low_branch, high_branch, node->w_low, node->w_high);
            } else if (node->qubit_level > control_qubit) {
                HDDNode* new_low = transform(node->low, negated);
                HDDNode* new_high = transform(node->high, negated);
                result = node_manager_.get_or_create_node(
                    node->qubit_level, new_low, new_high, node->w_low, node->w_high);
            } else {
                result = node;
            }

            memo.emplace(key, result);
            return result;
        };

    replace_root_node(transform(root_node_, false));
}

/**
 * 递归应用条件SUM门
 */
HDDNode* QuantumCircuit::apply_conditional_sum_recursive(
    HDDNode* node, int control_qubit, int qumode1, int qumode2, double theta, double phi) {

    if (node->is_terminal()) {
        if (std::abs(theta) < 1e-14) {
            return node;
        }

        const int duplicated_state = state_pool_.duplicate_state(node->tensor_id);
        if (duplicated_state < 0) {
            throw std::runtime_error("条件SUM失败：无法复制终端状态");
        }

        apply_sum_to_state(duplicated_state, qumode1, qumode2, theta, phi);
        return node_manager_.create_terminal_node(duplicated_state);
    }

    if (node->qubit_level == control_qubit) {
        HDDNode* low_branch = apply_conditional_sum_recursive(
            node->low, control_qubit, qumode1, qumode2, theta, phi);
        HDDNode* high_branch = apply_conditional_sum_recursive(
            node->high, control_qubit, qumode1, qumode2, -theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_conditional_sum_recursive(
            node->low, control_qubit, qumode1, qumode2, theta, phi);
        HDDNode* new_high = apply_conditional_sum_recursive(
            node->high, control_qubit, qumode1, qumode2, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用SUM门
 */
void QuantumCircuit::apply_sum_to_state(
    int state_id,
    int qumode1,
    int qumode2,
    double theta,
    double phi) {
    if (std::abs(phi) > 1e-14) {
        throw std::runtime_error("当前SUM实现仅支持 phi = 0");
    }
    if (qumode1 == qumode2) {
        throw std::runtime_error("SUM需要两个不同的目标qumode");
    }

    auto transfer_start = std::chrono::high_resolution_clock::now();

    int* d_state_id = state_pool_.upload_values_to_buffer(
        &state_id, 1, state_pool_.scratch_target_ids);

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    auto compute_start = std::chrono::high_resolution_clock::now();

    apply_sum_gate(&state_pool_, d_state_id, 1, theta, cv_truncation_, cv_truncation_,
                   qumode1, qumode2, num_qumodes_);

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

// ===== 混合型相互作用门实现 =====

/**
 * 执行Rabi振荡门 RB(θ)
 * RB(θ) = exp[-i θ σ_x ⊗ (a + a†)]
 * 混合型：Qubit和Qumode相互作用
 */
void QuantumCircuit::execute_rabi_interaction(const GateParams& gate) {
    const int control_qubit = gate.target_qubits[0];
    const int target_qumode = gate.target_qumodes[0];
    const double theta = gate.params.empty() ? 0.0 : gate.params[0].real();
    reserve_pairwise_hybrid_headroom("Rabi", root_node_, control_qubit, state_pool_);

    std::vector<int> low_ids;
    std::vector<int> high_ids;
    std::unordered_set<int> replaced_state_ids;
    std::unordered_map<WeightedNodePairKey, std::pair<HDDNode*, HDDNode*>, WeightedNodePairKeyHash>
        pair_memo;
    std::unordered_map<HDDNode*, HDDNode*> node_memo;
    auto remaining_replaced_state_uses =
        collect_pairwise_replaced_state_use_counts(root_node_, control_qubit);
    const bool allow_early_replaced_state_release =
        collect_retained_state_ids_outside_pairwise_region(root_node_, control_qubit).empty();

    std::function<std::pair<HDDNode*, HDDNode*>(HDDNode*, std::complex<double>, HDDNode*, std::complex<double>)>
        build_pairs =
            [&](HDDNode* low_node,
                std::complex<double> low_weight,
                HDDNode* high_node,
                std::complex<double> high_weight) -> std::pair<HDDNode*, HDDNode*> {
                if (!low_node || !high_node) {
                    throw std::runtime_error("Rabi分支配对失败：存在空的HDD分支");
                }

                const WeightedNodePairKey key{low_node, high_node, low_weight, high_weight};
                const auto memo_it = pair_memo.find(key);
                if (memo_it != pair_memo.end()) {
                    return memo_it->second;
                }

                std::pair<HDDNode*, HDDNode*> result;
                if (low_node->is_terminal() && high_node->is_terminal()) {
                    HDDNode* low_copy = nullptr;
                    HDDNode* high_copy = nullptr;
                    try {
                        low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
                        low_ids.push_back(low_copy->tensor_id);
                        high_copy = duplicate_scaled_terminal_node(high_node, high_weight);
                        high_ids.push_back(high_copy->tensor_id);
                        if (low_node->tensor_id >= 0 && low_node->tensor_id != shared_zero_state_id_) {
                            replaced_state_ids.insert(low_node->tensor_id);
                        }
                        if (high_node->tensor_id >= 0 && high_node->tensor_id != shared_zero_state_id_) {
                            replaced_state_ids.insert(high_node->tensor_id);
                        }
                        release_pairwise_replaced_state_if_safe(
                            low_node->tensor_id,
                            allow_early_replaced_state_release,
                            remaining_replaced_state_uses,
                            state_pool_,
                            shared_zero_state_id_);
                        release_pairwise_replaced_state_if_safe(
                            high_node->tensor_id,
                            allow_early_replaced_state_release,
                            remaining_replaced_state_uses,
                            state_pool_,
                            shared_zero_state_id_);
                        result = {low_copy, high_copy};
                    } catch (...) {
                        if (high_copy) {
                            release_transient_pairwise_node(
                                node_manager_, state_pool_, high_copy, low_ids, high_ids);
                        }
                        if (low_copy) {
                            release_transient_pairwise_node(
                                node_manager_, state_pool_, low_copy, low_ids, high_ids);
                        }
                        throw;
                    }
                } else {
                    if (low_node->is_terminal() || high_node->is_terminal()) {
                        throw std::runtime_error("Rabi分支配对失败：低/高分支HDD结构不一致");
                    }
                    if (low_node->qubit_level != high_node->qubit_level) {
                        throw std::runtime_error("Rabi分支配对失败：低/高分支层级不一致");
                    }

                    const auto low_pair = build_pairs(
                        low_node->low,
                        low_weight * low_node->w_low,
                        high_node->low,
                        high_weight * high_node->w_low);
                    const auto high_pair = build_pairs(
                        low_node->high,
                        low_weight * low_node->w_high,
                        high_node->high,
                        high_weight * high_node->w_high);
                    result = {
                        node_manager_.get_or_create_node(
                            low_node->qubit_level, low_pair.first, high_pair.first, 1.0, 1.0),
                        node_manager_.get_or_create_node(
                            high_node->qubit_level, low_pair.second, high_pair.second, 1.0, 1.0)};
                }

                pair_memo.emplace(key, result);
                return result;
            };

    std::function<HDDNode*(HDDNode*)> transform =
        [&](HDDNode* node) -> HDDNode* {
            if (!node || node->is_terminal()) {
                return node;
            }

            const auto memo_it = node_memo.find(node);
            if (memo_it != node_memo.end()) {
                return memo_it->second;
            }

            HDDNode* transformed = nullptr;
            if (node->qubit_level == control_qubit) {
                const auto coupled = build_pairs(node->low, node->w_low, node->high, node->w_high);
                transformed = node_manager_.get_or_create_node(
                    node->qubit_level, coupled.first, coupled.second, 1.0, 1.0);
            } else if (node->qubit_level > control_qubit) {
                transformed = node_manager_.get_or_create_node(
                    node->qubit_level,
                    transform(node->low),
                    transform(node->high),
                    node->w_low,
                    node->w_high);
            } else {
                transformed = node;
            }

            node_memo.emplace(node, transformed);
            return transformed;
        };

    HDDNode* new_root = nullptr;
    std::unordered_set<int> retained_state_ids;
    try {
        new_root = transform(root_node_);
        const std::vector<int> retained_state_ids_vec = collect_terminal_state_ids(new_root);
        retained_state_ids.insert(retained_state_ids_vec.begin(), retained_state_ids_vec.end());
        if (!low_ids.empty()) {
            apply_rabi_interaction_on_mode(
                &state_pool_, low_ids, high_ids, theta, target_qumode, num_qumodes_);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    } catch (...) {
        cleanup_pairwise_build_failure(
            node_manager_, state_pool_, pair_memo, node_memo, low_ids, high_ids);
        throw;
    }
    replace_root_node_preserving_terminals(new_root);
    for (int state_id : replaced_state_ids) {
        if (retained_state_ids.find(state_id) == retained_state_ids.end() &&
            state_pool_.is_valid_state(state_id)) {
            state_pool_.free_state(state_id);
        }
    }
}

/**
 * 递归应用Rabi相互作用
 */
HDDNode* QuantumCircuit::apply_rabi_interaction_recursive(
    HDDNode* node, int control_qubit, int target_qumode, double theta) {
    if (node->is_terminal()) {
        return node;
    }

    if (node->qubit_level == control_qubit) {
        const auto coupled_branches =
            apply_rabi_pair_recursive(
                node->low, node->w_low, node->high, node->w_high, target_qumode, theta);
        return node_manager_.get_or_create_node(
            node->qubit_level, coupled_branches.first, coupled_branches.second, 1.0, 1.0);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_rabi_interaction_recursive(
            node->low, control_qubit, target_qumode, theta);
        HDDNode* new_high = apply_rabi_interaction_recursive(
            node->high, control_qubit, target_qumode, theta);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用Rabi相互作用
 */
void QuantumCircuit::apply_rabi_to_state(int state_id, double theta) {
    throw std::runtime_error(
        "Rabi相互作用需要在控制qubit的低/高分支上成对应用，单状态入口不应被直接调用: state_id=" +
        std::to_string(state_id) + ", theta=" + std::to_string(theta));
}

std::pair<HDDNode*, HDDNode*> QuantumCircuit::apply_rabi_pair_recursive(
    HDDNode* low_node,
    std::complex<double> low_weight,
    HDDNode* high_node,
    std::complex<double> high_weight,
    int target_qumode,
    double theta) {

    if (!low_node || !high_node) {
        throw std::runtime_error("Rabi分支配对失败：存在空的HDD分支");
    }

    if (low_node->is_terminal() && high_node->is_terminal()) {
        std::vector<int> low_ids;
        std::vector<int> high_ids;
        HDDNode* low_copy = nullptr;
        HDDNode* high_copy = nullptr;
        try {
            low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
            low_ids.push_back(low_copy->tensor_id);
            high_copy = duplicate_scaled_terminal_node(high_node, high_weight);
            high_ids.push_back(high_copy->tensor_id);

            apply_rabi_interaction_on_mode(
                &state_pool_, low_ids, high_ids, theta, target_qumode, num_qumodes_);
            CHECK_CUDA(cudaDeviceSynchronize());
            return {low_copy, high_copy};
        } catch (...) {
            if (high_copy) {
                release_transient_pairwise_node(
                    node_manager_, state_pool_, high_copy, low_ids, high_ids);
            }
            if (low_copy) {
                release_transient_pairwise_node(
                    node_manager_, state_pool_, low_copy, low_ids, high_ids);
            }
            cleanup_duplicated_pairwise_states(state_pool_, low_ids, high_ids);
            throw;
        }
    }

    if (low_node->is_terminal() || high_node->is_terminal()) {
        throw std::runtime_error("Rabi分支配对失败：低/高分支HDD结构不一致");
    }

    if (low_node->qubit_level != high_node->qubit_level) {
        throw std::runtime_error("Rabi分支配对失败：低/高分支层级不一致");
    }

    const auto low_pair = apply_rabi_pair_recursive(
        low_node->low,
        low_weight * low_node->w_low,
        high_node->low,
        high_weight * high_node->w_low,
        target_qumode,
        theta);

    const auto high_pair = apply_rabi_pair_recursive(
        low_node->high,
        low_weight * low_node->w_high,
        high_node->high,
        high_weight * high_node->w_high,
        target_qumode,
        theta);

    HDDNode* new_low = node_manager_.get_or_create_node(
        low_node->qubit_level, low_pair.first, high_pair.first, 1.0, 1.0);
    HDDNode* new_high = node_manager_.get_or_create_node(
        high_node->qubit_level, low_pair.second, high_pair.second, 1.0, 1.0);
    return {new_low, new_high};
}

std::pair<HDDNode*, HDDNode*> QuantumCircuit::apply_jc_like_pair_recursive(
    HDDNode* low_node,
    std::complex<double> low_weight,
    HDDNode* high_node,
    std::complex<double> high_weight,
    int target_qumode,
    double theta,
    double phi,
    bool anti_jaynes_cummings) {

    if (!low_node || !high_node) {
        throw std::runtime_error("JC/AJC分支配对失败：存在空的HDD分支");
    }

    if (low_node->is_terminal() && high_node->is_terminal()) {
        std::vector<int> low_ids;
        std::vector<int> high_ids;
        HDDNode* low_copy = nullptr;
        HDDNode* high_copy = nullptr;
        try {
            low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
            low_ids.push_back(low_copy->tensor_id);
            high_copy = duplicate_scaled_terminal_node(high_node, high_weight);
            high_ids.push_back(high_copy->tensor_id);

            if (anti_jaynes_cummings) {
                apply_anti_jaynes_cummings_on_mode(
                    &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
            } else {
                apply_jaynes_cummings_on_mode(
                    &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            return {low_copy, high_copy};
        } catch (...) {
            if (high_copy) {
                release_transient_pairwise_node(
                    node_manager_, state_pool_, high_copy, low_ids, high_ids);
            }
            if (low_copy) {
                release_transient_pairwise_node(
                    node_manager_, state_pool_, low_copy, low_ids, high_ids);
            }
            cleanup_duplicated_pairwise_states(state_pool_, low_ids, high_ids);
            throw;
        }
    }

    if (low_node->is_terminal() || high_node->is_terminal()) {
        throw std::runtime_error("JC/AJC分支配对失败：低/高分支HDD结构不一致");
    }

    if (low_node->qubit_level != high_node->qubit_level) {
        throw std::runtime_error("JC/AJC分支配对失败：低/高分支层级不一致");
    }

    const auto low_pair = apply_jc_like_pair_recursive(
        low_node->low,
        low_weight * low_node->w_low,
        high_node->low,
        high_weight * high_node->w_low,
        target_qumode,
        theta,
        phi,
        anti_jaynes_cummings);

    const auto high_pair = apply_jc_like_pair_recursive(
        low_node->high,
        low_weight * low_node->w_high,
        high_node->high,
        high_weight * high_node->w_high,
        target_qumode,
        theta,
        phi,
        anti_jaynes_cummings);

    HDDNode* new_low = node_manager_.get_or_create_node(
        low_node->qubit_level, low_pair.first, high_pair.first, 1.0, 1.0);
    HDDNode* new_high = node_manager_.get_or_create_node(
        high_node->qubit_level, low_pair.second, high_pair.second, 1.0, 1.0);
    return {new_low, new_high};
}

/**
 * 执行Jaynes-Cummings相互作用 JC(θ,φ)
 * JC(θ,φ) = exp[-iθ(e^{iφ} σ- a† + e^{-iφ} σ+ a)]
 */
void QuantumCircuit::execute_jaynes_cummings(const GateParams& gate) {
    const int control_qubit = gate.target_qubits[0];
    const int target_qumode = gate.target_qumodes[0];
    const double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    const double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;
    reserve_pairwise_hybrid_headroom("JC", root_node_, control_qubit, state_pool_);

    std::vector<int> low_ids;
    std::vector<int> high_ids;
    std::unordered_set<int> replaced_state_ids;
    std::unordered_map<WeightedNodePairKey, std::pair<HDDNode*, HDDNode*>, WeightedNodePairKeyHash>
        pair_memo;
    std::unordered_map<HDDNode*, HDDNode*> node_memo;
    auto remaining_replaced_state_uses =
        collect_pairwise_replaced_state_use_counts(root_node_, control_qubit);
    const bool allow_early_replaced_state_release =
        collect_retained_state_ids_outside_pairwise_region(root_node_, control_qubit).empty();

    std::function<std::pair<HDDNode*, HDDNode*>(HDDNode*, std::complex<double>, HDDNode*, std::complex<double>)>
        build_pairs =
            [&](HDDNode* low_node,
                std::complex<double> low_weight,
                HDDNode* high_node,
                std::complex<double> high_weight) -> std::pair<HDDNode*, HDDNode*> {
                if (!low_node || !high_node) {
                    throw std::runtime_error("JC分支配对失败：存在空的HDD分支");
                }

                const WeightedNodePairKey key{low_node, high_node, low_weight, high_weight};
                const auto memo_it = pair_memo.find(key);
                if (memo_it != pair_memo.end()) {
                    return memo_it->second;
                }

                std::pair<HDDNode*, HDDNode*> result;
                if (low_node->is_terminal() && high_node->is_terminal()) {
                    HDDNode* low_copy = nullptr;
                    HDDNode* high_copy = nullptr;
                    try {
                        low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
                        low_ids.push_back(low_copy->tensor_id);
                        high_copy = duplicate_scaled_terminal_node(high_node, high_weight);
                        high_ids.push_back(high_copy->tensor_id);
                        if (low_node->tensor_id >= 0 && low_node->tensor_id != shared_zero_state_id_) {
                            replaced_state_ids.insert(low_node->tensor_id);
                        }
                        if (high_node->tensor_id >= 0 && high_node->tensor_id != shared_zero_state_id_) {
                            replaced_state_ids.insert(high_node->tensor_id);
                        }
                        release_pairwise_replaced_state_if_safe(
                            low_node->tensor_id,
                            allow_early_replaced_state_release,
                            remaining_replaced_state_uses,
                            state_pool_,
                            shared_zero_state_id_);
                        release_pairwise_replaced_state_if_safe(
                            high_node->tensor_id,
                            allow_early_replaced_state_release,
                            remaining_replaced_state_uses,
                            state_pool_,
                            shared_zero_state_id_);
                        result = {low_copy, high_copy};
                    } catch (...) {
                        if (high_copy) {
                            release_transient_pairwise_node(
                                node_manager_, state_pool_, high_copy, low_ids, high_ids);
                        }
                        if (low_copy) {
                            release_transient_pairwise_node(
                                node_manager_, state_pool_, low_copy, low_ids, high_ids);
                        }
                        throw;
                    }
                } else {
                    if (low_node->is_terminal() || high_node->is_terminal()) {
                        throw std::runtime_error("JC分支配对失败：低/高分支HDD结构不一致");
                    }
                    if (low_node->qubit_level != high_node->qubit_level) {
                        throw std::runtime_error("JC分支配对失败：低/高分支层级不一致");
                    }

                    const auto low_pair = build_pairs(
                        low_node->low,
                        low_weight * low_node->w_low,
                        high_node->low,
                        high_weight * high_node->w_low);
                    const auto high_pair = build_pairs(
                        low_node->high,
                        low_weight * low_node->w_high,
                        high_node->high,
                        high_weight * high_node->w_high);
                    result = {
                        node_manager_.get_or_create_node(
                            low_node->qubit_level, low_pair.first, high_pair.first, 1.0, 1.0),
                        node_manager_.get_or_create_node(
                            high_node->qubit_level, low_pair.second, high_pair.second, 1.0, 1.0)};
                }

                pair_memo.emplace(key, result);
                return result;
            };

    std::function<HDDNode*(HDDNode*)> transform =
        [&](HDDNode* node) -> HDDNode* {
            if (!node || node->is_terminal()) {
                return node;
            }

            const auto memo_it = node_memo.find(node);
            if (memo_it != node_memo.end()) {
                return memo_it->second;
            }

            HDDNode* transformed = nullptr;
            if (node->qubit_level == control_qubit) {
                const auto coupled = build_pairs(node->low, node->w_low, node->high, node->w_high);
                transformed = node_manager_.get_or_create_node(
                    node->qubit_level, coupled.first, coupled.second, 1.0, 1.0);
            } else if (node->qubit_level > control_qubit) {
                transformed = node_manager_.get_or_create_node(
                    node->qubit_level,
                    transform(node->low),
                    transform(node->high),
                    node->w_low,
                    node->w_high);
            } else {
                transformed = node;
            }

            node_memo.emplace(node, transformed);
            return transformed;
        };

    HDDNode* new_root = nullptr;
    std::unordered_set<int> retained_state_ids;
    try {
        new_root = transform(root_node_);
        const std::vector<int> retained_state_ids_vec = collect_terminal_state_ids(new_root);
        retained_state_ids.insert(retained_state_ids_vec.begin(), retained_state_ids_vec.end());
        FALLBACK_DEBUG_LOG << "[fallback] JC prepared pairs=" << low_ids.size()
                           << " replaced_state_ids=" << replaced_state_ids.size() << std::endl;
        if (!low_ids.empty()) {
            FALLBACK_DEBUG_LOG << "[fallback] JC launching paired kernel" << std::endl;
            apply_jaynes_cummings_on_mode(
                &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    } catch (...) {
        cleanup_pairwise_build_failure(
            node_manager_, state_pool_, pair_memo, node_memo, low_ids, high_ids);
        throw;
    }
    replace_root_node_preserving_terminals(new_root);
    for (int state_id : replaced_state_ids) {
        if (retained_state_ids.find(state_id) == retained_state_ids.end() &&
            state_pool_.is_valid_state(state_id)) {
            state_pool_.free_state(state_id);
        }
    }
    FALLBACK_DEBUG_LOG << "[fallback] JC old states released" << std::endl;
}

/**
 * 递归应用Jaynes-Cummings相互作用
 */
HDDNode* QuantumCircuit::apply_jaynes_cummings_recursive(
    HDDNode* node, int control_qubit, int target_qumode, double theta, double phi) {
    if (node->is_terminal()) {
        return node;
    }

    if (node->qubit_level == control_qubit) {
        const auto coupled_branches = apply_jc_like_pair_recursive(
            node->low, node->w_low, node->high, node->w_high, target_qumode, theta, phi, false);
        return node_manager_.get_or_create_node(
            node->qubit_level, coupled_branches.first, coupled_branches.second, 1.0, 1.0);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_jaynes_cummings_recursive(
            node->low, control_qubit, target_qumode, theta, phi);
        HDDNode* new_high = apply_jaynes_cummings_recursive(
            node->high, control_qubit, target_qumode, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用Jaynes-Cummings相互作用
 */
void QuantumCircuit::apply_jaynes_cummings_to_state(int state_id, double theta, double phi) {
    throw std::runtime_error(
        "JC相互作用需要在控制qubit的低/高分支上成对应用，单状态入口不应被直接调用: state_id=" +
        std::to_string(state_id) + ", theta=" + std::to_string(theta) + ", phi=" + std::to_string(phi));
}

/**
 * 执行Anti-Jaynes-Cummings相互作用 AJC(θ,φ)
 */
void QuantumCircuit::execute_anti_jaynes_cummings(const GateParams& gate) {
    const int control_qubit = gate.target_qubits[0];
    const int target_qumode = gate.target_qumodes[0];
    const double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    const double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;
    reserve_pairwise_hybrid_headroom("AJC", root_node_, control_qubit, state_pool_);

    std::vector<int> low_ids;
    std::vector<int> high_ids;
    std::unordered_set<int> replaced_state_ids;
    std::unordered_map<WeightedNodePairKey, std::pair<HDDNode*, HDDNode*>, WeightedNodePairKeyHash>
        pair_memo;
    std::unordered_map<HDDNode*, HDDNode*> node_memo;
    auto remaining_replaced_state_uses =
        collect_pairwise_replaced_state_use_counts(root_node_, control_qubit);
    const bool allow_early_replaced_state_release =
        collect_retained_state_ids_outside_pairwise_region(root_node_, control_qubit).empty();

    std::function<std::pair<HDDNode*, HDDNode*>(HDDNode*, std::complex<double>, HDDNode*, std::complex<double>)>
        build_pairs =
            [&](HDDNode* low_node,
                std::complex<double> low_weight,
                HDDNode* high_node,
                std::complex<double> high_weight) -> std::pair<HDDNode*, HDDNode*> {
                if (!low_node || !high_node) {
                    throw std::runtime_error("AJC分支配对失败：存在空的HDD分支");
                }

                const WeightedNodePairKey key{low_node, high_node, low_weight, high_weight};
                const auto memo_it = pair_memo.find(key);
                if (memo_it != pair_memo.end()) {
                    return memo_it->second;
                }

                std::pair<HDDNode*, HDDNode*> result;
                if (low_node->is_terminal() && high_node->is_terminal()) {
                    HDDNode* low_copy = nullptr;
                    HDDNode* high_copy = nullptr;
                    try {
                        low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
                        low_ids.push_back(low_copy->tensor_id);
                        high_copy = duplicate_scaled_terminal_node(high_node, high_weight);
                        high_ids.push_back(high_copy->tensor_id);
                        if (low_node->tensor_id >= 0 && low_node->tensor_id != shared_zero_state_id_) {
                            replaced_state_ids.insert(low_node->tensor_id);
                        }
                        if (high_node->tensor_id >= 0 && high_node->tensor_id != shared_zero_state_id_) {
                            replaced_state_ids.insert(high_node->tensor_id);
                        }
                        release_pairwise_replaced_state_if_safe(
                            low_node->tensor_id,
                            allow_early_replaced_state_release,
                            remaining_replaced_state_uses,
                            state_pool_,
                            shared_zero_state_id_);
                        release_pairwise_replaced_state_if_safe(
                            high_node->tensor_id,
                            allow_early_replaced_state_release,
                            remaining_replaced_state_uses,
                            state_pool_,
                            shared_zero_state_id_);
                        result = {low_copy, high_copy};
                    } catch (...) {
                        if (high_copy) {
                            release_transient_pairwise_node(
                                node_manager_, state_pool_, high_copy, low_ids, high_ids);
                        }
                        if (low_copy) {
                            release_transient_pairwise_node(
                                node_manager_, state_pool_, low_copy, low_ids, high_ids);
                        }
                        throw;
                    }
                } else {
                    if (low_node->is_terminal() || high_node->is_terminal()) {
                        throw std::runtime_error("AJC分支配对失败：低/高分支HDD结构不一致");
                    }
                    if (low_node->qubit_level != high_node->qubit_level) {
                        throw std::runtime_error("AJC分支配对失败：低/高分支层级不一致");
                    }

                    const auto low_pair = build_pairs(
                        low_node->low,
                        low_weight * low_node->w_low,
                        high_node->low,
                        high_weight * high_node->w_low);
                    const auto high_pair = build_pairs(
                        low_node->high,
                        low_weight * low_node->w_high,
                        high_node->high,
                        high_weight * high_node->w_high);
                    result = {
                        node_manager_.get_or_create_node(
                            low_node->qubit_level, low_pair.first, high_pair.first, 1.0, 1.0),
                        node_manager_.get_or_create_node(
                            high_node->qubit_level, low_pair.second, high_pair.second, 1.0, 1.0)};
                }

                pair_memo.emplace(key, result);
                return result;
            };

    std::function<HDDNode*(HDDNode*)> transform =
        [&](HDDNode* node) -> HDDNode* {
            if (!node || node->is_terminal()) {
                return node;
            }

            const auto memo_it = node_memo.find(node);
            if (memo_it != node_memo.end()) {
                return memo_it->second;
            }

            HDDNode* transformed = nullptr;
            if (node->qubit_level == control_qubit) {
                const auto coupled = build_pairs(node->low, node->w_low, node->high, node->w_high);
                transformed = node_manager_.get_or_create_node(
                    node->qubit_level, coupled.first, coupled.second, 1.0, 1.0);
            } else if (node->qubit_level > control_qubit) {
                transformed = node_manager_.get_or_create_node(
                    node->qubit_level,
                    transform(node->low),
                    transform(node->high),
                    node->w_low,
                    node->w_high);
            } else {
                transformed = node;
            }

            node_memo.emplace(node, transformed);
            return transformed;
        };

    HDDNode* new_root = nullptr;
    std::unordered_set<int> retained_state_ids;
    try {
        new_root = transform(root_node_);
        const std::vector<int> retained_state_ids_vec = collect_terminal_state_ids(new_root);
        retained_state_ids.insert(retained_state_ids_vec.begin(), retained_state_ids_vec.end());
        if (!low_ids.empty()) {
            apply_anti_jaynes_cummings_on_mode(
                &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    } catch (...) {
        cleanup_pairwise_build_failure(
            node_manager_, state_pool_, pair_memo, node_memo, low_ids, high_ids);
        throw;
    }
    replace_root_node_preserving_terminals(new_root);
    for (int state_id : replaced_state_ids) {
        if (retained_state_ids.find(state_id) == retained_state_ids.end() &&
            state_pool_.is_valid_state(state_id)) {
            state_pool_.free_state(state_id);
        }
    }
}

/**
 * 递归应用Anti-Jaynes-Cummings相互作用
 */
HDDNode* QuantumCircuit::apply_anti_jaynes_cummings_recursive(
    HDDNode* node, int control_qubit, int target_qumode, double theta, double phi) {
    if (node->is_terminal()) {
        return node;
    }

    if (node->qubit_level == control_qubit) {
        const auto coupled_branches = apply_jc_like_pair_recursive(
            node->low, node->w_low, node->high, node->w_high, target_qumode, theta, phi, true);
        return node_manager_.get_or_create_node(
            node->qubit_level, coupled_branches.first, coupled_branches.second, 1.0, 1.0);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_anti_jaynes_cummings_recursive(
            node->low, control_qubit, target_qumode, theta, phi);
        HDDNode* new_high = apply_anti_jaynes_cummings_recursive(
            node->high, control_qubit, target_qumode, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用Anti-Jaynes-Cummings相互作用
 */
void QuantumCircuit::apply_anti_jaynes_cummings_to_state(int state_id, double theta, double phi) {
    throw std::runtime_error(
        "AJC相互作用需要在控制qubit的低/高分支上成对应用，单状态入口不应被直接调用: state_id=" +
        std::to_string(state_id) + ", theta=" + std::to_string(theta) + ", phi=" + std::to_string(phi));
}

/**
 * 执行选择性Qubit旋转 SQR(θ,φ)
 */
void QuantumCircuit::execute_selective_qubit_rotation(const GateParams& gate) {
    const int target_qubit = gate.target_qubits[0];
    const int control_qumode = gate.target_qumodes[0];
    std::vector<double> theta_vec, phi_vec;

    // 从参数中提取θ和φ向量
    size_t num_params = gate.params.size();
    for (size_t i = 0; i < num_params / 2; ++i) {
        theta_vec.push_back(gate.params[2 * i].real());
        phi_vec.push_back(gate.params[2 * i + 1].real());
    }

    if (theta_vec.empty() || theta_vec.size() != phi_vec.size()) {
        throw std::runtime_error("SQR需要成对提供每个Fock层级的(theta, phi)参数");
    }

    const std::vector<double> expanded_thetas =
        expand_selective_rotation_profile(
            theta_vec,
            state_pool_.d_trunc,
            control_qumode,
            num_qumodes_,
            state_pool_.max_total_dim);
    const std::vector<double> expanded_phis =
        expand_selective_rotation_profile(
            phi_vec,
            state_pool_.d_trunc,
            control_qumode,
            num_qumodes_,
            state_pool_.max_total_dim);
    reserve_pairwise_hybrid_headroom("SQR", root_node_, target_qubit, state_pool_);

    std::vector<int> low_ids;
    std::vector<int> high_ids;
    std::unordered_set<int> replaced_state_ids;
    std::unordered_map<WeightedNodePairKey, std::pair<HDDNode*, HDDNode*>, WeightedNodePairKeyHash>
        pair_memo;
    std::unordered_map<HDDNode*, HDDNode*> node_memo;
    auto remaining_replaced_state_uses =
        collect_pairwise_replaced_state_use_counts(root_node_, target_qubit);
    const bool allow_early_replaced_state_release =
        collect_retained_state_ids_outside_pairwise_region(root_node_, target_qubit).empty();

    std::function<std::pair<HDDNode*, HDDNode*>(HDDNode*, std::complex<double>, HDDNode*, std::complex<double>)>
        build_pairs =
            [&](HDDNode* low_node,
                std::complex<double> low_weight,
                HDDNode* high_node,
                std::complex<double> high_weight) -> std::pair<HDDNode*, HDDNode*> {
                if (!low_node || !high_node) {
                    throw std::runtime_error("SQR分支配对失败：存在空的HDD分支");
                }

                const WeightedNodePairKey key{low_node, high_node, low_weight, high_weight};
                const auto memo_it = pair_memo.find(key);
                if (memo_it != pair_memo.end()) {
                    return memo_it->second;
                }

                std::pair<HDDNode*, HDDNode*> result;
                if (low_node->is_terminal() && high_node->is_terminal()) {
                    HDDNode* low_copy = nullptr;
                    HDDNode* high_copy = nullptr;
                    try {
                        low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
                        low_ids.push_back(low_copy->tensor_id);
                        high_copy = duplicate_scaled_terminal_node(high_node, high_weight);
                        high_ids.push_back(high_copy->tensor_id);
                        if (low_node->tensor_id >= 0 && low_node->tensor_id != shared_zero_state_id_) {
                            replaced_state_ids.insert(low_node->tensor_id);
                        }
                        if (high_node->tensor_id >= 0 && high_node->tensor_id != shared_zero_state_id_) {
                            replaced_state_ids.insert(high_node->tensor_id);
                        }
                        release_pairwise_replaced_state_if_safe(
                            low_node->tensor_id,
                            allow_early_replaced_state_release,
                            remaining_replaced_state_uses,
                            state_pool_,
                            shared_zero_state_id_);
                        release_pairwise_replaced_state_if_safe(
                            high_node->tensor_id,
                            allow_early_replaced_state_release,
                            remaining_replaced_state_uses,
                            state_pool_,
                            shared_zero_state_id_);
                        result = {low_copy, high_copy};
                    } catch (...) {
                        if (high_copy) {
                            release_transient_pairwise_node(
                                node_manager_, state_pool_, high_copy, low_ids, high_ids);
                        }
                        if (low_copy) {
                            release_transient_pairwise_node(
                                node_manager_, state_pool_, low_copy, low_ids, high_ids);
                        }
                        throw;
                    }
                } else {
                    if (low_node->is_terminal() || high_node->is_terminal()) {
                        throw std::runtime_error("SQR分支配对失败：低/高分支HDD结构不一致");
                    }
                    if (low_node->qubit_level != high_node->qubit_level) {
                        throw std::runtime_error("SQR分支配对失败：低/高分支层级不一致");
                    }

                    const auto low_pair = build_pairs(
                        low_node->low,
                        low_weight * low_node->w_low,
                        high_node->low,
                        high_weight * high_node->w_low);
                    const auto high_pair = build_pairs(
                        low_node->high,
                        low_weight * low_node->w_high,
                        high_node->high,
                        high_weight * high_node->w_high);
                    result = {
                        node_manager_.get_or_create_node(
                            low_node->qubit_level, low_pair.first, high_pair.first, 1.0, 1.0),
                        node_manager_.get_or_create_node(
                            high_node->qubit_level, low_pair.second, high_pair.second, 1.0, 1.0)};
                }

                pair_memo.emplace(key, result);
                return result;
            };

    std::function<HDDNode*(HDDNode*)> transform =
        [&](HDDNode* node) -> HDDNode* {
            if (!node || node->is_terminal()) {
                return node;
            }

            const auto memo_it = node_memo.find(node);
            if (memo_it != node_memo.end()) {
                return memo_it->second;
            }

            HDDNode* transformed = nullptr;
            if (node->qubit_level == target_qubit) {
                const auto rotated = build_pairs(node->low, node->w_low, node->high, node->w_high);
                transformed = node_manager_.get_or_create_node(
                    node->qubit_level, rotated.first, rotated.second, 1.0, 1.0);
            } else if (node->qubit_level > target_qubit) {
                transformed = node_manager_.get_or_create_node(
                    node->qubit_level,
                    transform(node->low),
                    transform(node->high),
                    node->w_low,
                    node->w_high);
            } else {
                transformed = node;
            }

            node_memo.emplace(node, transformed);
            return transformed;
        };

    HDDNode* new_root = nullptr;
    std::unordered_set<int> retained_state_ids;
    try {
        new_root = transform(root_node_);
        const std::vector<int> retained_state_ids_vec = collect_terminal_state_ids(new_root);
        retained_state_ids.insert(retained_state_ids_vec.begin(), retained_state_ids_vec.end());
        if (!low_ids.empty()) {
            apply_sqr(&state_pool_, low_ids, high_ids, expanded_thetas, expanded_phis);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    } catch (...) {
        cleanup_pairwise_build_failure(
            node_manager_, state_pool_, pair_memo, node_memo, low_ids, high_ids);
        throw;
    }
    replace_root_node_preserving_terminals(new_root);
    for (int state_id : replaced_state_ids) {
        if (retained_state_ids.find(state_id) == retained_state_ids.end() &&
            state_pool_.is_valid_state(state_id)) {
            state_pool_.free_state(state_id);
        }
    }
}

/**
 * 递归应用选择性Qubit旋转
 */
HDDNode* QuantumCircuit::apply_selective_qubit_rotation_recursive(
    HDDNode* node, int target_qubit, int control_qumode,
    const std::vector<double>& theta_vec, const std::vector<double>& phi_vec) {
    if (node->is_terminal()) {
        return node;
    }

    if (node->qubit_level == target_qubit) {
        const auto rotated_branches = apply_selective_qubit_rotation_pair_recursive(
            node->low,
            node->w_low,
            node->high,
            node->w_high,
            control_qumode,
            theta_vec,
            phi_vec);
        return node_manager_.get_or_create_node(
            node->qubit_level, rotated_branches.first, rotated_branches.second, 1.0, 1.0);
    } else if (node->qubit_level > target_qubit) {
        HDDNode* new_low = apply_selective_qubit_rotation_recursive(
            node->low, target_qubit, control_qumode, theta_vec, phi_vec);
        HDDNode* new_high = apply_selective_qubit_rotation_recursive(
            node->high, target_qubit, control_qumode, theta_vec, phi_vec);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

std::pair<HDDNode*, HDDNode*> QuantumCircuit::apply_selective_qubit_rotation_pair_recursive(
    HDDNode* low_node,
    std::complex<double> low_weight,
    HDDNode* high_node,
    std::complex<double> high_weight,
    int control_qumode,
    const std::vector<double>& theta_vec,
    const std::vector<double>& phi_vec) {
    if (!low_node || !high_node) {
        throw std::runtime_error("SQR分支配对失败：存在空的HDD分支");
    }

    if (low_node->is_terminal() && high_node->is_terminal()) {
        std::vector<int> low_ids;
        std::vector<int> high_ids;
        HDDNode* low_copy = nullptr;
        HDDNode* high_copy = nullptr;
        try {
            low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
            low_ids.push_back(low_copy->tensor_id);
            high_copy = duplicate_scaled_terminal_node(high_node, high_weight);
            high_ids.push_back(high_copy->tensor_id);

            const std::vector<double> expanded_thetas =
                expand_selective_rotation_profile(
                    theta_vec,
                    state_pool_.d_trunc,
                    control_qumode,
                    num_qumodes_,
                    state_pool_.max_total_dim);
            const std::vector<double> expanded_phis =
                expand_selective_rotation_profile(
                    phi_vec,
                    state_pool_.d_trunc,
                    control_qumode,
                    num_qumodes_,
                    state_pool_.max_total_dim);

            apply_sqr(
                &state_pool_,
                low_ids,
                high_ids,
                expanded_thetas,
                expanded_phis);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            return {low_copy, high_copy};
        } catch (...) {
            if (high_copy) {
                release_transient_pairwise_node(
                    node_manager_, state_pool_, high_copy, low_ids, high_ids);
            }
            if (low_copy) {
                release_transient_pairwise_node(
                    node_manager_, state_pool_, low_copy, low_ids, high_ids);
            }
            cleanup_duplicated_pairwise_states(state_pool_, low_ids, high_ids);
            throw;
        }
    }

    if (low_node->is_terminal() || high_node->is_terminal()) {
        throw std::runtime_error("SQR分支配对失败：低/高分支HDD结构不一致");
    }

    if (low_node->qubit_level != high_node->qubit_level) {
        throw std::runtime_error("SQR分支配对失败：低/高分支层级不一致");
    }

    const auto low_pair = apply_selective_qubit_rotation_pair_recursive(
        low_node->low,
        low_weight * low_node->w_low,
        high_node->low,
        high_weight * high_node->w_low,
        control_qumode,
        theta_vec,
        phi_vec);

    const auto high_pair = apply_selective_qubit_rotation_pair_recursive(
        low_node->high,
        low_weight * low_node->w_high,
        high_node->high,
        high_weight * high_node->w_high,
        control_qumode,
        theta_vec,
        phi_vec);

    HDDNode* new_low = node_manager_.get_or_create_node(
        low_node->qubit_level, low_pair.first, high_pair.first, 1.0, 1.0);
    HDDNode* new_high = node_manager_.get_or_create_node(
        high_node->qubit_level, low_pair.second, high_pair.second, 1.0, 1.0);
    return {new_low, new_high};
}

/**
 * 收集需要更新的状态ID
 */
std::vector<int> QuantumCircuit::collect_target_states(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::collect_target_states");
    (void)gate;
    return get_cached_target_states();
}

/**
 * 准备ELL算符
 */
FockELLOperator* QuantumCircuit::prepare_ell_operator(const GateParams& gate) {
    if (gate.params.empty()) {
        throw std::runtime_error("ELL算符构建失败：门参数为空");
    }

    Reference::Matrix dense_matrix;
    switch (gate.type) {
        case GateType::DISPLACEMENT:
            dense_matrix = Reference::create_displacement_matrix(cv_truncation_, gate.params[0]);
            break;
        case GateType::SQUEEZING:
            dense_matrix = Reference::create_squeezing_matrix(cv_truncation_, gate.params[0]);
            break;
        default:
            throw std::runtime_error("当前ELL路径仅支持位移门和挤压门");
    }

    int max_bandwidth = 0;
    std::vector<cuDoubleComplex> dense_flat;
    dense_flat.reserve(static_cast<size_t>(cv_truncation_) * cv_truncation_);
    for (int row = 0; row < cv_truncation_; ++row) {
        int row_nnz = 0;
        for (int col = 0; col < cv_truncation_; ++col) {
            const std::complex<double>& value = dense_matrix[row][col];
            if (std::abs(value) > 1e-12) {
                ++row_nnz;
            }
            dense_flat.push_back(make_cuDoubleComplex(value.real(), value.imag()));
        }
        max_bandwidth = std::max(max_bandwidth, row_nnz);
    }

    max_bandwidth = std::max(1, max_bandwidth);
    FockELLOperator* ell_op = new FockELLOperator(cv_truncation_, max_bandwidth);
    ell_op->build_from_dense(dense_flat);
    return ell_op;
}

/**
 * 准备挤压门的ELL算符
 */
FockELLOperator* QuantumCircuit::prepare_squeezing_ell_operator(std::complex<double> xi) {
    GateParams squeezing_gate(
        GateType::SQUEEZING,
        {},
        {0},
        {xi});
    return prepare_ell_operator(squeezing_gate);
}

/**
 * 获取状态振幅
 */
std::complex<double> QuantumCircuit::get_amplitude(
    const std::vector<int>& qubit_states,
    const std::vector<std::vector<std::complex<double>>>& qumode_states) {
    if (!root_node_) {
        return std::complex<double>(0.0, 0.0);
    }

    if (has_symbolic_terminals()) {
        materialize_symbolic_terminals_to_fock();
    }

    std::vector<int> padded_qubit_states = qubit_states;
    if (padded_qubit_states.size() < static_cast<size_t>(num_qubits_)) {
        padded_qubit_states.resize(num_qubits_, 0);
    }

    HDDNode* node = root_node_;
    std::complex<double> branch_weight(1.0, 0.0);

    while (node && !node->is_terminal()) {
        const int qubit_level = node->qubit_level;
        const int target_state = padded_qubit_states[qubit_level];
        if (target_state != 0 && target_state != 1) {
            throw std::invalid_argument("Qubit状态必须为0或1");
        }

        if (target_state == 0) {
            branch_weight *= node->w_low;
            node = node->low;
        } else {
            branch_weight *= node->w_high;
            node = node->high;
        }
    }

    if (!node || node->tensor_id < 0 || std::abs(branch_weight) < 1e-14) {
        return std::complex<double>(0.0, 0.0);
    }

    std::vector<cuDoubleComplex> state_data;
    state_pool_.download_state(node->tensor_id, state_data);
    return branch_weight * compute_qumode_overlap(state_data, cv_truncation_, num_qumodes_, qumode_states);
}

/**
 * 获取线路统计信息
 */
QuantumCircuit::CircuitStats QuantumCircuit::get_stats() const {
    const std::vector<int>& reachable_states = get_cached_target_states();
    const std::vector<int>& reachable_symbolic_states = get_cached_symbolic_terminal_ids();
    return {
        num_qubits_,
        num_qumodes_,
        cv_truncation_,
        static_cast<int>(gate_sequence_.size()),
        static_cast<int>(reachable_states.size() + reachable_symbolic_states.size()),
        count_reachable_hdd_nodes(root_node_)
    };
}

/**
 * 获取时间统计信息
 */
QuantumCircuit::TimeStats QuantumCircuit::get_time_stats() const {
    return {
        total_time_,
        transfer_time_,
        computation_time_,
        planning_time_
    };
}

/**
 * 方案 B：交互绘景执行引擎
 */
void QuantumCircuit::execute_with_interaction_picture() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. 初始化高斯参考系 (Frame) 和 HDD
    GaussianFrame current_frame(num_qumodes_);
    initialize_hdd(); 
    is_executed_ = true;

    // 2. 核心执行循环
    for (const auto& gate : gate_sequence_) {
        execute_gate_ip(gate, current_frame);
    }

    // 3. 最终物态还原 (将参考系中累积的高斯变换一次性作用到 Fock 终端)
    std::vector<int> terminal_ids = state_pool_.get_active_state_ids(); 
    for (int state_id : terminal_ids) {
        if (state_id != shared_zero_state_id_) {
            materialize_frame_to_fock(state_id, current_frame);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    total_time_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

void QuantumCircuit::execute_gate_ip(const GateParams& gate, GaussianFrame& frame) {
    switch (gate.type) {
        // --- 高斯门：仅更新符号参考系 (此处主要实现位移跟踪) ---
        case GateType::DISPLACEMENT: {
            int m = gate.target_qumodes[0];
            frame.alpha[m] += gate.params[0];
            break;
        }

        // --- 非高斯门：在参考系下进行参数偏移后应用 ---
        case GateType::KERR_GATE: {
            std::vector<int> targets = collect_target_states(gate);
            for (int sid : targets) {
                apply_displaced_non_gaussian_gate(sid, gate, frame);
            }
            break;
        }

        // --- 其他高斯门：如果尚未实现其参考系变换，则退化为即时作用并清空该 mode 的位移参考系 ---
        case GateType::PHASE_ROTATION:
        case GateType::SQUEEZING:
        case GateType::BEAM_SPLITTER: {
            // 在全量 IP 实现前，遇到这些门先将当前位移“落盘”回 Fock 态，再执行该门
            std::vector<int> targets = collect_target_states(gate);
            for (int sid : targets) {
                for (int m : gate.target_qumodes) {
                    if (std::abs(frame.alpha[m]) > 1e-9) {
                        apply_displacement_to_state(sid, frame.alpha[m], m);
                        frame.alpha[m] = 0.0;
                    }
                }
            }
            execute_gate(gate); 
            break;
        }

        // --- Qubit 和 混合门：默认处理 ---
        default:
            // 在执行混合门前，确保对应的 Qumode 参考系已同步到 Fock 空间
            if (!gate.target_qumodes.empty()) {
                std::vector<int> targets = collect_target_states(gate);
                for (int sid : targets) {
                    for (int m : gate.target_qumodes) {
                        if (std::abs(frame.alpha[m]) > 1e-9) {
                            apply_displacement_to_state(sid, frame.alpha[m], m);
                            frame.alpha[m] = 0.0;
                        }
                    }
                }
            }
            execute_gate(gate); 
            break;
    }
}

void QuantumCircuit::apply_displaced_non_gaussian_gate(int state_id, const GateParams& gate, const GaussianFrame& frame) {
    if (gate.type == GateType::KERR_GATE) {
        int m = gate.target_qumodes[0];
        std::complex<double> alpha = frame.alpha[m];

        if (std::abs(alpha) < 1e-9) {
            execute_level0_gate(gate);
            return;
        }

        // 交互绘景核心：状态向量保持在原点，通过临时位移应用非高斯算符
        // 这一步在未来可以优化为直接调用 ShiftedKerrKernel，从而避免两次位移操作
        apply_displacement_to_state(state_id, alpha, m);
        execute_level0_gate(gate);
        apply_displacement_to_state(state_id, -alpha, m);
    } else {
        execute_gate(gate);
    }
}

void QuantumCircuit::materialize_frame_to_fock(int state_id, const GaussianFrame& frame) {
    for (int m = 0; m < num_qumodes_; ++m) {
        if (std::abs(frame.alpha[m]) > 1e-9) {
            apply_displacement_to_state(state_id, frame.alpha[m], m);
        }
    }
}

// ===== 门操作便捷函数实现 =====

namespace Gates {
    GateParams PhaseRotation(int qumode, double theta) {
        return GateParams(GateType::PHASE_ROTATION, {}, {qumode}, {theta});
    }

    GateParams KerrGate(int qumode, double chi) {
        return GateParams(GateType::KERR_GATE, {}, {qumode}, {chi});
    }

    GateParams ConditionalParity(int qumode, double parity) {
        return GateParams(GateType::CONDITIONAL_PARITY, {}, {qumode}, {parity});
    }

    GateParams Snap(int qumode, double theta, int target_fock_state) {
        return GateParams(
            GateType::SNAP_GATE,
            {},
            {qumode},
            {std::complex<double>(theta, 0.0), std::complex<double>(static_cast<double>(target_fock_state), 0.0)});
    }

    GateParams MultiSNAP(int qumode, const std::vector<double>& phase_map) {
        std::vector<std::complex<double>> params;
        params.reserve(phase_map.size());
        for (double phase : phase_map) {
            params.emplace_back(phase, 0.0);
        }
        return GateParams(GateType::MULTI_SNAP_GATE, {}, {qumode}, params);
    }

    GateParams CrossKerr(int qumode1, int qumode2, double kappa) {
        return GateParams(GateType::CROSS_KERR_GATE, {}, {qumode1, qumode2}, {kappa});
    }

    GateParams CreationOperator(int qumode) {
        return GateParams(GateType::CREATION_OPERATOR, {}, {qumode});
    }

    GateParams AnnihilationOperator(int qumode) {
        return GateParams(GateType::ANNIHILATION_OPERATOR, {}, {qumode});
    }

    GateParams Displacement(int qumode, std::complex<double> alpha) {
        return GateParams(GateType::DISPLACEMENT, {}, {qumode}, {alpha});
    }

    GateParams Squeezing(int qumode, std::complex<double> xi) {
        return GateParams(GateType::SQUEEZING, {}, {qumode}, {xi});
    }

    GateParams BeamSplitter(int qumode1, int qumode2, double theta, double phi) {
        return GateParams(GateType::BEAM_SPLITTER, {}, {qumode1, qumode2}, {theta, phi});
    }

    GateParams ConditionalDisplacement(int control_qubit, int target_qumode, std::complex<double> alpha) {
        return GateParams(GateType::CONDITIONAL_DISPLACEMENT, {control_qubit}, {target_qumode}, {alpha});
    }

    GateParams ConditionalSqueezing(int control_qubit, int target_qumode, std::complex<double> xi) {
        return GateParams(GateType::CONDITIONAL_SQUEEZING, {control_qubit}, {target_qumode}, {xi});
    }

    GateParams ConditionalBeamSplitter(int control_qubit, int target_qumode1, int target_qumode2, double theta, double phi) {
        return GateParams(GateType::CONDITIONAL_BEAM_SPLITTER, {control_qubit}, {target_qumode1, target_qumode2}, {theta, phi});
    }

    GateParams ConditionalTwoModeSqueezing(int control_qubit, int target_qumode1, int target_qumode2, std::complex<double> xi) {
        return GateParams(GateType::CONDITIONAL_TWO_MODE_SQUEEZING, {control_qubit}, {target_qumode1, target_qumode2}, {xi});
    }

    GateParams ConditionalSUM(int control_qubit, int target_qumode1, int target_qumode2, double theta, double phi) {
        return GateParams(GateType::CONDITIONAL_SUM, {control_qubit}, {target_qumode1, target_qumode2}, {theta, phi});
    }

    GateParams RabiInteraction(int control_qubit, int target_qumode, double theta) {
        return GateParams(GateType::RABI_INTERACTION, {control_qubit}, {target_qumode}, {theta});
    }

    GateParams JaynesCummings(int control_qubit, int target_qumode, double theta, double phi) {
        return GateParams(GateType::JAYNES_CUMMINGS, {control_qubit}, {target_qumode}, {theta, phi});
    }

    GateParams AntiJaynesCummings(int control_qubit, int target_qumode, double theta, double phi) {
        return GateParams(GateType::ANTI_JAYNES_CUMMINGS, {control_qubit}, {target_qumode}, {theta, phi});
    }

    GateParams SelectiveQubitRotation(int target_qubit, int control_qumode, const std::vector<double>& theta_vec, const std::vector<double>& phi_vec) {
        std::vector<std::complex<double>> params;
        params.reserve(theta_vec.size() + phi_vec.size());
        for (double theta : theta_vec) params.push_back(theta);
        for (double phi : phi_vec) params.push_back(phi);
        return GateParams(GateType::SELECTIVE_QUBIT_ROTATION, {target_qubit}, {control_qumode}, params);
    }

    // Qubit门
    GateParams Hadamard(int qubit) {
        return GateParams(GateType::HADAMARD, {qubit});
    }

    GateParams PauliX(int qubit) {
        return GateParams(GateType::PAULI_X, {qubit});
    }

    GateParams PauliY(int qubit) {
        return GateParams(GateType::PAULI_Y, {qubit});
    }

    GateParams PauliZ(int qubit) {
        return GateParams(GateType::PAULI_Z, {qubit});
    }

    GateParams RotationX(int qubit, double theta) {
        return GateParams(GateType::ROTATION_X, {qubit}, {}, {theta});
    }

    GateParams RotationY(int qubit, double theta) {
        return GateParams(GateType::ROTATION_Y, {qubit}, {}, {theta});
    }

    GateParams RotationZ(int qubit, double theta) {
        return GateParams(GateType::ROTATION_Z, {qubit}, {}, {theta});
    }

    GateParams PhaseGateS(int qubit) {
        return GateParams(GateType::PHASE_GATE_S, {qubit});
    }

    GateParams PhaseGateT(int qubit) {
        return GateParams(GateType::PHASE_GATE_T, {qubit});
    }

    GateParams CNOT(int control, int target) {
        return GateParams(GateType::CNOT, {control, target});
    }

    GateParams CZ(int control, int target) {
        return GateParams(GateType::CZ, {control, target});
    }
}
