#pragma once
// circuit_internal.h — Shared declarations for quantum_circuit.cpp split files

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

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA错误: " + std::string(cudaGetErrorString(err))); \
        } \
    } while (0)

// ==================== GPU Kernel Forward Declarations ====================

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
void apply_controlled_displacement(CVStatePool* state_pool,
                                   const std::vector<int>& controlled_states,
                                   cuDoubleComplex alpha);
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


// ==================== Shared Internal Helpers ====================
namespace circuit_internal {

inline bool fallback_debug_logging_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("HYBRIDCVDV_FALLBACK_DEBUG");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

inline bool async_cv_pipeline_disabled() {
    static const bool disabled = []() {
        const char* env = std::getenv("HYBRIDCVDV_DISABLE_ASYNC_CV_PIPELINE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return disabled;
}

#define FALLBACK_DEBUG_LOG if (!fallback_debug_logging_enabled()) {} else std::cout

inline const char* block_progress_log_path() {
    static const char* path = std::getenv("HYBRIDCVDV_BLOCK_PROGRESS_LOG");
    return (path != nullptr && path[0] != '\0') ? path : nullptr;
}

inline size_t block_progress_log_interval() {
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

inline void log_block_progress_if_requested(size_t block_index, size_t total_blocks) {
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

inline int compute_qumode_right_stride(int trunc_dim, int target_qumode, int num_qumodes) {
    int right_stride = 1;
    for (int mode = target_qumode + 1; mode < num_qumodes; ++mode) {
        right_stride *= trunc_dim;
    }
    return right_stride;
}

inline size_t align_scratch_offset(size_t offset, size_t alignment) {
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

inline bool is_pairwise_hybrid_gate_type(GateType type) {
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

inline PairwiseHybridStorageEstimate estimate_pairwise_hybrid_storage(HDDNode* root,
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


// Forward declarations — pairwise hybrid helpers (circuit_pairwise_helpers.cpp)
std::unordered_map<int, size_t> collect_pairwise_replaced_state_use_counts(
    HDDNode* root, int control_qubit);
std::unordered_set<int> collect_retained_state_ids_outside_pairwise_region(
    HDDNode* root, int control_qubit);
size_t pairwise_hybrid_working_extra_elements(
    HDDNode* root, int control_qubit, CVStatePool& state_pool);
std::unordered_map<int, size_t> collect_pairwise_replaced_state_use_counts(
    HDDNode* root, int control_qubit);
std::unordered_set<int> collect_retained_state_ids_outside_pairwise_region(
    HDDNode* root, int control_qubit);
size_t pairwise_hybrid_working_extra_elements(
    HDDNode* root, int control_qubit, size_t state_dim,
    const PairwiseHybridStorageEstimate& full_estimate,
    bool* early_release_enabled = nullptr);
void release_pairwise_replaced_state_if_safe(
    int state_id, bool allow_early_release,
    std::unordered_map<int, size_t>& remaining_use_counts,
    CVStatePool& state_pool, int shared_zero_state_id);
void reserve_pairwise_hybrid_headroom(const char* gate_name,
    HDDNode* root, int control_qubit, CVStatePool& state_pool);
void cleanup_duplicated_pairwise_states(CVStatePool& state_pool,
    const std::vector<int>& low_ids, const std::vector<int>& high_ids);
void cleanup_pairwise_build_failure(
    HDDNodeManager& node_manager, CVStatePool& state_pool,
    const std::unordered_map<WeightedNodePairKey, std::pair<HDDNode*, HDDNode*>, WeightedNodePairKeyHash>& pair_memo,
    const std::unordered_map<HDDNode*, HDDNode*>& node_memo,
    const std::vector<int>& low_ids, const std::vector<int>& high_ids);
void release_transient_pairwise_node(
    HDDNodeManager& node_manager, CVStatePool& state_pool,
    HDDNode* node, const std::vector<int>& low_ids, const std::vector<int>& high_ids);
std::vector<double> expand_selective_rotation_profile(
    const std::vector<double>& per_photon_values, int trunc_dim,
    int control_qumode, int num_qumodes, int64_t max_total_dim);

constexpr double kVacuumTolerance = 1e-12;
constexpr double kSymbolicBranchPruneTolerance = 1e-14;
constexpr int kDefaultSymbolicBranchLimit = 64;
constexpr double kTargetMixtureFidelity = 0.9999;
constexpr int kMaxKerrMixtureBranches = 32;
constexpr int kMaxSnapMixtureBranches = 32;
constexpr int kMaxCrossKerrMixtureBranches = 64;
constexpr double kDiagonalCanonicalizationTolerance = 1e-14;

inline size_t integer_power(size_t base, int exponent) {
    size_t result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

inline bool is_nontrivial_phase(double theta) {
    return std::abs(theta) > kDiagonalCanonicalizationTolerance;
}

inline bool symbolic_mixture_would_exceed_branch_limit(
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

inline double conservative_fidelity_lower_bound_from_operator_error(double operator_error) {
    if (operator_error <= 0.0) {
        return 1.0;
    }
    if (operator_error >= 1.0) {
        return 0.0;
    }

    const double overlap_lower_bound = (1.0 - operator_error) / (1.0 + operator_error);
    return overlap_lower_bound * overlap_lower_bound;
}

inline HDDNode* find_all_zero_qubit_terminal(HDDNode* node) {
    HDDNode* current = node;
    while (current && !current->is_terminal()) {
        current = current->low;
    }
    return current;
}

inline bool is_vacuum_fock_state(const std::vector<cuDoubleComplex>& state) {
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

inline VacuumRayInfo classify_vacuum_ray(const std::vector<cuDoubleComplex>& state) {
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

inline VacuumRayInfo classify_vacuum_ray_on_device(CVStatePool& state_pool, int state_id) {
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

inline void clear_cuda_runtime_error_state() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess && err != cudaErrorNotReady) {
        cudaGetLastError();
    }

    err = cudaGetLastError();
    if (err != cudaSuccess && err != cudaErrorNotReady) {
        cudaGetLastError();
    }
}

inline std::vector<cuDoubleComplex> to_cuda_state(
    const std::vector<std::complex<double>>& state) {
    std::vector<cuDoubleComplex> cuda_state;
    cuda_state.reserve(state.size());
    for (const auto& amplitude : state) {
        cuda_state.push_back(make_cuDoubleComplex(amplitude.real(), amplitude.imag()));
    }
    return cuda_state;
}

inline std::vector<std::complex<double>> to_host_complex(
    const std::vector<cuDoubleComplex>& state) {
    std::vector<std::complex<double>> host_state;
    host_state.reserve(state.size());
    for (const cuDoubleComplex amplitude : state) {
        host_state.emplace_back(cuCreal(amplitude), cuCimag(amplitude));
    }
    return host_state;
}


// ==================== Forward Declarations (defined in circuit_compilation.cpp) ====================

void trim_trailing_zero_phases(std::vector<double>* phase_map);
bool is_unconditional_gaussian_gate(const GateParams& gate);
bool is_supported_controlled_gaussian_gate(const GateParams& gate);
bool is_gaussian_track_gate(const GateParams& gate);
bool is_pure_qubit_gate(const GateParams& gate);
const char* gate_type_name(GateType type);
bool validate_gaussian_track_gate(const GateParams& gate,
    int total_qubits, int total_qumodes, std::string* error_message);
std::vector<int> collect_gaussian_control_qubits(const std::vector<GateParams>& gates);
std::string make_control_assignment_key(const std::vector<int>& control_qubits,
    const std::vector<int>& assignment);
std::vector<int> assignment_from_key(const std::vector<int>& control_qubits,
    const std::string& key, int num_qubits);
std::optional<GateParams> resolve_gaussian_gate_for_assignment(
    const GateParams& gate, const std::vector<int>& qubit_assignment);
void scale_cuda_state(std::vector<cuDoubleComplex>* state,
    std::complex<double> scale);
void collect_symbolic_terminal_ids_recursive(
    HDDNode* node, std::unordered_set<size_t>& visited_nodes,
    std::unordered_set<int>& symbolic_ids);
int choose_snap_mixture_branch_cap(const GateParams& gate, int cutoff);
int choose_cross_kerr_mixture_branch_cap(int cutoff);
void apply_exact_diagonal_gate_host(std::vector<std::complex<double>>* state,
    const GateParams& gate, int cutoff, int num_qumodes);
bool is_pure_cv_diagonal_gate(const GateParams& gate);
bool is_pure_cv_diagonal_non_gaussian_gate(const GateParams& gate);
int choose_kerr_mixture_branch_cap(int cutoff);
int required_exact_branch_count_for_gate(const GateParams& gate, int cutoff);
bool compile_diagonal_gate_mixture_approximation(
    const GateParams& gate, int total_qumodes, int cutoff,
    double target_fidelity, GaussianMixtureApproximation* approximation,
    std::string* error_message);
std::vector<GateParams> canonicalize_pure_cv_diagonal_window(
    const std::vector<GateParams>& window);
void collect_terminal_state_ids_recursive(HDDNode* node,
    std::unordered_set<size_t>& visited_nodes,
    std::unordered_set<int>& state_ids);
std::vector<int> collect_terminal_state_ids(HDDNode* root);
size_t count_reachable_hdd_nodes_recursive(HDDNode* node,
    std::unordered_set<size_t>& visited_nodes);
size_t count_reachable_hdd_nodes(HDDNode* root);

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
    int d_trunc, int num_qumodes,
    const std::vector<std::vector<std::complex<double>>>& qumode_states);

// gate_to_symplectic: convert GateParams to symplectic matrix for Gaussian gates
SymplecticGate gate_to_symplectic(const GateParams& gate, int total_qumodes);

}  // namespace circuit_internal
