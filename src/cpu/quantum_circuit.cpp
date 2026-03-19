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
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include <chrono>
#include <future>
#include <map>
#include <optional>
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
                                int num_qumodes);

void zero_state_device(CVStatePool* pool, int state_id);
void initialize_vacuum_state_device(CVStatePool* pool, int state_id, int state_dim);
void copy_state_device(CVStatePool* pool, int src_state_id, int dst_state_id);
void axpy_state_device(CVStatePool* pool, int src_state_id, int dst_state_id, cuDoubleComplex weight);
void classify_vacuum_ray_device(CVStatePool* pool, int state_id, double tolerance,
                                int* is_zero, int* is_scaled_vacuum, cuDoubleComplex* scale);

// Level 1
void apply_creation_operator_on_mode(CVStatePool* pool, const int* targets, int batch_size,
                                     int target_qumode, int num_qumodes);
void apply_annihilation_operator_on_mode(CVStatePool* pool, const int* targets, int batch_size,
                                         int target_qumode, int num_qumodes);

// Level 2
void apply_single_mode_gate(CVStatePool* pool, FockELLOperator* ell_op,
                           const int* targets, int batch_size);
void apply_displacement_gate(CVStatePool* pool, const int* targets, int batch_size,
                            cuDoubleComplex alpha);

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

namespace {

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

std::vector<double> expand_selective_rotation_profile(
    const std::vector<double>& per_photon_values,
    int trunc_dim,
    int control_qumode,
    int num_qumodes,
    int max_total_dim) {
    std::vector<double> expanded(static_cast<size_t>(max_total_dim), 0.0);
    if (trunc_dim <= 0 || max_total_dim <= 0 || control_qumode < 0 || control_qumode >= num_qumodes) {
        return expanded;
    }

    const int right_stride = compute_qumode_right_stride(trunc_dim, control_qumode, num_qumodes);
    for (int flat_index = 0; flat_index < max_total_dim; ++flat_index) {
        const int photon_number = (flat_index / right_stride) % trunc_dim;
        if (photon_number < static_cast<int>(per_photon_values.size())) {
            expanded[static_cast<size_t>(flat_index)] = per_photon_values[static_cast<size_t>(photon_number)];
        }
    }
    return expanded;
}

constexpr double kVacuumTolerance = 1e-12;
constexpr double kSymbolicBranchPruneTolerance = 1e-14;
constexpr size_t kMaxSymbolicBranchesPerTerminal = 256;
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
    classify_vacuum_ray_device(&state_pool, state_id, kVacuumTolerance,
                               &is_zero, &is_scaled_vacuum, &scale);

    VacuumRayInfo info;
    info.is_zero = is_zero != 0;
    info.is_scaled_vacuum = is_scaled_vacuum != 0;
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

        int zero_dim = shared_zero_state_id_ >= 0 ? state_pool_.get_state_dim(shared_zero_state_id_) : 0;
        if (node->is_terminal()) {
            zero_dim = state_pool_.get_state_dim(node->tensor_id);
        }
        if (zero_dim <= 0) {
            zero_dim = state_pool_.get_max_total_dim();
        }

        std::vector<cuDoubleComplex> zeros(static_cast<size_t>(zero_dim),
                                           make_cuDoubleComplex(0.0, 0.0));
        state_pool_.upload_state(zero_id, zeros);
        return node_manager_.create_terminal_node(zero_id);
    }

    if (std::abs(weight - std::complex<double>(1.0, 0.0)) < 1e-14) {
        return node;
    }

    if (node->is_terminal()) {
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

            const int scaled_terminal_id = allocate_symbolic_terminal_id();
            symbolic_terminal_states_.emplace(
                scaled_terminal_id,
                SymbolicTerminalState{std::move(scaled_branches)});
            return node_manager_.create_terminal_node(scaled_terminal_id);
        }

        const int scaled_state_id = state_pool_.duplicate_state(node->tensor_id);
        if (scaled_state_id < 0) {
            throw std::runtime_error("HDD缩放失败：无法复制终端状态");
        }

        std::vector<cuDoubleComplex> scaled_state;
        state_pool_.download_state(scaled_state_id, scaled_state);
        const cuDoubleComplex weight_cu = make_cuDoubleComplex(weight.real(), weight.imag());
        for (cuDoubleComplex& amplitude : scaled_state) {
            amplitude = cuCmul(weight_cu, amplitude);
        }
        state_pool_.upload_state(scaled_state_id, scaled_state);
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

    const int scaled_state_id = state_pool_.duplicate_state(terminal_node->tensor_id);
    if (scaled_state_id < 0) {
        throw std::runtime_error("终端节点复制失败：无法复制状态");
    }

    if (std::abs(weight) < 1e-14) {
        int zero_dim = state_pool_.get_state_dim(scaled_state_id);
        if (zero_dim <= 0) {
            zero_dim = state_pool_.get_state_dim(terminal_node->tensor_id);
        }
        if (zero_dim <= 0) {
            zero_dim = state_pool_.get_max_total_dim();
        }

        std::vector<cuDoubleComplex> zeros(static_cast<size_t>(zero_dim),
                                           make_cuDoubleComplex(0.0, 0.0));
        state_pool_.upload_state(scaled_state_id, zeros);
        return node_manager_.create_terminal_node(scaled_state_id);
    }

    if (std::abs(weight - std::complex<double>(1.0, 0.0)) < 1e-14) {
        return node_manager_.create_terminal_node(scaled_state_id);
    }

    std::vector<cuDoubleComplex> scaled_state;
    state_pool_.download_state(scaled_state_id, scaled_state);
    const cuDoubleComplex weight_cu = make_cuDoubleComplex(weight.real(), weight.imag());
    for (cuDoubleComplex& amplitude : scaled_state) {
        amplitude = cuCmul(weight_cu, amplitude);
    }
    state_pool_.upload_state(scaled_state_id, scaled_state);
    return node_manager_.create_terminal_node(scaled_state_id);
}

/**
 * HDD节点加法: result = w1 * n1 + w2 * n2
 */
HDDNode* QuantumCircuit::hdd_add(HDDNode* n1, std::complex<double> w1, HDDNode* n2, std::complex<double> w2) {
    // 处理零权重
    if (std::abs(w1) < 1e-14) {
        if (std::abs(w2) < 1e-14) {
            return scale_hdd_node(n1 ? n1 : n2, std::complex<double>(0.0, 0.0));
        }
        return scale_hdd_node(n2, w2);
    }
    if (std::abs(w2) < 1e-14) {
        return scale_hdd_node(n1, w1);
    }

    // 基本情况：终端节点
    if (n1->is_terminal() && n2->is_terminal()) {
        int id1 = n1->tensor_id;
        int id2 = n2->tensor_id;
        if (id1 == id2) {
            return scale_hdd_node(n1, w1 + w2);
        }

        const bool symbolic1 = is_symbolic_terminal_id(id1);
        const bool symbolic2 = is_symbolic_terminal_id(id2);
        const bool zero1 = id1 == shared_zero_state_id_;
        const bool zero2 = id2 == shared_zero_state_id_;

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

            append_scaled_branches(id1, w1);
            append_scaled_branches(id2, w2);

            if (combined_branches.empty()) {
                return node_manager_.create_terminal_node(shared_zero_state_id_);
            }

            const int symbolic_terminal_id = allocate_symbolic_terminal_id();
            symbolic_terminal_states_.emplace(
                symbolic_terminal_id,
                SymbolicTerminalState{std::move(combined_branches)});
            return node_manager_.create_terminal_node(symbolic_terminal_id);
        }

        if (symbolic1) {
            id1 = project_symbolic_terminal_to_fock_state(id1);
        }
        if (symbolic2) {
            id2 = project_symbolic_terminal_to_fock_state(id2);
        }

        int new_id = state_pool_.allocate_state();
        
        // 统计传输时延
        auto transfer_start = std::chrono::high_resolution_clock::now();
        
        // 准备设备端数据
        int* d_src1 = nullptr;
        int* d_src2 = nullptr;
        int* d_dst = nullptr;
        cuDoubleComplex* d_w1 = nullptr;
        cuDoubleComplex* d_w2 = nullptr;
        
        cudaMalloc(&d_src1, sizeof(int));
        cudaMalloc(&d_src2, sizeof(int));
        cudaMalloc(&d_dst, sizeof(int));
        cudaMalloc(&d_w1, sizeof(cuDoubleComplex));
        cudaMalloc(&d_w2, sizeof(cuDoubleComplex));
        
        cudaMemcpy(d_src1, &id1, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_src2, &id2, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dst, &new_id, sizeof(int), cudaMemcpyHostToDevice);
        
        cuDoubleComplex w1_cu = make_cuDoubleComplex(w1.real(), w1.imag());
        cuDoubleComplex w2_cu = make_cuDoubleComplex(w2.real(), w2.imag());
        cudaMemcpy(d_w1, &w1_cu, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2, &w2_cu, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        
        auto transfer_end = std::chrono::high_resolution_clock::now();
        transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

        // 统计计算时延
        auto compute_start = std::chrono::high_resolution_clock::now();
        
        // 调用GPU内核
        add_states(&state_pool_, d_src1, d_w1, d_src2, d_w2, d_dst, 1);
        
        cudaDeviceSynchronize();
        
        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
        
        // 清理
        cudaFree(d_src1);
        cudaFree(d_src2);
        cudaFree(d_dst);
        cudaFree(d_w1);
        cudaFree(d_w2);
        
        return node_manager_.create_terminal_node(new_id);
    }
    
    // 递归步骤
    int level1 = n1->is_terminal() ? -1 : n1->qubit_level;
    int level2 = n2->is_terminal() ? -1 : n2->qubit_level;
    
    if (level1 != level2) {
        if (level1 > level2) {
            HDDNode* new_low = hdd_add(n1->low, w1 * n1->w_low, n2, w2);
            HDDNode* new_high = hdd_add(n1->high, w1 * n1->w_high, n2, w2);
            return node_manager_.get_or_create_node(level1, new_low, new_high, 1.0, 1.0);
        } else {
            // level2 > level1
            HDDNode* new_low = hdd_add(n1, w1, n2->low, w2 * n2->w_low);
            HDDNode* new_high = hdd_add(n1, w1, n2->high, w2 * n2->w_high);
            return node_manager_.get_or_create_node(level2, new_low, new_high, 1.0, 1.0);
        }
    }
    
    // 相同层级
    int level = level1;
    HDDNode* new_low = hdd_add(n1->low, w1 * n1->w_low, n2->low, w2 * n2->w_low);
    HDDNode* new_high = hdd_add(n1->high, w1 * n1->w_high, n2->high, w2 * n2->w_high);
    
    return node_manager_.get_or_create_node(level, new_low, new_high, 1.0, 1.0);
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
      gaussian_symbolic_mode_limit_(4), next_symbolic_terminal_id_(-2) {

    if (num_qubits <= 0 || num_qumodes <= 0 || cv_truncation <= 0) {
        throw std::invalid_argument("Qubit数量、Qumode数量和截断维度必须为正数");
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
    return !collect_symbolic_terminal_ids(root_node_).empty();
}

void QuantumCircuit::ensure_gaussian_state_pool() {
    if (gaussian_state_pool_) {
        return;
    }
    const int capacity = std::max(4096, state_pool_.capacity * 16);
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

    int* d_state_ids = nullptr;
    double* d_S = nullptr;
    double* d_dg = nullptr;

    CHECK_CUDA(cudaMalloc(&d_state_ids, gaussian_state_ids.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_S, gate.S.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_dg, gate.d.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(
        d_state_ids, gaussian_state_ids.data(),
        gaussian_state_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_S, gate.S.data(), gate.S.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_dg, gate.d.data(), gate.d.size() * sizeof(double), cudaMemcpyHostToDevice));

    apply_batched_symplectic_update(
        gaussian_state_pool_.get(),
        d_state_ids,
        static_cast<int>(gaussian_state_ids.size()),
        d_S,
        d_dg);

    CHECK_CUDA(cudaFree(d_state_ids));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_dg));
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

        std::cout << "Gaussian EDE块级加速已启用，块门数=" << compiled_block.gates.size() << std::endl;
        return true;
    } catch (const std::exception& e) {
        clear_cuda_runtime_error_state();
        std::cout << "Gaussian EDE块回退到全量Fock执行: " << e.what() << std::endl;
        return false;
    }
}

bool QuantumCircuit::apply_gaussian_mixture_approximation_on_gpu(
    int state_id,
    const GaussianMixtureApproximation& approximation) {
    if (approximation.branches.empty()) {
        return true;
    }

    const int state_dim = state_pool_.get_state_dim(state_id);
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
        if (d_single_target) {
            cudaFree(d_single_target);
            d_single_target = nullptr;
        }
        state_pool_.free_state(accum_state_id);
        state_pool_.free_state(scratch_state_id);
    };

    try {
        state_pool_.reserve_state_storage(scratch_state_id, state_dim);
        state_pool_.reserve_state_storage(accum_state_id, state_dim);
        zero_state_device(&state_pool_, accum_state_id);

        CHECK_CUDA(cudaMalloc(&d_single_target, sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_single_target, &scratch_state_id, sizeof(int), cudaMemcpyHostToDevice));

        for (const GaussianMixtureBranch& branch : approximation.branches) {
            if (branch.gaussian_gate.num_qumodes != num_qumodes_) {
                throw std::runtime_error("Gaussian Mixture分支qumode数量不匹配");
            }
            if (branch.target_qumodes.size() != branch.phase_rotation_thetas.size()) {
                throw std::runtime_error("Gaussian Mixture分支target/theta长度不匹配");
            }

            copy_state_device(&state_pool_, state_id, scratch_state_id);
            for (size_t idx = 0; idx < branch.target_qumodes.size(); ++idx) {
                const double theta = branch.phase_rotation_thetas[idx];
                if (std::abs(theta) < kDiagonalCanonicalizationTolerance) {
                    continue;
                }
                apply_phase_rotation_on_mode(
                    &state_pool_,
                    d_single_target,
                    1,
                    theta,
                    branch.target_qumodes[idx],
                    num_qumodes_);
            }

            axpy_state_device(
                &state_pool_,
                scratch_state_id,
                accum_state_id,
                make_cuDoubleComplex(branch.weight.real(), branch.weight.imag()));
        }

        copy_state_device(&state_pool_, accum_state_id, state_id);
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
            int* d_state_id = nullptr;
            CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));
            apply_phase_rotation_on_mode(
                &state_pool_,
                d_state_id,
                1,
                gate.params[0].real(),
                gate.target_qumodes[0],
                num_qumodes_);
            CHECK_CUDA(cudaFree(d_state_id));
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
    const auto it = symbolic_terminal_states_.find(terminal_id);
    if (it == symbolic_terminal_states_.end()) {
        throw std::runtime_error("symbolic terminal sidecar missing during Fock materialization");
    }

    if (it->second.branches.empty()) {
        return shared_zero_state_id_;
    }

    const int state_dim = state_pool_.get_max_total_dim();
    const int accum_state_id = state_pool_.allocate_state();
    if (accum_state_id < 0) {
        throw std::runtime_error("symbolic->Fock materialization failed: unable to allocate accumulator");
    }
    const int scratch_state_id = state_pool_.allocate_state();
    if (scratch_state_id < 0) {
        state_pool_.free_state(accum_state_id);
        throw std::runtime_error("symbolic->Fock materialization failed: unable to allocate scratch");
    }

    try {
        initialize_vacuum_state_device(&state_pool_, accum_state_id, state_dim);
        zero_state_device(&state_pool_, accum_state_id);
        state_pool_.reserve_state_storage(scratch_state_id, state_dim);

        for (const SymbolicGaussianBranch& branch : it->second.branches) {
            if (std::abs(branch.weight) < kSymbolicBranchPruneTolerance) {
                continue;
            }

            initialize_vacuum_state_device(&state_pool_, scratch_state_id, state_dim);
            for (const GateParams& replay_gate : branch.replay_gates) {
                apply_replayable_gaussian_gate_to_state(scratch_state_id, replay_gate);
            }

            axpy_state_device(
                &state_pool_,
                scratch_state_id,
                accum_state_id,
                make_cuDoubleComplex(branch.weight.real(), branch.weight.imag()));
        }
    } catch (...) {
        state_pool_.free_state(scratch_state_id);
        state_pool_.free_state(accum_state_id);
        throw;
    }

    state_pool_.free_state(scratch_state_id);
    return accum_state_id;
}

bool QuantumCircuit::materialize_symbolic_terminals_to_fock() {
    ScopedNvtxRange nvtx_range("qc::symbolic_to_fock");
    if (!has_symbolic_terminals()) {
        return true;
    }

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
            std::cout << "对角非高斯块Mixture预编译失败，回退到精确Fock执行: "
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
                            if (current_branches.size() > kMaxSymbolicBranchesPerTerminal) {
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
        std::cout << "对角非高斯块Gaussian Mixture回退到精确Fock执行: " << e.what() << std::endl;
        return false;
    }

    std::cout << "对角非高斯块Gaussian Mixture已启用，块门数="
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
    ScopedNvtxRange nvtx_range("qc::execute");
    if (!is_built_) {
        throw std::runtime_error("必须先构建量子线路");
    }

    if (is_executed_) {
        std::cout << "线路已执行，跳过重复执行" << std::endl;
        return;
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

    if (execution_blocks.empty()) {
        is_executed_ = true;
        auto end_total = std::chrono::high_resolution_clock::now();
        total_time_ = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        return;
    }

    CompiledExecutionBlock current_block =
        compile_execution_block(execution_sequence, execution_blocks, 0);
    planning_time_ += current_block.compile_time_ms;

    std::future<CompiledExecutionBlock> next_block_future;

    if (execution_blocks.size() > 1) {
        std::cout << "块级编译-执行流水线已启用，块数=" << execution_blocks.size() << std::endl;
    }

    for (size_t block_index = 0; block_index < execution_blocks.size(); ++block_index) {
        if (block_index + 1 < execution_blocks.size() && !next_block_future.valid()) {
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
            if (block_index + 1 < execution_blocks.size()) {
                current_block = next_block_future.get();
                planning_time_ += current_block.compile_time_ms;
                next_block_future = std::future<CompiledExecutionBlock>();
            }
            continue;
        }

        if (current_block.kind != ExecutionBlockKind::QubitOnly &&
            has_symbolic_terminals()) {
            materialize_symbolic_terminals_to_fock();
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
                    auto target_states = collect_target_states(fusable_gates[0]);
                    if (!target_states.empty()) {
                        auto transfer_start = std::chrono::high_resolution_clock::now();
                        const size_t ids_bytes = target_states.size() * sizeof(int);
                        int* d_target_ids = static_cast<int*>(
                            state_pool_.scratch_target_ids.ensure(ids_bytes));
                        CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
                                   ids_bytes, cudaMemcpyHostToDevice));
                        auto transfer_end = std::chrono::high_resolution_clock::now();
                        transfer_time_ += std::chrono::duration<double, std::milli>(
                            transfer_end - transfer_start).count();

                        auto compute_start = std::chrono::high_resolution_clock::now();
                        apply_fused_diagonal_gates(&state_pool_, d_target_ids,
                                                   static_cast<int>(target_states.size()),
                                                   ops_vec, num_qumodes_);
                        auto compute_end = std::chrono::high_resolution_clock::now();
                        computation_time_ += std::chrono::duration<double, std::milli>(
                            compute_end - compute_start).count();
                    }
                }

                // Execute remaining non-fusable gates normally
                for (const GateParams& gate : other_gates) {
                    execute_gate(gate);
                }

                if (block_index + 1 < execution_blocks.size()) {
                    current_block = next_block_future.get();
                    planning_time_ += current_block.compile_time_ms;
                    next_block_future = std::future<CompiledExecutionBlock>();
                }
                continue;
            }
        }
        // ── End fused diagonal ──────────────────────────────────────

        for (const GateParams& gate : current_block.gates) {
            execute_gate(gate);
        }

        if (block_index + 1 < execution_blocks.size()) {
            current_block = next_block_future.get();
            planning_time_ += current_block.compile_time_ms;
            next_block_future = std::future<CompiledExecutionBlock>();
        }
    }

    // 记录总结束时间
    auto end_total = std::chrono::high_resolution_clock::now();
    total_time_ = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    is_executed_ = true;
    std::cout << "量子线路执行完成" << std::endl;
    std::cout << "执行时间: " << total_time_ << " ms" << std::endl;
    std::cout << "传输时延: " << transfer_time_ << " ms" << std::endl;
    std::cout << "计算时延: " << computation_time_ << " ms" << std::endl;
    std::cout << "规划时延: " << planning_time_ << " ms" << std::endl;
}

/**
 * 重置量子线路状态
 */
void QuantumCircuit::reset() {
    // 同步所有GPU操作，确保在重置前所有操作完成
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

/**
 * 初始化HDD结构
 */
void QuantumCircuit::initialize_hdd() {
    const int vacuum_state_id = state_pool_.allocate_state();
    const int zero_state_id = state_pool_.allocate_state();
    if (vacuum_state_id < 0 || zero_state_id < 0) {
        throw std::runtime_error("初始化HDD失败：无法分配初始状态");
    }

    const int total_dim = state_pool_.get_max_total_dim();
    std::vector<cuDoubleComplex> vacuum_product_state(total_dim, make_cuDoubleComplex(0.0, 0.0));
    vacuum_product_state[0] = make_cuDoubleComplex(1.0, 0.0);
    state_pool_.upload_state(vacuum_state_id, vacuum_product_state);

    std::vector<cuDoubleComplex> zero_state(total_dim, make_cuDoubleComplex(0.0, 0.0));
    state_pool_.upload_state(zero_state_id, zero_state);
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

    if (old_root) {
        node_manager_.release_node(old_root);
    }

    size_t previous_cache_size = 0;
    do {
        previous_cache_size = node_manager_.get_cache_size();
        node_manager_.garbage_collect();
    } while (node_manager_.get_cache_size() < previous_cache_size);

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
    auto target_states = collect_target_states(gate);

    if (target_states.empty()) return;

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    // 上传状态ID到GPU (scratch buffer — no per-gate cudaMalloc)
    const size_t ids_bytes = target_states.size() * sizeof(int);
    int* d_target_ids = static_cast<int*>(state_pool_.scratch_target_ids.ensure(ids_bytes));
    CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
               ids_bytes, cudaMemcpyHostToDevice));

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    double param = gate.params.empty() ? 0.0 : gate.params[0].real();
    const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];

    // 统计计算时延
    auto compute_start = std::chrono::high_resolution_clock::now();

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            apply_phase_rotation_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                         target_qumode, num_qumodes_);
            break;
        case GateType::KERR_GATE:
            apply_kerr_gate_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                    target_qumode, num_qumodes_);
            break;
        case GateType::CONDITIONAL_PARITY:
            apply_conditional_parity_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                             target_qumode, num_qumodes_);
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
    CHECK_CUDA(cudaDeviceSynchronize());

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行Level 1门 (梯算符门)
 */
void QuantumCircuit::execute_level1_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_level1_gate");
    auto target_states = collect_target_states(gate);

    if (target_states.empty()) return;

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    const size_t ids_bytes = target_states.size() * sizeof(int);
    int* d_target_ids = static_cast<int*>(state_pool_.scratch_target_ids.ensure(ids_bytes));
    CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
               ids_bytes, cudaMemcpyHostToDevice));

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    // 统计计算时延
    auto compute_start = std::chrono::high_resolution_clock::now();
    const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];

    switch (gate.type) {
        case GateType::CREATION_OPERATOR:
            apply_creation_operator_on_mode(&state_pool_, d_target_ids, target_states.size(),
                                            target_qumode, num_qumodes_);
            break;
        case GateType::ANNIHILATION_OPERATOR:
            apply_annihilation_operator_on_mode(&state_pool_, d_target_ids, target_states.size(),
                                                target_qumode, num_qumodes_);
            break;
        default:
            break;
    }

    // 检查GPU内核执行错误
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行Level 2门 (单模门)
 */
void QuantumCircuit::execute_level2_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_level2_gate");
    auto target_states = collect_target_states(gate);

    if (target_states.empty()) return;

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    const size_t ids_bytes = target_states.size() * sizeof(int);
    int* d_target_ids = static_cast<int*>(state_pool_.scratch_target_ids.ensure(ids_bytes));
    CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
               ids_bytes, cudaMemcpyHostToDevice));

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    if (gate.type == GateType::DISPLACEMENT && !gate.params.empty()) {
        cuDoubleComplex alpha = make_cuDoubleComplex(gate.params[0].real(), gate.params[0].imag());
        const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];
        
        // 统计计算时延
        auto compute_start = std::chrono::high_resolution_clock::now();

        if (num_qumodes_ > 1 || target_qumode != 0) {
            apply_controlled_displacement_on_mode(
                &state_pool_, target_states, alpha, target_qumode, num_qumodes_);
        } else {
            apply_displacement_gate(&state_pool_, d_target_ids, target_states.size(), alpha);
        }
        
        // 检查GPU内核执行错误
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    } else if (gate.type == GateType::SQUEEZING && !gate.params.empty()) {
        const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];
        auto compute_start = std::chrono::high_resolution_clock::now();

        apply_squeezing_gate_gpu(&state_pool_,
                                 d_target_ids,
                                 static_cast<int>(target_states.size()),
                                 std::abs(gate.params[0]),
                                 std::arg(gate.params[0]),
                                 target_qumode,
                                 num_qumodes_);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

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
            
            apply_single_mode_gate(&state_pool_, ell_op, d_target_ids, target_states.size());
            
            // 检查GPU内核执行错误
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                delete ell_op;
                throw std::runtime_error("GPU单模门执行失败: " + std::string(cudaGetErrorString(err)));
            }
            
            // 在删除ELL操作符之前，确保GPU操作完成
            CHECK_CUDA(cudaDeviceSynchronize());
            
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
    auto target_states = collect_target_states(gate);

    if (target_states.empty()) return;

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    const size_t ids_bytes = target_states.size() * sizeof(int);
    int* d_target_ids = static_cast<int*>(state_pool_.scratch_target_ids.ensure(ids_bytes));
    CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
               ids_bytes, cudaMemcpyHostToDevice));

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
                                      theta, phi, target_qumode1, target_qumode2, num_qumodes_);

        // 检查GPU内核执行错误
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

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

    // 构造对应的单比特门矩阵U
    std::vector<std::complex<double>> u(4, std::complex<double>(0.0, 0.0));

    switch (gate.type) {
        case GateType::PAULI_X: {
            // X = [[0, 1], [1, 0]]
            u[1] = 1.0;  // (0,1)
            u[2] = 1.0;  // (1,0)
            break;
        }
        case GateType::PAULI_Y: {
            // Y = [[0, -i], [i, 0]]
            u[1] = std::complex<double>(0.0, -1.0);  // (0,1)
            u[2] = std::complex<double>(0.0, 1.0);   // (1,0)
            break;
        }
        case GateType::PAULI_Z: {
            // Z = [[1, 0], [0, -1]]
            u[0] = 1.0;   // (0,0)
            u[3] = -1.0;  // (1,1)
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
            break;
        }
        case GateType::PHASE_GATE_S: {
            // S = [[1, 0], [0, i]]
            u[0] = 1.0;  // (0,0)
            u[3] = std::complex<double>(0.0, 1.0); // (1,1)
            break;
        }
        case GateType::PHASE_GATE_T: {
            // T = [[1, 0], [0, e^(iπ/4)]]
            u[0] = 1.0;  // (0,0)
            u[3] = std::complex<double>(std::cos(M_PI/4.0), std::sin(M_PI/4.0)); // (1,1)
            break;
        }
        case GateType::CNOT: {
            // CNOT是双比特门，需要控制位和目标位
            if (gate.target_qubits.size() < 2) {
                throw std::runtime_error("CNOT门需要控制位和目标位");
            }
            int control = gate.target_qubits[0];
            int target = gate.target_qubits[1];
            replace_root_node(apply_cnot_recursive(root_node_, control, target));
            return;  // CNOT特殊处理，不走单比特门逻辑
        }
        case GateType::CZ: {
            // CZ也是双比特门
            if (gate.target_qubits.size() < 2) {
                throw std::runtime_error("CZ门需要控制位和目标位");
            }
            int control = gate.target_qubits[0];
            int target = gate.target_qubits[1];
            replace_root_node(apply_cz_recursive(root_node_, control, target));
            return;  // CZ特殊处理
        }
        default:
            throw std::runtime_error("不支持的Qubit门类型");
    }

    // 应用单比特门到HDD
    replace_root_node(apply_single_qubit_gate_recursive(root_node_, target_qubit, u));
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

    replace_root_node(apply_conditional_displacement_recursive(root_node_, control_qubit, target_qumode, alpha));
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
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
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

    // 递归遍历HDD，为每个分支应用相应的挤压
    replace_root_node(apply_conditional_squeezing_recursive(root_node_, control_qubit, target_qumode, xi));
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

    int* d_state_id = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    cudaGetLastError();

    auto compute_start = std::chrono::high_resolution_clock::now();
    apply_squeezing_gate_gpu(&state_pool_, d_state_id, 1, std::abs(xi), std::arg(xi),
                             target_qumode, num_qumodes_);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_state_id);
        throw std::runtime_error("GPU挤压门执行失败: " + std::string(cudaGetErrorString(err)));
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();

    CHECK_CUDA(cudaFree(d_state_id));
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

    // 递归遍历HDD，为每个分支应用相应的光束分裂器
    replace_root_node(apply_conditional_beam_splitter_recursive(root_node_, control_qubit, target_qumode1, target_qumode2, theta, phi));
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
    int* d_state_id = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));

    apply_beam_splitter_recursive(&state_pool_, d_state_id, 1, theta, phi,
                                  qumode1, qumode2, num_qumodes_);

    CHECK_CUDA(cudaFree(d_state_id));
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

    // 递归遍历HDD
    replace_root_node(apply_conditional_two_mode_squeezing_recursive(root_node_, control_qubit, target_qumode1, target_qumode2, xi));
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

    int* d_state_id = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    auto compute_start = std::chrono::high_resolution_clock::now();

    apply_two_mode_squeezing_recursive(&state_pool_, d_state_id, 1, std::abs(xi), std::arg(xi),
                                       qumode1, qumode2, num_qumodes_);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();

    CHECK_CUDA(cudaFree(d_state_id));
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

    replace_root_node(apply_conditional_sum_recursive(root_node_, control_qubit, target_qumode1, target_qumode2, theta, phi));
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

    int* d_state_id = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    auto compute_start = std::chrono::high_resolution_clock::now();

    apply_sum_gate(&state_pool_, d_state_id, 1, theta, cv_truncation_, cv_truncation_,
                   qumode1, qumode2, num_qumodes_);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();

    CHECK_CUDA(cudaFree(d_state_id));
}

// ===== 混合型相互作用门实现 =====

/**
 * 执行Rabi振荡门 RB(θ)
 * RB(θ) = exp[-i θ σ_x ⊗ (a + a†)]
 * 混合型：Qubit和Qumode相互作用
 */
void QuantumCircuit::execute_rabi_interaction(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    double theta = gate.params.empty() ? 0.0 : gate.params[0].real();

    // Rabi相互作用需要同时处理qubit和qumode状态
    // 这是一个复杂的混合操作，需要扩展HDD处理逻辑
    replace_root_node(apply_rabi_interaction_recursive(root_node_, control_qubit, target_qumode, theta));
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
        HDDNode* low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
        HDDNode* high_copy = duplicate_scaled_terminal_node(high_node, high_weight);

        std::vector<int> low_ids{low_copy->tensor_id};
        std::vector<int> high_ids{high_copy->tensor_id};
        apply_rabi_interaction_on_mode(
            &state_pool_, low_ids, high_ids, theta, target_qumode, num_qumodes_);
        CHECK_CUDA(cudaDeviceSynchronize());
        return {low_copy, high_copy};
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
        HDDNode* low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
        HDDNode* high_copy = duplicate_scaled_terminal_node(high_node, high_weight);

        std::vector<int> low_ids{low_copy->tensor_id};
        std::vector<int> high_ids{high_copy->tensor_id};
        if (anti_jaynes_cummings) {
            apply_anti_jaynes_cummings_on_mode(
                &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
        } else {
            apply_jaynes_cummings_on_mode(
                &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        return {low_copy, high_copy};
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
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;

    replace_root_node(apply_jaynes_cummings_recursive(root_node_, control_qubit, target_qumode, theta, phi));
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
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;

    replace_root_node(apply_anti_jaynes_cummings_recursive(root_node_, control_qubit, target_qumode, theta, phi));
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
    int target_qubit = gate.target_qubits[0];
    int control_qumode = gate.target_qumodes[0];
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

    replace_root_node(apply_selective_qubit_rotation_recursive(root_node_, target_qubit, control_qumode, theta_vec, phi_vec));
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
        HDDNode* low_copy = duplicate_scaled_terminal_node(low_node, low_weight);
        HDDNode* high_copy = duplicate_scaled_terminal_node(high_node, high_weight);

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
            std::vector<int>{low_copy->tensor_id},
            std::vector<int>{high_copy->tensor_id},
            expanded_thetas,
            expanded_phis);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        return {low_copy, high_copy};
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
    std::vector<int> state_ids = collect_terminal_state_ids(root_node_);
    state_ids.erase(
        std::remove(state_ids.begin(), state_ids.end(), shared_zero_state_id_),
        state_ids.end());
    return state_ids;
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
    std::vector<int> reachable_states = collect_terminal_state_ids(root_node_);
    reachable_states.erase(
        std::remove(reachable_states.begin(), reachable_states.end(), shared_zero_state_id_),
        reachable_states.end());
    const std::vector<int> reachable_symbolic_states = collect_symbolic_terminal_ids(root_node_);
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
