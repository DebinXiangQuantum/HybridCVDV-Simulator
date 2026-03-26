// circuit_cv_gates.cpp — Conditional displacement, squeezing, BS, TMS, sum

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

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
