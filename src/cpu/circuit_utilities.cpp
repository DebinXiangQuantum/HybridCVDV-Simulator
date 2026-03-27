// circuit_utilities.cpp — target states, amplitude, stats, interaction picture, Gates

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

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
