// circuit_gate_dispatch.cpp — Gate level dispatch (levels 0-4) and qubit gates

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

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
