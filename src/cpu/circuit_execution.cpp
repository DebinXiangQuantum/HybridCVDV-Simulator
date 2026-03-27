// circuit_execution.cpp — Gaussian EDE, mixture, projection, materialization

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

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

                    std::vector<GaussianComponent> transformed_comps;

                    if (is_symbolic_terminal_id(node->tensor_id)) {
                        const MixtureGaussianState& mixture =
                            symbolic_terminal_states_.at(node->tensor_id);
                        transformed_comps.reserve(mixture.components.size());
                        for (const GaussianComponent& comp : mixture.components) {
                            GaussianComponent t = comp;
                            // Always copy: we will mutate via symplectic update
                            t.gaussian_state_id =
                                duplicate_gaussian_state(comp.gaussian_state_id);
                            transformed_comps.push_back(std::move(t));
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
                        transformed_comps.push_back(
                            {gaussian_state_id, info.scale, {}});
                    }

                    std::vector<int> gaussian_state_ids;
                    gaussian_state_ids.reserve(transformed_comps.size());
                    for (const GaussianComponent& branch : transformed_comps) {
                        gaussian_state_ids.push_back(branch.gaussian_state_id);
                    }

                    auto compute_start = std::chrono::high_resolution_clock::now();
                    for (const GateParams& gate : resolved_gates) {
                        apply_symplectic_update_to_gaussian_states(
                            gaussian_state_ids,
                            gate_to_symplectic(gate, num_qumodes_));
                        for (GaussianComponent& branch : transformed_comps) {
                            branch.replay_gates.push_back(gate);
                        }
                    }
                    auto compute_end = std::chrono::high_resolution_clock::now();
                    computation_time_ +=
                        std::chrono::duration<double, std::milli>(compute_end - compute_start).count();

                    int symbolic_terminal_id = allocate_symbolic_terminal_id();
                    symbolic_terminal_states_.emplace(
                        symbolic_terminal_id,
                        MixtureGaussianState{std::move(transformed_comps)});
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

    if (gaussian_state_pool_ && it->second.components.size() > 1) {
        std::unordered_map<std::string, size_t> unique_branch_index;
        std::vector<GaussianComponent> coalesced_branches;
        coalesced_branches.reserve(it->second.components.size());

        std::vector<double> displacement;
        std::vector<double> covariance;
        for (GaussianComponent& branch : it->second.components) {
            if (branch.gaussian_state_id < 0 ||
                std::abs(branch.weight) < kSymbolicBranchPruneTolerance) {
                if (branch.gaussian_state_id >= 0) {
                    gaussian_state_pool_->release_ref(branch.gaussian_state_id);
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

            GaussianComponent& canonical_branch = coalesced_branches[dedup_it->second];
            canonical_branch.weight += branch.weight;
            gaussian_state_pool_->release_ref(branch.gaussian_state_id);
        }

        std::vector<GaussianComponent> pruned_branches;
        pruned_branches.reserve(coalesced_branches.size());
        for (GaussianComponent& branch : coalesced_branches) {
            if (std::abs(branch.weight) < kSymbolicBranchPruneTolerance) {
                if (branch.gaussian_state_id >= 0) {
                    gaussian_state_pool_->release_ref(branch.gaussian_state_id);
                }
                continue;
            }
            pruned_branches.push_back(std::move(branch));
        }

        if (pruned_branches.size() != it->second.components.size()) {
            FALLBACK_DEBUG_LOG << "[fallback] coalesced symbolic terminal " << terminal_id
                               << " branches: " << it->second.components.size()
                               << " -> " << pruned_branches.size() << std::endl;
        }
        it->second.components = std::move(pruned_branches);
    }

    if (it->second.components.empty()) {
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
                       << " to Fock, branches=" << it->second.components.size() << std::endl;

    try {
        state_pool_.reserve_state_storage(accum_state_id, state_dim);
        zero_state_device(&state_pool_, accum_state_id, nullptr, false);
        state_pool_.reserve_state_storage(scratch_state_id, state_dim);

        for (size_t branch_index = 0; branch_index < it->second.components.size(); ++branch_index) {
            const GaussianComponent& branch = it->second.components[branch_index];
            if (std::abs(branch.weight) < kSymbolicBranchPruneTolerance) {
                continue;
            }

            if (branch_index == 0 ||
                branch_index + 1 == it->second.components.size() ||
                ((branch_index + 1) % 32 == 0)) {
                FALLBACK_DEBUG_LOG << "[fallback] terminal " << terminal_id
                                   << " replay branch " << (branch_index + 1)
                                   << "/" << it->second.components.size() << std::endl;
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

    auto free_component_refs = [&](std::vector<GaussianComponent>* comps) {
        if (!gaussian_state_pool_) {
            return;
        }
        for (const GaussianComponent& comp : *comps) {
            if (comp.gaussian_state_id >= 0) {
                gaussian_state_pool_->release_ref(comp.gaussian_state_id);
            }
        }
        comps->clear();
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

                    std::vector<GaussianComponent> current_branches;

                    if (is_symbolic_terminal_id(node->tensor_id)) {
                        const MixtureGaussianState& symbolic_state =
                            symbolic_terminal_states_.at(node->tensor_id);
                        std::vector<std::complex<double>> initial_branch_weights;
                        initial_branch_weights.reserve(symbolic_state.components.size());
                        for (const GaussianComponent& existing_branch : symbolic_state.components) {
                            initial_branch_weights.push_back(existing_branch.weight);
                        }

                        size_t projected_branch_count = symbolic_state.components.size();
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

                        // Share existing components via refcount (no GPU copy)
                        current_branches.reserve(symbolic_state.components.size());
                        for (const GaussianComponent& comp : symbolic_state.components) {
                            GaussianComponent shared = comp;
                            if (comp.gaussian_state_id >= 0 && gaussian_state_pool_) {
                                gaussian_state_pool_->add_ref(comp.gaussian_state_id);
                            }
                            current_branches.push_back(std::move(shared));
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
                            std::vector<GaussianComponent> expanded_branches;
                            expanded_branches.reserve(std::min(
                                current_branches.size() * std::max<size_t>(size_t{1}, approximation.branches.size()),
                                static_cast<size_t>(symbolic_branch_limit_)));
                            for (const GaussianComponent& base_comp : current_branches) {
                                for (const GaussianMixtureBranch& mixture_branch : approximation.branches) {
                                    const std::complex<double> new_weight =
                                        base_comp.weight * mixture_branch.weight;
                                    if (std::abs(new_weight) < kSymbolicBranchPruneTolerance) {
                                        continue;
                                    }

                                    GaussianComponent expanded = base_comp;
                                    expanded.weight = new_weight;
                                    // Copy GPU state: we'll mutate it with phase rotations
                                    expanded.gaussian_state_id =
                                        duplicate_gaussian_state(base_comp.gaussian_state_id);

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
                                            {expanded.gaussian_state_id},
                                            gate_to_symplectic(phase_gate, num_qumodes_));
                                        expanded.replay_gates.push_back(phase_gate);
                                    }
                                    expanded_branches.push_back(std::move(expanded));
                                }
                            }

                            free_component_refs(&current_branches);
                            current_branches = std::move(expanded_branches);
                            if (current_branches.size() >
                                static_cast<size_t>(symbolic_branch_limit_)) {
                                free_component_refs(&current_branches);
                                throw std::runtime_error("symbolic mixture branch count exceeded limit");
                            }
                        }
                        auto compute_end = std::chrono::high_resolution_clock::now();
                        computation_time_ +=
                            std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
                    } catch (...) {
                        free_component_refs(&current_branches);
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
                        MixtureGaussianState{std::move(current_branches)});
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
