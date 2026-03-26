// circuit_selective_rotation.cpp — Selective qubit rotation gate

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

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
