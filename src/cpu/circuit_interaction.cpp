// circuit_interaction.cpp — Rabi, Jaynes-Cummings, Anti-Jaynes-Cummings

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

namespace {

bool is_unit_pairwise_weight(const std::complex<double>& weight) {
    return std::abs(weight - std::complex<double>(1.0, 0.0)) < 1e-14;
}

struct WeightedNodeKey {
    HDDNode* node = nullptr;
    std::complex<double> weight{1.0, 0.0};

    bool operator==(const WeightedNodeKey& other) const {
        return node == other.node && weight == other.weight;
    }
};

struct WeightedNodeKeyHash {
    size_t operator()(const WeightedNodeKey& key) const {
        size_t hash = std::hash<HDDNode*>()(key.node);
        hash ^= std::hash<double>()(key.weight.real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<double>()(key.weight.imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

}  // namespace

bool QuantumCircuit::try_collect_normalized_pairwise_terminal_ids(
    int control_qubit,
    std::vector<int>* low_ids,
    std::vector<int>* high_ids) const {
    if (!low_ids || !high_ids) {
        return false;
    }

    low_ids->clear();
    high_ids->clear();
    if (!root_node_) {
        return false;
    }

    std::unordered_set<WeightedNodePairKey, WeightedNodePairKeyHash> visited_pairs;
    std::unordered_set<HDDNode*> visited_nodes;

    std::function<bool(HDDNode*, HDDNode*)> collect_pairs =
        [&](HDDNode* low_node, HDDNode* high_node) -> bool {
            if (!low_node || !high_node) {
                return false;
            }

            const WeightedNodePairKey key{
                low_node,
                high_node,
                std::complex<double>(1.0, 0.0),
                std::complex<double>(1.0, 0.0)
            };
            if (!visited_pairs.insert(key).second) {
                return true;
            }

            if (low_node->is_terminal() && high_node->is_terminal()) {
                const int low_id = low_node->tensor_id;
                const int high_id = high_node->tensor_id;
                if (low_id < 0 ||
                    high_id < 0 ||
                    low_id == shared_zero_state_id_ ||
                    high_id == shared_zero_state_id_ ||
                    is_symbolic_terminal_id(low_id) ||
                    is_symbolic_terminal_id(high_id) ||
                    !state_pool_.is_valid_state(low_id) ||
                    !state_pool_.is_valid_state(high_id)) {
                    return false;
                }

                low_ids->push_back(low_id);
                high_ids->push_back(high_id);
                return true;
            }

            if (low_node->is_terminal() || high_node->is_terminal()) {
                return false;
            }
            if (low_node->qubit_level != high_node->qubit_level) {
                return false;
            }
            if (!is_unit_pairwise_weight(low_node->w_low) ||
                !is_unit_pairwise_weight(low_node->w_high) ||
                !is_unit_pairwise_weight(high_node->w_low) ||
                !is_unit_pairwise_weight(high_node->w_high)) {
                return false;
            }

            return collect_pairs(low_node->low, high_node->low) &&
                   collect_pairs(low_node->high, high_node->high);
        };

    std::function<bool(HDDNode*)> walk =
        [&](HDDNode* node) -> bool {
            if (!node || node->is_terminal()) {
                return true;
            }
            if (node->qubit_level < control_qubit) {
                return true;
            }
            if (!visited_nodes.insert(node).second) {
                return true;
            }
            if (node->qubit_level == control_qubit) {
                if (!is_unit_pairwise_weight(node->w_low) ||
                    !is_unit_pairwise_weight(node->w_high)) {
                    return false;
                }
                return collect_pairs(node->low, node->high);
            }
            return walk(node->low) && walk(node->high);
        };

    const bool ok = walk(root_node_);
    if (!ok || low_ids->empty() || low_ids->size() != high_ids->size()) {
        low_ids->clear();
        high_ids->clear();
        return false;
    }
    return true;
}

bool QuantumCircuit::normalize_root_for_pairwise_fastpath() {
    if (!root_node_ || has_symbolic_terminals()) {
        return false;
    }

    std::unordered_map<WeightedNodeKey, HDDNode*, WeightedNodeKeyHash> memo;
    std::vector<int> allocated_state_ids;

    try {
        std::function<HDDNode*(HDDNode*, std::complex<double>)> normalize_recursive =
            [&](HDDNode* node, std::complex<double> weight) -> HDDNode* {
                if (!node) {
                    return nullptr;
                }

                const WeightedNodeKey key{node, weight};
                const auto memo_it = memo.find(key);
                if (memo_it != memo.end()) {
                    return memo_it->second;
                }

                HDDNode* result = nullptr;
                if (node->is_terminal()) {
                    if (node->tensor_id < 0) {
                        throw std::runtime_error(
                            "pairwise fast-path root normalization encountered invalid terminal id");
                    }

                    const int preferred_device =
                        state_pool_.get_state_device_id(node->tensor_id);
                    result =
                        duplicate_scaled_terminal_node(node, weight, preferred_device);
                    allocated_state_ids.push_back(result->tensor_id);
                } else {
                    HDDNode* new_low =
                        normalize_recursive(node->low, weight * node->w_low);
                    HDDNode* new_high =
                        normalize_recursive(node->high, weight * node->w_high);
                    result = node_manager_.get_or_create_node(
                        node->qubit_level, new_low, new_high, 1.0, 1.0);
                }

                memo.emplace(key, result);
                return result;
            };

        HDDNode* new_root = normalize_recursive(root_node_, std::complex<double>(1.0, 0.0));
        replace_root_node(new_root);
        return true;
    } catch (...) {
        for (int state_id : allocated_state_ids) {
            if (state_pool_.is_valid_state(state_id)) {
                state_pool_.free_state(state_id);
            }
        }
        throw;
    }
}

bool QuantumCircuit::try_execute_inplace_pairwise_hybrid_gate(
    const GateParams& gate,
    const std::vector<int>& low_ids,
    const std::vector<int>& high_ids) {
    if (low_ids.empty() || low_ids.size() != high_ids.size() ||
        gate.target_qumodes.empty()) {
        return false;
    }

    const int target_qumode = gate.target_qumodes[0];
    switch (gate.type) {
        case GateType::RABI_INTERACTION: {
            const double theta =
                gate.params.empty() ? 0.0 : gate.params[0].real();
            apply_rabi_interaction_on_mode(
                &state_pool_, low_ids, high_ids, theta, target_qumode, num_qumodes_);
            CHECK_CUDA(cudaGetLastError());
            break;
        }
        case GateType::JAYNES_CUMMINGS: {
            const double theta =
                gate.params.empty() ? 0.0 : gate.params[0].real();
            const double phi =
                gate.params.size() > 1 ? gate.params[1].real() : 0.0;
            apply_jaynes_cummings_on_mode(
                &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
            CHECK_CUDA(cudaGetLastError());
            break;
        }
        case GateType::ANTI_JAYNES_CUMMINGS: {
            const double theta =
                gate.params.empty() ? 0.0 : gate.params[0].real();
            const double phi =
                gate.params.size() > 1 ? gate.params[1].real() : 0.0;
            apply_anti_jaynes_cummings_on_mode(
                &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
            CHECK_CUDA(cudaGetLastError());
            break;
        }
        default:
            return false;
    }

    state_pool_.synchronize_all_devices();
    return true;
}

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
                    int pair_device = execution_device_id_;
                    const int low_owner = state_pool_.get_state_device_id(low_node->tensor_id);
                    const int high_owner = state_pool_.get_state_device_id(high_node->tensor_id);
                    if (low_owner >= 0 && high_owner >= 0 && low_owner == high_owner) {
                        pair_device = low_owner;
                    } else if (low_owner >= 0 && high_owner < 0) {
                        pair_device = low_owner;
                    } else if (high_owner >= 0 && low_owner < 0) {
                        pair_device = high_owner;
                    }
                    HDDNode* low_copy = nullptr;
                    HDDNode* high_copy = nullptr;
                    try {
                        low_copy = duplicate_scaled_terminal_node(low_node, low_weight, pair_device);
                        low_ids.push_back(low_copy->tensor_id);
                        high_copy = duplicate_scaled_terminal_node(high_node, high_weight, pair_device);
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
        state_pool_.synchronize_all_devices();
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
        int pair_device = execution_device_id_;
        const int low_owner = state_pool_.get_state_device_id(low_node->tensor_id);
        const int high_owner = state_pool_.get_state_device_id(high_node->tensor_id);
        if (low_owner >= 0 && high_owner >= 0 && low_owner == high_owner) {
            pair_device = low_owner;
        } else if (low_owner >= 0 && high_owner < 0) {
            pair_device = low_owner;
        } else if (high_owner >= 0 && low_owner < 0) {
            pair_device = high_owner;
        }
        std::vector<int> low_ids;
        std::vector<int> high_ids;
        HDDNode* low_copy = nullptr;
        HDDNode* high_copy = nullptr;
        try {
            low_copy = duplicate_scaled_terminal_node(low_node, low_weight, pair_device);
            low_ids.push_back(low_copy->tensor_id);
            high_copy = duplicate_scaled_terminal_node(high_node, high_weight, pair_device);
            high_ids.push_back(high_copy->tensor_id);

            apply_rabi_interaction_on_mode(
                &state_pool_, low_ids, high_ids, theta, target_qumode, num_qumodes_);
            state_pool_.synchronize_all_devices();
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
        int pair_device = execution_device_id_;
        const int low_owner = state_pool_.get_state_device_id(low_node->tensor_id);
        const int high_owner = state_pool_.get_state_device_id(high_node->tensor_id);
        if (low_owner >= 0 && high_owner >= 0 && low_owner == high_owner) {
            pair_device = low_owner;
        } else if (low_owner >= 0 && high_owner < 0) {
            pair_device = low_owner;
        } else if (high_owner >= 0 && low_owner < 0) {
            pair_device = high_owner;
        }
        std::vector<int> low_ids;
        std::vector<int> high_ids;
        HDDNode* low_copy = nullptr;
        HDDNode* high_copy = nullptr;
        try {
            low_copy = duplicate_scaled_terminal_node(low_node, low_weight, pair_device);
            low_ids.push_back(low_copy->tensor_id);
            high_copy = duplicate_scaled_terminal_node(high_node, high_weight, pair_device);
            high_ids.push_back(high_copy->tensor_id);

            if (anti_jaynes_cummings) {
                apply_anti_jaynes_cummings_on_mode(
                    &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
            } else {
                apply_jaynes_cummings_on_mode(
                    &state_pool_, low_ids, high_ids, theta, phi, target_qumode, num_qumodes_);
            }
            state_pool_.synchronize_all_devices();
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
                    int pair_device = execution_device_id_;
                    const int low_owner = state_pool_.get_state_device_id(low_node->tensor_id);
                    const int high_owner = state_pool_.get_state_device_id(high_node->tensor_id);
                    if (low_owner >= 0 && high_owner >= 0 && low_owner == high_owner) {
                        pair_device = low_owner;
                    } else if (low_owner >= 0 && high_owner < 0) {
                        pair_device = low_owner;
                    } else if (high_owner >= 0 && low_owner < 0) {
                        pair_device = high_owner;
                    }
                    HDDNode* low_copy = nullptr;
                    HDDNode* high_copy = nullptr;
                    try {
                        low_copy = duplicate_scaled_terminal_node(low_node, low_weight, pair_device);
                        low_ids.push_back(low_copy->tensor_id);
                        high_copy = duplicate_scaled_terminal_node(high_node, high_weight, pair_device);
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
        state_pool_.synchronize_all_devices();
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
                    int pair_device = execution_device_id_;
                    const int low_owner = state_pool_.get_state_device_id(low_node->tensor_id);
                    const int high_owner = state_pool_.get_state_device_id(high_node->tensor_id);
                    if (low_owner >= 0 && high_owner >= 0 && low_owner == high_owner) {
                        pair_device = low_owner;
                    } else if (low_owner >= 0 && high_owner < 0) {
                        pair_device = low_owner;
                    } else if (high_owner >= 0 && low_owner < 0) {
                        pair_device = high_owner;
                    }
                    HDDNode* low_copy = nullptr;
                    HDDNode* high_copy = nullptr;
                    try {
                        low_copy = duplicate_scaled_terminal_node(low_node, low_weight, pair_device);
                        low_ids.push_back(low_copy->tensor_id);
                        high_copy = duplicate_scaled_terminal_node(high_node, high_weight, pair_device);
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
        state_pool_.synchronize_all_devices();
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
