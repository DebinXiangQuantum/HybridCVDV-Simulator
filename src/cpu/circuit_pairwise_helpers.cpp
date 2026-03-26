// circuit_pairwise_helpers.cpp — Pairwise hybrid gate storage estimation & helpers

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_state.h"

namespace circuit_internal {

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
    bool* early_release_enabled) {
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

}  // namespace circuit_internal
