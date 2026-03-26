// circuit_hdd_gaussian.cpp — HDD scale/add, qubit gate recursion

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;


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
            const MixtureGaussianState& mixture =
                symbolic_terminal_states_.at(node->tensor_id);
            // Zero-GPU-copy: share state via refcount, only scale weights
            std::vector<GaussianComponent> scaled_components;
            scaled_components.reserve(mixture.components.size());
            for (const GaussianComponent& comp : mixture.components) {
                GaussianComponent scaled = comp;
                scaled.weight *= weight;
                if (comp.gaussian_state_id >= 0 && gaussian_state_pool_) {
                    gaussian_state_pool_->add_ref(comp.gaussian_state_id);
                }
                scaled_components.push_back(std::move(scaled));
            }

            if (scaled_components.empty()) {
                return node_manager_.create_terminal_node(shared_zero_state_id_);
            }

            const int scaled_terminal_id = allocate_symbolic_terminal_id();
            symbolic_terminal_states_.emplace(
                scaled_terminal_id,
                MixtureGaussianState{std::move(scaled_components)});
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

    const auto free_temporary_components =
        [&](std::vector<GaussianComponent>* comps) {
            if (!gaussian_state_pool_) {
                comps->clear();
                return;
            }
            for (const GaussianComponent& comp : *comps) {
                if (comp.gaussian_state_id >= 0) {
                    gaussian_state_pool_->release_ref(comp.gaussian_state_id);
                }
            }
            comps->clear();
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
                        // Zero-GPU-copy: concat component lists, share states via refcount
                        std::vector<GaussianComponent> combined;

                        auto append_scaled_components =
                            [&](int terminal_id, std::complex<double> scale) {
                                if (std::abs(scale) < 1e-14 || !is_symbolic_terminal_id(terminal_id)) {
                                    return;
                                }
                                const MixtureGaussianState& mixture =
                                    symbolic_terminal_states_.at(terminal_id);
                                for (const GaussianComponent& comp : mixture.components) {
                                    GaussianComponent new_comp = comp;
                                    new_comp.weight *= scale;
                                    if (comp.gaussian_state_id >= 0 && gaussian_state_pool_) {
                                        gaussian_state_pool_->add_ref(comp.gaussian_state_id);
                                    }
                                    combined.push_back(std::move(new_comp));
                                }
                            };

                        append_scaled_components(id1, lhs_weight);
                        append_scaled_components(id2, rhs_weight);

                        if (combined.empty()) {
                            result = node_manager_.create_terminal_node(shared_zero_state_id_);
                            add_memo.emplace(key, result);
                            return result;
                        }

                        if (combined.size() <=
                            static_cast<size_t>(symbolic_branch_limit_)) {
                            const int symbolic_terminal_id = allocate_symbolic_terminal_id();
                            symbolic_terminal_states_.emplace(
                                symbolic_terminal_id,
                                MixtureGaussianState{std::move(combined)});
                            result = node_manager_.create_terminal_node(symbolic_terminal_id);
                            add_memo.emplace(key, result);
                            return result;
                        }

                        FALLBACK_DEBUG_LOG
                            << "[fallback] hdd_add materializing symbolic sum because combined branches would grow to "
                            << combined.size() << std::endl;
                        free_temporary_components(&combined);
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
