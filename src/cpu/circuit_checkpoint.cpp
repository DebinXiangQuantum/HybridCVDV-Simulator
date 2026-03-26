// circuit_checkpoint.cpp — Exact Fock checkpoint save/load

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

void QuantumCircuit::save_exact_fock_checkpoint(const std::string& path,
                                                size_t next_block_index,
                                                size_t total_blocks) const {
    if (!is_built_ || !root_node_) {
        throw std::runtime_error("无法保存checkpoint：线路未构建或根节点为空");
    }
    if (has_symbolic_terminals()) {
        throw std::runtime_error("当前checkpoint仅支持exact Fock状态，不支持symbolic terminals");
    }

    std::vector<int> state_ids = collect_terminal_state_ids(root_node_);
    if (shared_zero_state_id_ >= 0 &&
        std::find(state_ids.begin(), state_ids.end(), shared_zero_state_id_) == state_ids.end()) {
        state_ids.push_back(shared_zero_state_id_);
        std::sort(state_ids.begin(), state_ids.end());
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("无法打开checkpoint文件用于写入: " + path);
    }

    ExactFockCheckpointHeader header{};
    std::memcpy(header.magic, "HCVDVCK1", sizeof(header.magic));
    header.version = 1;
    header.num_qubits = static_cast<uint32_t>(num_qubits_);
    header.num_qumodes = static_cast<uint32_t>(num_qumodes_);
    header.cv_truncation = static_cast<uint32_t>(cv_truncation_);
    header.max_states = static_cast<uint32_t>(state_pool_.capacity);
    header.shared_zero_state_id = shared_zero_state_id_;
    header.next_block_index = static_cast<uint64_t>(next_block_index);
    header.total_blocks = static_cast<uint64_t>(total_blocks);
    header.state_count = static_cast<uint64_t>(state_ids.size());

    std::vector<ExactFockCheckpointStateRecord> state_records;
    state_records.reserve(state_ids.size());
    std::vector<std::vector<cuDoubleComplex>> state_payloads;
    state_payloads.reserve(state_ids.size());
    for (int state_id : state_ids) {
        std::vector<cuDoubleComplex> host_state;
        state_pool_.download_state(state_id, host_state);
        state_records.push_back(
            ExactFockCheckpointStateRecord{state_id, static_cast<int32_t>(host_state.size())});
        state_payloads.push_back(std::move(host_state));
    }

    std::vector<ExactFockCheckpointNodeRecord> node_records;
    std::unordered_map<HDDNode*, uint64_t> node_indices;
    std::function<uint64_t(HDDNode*)> serialize_node =
        [&](HDDNode* node) -> uint64_t {
            const auto cached = node_indices.find(node);
            if (cached != node_indices.end()) {
                return cached->second;
            }

            ExactFockCheckpointNodeRecord record{};
            if (node->is_terminal()) {
                record.is_terminal = 1;
                record.qubit_level = -1;
                record.tensor_id = node->tensor_id;
            } else {
                record.is_terminal = 0;
                record.qubit_level = node->qubit_level;
                record.low_index = serialize_node(node->low);
                record.high_index = serialize_node(node->high);
                record.w_low_real = node->w_low.real();
                record.w_low_imag = node->w_low.imag();
                record.w_high_real = node->w_high.real();
                record.w_high_imag = node->w_high.imag();
            }

            const uint64_t index = static_cast<uint64_t>(node_records.size());
            node_records.push_back(record);
            node_indices.emplace(node, index);
            return index;
        };
    header.root_index = serialize_node(root_node_);
    header.node_count = static_cast<uint64_t>(node_records.size());

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (size_t i = 0; i < state_records.size(); ++i) {
        out.write(reinterpret_cast<const char*>(&state_records[i]), sizeof(state_records[i]));
        const std::vector<cuDoubleComplex>& payload = state_payloads[i];
        if (!payload.empty()) {
            out.write(reinterpret_cast<const char*>(payload.data()),
                      static_cast<std::streamsize>(payload.size() * sizeof(cuDoubleComplex)));
        }
    }
    for (const ExactFockCheckpointNodeRecord& record : node_records) {
        out.write(reinterpret_cast<const char*>(&record), sizeof(record));
    }
    if (!out) {
        throw std::runtime_error("写入checkpoint失败: " + path);
    }
}

size_t QuantumCircuit::load_exact_fock_checkpoint(const std::string& path, size_t* total_blocks) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法打开checkpoint文件: " + path);
    }

    ExactFockCheckpointHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) {
        throw std::runtime_error("读取checkpoint头失败: " + path);
    }
    if (std::memcmp(header.magic, "HCVDVCK1", sizeof(header.magic)) != 0 || header.version != 1) {
        throw std::runtime_error("checkpoint格式不受支持: " + path);
    }
    if (header.num_qubits != static_cast<uint32_t>(num_qubits_) ||
        header.num_qumodes != static_cast<uint32_t>(num_qumodes_) ||
        header.cv_truncation != static_cast<uint32_t>(cv_truncation_) ||
        header.max_states != static_cast<uint32_t>(state_pool_.capacity)) {
        throw std::runtime_error("checkpoint配置与当前线路不匹配");
    }

    synchronize_async_cv_pipeline();
    if (root_node_) {
        node_manager_.release_node(root_node_);
        root_node_ = nullptr;
    }
    node_manager_.clear();
    clear_symbolic_terminals();
    state_pool_.reset();
    invalidate_root_caches();
    is_executed_ = false;
    pending_gc_replacements_ = 0;

    std::unordered_map<int32_t, int32_t> state_id_map;
    for (uint64_t state_index = 0; state_index < header.state_count; ++state_index) {
        ExactFockCheckpointStateRecord record{};
        in.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!in || record.state_dim < 0) {
            throw std::runtime_error("读取checkpoint状态元数据失败: " + path);
        }

        std::vector<cuDoubleComplex> payload(static_cast<size_t>(record.state_dim));
        if (!payload.empty()) {
            in.read(reinterpret_cast<char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size() * sizeof(cuDoubleComplex)));
            if (!in) {
                throw std::runtime_error("读取checkpoint状态数据失败: " + path);
            }
        }

        const int new_state_id = state_pool_.allocate_state();
        if (new_state_id < 0) {
            throw std::runtime_error("加载checkpoint失败：状态池容量不足");
        }
        state_pool_.upload_state(new_state_id, payload);
        state_id_map.emplace(record.state_id, new_state_id);
    }

    if (header.node_count == 0 || header.root_index >= header.node_count) {
        throw std::runtime_error("checkpoint节点图无效");
    }

    std::vector<HDDNode*> rebuilt_nodes(static_cast<size_t>(header.node_count), nullptr);
    for (uint64_t node_index = 0; node_index < header.node_count; ++node_index) {
        ExactFockCheckpointNodeRecord record{};
        in.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!in) {
            throw std::runtime_error("读取checkpoint节点失败: " + path);
        }

        if (record.is_terminal != 0) {
            const auto mapped = state_id_map.find(record.tensor_id);
            if (mapped == state_id_map.end()) {
                throw std::runtime_error("checkpoint引用了未知状态ID");
            }
            rebuilt_nodes[static_cast<size_t>(node_index)] =
                node_manager_.create_terminal_node(mapped->second);
        } else {
            if (record.low_index >= node_index || record.high_index >= node_index) {
                throw std::runtime_error("checkpoint节点拓扑顺序无效");
            }
            rebuilt_nodes[static_cast<size_t>(node_index)] =
                node_manager_.get_or_create_node(
                    record.qubit_level,
                    rebuilt_nodes[static_cast<size_t>(record.low_index)],
                    rebuilt_nodes[static_cast<size_t>(record.high_index)],
                    std::complex<double>(record.w_low_real, record.w_low_imag),
                    std::complex<double>(record.w_high_real, record.w_high_imag));
        }
    }

    root_node_ = rebuilt_nodes[static_cast<size_t>(header.root_index)];
    shared_zero_state_id_ = -1;
    if (header.shared_zero_state_id >= 0) {
        const auto mapped = state_id_map.find(header.shared_zero_state_id);
        if (mapped == state_id_map.end()) {
            throw std::runtime_error("checkpoint缺少shared zero state");
        }
        shared_zero_state_id_ = mapped->second;
    }
    if (total_blocks) {
        *total_blocks = static_cast<size_t>(header.total_blocks);
    }
    return static_cast<size_t>(header.next_block_index);
}

/**
 * 重置量子线路状态
 */
