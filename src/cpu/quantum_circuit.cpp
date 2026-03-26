// quantum_circuit.cpp — Core: ctor/dtor, async pipeline, build, execute, reset

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_circuit.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "reference_gates.h"
#include "squeezing_gate_gpu.h"
#include "two_mode_gates.h"

using namespace circuit_internal;

// ==================== Async CV Pipeline ====================

void QuantumCircuit::ensure_async_cv_pipeline() {
    if (compute_stream_ && upload_stream_) {
        return;
    }

    CHECK_CUDA(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking));
    for (TargetUploadSlot& slot : target_upload_slots_) {
        if (!slot.upload_ready) {
            CHECK_CUDA(cudaEventCreateWithFlags(&slot.upload_ready, cudaEventDisableTiming));
        }
        if (!slot.reusable) {
            CHECK_CUDA(cudaEventCreateWithFlags(&slot.reusable, cudaEventDisableTiming));
        }
        slot.reusable_recorded = false;
    }
    next_target_upload_slot_ = 0;
    async_cv_work_pending_ = false;
}

void QuantumCircuit::release_async_cv_pipeline() {
    synchronize_async_cv_pipeline();

    for (TargetUploadSlot& slot : target_upload_slots_) {
        if (slot.upload_ready) {
            cudaEventDestroy(slot.upload_ready);
            slot.upload_ready = nullptr;
        }
        if (slot.reusable) {
            cudaEventDestroy(slot.reusable);
            slot.reusable = nullptr;
        }
        slot.reusable_recorded = false;
        slot.device_buffer.release();
        slot.host_buffer.release();
    }

    if (upload_stream_) {
        cudaStreamDestroy(upload_stream_);
        upload_stream_ = nullptr;
    }
    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
        compute_stream_ = nullptr;
    }
    next_target_upload_slot_ = 0;
    async_cv_work_pending_ = false;
}

void QuantumCircuit::prewarm_async_target_upload_slots() {
    ensure_async_cv_pipeline();

    const size_t target_id_bytes = static_cast<size_t>(state_pool_.capacity) * sizeof(int);
    if (target_id_bytes == 0) {
        return;
    }

    for (TargetUploadSlot& slot : target_upload_slots_) {
        slot.device_buffer.ensure(target_id_bytes);
        slot.host_buffer.ensure(target_id_bytes);
    }
}

void QuantumCircuit::synchronize_async_cv_pipeline() {
    if (!async_cv_work_pending_) {
        return;
    }
    if (upload_stream_) {
        CHECK_CUDA(cudaStreamSynchronize(upload_stream_));
    }
    if (compute_stream_) {
        CHECK_CUDA(cudaStreamSynchronize(compute_stream_));
    }
    async_cv_work_pending_ = false;
}

std::pair<int*, size_t> QuantumCircuit::upload_target_states_for_compute(
    const std::vector<int>& target_states,
    size_t* slot_index) {
    if (target_states.empty()) {
        if (slot_index) {
            *slot_index = 0;
        }
        return {nullptr, 0};
    }

    if (!async_cv_pipeline_enabled_) {
        int* d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
        if (slot_index) {
            *slot_index = 0;
        }
        return {d_target_ids, target_states.size() * sizeof(int)};
    }

    ensure_async_cv_pipeline();
    const size_t ids_bytes = target_states.size() * sizeof(int);
    const size_t current_slot = next_target_upload_slot_;
    TargetUploadSlot& slot = target_upload_slots_[current_slot];

    if (slot.reusable_recorded) {
        CHECK_CUDA(cudaEventSynchronize(slot.reusable));
        slot.reusable_recorded = false;
    }

    int* staged_host = static_cast<int*>(slot.host_buffer.ensure(ids_bytes));
    std::memcpy(staged_host, target_states.data(), ids_bytes);
    int* device_target_ids = static_cast<int*>(slot.device_buffer.ensure(ids_bytes));

    CHECK_CUDA(cudaMemcpyAsync(device_target_ids,
                               staged_host,
                               ids_bytes,
                               cudaMemcpyHostToDevice,
                               upload_stream_));
    CHECK_CUDA(cudaEventRecord(slot.upload_ready, upload_stream_));
    CHECK_CUDA(cudaStreamWaitEvent(compute_stream_, slot.upload_ready, 0));

    next_target_upload_slot_ = (current_slot + 1) % target_upload_slots_.size();
    if (slot_index) {
        *slot_index = current_slot;
    }
    return {device_target_ids, ids_bytes};
}

void QuantumCircuit::mark_target_upload_slot_in_use(size_t slot_index) {
    if (!async_cv_pipeline_enabled_) {
        return;
    }
    if (slot_index >= target_upload_slots_.size()) {
        throw std::out_of_range("target upload slot index out of range");
    }
    TargetUploadSlot& slot = target_upload_slots_[slot_index];
    CHECK_CUDA(cudaEventRecord(slot.reusable, compute_stream_));
    slot.reusable_recorded = true;
    async_cv_work_pending_ = true;
}

void QuantumCircuit::invalidate_root_caches() {
    ++root_revision_;
    cached_target_state_revision_ = 0;
    cached_target_state_ids_.clear();
    cached_symbolic_terminal_revision_ = 0;
    cached_symbolic_terminal_ids_.clear();
}

const std::vector<int>& QuantumCircuit::get_cached_target_states() const {
    if (cached_target_state_revision_ == root_revision_) {
        return cached_target_state_ids_;
    }

    cached_target_state_ids_ = collect_terminal_state_ids(root_node_);
    cached_target_state_ids_.erase(
        std::remove(cached_target_state_ids_.begin(),
                    cached_target_state_ids_.end(),
                    shared_zero_state_id_),
        cached_target_state_ids_.end());
    cached_target_state_revision_ = root_revision_;
    return cached_target_state_ids_;
}

const std::vector<int>& QuantumCircuit::get_cached_symbolic_terminal_ids() const {
    if (cached_symbolic_terminal_revision_ == root_revision_) {
        return cached_symbolic_terminal_ids_;
    }

    cached_symbolic_terminal_ids_ = collect_symbolic_terminal_ids(root_node_);
    cached_symbolic_terminal_revision_ = root_revision_;
    return cached_symbolic_terminal_ids_;
}

// ==================== Constructor & Gaussian State Management ====================

QuantumCircuit::QuantumCircuit(int num_qubits, int num_qumodes, int cv_truncation, int max_states)
    : num_qubits_(num_qubits), num_qumodes_(num_qumodes), cv_truncation_(cv_truncation),
      root_node_(nullptr), state_pool_(cv_truncation, max_states, num_qumodes),
      gaussian_state_pool_(nullptr),
      is_built_(false), is_executed_(false), shared_zero_state_id_(-1),
      total_time_(0.0), transfer_time_(0.0), computation_time_(0.0), planning_time_(0.0),
      gaussian_symbolic_mode_limit_(4), symbolic_branch_limit_(kDefaultSymbolicBranchLimit),
      gaussian_state_pool_capacity_override_(0),
      next_symbolic_terminal_id_(-2),
      pending_gc_replacements_(0) {

    if (num_qubits < 0 || num_qumodes <= 0 || cv_truncation <= 0) {
        throw std::invalid_argument("Qubit数量不能为负数，Qumode数量和截断维度必须为正数");
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
    return !get_cached_symbolic_terminal_ids().empty();
}

void QuantumCircuit::ensure_gaussian_state_pool() {
    if (gaussian_state_pool_) {
        return;
    }
    const int capacity = gaussian_state_pool_capacity_override_ > 0
        ? gaussian_state_pool_capacity_override_
        : std::max(4096, state_pool_.capacity * 16);
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

    size_t offset = 0;
    offset = align_scratch_offset(offset, alignof(int));
    const size_t state_ids_offset = offset;
    offset += gaussian_state_ids.size() * sizeof(int);

    offset = align_scratch_offset(offset, alignof(double));
    const size_t s_offset = offset;
    offset += gate.S.size() * sizeof(double);

    offset = align_scratch_offset(offset, alignof(double));
    const size_t dg_offset = offset;
    offset += gate.d.size() * sizeof(double);

    const int dim = 2 * num_qumodes_;
    offset = align_scratch_offset(offset, alignof(double));
    const size_t old_offset = offset;
    offset += gaussian_state_ids.size() * static_cast<size_t>(dim) * sizeof(double);

    offset = align_scratch_offset(offset, alignof(double));
    const size_t temp_offset = offset;
    offset += gaussian_state_ids.size() * static_cast<size_t>(dim) * dim * sizeof(double);

    char* scratch = static_cast<char*>(state_pool_.scratch_aux.ensure(offset));
    int* d_state_ids = reinterpret_cast<int*>(scratch + state_ids_offset);
    double* d_S = reinterpret_cast<double*>(scratch + s_offset);
    double* d_dg = reinterpret_cast<double*>(scratch + dg_offset);
    double* d_old = reinterpret_cast<double*>(scratch + old_offset);
    double* d_temp = reinterpret_cast<double*>(scratch + temp_offset);

    char* staged = static_cast<char*>(state_pool_.host_transfer_staging.ensure(dg_offset + gate.d.size() * sizeof(double)));
    std::memcpy(staged + state_ids_offset,
                gaussian_state_ids.data(),
                gaussian_state_ids.size() * sizeof(int));
    std::memcpy(staged + s_offset,
                gate.S.data(),
                gate.S.size() * sizeof(double));
    std::memcpy(staged + dg_offset,
                gate.d.data(),
                gate.d.size() * sizeof(double));
    CHECK_CUDA(cudaMemcpy(scratch,
                          staged,
                          dg_offset + gate.d.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    apply_batched_symplectic_update(
        gaussian_state_pool_.get(),
        d_state_ids,
        static_cast<int>(gaussian_state_ids.size()),
        d_S,
        d_dg,
        d_old,
        d_temp,
        nullptr,
        true);
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

// ==================== Gate Addition & Build ====================

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

    async_cv_pipeline_enabled_ = !async_cv_pipeline_disabled();
    if (async_cv_pipeline_enabled_) {
        prewarm_async_target_upload_slots();
    } else {
        std::cout << "异步CV流水线已通过环境变量禁用" << std::endl;
    }

    const size_t target_id_bytes = static_cast<size_t>(state_pool_.capacity) * sizeof(int);
    if (target_id_bytes > 0) {
        state_pool_.scratch_target_ids.ensure(target_id_bytes);
        state_pool_.host_transfer_staging.ensure(target_id_bytes);
    }

    // 初始化HDD结构
    initialize_hdd();

    is_built_ = true;
    std::cout << "量子线路构建完成" << std::endl;
}


// ==================== Execute ====================

void QuantumCircuit::execute() {
    (void)execute_range(0, std::numeric_limits<size_t>::max());
}

size_t QuantumCircuit::get_execution_block_count() const {
    const std::vector<GateParams> execution_sequence = canonicalize_gate_sequence_for_execution();
    return partition_execution_blocks(execution_sequence).size();
}

size_t QuantumCircuit::execute_range(size_t start_block, size_t max_blocks) {
    ScopedNvtxRange nvtx_range("qc::execute");
    if (!is_built_) {
        throw std::runtime_error("必须先构建量子线路");
    }

    if (max_blocks == 0) {
        return start_block;
    }

    if (is_executed_ && start_block == 0) {
        std::cout << "线路已执行，跳过重复执行" << std::endl;
        return get_execution_block_count();
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

    const size_t total_blocks = execution_blocks.size();
    if (execution_blocks.empty()) {
        collect_hdd_garbage_if_needed(true);
        is_executed_ = true;
        auto end_total = std::chrono::high_resolution_clock::now();
        total_time_ = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        return 0;
    }

    if (start_block > total_blocks) {
        throw std::out_of_range("start_block 超出执行块范围");
    }
    if (start_block == total_blocks) {
        is_executed_ = true;
        auto end_total = std::chrono::high_resolution_clock::now();
        total_time_ = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        return total_blocks;
    }

    CompiledExecutionBlock current_block =
        compile_execution_block(execution_sequence, execution_blocks, start_block);
    planning_time_ += current_block.compile_time_ms;

    std::future<CompiledExecutionBlock> next_block_future;

    if (total_blocks > 1) {
        std::cout << "块级编译-执行流水线已启用，块数=" << total_blocks << std::endl;
    }

    size_t executed_blocks = 0;
    for (size_t block_index = start_block;
         block_index < total_blocks && executed_blocks < max_blocks;
         ++block_index, ++executed_blocks) {
        log_block_progress_if_requested(block_index, total_blocks);

        if (block_index + 1 < total_blocks &&
            executed_blocks + 1 < max_blocks &&
            !next_block_future.valid()) {
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
            synchronize_async_cv_pipeline();
            collect_hdd_garbage_if_needed(false);
            if (block_index + 1 < total_blocks &&
                executed_blocks + 1 < max_blocks) {
                current_block = next_block_future.get();
                planning_time_ += current_block.compile_time_ms;
                next_block_future = std::future<CompiledExecutionBlock>();
            }
            continue;
        }

        if (current_block.kind != ExecutionBlockKind::QubitOnly &&
            has_symbolic_terminals()) {
            const auto symbolic_terminal_ids = collect_symbolic_terminal_ids(root_node_);
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

                    size_t extra_pairwise_elements = 0;
                    PairwiseHybridStorageEstimate pairwise_estimate;
                    bool pairwise_early_release_enabled = false;
                    for (const GateParams& gate : current_block.gates) {
                        if (!is_pairwise_hybrid_gate_type(gate.type) ||
                            gate.target_qubits.empty()) {
                            continue;
                        }

                        const PairwiseHybridStorageEstimate candidate =
                            estimate_pairwise_hybrid_storage(
                                root_node_,
                                gate.target_qubits[0],
                                state_dim);
                        const size_t candidate_working_extra =
                            pairwise_hybrid_working_extra_elements(
                                root_node_,
                                gate.target_qubits[0],
                                state_dim,
                                candidate);
                        if (candidate_working_extra > extra_pairwise_elements) {
                            pairwise_estimate = candidate;
                            extra_pairwise_elements = candidate_working_extra;
                            pairwise_early_release_enabled =
                                candidate.extra_elements != 0 &&
                                candidate_working_extra < candidate.extra_elements;
                        }
                    }
                    if (extra_pairwise_elements != 0) {
                        FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                                           << " pre-reserving pairwise hybrid headroom"
                                           << " projected_terminals=" << projected_terminal_count
                                           << " pairwise_pairs=" << pairwise_estimate.pair_count
                                           << " duplicate_states="
                                           << pairwise_estimate.duplicate_state_count
                                           << " working_extra_elements=" << extra_pairwise_elements
                                           << " early_release="
                                           << (pairwise_early_release_enabled ? 1 : 0)
                                           << std::endl;
                    }

                    const size_t reserved_projection_elements =
                        (projected_terminal_count + 1) * state_dim;
                    size_t exact_phase_peak_elements = reserved_projection_elements;
                    if (projected_terminal_count >
                        std::numeric_limits<size_t>::max() / state_dim) {
                        throw std::overflow_error(
                            "pairwise hybrid active exact-state estimate overflow");
                    }
                    const size_t duplicated_gate_peak_elements =
                        projected_terminal_count * state_dim + extra_pairwise_elements;
                    exact_phase_peak_elements =
                        std::max(exact_phase_peak_elements, duplicated_gate_peak_elements);
                    state_pool_.reserve_total_storage_elements(
                        active_storage + exact_phase_peak_elements);
                }
            }
            FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                               << " materializing symbolic terminals to exact Fock"
                               << std::endl;
            materialize_symbolic_terminals_to_fock();
            FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                               << " symbolic materialization complete" << std::endl;
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
                    const auto& target_states = get_cached_target_states();
                    if (!target_states.empty()) {
                        auto transfer_start = std::chrono::high_resolution_clock::now();
                        size_t upload_slot = 0;
                        auto [d_target_ids, ids_bytes] = upload_target_states_for_compute(
                            target_states, &upload_slot);
                        auto transfer_end = std::chrono::high_resolution_clock::now();
                        transfer_time_ += std::chrono::duration<double, std::milli>(
                            transfer_end - transfer_start).count();

                        auto compute_start = std::chrono::high_resolution_clock::now();
                        apply_fused_diagonal_gates(&state_pool_, d_target_ids,
                                                   static_cast<int>(target_states.size()),
                                                   ops_vec, num_qumodes_,
                                                   async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                                   !async_cv_pipeline_enabled_);
                        if (async_cv_pipeline_enabled_) {
                            mark_target_upload_slot_in_use(upload_slot);
                        }
                        auto compute_end = std::chrono::high_resolution_clock::now();
                        computation_time_ += std::chrono::duration<double, std::milli>(
                            compute_end - compute_start).count();
                    }
                }

                // Execute remaining non-fusable gates normally
                FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                                   << " executing exact path after fused diagonal split"
                                   << " kind=" << static_cast<int>(current_block.kind)
                                   << " (remaining gates=" << other_gates.size() << ")"
                                   << std::endl;
                for (size_t gate_index = 0; gate_index < other_gates.size(); ++gate_index) {
                    const GateParams& gate = other_gates[gate_index];
                    FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                                       << " gate " << (gate_index + 1) << "/"
                                       << other_gates.size() << " "
                                       << gate_type_name(gate.type) << std::endl;
                    execute_gate(gate);
                }

                synchronize_async_cv_pipeline();
                collect_hdd_garbage_if_needed(false);
                FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                                   << " exact path complete after fused diagonal split"
                                   << std::endl;

                if (block_index + 1 < total_blocks &&
                    executed_blocks + 1 < max_blocks) {
                    current_block = next_block_future.get();
                    planning_time_ += current_block.compile_time_ms;
                    next_block_future = std::future<CompiledExecutionBlock>();
                }
                continue;
            }
        }
        // ── End fused diagonal ──────────────────────────────────────

        FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                           << " executing exact path with " << current_block.gates.size()
                           << " gates"
                           << " kind=" << static_cast<int>(current_block.kind) << std::endl;
        for (size_t gate_index = 0; gate_index < current_block.gates.size(); ++gate_index) {
            const GateParams& gate = current_block.gates[gate_index];
            FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                               << " gate " << (gate_index + 1) << "/"
                               << current_block.gates.size() << " "
                               << gate_type_name(gate.type) << std::endl;
            execute_gate(gate);
        }

        synchronize_async_cv_pipeline();
        collect_hdd_garbage_if_needed(false);
        FALLBACK_DEBUG_LOG << "[fallback] block " << block_index
                           << " exact path complete" << std::endl;

        if (block_index + 1 < total_blocks &&
            executed_blocks + 1 < max_blocks) {
            current_block = next_block_future.get();
            planning_time_ += current_block.compile_time_ms;
            next_block_future = std::future<CompiledExecutionBlock>();
        }
    }

    // 记录总结束时间
    auto end_total = std::chrono::high_resolution_clock::now();
    total_time_ = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    const size_t next_block_index =
        std::min(start_block + executed_blocks, total_blocks);
    is_executed_ = (next_block_index == total_blocks);
    if (is_executed_) {
        std::cout << "量子线路执行完成" << std::endl;
        std::cout << "执行时间: " << total_time_ << " ms" << std::endl;
        std::cout << "传输时延: " << transfer_time_ << " ms" << std::endl;
        std::cout << "计算时延: " << computation_time_ << " ms" << std::endl;
        std::cout << "规划时延: " << planning_time_ << " ms" << std::endl;
    }
    return next_block_index;
}


// ==================== Reset / Settings / HDD Init ====================

void QuantumCircuit::reset() {
    // 同步所有GPU操作，确保在重置前所有操作完成
    release_async_cv_pipeline();
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
    pending_gc_replacements_ = 0;
    async_cv_pipeline_enabled_ = false;
    invalidate_root_caches();
    
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

void QuantumCircuit::set_symbolic_branch_limit(int limit) {
    if (limit <= 0) {
        throw std::invalid_argument("symbolic branch limit must be positive");
    }
    symbolic_branch_limit_ = limit;
}

void QuantumCircuit::set_gaussian_state_pool_capacity(int capacity) {
    if (capacity <= 0) {
        throw std::invalid_argument("Gaussian state pool capacity must be positive");
    }

    const int desired_capacity = capacity;
    if (gaussian_state_pool_ &&
        gaussian_state_pool_->get_capacity() != desired_capacity) {
        throw std::logic_error(
            "Gaussian state pool capacity must be configured before the pool is initialized");
    }

    gaussian_state_pool_capacity_override_ = desired_capacity;
}

/**
 * 初始化HDD结构
 */
void QuantumCircuit::initialize_hdd() {
    const int vacuum_state_id = state_pool_.allocate_state();
    if (vacuum_state_id < 0) {
        throw std::runtime_error("初始化HDD失败：无法分配初始状态");
    }

    const int64_t total_dim = state_pool_.get_max_total_dim();
    initialize_vacuum_state_device(&state_pool_, vacuum_state_id, total_dim);

    // Pure CV (nq=0): no qubit branching, so skip zero_state allocation to save
    // one full state_dim of GPU memory (critical for nm=8 where state = 68.7GB).
    if (num_qubits_ == 0) {
        shared_zero_state_id_ = -1;
        HDDNode* active_branch = node_manager_.create_terminal_node(vacuum_state_id);
        root_node_ = active_branch;
        invalidate_root_caches();
        return;
    }

    const int zero_state_id = state_pool_.allocate_state();
    if (zero_state_id < 0) {
        throw std::runtime_error("初始化HDD失败：无法分配零状态");
    }

    state_pool_.reserve_state_storage(zero_state_id, total_dim);
    zero_state_device(&state_pool_, zero_state_id);
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
    invalidate_root_caches();
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
    invalidate_root_caches();

    if (old_root) {
        node_manager_.release_node(old_root);
    }
    ++pending_gc_replacements_;

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

    collect_hdd_garbage_if_needed(false);
}

void QuantumCircuit::replace_root_node_preserving_terminals(HDDNode* new_root) {
    if (new_root == root_node_) {
        return;
    }

    HDDNode* old_root = root_node_;
    root_node_ = new_root;
    invalidate_root_caches();

    if (old_root) {
        node_manager_.release_node(old_root);
    }
    ++pending_gc_replacements_;
    collect_hdd_garbage_if_needed(false);
}

void QuantumCircuit::collect_hdd_garbage_if_needed(bool force) {
    constexpr size_t kGarbageCollectReplacementInterval = 32;
    if (!force && pending_gc_replacements_ < kGarbageCollectReplacementInterval) {
        return;
    }

    size_t previous_cache_size = 0;
    do {
        previous_cache_size = node_manager_.get_cache_size();
        node_manager_.garbage_collect();
    } while (node_manager_.get_cache_size() < previous_cache_size);

    pending_gc_replacements_ = 0;
}

/**
 * 执行单个门操作
 */

// ==================== Gate Dispatch Router ====================

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
            synchronize_async_cv_pipeline();
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
            synchronize_async_cv_pipeline();
            execute_hybrid_gate(gate);
            break;

        default:
            throw std::runtime_error("不支持的门类型");
    }
}

/**
 * 执行Level 0门 (对角门)
 */

