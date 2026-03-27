#include "batch_scheduler.h"
#include "squeezing_gate_gpu.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// 包含GPU内核头文件
void apply_phase_rotation_on_mode(CVStatePool* pool, const int* targets, int batch_size, double theta,
                                  int target_qumode, int num_qumodes,
                                  cudaStream_t stream = nullptr, bool synchronize = true);
void apply_kerr_gate_on_mode(CVStatePool* pool, const int* targets, int batch_size, double chi,
                             int target_qumode, int num_qumodes,
                             cudaStream_t stream = nullptr, bool synchronize = true);
void apply_conditional_parity_on_mode(CVStatePool* pool, const int* targets, int batch_size, double parity,
                                      int target_qumode, int num_qumodes,
                                      cudaStream_t stream = nullptr, bool synchronize = true);
void apply_creation_operator_on_mode(CVStatePool* pool, const int* targets, int batch_size,
                                     int target_qumode, int num_qumodes,
                                     cudaStream_t stream = nullptr, bool synchronize = true);
void apply_annihilation_operator_on_mode(CVStatePool* pool, const int* targets, int batch_size,
                                         int target_qumode, int num_qumodes,
                                         cudaStream_t stream = nullptr, bool synchronize = true);
void apply_displacement_gate(CVStatePool* pool, const int* targets, int batch_size,
                            cuDoubleComplex alpha,
                            cudaStream_t stream = nullptr, bool synchronize = true);
void apply_single_mode_gate(CVStatePool* pool, FockELLOperator* ell_op,
                           const int* targets, int batch_size,
                           cudaStream_t stream = nullptr, bool synchronize = true);
void apply_controlled_displacement_on_mode(CVStatePool* state_pool,
                                           const std::vector<int>& controlled_states,
                                           cuDoubleComplex alpha,
                                           int target_qumode,
                                           int num_qumodes);

namespace {

int infer_num_qumodes_from_pool(const CVStatePool* state_pool) {
    if (!state_pool || state_pool->d_trunc <= 1 || state_pool->max_total_dim <= 0) {
        return 1;
    }

    int num_qumodes = 0;
    int64_t dim = 1;
    while (dim < state_pool->max_total_dim) {
        if (dim > state_pool->max_total_dim / state_pool->d_trunc) {
            return 1;
        }
        dim *= state_pool->d_trunc;
        ++num_qumodes;
    }

    return dim == state_pool->max_total_dim ? std::max(1, num_qumodes) : 1;
}

int get_primary_target_qumode(const BatchTask& task) {
    return task.target_qumodes.empty() ? 0 : task.target_qumodes.front();
}

bool same_batch_signature(const BatchTask& lhs, const BatchTask& rhs) {
    return lhs.gate_type == rhs.gate_type &&
           lhs.target_qubits == rhs.target_qubits &&
           lhs.target_qumodes == rhs.target_qumodes &&
           lhs.params == rhs.params;
}

bool is_level0_gate(GateType gate_type) {
    switch (gate_type) {
        case GateType::PHASE_ROTATION:
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
            return true;
        default:
            return false;
    }
}

void check_cuda_status(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(message) + cudaGetErrorString(err));
    }
}

}  // namespace

/**
 * BatchScheduler 构造函数
 */
BatchScheduler::BatchScheduler(CVStatePool* state_pool, size_t max_batch_size)
    : state_pool_(state_pool), max_batch_size_(max_batch_size),
      current_batch_memory_(0), total_tasks_processed_(0),
      total_batches_executed_(0), total_execution_time_(0.0) {

    if (!state_pool) {
        throw std::invalid_argument("状态池指针不能为空");
    }
}

BatchScheduler::~BatchScheduler() {
    reset_graph();
}

/**
 * 添加任务到调度器
 */
void BatchScheduler::add_task(const BatchTask& task) {
    // 检查是否可以添加到当前批次
    if (!pending_batch_.empty() && !can_add_to_batch(task)) {
        // 执行当前批次
        execute_batch(pending_batch_);
        pending_batch_.clear();
        current_batch_memory_ = 0;
    }

    pending_batch_.push_back(task);
    current_batch_memory_ += task.target_state_ids.size();

    // 检查是否达到批次大小限制
    if (pending_batch_.size() >= max_batch_size_) {
        execute_batch(pending_batch_);
        pending_batch_.clear();
        current_batch_memory_ = 0;
    }
}

/**
 * 批量添加任务
 */
void BatchScheduler::add_tasks(const std::vector<BatchTask>& tasks) {
    for (const auto& task : tasks) {
        add_task(task);
    }
}

/**
 * 执行所有待处理任务
 */
void BatchScheduler::execute_pending_tasks() {
    if (!pending_batch_.empty()) {
        execute_batch(pending_batch_);
        pending_batch_.clear();
        current_batch_memory_ = 0;
    }

    // 执行队列中的所有任务
    while (!task_queue_.empty()) {
        BatchTask task = task_queue_.top();
        task_queue_.pop();

        std::vector<BatchTask> single_batch = {task};
        execute_batch(single_batch);
    }
}

/**
 * 强制执行当前批次
 */
void BatchScheduler::flush_batch() {
    if (!pending_batch_.empty()) {
        execute_batch(pending_batch_);
        pending_batch_.clear();
        current_batch_memory_ = 0;
    }
}

/**
 * 清空所有任务
 */
void BatchScheduler::clear() {
    while (!task_queue_.empty()) {
        task_queue_.pop();
    }
    pending_batch_.clear();
    current_batch_memory_ = 0;
    total_tasks_processed_ = 0;
    total_batches_executed_ = 0;
    total_execution_time_ = 0.0;
    reset_graph();
}

// ==========================================
// HPC Optimization: CUDA Graphs 工作流卸载
// ==========================================

void BatchScheduler::execute_with_cuda_graph() {
    if (!pending_batch_.empty()) {
        std::string reason;
        if (!can_capture_pending_batch_with_cuda_graph(&reason)) {
            throw std::runtime_error("CUDA Graph capture unsupported: " + reason);
        }

        reset_graph();
        if (!stream_) {
            check_cuda_status(cudaStreamCreate(&stream_), "Failed to create CUDA stream: ");
        }

        const auto merged_batch = merge_similar_tasks(pending_batch_);
        captured_graph_tasks_.clear();
        captured_graph_tasks_.reserve(merged_batch.size());

        size_t total_target_ids = 0;
        for (const auto& task : merged_batch) {
            total_target_ids += task.target_state_ids.size();
        }
        int* captured_target_base = nullptr;
        if (total_target_ids > 0) {
            captured_target_base = static_cast<int*>(
                captured_graph_target_storage_.ensure(total_target_ids * sizeof(int)));
        }

        size_t target_offset = 0;
        for (const auto& task : merged_batch) {
            CapturedGraphTask prepared{task, nullptr};
            if (!task.target_state_ids.empty()) {
                prepared.d_target_ids = captured_target_base + target_offset;
                int* staged_ids = static_cast<int*>(
                    state_pool_->host_transfer_staging.ensure(
                        task.target_state_ids.size() * sizeof(int)));
                std::memcpy(staged_ids,
                            task.target_state_ids.data(),
                            task.target_state_ids.size() * sizeof(int));
                check_cuda_status(
                    cudaMemcpy(prepared.d_target_ids,
                               staged_ids,
                               task.target_state_ids.size() * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "Failed to upload CUDA Graph target buffer: ");
                target_offset += task.target_state_ids.size();
            }
            captured_graph_tasks_.push_back(prepared);
        }

        check_cuda_status(
            cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
            "Failed to begin CUDA Graph capture: ");
        try {
            execute_prepared_level0_batch(captured_graph_tasks_, stream_);
            check_cuda_status(
                cudaStreamEndCapture(stream_, &exec_graph_),
                "Failed to end CUDA Graph capture: ");
            check_cuda_status(
                cudaGraphInstantiate(&graph_instance_, exec_graph_, nullptr, nullptr, 0),
                "Failed to instantiate CUDA Graph: ");
            graph_captured_ = true;
            pending_batch_.clear();
            current_batch_memory_ = 0;
            std::cout << "[HPC Opt] CUDA Graph 录制完成" << std::endl;
        } catch (...) {
            cudaStreamEndCapture(stream_, &exec_graph_);
            release_captured_graph_tasks();
            throw;
        }
    } else if (!graph_captured_) {
        std::cout << "[HPC Opt] 没有可录制的CUDA Graph任务" << std::endl;
        return;
    }

    // 极速下发已录制的 CUDA Graph
    auto start_time = std::chrono::high_resolution_clock::now();

    check_cuda_status(cudaGraphLaunch(graph_instance_, stream_), "Failed to launch CUDA Graph: ");
    check_cuda_status(cudaStreamSynchronize(stream_), "Failed to synchronize CUDA Graph stream: ");

    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();

    // 这里做简单的性能统计，不准确累计 tasks 数量因为那是图内部的事
    total_execution_time_ += execution_time;
    total_batches_executed_ += 1;

    std::cout << "[HPC Opt] CUDA Graph 极速下发完成: 耗时 " 
              << execution_time * 1000 << " ms" << std::endl;
}

void BatchScheduler::reset_graph() {
    if (graph_captured_) {
        cudaGraphExecDestroy(graph_instance_);
        cudaGraphDestroy(exec_graph_);
        graph_captured_ = false;
    }
    release_captured_graph_tasks();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

bool BatchScheduler::can_capture_pending_batch_with_cuda_graph(std::string* reason) const {
    if (!task_queue_.empty()) {
        if (reason) {
            *reason = "priority queue tasks are not capture-safe yet";
        }
        return false;
    }
    if (pending_batch_.empty()) {
        if (reason) {
            *reason = "pending batch is empty";
        }
        return false;
    }

    for (const auto& task : pending_batch_) {
        if (!is_level0_gate(task.gate_type)) {
            if (reason) {
                *reason = "only Level 0 diagonal gates are supported, got gate type " +
                          std::to_string(static_cast<int>(task.gate_type));
            }
            return false;
        }
        if (task.target_state_ids.empty()) {
            if (reason) {
                *reason = "batch task has no target states";
            }
            return false;
        }
    }

    return true;
}

void BatchScheduler::release_captured_graph_tasks() {
    captured_graph_tasks_.clear();
    captured_graph_target_storage_.release();
}

void BatchScheduler::execute_prepared_level0_batch(const std::vector<CapturedGraphTask>& batch,
                                                   cudaStream_t stream) {
    const int num_qumodes = infer_num_qumodes_from_pool(state_pool_);

    for (const auto& prepared : batch) {
        if (!prepared.d_target_ids || prepared.task.target_state_ids.empty()) {
            continue;
        }

        const BatchTask& task = prepared.task;
        const double param = task.params.empty() ? 0.0 : task.params[0].real();
        const int target_qumode = get_primary_target_qumode(task);

        switch (task.gate_type) {
            case GateType::PHASE_ROTATION:
                apply_phase_rotation_on_mode(
                    state_pool_,
                    prepared.d_target_ids,
                    static_cast<int>(task.target_state_ids.size()),
                    param,
                    target_qumode,
                    num_qumodes,
                    stream,
                    false);
                break;
            case GateType::KERR_GATE:
                apply_kerr_gate_on_mode(
                    state_pool_,
                    prepared.d_target_ids,
                    static_cast<int>(task.target_state_ids.size()),
                    param,
                    target_qumode,
                    num_qumodes,
                    stream,
                    false);
                break;
            case GateType::CONDITIONAL_PARITY:
                apply_conditional_parity_on_mode(
                    state_pool_,
                    prepared.d_target_ids,
                    static_cast<int>(task.target_state_ids.size()),
                    param,
                    target_qumode,
                    num_qumodes,
                    stream,
                    false);
                break;
            default:
                throw std::runtime_error("unexpected non-Level0 gate during CUDA Graph capture");
        }
    }
}

/**
 * 检查是否可以添加到当前批次
 * 条件：相同门类型、相同目标 qubit/qumode 布局、相同参数、
 * 合并后 state 数不超过 max_batch_size_。
 */
bool BatchScheduler::can_add_to_batch(const BatchTask& task) const {
    if (pending_batch_.empty()) return true;

    const BatchTask& front = pending_batch_.front();

    // 门类型与目标布局必须一致才能合并执行
    if (task.gate_type != front.gate_type) return false;
    if (task.target_qubits != front.target_qubits) return false;
    if (task.target_qumodes != front.target_qumodes) return false;

    // 参数必须完全匹配（同一门参数才能共享内核 launch）
    if (task.params.size() != front.params.size()) return false;
    for (size_t i = 0; i < task.params.size(); ++i) {
        if (std::abs(task.params[i] - front.params[i]) > 1e-10) return false;
    }

    // 批次中总 state 数不超过 max_batch_size_
    size_t total_states = task.target_state_ids.size();
    for (const auto& existing : pending_batch_) {
        total_states += existing.target_state_ids.size();
    }
    if (total_states > max_batch_size_) return false;

    return true;
}

/**
 * 执行单个批次
 */
void BatchScheduler::execute_batch(const std::vector<BatchTask>& batch) {
    if (batch.empty()) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    // 合并相同类型的任务
    auto merged_batch = merge_similar_tasks(batch);

    // 根据门类型分发执行
    GateType gate_type = merged_batch.front().gate_type;

    switch (gate_type) {
        case GateType::PHASE_ROTATION:
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
            execute_level0_batch(merged_batch);
            break;

        case GateType::CREATION_OPERATOR:
        case GateType::ANNIHILATION_OPERATOR:
            execute_level1_batch(merged_batch);
            break;

        case GateType::DISPLACEMENT:
        case GateType::SQUEEZING:
            execute_level2_batch(merged_batch);
            break;

        default:
            std::cerr << "不支持的批处理门类型" << std::endl;
            break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();

    total_execution_time_ += execution_time;
    total_batches_executed_ += 1;
    total_tasks_processed_ += batch.size();

    std::cout << "执行批次完成: " << batch.size() << " 任务, 耗时 "
              << execution_time * 1000 << " ms" << std::endl;
}

/**
 * 合并相同类型的任务
 */
std::vector<BatchTask> BatchScheduler::merge_similar_tasks(const std::vector<BatchTask>& tasks) {
    if (tasks.size() <= 1) return tasks;

    std::vector<BatchTask> merged;

    for (const auto& task : tasks) {
        auto existing = std::find_if(merged.begin(), merged.end(),
                                     [&](const BatchTask& candidate) {
                                         return same_batch_signature(candidate, task);
                                     });

        if (existing == merged.end()) {
            merged.push_back(task);
            continue;
        }

        existing->target_state_ids.insert(existing->target_state_ids.end(),
                                          task.target_state_ids.begin(),
                                          task.target_state_ids.end());
        std::sort(existing->target_state_ids.begin(), existing->target_state_ids.end());
        auto last = std::unique(existing->target_state_ids.begin(), existing->target_state_ids.end());
        existing->target_state_ids.erase(last, existing->target_state_ids.end());
    }

    return merged;
}

/**
 * 执行Level 0门批次
 */
void BatchScheduler::execute_level0_batch(const std::vector<BatchTask>& batch) {
    const int num_qumodes = infer_num_qumodes_from_pool(state_pool_);
    size_t max_ids_bytes = 0;
    for (const auto& task : batch) {
        max_ids_bytes = std::max(max_ids_bytes, task.target_state_ids.size() * sizeof(int));
    }
    if (max_ids_bytes > 0) {
        state_pool_->scratch_target_ids.ensure(max_ids_bytes);
    }

    for (const auto& task : batch) {
        if (task.target_state_ids.empty()) continue;

        // 上传状态ID到GPU (scratch buffer)
        int* d_target_ids = state_pool_->upload_vector_to_buffer(
            task.target_state_ids, state_pool_->scratch_target_ids);

        double param = task.params.empty() ? 0.0 : task.params[0].real();
        const int target_qumode = get_primary_target_qumode(task);

        switch (task.gate_type) {
            case GateType::PHASE_ROTATION:
                apply_phase_rotation_on_mode(state_pool_, d_target_ids, task.target_state_ids.size(), param,
                                             target_qumode, num_qumodes, nullptr, false);
                break;
            case GateType::KERR_GATE:
                apply_kerr_gate_on_mode(state_pool_, d_target_ids, task.target_state_ids.size(), param,
                                        target_qumode, num_qumodes, nullptr, false);
                break;
            case GateType::CONDITIONAL_PARITY:
                apply_conditional_parity_on_mode(state_pool_, d_target_ids, task.target_state_ids.size(), param,
                                                 target_qumode, num_qumodes, nullptr, false);
                break;
            default:
                break;
        }
    }

    check_cuda_status(cudaDeviceSynchronize(), "execute_level0_batch: kernel synchronization failed: ");
}

/**
 * 执行Level 1门批次
 */
void BatchScheduler::execute_level1_batch(const std::vector<BatchTask>& batch) {
    const int num_qumodes = infer_num_qumodes_from_pool(state_pool_);

    for (const auto& task : batch) {
        if (task.target_state_ids.empty()) continue;

        int* d_target_ids = state_pool_->upload_vector_to_buffer(
            task.target_state_ids, state_pool_->scratch_target_ids);

        const int target_qumode = get_primary_target_qumode(task);

        switch (task.gate_type) {
            case GateType::CREATION_OPERATOR:
                apply_creation_operator_on_mode(state_pool_, d_target_ids, task.target_state_ids.size(),
                                                target_qumode, num_qumodes);
                break;
            case GateType::ANNIHILATION_OPERATOR:
                apply_annihilation_operator_on_mode(state_pool_, d_target_ids, task.target_state_ids.size(),
                                                    target_qumode, num_qumodes);
                break;
            default:
                break;
        }
    }
}

/**
 * 执行Level 2门批次
 */
void BatchScheduler::execute_level2_batch(const std::vector<BatchTask>& batch) {
    const int num_qumodes = infer_num_qumodes_from_pool(state_pool_);

    for (const auto& task : batch) {
        if (task.target_state_ids.empty()) continue;

        int* d_target_ids = state_pool_->upload_vector_to_buffer(
            task.target_state_ids, state_pool_->scratch_target_ids);

        const int target_qumode = get_primary_target_qumode(task);

        if (task.gate_type == GateType::DISPLACEMENT && !task.params.empty()) {
            cuDoubleComplex alpha = make_cuDoubleComplex(task.params[0].real(), task.params[0].imag());
            if (num_qumodes > 1 || target_qumode != 0) {
                apply_controlled_displacement_on_mode(state_pool_, task.target_state_ids, alpha,
                                                      target_qumode, num_qumodes);
            } else {
                apply_displacement_gate(state_pool_, d_target_ids, task.target_state_ids.size(), alpha);
            }
        } else if (task.gate_type == GateType::SQUEEZING && !task.params.empty()) {
            apply_squeezing_gate_gpu(state_pool_, d_target_ids, static_cast<int>(task.target_state_ids.size()),
                                     std::abs(task.params[0]), std::arg(task.params[0]),
                                     target_qumode, num_qumodes);
        }
    }
}

/**
 * 获取统计信息
 */
BatchScheduler::SchedulerStats BatchScheduler::get_stats() const {
    double avg_batch_size = total_batches_executed_ > 0 ?
                           static_cast<double>(total_tasks_processed_) / total_batches_executed_ : 0.0;
    double throughput = total_execution_time_ > 0 ?
                       static_cast<double>(total_tasks_processed_) / total_execution_time_ : 0.0;

    return {
        total_tasks_processed_,
        total_batches_executed_,
        total_execution_time_,
        avg_batch_size,
        throughput
    };
}

// ===== InstructionFusion 实现 =====

/**
 * InstructionFusion 构造函数
 */
InstructionFusion::InstructionFusion(size_t max_window)
    : max_fusion_window_(max_window), can_fuse_displacements_(true),
      displacement_fusion_threshold_(1e-6) {}

/**
 * 添加指令到融合缓冲区
 */
void InstructionFusion::add_instruction(const GateParams& instruction) {
    instruction_buffer_.push_back(instruction);

    // 如果缓冲区满了，自动融合
    if (instruction_buffer_.size() >= max_fusion_window_) {
        instruction_buffer_ = fuse_instructions();
    }
}

/**
 * 尝试融合缓冲区中的指令
 */
std::vector<GateParams> InstructionFusion::fuse_instructions() {
    if (instruction_buffer_.size() <= 1) {
        return instruction_buffer_;
    }

    return perform_fusion(instruction_buffer_);
}

/**
 * 清空缓冲区
 */
void InstructionFusion::clear() {
    instruction_buffer_.clear();
}

/**
 * 检查两个位移门是否可以融合
 */
bool InstructionFusion::can_fuse_displacements(const GateParams& g1, const GateParams& g2) const {
    if (g1.type != GateType::DISPLACEMENT || g2.type != GateType::DISPLACEMENT) return false;
    if (g1.target_qumodes != g2.target_qumodes) return false;
    if (g1.target_qubits != g2.target_qubits) return false;

    // 检查参数是否足够小 (可以忽略相位)
    if (g1.params.empty() || g2.params.empty()) return false;

    double alpha1_norm = std::abs(g1.params[0]);
    double alpha2_norm = std::abs(g2.params[0]);

    return alpha1_norm < displacement_fusion_threshold_ &&
           alpha2_norm < displacement_fusion_threshold_;
}

/**
 * 融合两个位移门
 */
GateParams InstructionFusion::fuse_displacements(const GateParams& g1, const GateParams& g2) const {
    // α1 + α2 (忽略相位)
    std::complex<double> fused_alpha = g1.params[0] + g2.params[0];
    return GateParams(GateType::DISPLACEMENT, g1.target_qubits, g1.target_qumodes, {fused_alpha});
}

/**
 * 执行指令融合
 */
std::vector<GateParams> InstructionFusion::perform_fusion(const std::vector<GateParams>& sequence) {
    std::vector<GateParams> fused;

    for (size_t i = 0; i < sequence.size(); ++i) {
        const GateParams& current = sequence[i];

        // 尝试与下一条指令融合
        if (i + 1 < sequence.size() && can_fuse_displacements(current, sequence[i + 1])) {
            GateParams fused_gate = fuse_displacements(current, sequence[i + 1]);
            fused.push_back(fused_gate);
            ++i;  // 跳过下一条指令
        } else {
            fused.push_back(current);
        }
    }

    return fused;
}

// ===== RuntimeScheduler 实现 =====

/**
 * RuntimeScheduler 构造函数
 */
RuntimeScheduler::RuntimeScheduler(QuantumCircuit* circuit, size_t max_batch_size)
    : batch_scheduler_(&circuit->get_state_pool(), max_batch_size),
      circuit_(circuit), fusion_enabled_(true), auto_flush_enabled_(true) {}

// 私有辅助：将单个门转换为 BatchTask 并入队（直接提交到 BatchScheduler）
void RuntimeScheduler::enqueue_gate(const GateParams& gate) {
    // 尝试收集目标状态ID（当前采用简单策略：对指定的qumodes或默认使用所有活跃状态）
    std::vector<int> target_state_ids;
    auto& state_pool = circuit_->get_state_pool();

    // 如果指定了qumodes，理想情况下应将每个qumode映射到具体的物理状态ID；
    // 目前使用一个简单的启发式：使用所有活跃状态作为作用目标（最佳努力）
    target_state_ids = state_pool.get_active_state_ids();

    BatchTask task(gate.type, target_state_ids, gate.params, 0,
                   gate.target_qubits, gate.target_qumodes);
    batch_scheduler_.add_task(task);
}

/**
 * 调度单个门操作
 */
void RuntimeScheduler::schedule_gate(const GateParams& gate) {
    if (fusion_enabled_) {
        instruction_fusion_.add_instruction(gate);

        if (auto_flush_enabled_ && instruction_fusion_.buffer_size() >= 5) {
            auto fused_gates = instruction_fusion_.fuse_instructions();
            // 直接把融合后的门提交到批调度器（绕过再次缓冲），避免递归调用
            for (const auto& fg : fused_gates) {
                enqueue_gate(fg);
            }
            instruction_fusion_.clear();
        }
    } else {
        // 未启用融合时直接提交到批调度器（修复原来的递归调用问题）
        enqueue_gate(gate);
    }
}

/**
 * 调度多个门操作
 */
void RuntimeScheduler::schedule_gates(const std::vector<GateParams>& gates) {
    for (const auto& gate : gates) {
        enqueue_gate(gate);
    }
}

/**
 * 执行所有待处理操作
 */
void RuntimeScheduler::execute_all() {
    if (fusion_enabled_) {
        auto fused_gates = instruction_fusion_.fuse_instructions();
        schedule_gates(fused_gates);
        instruction_fusion_.clear();
    }

    batch_scheduler_.execute_pending_tasks();
}

/**
 * 同步执行
 */
void RuntimeScheduler::synchronize() {
    execute_all();
    cudaDeviceSynchronize();
}

/**
 * 获取统计信息
 */
RuntimeScheduler::RuntimeStats RuntimeScheduler::get_stats() const {
    return {
        batch_scheduler_.get_stats(),
        instruction_fusion_.buffer_size(),
        fusion_enabled_,
        auto_flush_enabled_
    };
}
