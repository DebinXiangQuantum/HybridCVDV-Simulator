#include "batch_scheduler.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

// 包含GPU内核头文件
void apply_phase_rotation(CVStatePool* pool, const int* targets, int batch_size, double theta);
void apply_kerr_gate(CVStatePool* pool, const int* targets, int batch_size, double chi);
void apply_conditional_parity(CVStatePool* pool, const int* targets, int batch_size, double parity);
void apply_creation_operator(CVStatePool* pool, const int* targets, int batch_size);
void apply_annihilation_operator(CVStatePool* pool, const int* targets, int batch_size);
void apply_displacement_gate(CVStatePool* pool, const int* targets, int batch_size,
                            cuDoubleComplex alpha);
void apply_single_mode_gate(CVStatePool* pool, FockELLOperator* ell_op,
                           const int* targets, int batch_size);

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
}

/**
 * 检查是否可以添加到当前批次
 */
bool BatchScheduler::can_add_to_batch(const BatchTask& task) const {
    if (pending_batch_.empty()) return true;

    // 检查门类型是否相同 (简化版本)
    if (task.gate_type != pending_batch_.front().gate_type) return false;

    // 检查参数是否相同
    if (task.params.size() != pending_batch_.front().params.size()) return false;
    for (size_t i = 0; i < task.params.size(); ++i) {
        if (std::abs(task.params[i] - pending_batch_.front().params[i]) > 1e-10) return false;
    }

    // 检查内存限制
    size_t new_memory = current_batch_memory_ + task.target_state_ids.size();
    if (new_memory > max_batch_size_ * 10) return false;  // 经验值

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

    // 简化的合并逻辑：将所有任务合并为一个大任务
    BatchTask merged_task = tasks.front();

    for (size_t i = 1; i < tasks.size(); ++i) {
        merged_task.target_state_ids.insert(
            merged_task.target_state_ids.end(),
            tasks[i].target_state_ids.begin(),
            tasks[i].target_state_ids.end()
        );
    }

    // 去重状态ID
    std::sort(merged_task.target_state_ids.begin(), merged_task.target_state_ids.end());
    auto last = std::unique(merged_task.target_state_ids.begin(), merged_task.target_state_ids.end());
    merged_task.target_state_ids.erase(last, merged_task.target_state_ids.end());

    merged.push_back(merged_task);
    return merged;
}

/**
 * 执行Level 0门批次
 */
void BatchScheduler::execute_level0_batch(const std::vector<BatchTask>& batch) {
    for (const auto& task : batch) {
        if (task.target_state_ids.empty()) continue;

        // 上传状态ID到GPU
        int* d_target_ids = nullptr;
        cudaMalloc(&d_target_ids, task.target_state_ids.size() * sizeof(int));
        cudaMemcpy(d_target_ids, task.target_state_ids.data(),
                   task.target_state_ids.size() * sizeof(int), cudaMemcpyHostToDevice);

        double param = task.params.empty() ? 0.0 : task.params[0].real();

        switch (task.gate_type) {
            case GateType::PHASE_ROTATION:
                apply_phase_rotation(state_pool_, d_target_ids, task.target_state_ids.size(), param);
                break;
            case GateType::KERR_GATE:
                apply_kerr_gate(state_pool_, d_target_ids, task.target_state_ids.size(), param);
                break;
            case GateType::CONDITIONAL_PARITY:
                apply_conditional_parity(state_pool_, d_target_ids, task.target_state_ids.size(), param);
                break;
            default:
                break;
        }

        cudaFree(d_target_ids);
    }
}

/**
 * 执行Level 1门批次
 */
void BatchScheduler::execute_level1_batch(const std::vector<BatchTask>& batch) {
    for (const auto& task : batch) {
        if (task.target_state_ids.empty()) continue;

        int* d_target_ids = nullptr;
        cudaMalloc(&d_target_ids, task.target_state_ids.size() * sizeof(int));
        cudaMemcpy(d_target_ids, task.target_state_ids.data(),
                   task.target_state_ids.size() * sizeof(int), cudaMemcpyHostToDevice);

        switch (task.gate_type) {
            case GateType::CREATION_OPERATOR:
                apply_creation_operator(state_pool_, d_target_ids, task.target_state_ids.size());
                break;
            case GateType::ANNIHILATION_OPERATOR:
                apply_annihilation_operator(state_pool_, d_target_ids, task.target_state_ids.size());
                break;
            default:
                break;
        }

        cudaFree(d_target_ids);
    }
}

/**
 * 执行Level 2门批次
 */
void BatchScheduler::execute_level2_batch(const std::vector<BatchTask>& batch) {
    for (const auto& task : batch) {
        if (task.target_state_ids.empty()) continue;

        int* d_target_ids = nullptr;
        cudaMalloc(&d_target_ids, task.target_state_ids.size() * sizeof(int));
        cudaMemcpy(d_target_ids, task.target_state_ids.data(),
                   task.target_state_ids.size() * sizeof(int), cudaMemcpyHostToDevice);

        if (task.gate_type == GateType::DISPLACEMENT && !task.params.empty()) {
            cuDoubleComplex alpha = make_cuDoubleComplex(task.params[0].real(), task.params[0].imag());
            apply_displacement_gate(state_pool_, d_target_ids, task.target_state_ids.size(), alpha);
        }

        cudaFree(d_target_ids);
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

/**
 * 调度单个门操作
 */
void RuntimeScheduler::schedule_gate(const GateParams& gate) {
    if (fusion_enabled_) {
        instruction_fusion_.add_instruction(gate);

        if (auto_flush_enabled_ && instruction_fusion_.buffer_size() >= 5) {
            auto fused_gates = instruction_fusion_.fuse_instructions();
            schedule_gates(fused_gates);
            instruction_fusion_.clear();
        }
    } else {
        schedule_gate(gate);
    }
}

/**
 * 调度多个门操作
 */
void RuntimeScheduler::schedule_gates(const std::vector<GateParams>& gates) {
    for (const auto& gate : gates) {
        schedule_gate(gate);
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

