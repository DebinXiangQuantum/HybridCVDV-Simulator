#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <functional>
#include "quantum_circuit.h"

/**
 * 批处理任务结构体
 */
struct BatchTask {
    std::vector<int> target_state_ids;    // 目标状态ID列表
    GateType gate_type;                   // 门类型
    std::vector<std::complex<double>> params;  // 门参数
    int priority;                         // 任务优先级 (0=最高)

    // 构造函数
    BatchTask(GateType type, const std::vector<int>& state_ids,
              const std::vector<std::complex<double>>& p = {}, int prio = 0)
        : target_state_ids(state_ids), gate_type(type), params(p), priority(prio) {}

    // 比较运算符 (用于优先级队列)
    bool operator<(const BatchTask& other) const {
        return priority > other.priority;  // 优先级小的先执行
    }
};

/**
 * 批处理调度器
 * 负责收集、融合和批量执行GPU内核
 */
class BatchScheduler {
private:
    std::priority_queue<BatchTask> task_queue_;  // 任务队列
    std::vector<BatchTask> pending_batch_;       // 当前批次待处理任务
    size_t max_batch_size_;                      // 最大批次大小
    size_t current_batch_memory_;                // 当前批次内存使用量

    CVStatePool* state_pool_;                    // 状态池引用

    // 统计信息
    size_t total_tasks_processed_;
    size_t total_batches_executed_;
    double total_execution_time_;

public:
    /**
     * 构造函数
     * @param state_pool 状态池引用
     * @param max_batch_size 最大批次大小
     */
    BatchScheduler(CVStatePool* state_pool, size_t max_batch_size = 64);

    /**
     * 添加任务到调度器
     */
    void add_task(const BatchTask& task);

    /**
     * 批量添加任务
     */
    void add_tasks(const std::vector<BatchTask>& tasks);

    /**
     * 执行所有待处理任务
     */
    void execute_pending_tasks();

    /**
     * 强制执行当前批次 (无论大小)
     */
    void flush_batch();

    /**
     * 清空所有任务
     */
    void clear();

    /**
     * 获取统计信息
     */
    struct SchedulerStats {
        size_t total_tasks;
        size_t total_batches;
        double total_time;
        double avg_batch_size;
        double throughput;  // tasks/second
    };
    SchedulerStats get_stats() const;

    /**
     * 设置最大批次大小
     */
    void set_max_batch_size(size_t size) { max_batch_size_ = size; }

private:
    /**
     * 检查是否可以添加到当前批次
     */
    bool can_add_to_batch(const BatchTask& task) const;

    /**
     * 执行单个批次
     */
    void execute_batch(const std::vector<BatchTask>& batch);

    /**
     * 执行Level 0门批次
     */
    void execute_level0_batch(const std::vector<BatchTask>& batch);

    /**
     * 执行Level 1门批次
     */
    void execute_level1_batch(const std::vector<BatchTask>& batch);

    /**
     * 执行Level 2门批次
     */
    void execute_level2_batch(const std::vector<BatchTask>& batch);

    /**
     * 合并相同类型的任务
     */
    std::vector<BatchTask> merge_similar_tasks(const std::vector<BatchTask>& tasks);
};

/**
 * 指令融合器
 * 负责检测和合并可以融合的连续指令
 */
class InstructionFusion {
private:
    std::vector<GateParams> instruction_buffer_;  // 指令缓冲区
    size_t max_fusion_window_;                    // 最大融合窗口大小

    // 融合规则
    bool can_fuse_displacements_;                 // 是否融合位移门
    double displacement_fusion_threshold_;       // 位移融合阈值

public:
    /**
     * 构造函数
     */
    InstructionFusion(size_t max_window = 10);

    /**
     * 添加指令到融合缓冲区
     */
    void add_instruction(const GateParams& instruction);

    /**
     * 尝试融合缓冲区中的指令
     * @return 融合后的指令列表
     */
    std::vector<GateParams> fuse_instructions();

    /**
     * 清空缓冲区
     */
    void clear();

    /**
     * 获取当前缓冲区大小
     */
    size_t buffer_size() const { return instruction_buffer_.size(); }

    /**
     * 设置融合选项
     */
    void enable_displacement_fusion(bool enable, double threshold = 1e-6) {
        can_fuse_displacements_ = enable;
        displacement_fusion_threshold_ = threshold;
    }

private:
    /**
     * 检查两个位移门是否可以融合
     */
    bool can_fuse_displacements(const GateParams& g1, const GateParams& g2) const;

    /**
     * 融合两个位移门
     */
    GateParams fuse_displacements(const GateParams& g1, const GateParams& g2) const;

    /**
     * 检查指令序列是否可以融合
     */
    bool can_fuse_sequence(const std::vector<GateParams>& sequence) const;

    /**
     * 执行指令融合
     */
    std::vector<GateParams> perform_fusion(const std::vector<GateParams>& sequence);
};

/**
 * 完整的运行时调度器
 * 结合批处理和指令融合
 */
class RuntimeScheduler {
private:
    BatchScheduler batch_scheduler_;
    InstructionFusion instruction_fusion_;
    QuantumCircuit* circuit_;  // 关联的量子电路

    bool fusion_enabled_;
    bool auto_flush_enabled_;

    // 私有辅助：直接将门转换为批处理任务并入队
    void enqueue_gate(const GateParams& gate);

public:
    /**
     * 构造函数
     * @param circuit 关联的量子电路
     * @param max_batch_size 最大批次大小
     */
    RuntimeScheduler(QuantumCircuit* circuit, size_t max_batch_size = 64);

    /**
     * 调度单个门操作
     */
    void schedule_gate(const GateParams& gate);

    /**
     * 调度多个门操作
     */
    void schedule_gates(const std::vector<GateParams>& gates);

    /**
     * 执行所有待处理操作
     */
    void execute_all();

    /**
     * 同步执行 (立即执行所有待处理操作)
     */
    void synchronize();

    /**
     * 获取调度器统计信息
     */
    struct RuntimeStats {
        BatchScheduler::SchedulerStats batch_stats;
        size_t fusion_buffer_size;
        bool fusion_enabled;
        bool auto_flush_enabled;
    };
    RuntimeStats get_stats() const;

    /**
     * 配置调度选项
     */
    void enable_fusion(bool enable) { fusion_enabled_ = enable; }
    void enable_auto_flush(bool enable) { auto_flush_enabled_ = enable; }
    void set_max_batch_size(size_t size) { batch_scheduler_.set_max_batch_size(size); }
};
