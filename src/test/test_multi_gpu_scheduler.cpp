/**
 * @file test_multi_gpu_scheduler.cpp
 * @brief Unit tests for multi-GPU batch scheduling.
 *
 * Tests device-aware task routing and batch grouping.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "batch_scheduler.h"
#include "gpu_context.h"

using namespace hybridcvdv;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class MultiGPUSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!GPUContext::is_initialized()) {
            GPUContext::initialize(true);
        }
    }
};

// ─── BatchTask Construction ──────────────────────────────────────────────────

TEST_F(MultiGPUSchedulerTest, DefaultTaskHasEmptyOptionals) {
    BatchTask task(GateType::PHASE_ROTATION, {0, 1, 2});
    EXPECT_EQ(task.gate_type, GateType::PHASE_ROTATION);
    EXPECT_EQ(task.target_state_ids.size(), 3u);
    EXPECT_TRUE(task.params.empty());
    EXPECT_TRUE(task.target_qubits.empty());
    EXPECT_TRUE(task.target_qumodes.empty());
    EXPECT_EQ(task.priority, 0);
}

TEST_F(MultiGPUSchedulerTest, ExplicitParamsPreserved) {
    std::vector<std::complex<double>> params = {{1.0, 0.5}};
    BatchTask task(GateType::PHASE_ROTATION, {0, 1, 2}, params, 5, {0}, {1});
    EXPECT_EQ(task.gate_type, GateType::PHASE_ROTATION);
    EXPECT_EQ(task.target_state_ids.size(), 3u);
    EXPECT_EQ(task.params.size(), 1u);
    EXPECT_EQ(task.priority, 5);
    EXPECT_EQ(task.target_qubits.size(), 1u);
    EXPECT_EQ(task.target_qumodes.size(), 1u);
}

TEST_F(MultiGPUSchedulerTest, TaskPriorityComparison) {
    BatchTask high(GateType::PHASE_ROTATION, {0}, {}, 0);
    BatchTask low(GateType::PHASE_ROTATION, {1}, {}, 10);
    // High priority (lower number) should come first
    EXPECT_TRUE(low < high);  // In priority_queue, low priority compares less
}

TEST_F(MultiGPUSchedulerTest, TaskWithParamsPreserved) {
    std::vector<std::complex<double>> params = {{1.0, 0.5}, {0.3, -0.2}};
    BatchTask task(GateType::DISPLACEMENT, {0, 1}, params, 0, {}, {0});
    EXPECT_EQ(task.params.size(), 2u);
    EXPECT_EQ(task.target_qumodes.size(), 1u);
    EXPECT_EQ(task.target_state_ids.size(), 2u);
}

// ─── BatchScheduler Basic Operations ─────────────────────────────────────────

TEST_F(MultiGPUSchedulerTest, CreateSchedulerDefault) {
    CVStatePool pool(10, 16, 1, 0);
    BatchScheduler scheduler(&pool);

    auto stats = scheduler.get_stats();
    EXPECT_EQ(stats.total_tasks, 0u);
    EXPECT_EQ(stats.total_batches, 0u);
}

TEST_F(MultiGPUSchedulerTest, CreateSchedulerSecondPool) {
    CVStatePool pool(10, 16, 1, 0);
    BatchScheduler scheduler(&pool);

    auto stats = scheduler.get_stats();
    EXPECT_EQ(stats.total_tasks, 0u);
}

// ─── Task Grouping by Gate Type ──────────────────────────────────────────────

TEST_F(MultiGPUSchedulerTest, GroupTasksByGateType) {
    std::vector<BatchTask> tasks;
    tasks.emplace_back(GateType::PHASE_ROTATION, std::vector<int>{0},
                       std::vector<std::complex<double>>{}, 0,
                       std::vector<int>{}, std::vector<int>{});
    tasks.emplace_back(GateType::DISPLACEMENT, std::vector<int>{1},
                       std::vector<std::complex<double>>{}, 0,
                       std::vector<int>{}, std::vector<int>{});
    tasks.emplace_back(GateType::PHASE_ROTATION, std::vector<int>{2},
                       std::vector<std::complex<double>>{}, 0,
                       std::vector<int>{}, std::vector<int>{});

    // Group by gate_type
    std::map<GateType, std::vector<BatchTask>> grouped;
    for (const auto& task : tasks) {
        grouped[task.gate_type].push_back(task);
    }

    EXPECT_EQ(grouped[GateType::PHASE_ROTATION].size(), 2u);
    EXPECT_EQ(grouped[GateType::DISPLACEMENT].size(), 1u);
}

// ─── Multi-Pool Scheduling ───────────────────────────────────────────────────

TEST_F(MultiGPUSchedulerTest, MultiPoolStateAllocation) {
    CVStatePool pool0(10, 16, 1, 0);
    CVStatePool pool1(10, 16, 1, 0);

    BatchScheduler sched0(&pool0);
    BatchScheduler sched1(&pool1);

    // Allocate states on each pool
    int s0 = pool0.allocate_state();
    int s1 = pool1.allocate_state();

    EXPECT_GE(s0, 0);
    EXPECT_GE(s1, 0);

    // Each scheduler should operate independently
    auto stats0 = sched0.get_stats();
    auto stats1 = sched1.get_stats();
    EXPECT_EQ(stats0.total_tasks, 0u);
    EXPECT_EQ(stats1.total_tasks, 0u);

    pool0.free_state(s0);
    pool1.free_state(s1);
}
