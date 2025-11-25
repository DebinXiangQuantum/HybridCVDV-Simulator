#include <gtest/gtest.h>
#include "quantum_circuit.h"
#include "batch_scheduler.h"
#include <iostream>

/**
 * 系统集成测试
 */
class SystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        circuit = new QuantumCircuit(2, 2, 8, 16);  // 2 qubits, 2 qumodes, 截断8, 最大16状态
        circuit->build();
    }

    void TearDown() override {
        delete circuit;
    }

    QuantumCircuit* circuit;
};

TEST_F(SystemTest, CircuitInitialization) {
    auto stats = circuit->get_stats();
    EXPECT_EQ(stats.num_qubits, 2);
    EXPECT_EQ(stats.num_qumodes, 2);
    EXPECT_EQ(stats.cv_truncation, 8);
    EXPECT_EQ(stats.num_gates, 0);
}

TEST_F(SystemTest, AddGates) {
    // 添加一些门操作
    circuit->add_gate(Gates::PhaseRotation(0, M_PI / 4.0));
    circuit->add_gate(Gates::Displacement(0, std::complex<double>(0.1, 0.0)));
    circuit->add_gate(Gates::BeamSplitter(0, 1, M_PI / 2.0));

    auto stats = circuit->get_stats();
    EXPECT_EQ(stats.num_gates, 3);
}

TEST_F(SystemTest, ExecuteCircuit) {
    // 创建一个简单的电路
    circuit->add_gate(Gates::PhaseRotation(0, M_PI));
    circuit->add_gate(Gates::Displacement(0, std::complex<double>(0.1, 0.0)));

    // 执行电路
    circuit->execute();

    // 验证可以获取状态
    std::complex<double> amplitude = circuit->get_amplitude({0}, {});
    EXPECT_NEAR(std::abs(amplitude), 1.0, 1e-10);  // 应该有非零振幅
}

TEST_F(SystemTest, BatchSchedulerIntegration) {
    RuntimeScheduler scheduler(circuit, 4);

    // 添加多个门操作
    scheduler.schedule_gate(Gates::PhaseRotation(0, M_PI / 4.0));
    scheduler.schedule_gate(Gates::Displacement(0, std::complex<double>(0.05, 0.0)));
    scheduler.schedule_gate(Gates::CreationOperator(0));

    // 执行调度
    scheduler.execute_all();

    auto stats = scheduler.get_stats();
    EXPECT_GE(stats.batch_stats.total_tasks, 3);
}

TEST_F(SystemTest, StatePoolIntegration) {
    auto& state_pool = circuit->get_state_pool();

    // 验证状态池初始化
    EXPECT_EQ(state_pool.capacity, 16);
    EXPECT_EQ(state_pool.d_trunc, 8);
    EXPECT_EQ(state_pool.active_count, 1);  // 初始真空态
}

TEST_F(SystemTest, MemoryManagement) {
    // 测试多次执行不会导致内存泄漏
    for (int i = 0; i < 5; ++i) {
        circuit->reset();
        circuit->add_gate(Gates::PhaseRotation(0, M_PI / (i + 1)));
        circuit->build();
        circuit->execute();
    }

    // 验证状态池状态
    auto& state_pool = circuit->get_state_pool();
    EXPECT_EQ(state_pool.active_count, 1);  // 应该只有一个活动状态
}

TEST_F(SystemTest, ComplexCircuit) {
    // 创建一个更复杂的电路
    circuit->add_gates({
        Gates::PhaseRotation(0, M_PI / 4.0),
        Gates::Displacement(0, std::complex<double>(0.2, 0.1)),
        Gates::BeamSplitter(0, 1, M_PI / 3.0),
        Gates::ControlledDisplacement(0, 1, std::complex<double>(0.1, 0.0)),
        Gates::Squeezing(1, std::complex<double>(0.05, 0.0))
    });

    circuit->execute();

    // 验证电路执行成功
    auto stats = circuit->get_stats();
    EXPECT_EQ(stats.num_gates, 5);
    EXPECT_GE(circuit->get_state_pool().active_count, 1);
}

TEST_F(SystemTest, ResetAndReuse) {
    // 第一次使用
    circuit->add_gate(Gates::PhaseRotation(0, M_PI));
    circuit->execute();

    // 重置并重新使用
    circuit->reset();
    circuit->add_gate(Gates::Displacement(0, std::complex<double>(0.1, 0.0)));
    circuit->build();
    circuit->execute();

    // 验证重置有效
    auto stats = circuit->get_stats();
    EXPECT_EQ(stats.num_gates, 1);
}

// 性能测试
TEST_F(SystemTest, DISABLED_PerformanceTest) {
    const int num_iterations = 100;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        circuit->reset();
        circuit->add_gates({
            Gates::PhaseRotation(0, M_PI / 4.0),
            Gates::Displacement(0, std::complex<double>(0.1, 0.0)),
            Gates::CreationOperator(0),
            Gates::BeamSplitter(0, 1, M_PI / 2.0)
        });
        circuit->build();
        circuit->execute();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "性能测试: " << num_iterations << " 次迭代, 总时间: "
              << total_time << " 秒, 平均: " << total_time / num_iterations << " 秒/次" << std::endl;
}
