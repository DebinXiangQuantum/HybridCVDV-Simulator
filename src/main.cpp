#include <iostream>
#include <iomanip>
#include <chrono>
#include "quantum_circuit.h"
#include "batch_scheduler.h"

/**
 * HybridCVDV-Simulator 主程序
 *
 * 演示如何使用混合CV-DV量子模拟器
 */
int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << "   Hybrid Tensor-DD 量子模拟器" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "版本: v1.5" << std::endl;
    std::cout << "架构: Hybrid CV-DV Quantum Systems" << std::endl;
    std::cout << "硬件: CPU (逻辑控制) + NVIDIA GPU (张量计算)" << std::endl;
    std::cout << "=========================================" << std::endl << std::endl;

    try {
        // 创建量子电路: 2 qubits, 2 qumodes, 截断维度16, 最大32个状态
        QuantumCircuit circuit(4, 6, 16, 64);

        std::cout << "创建量子电路: 4 qubits, 6 qumodes, 截断维度=16" << std::endl;
        std::cout << "初始化状态池: 容量=64" << std::endl << std::endl;

        // 示例1: 简单的连续变量操作
        std::cout << "示例1: 连续变量位移和挤压" << std::endl;
        circuit.add_gates({
            Gates::Displacement(0, std::complex<double>(0.5, 0.2)),  // 在qumode 0上应用位移
            Gates::Squeezing(1, std::complex<double>(0.3, 0.1)),      // 在qumode 1上应用挤压
            Gates::BeamSplitter(1, 2, M_PI / 3.0),                   // 光束分裂器
            Gates::ControlledDisplacement(2, 1, std::complex<double>(0.3, 0.0)),// 受控位移
            Gates::ControlledSqueezing(1, 1, std::complex<double>(0.3, 0.0))  // 受控挤压
        });

        circuit.build();

        auto start_time = std::chrono::high_resolution_clock::now();
        circuit.execute();
        auto end_time = std::chrono::high_resolution_clock::now();

        double execution_time = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "执行时间: " << std::fixed << std::setprecision(6) << execution_time << " 秒" << std::endl;

        auto stats = circuit.get_stats();
        std::cout << "电路统计: " << stats.num_gates << " 个门, "
                  << stats.active_states << " 个活跃状态" << std::endl << std::endl;

        // 示例2: 混合量子操作
        std::cout << "示例2: Qubit控制的连续变量操作" << std::endl;
        {
            QuantumCircuit circuit2(4, 2, 16, 32);
            circuit2.add_gates({
                Gates::PhaseRotation(0, M_PI / 4.0),                        // Qubit旋转
                Gates::ControlledDisplacement(0, 0, std::complex<double>(0.4, 0.0)), // 受控位移
                Gates::CreationOperator(1),                                 // 光子创建
                Gates::KerrGate(1, 0.1)                                    // Kerr非线性
            });
            circuit2.build();

            start_time = std::chrono::high_resolution_clock::now();
            circuit2.execute();
            end_time = std::chrono::high_resolution_clock::now();

            execution_time = std::chrono::duration<double>(end_time - start_time).count();
            std::cout << "执行时间: " << std::fixed << std::setprecision(6) << execution_time << " 秒" << std::endl;

            auto stats2 = circuit2.get_stats();
            std::cout << "电路统计: " << stats2.num_gates << " 个门, "
                      << stats2.active_states << " 个活跃状态" << std::endl << std::endl;
        }

        execution_time = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "执行时间: " << std::fixed << std::setprecision(6) << execution_time << " 秒" << std::endl;

        auto stats1 = circuit.get_stats();
        std::cout << "电路统计: " << stats1.num_gates << " 个门, "
                  << stats1.active_states << " 个活跃状态" << std::endl << std::endl;

        // 示例3: 批处理调度器演示
        std::cout << "示例3: 批处理调度器性能演示" << std::endl;
        RuntimeScheduler scheduler(&circuit, 8);  // 批大小为8

        // 添加多个门操作进行批处理
        for (int i = 0; i < 20; ++i) {
            scheduler.schedule_gate(Gates::PhaseRotation(0, M_PI / (i + 1)));
            scheduler.schedule_gate(Gates::Displacement(0, std::complex<double>(0.1 * (i + 1), 0.0)));
        }

        start_time = std::chrono::high_resolution_clock::now();
        scheduler.execute_all();
        end_time = std::chrono::high_resolution_clock::now();

        execution_time = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "批处理执行时间: " << std::fixed << std::setprecision(6) << execution_time << " 秒" << std::endl;

        auto runtime_stats = scheduler.get_stats();
        std::cout << "批处理统计: " << runtime_stats.batch_stats.total_tasks << " 个任务, "
                  << runtime_stats.batch_stats.total_batches << " 个批次, "
                  << "平均批大小: " << runtime_stats.batch_stats.avg_batch_size << std::endl;

        // 内存使用情况
        auto& state_pool = circuit.get_state_pool();
        std::cout << "内存使用: " << state_pool.active_count << "/" << state_pool.capacity << " 状态槽位" << std::endl;

        std::cout << std::endl << "=========================================" << std::endl;
        std::cout << "模拟器运行完成！" << std::endl;
        std::cout << "=========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "未知错误发生" << std::endl;
        return 1;
    }
}
