#include <iostream>
#include <complex>
#include "quantum_circuit.h"

/**
 * 基本使用示例
 *
 * 演示HybridCVDV-Simulator的基本用法
 */
int main() {
    std::cout << "HybridCVDV-Simulator 基本使用示例" << std::endl;
    std::cout << "=====================================" << std::endl;

    try {
        // 1. 创建量子电路
        std::cout << "1. 创建量子电路 (1 qubit, 2 qumodes, 截断维度=8)" << std::endl;
        QuantumCircuit circuit(1, 2, 8, 16);
        circuit.build();

        // 2. 构建量子线路
        std::cout << "2. 添加量子门操作" << std::endl;

        // 添加一些基本的门操作
        circuit.add_gate(Gates::PhaseRotation(0, M_PI / 4.0));
        circuit.add_gate(Gates::Displacement(0, std::complex<double>(0.3, 0.1)));
        circuit.add_gate(Gates::Squeezing(1, std::complex<double>(0.2, 0.0)));
        circuit.add_gate(Gates::BeamSplitter(0, 1, M_PI / 3.0));
        circuit.add_gate(Gates::CreationOperator(0));

        // 3. 执行量子线路
        std::cout << "3. 执行量子线路" << std::endl;
        circuit.execute();

        // 4. 获取结果统计
        std::cout << "4. 结果统计" << std::endl;
        auto stats = circuit.get_stats();
        std::cout << "   - Qubits: " << stats.num_qubits << std::endl;
        std::cout << "   - Qumodes: " << stats.num_qumodes << std::endl;
        std::cout << "   - 截断维度: " << stats.cv_truncation << std::endl;
        std::cout << "   - 门操作数: " << stats.num_gates << std::endl;
        std::cout << "   - 活跃状态数: " << stats.active_states << std::endl;
        std::cout << "   - HDD节点数: " << stats.hdd_nodes << std::endl;

        // 5. 获取状态振幅 (示例)
        std::cout << "5. 状态振幅示例" << std::endl;
        std::complex<double> amp = circuit.get_amplitude({0}, {});
        std::cout << "   振幅 |0⟩: " << amp << " (模方: " << std::norm(amp) << ")" << std::endl;

        std::cout << std::endl << "示例运行完成！" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}
