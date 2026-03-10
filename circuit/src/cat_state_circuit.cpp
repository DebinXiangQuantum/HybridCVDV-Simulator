#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include "quantum_circuit.h"

/**
 * 猫态制备电路
 * 创建 |cat> = |alpha> + |-alpha> 态
 */
void run_cat_state_circuit(int num_qubits, int num_qumodes, int cutoff, double alpha) {
    QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
    
    // 实现猫态制备
    // 1. 对qubit 0应用Hadamard门
    circuit.add_gate(Gates::Hadamard(0));
    
    // 2. 应用条件位移门
    std::complex<double> displacement_param(alpha / std::sqrt(2), 0.0);
    circuit.add_gate(Gates::ConditionalDisplacement(0, 0, displacement_param));
    
    // 3. 再次应用Hadamard门
    circuit.add_gate(Gates::Hadamard(0));
    
    // 4. 应用S†门
    circuit.add_gate(Gates::PhaseGateS(0));
    
    // 5. 应用Hadamard门
    circuit.add_gate(Gates::Hadamard(0));
    
    // 6. 应用条件位移门 (虚数部分)
    std::complex<double> displacement_param2(0.0, M_PI / (8 * alpha * std::sqrt(2)));
    circuit.add_gate(Gates::ConditionalDisplacement(0, 0, displacement_param2));
    
    // 7. 应用Hadamard门
    circuit.add_gate(Gates::Hadamard(0));
    
    // 8. 应用S门
    circuit.add_gate(Gates::PhaseGateS(0));
    
    // 构建并执行电路
    circuit.build();
    circuit.execute();
    
    // 获取统计信息
    auto stats = circuit.get_stats();
    std::cout << "电路统计: " << stats.num_gates << " 个门, "
              << stats.active_states << " 个活跃状态" << std::endl;
    
    // 获取时间统计信息
    auto time_stats = circuit.get_time_stats();
    std::cout << "时间统计: 总时间=" << time_stats.total_time << " ms, "
              << "传输时延=" << time_stats.transfer_time << " ms, "
              << "计算时延=" << time_stats.computation_time << " ms" << std::endl;
    
    // 保存时间结果到文件
    std::stringstream filename;
    filename << "result/cat_state_time_alpha_" << alpha << "_cutoff_" << cutoff << ".csv";
    std::ofstream outfile(filename.str());
    if (outfile.is_open()) {
        outfile << "总时间,传输时延,计算时延\n";
        outfile << time_stats.total_time << "," << time_stats.transfer_time << "," << time_stats.computation_time << std::endl;
        outfile.close();
        std::cout << "时间结果已保存到: " << filename.str() << std::endl;
    } else {
        std::cerr << "无法创建时间结果文件" << std::endl;
    }
}

int main() {
    try {
        // 创建猫态电路
        int num_qubits = 1;
        int num_qumodes = 1;
        int cutoff = 16;
        double alpha = 1.0;
        
        std::cout << "猫态制备电路创建成功" << std::endl;
        std::cout << "参数: alpha = " << alpha << ", 截断维度 = " << cutoff << std::endl;
        
        run_cat_state_circuit(num_qubits, num_qumodes, cutoff, alpha);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
