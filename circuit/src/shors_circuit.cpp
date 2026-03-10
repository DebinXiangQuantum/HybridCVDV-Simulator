#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include "quantum_circuit.h"

/**
 * GKP态制备辅助函数
 */
void prepare_gkp_state(QuantumCircuit& circuit, int qubit_idx, int qumode_idx, int cutoff, double r = 0.222) {
    double alpha = std::sqrt(M_PI);
    
    // 初始挤压操作
    std::complex<double> squeezing_param(r, 0.0);
    circuit.add_gate(Gates::Squeezing(qumode_idx, squeezing_param));
    
    // 多轮条件位移操作
    for (int i = 1; i < 9; ++i) {
        circuit.add_gate(Gates::Hadamard(qubit_idx));
        
        std::complex<double> displacement_param(alpha / std::sqrt(2), 0.0);
        circuit.add_gate(Gates::ConditionalDisplacement(qubit_idx, qumode_idx, displacement_param));
        
        circuit.add_gate(Gates::Hadamard(qubit_idx));
        circuit.add_gate(Gates::PhaseGateS(qubit_idx));
        circuit.add_gate(Gates::Hadamard(qubit_idx));
        
        std::complex<double> displacement_param2(0.0, M_PI / (8 * alpha * std::sqrt(2)));
        circuit.add_gate(Gates::ConditionalDisplacement(qubit_idx, qumode_idx, displacement_param2));
        
        circuit.add_gate(Gates::Hadamard(qubit_idx));
        circuit.add_gate(Gates::PhaseGateS(qubit_idx));
    }
}

/**
 * Shor's算法电路
 * 用于因子分解问题
 */
void run_shors_circuit(int num_qubits, int num_qumodes, int cutoff, int N, int m, int R, int a, double delta) {
    QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
    
    // 在qumode 0和1上制备GKP态
    prepare_gkp_state(circuit, 0, 0, cutoff);
    prepare_gkp_state(circuit, 0, 1, cutoff);
    
    // 在qumode 2上应用挤压
    std::complex<double> squeezing_param(-std::log(delta), 0.0);
    circuit.add_gate(Gates::Squeezing(2, squeezing_param));
    
    // 模算术操作
    // 1. 平移操作 R
    for (int i = 0; i < R; ++i) {
        std::complex<double> displacement_param(1.0, 0.0);
        circuit.add_gate(Gates::Displacement(0, displacement_param));
    }
    
    // 2. 乘法操作 N
    for (int i = 0; i < N; ++i) {
        std::complex<double> displacement_param(1.0, 0.0);
        circuit.add_gate(Gates::Displacement(1, displacement_param));
    }
    
    // 3. 模指数运算 U_aNm
    for (int i = 0; i < m; ++i) {
        // 模拟模指数运算
        circuit.add_gate(Gates::JaynesCummings(0, 0, M_PI / 4.0));
        circuit.add_gate(Gates::JaynesCummings(0, 1, M_PI / 4.0));
        circuit.add_gate(Gates::JaynesCummings(0, 2, M_PI / 4.0));
    }
    
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
    filename << "result/shors_circuit_N_" << N << "_a_" << a << "_cutoff_" << cutoff << ".csv";
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
        // 创建Shor's算法电路
        int num_qubits = 1;
        int num_qumodes = 2;  // 从3减少到2
        int cutoff = 8;       // 从16减少到8
        int N = 15; // 要分解的数
        int m = 4; // 模指数参数
        int R = 4; // 参考参数
        int a = 2; // 指数基
        double delta = 0.1; // GKP挤压参数
        
        std::cout << "Shor's算法电路创建成功" << std::endl;
        std::cout << "参数: N = " << N << ", a = " << a << ", 截断维度 = " << cutoff << std::endl;
        
        run_shors_circuit(num_qubits, num_qumodes, cutoff, N, m, R, a, delta);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
