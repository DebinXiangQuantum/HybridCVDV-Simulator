#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "quantum_circuit.h"

/**
 * 应用基变换
 */
void apply_basis_transformation(QuantumCircuit& circuit, int num_qubits) {
    for (int i = 0; i < num_qubits; ++i) {
        circuit.add_gate(Gates::Hadamard(i));
        if (i == num_qubits - 1) { // MSB
            circuit.add_gate(Gates::PauliX(i));
            circuit.add_gate(Gates::PauliZ(i));
        } else if (i == 0) { // LSB
            circuit.add_gate(Gates::PauliZ(i));
        } else { // 中间量子比特
            circuit.add_gate(Gates::PauliX(i));
        }
    }
}

/**
 * 应用逆基变换
 */
void apply_basis_transformation_reverse(QuantumCircuit& circuit, int num_qubits) {
    for (int i = 0; i < num_qubits; ++i) {
        if (i == num_qubits - 1) { // MSB
            circuit.add_gate(Gates::PauliZ(i));
            circuit.add_gate(Gates::PauliX(i));
            circuit.add_gate(Gates::Hadamard(i));
        } else if (i == 0) { // LSB
            circuit.add_gate(Gates::PauliZ(i));
            circuit.add_gate(Gates::Hadamard(i));
        } else { // 中间量子比特
            circuit.add_gate(Gates::PauliX(i));
            circuit.add_gate(Gates::Hadamard(i));
        }
    }
}

/**
 * CV到DV的状态转移电路
 */
void run_state_transfer_CVtoDV(int num_qubits, int num_qumodes, int cutoff, double lambda = 0.29, bool apply_basis = true) {
    QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
    
    // 实现V和W门操作序列
    for (int j = 1; j <= num_qubits; ++j) {
        // 这里使用简化的实现，实际项目中可能需要更复杂的门操作
        // 模拟V_j门
        for (int q = 0; q < num_qubits; ++q) {
            circuit.add_gate(Gates::RotationX(q, M_PI / 4.0));
        }
        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(Gates::Displacement(qm, std::complex<double>(lambda, 0.0)));
        }
        
        // 模拟W_j门
        for (int q = 0; q < num_qubits; ++q) {
            circuit.add_gate(Gates::RotationZ(q, M_PI / 4.0));
        }
        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(Gates::Squeezing(qm, std::complex<double>(lambda, 0.0)));
        }
    }
    
    // 应用基变换
    if (apply_basis) {
        apply_basis_transformation(circuit, num_qubits);
    }
    
    // 构建并执行电路
    circuit.build();
    circuit.execute();
    
    // 获取统计信息
    auto stats = circuit.get_stats();
    std::cout << "CV到DV电路统计: " << stats.num_gates << " 个门, "
              << stats.active_states << " 个活跃状态" << std::endl;
    
    // 获取时间统计信息
    auto time_stats = circuit.get_time_stats();
    std::cout << "时间统计: 总时间=" << time_stats.total_time << " ms, "
              << "传输时延=" << time_stats.transfer_time << " ms, "
              << "计算时延=" << time_stats.computation_time << " ms" << std::endl;
    
    // 保存时间结果到文件
    std::stringstream filename;
    filename << "result/state_transfer_CVtoDV_qubits_" << num_qubits << "_cutoff_" << cutoff << ".csv";
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

/**
 * DV到CV的状态转移电路
 */
void run_state_transfer_DVtoCV(int num_qubits, int num_qumodes, int cutoff, double lambda = 0.29, bool apply_basis = true) {
    QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
    
    // 应用逆基变换
    if (apply_basis) {
        apply_basis_transformation_reverse(circuit, num_qubits);
    }
    
    // 反向实现W和V门操作序列
    for (int j = num_qubits; j >= 1; --j) {
        // 模拟W_j门
        for (int q = 0; q < num_qubits; ++q) {
            circuit.add_gate(Gates::RotationZ(q, M_PI / 4.0));
        }
        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(Gates::Squeezing(qm, std::complex<double>(lambda, 0.0)));
        }
        
        // 模拟V_j门
        for (int q = 0; q < num_qubits; ++q) {
            circuit.add_gate(Gates::RotationX(q, M_PI / 4.0));
        }
        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(Gates::Displacement(qm, std::complex<double>(lambda, 0.0)));
        }
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
    filename << "result/state_transfer_DVtoCV_qubits_" << num_qubits << "_cutoff_" << cutoff << ".csv";
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
        // 创建CV到DV的状态转移电路
        int num_qubits = 2;
        int num_qumodes = 1;
        int cutoff = 16;
        double lambda = 0.29;
        
        std::cout << "CV到DV状态转移电路创建成功" << std::endl;
        run_state_transfer_CVtoDV(num_qubits, num_qumodes, cutoff, lambda);
        
        // 创建DV到CV的状态转移电路
        std::cout << "\nDV到CV状态转移电路创建成功" << std::endl;
        run_state_transfer_DVtoCV(num_qubits, num_qumodes, cutoff, lambda);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
