#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "core/circuit.h"
#include "gates/gates.h"

/**
 * 应用基变换
 */
void apply_basis_transformation(gpu::Circuit& circuit, int num_qubits) {
    for (int i = 0; i < num_qubits; ++i) {
        circuit.add_gate(gpu::Gates::Hadamard(i));
        if (i == num_qubits - 1) { // MSB
            circuit.add_gate(gpu::Gates::PauliX(i));
            circuit.add_gate(gpu::Gates::PauliZ(i));
        } else if (i == 0) { // LSB
            circuit.add_gate(gpu::Gates::PauliZ(i));
        } else { // 中间量子比特
            circuit.add_gate(gpu::Gates::PauliX(i));
        }
    }
}

/**
 * 应用逆基变换
 */
void apply_basis_transformation_reverse(gpu::Circuit& circuit, int num_qubits) {
    for (int i = 0; i < num_qubits; ++i) {
        if (i == num_qubits - 1) { // MSB
            circuit.add_gate(gpu::Gates::PauliZ(i));
            circuit.add_gate(gpu::Gates::PauliX(i));
            circuit.add_gate(gpu::Gates::Hadamard(i));
        } else if (i == 0) { // LSB
            circuit.add_gate(gpu::Gates::PauliZ(i));
            circuit.add_gate(gpu::Gates::Hadamard(i));
        } else { // 中间量子比特
            circuit.add_gate(gpu::Gates::PauliX(i));
            circuit.add_gate(gpu::Gates::Hadamard(i));
        }
    }
}

/**
 * CV到DV的状态转移电路
 */
void run_state_transfer_CVtoDV(int num_qubits, int num_qumodes, int cutoff, double lambda = 0.29, bool apply_basis = true) {
    gpu::Circuit circuit(num_qubits, num_qumodes, cutoff);
    
    // 实现V和W门操作序列
    for (int j = 1; j <= num_qubits; ++j) {
        // 这里使用简化的实现，实际项目中可能需要更复杂的门操作
        // 模拟V_j门
        for (int q = 0; q < num_qubits; ++q) {
            circuit.add_gate(gpu::Gates::RotationX(q, M_PI / 4.0));
        }
        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(gpu::Gates::Displacement(qm, std::complex<double>(lambda, 0.0)));
        }
        
        // 模拟W_j门
        for (int q = 0; q < num_qubits; ++q) {
            circuit.add_gate(gpu::Gates::RotationZ(q, M_PI / 4.0));
        }
        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(gpu::Gates::Squeezing(qm, std::complex<double>(lambda, 0.0)));
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
    std::cout << "CV到DV电路统计: " << stats.num_gates << " 个门" << std::endl;
    
    // 获取时间统计信息
    std::cout << "时间统计: 总时间=" << stats.total_time_ms << " ms, "
              << "传输时延=" << stats.transfer_time_ms << " ms, "
              << "计算时延=" << stats.computation_time_ms << " ms" << std::endl;
    
    // 保存时间结果到文件
    std::stringstream filename;
    filename << "result/state_transfer_CVtoDV_qubits_" << num_qubits << "_cutoff_" << cutoff << ".csv";
    std::ofstream outfile(filename.str());
    if (outfile.is_open()) {
        outfile << "电路类型,参数,门数量,总时间,传输时间,计算时间,内存占用\n";
        outfile << "state_transfer_circuit,CVtoDV,qubits=" << num_qubits << ",cutoff=" << cutoff << "," << stats.num_gates << "," << stats.total_time_ms << "," << stats.transfer_time_ms << "," << stats.computation_time_ms << "," << stats.memory_usage_bytes << std::endl;
        outfile.close();
        std::cout << "结果已保存到: " << filename.str() << std::endl;
    } else {
        std::cerr << "无法创建时间结果文件" << std::endl;
    }
}

/**
 * DV到CV的状态转移电路
 */
void run_state_transfer_DVtoCV(int num_qubits, int num_qumodes, int cutoff, double lambda = 0.29, bool apply_basis = true) {
    gpu::Circuit circuit(num_qubits, num_qumodes, cutoff);
    
    // 应用逆基变换
    if (apply_basis) {
        apply_basis_transformation_reverse(circuit, num_qubits);
    }
    
    // 实现V†和W†门操作序列
    for (int j = num_qubits; j >= 1; --j) {
        // 模拟W_j†门
        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(gpu::Gates::Squeezing(qm, std::complex<double>(-lambda, 0.0)));
        }
        for (int q = 0; q < num_qubits; ++q) {
            circuit.add_gate(gpu::Gates::RotationZ(q, -M_PI / 4.0));
        }
        
        // 模拟V_j†门
        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(gpu::Gates::Displacement(qm, std::complex<double>(-lambda, 0.0)));
        }
        for (int q = 0; q < num_qubits; ++q) {
            circuit.add_gate(gpu::Gates::RotationX(q, -M_PI / 4.0));
        }
    }
    
    // 构建并执行电路
    circuit.build();
    circuit.execute();
    
    // 获取统计信息
    auto stats = circuit.get_stats();
    std::cout << "DV到CV电路统计: " << stats.num_gates << " 个门" << std::endl;
    
    // 获取时间统计信息
    std::cout << "时间统计: 总时间=" << stats.total_time_ms << " ms, "
              << "传输时延=" << stats.transfer_time_ms << " ms, "
              << "计算时延=" << stats.computation_time_ms << " ms" << std::endl;
    
    // 保存时间结果到文件
    std::stringstream filename;
    filename << "result/state_transfer_DVtoCV_qubits_" << num_qubits << "_cutoff_" << cutoff << ".csv";
    std::ofstream outfile(filename.str());
    if (outfile.is_open()) {
        outfile << "电路类型,参数,门数量,总时间,传输时间,计算时间,内存占用\n";
        outfile << "state_transfer_circuit,DVtoCV,qubits=" << num_qubits << ",cutoff=" << cutoff << "," << stats.num_gates << "," << stats.total_time_ms << "," << stats.transfer_time_ms << "," << stats.computation_time_ms << "," << stats.memory_usage_bytes << std::endl;
        outfile.close();
        std::cout << "结果已保存到: " << filename.str() << std::endl;
    } else {
        std::cerr << "无法创建时间结果文件" << std::endl;
    }
}

#ifndef NO_MAIN
int main() {
    try {
        // 创建态传输电路
        int num_qubits = 2;
        int num_qumodes = 1;
        int cutoff = 16;
        double lambda = 0.29; // 耦合参数
        
        std::cout << "量子态传输电路创建成功" << std::endl;
        std::cout << "参数: 量子比特数=" << num_qubits << ", 连续模式数=" << num_qumodes 
                  << ", lambda=" << lambda << ", 截断维度=" << cutoff << std::endl;
        
        // 测试CV到DV的传输
        std::cout << "\n=== CV到DV态传输 ===" << std::endl;
        run_state_transfer_CVtoDV(num_qubits, num_qumodes, cutoff, lambda, true);
        
        // 测试DV到CV的传输
        std::cout << "\n=== DV到CV态传输 ===" << std::endl;
        run_state_transfer_DVtoCV(num_qubits, num_qumodes, cutoff, lambda, true);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
#endif
