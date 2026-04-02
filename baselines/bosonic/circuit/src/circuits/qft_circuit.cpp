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
 * 应用基准备
 */
void apply_basis_preparation(gpu::Circuit& circuit, const std::vector<int>& qubits) {
    circuit.add_gate(gpu::Gates::PauliZ(qubits[0]));
    circuit.add_gate(gpu::Gates::PauliZ(qubits.back()));
    circuit.add_gate(gpu::Gates::Hadamard(qubits.back()));
    
    for (size_t i = 0; i < qubits.size() - 1; ++i) {
        circuit.add_gate(gpu::Gates::PauliX(qubits[i]));
        circuit.add_gate(gpu::Gates::Hadamard(qubits[i]));
    }
}

/**
 * 应用逆基准备
 */
void apply_basis_preparation_reverse(gpu::Circuit& circuit, const std::vector<int>& qubits) {
    for (size_t i = 0; i < qubits.size() - 1; ++i) {
        circuit.add_gate(gpu::Gates::Hadamard(qubits[i]));
        circuit.add_gate(gpu::Gates::PauliX(qubits[i]));
    }
    circuit.add_gate(gpu::Gates::Hadamard(qubits.back()));
    circuit.add_gate(gpu::Gates::PauliZ(qubits[0]));
    circuit.add_gate(gpu::Gates::PauliZ(qubits.back()));
}

/**
 * QFT电路
 * 使用CV-DV混合系统实现量子傅里叶变换
 */
void run_qft_circuit(int num_qubits, int num_qumodes, int cutoff, double delta, int n, int a, int append) {
    int total = n + a + append;
    gpu::Circuit circuit(num_qubits, num_qumodes, cutoff);
    
    // 初始化辅助量子比特
    for (int i = 0; i < a; ++i) {
        circuit.add_gate(gpu::Gates::Hadamard(i));
    }
    
    // 构建总寄存器
    std::vector<int> total_reg;
    for (int i = 0; i < a; ++i) total_reg.push_back(i); // 辅助量子比特
    for (int i = a; i < a + n; ++i) total_reg.push_back(i); // 主量子比特
    for (int i = a + n; i < total; ++i) total_reg.push_back(i); // 附加量子比特
    
    // 反转寄存器顺序
    std::vector<int> reversed_reg(total_reg.rbegin(), total_reg.rend());
    
    // 应用基准备
    apply_basis_preparation(circuit, reversed_reg);
    
    // DV到CV转移 (无基变换和测量)
    // 模拟状态转移操作
    for (int q = 0; q < total; ++q) {
        circuit.add_gate(gpu::Gates::RotationX(q, M_PI / 4.0));
    }
    circuit.add_gate(gpu::Gates::Displacement(0, std::complex<double>(0.29, 0.0)));
    
    // CV空间中的QFT操作
    double delta_prime = (2 * M_PI) / (std::pow(2, total) * delta);
    circuit.add_gate(gpu::Gates::Displacement(0, std::complex<double>(delta / 2, 0.0)));
    circuit.add_gate(gpu::Gates::PhaseRotation(0, M_PI / 2.0));
    circuit.add_gate(gpu::Gates::Displacement(0, std::complex<double>(-delta_prime / 2, 0.0)));
    
    // CV到DV转移 (无基变换和测量)
    for (int q = 0; q < total; ++q) {
        circuit.add_gate(gpu::Gates::RotationZ(q, M_PI / 4.0));
    }
    circuit.add_gate(gpu::Gates::Squeezing(0, std::complex<double>(0.29, 0.0)));
    
    // 应用逆基准备
    apply_basis_preparation_reverse(circuit, reversed_reg);
    
    // 测量主量子比特
    int start_index = a;
    for (int i = 0; i < n; ++i) {
        // 这里简化处理，实际项目中可能需要更复杂的测量操作
        circuit.add_gate(gpu::Gates::Hadamard(total_reg[start_index + i]));
    }
    
    // 构建并执行电路
    circuit.build();
    circuit.execute();
    
    // 获取统计信息
    auto stats = circuit.get_stats();
    std::cout << "电路统计: " << stats.num_gates << " 个门" << std::endl;
    
    // 获取时间统计信息
    std::cout << "时间统计: 总时间=" << stats.total_time_ms << " ms, "
              << "传输时延=" << stats.transfer_time_ms << " ms, "
              << "计算时延=" << stats.computation_time_ms << " ms" << std::endl;
    
    // 保存时间结果到文件
    std::stringstream filename;
    filename << "result/qft_circuit_n_" << n << "_a_" << a << "_append_" << append << "_cutoff_" << cutoff << ".csv";
    std::ofstream outfile(filename.str());
    if (outfile.is_open()) {
        outfile << "电路类型,参数,门数量,总时间,传输时间,计算时间,内存占用\n";
        outfile << "qft_circuit,n=" << n << ",a=" << a << ",append=" << append << ",cutoff=" << cutoff << "," << stats.num_gates << "," << stats.total_time_ms << "," << stats.transfer_time_ms << "," << stats.computation_time_ms << "," << stats.memory_usage_bytes << std::endl;
        outfile.close();
        std::cout << "结果已保存到: " << filename.str() << std::endl;
    } else {
        std::cerr << "无法创建时间结果文件" << std::endl;
    }
}

#ifndef NO_MAIN
int main() {
    try {
        // 创建QFT电路
        int num_qubits = 4;  // 总量子比特数
        int num_qumodes = 1; // 连续变量模式数
        int cutoff = 16;
        double delta = 0.1;  // 网格间距
        int n = 2;           // 主量子比特数
        int a = 1;           // 辅助量子比特数
        int append = 1;      // 附加量子比特数
        
        std::cout << "量子傅里叶变换电路创建成功" << std::endl;
        std::cout << "参数: n=" << n << ", a=" << a << ", append=" << append 
                  << ", delta=" << delta << ", 截断维度=" << cutoff << std::endl;
        
        run_qft_circuit(num_qubits, num_qumodes, cutoff, delta, n, a, append);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
#endif
