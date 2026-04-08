#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include "core/circuit.h"
#include "gates/gates.h"

void run_cat_state_circuit(int num_qubits, int num_qumodes, int cutoff, double alpha) {
    gpu::Circuit circuit(num_qubits, num_qumodes, cutoff);
    
    // 1. 对qubit 0应用Hadamard门
    circuit.add_gate(gpu::Gates::Hadamard(0));
    
    // 2. 应用条件位移门
    std::complex<double> displacement_param(alpha / std::sqrt(2), 0.0);
    circuit.add_gate(gpu::Gates::ConditionalDisplacement(0, 0, displacement_param));
    
    // 3. 再次应用Hadamard门
    circuit.add_gate(gpu::Gates::Hadamard(0));
    
    // 4. 应用S†门 (使用PhaseS的共轭)
    circuit.add_gate(gpu::Gates::PhaseS(0));
    circuit.add_gate(gpu::Gates::PhaseS(0));
    circuit.add_gate(gpu::Gates::PhaseS(0));
    
    // 5. 应用Hadamard门
    circuit.add_gate(gpu::Gates::Hadamard(0));
    
    // 6. 应用条件位移门 (虚数部分)
    std::complex<double> displacement_param2(0.0, M_PI / (8 * alpha * std::sqrt(2)));
    circuit.add_gate(gpu::Gates::ConditionalDisplacement(0, 0, displacement_param2));
    
    // 7. 应用Hadamard门
    circuit.add_gate(gpu::Gates::Hadamard(0));
    
    // 8. 应用S门
    circuit.add_gate(gpu::Gates::PhaseS(0));
    
    circuit.build();
    circuit.execute();
    
    auto stats = circuit.get_stats();
    std::cout << "电路统计: " << stats.num_gates << " 个门" << std::endl;
    std::cout << "时间统计: 总时间=" << stats.total_time_ms << " ms, "
              << "传输时延=" << stats.transfer_time_ms << " ms, "
              << "计算时延=" << stats.computation_time_ms << " ms" << std::endl;
    
    std::stringstream filename;
    filename << "result/cat_state_time_alpha_" << alpha << "_cutoff_" << cutoff << ".csv";
    std::ofstream outfile(filename.str());
    if (outfile.is_open()) {
        outfile << "电路类型,参数,门数量,总时间,传输时间,计算时间,内存占用\n";
        outfile << "cat_state_circuit,num_qubits=" << num_qubits << ",num_qumodes=" << num_qumodes << ",alpha=" << alpha << ",cutoff=" << cutoff << "," << stats.num_gates << "," << stats.total_time_ms << "," << stats.transfer_time_ms << "," << stats.computation_time_ms << "," << stats.memory_usage_bytes << std::endl;
        outfile.close();
        std::cout << "结果已保存到: " << filename.str() << std::endl;
    }
}

#ifndef NO_MAIN
int main() {
    try {
        int num_qubits = 1;
        int num_qumodes = 1;
        int cutoff = 16;
        double alpha = 1.0;
        
        std::cout << "猫态制备电路" << std::endl;
        std::cout << "参数: alpha = " << alpha << ", 截断维度 = " << cutoff << std::endl;
        
        run_cat_state_circuit(num_qubits, num_qumodes, cutoff, alpha);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
#endif
