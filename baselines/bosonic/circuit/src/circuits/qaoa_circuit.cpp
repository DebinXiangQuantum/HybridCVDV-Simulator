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
 * CV-QAOA电路
 * 连续变量量子近似优化算法
 */
void run_cv_qaoa_circuit(int num_qubits, int num_qumodes, int cutoff, const std::vector<double>& params, double s, double a, int p, int n) {
    gpu::Circuit circuit(num_qubits, num_qumodes, cutoff);
    
    // 初始挤压操作
    for (int qm = 0; qm < num_qumodes; ++qm) {
        std::complex<double> squeezing_param(s, 0.0);
        circuit.add_gate(gpu::Gates::Squeezing(qm, squeezing_param));
    }
    
    // 构建QAOA层
    for (int i = 0; i < p; ++i) {
        // 获取当前层的参数
        double gamma = params[i];
        double eta = params[p + i];
        
        // 应用Cost Hamiltonian演化
        for (int qm = 0; qm < num_qumodes; ++qm) {
            // 实现位置相关的演化
            std::complex<double> displacement_param(a * gamma, 0.0);
            circuit.add_gate(gpu::Gates::Displacement(qm, displacement_param));
        }
        
        // 应用Mixing Hamiltonian演化
        for (int qm = 0; qm < num_qumodes; ++qm) {
            std::complex<double> squeezing_param(eta, 0.0);
            circuit.add_gate(gpu::Gates::Squeezing(qm, squeezing_param));
        }
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
    filename << "result/qaoa_circuit_qumodes_" << num_qumodes << "_layers_" << p << "_cutoff_" << cutoff << ".csv";
    std::ofstream outfile(filename.str());
    if (outfile.is_open()) {
        outfile << "电路类型,参数,门数量,总时间,传输时间,计算时间,内存占用\n";
        outfile << "qaoa_circuit,qumodes=" << num_qumodes << ",layers=" << p << ",cutoff=" << cutoff << "," << stats.num_gates << "," << stats.total_time_ms << "," << stats.transfer_time_ms << "," << stats.computation_time_ms << "," << stats.memory_usage_bytes << std::endl;
        outfile.close();
        std::cout << "结果已保存到: " << filename.str() << std::endl;
    } else {
        std::cerr << "无法创建时间结果文件" << std::endl;
    }
}

#ifndef NO_MAIN
int main() {
    try {
        // 创建CV-QAOA电路
        int num_qubits = 1;
        int num_qumodes = 1;
        int cutoff = 16;
        int p = 2; // QAOA层数
        double s = 0.5; // 挤压参数
        double a = 1.0; // 目标位置值
        int n = 2; // 成本函数幂次
        
        // 生成随机参数
        std::vector<double> params(2 * p);
        for (int i = 0; i < 2 * p; ++i) {
            params[i] = 2 * M_PI * (i + 1) / (2 * p);
        }
        
        std::cout << "CV-QAOA电路创建成功" << std::endl;
        std::cout << "参数: 层数 = " << p << ", 挤压参数 = " << s << ", 目标位置 = " << a << std::endl;
        
        run_cv_qaoa_circuit(num_qubits, num_qumodes, cutoff, params, s, a, p, n);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
#endif
