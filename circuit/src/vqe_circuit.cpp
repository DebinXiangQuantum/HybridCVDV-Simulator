#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include "quantum_circuit.h"

/**
 * VQE电路
 * 用于二进制背包问题的变分量子本征求解
 */
void run_binary_knapsack_vqe_circuit(int num_qubits, int num_qumodes, int cutoff, int ndepth, const std::vector<int>& nfocks, const std::vector<double>* Xvec = nullptr) {
    QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
    
    std::vector<double> params;
    
    // 如果没有提供参数，生成随机参数
    if (Xvec == nullptr || Xvec->empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist_mag(0, 3);
        std::uniform_real_distribution<double> dist_arg(0, M_PI);
        
        for (int i = 0; i < ndepth; ++i) {
            for (int j = 0; j < 2; ++j) {
                params.push_back(dist_mag(gen));  // beta_mag
                params.push_back(dist_arg(gen));  // beta_arg
                params.push_back(dist_arg(gen));  // theta
                params.push_back(dist_arg(gen));  // phi
            }
        }
    } else {
        params = *Xvec;
    }
    
    // 构建VQE ansatz电路
    int param_idx = 0;
    for (int d = 0; d < ndepth; ++d) {
        // 应用位移操作
        for (int qm = 0; qm < num_qumodes; ++qm) {
            double beta_mag = params[param_idx++];
            double beta_arg = params[param_idx++];
            std::complex<double> beta(beta_mag * cos(beta_arg), beta_mag * sin(beta_arg));
            circuit.add_gate(Gates::Displacement(qm, beta));
        }
        
        // 应用旋转操作
        for (int q = 0; q < num_qubits; ++q) {
            double theta = params[param_idx++];
            double phi = params[param_idx++];
            circuit.add_gate(Gates::RotationX(q, theta));
            circuit.add_gate(Gates::RotationZ(q, phi));
        }
        
        // 应用耦合操作
        for (int q = 0; q < num_qubits; ++q) {
            for (int qm = 0; qm < num_qumodes; ++qm) {
                circuit.add_gate(Gates::JaynesCummings(q, qm, M_PI / 4.0));
            }
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
    filename << "result/vqe_circuit_qubits_" << num_qubits << "_qumodes_" << num_qumodes << "_depth_" << ndepth << "_cutoff_" << cutoff << ".csv";
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
        // 创建VQE电路
        int num_qubits = 2;
        int num_qumodes = 2;
        int cutoff = 16;
        int ndepth = 2;
        std::vector<int> nfocks = {16, 16};
        
        std::cout << "VQE电路创建成功" << std::endl;
        std::cout << "参数: 电路深度 = " << ndepth << ", 截断维度 = " << cutoff << std::endl;
        
        run_binary_knapsack_vqe_circuit(num_qubits, num_qumodes, cutoff, ndepth, nfocks);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
