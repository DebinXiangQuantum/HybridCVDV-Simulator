#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include "core/circuit.h"
#include "gates/gates.h"

/**
 * JCH模拟电路
 * 实现JCH哈密顿量的单时间步演化
 */
void run_jch_simulation_circuit(int Nsites, int Nqubits, int cutoff, double J, double omega_r, double omega_q, double g, double tau, int timesteps) {
    gpu::Circuit circuit(Nqubits, Nsites, cutoff);
    
    // 实现JCH哈密顿量的Trotter分解
    
    // 1. 谐振子项 (omega_r * a†a)
    for (int i = 0; i < Nsites; ++i) {
        double angle = omega_r * tau;
        circuit.add_gate(gpu::Gates::PhaseRotation(i, angle));
    }
    
    // 2. 量子比特项 (omega_q * σz/2)
    for (int i = 0; i < Nqubits; ++i) {
        double angle = omega_q * tau / 2.0;
        circuit.add_gate(gpu::Gates::RotationZ(i, angle));
    }
    
    // 3. 耦合项 (g * (a†σ- + aσ+))
    for (int i = 0; i < std::min(Nsites, Nqubits); ++i) {
        double angle = g * tau;
        circuit.add_gate(gpu::Gates::JaynesCummings(i, i, angle));
    }
    
    // 4. 跳跃项 (J * (a†b + ab†))
    for (int i = 0; i < Nsites - 1; ++i) {
        double angle = J * tau;
        circuit.add_gate(gpu::Gates::BeamSplitter(i, i+1, angle));
    }
    
    // 构建并执行电路
    circuit.build();
    circuit.execute();
    
    // 获取统计信息
    auto stats = circuit.get_stats();
    std::cout << "单时间步电路统计: " << stats.num_gates << " 个门" << std::endl;
    
    // 获取时间统计信息
    std::cout << "时间统计: 总时间=" << stats.total_time_ms << " ms, "
              << "传输时延=" << stats.transfer_time_ms << " ms, "
              << "计算时延=" << stats.computation_time_ms << " ms" << std::endl;
    
    // 保存时间结果到文件
    std::stringstream filename;
    filename << "result/jch_simulation_multi_qubits_" << Nqubits << "_sites_" << Nsites << "_timesteps_" << timesteps << "_cutoff_" << cutoff << ".csv";
    std::ofstream outfile(filename.str());
    if (outfile.is_open()) {
        outfile << "电路类型,参数,门数量,总时间,传输时间,计算时间,内存占用\n";
        outfile << "jch_simulation_circuit,num_qubits=" << Nqubits << ",num_modes=" << Nsites << ",timesteps=" << timesteps << ",cutoff=" << cutoff << ",J=" << J << ",omega_r=" << omega_r << ",omega_q=" << omega_q << ",g=" << g << ",tau=" << tau << "," << stats.num_gates << "," << stats.total_time_ms << "," << stats.transfer_time_ms << "," << stats.computation_time_ms << "," << stats.memory_usage_bytes << std::endl;
        outfile.close();
        std::cout << "结果已保存到: " << filename.str() << std::endl;
    } else {
        std::cerr << "无法创建结果文件" << std::endl;
    }
}

/**
 * 多时间步JCH模拟电路
 */
void run_jch_simulation_circuit_display(int Nsites, int Nqubits, int cutoff, double J, double omega_r, double omega_q, double g, double tau, int timesteps) {
    gpu::Circuit circuit(Nqubits, Nsites, cutoff);
    
    // 多次应用单时间步电路
    for (int t = 0; t < timesteps; ++t) {
        // 1. 谐振子项
        for (int i = 0; i < Nsites; ++i) {
            double angle = omega_r * tau;
            circuit.add_gate(gpu::Gates::PhaseRotation(i, angle));
        }
        
        // 2. 量子比特项
        for (int i = 0; i < Nqubits; ++i) {
            double angle = omega_q * tau / 2.0;
            circuit.add_gate(gpu::Gates::RotationZ(i, angle));
        }
        
        // 3. 耦合项
        for (int i = 0; i < std::min(Nsites, Nqubits); ++i) {
            double angle = g * tau;
            circuit.add_gate(gpu::Gates::JaynesCummings(i, i, angle));
        }
        
        // 4. 跳跃项
        for (int i = 0; i < Nsites - 1; ++i) {
            double angle = J * tau;
            circuit.add_gate(gpu::Gates::BeamSplitter(i, i+1, angle));
        }
    }
    
    // 构建并执行电路
    circuit.build();
    circuit.execute();
    
    // 获取统计信息
    auto stats = circuit.get_stats();
    std::cout << "多时间步电路统计: " << stats.num_gates << " 个门" << std::endl;
    
    // 获取时间统计信息
    std::cout << "时间统计: 总时间=" << stats.total_time_ms << " ms, "
              << "传输时延=" << stats.transfer_time_ms << " ms, "
              << "计算时延=" << stats.computation_time_ms << " ms" << std::endl;
    
    // 保存时间结果到文件
    std::stringstream filename;
    filename << "result/jch_simulation_multi_qubits_" << Nqubits << "_sites_" << Nsites << "_timesteps_" << timesteps << "_cutoff_" << cutoff << ".csv";
    std::ofstream outfile(filename.str());
    if (outfile.is_open()) {
        outfile << "电路类型,参数,门数量,总时间,传输时间,计算时间,内存占用\n";
        outfile << "jch_simulation_circuit,num_qubits=" << Nqubits << ",num_modes=" << Nsites << ",timesteps=" << timesteps << ",cutoff=" << cutoff << ",J=" << J << ",omega_r=" << omega_r << ",omega_q=" << omega_q << ",g=" << g << ",tau=" << tau << "," << stats.num_gates << "," << stats.total_time_ms << "," << stats.transfer_time_ms << "," << stats.computation_time_ms << "," << stats.memory_usage_bytes << std::endl;
        outfile.close();
        std::cout << "时间结果已保存到: " << filename.str() << std::endl;
    } else {
        std::cerr << "无法创建时间结果文件" << std::endl;
    }
}

#ifndef NO_MAIN
int main() {
    try {
        // JCH模拟参数
        int Nsites = 2;
        int Nqubits = 2;
        int cutoff = 16;
        double J = 0.1;      // 跳跃强度
        double omega_r = 1.0; // 谐振子频率
        double omega_q = 1.0; // 量子比特频率
        double g = 0.1;      // 耦合强度
        double tau = 0.1;    // 时间步长
        int timesteps = 10;  // 时间步数
        
        std::cout << "JCH Hamiltonian模拟电路创建成功" << std::endl;
        std::cout << "参数: Nsites=" << Nsites << ", Nqubits=" << Nqubits 
                  << ", J=" << J << ", omega_r=" << omega_r 
                  << ", omega_q=" << omega_q << ", g=" << g 
                  << ", tau=" << tau << ", 截断维度=" << cutoff << std::endl;
        
        // 运行单时间步模拟
        std::cout << "\n--- 单时间步模拟 ---" << std::endl;
        run_jch_simulation_circuit(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau, 1);
        
        // 运行多时间步模拟
        std::cout << "\n--- 多时间步模拟 (" << timesteps << " 步) ---" << std::endl;
        run_jch_simulation_circuit_display(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau, timesteps);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
#endif
