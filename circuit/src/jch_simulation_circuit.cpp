#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include "quantum_circuit.h"

/**
 * JCH模拟电路
 * 实现JCH哈密顿量的单时间步演化
 */
void run_jch_simulation_circuit(int Nsites, int Nqubits, int cutoff, double J, double omega_r, double omega_q, double g, double tau) {
    QuantumCircuit circuit(Nqubits, Nsites, cutoff, 32);
    
    // 实现JCH哈密顿量的Trotter分解
    
    // 1. 谐振子项 (omega_r * a†a)
    for (int i = 0; i < Nsites; ++i) {
        double angle = omega_r * tau;
        circuit.add_gate(Gates::PhaseRotation(i, angle));
    }
    
    // 2. 量子比特项 (omega_q * σz/2)
    for (int i = 0; i < Nqubits; ++i) {
        double angle = omega_q * tau / 2.0;
        circuit.add_gate(Gates::RotationZ(i, angle));
    }
    
    // 3. 耦合项 (g * (a†σ- + aσ+))
    for (int i = 0; i < std::min(Nsites, Nqubits); ++i) {
        double angle = g * tau;
        circuit.add_gate(Gates::JaynesCummings(i, i, angle));
    }
    
    // 4. 跳跃项 (J * (a†b + ab†))
    for (int i = 0; i < Nsites - 1; ++i) {
        double angle = J * tau;
        circuit.add_gate(Gates::BeamSplitter(i, i+1, angle));
    }
    
    // 构建并执行电路
    circuit.build();
    circuit.execute();
    
    // 获取统计信息
    auto stats = circuit.get_stats();
    std::cout << "单时间步电路统计: " << stats.num_gates << " 个门, "
              << stats.active_states << " 个活跃状态" << std::endl;
    
    // 获取时间统计信息
    auto time_stats = circuit.get_time_stats();
    std::cout << "时间统计: 总时间=" << time_stats.total_time << " ms, "
              << "传输时延=" << time_stats.transfer_time << " ms, "
              << "计算时延=" << time_stats.computation_time << " ms" << std::endl;
    
    // 保存时间结果到文件
    std::stringstream filename;
    filename << "result/jch_simulation_single_qubits_" << Nqubits << "_sites_" << Nsites << "_cutoff_" << cutoff << ".csv";
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
 * 多时间步JCH模拟电路
 */
void run_jch_simulation_circuit_display(int Nsites, int Nqubits, int cutoff, double J, double omega_r, double omega_q, double g, double tau, int timesteps) {
    QuantumCircuit circuit(Nqubits, Nsites, cutoff, 32);
    
    // 多次应用单时间步电路
    for (int t = 0; t < timesteps; ++t) {
        // 1. 谐振子项
        for (int i = 0; i < Nsites; ++i) {
            double angle = omega_r * tau;
            circuit.add_gate(Gates::PhaseRotation(i, angle));
        }
        
        // 2. 量子比特项
        for (int i = 0; i < Nqubits; ++i) {
            double angle = omega_q * tau / 2.0;
            circuit.add_gate(Gates::RotationZ(i, angle));
        }
        
        // 3. 耦合项
        for (int i = 0; i < std::min(Nsites, Nqubits); ++i) {
            double angle = g * tau;
            circuit.add_gate(Gates::JaynesCummings(i, i, angle));
        }
        
        // 4. 跳跃项
        for (int i = 0; i < Nsites - 1; ++i) {
            double angle = J * tau;
            circuit.add_gate(Gates::BeamSplitter(i, i+1, angle));
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
    filename << "result/jch_simulation_multi_qubits_" << Nqubits << "_sites_" << Nsites << "_timesteps_" << timesteps << "_cutoff_" << cutoff << ".csv";
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
        // 创建JCH模拟电路
        int Nsites = 2;
        int Nqubits = 2;
        int cutoff = 16;
        double J = 1.0;
        double omega_r = 1.0;
        double omega_q = 1.0;
        double g = 0.5;
        double tau = 0.1;
        int timesteps = 5;
        
        std::cout << "单时间步JCH模拟电路创建成功" << std::endl;
        run_jch_simulation_circuit(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau);
        
        // 创建多时间步JCH模拟电路
        std::cout << "\n多时间步JCH模拟电路创建成功" << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
        run_jch_simulation_circuit_display(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau, timesteps);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
