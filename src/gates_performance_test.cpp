#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <fstream>
#include <functional>
#include "quantum_circuit.h"

// 使用 Gates 命名空间
using namespace Gates;

// 门类型和名称映射
std::map<GateType, std::string> gate_type_names = {
    // CPU端纯Qubit门
    {GateType::HADAMARD, "Hadamard Gate"},
    {GateType::PAULI_X, "Pauli X Gate"},
    {GateType::PAULI_Y, "Pauli Y Gate"},
    {GateType::PAULI_Z, "Pauli Z Gate"},
    {GateType::ROTATION_X, "Rotation X Gate"},
    {GateType::ROTATION_Y, "Rotation Y Gate"},
    {GateType::ROTATION_Z, "Rotation Z Gate"},
    {GateType::PHASE_GATE_S, "Phase S Gate"},
    {GateType::PHASE_GATE_T, "Phase T Gate"},
    {GateType::CNOT, "CNOT Gate"},
    {GateType::CZ, "CZ Gate"},
    
    // GPU端纯Qumode门
    {GateType::PHASE_ROTATION, "Phase Rotation"},
    {GateType::KERR_GATE, "Kerr Gate"},
    {GateType::CREATION_OPERATOR, "Creation Operator"},
    {GateType::ANNIHILATION_OPERATOR, "Annihilation Operator"},
    {GateType::DISPLACEMENT, "Displacement Gate"},
    {GateType::SQUEEZING, "Squeezing Gate"},
    {GateType::BEAM_SPLITTER, "Beam Splitter"},
    
    // 混合门
    {GateType::CONDITIONAL_DISPLACEMENT, "Conditional Displacement"},
    {GateType::CONDITIONAL_SQUEEZING, "Conditional Squeezing"},
    {GateType::CONDITIONAL_BEAM_SPLITTER, "Conditional Beam Splitter"},
    {GateType::CONDITIONAL_TWO_MODE_SQUEEZING, "Conditional Two-Mode Squeezing"},
    {GateType::CONDITIONAL_SUM, "Conditional SUM"},
    {GateType::RABI_INTERACTION, "Rabi Interaction"},
    {GateType::JAYNES_CUMMINGS, "Jaynes-Cummings"},
    {GateType::ANTI_JAYNES_CUMMINGS, "Anti Jaynes-Cummings"},
    {GateType::SELECTIVE_QUBIT_ROTATION, "Selective Qubit Rotation"}
};

// 性能测试结果结构
struct GatePerformanceResult {
    std::string gate_name;
    std::string gate_full_name;
    double single_gate_latency_ms;
    double batch_throughput;
    double memory_efficiency;
};

// 测量单个门延迟
std::pair<double, double> measure_single_gate_latency(const std::function<void(QuantumCircuit&)>& gate_func, 
                                                       int num_qubits, int num_qumodes, int cutoff) {
    const int num_iterations = 100;
    
    // 第一次执行以确保所有初始化都完成
    {
        QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
        gate_func(circuit);
        circuit.build();
        circuit.execute();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
        gate_func(circuit);
        circuit.build();
        circuit.execute();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // 计算内存效率
    QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
    gate_func(circuit);
    circuit.build();
    circuit.execute();
    auto stats = circuit.get_stats();
    double memory_efficiency = static_cast<double>(stats.active_states) / stats.cv_truncation;
    
    return std::make_pair((total_time / num_iterations) * 1000.0, memory_efficiency);
}

// 测量批处理吞吐量
double measure_batch_throughput(const std::function<void(QuantumCircuit&)>& gate_func,
                                int num_qubits, int num_qumodes, int cutoff) {
    const int num_tasks = 100;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_tasks; ++i) {
        QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, 32);
        gate_func(circuit);
        circuit.build();
        circuit.execute();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // 计算吞吐量 (tasks/second)
    return num_tasks / total_time;
}

// 测试所有量子比特门的性能
std::vector<GatePerformanceResult> test_qubit_gates() {
    std::vector<GatePerformanceResult> results;
    int cutoff = 16;
    
    std::cout << "测试量子比特门性能..." << std::endl;
    
    // 定义测试的门 - 量子比特门需要至少1个qumode来创建有效的量子电路
    std::vector<std::pair<GateType, std::function<void(QuantumCircuit&)>>> gates_to_test;
    gates_to_test.push_back(std::make_pair(GateType::HADAMARD, [](QuantumCircuit& c) { c.add_gate(Hadamard(0)); }));
    gates_to_test.push_back(std::make_pair(GateType::PAULI_X, [](QuantumCircuit& c) { c.add_gate(PauliX(0)); }));
    gates_to_test.push_back(std::make_pair(GateType::PAULI_Y, [](QuantumCircuit& c) { c.add_gate(PauliY(0)); }));
    gates_to_test.push_back(std::make_pair(GateType::PAULI_Z, [](QuantumCircuit& c) { c.add_gate(PauliZ(0)); }));
    gates_to_test.push_back(std::make_pair(GateType::ROTATION_X, [](QuantumCircuit& c) { c.add_gate(RotationX(0, M_PI / 4.0)); }));
    gates_to_test.push_back(std::make_pair(GateType::ROTATION_Y, [](QuantumCircuit& c) { c.add_gate(RotationY(0, M_PI / 4.0)); }));
    gates_to_test.push_back(std::make_pair(GateType::ROTATION_Z, [](QuantumCircuit& c) { c.add_gate(RotationZ(0, M_PI / 4.0)); }));
    gates_to_test.push_back(std::make_pair(GateType::PHASE_GATE_S, [](QuantumCircuit& c) { c.add_gate(PhaseGateS(0)); }));
    gates_to_test.push_back(std::make_pair(GateType::PHASE_GATE_T, [](QuantumCircuit& c) { c.add_gate(PhaseGateT(0)); }));
    gates_to_test.push_back(std::make_pair(GateType::CNOT, [](QuantumCircuit& c) { c.add_gate(CNOT(0, 1)); }));
    gates_to_test.push_back(std::make_pair(GateType::CZ, [](QuantumCircuit& c) { c.add_gate(CZ(0, 1)); }));
    
    for (const auto& gate_pair : gates_to_test) {
        GateType gate_type = gate_pair.first;
        std::function<void(QuantumCircuit&)> gate_func = gate_pair.second;
        std::string gate_name = gate_type_names[gate_type];
        std::cout << "测试: " << gate_name << std::endl;
        
        int num_qubits = (gate_type == GateType::CNOT || gate_type == GateType::CZ) ? 2 : 1;
        // 修复：量子比特门也需要至少1个qumode，否则电路无法正确初始化
        int num_qumodes = 1;
        
        try {
            // 测量单个门延迟和内存效率
            std::pair<double, double> latency_result = measure_single_gate_latency(gate_func, num_qubits, num_qumodes, cutoff);
            double latency = latency_result.first;
            double memory_efficiency = latency_result.second;
            
            // 测量批处理吞吐量
            double throughput = measure_batch_throughput(gate_func, num_qubits, num_qumodes, cutoff);
            
            // 保存结果
            GatePerformanceResult result;
            result.gate_name = std::to_string(static_cast<int>(gate_type));
            result.gate_full_name = gate_name;
            result.single_gate_latency_ms = latency;
            result.batch_throughput = throughput;
            result.memory_efficiency = memory_efficiency;
            results.push_back(result);
            
            std::cout << "  单个门延迟: " << std::fixed << std::setprecision(6) << latency << " ms" << std::endl;
            std::cout << "  批处理吞吐量: " << std::fixed << std::setprecision(2) << throughput << " tasks/second" << std::endl;
            std::cout << "  内存效率: " << std::fixed << std::setprecision(4) << memory_efficiency << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  测试失败: " << e.what() << std::endl;
            
            // 保存失败结果
            GatePerformanceResult result;
            result.gate_name = std::to_string(static_cast<int>(gate_type));
            result.gate_full_name = gate_name;
            result.single_gate_latency_ms = -1.0;
            result.batch_throughput = -1.0;
            result.memory_efficiency = -1.0;
            results.push_back(result);
        }
        
        std::cout << std::endl;
    }
    
    return results;
}

// 测试所有连续变量操作符的性能
std::vector<GatePerformanceResult> test_cv_operators() {
    std::vector<GatePerformanceResult> results;
    int cutoff = 16;
    
    std::cout << "测试连续变量操作符性能..." << std::endl;
    
    // 定义测试的操作符
    std::vector<std::pair<GateType, std::function<void(QuantumCircuit&)>>> ops_to_test;
    ops_to_test.push_back(std::make_pair(GateType::PHASE_ROTATION, [](QuantumCircuit& c) { c.add_gate(PhaseRotation(0, M_PI / 4.0)); }));
    ops_to_test.push_back(std::make_pair(GateType::KERR_GATE, [](QuantumCircuit& c) { c.add_gate(KerrGate(0, 0.5)); }));
    ops_to_test.push_back(std::make_pair(GateType::CREATION_OPERATOR, [](QuantumCircuit& c) { c.add_gate(CreationOperator(0)); }));
    ops_to_test.push_back(std::make_pair(GateType::ANNIHILATION_OPERATOR, [](QuantumCircuit& c) { c.add_gate(AnnihilationOperator(0)); }));
    ops_to_test.push_back(std::make_pair(GateType::DISPLACEMENT, [](QuantumCircuit& c) { c.add_gate(Displacement(0, std::complex<double>(0.5, 0.2))); }));
    ops_to_test.push_back(std::make_pair(GateType::SQUEEZING, [](QuantumCircuit& c) { c.add_gate(Squeezing(0, std::complex<double>(0.3, 0.1))); }));
    ops_to_test.push_back(std::make_pair(GateType::BEAM_SPLITTER, [](QuantumCircuit& c) { c.add_gate(BeamSplitter(0, 1, M_PI / 3.0, 0.0)); }));
    ops_to_test.push_back(std::make_pair(GateType::CONDITIONAL_DISPLACEMENT, [](QuantumCircuit& c) { c.add_gate(ConditionalDisplacement(0, 0, std::complex<double>(0.4, 0.0))); }));
    ops_to_test.push_back(std::make_pair(GateType::CONDITIONAL_SQUEEZING, [](QuantumCircuit& c) { c.add_gate(ConditionalSqueezing(0, 0, std::complex<double>(0.3, 0.1))); }));
    ops_to_test.push_back(std::make_pair(GateType::CONDITIONAL_BEAM_SPLITTER, [](QuantumCircuit& c) { c.add_gate(ConditionalBeamSplitter(0, 0, 1, M_PI / 3.0, 0.0)); }));
    ops_to_test.push_back(std::make_pair(GateType::CONDITIONAL_TWO_MODE_SQUEEZING, [](QuantumCircuit& c) { c.add_gate(ConditionalTwoModeSqueezing(0, 0, 1, std::complex<double>(0.2, 0.1))); }));
    ops_to_test.push_back(std::make_pair(GateType::CONDITIONAL_SUM, [](QuantumCircuit& c) { c.add_gate(ConditionalSUM(0, 0, 1, 0.5, 0.0)); }));
    ops_to_test.push_back(std::make_pair(GateType::RABI_INTERACTION, [](QuantumCircuit& c) { c.add_gate(RabiInteraction(0, 0, M_PI / 4.0)); }));
    ops_to_test.push_back(std::make_pair(GateType::JAYNES_CUMMINGS, [](QuantumCircuit& c) { c.add_gate(JaynesCummings(0, 0, M_PI / 4.0, 0.0)); }));
    ops_to_test.push_back(std::make_pair(GateType::ANTI_JAYNES_CUMMINGS, [](QuantumCircuit& c) { c.add_gate(AntiJaynesCummings(0, 0, M_PI / 4.0, 0.0)); }));
    
    // SelectiveQubitRotation 需要特殊处理参数
    std::vector<double> theta_vec(16, M_PI / 2.0);
    std::vector<double> phi_vec(16, 0.0);
    ops_to_test.push_back(std::make_pair(GateType::SELECTIVE_QUBIT_ROTATION, [theta_vec, phi_vec](QuantumCircuit& c) { 
        c.add_gate(SelectiveQubitRotation(0, 0, theta_vec, phi_vec)); 
    }));
    
    for (const auto& op_pair : ops_to_test) {
        GateType op_type = op_pair.first;
        std::function<void(QuantumCircuit&)> op_func = op_pair.second;
        std::string op_name = gate_type_names[op_type];
        std::cout << "测试: " << op_name << std::endl;
        
        int num_qubits = 1;
        int num_qumodes = 1;
        
        // 根据门类型调整qubit和qumode数量
        if (op_type == GateType::BEAM_SPLITTER || 
            op_type == GateType::CONDITIONAL_TWO_MODE_SQUEEZING ||
            op_type == GateType::CONDITIONAL_SUM) {
            num_qumodes = 2;
        }
        
        try {
            // 测量单个操作符延迟和内存效率
            std::pair<double, double> latency_result = measure_single_gate_latency(op_func, num_qubits, num_qumodes, cutoff);
            double latency = latency_result.first;
            double memory_efficiency = latency_result.second;
            
            // 测量批处理吞吐量
            double throughput = measure_batch_throughput(op_func, num_qubits, num_qumodes, cutoff);
            
            // 保存结果
            GatePerformanceResult result;
            result.gate_name = std::to_string(static_cast<int>(op_type));
            result.gate_full_name = op_name;
            result.single_gate_latency_ms = latency;
            result.batch_throughput = throughput;
            result.memory_efficiency = memory_efficiency;
            results.push_back(result);
            
            std::cout << "  单个操作符延迟: " << std::fixed << std::setprecision(6) << latency << " ms" << std::endl;
            std::cout << "  批处理吞吐量: " << std::fixed << std::setprecision(2) << throughput << " tasks/second" << std::endl;
            std::cout << "  内存效率: " << std::fixed << std::setprecision(4) << memory_efficiency << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  测试失败: " << e.what() << std::endl;
            
            // 保存失败结果
            GatePerformanceResult result;
            result.gate_name = std::to_string(static_cast<int>(op_type));
            result.gate_full_name = op_name;
            result.single_gate_latency_ms = -1.0;
            result.batch_throughput = -1.0;
            result.memory_efficiency = -1.0;
            results.push_back(result);
        }
        
        std::cout << std::endl;
    }
    
    return results;
}

// 将结果保存为JSON文件
void save_results_to_json(
    const std::vector<GatePerformanceResult>& qubit_results,
    const std::vector<GatePerformanceResult>& cv_results) {
    std::ofstream file("hybridcvdv_gates_performance_results.json");
    
    file << "{\n";
    file << "  \"simulator\": \"HybridCVDV-Simulator\",\n";
    file << "  \"truncation_dimension\": 16,\n";
    file << "  \"timestamp\": \"" << std::time(nullptr) << "\",\n";
    file << "  \"qubit_gates\": [\n";
    
    for (size_t i = 0; i < qubit_results.size(); ++i) {
        const GatePerformanceResult& result = qubit_results[i];
        file << "    {\n";
        file << "      \"gate_type\": " << result.gate_name << ",\n";
        file << "      \"gate_full_name\": \"" << result.gate_full_name << "\",\n";
        file << "      \"single_gate_latency_ms\": " << std::fixed << std::setprecision(6) << result.single_gate_latency_ms << ",\n";
        file << "      \"batch_throughput\": " << std::fixed << std::setprecision(2) << result.batch_throughput << ",\n";
        file << "      \"memory_efficiency\": " << std::fixed << std::setprecision(4) << result.memory_efficiency << "\n";
        file << "    }";
        if (i < qubit_results.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ],\n";
    file << "  \"cv_operators\": [\n";
    
    for (size_t i = 0; i < cv_results.size(); ++i) {
        const GatePerformanceResult& result = cv_results[i];
        file << "    {\n";
        file << "      \"gate_type\": " << result.gate_name << ",\n";
        file << "      \"gate_full_name\": \"" << result.gate_full_name << "\",\n";
        file << "      \"single_gate_latency_ms\": " << std::fixed << std::setprecision(6) << result.single_gate_latency_ms << ",\n";
        file << "      \"batch_throughput\": " << std::fixed << std::setprecision(2) << result.batch_throughput << ",\n";
        file << "      \"memory_efficiency\": " << std::fixed << std::setprecision(4) << result.memory_efficiency << "\n";
        file << "    }";
        if (i < cv_results.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    std::cout << "性能结果已保存到 hybridcvdv_gates_performance_results.json" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << "   HybridCVDV-Simulator 门性能测试" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "测试目的: 测量HybridCVDV门操作的性能指标" << std::endl;
    std::cout << "截断维度: 16" << std::endl;
    std::cout << "=========================================" << std::endl << std::endl;
    
    try {
        // 测试所有量子比特门的性能
        std::vector<GatePerformanceResult> qubit_results = test_qubit_gates();
        
        // 测试所有连续变量操作符的性能
        std::vector<GatePerformanceResult> cv_results = test_cv_operators();
        
        // 打印量子比特门汇总表
        std::cout << "=========================================" << std::endl;
        std::cout << "量子比特门性能测试汇总" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << std::setw(25) << std::left << "门操作";
        std::cout << std::setw(25) << std::left << "单个门延迟(ms)";
        std::cout << std::setw(25) << std::left << "批处理吞吐量(tasks/s)";
        std::cout << std::setw(15) << std::left << "内存效率" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        for (const auto& result : qubit_results) {
            std::cout << std::setw(25) << std::left << result.gate_full_name;
            std::cout << std::setw(25) << std::left << std::fixed << std::setprecision(6) << result.single_gate_latency_ms;
            std::cout << std::setw(25) << std::left << std::fixed << std::setprecision(2) << result.batch_throughput;
            std::cout << std::setw(15) << std::left << std::fixed << std::setprecision(4) << result.memory_efficiency << std::endl;
        }
        
        // 打印连续变量操作符汇总表
        std::cout << "\n=========================================" << std::endl;
        std::cout << "连续变量操作符性能测试汇总" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << std::setw(30) << std::left << "操作符";
        std::cout << std::setw(25) << std::left << "单个操作符延迟(ms)";
        std::cout << std::setw(25) << std::left << "批处理吞吐量(tasks/s)";
        std::cout << std::setw(15) << std::left << "内存效率" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        for (const auto& result : cv_results) {
            std::cout << std::setw(30) << std::left << result.gate_full_name;
            std::cout << std::setw(25) << std::left << std::fixed << std::setprecision(6) << result.single_gate_latency_ms;
            std::cout << std::setw(25) << std::left << std::fixed << std::setprecision(2) << result.batch_throughput;
            std::cout << std::setw(15) << std::left << std::fixed << std::setprecision(4) << result.memory_efficiency << std::endl;
        }
        
        // 保存结果到JSON文件
        save_results_to_json(qubit_results, cv_results);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
