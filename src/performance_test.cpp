#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <fstream>
#include "quantum_circuit.h"
#include "batch_scheduler.h"

// 门类型和名称映射
std::map<GateType, std::string> gate_type_names = {
    {GateType::PHASE_ROTATION, "PHASE_ROTATION"},
    {GateType::KERR_GATE, "KERR_GATE"},
    {GateType::CONDITIONAL_PARITY, "CONDITIONAL_PARITY"},
    {GateType::CREATION_OPERATOR, "CREATION_OPERATOR"},
    {GateType::ANNIHILATION_OPERATOR, "ANNIHILATION_OPERATOR"},
    {GateType::DISPLACEMENT, "DISPLACEMENT"},
    {GateType::SQUEEZING, "SQUEEZING"},
    {GateType::BEAM_SPLITTER, "BEAM_SPLITTER"},
    {GateType::CONDITIONAL_DISPLACEMENT, "CONDITIONAL_DISPLACEMENT"},
    {GateType::CONDITIONAL_SQUEEZING, "CONDITIONAL_SQUEEZING"},
    {GateType::CONDITIONAL_BEAM_SPLITTER, "CONDITIONAL_BEAM_SPLITTER"},
    {GateType::CONDITIONAL_TWO_MODE_SQUEEZING, "CONDITIONAL_TWO_MODE_SQUEEZING"},
    {GateType::CONDITIONAL_SUM, "CONDITIONAL_SUM"},
    {GateType::RABI_INTERACTION, "RABI_INTERACTION"},
    {GateType::JAYNES_CUMMINGS, "JAYNES_CUMMINGS"},
    {GateType::ANTI_JAYNES_CUMMINGS, "ANTI_JAYNES_CUMMINGS"},
    {GateType::SELECTIVE_QUBIT_ROTATION, "SELECTIVE_QUBIT_ROTATION"}
};

// 性能测试结果结构
struct GatePerformanceResult {
    std::string gate_name;
    double single_gate_latency_ms;
    double batch_throughput;
    double memory_efficiency;
};

// 单个门性能测试
double measure_single_gate_latency(GateType gate_type, const std::vector<GateParams>& gates) {
    QuantumCircuit circuit(2, 2, 16, 32);
    circuit.add_gates(gates);
    circuit.build();
    
    const int num_iterations = 100;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        circuit.reset();
        circuit.build();  // 重置后需要重新构建
        circuit.execute();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    return (total_time / num_iterations) * 1000.0; // 转换为毫秒
}

// 批处理吞吐量测试
double measure_batch_throughput(GateType gate_type, const std::vector<GateParams>& gates) {
    QuantumCircuit circuit(2, 2, 16, 32);
    RuntimeScheduler scheduler(&circuit, 8);
    
    const int num_tasks = 100;
    for (int i = 0; i < num_tasks; ++i) {
        scheduler.schedule_gate(gates[0]);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    scheduler.execute_all();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // 计算吞吐量 (tasks/second)
    return num_tasks / total_time;
}

// 内存效率测试
double measure_memory_efficiency(const QuantumCircuit& circuit) {
    auto stats = circuit.get_stats();
    
    // 计算内存使用效率 (active_states / cv_truncation)
    return static_cast<double>(stats.active_states) / stats.cv_truncation;
}

// 测试所有门的性能
std::vector<GatePerformanceResult> test_all_gates() {
    std::vector<GatePerformanceResult> results;
    
    // 定义测试的门和参数
    std::vector<std::pair<GateType, std::vector<GateParams>>> gates_to_test = {
        {GateType::PHASE_ROTATION, {Gates::PhaseRotation(0, M_PI / 4.0)}},
        {GateType::KERR_GATE, {Gates::KerrGate(0, 0.5)}},
        {GateType::CREATION_OPERATOR, {Gates::CreationOperator(0)}},
        {GateType::ANNIHILATION_OPERATOR, {Gates::AnnihilationOperator(0)}},
        {GateType::DISPLACEMENT, {Gates::Displacement(0, std::complex<double>(0.5, 0.2))}},
        {GateType::SQUEEZING, {Gates::Squeezing(0, std::complex<double>(0.3, 0.1))}},
        {GateType::BEAM_SPLITTER, {Gates::BeamSplitter(0, 1, M_PI / 3.0)}},
        {GateType::CONDITIONAL_DISPLACEMENT, {Gates::ConditionalDisplacement(0, 0, std::complex<double>(0.4, 0.0))}},
        {GateType::CONDITIONAL_SQUEEZING, {Gates::ConditionalSqueezing(0, 0, std::complex<double>(0.3, 0.1))}},
        {GateType::RABI_INTERACTION, {Gates::RabiInteraction(0, 0, M_PI / 4.0)}}
    };
    
    for (const auto& [gate_type, gates] : gates_to_test) {
        std::cout << "测试门: " << gate_type_names[gate_type] << std::endl;
        
        // 单个门延迟测试
        double latency = measure_single_gate_latency(gate_type, gates);
        
        // 批处理吞吐量测试
        double throughput = measure_batch_throughput(gate_type, gates);
        
        // 内存效率测试
        QuantumCircuit circuit(2, 2, 16, 32);
        circuit.add_gates(gates);
        circuit.build();
        circuit.execute();
        double memory_efficiency = measure_memory_efficiency(circuit);
        
        // 保存结果
        results.push_back({
            gate_type_names[gate_type],
            latency,
            throughput,
            memory_efficiency
        });
        
        std::cout << "  单个门延迟: " << std::fixed << std::setprecision(6) << latency << " ms" << std::endl;
        std::cout << "  批处理吞吐量: " << std::fixed << std::setprecision(2) << throughput << " tasks/second" << std::endl;
        std::cout << "  内存效率: " << std::fixed << std::setprecision(4) << memory_efficiency << std::endl;
        std::cout << std::endl;
    }
    
    return results;
}

// 将结果保存为JSON文件
void save_results_to_json(const std::vector<GatePerformanceResult>& results) {
    std::ofstream file("performance_results.json");
    
    file << "{\n";
    file << "  \"simulator\": \"HybridCVDV-Simulator\",\n";
    file << "  \"truncation_dimension\": 16,\n";
    file << "  \"timestamp\": \"" << std::time(nullptr) << "\",\n";
    file << "  \"results\": [\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        file << "    {\n";
        file << "      \"gate_name\": \"" << result.gate_name << "\",\n";
        file << "      \"single_gate_latency_ms\": " << std::fixed << std::setprecision(6) << result.single_gate_latency_ms << ",\n";
        file << "      \"batch_throughput\": " << std::fixed << std::setprecision(2) << result.batch_throughput << ",\n";
        file << "      \"memory_efficiency\": " << std::fixed << std::setprecision(4) << result.memory_efficiency << "\n";
        file << "    }";
        if (i < results.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    std::cout << "性能结果已保存到 performance_results.json" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << "   HybridCVDV-Simulator 性能测试" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "测试目的: 测量每个门的性能指标" << std::endl;
    std::cout << "截断维度: 16" << std::endl;
    std::cout << "=========================================" << std::endl << std::endl;
    
    try {
        // 测试所有门的性能
        auto results = test_all_gates();
        
        // 打印汇总表
        std::cout << "=========================================" << std::endl;
        std::cout << "性能测试汇总" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << std::setw(30) << std::left << "门类型";
        std::cout << std::setw(20) << std::left << "单个门延迟(ms)";
        std::cout << std::setw(20) << std::left << "批处理吞吐量(tasks/s)";
        std::cout << std::setw(15) << std::left << "内存效率" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(30) << std::left << result.gate_name;
            std::cout << std::setw(20) << std::left << std::fixed << std::setprecision(6) << result.single_gate_latency_ms;
            std::cout << std::setw(20) << std::left << std::fixed << std::setprecision(2) << result.batch_throughput;
            std::cout << std::setw(15) << std::left << std::fixed << std::setprecision(4) << result.memory_efficiency << std::endl;
        }
        
        // 保存结果到JSON文件
        save_results_to_json(results);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
