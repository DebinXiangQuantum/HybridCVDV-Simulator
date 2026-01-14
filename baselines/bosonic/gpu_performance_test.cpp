#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <fstream>
#include <functional>
#include "gpu/operators.cuh"

// 使用gpu命名空间
using namespace gpu;

// 操作符类型和名称映射
std::map<std::string, std::string> operator_type_names = {
    {"I", "Identity Gate"},
    {"X", "Pauli X Gate"},
    {"Y", "Pauli Y Gate"},
    {"Z", "Pauli Z Gate"},
    {"SPLUS", "S+ Gate"},
    {"SMINUS", "S- Gate"},
    {"P0", "Projection to |0>"},
    {"P1", "Projection to |1>"},
    {"r", "Phase Rotation"},
    {"d", "Displacement"},
    {"s", "Squeezing"},
    {"s2", "Two-mode Squeezing"},
    {"bs", "Beam Splitter"},
    {"cr", "Conditional Rotation"},
    {"cd", "Conditional Displacement"},
    {"ecd", "Echo Conditional Displacement"},
    {"cbs", "Conditional Beam Splitter"},
    {"snap", "SNAP Gate"},
    {"csnap", "Conditional SNAP Gate"},
    {"multisnap", "Multi-SNAP Gate"},
    {"multicsnap", "Multi-Conditional SNAP Gate"},
    {"sqr", "SQR Gate"},
    {"pnr", "Photon Number Resolving"},
    {"eswap", "Exponential SWAP"},
    {"csq", "Conditional Squeezing"},
    {"sum", "Sum Gate"},
    {"csum", "Conditional Sum Gate"},
    {"jc", "Jaynes-Cummings"},
    {"ajc", "Anti Jaynes-Cummings"},
    {"rb", "Rabi Interaction"}
};

// 性能测试结果结构
struct OperatorPerformanceResult {
    std::string operator_name;
    double single_operator_latency_ms;
    double batch_throughput;
    double memory_efficiency;
};

// 测量单个操作符延迟
std::pair<double, double> measure_single_operator_latency(const std::function<CUDASparseMatrix()>& op_func) {
    const int num_iterations = 100;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 第一次执行以确保所有初始化都完成
    CUDASparseMatrix result = op_func();
    
    // 测量实际执行时间
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        result = op_func();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // 计算内存效率 (非零元素数量 / (行数 * 列数))
    result.downloadFromDevice();
    double memory_efficiency = 0.0;
    if (result.rows() > 0 && result.cols() > 0) {
        // 由于CUDASparseMatrix没有直接获取非零元素数量的方法，我们通过temp_values_的大小来获取
        // 注意：这里需要访问private成员，所以我们需要修改CUDASparseMatrix类或添加一个方法来获取非零元素数量
        // 由于我们不能修改原始类，这里使用一种替代方法：将矩阵转换为密集矩阵并计算非零元素数量
        auto dense = result.toDense();
        int non_zero_count = 0;
        for (const auto& row : dense) {
            for (const auto& elem : row) {
                if (elem.real != 0.0 || elem.imag != 0.0) {
                    non_zero_count++;
                }
            }
        }
        memory_efficiency = static_cast<double>(non_zero_count) / (result.rows() * result.cols());
    }
    
    return { (total_time / num_iterations) * 1000.0, memory_efficiency };
}

// 测量批处理吞吐量
double measure_batch_throughput(const std::function<CUDASparseMatrix()>& op_func) {
    const int num_tasks = 100;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_tasks; ++i) {
        CUDASparseMatrix result = op_func();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // 计算吞吐量 (tasks/second)
    return num_tasks / total_time;
}

// 测试所有量子比特门的性能
std::vector<OperatorPerformanceResult> test_qubit_gates() {
    std::vector<OperatorPerformanceResult> results;
    
    std::cout << "测试量子比特门性能..." << std::endl;
    
    // 定义测试的门
    std::vector<std::pair<std::string, std::function<CUDASparseMatrix()>>> gates_to_test = {
        {"I", []() { return QubitGatesGPU::I(); }} ,
        {"X", []() { return QubitGatesGPU::X(); }} ,
        {"Y", []() { return QubitGatesGPU::Y(); }} ,
        {"Z", []() { return QubitGatesGPU::Z(); }} ,
        {"SPLUS", []() { return QubitGatesGPU::SPLUS(); }} ,
        {"SMINUS", []() { return QubitGatesGPU::SMINUS(); }} ,
        {"P0", []() { return QubitGatesGPU::P0(); }} ,
        {"P1", []() { return QubitGatesGPU::P1(); }} 
    };
    
    for (const auto& [gate_name, gate_func] : gates_to_test) {
        std::cout << "测试: " << operator_type_names[gate_name] << " (" << gate_name << ")" << std::endl;
        
        // 测量单个门延迟和内存效率
        auto [latency, memory_efficiency] = measure_single_operator_latency(gate_func);
        
        // 测量批处理吞吐量
        double throughput = measure_batch_throughput(gate_func);
        
        // 保存结果
        results.push_back({
            gate_name,
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

// 测试所有连续变量操作符的性能
std::vector<OperatorPerformanceResult> test_cv_operators() {
    std::vector<OperatorPerformanceResult> results;
    CVOperatorsGPU cv_ops;
    
    std::cout << "测试连续变量操作符性能..." << std::endl;
    
    // 测试截断维度
    int cutoff = 16;
    int cutoff_a = 8;
    int cutoff_b = 8;
    
    // 定义测试的操作符
    std::vector<std::pair<std::string, std::function<CUDASparseMatrix()>>> ops_to_test = {
        {"r", [&cv_ops, cutoff]() { return cv_ops.r(M_PI / 4.0, cutoff); }} ,
        {"d", [&cv_ops, cutoff]() { return cv_ops.d(Complex(0.5, 0.2), cutoff); }} ,
        {"s", [&cv_ops, cutoff]() { return cv_ops.s(Complex(0.3, 0.1), cutoff); }} ,
        {"s2", [&cv_ops, cutoff_a, cutoff_b]() { return cv_ops.s2(Complex(0.2, 0.1), cutoff_a, cutoff_b); }} ,
        {"bs", [&cv_ops, cutoff_a, cutoff_b]() { return cv_ops.bs(Complex(M_PI / 3.0, 0), cutoff_a, cutoff_b); }} ,
        {"cr", [&cv_ops, cutoff]() { return cv_ops.cr(M_PI / 4.0, cutoff); }} ,
        {"cd", [&cv_ops, cutoff]() { return cv_ops.cd(Complex(0.4, 0.0), nullptr, cutoff); }} ,
        {"ecd", [&cv_ops, cutoff]() { return cv_ops.ecd(Complex(0.3, 0.0), cutoff); }} ,
        {"cbs", [&cv_ops, cutoff_a, cutoff_b]() { return cv_ops.cbs(Complex(M_PI / 3.0, 0), cutoff_a, cutoff_b); }} ,
        {"snap", [&cv_ops, cutoff]() { return cv_ops.snap(M_PI / 2.0, 2, cutoff); }} ,
        {"csnap", [&cv_ops, cutoff]() { return cv_ops.csnap(M_PI / 2.0, 2, cutoff); }} ,
        {"multisnap", [&cv_ops, cutoff]() { 
            std::vector<double> params = {M_PI / 2.0, M_PI / 4.0, 2.0, 3.0};
            return cv_ops.multisnap(params, cutoff); 
        }} ,
        {"multicsnap", [&cv_ops, cutoff]() { 
            std::vector<double> params = {M_PI / 2.0, M_PI / 4.0, 2.0, 3.0};
            return cv_ops.multicsnap(params, cutoff); 
        }} ,
        {"sqr", [&cv_ops, cutoff]() { 
            std::vector<double> params = {M_PI / 2.0, 0.0, 2.0};
            return cv_ops.sqr(params, cutoff); 
        }} ,
        {"pnr", [&cv_ops, cutoff]() { return cv_ops.pnr(4, cutoff); }} ,
        {"eswap", [&cv_ops, cutoff_a, cutoff_b]() { return cv_ops.eswap(M_PI / 4.0, cutoff_a, cutoff_b); }} ,
        {"csq", [&cv_ops, cutoff]() { return cv_ops.csq(Complex(0.3, 0.1), cutoff); }} ,
        {"sum", [&cv_ops, cutoff_a, cutoff_b]() { return cv_ops.sum(0.5, cutoff_a, cutoff_b); }} ,
        {"csum", [&cv_ops, cutoff_a, cutoff_b]() { return cv_ops.csum(0.5, cutoff_a, cutoff_b); }} ,
        {"jc", [&cv_ops, cutoff]() { return cv_ops.jc(M_PI / 4.0, 0.0, cutoff); }} ,
        {"ajc", [&cv_ops, cutoff]() { return cv_ops.ajc(M_PI / 4.0, 0.0, cutoff); }} ,
        {"rb", [&cv_ops, cutoff]() { return cv_ops.rb(Complex(0.3, 0.1), cutoff); }} 
    };
    
    for (const auto& [op_name, op_func] : ops_to_test) {
        std::cout << "测试: " << operator_type_names[op_name] << " (" << op_name << ")" << std::endl;
        
        try {
            // 测量单个操作符延迟和内存效率
            auto [latency, memory_efficiency] = measure_single_operator_latency(op_func);
            
            // 测量批处理吞吐量
            double throughput = measure_batch_throughput(op_func);
            
            // 保存结果
            results.push_back({
                op_name,
                latency,
                throughput,
                memory_efficiency
            });
            
            std::cout << "  单个操作符延迟: " << std::fixed << std::setprecision(6) << latency << " ms" << std::endl;
            std::cout << "  批处理吞吐量: " << std::fixed << std::setprecision(2) << throughput << " tasks/second" << std::endl;
            std::cout << "  内存效率: " << std::fixed << std::setprecision(4) << memory_efficiency << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  测试失败: " << e.what() << std::endl;
            
            // 保存失败结果
            results.push_back({
                op_name,
                -1.0,  // 失败标记
                -1.0,  // 失败标记
                -1.0   // 失败标记
            });
        }
        
        std::cout << std::endl;
    }
    
    return results;
}

// 将结果保存为JSON文件
void save_results_to_json(
    const std::vector<OperatorPerformanceResult>& qubit_results,
    const std::vector<OperatorPerformanceResult>& cv_results) {
    std::ofstream file("gpu_performance_results.json");
    
    file << "{\n";
    file << "  \"simulator\": \"HybridCVDV-Simulator-GPU\",\n";
    file << "  \"truncation_dimension\": 16,\n";
    file << "  \"timestamp\": \"" << std::time(nullptr) << "\",\n";
    file << "  \"qubit_gates\": [\n";
    
    for (size_t i = 0; i < qubit_results.size(); ++i) {
        const auto& result = qubit_results[i];
        file << "    {\n";
        file << "      \"operator_name\": \"" << result.operator_name << "\",\n";
        file << "      \"operator_full_name\": \"" << operator_type_names[result.operator_name] << "\",\n";
        file << "      \"single_operator_latency_ms\": " << std::fixed << std::setprecision(6) << result.single_operator_latency_ms << ",\n";
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
        const auto& result = cv_results[i];
        file << "    {\n";
        file << "      \"operator_name\": \"" << result.operator_name << "\",\n";
        file << "      \"operator_full_name\": \"" << operator_type_names[result.operator_name] << "\",\n";
        file << "      \"single_operator_latency_ms\": " << std::fixed << std::setprecision(6) << result.single_operator_latency_ms << ",\n";
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
    std::cout << "性能结果已保存到 gpu_performance_results.json" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << "   HybridCVDV-Simulator GPU性能测试" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "测试目的: 测量GPU操作符的性能指标" << std::endl;
    std::cout << "截断维度: 16" << std::endl;
    std::cout << "=========================================" << std::endl << std::endl;
    
    try {
        // 测试所有量子比特门的性能
        auto qubit_results = test_qubit_gates();
        
        // 测试所有连续变量操作符的性能
        auto cv_results = test_cv_operators();
        
        // 打印量子比特门汇总表
        std::cout << "=========================================" << std::endl;
        std::cout << "量子比特门性能测试汇总" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << std::setw(20) << std::left << "操作符";
        std::cout << std::setw(25) << std::left << "单个门延迟(ms)";
        std::cout << std::setw(25) << std::left << "批处理吞吐量(tasks/s)";
        std::cout << std::setw(15) << std::left << "内存效率" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        for (const auto& result : qubit_results) {
            std::cout << std::setw(20) << std::left << result.operator_name;
            std::cout << std::setw(25) << std::left << std::fixed << std::setprecision(6) << result.single_operator_latency_ms;
            std::cout << std::setw(25) << std::left << std::fixed << std::setprecision(2) << result.batch_throughput;
            std::cout << std::setw(15) << std::left << std::fixed << std::setprecision(4) << result.memory_efficiency << std::endl;
        }
        
        // 打印连续变量操作符汇总表
        std::cout << "\n=========================================" << std::endl;
        std::cout << "连续变量操作符性能测试汇总" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << std::setw(20) << std::left << "操作符";
        std::cout << std::setw(25) << std::left << "单个操作符延迟(ms)";
        std::cout << std::setw(25) << std::left << "批处理吞吐量(tasks/s)";
        std::cout << std::setw(15) << std::left << "内存效率" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        for (const auto& result : cv_results) {
            std::cout << std::setw(20) << std::left << result.operator_name;
            std::cout << std::setw(25) << std::left << std::fixed << std::setprecision(6) << result.single_operator_latency_ms;
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
