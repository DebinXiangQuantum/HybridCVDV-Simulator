#include "QBatchSimulator.hpp"
#include "cxxopts.hpp"
#include "dd/Export.hpp"
#include "nlohmann/json.hpp"


#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <regex>
#include <filesystem>

namespace nl = nlohmann;
namespace fs = std::filesystem;

// 检查并创建对应的 input_batch 文件
void check_and_create_input_batch_file(const std::string& qasm_file);

// 异常处理函数，用于捕获 CUDA 内存不足等异常
void handle_cuda_exception(const std::string& qasm_file, const std::string& output_path);

// 检查并创建对应的 input_batch 文件
void check_and_create_input_batch_file(const std::string& qasm_file) {
    std::cout << "Checking input_batch file for: " << qasm_file << std::endl;
    // 从 QASM 文件名中提取量子比特数
    std::regex qubit_pattern(R"(nq(\d+))");
    std::smatch match;
    if (std::regex_search(qasm_file, match, qubit_pattern)) {
        int num_qubits = std::stoi(match[1]);
        std::cout << "Extracted num_qubits: " << num_qubits << std::endl;
        
        // 构建 input_batch 文件路径
        std::string input_batch_file = "./input_batch/n" + std::to_string(num_qubits) + ".txt";
        std::cout << "Checking input_batch file: " << input_batch_file << std::endl;
        
        // 检查文件是否存在
        if (!fs::exists(input_batch_file)) {
            std::cout << "Creating input_batch file: " << input_batch_file << std::endl;
            
            // 确保 input_batch 目录存在
            fs::create_directories("./input_batch");
            
            // 计算量子态维度
            size_t n_dim = 1ULL << num_qubits; // 2^num_qubits
            std::cout << "Calculated n_dim: " << n_dim << std::endl;
            
            // 创建并写入文件
            std::ofstream out_file(input_batch_file);
            if (out_file.is_open()) {
                for (size_t i = 0; i < n_dim; i++) {
                    if (i == 0) {
                        out_file << "1.0 0.0\n"; // 初始态 |000...0>
                    } else {
                        out_file << "0.0 0.0\n";
                    }
                }
                out_file.close();
                std::cout << "Created input_batch file with " << n_dim << " entries" << std::endl;
            } else {
                std::cerr << "Failed to create input_batch file: " << input_batch_file << std::endl;
            }
        } else {
            std::cout << "Input_batch file already exists: " << input_batch_file << std::endl;
        }
    } else {
        std::cout << "No qubit pattern found in filename: " << qasm_file << std::endl;
    }
}

int main(int argc, char** argv) { // NOLINT(bugprone-exception-escape)
    cxxopts::Options options("Quantum Batch Sim", "Quantum Batch Sim");
    // clang-format off
    options.add_options()
        ("h,help", "produce help message")
        ("pv", "save the state vector")
        ("ps", "print simulation stats (applied gates, sim. time, and maximal size of the DD")
        ("export_fused_gates", "export the fused gates")
        ("batch_size", "number of states in a batch (integer)", cxxopts::value<int>())
        ("num_batch", "number of batches (integer)", cxxopts::value<int>())
        ("conversion_type", "DD-to-ELL conversion type: GPU (0), CPU (1), Mixed (2)", cxxopts::value<int>())
        ("output", "write machine-readable JSON result", cxxopts::value<std::string>())
        ("file", "simulate a quantum circuit given by file (detection by the file extension)", cxxopts::value<std::string>());

    // clang-format on

    auto vm = options.parse(argc, argv);
    if (vm.count("help") > 0) {
        std::cout << options.help();
        std::exit(0);
    }

    const int batch_size     = vm["batch_size"].as<int>();
    const int num_batch      = vm["num_batch"].as<int>();
    const int ddell_conversion = vm["conversion_type"].as<int>();
    // const int conversion_edge_thresh = vm["conversion_edge_thresh"].as<int>();

    std::unique_ptr<qc::QuantumComputation>              quantumComputation;
    std::unique_ptr<QBatchSimulator<dd::DDPackageConfig>>  qbatchsim{nullptr};
    const bool                                           verbose = vm.count("verbose") > 0;
    std::string fname = "";
    const std::string output_path =
        vm.count("output") > 0 ? vm["output"].as<std::string>() : "";

    try {
        if (vm.count("file") > 0) {
            fname = vm["file"].as<std::string>();
            
            // 检查并创建对应的 input_batch 文件
            check_and_create_input_batch_file(fname);
            
            quantumComputation      = std::make_unique<qc::QuantumComputation>(fname);
            qbatchsim               = std::make_unique<QBatchSimulator<dd::DDPackageConfig>>(std::move(quantumComputation), batch_size, num_batch);
        } else {
            std::cerr << "Did not find anything to simulate. See help below.\n"
                      << options.help() << "\n";
            std::exit(1);
        }

        if (qbatchsim->getNumberOfQubits() > 100) {
            std::clog << "[WARNING] Quantum computation contains quite a few qubits. You're jumping into the deep end.\n";
        }
        if (vm.count("export_fused_gates") > 0) {
            qbatchsim->export_fused_gates = true;
        }
        qbatchsim->ddell_conversion = ddell_conversion;
        // qbatchsim->conversion_edge_thresh = conversion_edge_thresh;
        auto begin = std::chrono::high_resolution_clock::now();
        qbatchsim->simulate();
        auto end = std::chrono::high_resolution_clock::now();


        std::cout << "Simulation finished" << std::endl;
        // init check
        bool *identical_d;
        bool *identical_h;
        checkCudaErrors(cudaMalloc((void**)&identical_d, qbatchsim->nDim*sizeof(bool)));
        checkCudaErrors(cudaMallocHost((void**)&identical_h, qbatchsim->nDim*sizeof(bool)));
        initial_check<<<qbatchsim->nDim, batch_size, batch_size*sizeof(bool)>>>(qbatchsim->d_batch[qbatchsim->final_state_idx_gpu], identical_d, batch_size);
        checkCudaErrors(cudaMemcpy(identical_h, identical_d, qbatchsim->nDim*sizeof(bool), cudaMemcpyDeviceToHost));
        bool identical_res = true;
        for (int i = 0; i < qbatchsim->nDim; i++) {
            if (!identical_h[i]) {
            identical_res = false;
            break;
            }
        }
        std::cout << "Initial check: "<<identical_res << std::endl;
        checkCudaErrors(cudaFree(identical_d));
        checkCudaErrors(cudaFreeHost(identical_h));

        
        nl::json outputObj;

        if (vm.count("pv") > 0) {
            cuDoubleComplex* state_vector;
            state_vector = qbatchsim->getVector();
            std::ofstream outputFile("../../log/results/state/qbsim_"+qbatchsim->getName()+".txt");
            if (outputFile.is_open()) {
                // Write the vector to the file
                for (size_t i = 0; i < qbatchsim->nDim; i++) {
                    outputFile << state_vector[i*batch_size].x << " " << state_vector[i*batch_size].y << std::endl;
                }
                // Close the file
                outputFile.close();
                std::cout << "Data saved to file." << std::endl;
            } else {
                std::cerr << "Failed to open the file." << std::endl;
            }
        }
        
        outputObj["statistics"] = {
                {"simulation_time", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()},
                {"benchmark", qbatchsim->getName()},
                {"n_qubits", +qbatchsim->getNumberOfQubits()},
                {"applied_gates", qbatchsim->getNumberOfOps()}
        };


        std::cout << std::setw(2) << outputObj << std::endl;
        
        // 输出到 CSV 文件
        std::string csv_filename = "./log/results/bqsim_results.csv";
        
        std::ofstream csv_file;
        if (output_path.empty()) {
            std::filesystem::create_directories("./log/results");
            bool file_exists = std::filesystem::exists(csv_filename);
            if (!file_exists) {
                csv_file.open(csv_filename);
                csv_file << "电路名,电路类型,总时间,传输时间,计算时间,内存占用,门数" << std::endl;
            } else {
                csv_file.open(csv_filename, std::ios_base::app);
            }
        }
        
        // 提取电路类型
        std::string circuit_type = "unknown";
        std::string circuit_name = qbatchsim->getName();
        if (circuit_name.find("jch") != std::string::npos) {
            circuit_type = "jch";
        } else if (circuit_name.find("vqe") != std::string::npos) {
            circuit_type = "vqe";
        } else if (circuit_name.find("transfer") != std::string::npos) {
            circuit_type = "transfer";
        } else if (circuit_name.find("qft") != std::string::npos) {
            circuit_type = "qft";
        } else if (circuit_name.find("qaoa") != std::string::npos) {
            circuit_type = "qaoa";
        } else if (circuit_name.find("shors") != std::string::npos) {
            circuit_type = "shors";
        } else if (circuit_name.find("cat") != std::string::npos) {
            circuit_type = "cat";
        } else if (circuit_name.find("gkp") != std::string::npos) {
            circuit_type = "gkp";
        }
        
        // 计算时间和内存
        double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        double transfer_time = qbatchsim->get_transfer_time();
        double computation_time = qbatchsim->get_computation_time();
        size_t peak_memory = qbatchsim->get_peak_memory_usage();
        int gate_count = qbatchsim->getNumberOfOps();

        double checksum = 0.0;
        cuDoubleComplex* final_state = qbatchsim->getVector();
        for (size_t index = 0; index < qbatchsim->nDim; ++index) {
            checksum += (static_cast<double>(index) + 1.0) *
                        (final_state[index * batch_size].x + final_state[index * batch_size].y);
        }

        if (!output_path.empty()) {
            fs::path result_path(output_path);
            if (result_path.has_parent_path()) {
                fs::create_directories(result_path.parent_path());
            }
            nl::json result = {
                {"schema_version", "3.0"},
                {"status", "ok"},
                {"system", "bqsim"},
                {"case_name", circuit_name},
                {"gpu_count", 1},
                {"timing", {
                    {"total_wall_ms", total_time},
                    {"simulation_ms", total_time},
                    {"gpu_compute_ms", computation_time},
                    {"h2d_ms", qbatchsim->get_h2d_time()},
                    {"d2h_ms", qbatchsim->get_d2h_time()},
                }},
                {"memory", {{"gpu_memory_peak_bytes", peak_memory}}},
                {"communication", nl::json::object()},
                {"throughput", {
                    {"completed_input_states", static_cast<long long>(batch_size) * num_batch},
                    {"completed_batches", num_batch},
                    {"completed_gate_applications",
                     static_cast<long long>(gate_count) * batch_size * num_batch},
                }},
                {"correctness", {
                    {"initial_check", identical_res},
                    {"checksum", checksum},
                }},
            };
            std::ofstream result_file(output_path);
            result_file << std::setw(2) << result << "\n";
        }
        if (output_path.empty()) {
            csv_file << circuit_name << "," << circuit_type << "," << total_time << "," << transfer_time << "," << computation_time << "," << peak_memory << "," << gate_count << std::endl;
            std::cout << "Results saved to " << csv_filename << std::endl;
            csv_file.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error running BQSim: " << e.what() << std::endl;
        // 处理异常，保存错误信息到 CSV
        handle_cuda_exception(fname, output_path);
        return 1;
    }
    return 0;
}

// 异常处理函数，用于捕获 CUDA 内存不足等异常
void handle_cuda_exception(const std::string& qasm_file, const std::string& output_path) {
    // 从 QASM 文件名中提取电路名称
    std::string circuit_name = qasm_file;
    size_t last_slash = circuit_name.find_last_of('/');
    if (last_slash != std::string::npos) {
        circuit_name = circuit_name.substr(last_slash + 1);
    }
    size_t last_dot = circuit_name.find_last_of('.');
    if (last_dot != std::string::npos) {
        circuit_name = circuit_name.substr(0, last_dot);
    }
    
    // 提取电路类型
    std::string circuit_type = "unknown";
    if (circuit_name.find("jch") != std::string::npos) {
        circuit_type = "jch";
    } else if (circuit_name.find("vqe") != std::string::npos) {
        circuit_type = "vqe";
    } else if (circuit_name.find("transfer") != std::string::npos) {
        circuit_type = "transfer";
    } else if (circuit_name.find("qft") != std::string::npos) {
        circuit_type = "qft";
    } else if (circuit_name.find("qaoa") != std::string::npos) {
        circuit_type = "qaoa";
    } else if (circuit_name.find("shors") != std::string::npos) {
        circuit_type = "shors";
    } else if (circuit_name.find("cat") != std::string::npos) {
        circuit_type = "cat";
    } else if (circuit_name.find("gkp") != std::string::npos) {
        circuit_type = "gkp";
    }
    
    if (!output_path.empty()) {
        fs::path result_path(output_path);
        if (result_path.has_parent_path()) {
            fs::create_directories(result_path.parent_path());
        }
        nl::json result = {
            {"schema_version", "3.0"},
            {"status", "crash_cuda"},
            {"system", "bqsim"},
            {"case_name", circuit_name},
        };
        std::ofstream result_file(output_path);
        result_file << std::setw(2) << result << "\n";
        return;
    }

    // Legacy CSV fallback.
    std::string csv_filename = "./log/results/bqsim_results.csv";
    
    // 确保目录存在
    std::filesystem::create_directories("./log/results");
    
    // 检查文件是否存在，如果不存在则创建并写入表头
    bool file_exists = std::filesystem::exists(csv_filename);
    std::ofstream csv_file;
    
    if (!file_exists) {
        csv_file.open(csv_filename);
        csv_file << "电路名,电路类型,总时间,传输时间,计算时间,内存占用,门数" << std::endl;
    } else {
        csv_file.open(csv_filename, std::ios_base::app);
    }
    
    // 写入错误信息
    csv_file << circuit_name << "," << circuit_type << ",-1,-1,-1,-1,-1" << std::endl;
    csv_file.close();
    
    std::cout << "Error results saved to " << csv_filename << std::endl;
}
