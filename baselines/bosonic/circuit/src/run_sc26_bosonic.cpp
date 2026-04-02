#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <future>
#include <mutex>
#include "nlohmann/json.hpp"
#include "core/circuit.h"
#include "gates/gates.h"

// 外部电路函数声明
void run_binary_knapsack_vqe_circuit(int num_qubits, int num_qumodes, int cutoff, int ndepth, const std::vector<int>& nfocks, const std::vector<double>* Xvec = nullptr);
void run_jch_simulation_circuit_display(int Nsites, int Nqubits, int cutoff, double J, double omega_r, double omega_q, double g, double tau, int timesteps=5);
void run_cv_qaoa_circuit(int num_qubits, int num_qumodes, int cutoff, const std::vector<double>& params, double s, double a, int p, int n);
void run_qft_circuit(int num_qubits, int num_qumodes, int cutoff, double delta, int n, int a, int append);
void run_shors_circuit(int num_qubits, int num_qumodes, int cutoff, int N, int m, int R, int a, double delta);
void run_cat_state_circuit(int num_qubits, int num_qumodes, int cutoff, double alpha);
void run_gkp_state_circuit(int num_qubits, int num_qumodes, int cutoff, int N_rounds = 9, double r = 0.222, int qumode_idx = 0);
void run_state_transfer_CVtoDV(int num_qubits, int num_qumodes, int cutoff, double lambda = 0.29, bool apply_basis = true);
void run_state_transfer_DVtoCV(int num_qubits, int num_qumodes, int cutoff, double lambda = 0.29, bool apply_basis = true);

using json = nlohmann::json;

namespace fs = std::filesystem;

std::mutex results_mutex;

struct CircuitCase {
    std::string name;
    std::string workload;
    int cutoff;
    int num_modes;
    int num_qubits;
    std::map<std::string, double> params;
    std::map<std::string, int> int_params;
    std::vector<int> nfocks;
};

struct CircuitResult {
    std::string name;
    std::string workload;
    std::string status;
};

std::vector<CircuitCase> parse_json_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        exit(1);
    }

    json data;
    try {
        file >> data;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        exit(1);
    }

    std::vector<CircuitCase> cases;

    if (data.contains("cases") && data["cases"].is_array()) {
        for (const auto& case_data : data["cases"]) {
            CircuitCase c;

            if (case_data.contains("name")) {
                c.name = case_data["name"];
            }

            if (case_data.contains("workload")) {
                c.workload = case_data["workload"];
            }

            if (case_data.contains("cutoff")) {
                c.cutoff = case_data["cutoff"];
            }

            if (case_data.contains("num_modes")) {
                c.num_modes = case_data["num_modes"];
            }

            if (case_data.contains("num_qubits")) {
                c.num_qubits = case_data["num_qubits"];
            }
           
            if (case_data.contains("nfocks") && case_data["nfocks"].is_array()) {
                for (const auto& val : case_data["nfocks"]) {
                    c.nfocks.push_back(val);
                }
            }

            for (auto& [key, value] : case_data.items()) {
                if (key == "name" || key == "workload" || key == "cutoff" ||
                    key == "num_modes" || key == "num_qubits" || key == "nfocks") {
                    continue;
                }

                if (value.is_number_integer()) {
                    c.int_params[key] = value;
                } else if (value.is_number()) {
                    c.params[key] = value;
                }
            }

            cases.push_back(c);
        }
    }

    return cases;
}

CircuitResult run_circuit(const CircuitCase& c, const std::string& result_dir) {
    CircuitResult result;
    result.name = c.name;
    result.workload = c.workload;
    result.status = "success";

    std::string result_file;
    if (c.workload == "vqe_circuit") {
        result_file = result_dir + "/vqe_circuit_qubits_" + std::to_string(c.num_qubits) + "_qumodes_" + std::to_string(c.num_modes) + "_depth_2_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else if (c.workload == "qaoa_circuit") {
        result_file = result_dir + "/qaoa_circuit_qumodes_" + std::to_string(c.num_modes) + "_layers_1_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else if (c.workload == "shors_circuit") {
        result_file = result_dir + "/shors_circuit_N_15_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else if (c.workload == "cat_state_circuit") {
        result_file = result_dir + "/cat_state_circuit_qumodes_" + std::to_string(c.num_modes) + "_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else if (c.workload == "gkp_state_circuit") {
        result_file = result_dir + "/gkp_state_circuit_qumodes_" + std::to_string(c.num_modes) + "_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else if (c.workload == "state_transfer_CVtoDV_circuit") {
        result_file = result_dir + "/state_transfer_CVtoDV_qubits_" + std::to_string(c.num_qubits) + "_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else if (c.workload == "state_transfer_DVtoCV_circuit") {
        result_file = result_dir + "/state_transfer_DVtoCV_qubits_" + std::to_string(c.num_qubits) + "_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else if (c.workload == "qft_circuit") {
        result_file = result_dir + "/qft_circuit_qumodes_" + std::to_string(c.num_modes) + "_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else if (c.workload == "jch_simulation_circuit") {
        int timesteps = 5;
        if (c.int_params.find("timesteps") != c.int_params.end()) {
            timesteps = c.int_params.at("timesteps");
        }
        result_file = result_dir + "/jch_simulation_multi_qubits_" + std::to_string(c.num_qubits) + "_sites_" + std::to_string(c.num_modes) + "_timesteps_" + std::to_string(timesteps) + "_cutoff_" + std::to_string(c.cutoff) + ".csv";
    } else {
        result_file = result_dir + "/" + c.workload + "_" + c.name + ".csv";
    }
    
    if (fs::exists(result_file)) {
        std::cout << "Skipping: result file already exists" << std::endl;
        result.status = "already_completed";
        return result;
    }

    try {
        // 取消 cutoff 限制
        // if (c.cutoff >= 32) {
        //     std::cout << "  Skipping: cutoff >= 32 (too large)" << std::endl;
        //     result.status = "skipped_large_cutoff";
        // } else {
            std::cout << "Running circuit: " << c.name << std::endl;
            std::cout << "  Workload: " << c.workload << std::endl;
            std::cout << "  Cutoff: " << c.cutoff << std::endl;
            std::cout << "  Num Modes: " << c.num_modes << std::endl;
            std::cout << "  Num Qubits: " << c.num_qubits << std::endl;

        if (c.workload == "vqe_circuit") {
            // 取消 num_modes 和 num_qubits 限制
            // if (c.num_modes >= 9) {
            //     std::cout << "  Skipping: num_modes >= 9 (too large)" << std::endl;
            //     result.status = "skipped_large_qumode";
            // } else if (c.num_qubits >= 9) {
            //     std::cout << "  Skipping: num_qubits >= 9 (too large)" << std::endl;
            //     result.status = "skipped_large_qubit";
            // } else {
                int layers = 2;
                if (c.int_params.find("layers") != c.int_params.end()) {
                    layers = c.int_params.at("layers");
                }
                run_binary_knapsack_vqe_circuit(c.num_qubits, c.num_modes, c.cutoff, layers, c.nfocks);
            // }
        } else if (c.workload == "jch_simulation_circuit") {
            double J = c.params.at("J");
            double omega_r = c.params.at("omega_r");
            double omega_q = c.params.at("omega_q");
            double g = c.params.at("g");
            double tau = c.params.at("tau");
            int timesteps = c.int_params.at("timesteps");
            run_jch_simulation_circuit_display(c.num_modes, c.num_qubits, c.cutoff, J, omega_r, omega_q, g, tau,timesteps);
        } else if (c.workload == "qaoa_circuit") {
            int p = 1;
            if (c.int_params.find("p") != c.int_params.end()) {
                p = c.int_params.at("p");
            }
            int n = 2;
            if (c.int_params.find("n") != c.int_params.end()) {
                n = c.int_params.at("n");
            }
            double s = 0.5;
            if (c.params.find("s") != c.params.end()) {
                s = c.params.at("s");
            }
            double a = 1.0;
            if (c.params.find("a") != c.params.end()) {
                a = c.params.at("a");
            }
            std::vector<double> params(2 * p);
            for (int i = 0; i < 2 * p; ++i) {
                params[i] = 2 * M_PI * (i + 1) / (2 * p);
            }
            run_cv_qaoa_circuit(c.num_qubits, c.num_modes, c.cutoff, params, s, a, p, n);
        } else if (c.workload == "qft_circuit") {
            int n = 3;
            if (c.int_params.find("n") != c.int_params.end()) {
                n = c.int_params.at("n");
            }
            int a = 1;
            if (c.int_params.find("a") != c.int_params.end()) {
                a = c.int_params.at("a");
            }
            int append = 2;
            if (c.int_params.find("append") != c.int_params.end()) {
                append = c.int_params.at("append");
            }
            double delta = 0.1;
            if (c.params.find("delta") != c.params.end()) {
                delta = c.params.at("delta");
            }
            run_qft_circuit(c.num_qubits, c.num_modes, c.cutoff, delta, n, a, append);
        } else if (c.workload == "shors_circuit") {
            std::cout << "  Skipping: shors_circuit (known issue)" << std::endl;
            result.status = "skipped_shors_circuit";
        } else if (c.workload == "cat_state_circuit") {
            double alpha = 1.0;
            if (c.params.find("alpha") != c.params.end()) {
                alpha = c.params.at("alpha");
            }
            run_cat_state_circuit(c.num_qubits, c.num_modes, c.cutoff, alpha);
        } else if (c.workload == "gkp_state_circuit") {
            int rounds = 9;
            if (c.int_params.find("rounds") != c.int_params.end()) {
                rounds = c.int_params.at("rounds");
            }
            run_gkp_state_circuit(c.num_qubits, c.num_modes, c.cutoff, rounds);
        } else if (c.workload == "state_transfer_CVtoDV_circuit") {
            double lambda = 0.29;
            if (c.params.find("lambda") != c.params.end()) {
                lambda = c.params.at("lambda");
            }
            run_state_transfer_CVtoDV(c.num_qubits, c.num_modes, c.cutoff, lambda);
        } else if (c.workload == "state_transfer_DVtoCV_circuit") {
            double lambda = 0.29;
            if (c.params.find("lambda") != c.params.end()) {
                lambda = c.params.at("lambda");
            }
            run_state_transfer_DVtoCV(c.num_qubits, c.num_modes, c.cutoff, lambda);
        } else {
            result.status = "unknown_workload";
            std::cerr << "Unknown workload: " << c.workload << std::endl;
        }

        std::cout << "  Status: " << result.status << std::endl;
        std::cout << std::endl;
        // }
    } catch (const std::exception& e) {
        result.status = std::string("error: ") + e.what();
        std::cerr << "Error running circuit " << c.name << ": " << e.what() << std::endl;
        std::cout << "  Status: " << result.status << std::endl;
        std::cout << std::endl;
        
        // 保存错误信息到CSV文件
        std::ofstream error_file(result_file);
        if (error_file.is_open()) {
            error_file << "电路类型,参数,门数量,总时间,传输时间,计算时间,内存占用,错误信息\n";
            std::string params_str;
            if (c.workload == "vqe_circuit") {
                params_str = "num_qubits=" + std::to_string(c.num_qubits) + ",num_modes=" + std::to_string(c.num_modes) + ",cutoff=" + std::to_string(c.cutoff);
            } else if (c.workload == "jch_simulation_circuit") {
                int timesteps = 5;
                if (c.int_params.find("timesteps") != c.int_params.end()) {
                    timesteps = c.int_params.at("timesteps");
                }
                params_str = "num_qubits=" + std::to_string(c.num_qubits) + ",num_modes=" + std::to_string(c.num_modes) + ",timesteps=" + std::to_string(timesteps) + ",cutoff=" + std::to_string(c.cutoff);
            } else if (c.workload == "qft_circuit") {
                params_str = "num_qubits=" + std::to_string(c.num_qubits) + ",num_modes=" + std::to_string(c.num_modes) + ",cutoff=" + std::to_string(c.cutoff);
            } else if (c.workload == "cat_state_circuit") {
                params_str = "num_qubits=" + std::to_string(c.num_qubits) + ",num_modes=" + std::to_string(c.num_modes) + ",cutoff=" + std::to_string(c.cutoff);
            } else if (c.workload == "gkp_state_circuit") {
                params_str = "num_qubits=" + std::to_string(c.num_qubits) + ",num_modes=" + std::to_string(c.num_modes) + ",cutoff=" + std::to_string(c.cutoff);
            } else if (c.workload == "state_transfer_CVtoDV_circuit") {
                params_str = "num_qubits=" + std::to_string(c.num_qubits) + ",num_modes=" + std::to_string(c.num_modes) + ",cutoff=" + std::to_string(c.cutoff);
            } else if (c.workload == "state_transfer_DVtoCV_circuit") {
                params_str = "num_qubits=" + std::to_string(c.num_qubits) + ",num_modes=" + std::to_string(c.num_modes) + ",cutoff=" + std::to_string(c.cutoff);
            } else if (c.workload == "qaoa_circuit") {
                params_str = "num_qubits=" + std::to_string(c.num_qubits) + ",num_modes=" + std::to_string(c.num_modes) + ",cutoff=" + std::to_string(c.cutoff);
            } else {
                params_str = "cutoff=" + std::to_string(c.cutoff);
            }
            error_file << c.workload << "," << params_str << ",0,0,0,0,0," << e.what() << std::endl;
            error_file.close();
            std::cout << "Error info saved to: " << result_file << std::endl;
        }
    }

    return result;
}

void save_results_to_json(const std::vector<CircuitResult>& results, const std::string& output_path) {
    json output;

    output["schema_version"] = "2.0";
    output["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
    output["total_cases"] = results.size();
    output["successful_cases"] = std::count_if(results.begin(), results.end(),
        [](const CircuitResult& r) { return r.status == "success"; });
    output["failed_cases"] = std::count_if(results.begin(), results.end(),
        [](const CircuitResult& r) { return r.status != "success"; });

    json cases_json = json::array();
    for (const auto& result : results) {
        json case_json;
        case_json["name"] = result.name;
        case_json["workload"] = result.workload;
        case_json["status"] = result.status;
        cases_json.push_back(case_json);
    }
    output["cases"] = cases_json;

    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create output file " << output_path << std::endl;
        return;
    }

    file << output.dump(4);
    file.close();

    std::cout << "Results saved to: " << output_path << std::endl;
}

int main(int argc, char* argv[]) {
    std::string json_file;
    std::string result_dir = "result";

    if (argc > 1) {
        json_file = argv[1];
    } else {
        json_file = "Hybirdcvdv/circuit/sc26_scaling.json";
    }

    if (argc > 2) {
        result_dir = argv[2];
    }

    std::cout << "Loading JSON file: " << json_file << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    auto cases = parse_json_file(json_file);

    std::cout << "Total cases: " << cases.size() << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::sort(cases.begin(), cases.end(), [](const CircuitCase& a, const CircuitCase& b) {
        if (a.cutoff != b.cutoff) return a.cutoff < b.cutoff;
        if (a.num_modes != b.num_modes) return a.num_modes < b.num_modes;
        return a.num_qubits < b.num_qubits;
    });

    std::cout << "Cases sorted by cutoff, then num_modes, then num_qubits" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::vector<CircuitResult> results;
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::vector<CircuitCase>> thread_cases(num_threads);

    // 分配任务到各个线程
    for (size_t i = 0; i < cases.size(); ++i) {
        thread_cases[i % num_threads].push_back(cases[i]);
    }

    // 启动线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            std::vector<CircuitResult> thread_results;
            for (const auto& c : thread_cases[i]) {
                auto result = run_circuit(c, result_dir);
                thread_results.push_back(result);
            }
            
            // 合并结果
            std::lock_guard<std::mutex> lock(results_mutex);
            results.insert(results.end(), thread_results.begin(), thread_results.end());
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    fs::create_directories(result_dir);

    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << result_dir << "/sc26_results_" << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S") << ".json";
    std::string output_path = ss.str();

    save_results_to_json(results, output_path);

    std::cout << "========================================" << std::endl;
    std::cout << "All circuits executed!" << std::endl;
    std::cout << "Results saved to: " << output_path << std::endl;

    return 0;
}
