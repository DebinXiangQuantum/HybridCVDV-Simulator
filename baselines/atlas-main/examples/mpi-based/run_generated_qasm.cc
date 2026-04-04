#include <string.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <sys/resource.h>

#include "circuit.h"
#include "quartz/context/param_info.h"

using quartz::GateType;

// 测量内存使用
long get_memory_usage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss; // 返回KB
}

// 从文件名提取电路类型
std::string extract_circuit_type(const std::string& filename) {
  size_t underscore_pos = filename.find('_');
  if (underscore_pos != std::string::npos) {
    std::string type_part = filename.substr(0, underscore_pos);
    if (type_part == "sc26") {
      size_t next_underscore = filename.find('_', underscore_pos + 1);
      if (next_underscore != std::string::npos) {
        return filename.substr(underscore_pos + 1, next_underscore - underscore_pos - 1);
      }
    }
  }
  return "unknown";
}

int main(int argc, char *argv[]) {

  MPICHECK(MPI_Init(&argc, &argv));

  int myRank, nRanks;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  printf("Num ranks: %d, myrank: %d\n", nRanks, myRank);

  std::string circuit_file;
  unsigned nqubits = 2; // 设置默认值为2
  unsigned nlocal = 2; // 设置默认值为2
  int ndevice = 1; // 设置默认值为1
  bool use_ilp = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--import-circuit")) {
      circuit_file = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--n")) {
      nqubits = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--local")) {
      nlocal = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--device")) {
      ndevice = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--use-ilp"))) {
      use_ilp = true;
      continue;
    }
  }

  // 开始总时间测量
  auto start_total = std::chrono::high_resolution_clock::now();

  quartz::init_python_interpreter();
  quartz::PythonInterpreter interpreter;
  // 使用动态分配的 ParamInfo 对象，确保它在整个程序运行期间都存在
  quartz::ParamInfo *param_info = new quartz::ParamInfo();
  quartz::Context ctx({GateType::input_qubit, GateType::input_param,
                       GateType::h, GateType::x, GateType::ry, GateType::u2,
                       GateType::u3, GateType::cx, GateType::cz, GateType::cp,
                       GateType::p, GateType::z, GateType::rz, GateType::swap}, param_info);
  // 加载 generate_qasm.py 生成的 QASM 文件
  auto seq = quartz::CircuitSeq::from_qasm_file(
      &ctx, std::string("../qasm/") + circuit_file + ".qasm");
  
  // 检查 QASM 文件是否成功加载
  if (!seq) {
    std::cerr << "Error: Failed to load QASM file: ../qasm/" << circuit_file << ".qasm" << std::endl;
    delete param_info;
    MPICHECK(MPI_Finalize());
    return 1;
  }
  
  // 统计门数
  int gate_count = seq->get_num_gates();
  
  // 使用从QASM文件中读取的实际量子比特数
  int actual_num_qubits = seq->get_num_qubits();
  printf("Actual number of qubits in circuit: %d\n", actual_num_qubits);
  
  // 确保num_local_qubits不超过实际的量子比特数
  if (nlocal > actual_num_qubits) {
    printf("Warning: num_local_qubits (%d) > actual_num_qubits (%d), setting num_local_qubits to actual_num_qubits\n", nlocal, actual_num_qubits);
    nlocal = actual_num_qubits;
  }
  
  sim::qcircuit::Circuit<double> circuit(actual_num_qubits, nlocal, ndevice, myRank, nRanks);
  
  // 开始编译时间测量
  auto start_compile = std::chrono::high_resolution_clock::now();
  circuit.compile(seq.get(), &ctx, &interpreter, use_ilp, "../schedules/" +
                                      circuit_file +
                                      std::to_string(nqubits) + "_" +
                                      std::to_string(nlocal));
  auto end_compile = std::chrono::high_resolution_clock::now();
  
  // 开始模拟时间测量
  auto start_simulate = std::chrono::high_resolution_clock::now();
  circuit.simulate(true);
  auto end_simulate = std::chrono::high_resolution_clock::now();
  
  // 结束总时间测量
  auto end_total = std::chrono::high_resolution_clock::now();
  
  // 计算各时间
  double total_time = std::chrono::duration<double>(end_total - start_total).count();
  double compile_time = std::chrono::duration<double>(end_compile - start_compile).count();
  double simulate_time = std::chrono::duration<double>(end_simulate - start_simulate).count();
  double compute_time = compile_time + simulate_time;
  double transfer_time = total_time - compute_time; // 传输时间 = 总时间 - 计算时间
  
  // 测量内存占用
  long memory_usage = get_memory_usage();
  
  // 提取电路类型
  std::string circuit_type = extract_circuit_type(circuit_file);
  
  // 释放动态分配的 ParamInfo 对象
  delete param_info;

  // 保存结果到文件（只在 rank 0 执行）
  if (myRank == 0) {
    std::string result_dir = "../result";
    std::string result_file = result_dir + "/atlas_results.csv";
    
    // 创建结果目录
    system(("mkdir -p " + result_dir).c_str());
    
    // 检查文件是否存在，不存在则创建并写入表头
    std::ifstream check_file(result_file);
    bool file_exists = check_file.good();
    check_file.close();
    
    std::ofstream outfile;
    if (!file_exists) {
      outfile.open(result_file);
      outfile << "电路名,电路类型,总时间,传输时间,计算时间,内存占用,门数\n";
    } else {
      outfile.open(result_file, std::ios_base::app);
    }
    
    // 写入结果
    outfile << circuit_file << "," 
            << circuit_type << "," 
            << std::fixed << std::setprecision(6) << total_time << "," 
            << std::fixed << std::setprecision(6) << transfer_time << "," 
            << std::fixed << std::setprecision(6) << compute_time << "," 
            << memory_usage << "," 
            << gate_count << "\n";
    
    outfile.close();
    
    // 打印结果
    printf("\n=== 运行结果 ===\n");
    printf("电路名: %s\n", circuit_file.c_str());
    printf("电路类型: %s\n", circuit_type.c_str());
    printf("总时间: %.6f 秒\n", total_time);
    printf("传输时间: %.6f 秒\n", transfer_time);
    printf("计算时间: %.6f 秒\n", compute_time);
    printf("内存占用: %ld KB\n", memory_usage);
    printf("门数: %d\n", gate_count);
    printf("结果已保存到: %s\n", result_file.c_str());
  }

  MPICHECK(MPI_Finalize());

  return 0;
}
