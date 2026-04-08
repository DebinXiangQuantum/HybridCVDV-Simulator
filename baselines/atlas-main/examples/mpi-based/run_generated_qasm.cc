#include <string.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <sys/resource.h>

#include "circuit.h"
#include "quartz/context/param_info.h"
#include "performance_monitor_global.h"

using quartz::GateType;

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

  // 初始化全局性能监控器
  sim::GlobalPerfMonitor::init(ndevice);
  auto* perf_monitor = sim::GlobalPerfMonitor::get();

  // 开始总时间测量
  perf_monitor->start_timer("total");

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
  
  // 在非 ILP 模式下，使用实际的量子比特数作为本地量子比特数，确保所有门都能被处理
  if (!use_ilp) {
    printf("Using non-ILP mode, setting num_local_qubits to actual_num_qubits (%d)\n", actual_num_qubits);
    nlocal = actual_num_qubits;
  }

  sim::qcircuit::Circuit<double> circuit(actual_num_qubits, nlocal, ndevice, myRank, nRanks);

  // 开始编译时间测量
  perf_monitor->start_timer("compile");
  circuit.compile(seq.get(), &ctx, &interpreter, use_ilp, "../schedules/" +
                                      circuit_file +
                                      std::to_string(nqubits) + "_" +
                                      std::to_string(nlocal));
  perf_monitor->stop_timer("compile");
  perf_monitor->update_memory_peak();

  // 开始模拟时间测量
  perf_monitor->start_timer("simulate");
  circuit.simulate(true);
  perf_monitor->stop_timer("simulate");
  perf_monitor->update_memory_peak();

  // 结束总时间测量
  perf_monitor->stop_timer("total");

  // 收集性能指标
  sim::PerformanceMetrics metrics = perf_monitor->get_metrics();
  metrics.total_time = perf_monitor->get_elapsed_time("total");
  metrics.compile_time = perf_monitor->get_elapsed_time("compile");
  metrics.simulate_time = perf_monitor->get_elapsed_time("simulate");
  metrics.num_gates = gate_count;
  metrics.num_qubits = actual_num_qubits;
  metrics.num_devices = ndevice;

  // 注意：h2d_transfer_time, d2h_transfer_time, d2d_transfer_time 已在 simulator 中记录
  // 这些值会通过全局监控器自动累积

  perf_monitor->set_metrics(metrics);

  // 提取电路类型
  std::string circuit_type = extract_circuit_type(circuit_file);

  // 释放动态分配的 ParamInfo 对象
  delete param_info;

  // 保存结果到文件（只在 rank 0 执行）
  if (myRank == 0) {
    std::string result_dir = "../result";
    // 创建结果目录
    system(("mkdir -p " + result_dir).c_str());
    // 打印详细报告
    perf_monitor->print_report(circuit_file);
    
    // 保存原始格式的 CSV 文件
    std::string csv_file = result_dir + "/atlas_results.csv";
    std::ifstream check_file(csv_file);
    bool file_exists = check_file.good();
    check_file.close();
    
    std::ofstream outfile;
    if (!file_exists) {
      outfile.open(csv_file);
      outfile << "电路名,电路类型,总时间,编译时间,模拟时间,CPU内存峰值,GPU显存峰值,GPU功耗峰值,GPU平均功耗,GPU利用率峰值,GPU平均利用率,门数,量子比特数\n";
    } else {
      outfile.open(csv_file, std::ios_base::app);
    }
    
    outfile << circuit_file << ","
            << circuit_type << ","
            << std::fixed << std::setprecision(6)
            << metrics.total_time * 1000 << ","    // ms
            << metrics.compile_time * 1000 << ","  // ms
            << metrics.simulate_time * 1000 << "," // ms
            << std::setprecision(2)
            << static_cast<double>(metrics.cpu_memory_peak) << ","  // bytes
            << static_cast<double>(metrics.gpu_memory_peak) << ","  // bytes
            << std::setprecision(1)
            << metrics.gpu_power_peak_w << ","           // W
            << metrics.gpu_power_avg_w << ","            // W
            << metrics.gpu_utilization_peak_pct << ","   // %
            << std::setprecision(1)
            << metrics.gpu_utilization_avg_pct << ","    // %
            << metrics.num_gates << ","
            << metrics.num_qubits << "\n";
    
    outfile.close();

  }

  // 清理全局监控器
  sim::GlobalPerfMonitor::cleanup();

  MPICHECK(MPI_Finalize());

  return 0;
}