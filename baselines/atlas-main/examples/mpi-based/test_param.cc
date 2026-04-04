#include <string.h>

#include "circuit.h"

using quartz::GateType;

int main(int argc, char *argv[]) {

  MPICHECK(MPI_Init(&argc, &argv));

  int myRank, nRanks;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  printf("Num ranks: %d, myrank: %d\n", nRanks, myRank);

  std::string circuit_file;
  unsigned nqubits;
  unsigned nlocal;
  int ndevice;
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

  quartz::init_python_interpreter();
  quartz::PythonInterpreter interpreter;
  
  // 使用动态分配的 ParamInfo 对象
  quartz::ParamInfo *param_info = new quartz::ParamInfo();
  
  quartz::Context ctx({GateType::input_qubit, GateType::input_param,
                       GateType::h, GateType::x, GateType::ry, GateType::u2,
                       GateType::u3, GateType::cx, GateType::cz, GateType::cp,
                       GateType::p, GateType::z, GateType::rz, GateType::swap}, param_info);
  
  // 加载 QASM 文件
  auto seq = quartz::CircuitSeq::from_qasm_file(
      &ctx, std::string("../qasm/") + circuit_file + ".qasm");
  
  // 打印参数数量和值
  printf("Number of parameters: %d\n", param_info->get_num_parameters());
  for (int i = 0; i < param_info->get_num_parameters(); i++) {
    printf("Parameter %d: %f\n", i, param_info->get_param_value(i));
  }
  
  // 编译和模拟电路
  sim::qcircuit::Circuit<double> circuit(nqubits, nlocal, ndevice, myRank, nRanks);
  circuit.compile(seq.get(), &ctx, &interpreter, use_ilp, "../schedules/" +
                                      circuit_file +
                                      std::to_string(nqubits) + "_" +
                                      std::to_string(nlocal));
  circuit.simulate(true);

  MPICHECK(MPI_Finalize());

  // 释放动态分配的 ParamInfo 对象
  delete param_info;

  return 0;
}