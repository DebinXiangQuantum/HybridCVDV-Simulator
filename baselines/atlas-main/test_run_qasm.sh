#!/bin/bash

# 测试脚本用于运行存在的 QASM 文件

# 定义变量
QASM_DIR="/root/Hybirdcvdv/baselines/atlas-main/qasm"
BUILD_DIR="/root/Hybirdcvdv/baselines/atlas-main/build"
RUN_PROGRAM="${BUILD_DIR}/examples/mpi-based/run_generated_qasm"

# 检查 run_generated_qasm 是否存在
if [ ! -f "${RUN_PROGRAM}" ]; then
    echo "Error: run_generated_qasm not found at ${RUN_PROGRAM}"
    exit 1
 fi

# 检查 qasm 目录是否存在
if [ ! -d "${QASM_DIR}" ]; then
    echo "Error: qasm directory not found at ${QASM_DIR}"
    exit 1
 fi

# 测试存在的 QASM 文件
test_circuits=("sc26_vqe_nq3_nm3_c4" "sc26_vqe_nq3_nm3_c8" "sc26_vqe_nq3_nm3_c16")

# 遍历测试电路
for circuit_name in "${test_circuits[@]}"; do
    echo "Running circuit: ${circuit_name}"
    
    # 检查 QASM 文件是否存在
    if [ ! -f "${QASM_DIR}/${circuit_name}.qasm" ]; then
        echo "Warning: QASM file ${QASM_DIR}/${circuit_name}.qasm not found"
        echo "--------------------------------------------------"
        continue
    fi
    
    # 运行 run_generated_qasm 程序（在 build 目录中运行，以确保相对路径正确）
    # 使用 nlocal 小于 nqubits，避免 num_global_qubits 为负数
    cd "${BUILD_DIR}" && mpirun -n 1 --allow-run-as-root ./examples/mpi-based/run_generated_qasm --import-circuit "${circuit_name}" --n 16 --local 8 --device 0 && cd "${QASM_DIR}/.."
    
    # 检查运行是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run circuit ${circuit_name}"
    else
        echo "Successfully ran circuit ${circuit_name}"
    fi
    
    echo "--------------------------------------------------"
done

echo "All test circuits have been processed."
