#!/bin/bash

# 脚本用于运行所有 QASM 文件

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

# 遍历 qasm 目录中的所有 QASM 文件
for qasm_file in "${QASM_DIR}"/*.qasm; do
    # 提取文件名（不包括路径和 .qasm 后缀）
    circuit_name=$(basename "${qasm_file}" .qasm)
    
    echo "Running circuit: ${circuit_name}"
    
    # 从 QASM 文件中提取量子比特数
    # 查找 qreg q[num]; 行
    qubits_line=$(grep "qreg q\[" "${qasm_file}" | head -1)
    if [ -n "${qubits_line}" ]; then
        # 提取数字部分
        num_qubits=$(echo "${qubits_line}" | sed 's/[^0-9]*//g')
        # 确保 num_qubits 是一个正整数
        if [[ "${num_qubits}" =~ ^[0-9]+$ && ${num_qubits} -gt 0 ]]; then
            # 计算 nlocal，确保 nlocal < num_qubits
            nlocal=$((num_qubits > 1 ? num_qubits / 2 : 1))
            echo "Detected ${num_qubits} qubits, using --n ${num_qubits} --local ${nlocal}"
        else
            # 默认值
            num_qubits=16
            nlocal=8
            echo "Could not detect qubit count, using default --n ${num_qubits} --local ${nlocal}"
        fi
    else
        # 默认值
        num_qubits=16
        nlocal=8
        echo "Could not detect qubit count, using default --n ${num_qubits} --local ${nlocal}"
    fi
    
    # 运行 run_generated_qasm 程序
    # 在 build 目录中运行，以确保相对路径正确
    cd "${BUILD_DIR}" && mpirun -n 1 --allow-run-as-root ./examples/mpi-based/run_generated_qasm --import-circuit "${circuit_name}" --n ${num_qubits} --local ${nlocal} --device 0 && cd "${QASM_DIR}/.."
    
    # 检查运行是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run circuit ${circuit_name}"
    else
        echo "Successfully ran circuit ${circuit_name}"
    fi
    
    echo "--------------------------------------------------"
done

echo "All circuits have been processed."
