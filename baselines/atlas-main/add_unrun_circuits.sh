#!/bin/bash

# 定义路径
QASM_DIR="/root/Hybirdcvdv/baselines/atlas-main/qasm"
RESULTS_FILE="/root/Hybirdcvdv/baselines/atlas-main/result/atlas_results.csv"

# 创建临时文件
temp_qasm="$(mktemp)"
temp_results="$(mktemp)"

# 提取 qasm 目录中的电路名称
echo "Extracting circuit names from qasm directory..."
for qasm_file in "${QASM_DIR}"/*.qasm; do
    circuit_name=$(basename "${qasm_file}" .qasm)
    echo "${circuit_name}" >> "${temp_qasm}"
done

# 提取 atlas_results.csv 中的电路名称
echo "Extracting circuit names from results file..."
if [ -f "${RESULTS_FILE}" ]; then
    # 跳过表头，提取第一列
    tail -n +2 "${RESULTS_FILE}" | cut -d ',' -f 1 >> "${temp_results}"
else
    echo "Results file not found!"
    exit 1
fi

# 找出未运行的电路
echo "Finding unrun circuits..."
unrun_circuits=$(comm -23 <(sort "${temp_qasm}") <(sort "${temp_results}"))

# 为未运行的电路添加记录到 atlas_results.csv
echo "Adding unrun circuits to atlas_results.csv..."
if [ -n "${unrun_circuits}" ]; then
    echo "Adding ${unrun_circuits} circuits..."
    echo "${unrun_circuits}" | while read -r circuit_name; do
        # 推断电路类型
        if [[ "${circuit_name}" == *"cat"* ]]; then
            circuit_type="cat"
        elif [[ "${circuit_name}" == *"gkp"* ]]; then
            circuit_type="gkp"
        elif [[ "${circuit_name}" == *"jch"* ]]; then
            circuit_type="jch"
        elif [[ "${circuit_name}" == *"qaoa"* ]]; then
            circuit_type="qaoa"
        elif [[ "${circuit_name}" == *"qft"* ]]; then
            circuit_type="qft"
        elif [[ "${circuit_name}" == *"shors"* ]]; then
            circuit_type="shors"
        elif [[ "${circuit_name}" == *"transfer"* ]]; then
            circuit_type="transfer"
        elif [[ "${circuit_name}" == *"vqe"* ]]; then
            circuit_type="vqe"
        else
            circuit_type="unknown"
        fi
        
        # 添加记录到文件
        echo "${circuit_name},${circuit_type},0,0,0,0,0,nlocal 超出可用范围" >> "${RESULTS_FILE}"
    done
    echo "Added ${unrun_circuits} circuits to atlas_results.csv"
else
    echo "All circuits have been run!"
fi

# 清理临时文件
rm "${temp_qasm}" "${temp_results}"
