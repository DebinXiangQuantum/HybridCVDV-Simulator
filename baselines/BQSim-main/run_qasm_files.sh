#!/bin/bash

# 定义 BQSim 可执行文件路径
BQSIM_EXEC="/root/Hybirdcvdv/baselines/BQSim-main/build/apps/BQSim"

# 定义 QASM 文件目录
QASM_DIR="/root/Hybirdcvdv/baselines/bosonic-qiskit-implementation/qasm_output"

# 定义输出目录
OUTPUT_DIR="/root/Hybirdcvdv/baselines/BQSim-main/results"
mkdir -p $OUTPUT_DIR

# 遍历目录中的所有 .qasm 文件
for qasm_file in "$QASM_DIR"/*.qasm; do
    # 获取文件名（不含路径）
    filename=$(basename "$qasm_file")
    # 获取文件名（不含扩展名）
    filename_no_ext=$(basename "$qasm_file" .qasm)
    
    # 定义输出文件路径
    output_file="$OUTPUT_DIR/${filename_no_ext}_result.txt"
    
    echo "Running BQSim on $filename..."
    
    # 运行 BQSim 并保存输出
    $BQSIM_EXEC --file "$qasm_file" --batch_size 1 --num_batch 1 --conversion_type 0 --ps > "$output_file"
    
    echo "Result saved to $output_file"
    echo "----------------------------------"
done

echo "All QASM files have been processed."
