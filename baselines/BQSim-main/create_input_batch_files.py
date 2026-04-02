#!/usr/bin/env python3
import os
import re
import math

# 定义目录路径
qasm_dir = "./qasm"
input_batch_dir = "./input_batch"

# 确保 input_batch 目录存在
os.makedirs(input_batch_dir, exist_ok=True)

# 正则表达式匹配文件名中的量子比特数
qubit_pattern = re.compile(r'nq(\d+)')

# 存储已处理的量子比特数，避免重复创建文件
processed_qubits = set()

# 遍历 qasm 目录下的所有文件
for filename in os.listdir(qasm_dir):
    if filename.endswith('.qasm'):
        # 提取量子比特数
        match = qubit_pattern.search(filename)
        if match:
            num_qubits = int(match.group(1))
            # 检查是否已经处理过
            if num_qubits not in processed_qubits:
                # 计算量子态维度
                n_dim = 2 ** num_qubits
                # 生成对应的 n?.txt 文件
                output_file = os.path.join(input_batch_dir, f"n{num_qubits}.txt")
                # 写入内容：生成 n_dim 个复数，第一个为 1.0 0.0，其余为 0.0 0.0
                with open(output_file, 'w') as f:
                    for i in range(n_dim):
                        if i == 0:
                            f.write("1.0 0.0\n")  # 初始态 |000...0>
                        else:
                            f.write("0.0 0.0\n")
                print(f"Created {output_file} with {n_dim} entries")
                processed_qubits.add(num_qubits)

print("\nAll input batch files created successfully!")
print(f"Processed {len(processed_qubits)} different qubit counts")
