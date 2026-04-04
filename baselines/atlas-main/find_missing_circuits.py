import os
import csv

# 读取已运行的电路
results_file = '/root/Hybirdcvdv/baselines/atlas-main/result/atlas_results.csv'
running_circuits = set()

with open(results_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        circuit_name = row[0]
        running_circuits.add(circuit_name)

# 读取qasm目录下的所有电路
qasm_dir = '/root/Hybirdcvdv/baselines/atlas-main/qasm'
all_circuits = set()

for file in os.listdir(qasm_dir):
    if file.endswith('.qasm'):
        circuit_name = file[:-5]  # 去掉.qasm后缀
        all_circuits.add(circuit_name)

# 找出未运行的电路
missing_circuits = all_circuits - running_circuits

# 输出结果
print(f"总共的电路数量: {len(all_circuits)}")
print(f"已运行的电路数量: {len(running_circuits)}")
print(f"未运行的电路数量: {len(missing_circuits)}")
print("\n未运行的电路列表:")
for circuit in sorted(missing_circuits):
    print(circuit)

# 将未运行的电路保存到文件
missing_file = '/root/Hybirdcvdv/baselines/atlas-main/build/missing_circuits.txt'
with open(missing_file, 'w', encoding='utf-8') as f:
    for circuit in sorted(missing_circuits):
        f.write(circuit + '\n')

print(f"\n未运行的电路已保存到: {missing_file}")
