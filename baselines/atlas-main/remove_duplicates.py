#!/usr/bin/env python3
"""
删除 atlas_results.csv 中的重复数据
"""

import csv
import os

# 定义文件路径
csv_file = '/root/Hybirdcvdv/baselines/atlas-main/result/atlas_results.csv'

# 读取文件并去重
seen = set()
header = None
rows = []

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # 读取表头
    for row in reader:
        circuit_name = row[0]  # 电路名是第一列
        if circuit_name not in seen:
            seen.add(circuit_name)
            rows.append(row)

# 写回文件
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"去重完成，删除了 {len(seen) - len(rows)} 条重复记录")
print(f"现在文件中有 {len(rows)} 条记录")
