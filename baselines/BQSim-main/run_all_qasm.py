#!/usr/bin/env python3
"""
运行 qasm 目录下的所有 QASM 文件，并检查是否已经运行过
"""

import os
import subprocess
import csv
import re

QASM_DIR = os.path.join(os.path.dirname(__file__), 'qasm')
RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'log', 'results', 'bqsim_results.csv')
BQSIM_EXEC = os.path.join(os.path.dirname(__file__), 'build', 'apps', 'BQSim')

def get_circuit_name(qasm_file):
    """从 QASM 文件名中提取电路名"""
    filename = os.path.basename(qasm_file)
    return os.path.splitext(filename)[0]

def get_circuit_type(circuit_name):
    """根据电路名判断电路类型"""
    if 'vqe' in circuit_name:
        return 'vqe'
    elif 'jch' in circuit_name:
        return 'jch'
    elif 'cat' in circuit_name:
        return 'cat'
    elif 'gkp' in circuit_name:
        return 'gkp'
    elif 'qaoa' in circuit_name:
        return 'qaoa'
    elif 'qft' in circuit_name:
        return 'qft'
    elif 'shors' in circuit_name:
        return 'shors'
    elif 'transfer' in circuit_name:
        return 'transfer'
    else:
        return 'quantum'

def load_existing_results():
    """加载已有的运行结果"""
    existing_circuits = set()
    
    if not os.path.exists(RESULTS_CSV):
        return existing_circuits
    
    try:
        with open(RESULTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_circuits.add(row['电路名'])
    except Exception as e:
        print(f"Error loading existing results: {e}")
    
    return existing_circuits

def run_bqsim(qasm_file):
    """运行 BQSim 并返回结果"""
    cmd = [
        BQSIM_EXEC,
        '--batch_size', '1',
        '--num_batch', '1',
        '--conversion_type', '0',
        '--file', qasm_file
    ]
    
    print(f"Running BQSim on {os.path.basename(qasm_file)}...")
    
    try:
        # 使用绝对路径，不设置 cwd
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running BQSim: {result.stderr}")
            return None
        
        # 解析输出结果
        output = result.stdout
        
        # 提取统计信息
        stats = {
            '总时间': 0,
            '传输时间': 0,
            '计算时间': 0,
            '内存占用': 0,
            '门数': 0
        }
        
        # 提取总时间
        total_time_match = re.search(r'time: (\d+)', output)
        if total_time_match:
            stats['总时间'] = total_time_match.group(1)
        
        # 提取传输时间
        transfer_time_match = re.search(r'传输时间: ([\d.e-]+) ms', output)
        if transfer_time_match:
            stats['传输时间'] = transfer_time_match.group(1)
        
        # 提取计算时间
        computation_time_match = re.search(r'计算时间: ([\d.e-]+) ms', output)
        if computation_time_match:
            stats['计算时间'] = computation_time_match.group(1)
        
        # 提取内存占用
        memory_match = re.search(r'峰值内存使用: (\d+) 字节', output)
        if memory_match:
            stats['内存占用'] = memory_match.group(1)
        
        # 提取门数
        gates_match = re.search(r'applied_gates": (\d+)', output)
        if gates_match:
            stats['门数'] = gates_match.group(1)
        
        return stats
        
    except Exception as e:
        print(f"Exception running BQSim: {e}")
        return None

def write_result(circuit_name, circuit_type, stats):
    """将结果写入 CSV 文件"""
    row = {
        '电路名': circuit_name,
        '电路类型': circuit_type,
        '总时间': stats.get('总时间', 0),
        '传输时间': stats.get('传输时间', 0),
        '计算时间': stats.get('计算时间', 0),
        '内存占用': stats.get('内存占用', 0),
        '门数': stats.get('门数', 0)
    }
    
    # 检查文件是否存在
    file_exists = os.path.exists(RESULTS_CSV)
    
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
    
    print(f"Result written for {circuit_name}")

def main():
    # 确保 BQSim 可执行文件存在
    if not os.path.exists(BQSIM_EXEC):
        print(f"BQSim executable not found at {BQSIM_EXEC}")
        print("Please build BQSim first.")
        return 1
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    
    # 加载已有的运行结果
    existing_circuits = load_existing_results()
    print(f"Loaded {len(existing_circuits)} existing results")
    
    # 遍历 qasm 目录下的所有 QASM 文件
    qasm_files = [f for f in os.listdir(QASM_DIR) if f.endswith('.qasm')]
    print(f"Found {len(qasm_files)} QASM files")
    
    # 统计变量
    total_files = len(qasm_files)
    skipped_files = 0
    run_files = 0
    error_files = 0
    
    for qasm_file in qasm_files:
        qasm_path = os.path.join(QASM_DIR, qasm_file)
        circuit_name = get_circuit_name(qasm_path)
        
        # 检查是否已经运行过
        if circuit_name in existing_circuits:
            print(f"[SKIP] {qasm_file} (already run)")
            skipped_files += 1
            continue
        
        # 运行 BQSim
        stats = run_bqsim(qasm_path)
        
        if stats:
            circuit_type = get_circuit_type(circuit_name)
            write_result(circuit_name, circuit_type, stats)
            # 更新已运行的电路集合，避免重复运行
            existing_circuits.add(circuit_name)
            run_files += 1
        else:
            print(f"[ERROR] Failed to run {qasm_file}")
            error_files += 1
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print(f"Total files: {total_files}")
    print(f"Skipped (already run): {skipped_files}")
    print(f"Run successfully: {run_files}")
    print(f"Run failed: {error_files}")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    exit(main())
