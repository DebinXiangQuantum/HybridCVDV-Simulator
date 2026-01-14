import json
import matplotlib.pyplot as plt
import numpy as np
import os



# Read performance data
def read_performance_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {result['gate_name']: result for result in data['results']}

# Get common gates from both datasets
def get_common_gates(cpu_data, gpu_data):
    return sorted(set(cpu_data.keys()) & set(gpu_data.keys()))

# Plot performance metrics comparison
def plot_performance_comparison(cpu_data, gpu_data, common_gates, metric, title, ylabel, filename, log_scale=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(common_gates))
    width = 0.35
    
    # 获取CPU和GPU的指标值
    cpu_values = [cpu_data[gate][metric] for gate in common_gates]
    gpu_values = [gpu_data[gate][metric] for gate in common_gates]
    
    # 创建条形图
    bars_cpu = ax.bar(x - width/2, cpu_values, width, label='CPU', color='#1f77b4')  # 蓝色
    bars_gpu = ax.bar(x + width/2, gpu_values, width, label='GPU', color='#ffd700')  # 黄色
    
    # 设置图表属性
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(common_gates, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    
    if log_scale:
        ax.set_yscale('log')
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_labels(bars_cpu)
    add_labels(bars_gpu)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {filename}")

# 主函数
def main():
    # Read data files
    gpu_file = 'gpu.json'
    cpu_file = 'cpu.json'
    
    gpu_data = read_performance_data(gpu_file)
    cpu_data = read_performance_data(cpu_file)
    
    # 获取共有门名
    common_gates = get_common_gates(cpu_data, gpu_data)
    
    print(f"Found {len(common_gates)} common gates in both datasets")
    print("Gate names: ", common_gates)
    
    # Plot performance metrics comparison
    metrics = [
        ('single_gate_latency_ms', 'Single Gate Latency Comparison (ms)', 'Latency (ms)', 'latency_comparison.png'),
        ('batch_throughput', 'Batch Throughput Comparison (tasks/s)', 'Throughput (tasks/s)', 'throughput_comparison.png'),
        ('memory_efficiency', 'Memory Efficiency Comparison', 'Memory Efficiency', 'memory_efficiency_comparison.png', False)
    ]
    
    # 输出目录
    output_dir = '/home/zhenyusen/HybridCVDV-Simulator/baselines/compare_fig'
    
    for metric, title, ylabel, filename, *log_scale in metrics:
        log = log_scale[0] if log_scale else True
        full_filename = os.path.join(output_dir, filename)
        plot_performance_comparison(cpu_data, gpu_data, common_gates, metric, title, ylabel, full_filename, log)
    
    print("All charts generated successfully!")

if __name__ == "__main__":
    main()
