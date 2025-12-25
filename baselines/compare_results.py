#!/usr/bin/env python3
"""
性能对比脚本
比较 HybridCVDV-Simulator 和 Strawberry Fields 的性能和精度
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys


class ResultComparator:
    """结果对比器"""
    
    def __init__(self, sf_results_file: str, hybrid_results_file: str):
        """
        初始化对比器
        
        Args:
            sf_results_file: Strawberry Fields 结果文件路径
            hybrid_results_file: HybridCVDV-Simulator 结果文件路径
        """
        with open(sf_results_file, 'r') as f:
            self.sf_results = json.load(f)
        
        with open(hybrid_results_file, 'r') as f:
            self.hybrid_results = json.load(f)
        
        self.comparison_results = {
            "comparisons": [],
            "summary": {}
        }
    
    def calculate_fidelity(self, state1: Dict, state2: Dict) -> float:
        """
        计算两个量子态的保真度
        
        Fidelity = |⟨ψ₁|ψ₂⟩|²
        
        Args:
            state1: 第一个状态向量 {"real": [...], "imag": [...]}
            state2: 第二个状态向量
            
        Returns:
            保真度 (0到1之间)
        """
        # 构建复数向量
        psi1 = np.array(state1["real"]) + 1j * np.array(state1["imag"])
        psi2 = np.array(state2["real"]) + 1j * np.array(state2["imag"])
        
        # 确保长度相同
        min_len = min(len(psi1), len(psi2))
        psi1 = psi1[:min_len]
        psi2 = psi2[:min_len]
        
        # 归一化
        psi1 = psi1 / np.linalg.norm(psi1)
        psi2 = psi2 / np.linalg.norm(psi2)
        
        # 计算内积
        overlap = np.abs(np.vdot(psi1, psi2))
        fidelity = overlap ** 2
        
        return float(fidelity)
    
    def calculate_trace_distance(self, state1: Dict, state2: Dict) -> float:
        """
        计算迹距离
        
        对于纯态: D(ρ₁, ρ₂) = √(1 - F(ρ₁, ρ₂))
        
        Args:
            state1: 第一个状态向量
            state2: 第二个状态向量
            
        Returns:
            迹距离
        """
        fidelity = self.calculate_fidelity(state1, state2)
        trace_distance = np.sqrt(1 - fidelity)
        return float(trace_distance)
    
    def calculate_l2_error(self, state1: Dict, state2: Dict) -> float:
        """
        计算L2误差
        
        Args:
            state1: 第一个状态向量
            state2: 第二个状态向量
            
        Returns:
            L2误差
        """
        psi1 = np.array(state1["real"]) + 1j * np.array(state1["imag"])
        psi2 = np.array(state2["real"]) + 1j * np.array(state2["imag"])
        
        min_len = min(len(psi1), len(psi2))
        psi1 = psi1[:min_len]
        psi2 = psi2[:min_len]
        
        diff = psi1 - psi2
        l2_error = np.linalg.norm(diff)
        
        return float(l2_error)
    
    def compare_test(self, sf_test: Dict, hybrid_test: Dict) -> Dict:
        """
        比较单个测试的结果
        
        Args:
            sf_test: Strawberry Fields 测试结果
            hybrid_test: HybridCVDV-Simulator 测试结果
            
        Returns:
            对比结果字典
        """
        test_name = sf_test["test_name"]
        print(f"\n比较测试: {test_name}")
        
        # 性能对比
        sf_time = sf_test["avg_time_ms"]
        hybrid_time = hybrid_test["avg_time_ms"]
        speedup = sf_time / hybrid_time if hybrid_time > 0 else float('inf')
        
        print(f"  Strawberry Fields 平均时间: {sf_time:.4f} ms")
        print(f"  HybridCVDV-Simulator 平均时间: {hybrid_time:.4f} ms")
        print(f"  加速比: {speedup:.2f}x")
        
        # 精度对比
        if "state_vector" in sf_test and "state_vector" in hybrid_test:
            fidelity = self.calculate_fidelity(
                sf_test["state_vector"], 
                hybrid_test["state_vector"]
            )
            trace_distance = self.calculate_trace_distance(
                sf_test["state_vector"],
                hybrid_test["state_vector"]
            )
            l2_error = self.calculate_l2_error(
                sf_test["state_vector"],
                hybrid_test["state_vector"]
            )
            
            print(f"  保真度: {fidelity:.10f}")
            print(f"  迹距离: {trace_distance:.10e}")
            print(f"  L2误差: {l2_error:.10e}")
        else:
            fidelity = None
            trace_distance = None
            l2_error = None
            print("  警告: 无法比较精度（缺少状态向量）")
        
        comparison = {
            "test_name": test_name,
            "gate_type": sf_test["gate_type"],
            "parameters": sf_test["parameters"],
            "performance": {
                "sf_avg_time_ms": sf_time,
                "sf_std_time_ms": sf_test["std_time_ms"],
                "hybrid_avg_time_ms": hybrid_time,
                "hybrid_std_time_ms": hybrid_test["std_time_ms"],
                "speedup": speedup
            },
            "accuracy": {
                "fidelity": fidelity,
                "trace_distance": trace_distance,
                "l2_error": l2_error
            }
        }
        
        return comparison
    
    def compare_all(self):
        """比较所有测试"""
        print("=" * 60)
        print("性能和精度对比分析")
        print("=" * 60)
        
        sf_tests = {test["test_name"]: test for test in self.sf_results["tests"]}
        hybrid_tests = {test["test_name"]: test for test in self.hybrid_results["tests"]}
        
        # 找到共同的测试
        common_tests = set(sf_tests.keys()) & set(hybrid_tests.keys())
        
        if not common_tests:
            print("错误: 没有找到共同的测试项目！")
            return
        
        print(f"\n找到 {len(common_tests)} 个共同测试项目")
        
        for test_name in sorted(common_tests):
            comparison = self.compare_test(sf_tests[test_name], hybrid_tests[test_name])
            self.comparison_results["comparisons"].append(comparison)
        
        # 计算总体统计
        self._calculate_summary()
    
    def _calculate_summary(self):
        """计算总体统计信息"""
        comparisons = self.comparison_results["comparisons"]
        
        speedups = [c["performance"]["speedup"] for c in comparisons]
        fidelities = [c["accuracy"]["fidelity"] for c in comparisons 
                     if c["accuracy"]["fidelity"] is not None]
        
        self.comparison_results["summary"] = {
            "num_tests": len(comparisons),
            "avg_speedup": float(np.mean(speedups)),
            "min_speedup": float(np.min(speedups)),
            "max_speedup": float(np.max(speedups)),
            "avg_fidelity": float(np.mean(fidelities)) if fidelities else None,
            "min_fidelity": float(np.min(fidelities)) if fidelities else None
        }
        
        print("\n" + "=" * 60)
        print("总体统计")
        print("=" * 60)
        print(f"测试数量: {self.comparison_results['summary']['num_tests']}")
        print(f"平均加速比: {self.comparison_results['summary']['avg_speedup']:.2f}x")
        print(f"最小加速比: {self.comparison_results['summary']['min_speedup']:.2f}x")
        print(f"最大加速比: {self.comparison_results['summary']['max_speedup']:.2f}x")
        
        if fidelities:
            print(f"平均保真度: {self.comparison_results['summary']['avg_fidelity']:.10f}")
            print(f"最小保真度: {self.comparison_results['summary']['min_fidelity']:.10f}")
    
    def save_comparison(self, filename: str = "baselines/comparison_results.json"):
        """保存对比结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_results, f, indent=2, ensure_ascii=False)
        print(f"\n对比结果已保存到: {filename}")
    
    def plot_performance(self, output_file: str = "baselines/performance_comparison.png"):
        """绘制性能对比图"""
        comparisons = self.comparison_results["comparisons"]
        
        test_names = [c["test_name"] for c in comparisons]
        sf_times = [c["performance"]["sf_avg_time_ms"] for c in comparisons]
        hybrid_times = [c["performance"]["hybrid_avg_time_ms"] for c in comparisons]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Execution time comparison
        ax1.bar(x - width/2, sf_times, width, label='Strawberry Fields', alpha=0.8)
        ax1.bar(x + width/2, hybrid_times, width, label='HybridCVDV-Simulator', alpha=0.8)
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Average Execution Time (ms)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Speedup
        speedups = [c["performance"]["speedup"] for c in comparisons]
        ax2.bar(x, speedups, alpha=0.8, color='green')
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1x)')
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('Speedup')
        ax2.set_title('HybridCVDV-Simulator Speedup vs Strawberry Fields')
        ax2.set_xticks(x)
        ax2.set_xticklabels(test_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Performance comparison plot saved to: {output_file}")
        plt.close()
    
    def plot_accuracy(self, output_file: str = "baselines/accuracy_comparison.png"):
        """绘制精度对比图"""
        comparisons = self.comparison_results["comparisons"]
        
        # Filter out tests without accuracy data
        valid_comparisons = [c for c in comparisons 
                           if c["accuracy"]["fidelity"] is not None]
        
        if not valid_comparisons:
            print("Warning: No accuracy data available for plotting")
            return
        
        test_names = [c["test_name"] for c in valid_comparisons]
        fidelities = [c["accuracy"]["fidelity"] for c in valid_comparisons]
        trace_distances = [c["accuracy"]["trace_distance"] for c in valid_comparisons]
        
        x = np.arange(len(test_names))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Fidelity
        ax1.bar(x, fidelities, alpha=0.8, color='blue')
        ax1.axhline(y=0.9999, color='r', linestyle='--', label='High Fidelity Threshold (0.9999)')
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Fidelity')
        ax1.set_title('State Fidelity Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.set_ylim([min(fidelities) - 0.0001, 1.0001])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Trace distance (log scale)
        ax2.bar(x, trace_distances, alpha=0.8, color='orange')
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('Trace Distance')
        ax2.set_title('Trace Distance Comparison (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(test_names, rotation=45, ha='right')
        ax2.set_yscale('log')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Accuracy comparison plot saved to: {output_file}")
        plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("HybridCVDV-Simulator vs Strawberry Fields 性能对比")
    print("=" * 60)
    
    # 文件路径
    sf_results = "baselines/strawberryfields_results.json"
    hybrid_results = "tests/hybridcvdv_results.json"
    
    # 检查文件是否存在
    import os
    if not os.path.exists(sf_results):
        print(f"错误: 找不到 Strawberry Fields 结果文件: {sf_results}")
        print("请先运行: uv run baselines/test_strawberryfields.py")
        sys.exit(1)
    
    if not os.path.exists(hybrid_results):
        print(f"错误: 找不到 HybridCVDV-Simulator 结果文件: {hybrid_results}")
        print("请先编译并运行 C++ 基准测试")
        sys.exit(1)
    
    # 创建对比器并运行对比
    comparator = ResultComparator(sf_results, hybrid_results)
    comparator.compare_all()
    comparator.save_comparison()
    
    # 绘制对比图
    try:
        comparator.plot_performance()
        comparator.plot_accuracy()
    except Exception as e:
        print(f"警告: 绘图失败: {e}")
        print("请确保已安装 matplotlib")
    
    print("\n" + "=" * 60)
    print("对比完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
