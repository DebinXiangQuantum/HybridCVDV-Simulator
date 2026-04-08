#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import os

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.configs.paper_style import apply_paper_style, save_figure, SINGLE_COLUMN_PT, BASE_FONT_SIZE

def plot_multigpu_scaling(output_dir):
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, ncols=1, nrows=1, panel_aspect=1.5)
    fig, ax = plt.subplots()
    
    # Data from sc26_jch_nm5_compare_20260329
    labels = ['1 GPU', '2 GPUs']
    total_time = [21097.7, 256.9]
    compute_time = [404.5, 1.88]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, total_time, width, label='Total Time (ms)', color='#1b9e77')
    ax.bar(x + width/2, compute_time, width, label='Compute Time (ms)', color='#d95f02')
    
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    
    save_figure(fig, output_dir, "sc26_multigpu_scaling")
    plt.close(fig)

def plot_precision_validation(output_dir):
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, ncols=1, nrows=1, panel_aspect=1.8)
    fig, ax = plt.subplots()
    
    # Data from sf_precision_test_output.txt
    gates = ['D(0.3)', 'D(0.5)', 'D(1.0)', 'S(0.2)', 'S(0.5)', 'S(0.8)', 'BS(0.3)', 'Kerr(0.1)']
    l2_errors = [1.8e-8, 1.1e-8, 1.4e-8, 2.2e-8, 3.9e-8, 2.4e-8, 2.4e-8, 1.4e-15]
    max_errors = [1.5e-8, 9.2e-9, 8.9e-9, 2.2e-8, 3.5e-8, 1.4e-8, 1.7e-8, 1.4e-15]
    
    x = np.arange(len(gates))
    width = 0.35
    
    ax.bar(x - width/2, l2_errors, width, label='L2 Error', color='#7570b3')
    ax.bar(x + width/2, max_errors, width, label='Max Error', color='#e7298a')
    
    ax.set_ylabel('Error Magnitude')
    ax.set_xticks(x)
    ax.set_xticklabels(gates, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    
    save_figure(fig, output_dir, "sc26_precision_validation")
    plt.close(fig)

def plot_oom_boundary(output_dir):
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, ncols=1, nrows=1, panel_aspect=1.5)
    fig, ax = plt.subplots()
    
    # Data from probe.tsv
    qubits = [7, 8, 9, 10]
    # Memory in GB = elements * 16 / (1024**3)
    jch_elements = [1879048192] * 4
    vqe_elements = [35165044736, 69524783104, 138244259840, 275683213312]
    
    jch_mem = [e * 16 / (1024**3) for e in jch_elements]
    vqe_mem = [e * 16 / (1024**3) for e in vqe_elements]
    
    ax.plot(qubits, vqe_mem, marker='o', color='#d95f02', label='VQE (Dense/No Sharing)')
    ax.plot(qubits, jch_mem, marker='s', color='#1b9e77', label='JCH (HDD Sharing)')
    
    ax.axhline(y=48, color='r', linestyle='--', label='1x L20 Capacity')
    ax.axhline(y=96, color='b', linestyle='--', label='2x L20 Capacity')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Required Memory (GB)')
    ax.set_xticks(qubits)
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    
    save_figure(fig, output_dir, "sc26_oom_boundary")
    plt.close(fig)

def main():
    output_dir = REPO_ROOT / "SC26submission" / "expplots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_multigpu_scaling(output_dir)
    plot_precision_validation(output_dir)
    plot_oom_boundary(output_dir)
    print(f"Figures saved to {output_dir}")

if __name__ == "__main__":
    main()
