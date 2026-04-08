#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add the project root to sys.path to import paper_style
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from experiments.configs.paper_style import apply_paper_style, save_figure

def geometric_mean(iterable):
    vals = np.array(iterable)
    vals = vals[vals > 0]
    if len(vals) == 0: return 0
    return np.exp(np.mean(np.log(vals)))

def get_mapping(k_name):
    """
    Final mapping based on source code analysis and ncu_full_analysis_report.csv.
    Categorizes all 17 unique kernels into SC26 terminology.
    """
    k = k_name.lower()
    
    # Gaussian (L2) -> Gaussian
    if 'fused_copy_displacement' in k:
        return 'Gaussian', 'Gaussian Track', 'displacement\n compute'
    if 'fused_covariance' in k:
        return 'Gaussian', 'Gaussian Track', 'covariance\n compute'
    # Level 0 Ops -> Diag Ops(L0)
    if any(x in k for x in ['phase_rotation', 'conditional_parity', 'multisnap', 'fused_diagonal', 'ckgate_multimode']):
        return 'Diag Ops(L0)', 'Fock Track', None
    
    # Level 1 Ops -> Ladder (L1)
    if 'creation_operator' in k or 'annihilation_operator' in k:
        return 'Ladder (L1)', 'Fock Track', None
    
    # Level 2 Ops -> ELL Ops (L2)
    if 'squeezing_ell' in k:
        return 'ELL Ops (L2)', 'Fock Track', 'squeezing gate'
    
    if 'apply_controlled_displacement' in k:
        return 'ELL Ops (L2)', 'Fock Track', ' \n displacement gate'

    # Level 3 Ops -> Subspace (L3)
    if 'bs_subspace' in k:
        return 'Subspace (L3)', 'Fock Track', 'BS gate'
    
    # State Copy
    if any(x in k for x in ['copy_back_two_mode', 'copy_back_ladder', 'copy_result']):
        return 'State Copy', 'Fock Track', None
    
    # State Conversion
    if any(x in k for x in ['inspect_scaled_vacuum', 'axpy_state_vector']):
        return 'State Conversion', 'State Conversion', None
    
    return "Unknown", "Unknown", None

def process_data():
    csv_path = project_root / 'artifacts/remote_ncu_reports_20260329/ncu_random_circuit_mgpu_20260329_1808.csv'
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Metric Indices
        idx_sm_a = header.index('sm__instruction_throughput.avg.pct_of_peak_sustained_active')
        idx_l1_a = header.index('l1tex__throughput.avg.pct_of_peak_sustained_active')
        idx_sm_e = header.index('sm__throughput.avg.pct_of_peak_sustained_elapsed')
        idx_mem_e = header.index('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed')
        idx_k = 4
        next(reader) # units
        
        kernel_data = {} # {kernel_name: {metrics...}}
        
        for row in reader:
            try:
                k_name = row[idx_k].split('(')[0]
                # Filter out namespaces if present
                if '::' in k_name:
                    k_name = k_name.split('::')[-1]
                
                sm_a = float(row[idx_sm_a].replace(',',''))
                l1_a = float(row[idx_l1_a].replace(',',''))
                sm_e = float(row[idx_sm_e].replace(',',''))
                mem_e = float(row[idx_mem_e].replace(',',''))
                
                if k_name not in kernel_data:
                    kernel_data[k_name] = {'sm_a':[], 'l1_a':[], 'sm_e':[], 'mem_e':[]}
                kernel_data[k_name]['sm_a'].append(sm_a)
                kernel_data[k_name]['l1_a'].append(l1_a)
                kernel_data[k_name]['sm_e'].append(sm_e)
                kernel_data[k_name]['mem_e'].append(mem_e)
            except: continue

    # Compute GM for each kernel point
    plot_points = []
    for kname, stats in kernel_data.items():
        label, track, text = get_mapping(kname)
        gm_sm_a = geometric_mean(stats['sm_a'])
        gm_l1_a = geometric_mean(stats['l1_a'])
        gm_sm_e = geometric_mean(stats['sm_e'])
        gm_mem_e = geometric_mean(stats['mem_e'])
        
        plot_points.append({
            'kernel': kname,
            'label': label,
            'track': track,
            'sm_a': gm_sm_a,
            'l1_a': gm_l1_a,
            'sm_e': gm_sm_e,
            'mem_e': gm_mem_e,
            'text': text
        })
        
    return plot_points

def create_scatter(data, x_key, y_key, x_label, y_label, x_lim, filename):
    figsize = apply_paper_style(width_pt=240.0, panel_aspect=1.8)
    fig, ax = plt.subplots(figsize=figsize)
    
    # User defined colors for Terminologies
    term_colors = {
        'Gaussian': "#5F8B4C",
        'State Copy': "#3C77B4",
        'State Conversion': "#7570b3", # Added 7th color
        
        'Diag Ops(L0)': "#FFDDAB",
        'Ladder (L1)': "#f84848",
        'ELL Ops (L2)': "#ffa806",
        'Subspace (L3)': "#945034",
        
    }
    
    # Markers for Tracks
    track_markers = {
        'Gaussian Track': 'o',
        'Fock Track': 's',
        'State Conversion': '^'
    }
    
    added_labels = set()
    
    # Sort data to ensure consistent legend order
    sorted_data = sorted(data, key=lambda x: (x['track'], x['label']))
    
    for d in sorted_data:
        label = d['label']
        track = d['track']
        
        # Plot point
        ax.scatter(d[x_key], d[y_key], 
                   c=term_colors.get(label, 'gray'), 
                   marker=track_markers.get(track, 'x'),
                   s=20, alpha=0.9, edgecolors='black', linewidths=0.2)
        
        # Annotate kernel name briefly or just the terminology? 
        # User asked to label by Terminology.
        # Since multiple points might have same Terminology, we label individually.
        if d['text']:
            ax.annotate(d['text'], (d[x_key], d[y_key]-2), xytext=(3, 1), 
                        textcoords='offset points', fontsize=6, alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_lim: ax.set_xlim(x_lim)
    
    # Custom Legend Construction
    from matplotlib.lines import Line2D
    
    # Define order for legend
    col1_labels = ['Gaussian', 'State Copy', 'State Conversion']
    col2_labels = ['Diag Ops(L0)', 'Ladder (L1)', 'ELL Ops (L2)', 'Subspace (L3)']
    
    def make_proxy(label):
        if label == 'Fock Track':
            return Line2D([0], [0], color='none', label='Fock Track')
        if label == ' ':
            return Line2D([0], [0], color='none', label=' ')
        # Determine track for marker
        track = 'Fock Track'
        if label == 'Gaussian': track = 'Gaussian Track'
        if label == 'State Conversion': track = 'State Conversion'
        
        return Line2D([0], [0], marker=track_markers[track], color='w', 
                      markerfacecolor=term_colors[label], 
                      markeredgecolor='black', markeredgewidth=0.2,
                      markersize=5, label=label)

    # Construct the handles list for ncol=2
    # Row-major ordering for ncol=2: [R1C1, R1C2, R2C1, R2C2, ...]
    handles = []
    # R1
    handles.append(make_proxy(' '))

    handles.append(make_proxy('Gaussian'))

    # R2
    handles.append(make_proxy('State Copy'))
    handles.append(make_proxy('State Conversion'))

    handles.append(make_proxy(' '))
    handles.append(make_proxy('Fock Track'))
    handles.append(make_proxy('Diag Ops(L0)'))
    # R3
    handles.append(make_proxy('Ladder (L1)'))
    # R4
    handles.append(make_proxy('ELL Ops (L2)'))
    # R5
    handles.append(make_proxy('Subspace (L3)'))

    labels = [h.get_label() for h in handles]
    
    ax.legend(handles, labels, loc='best', frameon=False, fontsize=5, 
              ncol=2, columnspacing=0.5, handletextpad=0.2)

    save_figure(fig, project_root / 'SC26submission' / 'expplots', filename)
    plt.close(fig)

if __name__ == '__main__':
    pdata = process_data()
    if pdata:
        print(f"Plotting {len(pdata)} kernel points...")
        create_scatter(pdata, 'l1_a', 'sm_a', 'L1TEX Active %', 'SM Instruction Active %', (0, 30), 'sc26_microarch_scatter')
        create_scatter(pdata, 'mem_e', 'sm_e', 'DRAM Throughput % (Elapsed)', 'SM Throughput % (Elapsed)', None, 'sc26_microarch_util')
        print("Done.")
