# 论文图目录说明

本目录保存 `distributed_8xH800` 实验数据生成的 SVG 论文图。图命名统一为 `fig01` 到 `fig14`。建议论文主线围绕 HybridCVDV 的覆盖范围、成功率、可解边界和稳健性展开，不建议把 raw throughput 作为主要优势证据。

生成这些图片所需代码已集中放在：

`figure_generation/`

| 文件 | 作用 |
|---|---|
| `figure_generation/plot_multigpu_paper_figures.py` | 从原始 distributed result/manifest 直接生成 fig01 到 fig08、fig11 等主图。 |
| `figure_generation/summarize_distributed_results.py` | 汇总原始实验结果，生成 `paper_data_summary` 下的 CSV 表。 |
| `figure_generation/plot_paper_data_summary.py` | 从 `paper_data_summary` 的 CSV 表生成 fig09 到 fig14 等汇总图。 |

## 图表索引

| 序号 | 文件 | 建议位置 | 图的含义 | 能体现的优势 |
|---|---|---|---|---|
| 1 | `fig01_successful_coverage_by_system.svg` | 正文主图 | 对比 HybridCVDV、ATLAS、BQSim 成功跑通的 unique case x GPU configuration 数量。 | 最能体现 HybridCVDV 的覆盖范围优势，尤其在 JCH、VQE、QAOA、QFT、transfer 等混合 CV-DV 电路族上。 |
| 2 | `fig02_phase_e_family_status_pies.svg` | 正文或附录 | Phase E feasibility scan 和 formal rerun 中，各 workload family 的成功/失败状态占比。 | 展示 HybridCVDV 在多数正式重跑电路族中 ok 占主导，失败集中在少数高压力 case。 |
| 3 | `fig03_family_gpu_success_heatmap.svg` | 正文主图 | HybridCVDV 在 family x GPU count 维度上的 Phase E 成功率。 | 说明 GPU 数增加没有引入大面积额外失败，系统在多 GPU 配置下保持稳定。 |
| 4 | `fig04_phase_e_gpu_coverage_pies.svg` | 附录 | Phase E feasibility/rerun 按 GPU 数分组的状态分布。 | 展示 1/2/4/6/8 GPU 下实验覆盖和失败类型。 |
| 5 | `fig05_capacity_scaling_status_pies.svg` | 正文或附录 | Phase C capacity scaling 中 HybridCVDV 与 ATLAS 在各 GPU 数下的状态组成。 | 可用于说明 capacity 实验中的失败模式，以及 baseline 存在 unsupported GPU count 等限制。 |
| 6 | `fig06_cutoff_family_success_heatmap.svg` | 正文主图 | HybridCVDV feasibility scan 中，不同 family 和 cutoff 的成功率。 | 说明哪些 family/cutoff 区域可稳定运行，支撑可解范围分析。 |
| 7 | `fig07_runtime_breakdown_pies.svg` | 附录 | HybridCVDV Phase E 在不同 GPU 数下的 GPU compute、host orchestration、communication、other 时间占比。 | 解释为什么多 GPU 不一定带来线性加速：host/communication 开销会抵消小 case 的并行收益。 |
| 8 | `fig08_throughput_scaling_diagnostic.svg` | 诊断图 | Phase D GKP case 的绝对吞吐和自归一化扩展趋势。 | 可说明 HybridCVDV 自身在部分 GKP case 上有扩展趋势，但不适合作为对 BQSim 的主优势证据。 |
| 9 | `fig09_manifest_status_summary.svg` | 附录 | 每个实验批次的 artifact 数量和整体状态组成。 | 展示数据来源、实验规模和批次完整性。 |
| 10 | `fig10_phase_d_throughput_medians.svg` | 诊断图 | 所有带 throughput telemetry 的电路族吞吐中位数随 GPU 数变化。 | 用于诚实呈现不同电路类型的 throughput 行为；不建议作为 HybridCVDV 主要优势图。 |
| 11 | `fig11_solvable_frontier_by_system.svg` | 正文或附录 | 三套系统在不同 family 上的最大可解有效状态空间规模。 | 展示 HybridCVDV 支持更广的 workload family；但 ATLAS 在个别 family 最大规模上可能更高，表述需谨慎。 |
| 12 | `fig12_family_success_rate_by_system.svg` | 正文或附录 | HybridCVDV、ATLAS、BQSim 按 family 聚合后的成功率对比。 | 展示不同 workload family 上的运行稳健性差异。 |
| 13 | `fig13_run_gpu_success_heatmap.svg` | 附录 | 按 run、system、GPU count 展示成功率。 | 帮助定位失败集中在哪些实验阶段、系统或 GPU 配置。 |
| 14 | `fig14_successful_coverage_summary.svg` | 备用图 | 从 `paper_data_summary` 生成的成功覆盖范围汇总图。 | 与 fig01 类似，可作为覆盖优势的备用版本。 |

## 推荐正文主图组合

建议正文优先使用：

1. `fig01_successful_coverage_by_system.svg`
2. `fig03_family_gpu_success_heatmap.svg`
3. `fig06_cutoff_family_success_heatmap.svg`
4. `fig11_solvable_frontier_by_system.svg`

如果篇幅允许，再加入：

5. `fig02_phase_e_family_status_pies.svg`
6. `fig07_runtime_breakdown_pies.svg`

## 复现命令

从仓库根目录 `/Users/ghost/Downloads/gpu` 运行：

```bash
python3 experiments/results/distributed_8xH800/paper_figures/figure_generation/summarize_distributed_results.py
python3 experiments/results/distributed_8xH800/paper_figures/figure_generation/plot_multigpu_paper_figures.py --output-dir experiments/results/distributed_8xH800/paper_figures
python3 experiments/results/distributed_8xH800/paper_figures/figure_generation/plot_paper_data_summary.py --summary-dir experiments/results/distributed_8xH800/paper_data_summary --output-dir experiments/results/distributed_8xH800/paper_figures
```

## 目前缺少的数据或实验

| 缺少项 | 为什么重要 |
|---|---|
| 三套系统严格 matched 的 canonical case 集合 | 当前覆盖图很强，但部分 family 不是三套系统完全同 case 对比；若要做公平性能比较，需要三套系统都跑同一组 case。 |
| BQSim / ATLAS 的完整 Phase E feasibility scan | 现在 BQSim/ATLAS 的覆盖少于 HybridCVDV，但需要进一步区分是系统不支持、实验未跑完，还是配置过滤导致。 |
| 多 GPU memory peak / aggregate memory 对比 | 多 GPU 的核心价值可能是容量和可承载规模，不只是速度；需要 per-GPU memory peak、总显存使用和 OOM 边界。 |
| 更大 cutoff、更大 mode/qubit 的 frontier sweep | 目前 fig11 展示了可解边界，但还没有完全推到系统极限；更大规模 sweep 能更强地证明边界扩展。 |
| 三系统统一的 runtime breakdown | 目前 runtime breakdown 主要来自 HybridCVDV；如果要解释 BQSim/ATLAS 在某些吞吐上更快，需要三套系统同口径分解。 |
| 重复实验方差、IQR 或 error bar | 吞吐和 runtime 图最好给 median + IQR，增强论文可信度。 |
| 正确性误差随规模变化的数据 | 若要强调 HybridCVDV 不只是能跑，还能正确稳定，需要 fidelity/error/checksum 随 cutoff、family、GPU 数变化的图。 |
