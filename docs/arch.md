  HTDD v2.0 技术设计规格书：自适应辛-张量混合执行架构

  1. 架构设计：双轨并行执行引擎 (Dual-Track Engine)


  HTDD v2.0 放弃了单一的 Fock 空间模拟路径，转而采用一种基于“状态演化阶段”的动态切换架构。


  1.1 核心组件
   * Symbolic Execution Track (SET)：维护高斯态的辛矩阵（Symplectic Matrix）表示。适用于电路的高斯区域（Displacement, Squeezing, Beam
     Splitter）。
   * Tensor Execution Track (TET)：使用 Fock 空间的张量表示。适用于非高斯门（Kerr, Cubic Phase, SNAP）和强纠缠区域。
   * Execution Decision Engine (EDE)：在每一层门操作前，评估纠缠模数 $M$ 和非高斯度 $\chi$，动态决定分支的演化轨道。

  ---

  2. 存储优化：多级压缩体系


  2.1 辛分量池 (Symplectic Pool)
  对于 SET 轨道的分支，状态不再存储为 $D^M$ 向量，而是存储为一个紧凑的元组 $(d, \mathbf{\Sigma})$：
   * 位移向量 $d \in \mathbb{R}^{2M}$。
   * 协方差矩阵 $\mathbf{\Sigma} \in \mathbb{R}^{2M \times 2M}$（或等价的辛矩阵 $S$）。
   * 存储增益：从 $O(16^M)$ 降至 $O(M^2)$。


  2.2 HDD 逻辑分裂 (Mixture Branching)
  当遇到非高斯操作且不满足“弹出”条件时，HDD 节点执行 Gaussian Mixture Decomposition：
   * 将非高斯门近似分解为 $K$ 个高斯分支的叠加。
   * HDD 增加 $K$ 个叶子节点，保持在 SET 轨道运行，避免进入 TET 轨道。

  ---


  3. 关键 Kernel 算法描述

  以下是 HTDD v2.0 的核心 GPU 内核设计，这些算法保证了数学严谨性与 HPC 性能。


  Kernel A: 批量辛阵变换 (Batched Symplectic Update)
  目标：在 GPU 上同时更新数千个高斯分支的状态。


   * 数学逻辑：
      高斯门 $G$ 对应一个辛变换 $(S_g, d_g)$。
      状态更新公式：
      $$ \mathbf{\Sigma}_{new} = S_g \mathbf{\Sigma}_{old} S_g^T $$
      $$ d_{new} = S_g d_{old} + d_g $$
   * 算法流程：
       1. 将 $N$ 个分支的 $\mathbf{\Sigma}_i$ 打包为维度为 $[N, 2M, 2M]$ 的张量。
       2. 调用 cuBLAS Batched GEMM：
           * Step 1: $Tmp_i = S_g \cdot \mathbf{\Sigma}_i$
           * Step 2: $\mathbf{\Sigma}_{new, i} = Tmp_i \cdot S_g^T$
       3. 使用向量化线程块并行更新 $d_i$。
   * HPC 创新：通过提高 $N$ (Batch Size) 掩盖小矩阵运算的内存潜伏期。


  Kernel B: 高斯-Fock 解析投影 (Gaussian-to-Fock Projection/Ejection)
  目标：当状态必须转为张量表示时，直接在 GPU 显存内生成 Fock 振幅。


   * 数学逻辑：
      对于单模高斯态 $|d, \Sigma\rangle$，其 Fock 空间振幅 $c_n = \langle n | \psi \rangle$ 的解析式涉及 Hermite 多项式：
      $$ c_n = \frac{1}{\sqrt{n!}} \exp(A) \cdot B^n \cdot H_n(C) $$
      其中 $A, B, C$ 是由 $(d, \Sigma)$ 导出的解析系数。
   * 算法流程：
       1. Thread Mapping：每个线程处理一个 Fock 索引 $n \in [0, D-1]$。
       2. Recurrence Relation：由于直接计算阶乘和高阶多项式会导致数值溢出，线程内部采用稳定递推关系：
          $$ \sqrt{n+1} c_{n+1} = (\text{解析项}) c_n + (\text{解析项}) c_{n-1} $$
       3. Normalization：计算所有线程的和进行归一化。
   * 数学边界：对于多模纠缠态，使用多维 Hermite 多项式递推，或使用梯算符 $a^\dagger$ 在 GPU 上的迭代作用。


  Kernel C: 批量保真度去重 (Batched Overlap/Fidelity)
  目标：计算分支间的重叠度，用于 HDD 的分支合并（Merge）。


   * 数学逻辑：
      两个高斯态 $\rho_1(d_1, \Sigma_1)$ 和 $\rho_2(d_2, \Sigma_2)$ 的保真度 $F$：
      $$ F = \frac{2^M}{\sqrt{\det(\mathbf{\Sigma}_1 + \mathbf{\Sigma}_2)}} \exp \left( -\frac{1}{2} (d_1 - d_2)^T (\mathbf{\Sigma}_1 +
  \mathbf{\Sigma}_2)^{-1} (d_1 - d_2) \right) $$
   * 算法流程：
       1. Batch Matrix Inversion：调用 cuSolver 批量计算矩阵和 $(\mathbf{\Sigma}_1 + \mathbf{\Sigma}_2)^{-1}$。
       2. Reduction Kernel：每个线程块计算一个 $d^T \Sigma^{-1} d$ 的内积。
       3. Threshold Pruning：若 $F > 1 - \epsilon$，则在 CPU 端标记合并。
   * HPC 优势：全解析计算，计算量 $O(M^3)$ 远小于张量内积的 $O(D^M)$。

  ---

  4. 论文讲解逻辑梳理


  在 SC26 投稿中，应按以下策略分章节强调技术突破：


  4.1 架构设计：自适应资源感知 (Resource-Aware Hybridism)
   * 讲解点：强调 EDE (决策引擎)。它不只是简单的开关，而是基于 GPU 实时显存负载和算子数学特性（非高斯度）的闭环控制系统。
   * 亮点：展现了系统在复杂线路下的“弹性”。


  4.2 存储优化：特征空间压缩 (Eigenspace Compression)
   * 讲解点：
       1. 辛分量池如何将存储密度提升 1000x 以上。
       2. HDD-Symbolic Link：解释 HDD 的叶子节点如何动态重定向到辛分量池或 Fock-ELL 张量块。
   * 亮点：通过数学对称性解决物理存储瓶颈。


  4.3 计算优化：解析加速与批处理 (Analytical Batching)
   * 讲解点：
       1. SET 轨道的批处理吞吐量：展示在 GPU 上同时处理 10^5 个辛矩阵的效率。
       2. TET 轨道的稀疏优化：现有的 Fock-ELL SpMV 性能。
       3. 跨轨转换开销：证明 Kernel B 的解析投影速度远快于数据传输。
   * 亮点：将模拟任务从“内存受限”转化为“计算受限”，充分利用 GPU 算力。

  ---


  5. 实现阶段规划
   1. Phase I: 封装 GaussianState 类，包含辛矩阵和位移向量。
   2. Phase II: 开发 Kernel A 和 Kernel C，实现 SET 轨道的闭环运行。
   3. Phase III: 开发 Kernel B (Ejection)，实现从 SET 到原有 TET 轨道的单向回退。
   4. Phase IV: 集成决策引擎，进行大规模 benchmark 测试。