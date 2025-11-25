这是一份规范的**Hybrid Tensor-DD (HTDD) 量子模拟器技术实现规格说明书**。该文档整合了之前的架构讨论，针对高性能计算（HPC）场景下的 GPU 加速器设计进行了标准化描述。

---

# Hybrid Tensor-DD (HTDD) 量子模拟器技术规格书

**版本**: v1.5
**适用架构**: Hybrid CV-DV Quantum Systems (e.g., Qubit-Qumode Architecture)
**硬件目标**: CPU (逻辑控制) + NVIDIA GPU (张量计算)

---

## 1. 架构设计概览

本系统采用 **控制层-物理层分离 (Control-Physical Separation)** 架构，旨在解决混合量子系统中的“状态空间爆炸”问题。

*   **控制层 (CPU Host)**: 使用 **混合决策图 (HDD, Hybrid Decision Diagram)** 压缩 $N$ 个 Qubit 的离散希尔伯特空间。负责处理纠缠结构、逻辑分支剪枝和任务调度。
*   **物理层 (GPU Device)**: 使用 **Tensor Pool** 和 **Fock-ELL** 格式存储 $M$ 个 Qumode 的连续变量状态。负责高吞吐量的线性代数运算。

---

## 2. 数据结构定义

### 2.1 GPU 物理层 (Device Memory)

物理层不感知 Qubit 的逻辑结构，仅将其视为一堆待处理的 Fock 态向量。

#### 2.1.1 全局状态池 (Global CV State Pool)
存储所有活跃的 Qumode 态。采用 **Structure of Arrays (SoA)** 布局以利于合并访存。

```cpp
struct CVStatePool {
    // 物理存储：扁平化的大数组
    // 维度: [Capacity * D_total]
    // D_total = d^M (单个CV分支的维度)
    cuDoubleComplex* data; 

    // 状态元数据
    int d_trunc;        // 截断维数 D
    int capacity;       // 最大支持的独立状态数
    int active_count;   // 当前活跃状态数
    int* free_list;     // 垃圾回收链表
};
```

#### 2.1.2 稀疏算符存储 (Fock-ELL Operator)
针对 Displacement 等带状矩阵的优化存储格式。

```cpp
struct FockELLOperator {
    // 存储非零元素的值
    // 维度: [D, K_eff] (Row-Major)
    cuDoubleComplex* ell_val; 

    // 存储对应的列索引
    // 维度: [D, K_eff]
    // 值 -1 表示该位置为 Padding
    int* ell_col;

    int max_bandwidth; // K_eff (有效带宽)
    int dim;           // D
};
```

### 2.2 CPU 逻辑层 (Host Memory)

#### 2.2.1 混合决策图节点 (HDD Node)
HDD 的叶子节点充当指向 GPU 显存的指针。

```cpp
struct HDDNode {
    // 唯一标识 (用于 Hash Table 去重)
    size_t unique_id;
    
    // Qubit 层级 (0 to N-1). -1 表示终端节点
    int16_t qubit_level;
    
    // 引用计数 (用于 GPU 显存垃圾回收)
    std::atomic<int> ref_count;

    // 子节点指针与权重 (Qubit 分支)
    HDDNode* low;                 // |0> 分支
    HDDNode* high;                // |1> 分支
    std::complex<double> w_low;   // 权重 alpha
    std::complex<double> w_high;  // 权重 beta

    // GPU 句柄 (仅 Terminal Node 有效)
    // 指向 CVStatePool 中的索引。若为 -1，表示该分支概率为0
    int32_t tensor_id; 
};
```

---

## 3. 门操作计算规范与公式

为了极致性能，系统根据门的数学特性将计算分为四个优化层级。

### Level 0: 对角门 (Diagonal Gates)
**特性**: 在 Fock 基底是对角阵。不涉及矩阵乘法，仅做 Element-wise 相乘。

*   **典型门**: Phase Rotation $R(\theta)$, Kerr $K(\chi)$, Conditional Parity $CP$.
*   **数学公式**:
    $$ \vec{\psi}_{out}[n] = \vec{\psi}_{in}[n] \cdot e^{-i \cdot f(n)} $$
    其中 $f(n)$ 对 $R(\theta)$ 为 $\theta n$，对 Kerr 为 $\chi n^2$。
*   **GPU 实现**:
    *   **Pre-computation**: 在 Constant Memory 中预存相位表 `PhaseTable[n]`。
    *   **Kernel**:
        ```cpp
        state[id * D + n] = cuCmul(state[id * D + n], PhaseTable[n]);
        ```
*   **复杂度**: $O(1)$ 显存读取/线程。

### Level 1: 梯算符门 (Ladder/Shift Gates)
**特性**: 矩阵仅有一条非零对角线（次对角线）。无需存储矩阵，系数实时计算。

*   **典型门**: Photon Creation $\hat{a}^\dagger$, Annihilation $\hat{a}$.
*   **数学公式**:
    *   Creation: $\vec{\psi}_{out}[n] = \sqrt{n} \cdot \vec{\psi}_{in}[n-1]$
    *   Annihilation: $\vec{\psi}_{out}[n] = \sqrt{n+1} \cdot \vec{\psi}_{in}[n+1]$
*   **GPU 实现**:
    *   **Kernel**: 使用 `__shfl_up/down` (Warp Shuffle) 或直接访存 `in[n-1]`。
        ```cpp
        // Creation
        val = (n > 0) ? sqrt(double(n)) * in[n-1] : 0;
        out[n] = val;
        ```
*   **复杂度**: $O(1)$ 显存读取/线程 (Matrix-Free)。

### Level 2: 通用单模门 (General Single-Mode)
**特性**: 矩阵为带状稀疏矩阵。需使用 Fock-ELL 格式。

*   **典型门**: Displacement $D(\alpha)$, Squeezing $S(\xi)$.
*   **数学公式**:
    $$ \vec{\psi}_{out}[n] = \sum_{k=0}^{K-1} \text{ELL\_Val}[n][k] \cdot \vec{\psi}_{in}[ \text{ELL\_Col}[n][k] ] $$
*   **GPU 实现**:
    *   **Optimization**: 动态计算有效带宽 $K_{eff}$。对于 Hamiltonian Simulation 中的微小 $\alpha$，通常 $K_{eff} \ll D$。
    *   **Kernel**: 经典的 ELL-SpMV (Sparse Matrix-Vector Multiplication)。
*   **复杂度**: $O(K_{eff})$ 显存读取/线程。

### Level 3: 双模混合门 (Two-Mode Mixing)
**特性**: 作用于两个 Qumode。矩阵巨大 ($D^2 \times D^2$)。

*   **典型门**: Beam Splitter $BS(\theta, \phi)$.
*   **物理性质**: 光子数守恒 ($n_1 + n_2 = N_{total} = \text{const}$)。矩阵是块对角化的 (Block Diagonal)。
*   **数学公式**:
    将状态空间分解为 $2D$ 个子空间。对于总光子数 $k$ 的子空间，状态向量 $\vec{v}_k$ 长度为 $k+1$。
    $$ \vec{v}_{k, out} = U_{BS}^{(k)} \cdot \vec{v}_{k, in} $$
    其中 $U_{BS}^{(k)}$ 是 $(k+1) \times (k+1)$ 的稠密矩阵。
*   **GPU 实现**:
    *   **Mapping**: 每个 CUDA Block 处理一个子空间 $k$。
    *   **Memory**: 将小的 $\vec{v}_k$ 和矩阵 $U_{BS}^{(k)}$ 加载到 Shared Memory 中进行 Dense MV 计算。
*   **复杂度**: $O(D^3)$ 总计算量 (远优于 $O(D^4)$)。

### Level 4: 混合控制门 (Hybrid Control Gates)
**特性**: Qubit 控制 Qumode。涉及 HDD 结构变更。

*   **典型门**: Controlled-Displacement $CD(\alpha)$.
*   **逻辑流程**:
    1.  **Traverse**: CPU 遍历 HDD。
    2.  **Check Identity**: 对于 Control=$|0\rangle$ 分支，跳过。
    3.  **Copy-on-Write (CoW)**: 对于 Control=$|1\rangle$ 分支：
        *   检查指向的 `tensor_id` 是否被其他分支共享（引用计数 > 1）。
        *   若共享，执行 **Device-to-Device Memcpy** 复制一份副本，得到 `new_id`，并更新 HDD 指针。
    4.  **Batch Execution**: 将所有涉及修改的 ID 加入 Batch List，调用 Level 2 Kernel。

---

## 4. 运行时调度流程 (Runtime Pipeline)

### 4.1 指令融合 (Instruction Fusion)
为了减少 Kernel Launch 开销：
1.  **Accumulation**: CPU 缓存连续的 Level 0/1/2 门指令。
2.  **Fusion**: 将连续的 Displacement $D(\alpha_1) \cdot D(\alpha_2)$ 合并为 $D(\alpha_1 + \alpha_2)$（忽略相位）。
3.  **Trigger**: 遇到测量、非对角 Qubit 门或显存压力阈值时，触发执行。

### 4.2 批处理执行 (Batched Execution)
所有 GPU Kernel 必须支持 Batch 模式：

```cpp
__global__ void Batched_Gate_Kernel(
    CVStatePool pool, 
    int* target_indices, // 需要更新的状态 ID 列表
    int batch_size,
    GateParams params
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;
    
    int state_idx = target_indices[batch_id];
    cuDoubleComplex* psi = &pool.data[state_idx * pool.d_trunc];
    
    // ... 执行具体的 Level 0-3 计算逻辑 ...
}
```

### 4.3 内存生命周期管理
*   **Allocation**: 使用 `cudaMalloc` 预分配大块显存池，自行管理分块，避免频繁系统调用。
*   **Deduplication (去重)**: 
    *   周期性扫描 HDD。
    *   若两个 `tensor_id` 的内容保真度 $|\langle \psi_a | \psi_b \rangle|^2 > 1 - \epsilon$，则销毁其中一个，合并 HDD 路径。

---

## 5. 性能关键指标 (KPIs)

*   **显存带宽利用率**: Level 0/1 Kernel 应达到 GPU 理论带宽的 80% 以上。
*   **HDD 压缩率**: 在弱纠缠电路中，活跃 `tensor_id` 数量应远小于 $2^N$。
*   **Batch Size**: 目标 Batch Size 应 $> 64$ 以掩盖 Kernel Launch Latency。

### 实现建议
采用 c++ + cuda 实现，同时采用一般的矩阵向量乘实现相同的操作，对比结果进行单元测试，知道所有门函数的输入输出都是正确的之后再进行整个量子线路的测试。