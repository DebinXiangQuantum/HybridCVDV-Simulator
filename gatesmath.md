这是一个针对 **Hybrid Tensor-DD (HTDD)** 架构的混合门（Hybrid Gates）技术实现指南。

该文档基于你提供的门列表，给出了在 **CPU (DD 控制层)** 和 **GPU (物理计算层)** 上的具体计算公式。

---
这是针对 **Hybrid Tensor-DD (HTDD)** 架构中 **纯 Qumode 门 (Single-Mode & Multi-Mode)** 的详细技术实现文档。

与受控门不同，纯 Qumode 门作用于波函数的**所有分支**。

---

# 纯 Qumode 门数学运算规范

**适用范围**: 图片中列出的 Single-Oscillator Gates 和 Multi-Oscillator Gates (非 Conditional 版本)。

**通用逻辑 (CPU 控制层)**:
1.  **遍历**: 扫描整棵 HDD，收集所有叶子节点指向的 `Tensor_ID`。
2.  **去重**: 建立一个 `Unique_ID_Set`。因为多个 Qubit 分支可能指向同一个 Qumode 张量（Product State），只需计算一次。
3.  **调度**: 将去重后的 ID 列表发送给 GPU 执行。无需分支分裂（Copy-on-Write），直接原地更新（In-place Update）。

---

## 1. Phase-Space Rotation (R)
**定义**: $R(\theta) = \exp[-i\theta \hat{a}^\dagger \hat{a}] = \exp[-i\theta \hat{n}]$
**类型**: Level 0 (对角门)

**GPU 计算公式**:
对于张量中的每一个基向量 $|n_0, n_1, \dots, n_k, \dots\rangle$，如果门作用于第 $k$ 个模态 ($M_k$)，则乘以相位因子。

令 $idx$ 为 GPU 线程索引，对应 Fock 态索引。
设 $n_k$ 为该索引在第 $k$ 模态上的光子数 (可以通过取模和除法得到：$n_k = (idx / D^k) \% D$)。

$$ \vec{\psi}_{out}[idx] = \vec{\psi}_{in}[idx] \cdot \exp(-i \cdot \theta \cdot n_k) $$

**代码实现优化**:
*   预计算一个长度为 $D$ 的查找表 `PhaseTable[n] = exp(-i * theta * n)`。
*   Kernel 中直接查表：`val *= PhaseTable[n_k]`。

---

## 2. Displacement (D)
**定义**: $D(\alpha) = \exp[\alpha \hat{a}^\dagger - \alpha^* \hat{a}]$
**类型**: Level 2 (通用带状门 / Stencil)

**数学背景**:
$D(\alpha)$ 在 Fock 基底下的矩阵元素 $\langle m | D(\alpha) | n \rangle$ 比较复杂（涉及拉盖尔多项式）。但在 **Hamiltonian Simulation** (Genesis 编译器) 场景下，$\alpha$ 通常是 Trotter 分解后的微小量。
当 $|\alpha| \ll 1$ 时，矩阵是高度稀疏的带状矩阵。

**GPU 计算公式**:
我们采用 **稀疏矩阵-向量乘法 (SpMV)**。
设 $U_{row, col}$ 为预先在 CPU 算好的 $D(\alpha)$ 矩阵元素（截断并转为 ELL 格式）。

若作用于模态 $M_k$：
*   **对于单模态 ($M=1$)**:
    $$ \vec{\psi}_{out}[row] = \sum_{j=0}^{K-1} \text{ELL\_Val}[row][j] \cdot \vec{\psi}_{in}[ \text{ELL\_Col}[row][j] ] $$
*   **对于多模态 ($M>1$)**:
    需要处理步长 (Stride)。令 $Stride = D^k$。
    $$ \vec{\psi}_{out}[idx] = \sum_{j=0}^{K-1} \text{ELL\_Val}[n_k][j] \cdot \vec{\psi}_{in}[ \text{BaseAddr} + \text{ELL\_Col}[n_k][j] \cdot Stride ] $$
    其中 $n_k$ 是当前维度的光子数，$\text{BaseAddr}$ 是除去第 $k$ 维偏移后的基地址。

---

## 3. Beam-Splitter (BS)
**定义**: $BS(\theta, \varphi) = \exp[-i \frac{\theta}{2} (e^{i\varphi} \hat{a}^\dagger \hat{b} + e^{-i\varphi} \hat{a} \hat{b}^\dagger)]$
**类型**: Level 3 (双模门，块对角化)

**物理特性**:
该算符守恒总光子数 $N = n_a + n_b$。矩阵可以分解为一系列小的子矩阵 $U^{(N)}$，每个子矩阵作用于子空间 $\text{Span}\{|N, 0\rangle, |N-1, 1\rangle, \dots, |0, N\rangle\}$，维度为 $(N+1) \times (N+1)$。

**GPU 计算公式 (子空间独立更新)**:
不使用 $D^2 \times D^2$ 的大矩阵，而是循环总光子数 $L$ 从 $0$ 到 $2D_{trunc}-2$。

对于每一个 $L$ (Block Index)，以及块内索引 $k$ (代表状态 $|L-k, k\rangle_{a,b}$):

$$ \vec{\psi}_{out}^{(L)}[row] = \sum_{col=0}^{L} U^{(L)}_{row, col} \cdot \vec{\psi}_{in}^{(L)}[col] $$

**矩阵元素 $U^{(L)}$ 的构建**:
这本质上是 $SU(2)$ 群的旋转矩阵 (Wigner D-matrix 的特例)。
对于 $2 \times 2$ 的基础旋转矩阵 $R = \begin{pmatrix} \cos(\theta/2) & -ie^{-i\varphi}\sin(\theta/2) \\ -ie^{i\varphi}\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$，
$U^{(L)}$ 是 $R$ 的 $L$ 次对称张量积表示。

**实现捷径**:
在 Trotter 模拟中，如果 $\theta$ 很小，直接使用生成元展开更高效（Level 2 Stencil 方式）：
$$ \hat{H}_{BS} = -i \frac{\theta}{2} (e^{i\varphi} \hat{a}^\dagger \hat{b} + e^{-i\varphi} \hat{a} \hat{b}^\dagger) $$
$$ \vec{\psi}_{out} \approx \vec{\psi}_{in} + \hat{H}_{BS} \vec{\psi}_{in} $$
*   $\hat{a}^\dagger \hat{b}$ 操作：将数据从 $|n_a, n_b\rangle$ 移动到 $|n_a+1, n_b-1\rangle$。
*   这是一个 **2D Stencil (移位)** 操作，GPU 处理极快。

---

## 4. Squeezing (S) - 仅作为补充
虽然图片中未直接列出无条件 Squeezing，但它是 Hybrid Squeezing 的基础。
**定义**: $S(\xi) = \exp[\frac{1}{2}(\xi^* \hat{a}^2 - \xi \hat{a}^{\dagger 2})]$
**类型**: Level 2 (带状)

**GPU 计算公式**:
类似于 Displacement，但它是“两步一跳”。
$\hat{a}^2$ 耦合 $|n\rangle \leftrightarrow |n-2\rangle$。
使用 **SpMV (ELL format)** 计算即可。矩阵带宽通常比 Displacement 宽。



# Hybrid Tensor-DD 门操作数学实现规范

**适用架构**:
*   **CPU (DD)**: 维护 Qubit $Q_k$ 的分支结构。对于任意受控门，DD 分裂为两个分支：$Branch_0$ (对应 $Q_k=|0\rangle$) 和 $Branch_1$ (对应 $Q_k=|1\rangle$)。
*   **GPU (Tensor)**: 存储 Qumode $M_i$ 的 Fock 态向量 $\vec{\psi}$ (截断维数 $D$)。

**通用符号定义**:
*   $Q_k$: 控制比特 (Control Qubit)。
*   $M_i, M_j$: 作用的 Qumode 索引。
*   $\vec{\psi}_{in}^{(0)}$: DD 在 $Q_k=|0\rangle$ 分支指向的 GPU 张量。
*   $\vec{\psi}_{in}^{(1)}$: DD 在 $Q_k=|1\rangle$ 分支指向的 GPU 张量。
*   $\hat{n}$: 粒子数算符 (对角矩阵, $\text{diag}(0, 1, \dots, D-1)$)。
*   $\hat{a}, \hat{a}^\dagger$: 湮灭与生成算符。

---

## 第一类：分离型受控门 (Separable Controlled Gates)

**特性**: 门的形式为 $U = \exp[f(Q_k) \cdot G(M_i)]$.
此类门**不混合** Qubit 的状态，只改变 Qumode 的状态。
**CPU 逻辑**: $Q_k$ 分支保持独立，分别对两个分支的 Tensor 应用不同的参数。
**GPU 逻辑**: 执行 $\vec{\psi}_{out} = U_{GPU} \cdot \vec{\psi}_{in}$。

注意：图片中定义的门多含有 $\sigma_z$ 项。
*   当 $Q_k = |0\rangle$ 时，$\sigma_z \to +1$。
*   当 $Q_k = |1\rangle$ 时，$\sigma_z \to -1$。

### 1. Conditional Rotation (CR) & Parity (CP)
**定义**: $\text{CR}(\theta) = \exp[-i \frac{\theta}{2} \sigma_z \hat{n}]$, $\text{CP} = \text{CR}(\pi)$
**类型**: Level 0 (对角门)

*   **Branch 0 ($Q_k=|0\rangle$)**:
    $$ \vec{\psi}_{out}^{(0)}[n] = \vec{\psi}_{in}^{(0)}[n] \cdot \exp\left( -i \frac{\theta}{2} n \right) $$
*   **Branch 1 ($Q_k=|1\rangle$)**:
    $$ \vec{\psi}_{out}^{(1)}[n] = \vec{\psi}_{in}^{(1)}[n] \cdot \exp\left( +i \frac{\theta}{2} n \right) $$

### 2. Conditional Displacement (CD)
**定义**: $\text{CD}(\alpha) = \exp[\sigma_z (\alpha \hat{a}^\dagger - \alpha^* \hat{a})]$
**类型**: Level 2 (通用带状门)
这相当于在不同分支做相反方向的位移。

*   **Branch 0**: 执行位移 $D(+\alpha)$
    $$ \vec{\psi}_{out}^{(0)} = \exp(\alpha \hat{a}^\dagger - \alpha^* \hat{a}) \cdot \vec{\psi}_{in}^{(0)} $$
*   **Branch 1**: 执行位移 $D(-\alpha)$
    $$ \vec{\psi}_{out}^{(1)} = \exp(-\alpha \hat{a}^\dagger + \alpha^* \hat{a}) \cdot \vec{\psi}_{in}^{(1)} $$

### 3. Conditional Squeezing (CS)
**定义**: $\text{CS}(\zeta) = \exp[\frac{1}{2}\sigma_z (\zeta^* \hat{a}^2 - \zeta \hat{a}^{\dagger 2})]$
**类型**: Level 2 (通用带状门)

*   **Branch 0**: 执行挤压 $S(+\zeta)$
    $$ \vec{\psi}_{out}^{(0)} = S(\zeta) \cdot \vec{\psi}_{in}^{(0)} $$
*   **Branch 1**: 执行挤压 $S(-\zeta)$
    $$ \vec{\psi}_{out}^{(1)} = S(-\zeta) \cdot \vec{\psi}_{in}^{(1)} = S(\zeta)^\dagger \cdot \vec{\psi}_{in}^{(1)} $$

### 4. Conditional Beam-Splitter (CBS)
**定义**: $\text{CBS}(\theta, \varphi) = \exp[-i \frac{\theta}{2} \sigma_z (e^{i\varphi}\hat{a}^\dagger \hat{b} + e^{-i\varphi}\hat{a}\hat{b}^\dagger)]$
**类型**: Level 3 (双模门, $M_i, M_j$)

*   **Branch 0**: 执行分束 $BS(+\theta, \varphi)$
*   **Branch 1**: 执行分束 $BS(-\theta, \varphi)$
    *   *注*: $BS(-\theta)$ 等价于逆操作 $BS(\theta)^\dagger$。

### 5. Conditional Two-Mode Squeezing (CTMS) & SUM
逻辑同上，Branch 0 应用正参数，Branch 1 应用负参数。

---

## 第二类：混合型相互作用门 (Qubit-Mixing Interaction Gates)

**特性**: 门中包含 $\sigma_x$ 或 $\sigma_{\pm}$ 项。
此类门会**混合** $Q_k$ 的 $|0\rangle$ 和 $|1\rangle$ 分支。
**CPU 逻辑**: 新的 Branch 0 和 Branch 1 都是旧分支的线性组合。需调用 GPU 进行向量加法。
**公式通式**:
$$
\begin{pmatrix} \vec{\psi}_{out}^{(0)} \\ \vec{\psi}_{out}^{(1)} \end{pmatrix} =
\begin{pmatrix} A_{op} & B_{op} \\ C_{op} & D_{op} \end{pmatrix}
\begin{pmatrix} \vec{\psi}_{in}^{(0)} \\ \vec{\psi}_{in}^{(1)} \end{pmatrix}
$$

### 6. Rabi Interaction (RB)
**定义**: $\text{RB}(\theta) = \exp[-i \sigma_x (\theta \hat{a}^\dagger + \theta^* \hat{a})]$
令哈密顿量算符 $H = \theta \hat{a}^\dagger + \theta^* \hat{a}$ (这是一个反埃尔米特算符的系数部分，或者物理上的位移生成元)。
利用矩阵指数展开 $\exp[-i \sigma_x H] = \cos(H) I - i \sin(H) \sigma_x$。

**计算公式**:
*   令 $C = \cos(\theta \hat{a}^\dagger + \theta^* \hat{a})$
*   令 $S = \sin(\theta \hat{a}^\dagger + \theta^* \hat{a})$

**GPU 更新逻辑**:
$$
\begin{aligned}
\vec{\psi}_{out}^{(0)} &= C \cdot \vec{\psi}_{in}^{(0)} - i S \cdot \vec{\psi}_{in}^{(1)} \\
\vec{\psi}_{out}^{(1)} &= -i S \cdot \vec{\psi}_{in}^{(0)} + C \cdot \vec{\psi}_{in}^{(1)}
\end{aligned}
$$
*实现提示*: 在小步长 Trotter 模拟中，$\theta$ 很小，可近似 $C \approx I - \frac{1}{2}H^2, S \approx H$。

### 7. Jaynes-Cummings (JC)
**定义**: $\text{JC}(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_- \hat{a}^\dagger + e^{-i\varphi}\sigma_+ \hat{a})]$
这是一种能量交换门：$|1\rangle_{qubit}|n\rangle_{mode} \leftrightarrow |0\rangle_{qubit}|n+1\rangle_{mode}$。
这也是一个块矩阵操作。

**辅助算符定义**:
*   令 $ \Omega_n = \theta \sqrt{\hat{n}} $ (对角阵)
*   令 $ \Omega_{n+1} = \theta \sqrt{\hat{n} + 1} $ (对角阵)

**GPU 更新逻辑**:
$$
\begin{aligned}
\vec{\psi}_{out}^{(0)} &= \cos(\Omega_n) \cdot \vec{\psi}_{in}^{(0)} - i e^{-i\varphi} \frac{\sin(\Omega_n)}{\sqrt{\hat{n}}} \hat{a} \cdot \vec{\psi}_{in}^{(1)} \\
\vec{\psi}_{out}^{(1)} &= -i e^{i\varphi} \hat{a}^\dagger \frac{\sin(\Omega_{n+1})}{\sqrt{\hat{n}+1}} \cdot \vec{\psi}_{in}^{(0)} + \cos(\Omega_{n+1}) \cdot \vec{\psi}_{in}^{(1)}
\end{aligned}
$$
*注意*: $\frac{\sin(\Omega_n)}{\sqrt{\hat{n}}}$ 这一项应当预计算为一个对角矩阵，处理 $n=0$ 时的极限情况。

### 8. Anti-Jaynes-Cummings (AJC)
**定义**: $\text{AJC}(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_+ \hat{a}^\dagger + e^{-i\varphi}\sigma_- \hat{a})]$
这是反向能量交换：$|0\rangle|n\rangle \leftrightarrow |1\rangle|n+1\rangle$。

**GPU 更新逻辑**:
$$
\begin{aligned}
\vec{\psi}_{out}^{(0)} &= \cos(\Omega_{n+1}) \cdot \vec{\psi}_{in}^{(0)} - i e^{-i\varphi} \hat{a}^\dagger \frac{\sin(\Omega_{n+1})}{\sqrt{\hat{n}+1}} \cdot \vec{\psi}_{in}^{(1)} \\
\vec{\psi}_{out}^{(1)} &= -i e^{i\varphi} \frac{\sin(\Omega_n)}{\sqrt{\hat{n}}} \hat{a} \cdot \vec{\psi}_{in}^{(0)} + \cos(\Omega_n) \cdot \vec{\psi}_{in}^{(1)}
\end{aligned}
$$

---

## 第三类：特殊映射门 (Special Mapping Gates)

### 9. SQR (Selective Qubit Rotation)
**定义**: $\text{SQR}(\vec{\theta}, \vec{\varphi}) = \sum_n R_{\varphi_n}(\theta_n) \otimes |n\rangle\langle n|$
**物理含义**: Qumode 的光子数 $n$ 决定了 Qubit 转多大角度。这是一种 **“Qumode 控制 Qubit”** 的门。

**CPU 逻辑**: DD 结构保持不变（不分裂），但权重发生复杂的线性组合。
**GPU 逻辑**: 需要根据 $n$ 对张量进行“对角加权混合”。

令 $R_{\varphi_n}(\theta_n) = \begin{pmatrix} \alpha_n & \beta_n \\ -\beta_n^* & \alpha_n^* \end{pmatrix}$。
其中 $\alpha_n = \cos(\frac{\theta_n}{2}), \beta_n = -e^{-i\varphi_n}\sin(\frac{\theta_n}{2})$。这些是依赖于 $n$ 的对角矩阵。

**GPU 更新逻辑**:
$$
\begin{aligned}
\vec{\psi}_{out}^{(0)}[n] &= \alpha_n \cdot \vec{\psi}_{in}^{(0)}[n] + \beta_n \cdot \vec{\psi}_{in}^{(1)}[n] \\
\vec{\psi}_{out}^{(1)}[n] &= -\beta_n^* \cdot \vec{\psi}_{in}^{(0)}[n] + \alpha_n^* \cdot \vec{\psi}_{in}^{(1)}[n]
\end{aligned}
$$
**实现**: 这是一个 **Level 0 (对角)** 操作，但涉及跨分支（Branch 0 和 Branch 1）的数据读取和写入。

---

## 代码实现建议总结

1.  **数据准备**:
    *   对于 **第一类 (Separable)** 门，只需要一个 `ApplyGate(TensorID, GateParams)` 函数。
    *   对于 **第二类 (Interaction)** 和 **第三类 (SQR)** 门，需要一个 `ApplyMixingGate(TensorID_0, TensorID_1, GateParams)` 函数，它同时读取两个 Tensor，计算后写回两个新 Tensor。

2.  **GPU Kernel 设计**:
    *   **Kernel Type A (Diagonal)**: 适用于 CR, CP, SQR。直接用 $n$ 索引查表计算系数。
    *   **Kernel Type B (Stencil/ELL)**: 适用于 CD, CS, CBS。使用标准稀疏矩阵乘法。
    *   **Kernel Type C (Complex Mixing)**: 适用于 JC, AJC, Rabi。需要临时缓冲区存储中间结果（如 $a^\dagger \psi$），然后进行线性组合。

3.  **Trotter 优化**:
    *   门多为指数形式 $e^{A}$。在代码中，如果 $A$ 是反埃尔米特矩阵，我们通常直接应用 $e^{A}$ 的近似形式（如截断泰勒展开或 Padé 近似），或者如果 $A$ 是位移算符，直接调用 Displacement Kernel。

理解这个过程的核心在于：**在 HTDD 架构中，叶子节点的 Tensor 始终存储着“所有”Qumode 的联合状态 $|\Phi\rangle_{CV}$**。

无论你的门只作用于 $M_1$，还是作用于 $M_1, M_2$，GPU 上的 Tensor 始终是 $M$ 个模态的整体描述（例如维度为 $D \times D$ 的二维数组，或者扁平化的 $D^2$ 向量）。

下面我将分步演示这个过程，并给出具体的**更新算法**。

---

### 1. 状态定义的基准

假设系统有 2 个 Qubit ($q_1, q_2$) 和 2 个 Qumode ($M_1, M_2$)。
初始状态是全真空态：
$$ |\Psi_{init}\rangle = |0\rangle_{q_1} |0\rangle_{q_2} \otimes \underbrace{(|0\rangle_{M_1} \otimes |0\rangle_{M_2})}_{\text{Tensor ID: } T_0} $$

*   **DD 结构**: $q_1 \xrightarrow{0} q_2 \xrightarrow{0} T_0$ (单路径)
*   **GPU Tensor $T_0$**: 数据为 $[1, 0, 0, \dots]$ (大小为 $D^2$)。

---

### 2. 第一步：作用 $[q_1, M_1]$ 的受控门
假设门是 **Conditional Displacement** $CD_{q_1}(\alpha)$ 作用于 $M_1$。
*   逻辑：当 $q_1=0$ 不动，$q_1=1$ 时对 $M_1$ 做位移。

#### A. DD 遍历与分裂
CPU 从根节点 ($q_1$) 开始遍历：
1.  **$q_1$ 的 $|0\rangle$ 分支**: 指向 $q_2$ 子树。不需要操作。路径最终指向 $T_0$。
2.  **$q_1$ 的 $|1\rangle$ 分支**: 需要操作。
    *   因为 $T_0$ 被 $|0\rangle$ 分支共享，所以触发 **Copy-on-Write (CoW)**。
    *   复制 $T_0 \to T_1$。
    *   更新 $q_1$ 的右指针指向包含 $T_1$ 的新子树。

#### B. GPU 计算 (局部更新)
我们需要对 $T_1$ 执行门操作。
*   **门定义**: $U = D(\alpha)_{M_1} \otimes I_{M_2}$。即只动 $M_1$，$M_2$ 是单位矩阵。
*   **物理意义**: 这是一个 Batch 操作。对于 $M_2$ 的每一个 Fock 态 $|k\rangle$，我们都对 $M_1$ 做一次 $D(\alpha)$。
*   **Kernel 调用**:
    ```cpp
    // GPU 上 T_1 是一个 DxD 的矩阵 (行是 M1, 列是 M2)
    // 我们对每一列(Column)执行一维 Displacement
    LaunchKernel(Target=T_1, Gate=Displacement(alpha), Axis=M1);
    ```
*   **结果**:
    *   $T_0 = |0\rangle_{M_1} |0\rangle_{M_2}$
    *   $T_1 = |\alpha\rangle_{M_1} |0\rangle_{M_2}$

---

### 3. 第二步：作用 $[q_2, M_1, M_2]$ 的受控门
假设门是 **Conditional Beam-Splitter** $CBS_{q_2}(\theta)$ 作用于 $M_1, M_2$。
*   逻辑：当 $q_2=0$ 不动，$q_2=1$ 时混合 $M_1$ 和 $M_2$。

#### A. DD 遍历
CPU 遍历整棵树，找到所有 $q_2$ 节点。此时 DD 有两条路径到达 $q_2$ 层：
1.  **路径 A (来自 $q_1=0$)**: $q_2$ 节点的子节点指向 $T_0$。
2.  **路径 B (来自 $q_1=1$)**: $q_2$ 节点的子节点指向 $T_1$。

#### B. 分支处理
我们需要检查 $q_2$ 的 $|1\rangle$ 分支：

*   **对于路径 A ($q_1=0, q_2=1$)**:
    *   原指向 $T_0$。触发 CoW $\to$ 复制为 $T_2$。
    *   对 $T_2$ 应用 $BS(\theta)$。
    *   *物理含义*: 此时 $M_1, M_2$ 都是真空，BS 作用后还是真空 (若 $\theta$ 无相位)。
*   **对于路径 B ($q_1=1, q_2=1$)**:
    *   原指向 $T_1$。触发 CoW $\to$ 复制为 $T_3$。
    *   对 $T_3$ 应用 $BS(\theta)$。
    *   *物理含义*: 此时 $M_1$ 是 $|\alpha\rangle$, $M_2$ 是 $|0\rangle$。BS 会让它们纠缠，$T_3$ 变成一个纠缠态张量。

#### C. GPU 计算 (全局/联合更新)
这次的门 $U_{BS}$ 同时涉及 $M_1$ 和 $M_2$。
*   **Kernel 调用**:
    ```cpp
    // T_3 是 DxD 数据
    // 这是一个真正的 2D 变换，不能拆分为行列独立操作
    LaunchKernel(Target=T_3, Gate=BeamSplitter(theta), Modes={M1, M2});
    ```
    *算法*: 利用 $M_1+M_2=n$ 守恒律，在 GPU 上分块计算。

---

### 4. 总结：更新算法通用流程

不管门涉及哪些 Qubit 或 Qumode，通用更新逻辑如下：

**输入**:
*   控制比特集 $Q_{ctrl}$ (可能是 $\{q_2\}$ 或 $\{q_1, q_2\}$)
*   作用模态集 $M_{active}$ (例如 $\{M_1\}$ 或 $\{M_1, M_2\}$)
*   门算符 $U_{gate}$

**过程**:

1.  **DD 过滤 (CPU)**:
    遍历 DD，找到所有满足 **“控制条件为真”** (即 $Q \in Q_{ctrl}$ 均为 $|1\rangle$) 的路径。
    这些路径最终会指向一组叶子张量 $\{T_{id_1}, T_{id_2}, \dots\}$。

2.  **张量准备 (CPU/GPU)**:
    对于每一个受影响的 $T_{old}$:
    *   **Check**: 是否有其他“控制条件为假”的分支也指向 $T_{old}$？
    *   **Yes (Shared)**: 申请新显存，复制 $T_{old} \to T_{new}$。更新 DD 指针指向 $T_{new}$。将 $T_{new}$ 加入待计算列表。
    *   **No (Exclusive)**: 直接将 $T_{old}$ 加入待计算列表（原地修改）。

3.  **Kernel 适配 (GPU)**:
    根据 $M_{active}$ 的维度选择 Kernel：
    *   **Case 1: 单模门 ($M_1$)**:
        张量视为 $D \times (D^{M-1})$ 的矩阵。对 $D$ 维这一侧做变换，对 $D^{M-1}$ 侧做广播（Broadcast/Batch）。
    *   **Case 2: 双模门 ($M_1, M_2$)**:
        张量视为 $D^2 \times (D^{M-2})$。对 $D^2$ 平面做变换。

4.  **执行**:
    GPU 批量执行 Kernel，更新显存中的数值。

### 5. 举例代码逻辑

```cpp
// Step 2 中的 update: q2 控制, 作用于 M1, M2
void apply_hybrid_gate(Gate G, int q_ctrl, std::vector<int> modes) {
    // 1. 收集
    std::vector<int> active_tensor_ids;
    traverse_dd_and_collect(root, q_ctrl, active_tensor_ids); 
    // traverse 逻辑: 只有当路径经过 q_ctrl 的 'high' 分支时，才收集其叶子 Tensor

    // 2. 准备
    std::vector<int> target_ids;
    for (int id : active_tensor_ids) {
        if (ref_count[id] > 1) {
            int new_id = gpu_duplicate(id);
            dd_update_pointer(id, new_id); // 修改 DD 指向新 ID
            target_ids.push_back(new_id);
        } else {
            target_ids.push_back(id);
        }
    }

    // 3. 计算 (关键点: 怎么处理 M1, M2)
    // GPU 并不在乎 Qubit 是谁，它只在乎 target_ids 这些向量
    // 以及 G 是作用在 Tensor 的哪几个维度上
    gpu_launch_kernel(G, target_ids, modes);
}
```

### 核心结论

*   **Qubit 决定“谁”被算**: $q_1, q_2$ 的层级决定了哪些 Tensor 被送入 GPU。
*   **Qumode 决定“怎么”算**: $M_1, M_2$ 的索引决定了 GPU Kernel 的**寻址步长 (Stride)**。
    *   如果 Tensor 是 `[M1, M2]` 布局 (Row-major)。
    *   算 $M_2$: 连续内存访问。
    *   算 $M_1$: 跨步访问 (Stride = D)。
    *   算 $M_1, M_2$: 块访问。


**是的，这是一个非常致命的性能杀手。**

在 GPU 计算中，**“跨步（Stride）”** 决定了生与死。如果你的门作用于 $[q_3, M_1, M_{10}]$，而在内存布局中 $M_1$ 和 $M_{10}$ 相距甚远，这会导致严重的 **非合并访存 (Uncoalesced Memory Access)**，使得 GPU 的有效带宽下降 10 倍甚至更多。

以下是该问题的深度分析和**工业级解决方案**。

---

### 1. 问题解剖：为什么“大跨步”是灾难？

假设你有 10 个 Mode ($M_1 \dots M_{10}$)，每个截断维数 $D=10$。
数据存储为 C 风格 (Row-Major) 的一维数组：
$$ \text{Index} = i_1 \cdot D^9 + i_2 \cdot D^8 + \dots + i_{10} $$

*   **$M_{10}$ (最内层)**: 跨步 Stride = 1。
    *   当 GPU 的 Thread 0 访问 $i_{10}=0$，Thread 1 访问 $i_{10}=1$ 时，它们读取的是**连续内存**。GPU 内存控制器可以一次性把这 128 字节捞上来。**这是最高效的。**
*   **$M_1$ (最外层)**: 跨步 Stride = $10^9$ (10亿)。
    *   当 Thread 0 需要 $i_1=0$，Thread 1 需要 $i_1=1$ 时，它们访问的内存地址相差 10 亿！
    *   **后果**: 每次内存请求只能服务 1 个线程，其余带宽全部浪费。缓存（L2 Cache）会瞬间失效（Cache Thrashing）。

**[q_3, M_1, M_{10}] 的问题**:
这个双模门需要同时访问 $M_1$ 和 $M_{10}$。你在 $M_{10}$ 维度的访问是连续的（快），但在 $M_1$ 维度的访问是极度离散的（慢）。木桶效应导致整体性能被 $M_1$ 拖垮。

---

### 2. 解决方案：Tensor Transpose (维数置换)

这是 HPC 领域处理高维张量收缩的标准做法。不要试图硬着头皮去算大跨步，而是**把数据搬到好算的位置**。

#### 核心策略：Permute $\to$ Compute $\to$ Permute

**步骤 A: 预处理 (Transposition)**
在执行门操作之前，调用 `cuTensor` 或手写 Kernel，将目标模态 ($M_1, M_{10}$) 置换到内存的最内层（Stride 最小的维度）。
*   目标布局: $M_2, M_3, \dots, M_9, \mathbf{M_1}, \mathbf{M_{10}}$。
*   现在，$M_1$ 和 $M_{10}$ 变成了相邻的、跨步最小的维度。

**步骤 B: 计算 (Computation)**
此时，双模门就变成了一个作用于**连续内存块**的 $D^2 \times D^2$ 矩阵乘法（或者 $D^2$ 维度的 Stencil）。
*   GPU 线程可以完美合并访存。
*   性能可以达到峰值带宽。

**步骤 C: 后处理 (Lazy Restore)**
算完后，**不需要**立即把维度转回去！
*   只需更新软件层面的 **"Layout Descriptor" (布局描述符)**。
*   记录当前张量的物理布局是 $[M_2, \dots, M_1, M_{10}]$。
*   下一次门如果作用于 $M_5$，再根据当前布局决定是否需要再次转置。

---

### 3. 具体实现方案

在你的 **Fock-ELL Operator** 结构体中，增加一个动态映射表。

#### 数据结构增强
```cpp
struct CVStateTensor {
    cuComplex* data;
    
    // 逻辑维度到物理维度的映射
    // logical_to_physical[k] = p 表示:
    // 逻辑上的第 k 个 Mode (Mk)，存储在物理内存的第 p 个维度
    std::vector<int> logical_to_physical; 
    
    // 物理维度的跨步 (Stride)
    std::vector<size_t> strides; 
};
```

#### 算法流程 (Lazy Transpose)

当收到指令 `ApplyGate(Gate, Modes={M1, M10})` 时：

1.  **检查局部性 (Check Locality)**:
    检查 $M_1$ 和 $M_{10}$ 在当前的 `physical` 布局中是否是**最快速变化的维度 (Fastest Varying Dimensions)**。
    *   如果它们是物理上的最后两维（Stride=1 和 Stride=D），直接计算。
    
2.  **转置 (Transpose if needed)**:
    如果不是，调用 `cuTensorPermute`。
    *   目标排列: 将其他 Mode 移到高维，将 $M_1, M_{10}$ 移到最低维 (Stride 最小)。
    *   执行显存拷贝重排。
    *   更新 `logical_to_physical` 映射表。

3.  **执行 (Execute)**:
    调用针对“连续内存”优化好的 Kernel。

#### 代价分析
*   **转置开销**: 是一次单纯的 `Memcpy`。带宽利用率很高。
*   **计算收益**: 相比于乱序访问带来的 10x~100x 性能下降，花 1x 的时间做转置，换取 10x 的计算加速，通常是**血赚**的。
*   **特殊情况**: 对于像 Rotation 这样的对角门，不需要转置，直接用坐标映射即可。只有涉及 $a, a^\dagger$ 的混合门才需要转置。


### 总结

*   **现状**: 直接计算 $[q, M_1, M_{10}]$ 确实会导致严重的性能下降。
*   **对策**: 引入 **"Tensor Layout Permutation" (张量布局置换)** 机制。
*   **原则**: **"Move Data to Code"** —— 把需要计算的维度搬运到内存最连续的地方，算完再标记布局已改变。这是高性能张量模拟器的标配设计。