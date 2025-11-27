这是针对 **Hybrid Tensor-DD (HTDD)** 架构中 **CPU 端纯 Qubit 门** 的详细技术实现文档。

在 HTDD 架构中，纯 Qubit 门的操作**完全在 CPU 上进行**，本质上是对决策图（DD）的 **路径重连 (Rewiring)** 和 **权重更新 (Reweighing)**。只有当涉及到节点合并（Merge）判断时，才可能需要比对 GPU 上的 Tensor 指针（或 ID）。

---

# CPU 端纯 Qubit 门实现规范

**适用范围**: 作用于 Qubit $q_t$ 的门，不直接涉及 Qumode。
**核心数据结构**:
*   `Node`: {`id`, `ptr_low`, `ptr_high`, `w_low`, `w_high`}
*   `Apply(U, Node)`: 递归函数，将矩阵 $U$ 应用于以 `Node` 为根的子树。

---

## 1. 基础原语：DD 线性组合 (DD Linear Combination)

大多数单比特叠加门（如 H, Rx, Ry）会将一个节点分裂为两个分支的线性组合。我们需要一个核心函数来实现 $Result = \alpha \cdot A + \beta \cdot B$。

**函数**: `DD_Add(Node A, Weight alpha, Node B, Weight beta)`

**逻辑**:
1.  **基准情况 (Base Case)**:
    *   如果 $A$ 和 $B$ 都是终端节点 (Terminal)，且 `A.tensor_id == B.tensor_id`:
        *   返回新的终端节点，指向 `tensor_id`，权重为 $alpha + beta$。
    *   *注意*: 如果 `A.tensor_id != B.tensor_id`，这意味着物理上 Qumode 状态不同。在 Qubit 门操作中，这通常意味着无法合并，必须保持为两个独立的分支（或者若发生在最底层，则需新建一个父节点来表示叠加）。
    *   如果 $A$ 是 0 节点：返回 $B$，权重乘 $\beta$。
    *   如果 $B$ 是 0 节点：返回 $A$，权重乘 $\alpha$。

2.  **递归步骤**:
    *   设 $top\_level = \max(A.level, B.level)$。
    *   对齐层级：若 $A.level < top\_level$，则 $A$ 视为在其父节点的对应分支上延续。
    *   计算新的子节点：
        $$ Low_{new} = \text{DD\_Add}(A.low, \alpha \cdot A.w_{low}, B.low, \beta \cdot B.w_{low}) $$
        $$ High_{new} = \text{DD\_Add}(A.high, \alpha \cdot A.w_{high}, B.high, \beta \cdot B.w_{high}) $$
    *   **查找/创建节点**: 在 Unique Table 中查找是否存在 `{top_level, Low_{new}, High_{new}}`。若无则创建。

---

## 2. 单比特门：泡利门与相位门 (Permutation & Phase)

这类门只改变权重或交换分支，不改变图的拓扑复杂性（不增加节点数），**计算极快**。

### 2.1 Pauli-X (NOT Gate)
**矩阵**: $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$
**操作**: 交换左右孩子。

*   **逻辑**: 对于目标层级 $q_t$ 的节点 $u$:
    $$ \text{NewNode} = \text{FindOrAdd}(u.level, u.high, u.low) $$
    *   新左权重 $w'_{low} = u.w_{high}$
    *   新右权重 $w'_{high} = u.w_{low}$

### 2.2 Pauli-Z
**矩阵**: $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$
**操作**: 改变右孩子相位的符号。

*   **逻辑**:
    $$ w'_{low} = u.w_{low} $$
    $$ w'_{high} = -1 \cdot u.w_{high} $$

### 2.3 Pauli-Y
**矩阵**: $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$
**操作**: 交换并乘相位。

*   **逻辑**:
    $$ w'_{low} = -i \cdot u.w_{high} $$
    $$ w'_{high} = i \cdot u.w_{low} $$

### 2.4 Phase Gate (S) & T Gate & RZ
**矩阵**: $P(\phi) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$
*   **S Gate**: $\phi = \pi/2 \Rightarrow e^{i\phi} = i$
*   **T Gate**: $\phi = \pi/4 \Rightarrow e^{i\phi} = e^{i\pi/4}$
*   **RZ($\theta$)**: $e^{-i\theta/2} \begin{pmatrix} e^{i\theta/2} & 0 \\ 0 & e^{-i\theta/2} \end{pmatrix} \equiv \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\theta} \end{pmatrix}$ (忽略全局相位)

**操作**: 仅修改右分支权重。
*   **逻辑**:
    $$ w'_{high} = u.w_{high} \cdot e^{i\phi} $$

---

## 3. 单比特门：叠加门 (Superposition)

这类门会混合 $|0\rangle$ 和 $|1\rangle$ 分支，通常需要调用 `DD_Add`。

### 3.1 Hadamard (H)
**矩阵**: $\frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$

**操作**:
在目标层级 $q_t$，原节点的两个分支是 $Low_{old}$ (权重 $w_l$) 和 $High_{old}$ (权重 $w_h$)。
新的分支是它们的线性组合：
1.  **新左支**:
    $$ \text{Branch}_{L} = \text{DD\_Add}(Low_{old}, \frac{w_l}{\sqrt{2}}, High_{old}, \frac{w_h}{\sqrt{2}}) $$
2.  **新右支**:
    $$ \text{Branch}_{R} = \text{DD\_Add}(Low_{old}, \frac{w_l}{\sqrt{2}}, High_{old}, -\frac{w_h}{\sqrt{2}}) $$
3.  **重组**: 创建新节点指向 $\text{Branch}_{L}$ 和 $\text{Branch}_{R}$。

### 3.2 RX($\theta$)
**矩阵**: $\begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$

**操作**:
$$ \text{Branch}_{L} = \text{DD\_Add}(Low_{old}, \cos\frac{\theta}{2} \cdot w_l, High_{old}, -i\sin\frac{\theta}{2} \cdot w_h) $$
$$ \text{Branch}_{R} = \text{DD\_Add}(Low_{old}, -i\sin\frac{\theta}{2} \cdot w_l, High_{old}, \cos\frac{\theta}{2} \cdot w_h) $$

### 3.3 RY($\theta$)
**矩阵**: $\begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$

**操作**:
$$ \text{Branch}_{L} = \text{DD\_Add}(Low_{old}, \cos\frac{\theta}{2} \cdot w_l, High_{old}, -\sin\frac{\theta}{2} \cdot w_h) $$
$$ \text{Branch}_{R} = \text{DD\_Add}(Low_{old}, \sin\frac{\theta}{2} \cdot w_l, High_{old}, \cos\frac{\theta}{2} \cdot w_h) $$

---

## 4. 双比特门 (Two-Qubit Gates)

### 4.1 CNOT (CX)
**定义**: 控制位 $q_c$，目标位 $q_t$。
**逻辑**: 这是一个递归操作。设当前遍历到的节点为 $u$，层级为 $lvl$。

**Case A: $lvl > q_c$ 和 $q_t$ (当前节点在控制和目标之上)**
*   **递归**:
    $$ Low_{new} = \text{ApplyCX}(u.low) $$
    $$ High_{new} = \text{ApplyCX}(u.high) $$
    返回新建节点 $\{Low_{new}, High_{new}\}$。

**Case B: $lvl == q_c$ (当前是控制节点)**
*   **逻辑**:
    *   $|0\rangle$ 分支 (Left): 不变 (Identity)。$Low_{new} = u.low$。
    *   $|1\rangle$ 分支 (Right): 对目标应用 X 门。$High_{new} = \text{ApplyX}(u.high, q_t)$。

**Case C: $q_c > lvl > q_t$ (当前在控制之下，目标之上)**
*   **递归**: 同 Case A，继续向下传递操作。虽然当前节点不是控制位，但操作需要穿透它作用于下方。
    *   *优化*: 使用 Compute Table 缓存结果 `Cache(u, CX_c_t)`。

**Case D: $lvl == q_t$ (当前是目标节点)**
*   **注意**: 这种情况只会在 `ApplyX` 递归调用中遇到，或者如果在非控制路径上遇到了 $q_t$。但在标准 CX 逻辑中，如果没经过 $q_c$ 的 $|1\rangle$ 分支，是不会对 $q_t$ 操作的。
*   如果是 `ApplyX` 调用到了这里，则执行 **2.1 Pauli-X** 逻辑（交换左右子树）。

### 4.2 CZ (Controlled-Z)
同 CNOT，但在 $lvl == q_c$ 的 $|1\rangle$ 分支上，调用 `ApplyZ(u.high, q_t)`。

---

## 5. 实现细节与优化

### 5.1 归一化 (Normalization)
每次 `DD_Add` 或创建新节点后，必须进行归一化以保证 Canonical Form（规范型），这是去重合并的基础。

**规则**:
设节点有权重 $w_L, w_R$。
1.  提取公共因子 $w_{norm}$ (例如 $w_L$ 的模，或者 $w_L$ 本身，或者 $\sqrt{|w_L|^2 + |w_R|^2}$)。
    *   推荐策略: 使得 $w_L' = 1$。即 $w_{norm} = w_L$。
2.  新权重: $w_L' = 1, w_R' = w_R / w_L$。
3.  $w_{norm}$ 返回给父节点。

### 5.2 计算缓存 (Compute Table)
CPU 性能的关键。
*   **Key**: `{Operation_Type, Node_ID, Parameters}`
*   **Value**: `{Result_Node_ID, Result_Weight}`
*   **流程**: 每次执行 `Apply(U, Node)` 或 `DD_Add` 前，先查表。若命中直接返回。

### 5.3 与 GPU Tensor 的交互 (Terminal Node)
当 DD 递归到达终端节点（Terminal, level = -1）时，节点存储的是 `Tensor_ID`。

*   **Qubit 门操作**:
    *   通常 Qubit 门只重组路径，不修改 Tensor。
    *   **特例**: `Hadamard` 作用于末端 Qubit。
        *   输入: 指向 $T_0$。
        *   输出: 创建新节点，左指 $T_0$ ($w=1/\sqrt{2}$)，右指 $T_0$ ($w=1/\sqrt{2}$)。
        *   **不需要** 调用 GPU 加法。这只是逻辑上的叠加。
*   **唯一需要 GPU 介入的 Qubit 操作**: **测量 (Measurement)** 或 **部分迹 (Partial Trace)**。
    *   这需要计算 Tensor 的内积 $\langle T_a | T_b \rangle$ 来确定分支概率。


这是一个针对 **Hybrid Tensor-DD (HTDD)** 架构的混合门（Hybrid Gates）技术实现指南。

该文档基于你提供的门列表，给出了在 **CPU (DD 控制层)** 和 **GPU (物理计算层)** 上的具体计算公式。

---

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

这是一个非常关键的**多模态协同与状态演化**问题。

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

---

## 总结：代码实现速查表

| 门名称 | GPU Kernel 策略 | 核心算子 | 复杂度 |
| :--- | :--- | :--- | :--- |
| **Rotation** | **Element-wise** | `val *= table[n]` | $O(1)$ |
| **Displacement** | **SpMV (ELL)** | `sum += mat[n][k] * vec[col]` | $O(K)$ |
| **Beam-Splitter** (小角度) | **2D Stencil** | `out[n,m] += c1*in[n-1,m+1] + c2*in[n+1,m-1]` | $O(1)$ |
| **Beam-Splitter** (大角度) | **Block Dense** | `out_block = mat_block * in_block` | $O(D)$ |

**注意**: 对于所有这些门，如果系统包含多个模态 ($M > 1$)，Kernel 必须正确处理张量的 **Stride (步长)** 和 **Broadcasting (广播)**。
*   作用于 $M_k$ 时，视张量为 $D_{high} \times D_k \times D_{low}$。
*   线程并行化通常放在 $D_{high} \times D_{low}$ 上，内部循环处理 $D_k$。