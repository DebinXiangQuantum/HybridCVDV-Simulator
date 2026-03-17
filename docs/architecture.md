# Technical Specification: Hybrid CPU-GPU DD-CV Simulator Architecture

## 1. Architectural Overview: The Dual-Track Engine

The Hybrid Tensor-Decision Diagram (HTDD) simulator employs a strict **Control-Physical Separation** architecture. This design optimally resolves the "state space explosion" inherent in mixed continuous-variable (CV) and discrete-variable (DV) quantum systems.

### 1.1 Control-Physical Separation
*   **CPU Layer (Control Layer):** Manages the **Hybrid Decision Diagram (HDD)**. It abstracts the discrete Hilbert space (Qubits) into a directed acyclic graph. It is responsible for routing, branching, garbage collection, and dynamic graph rewiring without computing the physical tensor data directly.
*   **GPU Layer (Physical Layer):** Executes the heavy linear algebra and tensor contractions for the Continuous Variables (Qumodes).

### 1.2 The Dual-Track Execution Engine
To maintain high performance across both Gaussian and Non-Gaussian regimes, the GPU layer implements an **Execution Decision Engine (EDE)** that dynamically routes CV execution between two tracks:
*   **Symbolic Execution Track (SET):** Represents Gaussian states via highly compact Symplectic matrices and displacement vectors. Used for highly parallel, memory-efficient simulation of Gaussian regions.
*   **Tensor Execution Track (TET):** Represents states as dense or sparse tensors in the truncated Fock space (dimension $D$). Used for non-Gaussian operations or regions of massive entanglement where symbolic tracking breaks down.

---

## 2. CPU Layer: DD Graph Updates and Pure Qubit Gates

All pure Qubit operations and routing for hybrid gates are resolved on the CPU by rewriting paths (Rewiring) and updating edge weights (Reweighing) within the HDD. The fundamental object is an HDD Node containing left ($|0\rangle$) and right ($|1\rangle$) branches, along with corresponding complex weights ($w_{low}$, $w_{high}$).

### 2.1 The DD Addition Primitive
For gates that mix branches, the CPU employs a fundamental DD Addition primitive to merge branches and prevent duplicate subgraph creation via a Unique Table:
$$ \text{Result\_Node} = \text{DD\_Add}(\text{Node}_A, \alpha, \text{Node}_B, \beta) $$

### 2.2 Pure Qubit Gate Updates

#### 2.2.1 Permutation and Phase Gates
Gates that do not increase topological complexity modify weights or swap pointers.
*   **Pauli-X (NOT Gate):** Swaps the branches of the target node.
    $$ w'_{low} = w_{high}, \quad w'_{high} = w_{low} $$
*   **Pauli-Z:** Flips the sign of the $|1\rangle$ branch.
    $$ w'_{low} = w_{low}, \quad w'_{high} = -w_{high} $$
*   **Pauli-Y:** Swaps branches and applies phase.
    $$ w'_{low} = -i \cdot w_{high}, \quad w'_{high} = i \cdot w_{low} $$
*   **Phase Gates (S, T, RZ($\theta$)):** Introduces a phase factor $e^{i\phi}$ strictly to the right ($|1\rangle$) branch.
    $$ w'_{low} = w_{low}, \quad w'_{high} = w_{high} \cdot e^{i\phi} $$

#### 2.2.2 Superposition Gates
Gates that create a linear combination of the $|0\rangle$ and $|1\rangle$ basis.
*   **Hadamard (H):** Branches are split and recombined using the `DD_Add` primitive.
    $$ \text{Branch}_{L\_new} = \text{DD\_Add}\left(Low, \frac{w_{low}}{\sqrt{2}}, High, \frac{w_{high}}{\sqrt{2}}\right) $$
    $$ \text{Branch}_{R\_new} = \text{DD\_Add}\left(Low, \frac{w_{low}}{\sqrt{2}}, High, -\frac{w_{high}}{\sqrt{2}}\right) $$
*   **RX($\theta$):**
    $$ \text{Branch}_{L} = \text{DD\_Add}\left(Low, \cos\frac{\theta}{2} \cdot w_{low}, High, -i\sin\frac{\theta}{2} \cdot w_{high}\right) $$
    $$ \text{Branch}_{R} = \text{DD\_Add}\left(Low, -i\sin\frac{\theta}{2} \cdot w_{low}, High, \cos\frac{\theta}{2} \cdot w_{high}\right) $$
*   **RY($\theta$):**
    $$ \text{Branch}_{L} = \text{DD\_Add}\left(Low, \cos\frac{\theta}{2} \cdot w_{low}, High, -\sin\frac{\theta}{2} \cdot w_{high}\right) $$
    $$ \text{Branch}_{R} = \text{DD\_Add}\left(Low, \sin\frac{\theta}{2} \cdot w_{low}, High, \cos\frac{\theta}{2} \cdot w_{high}\right) $$

#### 2.2.3 Two-Qubit Gates
*   **CNOT (CX):** Traverses recursively. When the control qubit $q_c$ is reached, the $|0\rangle$ branch remains identical, while the $|1\rangle$ branch applies a Pauli-X operation recursively down to the target qubit $q_t$.
*   **CZ (Controlled-Z):** Similar to CNOT, but applies a Pauli-Z to the target qubit $q_t$ on the $|1\rangle$ branch of the control.

---

## 3. Hybrid Controlled Gates (Qubit-Qumode Interaction)

When a gate spans both a Qubit $Q_k$ (Control) and Qumode(s) $M_i$ (Target), the CPU traverses to the level of $Q_k$ and dispatches targeted instructions to the GPU based on the gate's physical properties.

### 3.1 Separable Controlled Gates
These gates do not mix the Qubit basis ($|0\rangle$ and $|1\rangle$ branches remain independent), but they alter the Qumode state conditionally. The right branch ($|1\rangle$) typically triggers a **Copy-on-Write (CoW)** if the leaf tensor is shared.

*   **Conditional Rotation (CR) & Parity (CP):** $CR(\theta) = \exp[-i \frac{\theta}{2} \sigma_z \hat{n}]$
    $$ \text{Branch 0 } (|0\rangle): \vec{\psi}_{out}^{(0)}[n] = \vec{\psi}_{in}^{(0)}[n] \cdot \exp\left(-i \frac{\theta}{2} n\right) $$
    $$ \text{Branch 1 } (|1\rangle): \vec{\psi}_{out}^{(1)}[n] = \vec{\psi}_{in}^{(1)}[n] \cdot \exp\left(+i \frac{\theta}{2} n\right) $$
*   **Conditional Displacement (CD):** $CD(\alpha) = \exp[\sigma_z (\alpha \hat{a}^\dagger - \alpha^* \hat{a})]$
    $$ \text{Branch 0 } (|0\rangle): \vec{\psi}_{out}^{(0)} = D(+\alpha) \cdot \vec{\psi}_{in}^{(0)} $$
    $$ \text{Branch 1 } (|1\rangle): \vec{\psi}_{out}^{(1)} = D(-\alpha) \cdot \vec{\psi}_{in}^{(1)} $$
*   **Conditional Squeezing (CS):** $CS(\zeta) = \exp[\frac{1}{2}\sigma_z (\zeta^* \hat{a}^2 - \zeta \hat{a}^{\dagger 2})]$
    $$ \text{Branch 0 } (|0\rangle): \vec{\psi}_{out}^{(0)} = S(+\zeta) \cdot \vec{\psi}_{in}^{(0)} $$
    $$ \text{Branch 1 } (|1\rangle): \vec{\psi}_{out}^{(1)} = S(-\zeta) \cdot \vec{\psi}_{in}^{(1)} $$
*   **Conditional Beam-Splitter (CBS):** $CBS(\theta, \varphi) = \exp[-i \frac{\theta}{2} \sigma_z (e^{i\varphi}\hat{a}^\dagger \hat{b} + e^{-i\varphi}\hat{a}\hat{b}^\dagger)]$
    $$ \text{Branch 0 } (|0\rangle): \vec{\psi}_{out}^{(0)} = BS(+\theta, \varphi) \cdot \vec{\psi}_{in}^{(0)} $$
    $$ \text{Branch 1 } (|1\rangle): \vec{\psi}_{out}^{(1)} = BS(-\theta, \varphi) \cdot \vec{\psi}_{in}^{(1)} $$
*   **Conditional Two-Mode Squeezing (CTMS) & SUM:** Analogous logic, applying the positive parameter to Branch 0 and the negative parameter to Branch 1.

### 3.2 Qubit-Mixing Interaction Gates
These gates contain $\sigma_x$ or $\sigma_\pm$ operators, physically entangling the Qubit and the Qumode, mixing both branches. The CPU instructs the GPU to perform a cross-branch linear combination.

*   **Rabi Interaction (RB):** $RB(\theta) = \exp[-i \sigma_x (\theta \hat{a}^\dagger + \theta^* \hat{a})]$
    Let $C = \cos(\theta \hat{a}^\dagger + \theta^* \hat{a})$ and $S = \sin(\theta \hat{a}^\dagger + \theta^* \hat{a})$.
    $$ \vec{\psi}_{out}^{(0)} = C \cdot \vec{\psi}_{in}^{(0)} - i S \cdot \vec{\psi}_{in}^{(1)} $$
    $$ \vec{\psi}_{out}^{(1)} = -i S \cdot \vec{\psi}_{in}^{(0)} + C \cdot \vec{\psi}_{in}^{(1)} $$
*   **Jaynes-Cummings (JC):** $JC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_- \hat{a}^\dagger + e^{-i\varphi}\sigma_+ \hat{a})]$
    Let $\Omega_n = \theta \sqrt{\hat{n}}$ and $\Omega_{n+1} = \theta \sqrt{\hat{n} + 1}$.
    $$ \vec{\psi}_{out}^{(0)} = \cos(\Omega_n) \cdot \vec{\psi}_{in}^{(0)} - i e^{-i\varphi} \frac{\sin(\Omega_n)}{\sqrt{\hat{n}}} \hat{a} \cdot \vec{\psi}_{in}^{(1)} $$
    $$ \vec{\psi}_{out}^{(1)} = -i e^{i\varphi} \hat{a}^\dagger \frac{\sin(\Omega_{n+1})}{\sqrt{\hat{n}+1}} \cdot \vec{\psi}_{in}^{(0)} + \cos(\Omega_{n+1}) \cdot \vec{\psi}_{in}^{(1)} $$
*   **Anti-Jaynes-Cummings (AJC):** $AJC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_+ \hat{a}^\dagger + e^{-i\varphi}\sigma_- \hat{a})]$
    $$ \vec{\psi}_{out}^{(0)} = \cos(\Omega_{n+1}) \cdot \vec{\psi}_{in}^{(0)} - i e^{-i\varphi} \hat{a}^\dagger \frac{\sin(\Omega_{n+1})}{\sqrt{\hat{n}+1}} \cdot \vec{\psi}_{in}^{(1)} $$
    $$ \vec{\psi}_{out}^{(1)} = -i e^{i\varphi} \frac{\sin(\Omega_n)}{\sqrt{\hat{n}}} \hat{a} \cdot \vec{\psi}_{in}^{(0)} + \cos(\Omega_n) \cdot \vec{\psi}_{in}^{(1)} $$

### 3.3 Special Mapping Gates
*   **Selective Qubit Rotation (SQR):** $SQR(\vec{\theta}, \vec{\varphi}) = \sum_n R_{\varphi_n}(\theta_n) \otimes |n\rangle\langle n|$
    A Qumode-controlled Qubit gate. For each Fock state $n$, a different rotation $R_{\varphi_n}(\theta_n)$ is applied to the Qubit. Let $\alpha_n = \cos(\frac{\theta_n}{2})$ and $\beta_n = -e^{-i\varphi_n}\sin(\frac{\theta_n}{2})$.
    $$ \vec{\psi}_{out}^{(0)}[n] = \alpha_n \cdot \vec{\psi}_{in}^{(0)}[n] + \beta_n \cdot \vec{\psi}_{in}^{(1)}[n] $$
    $$ \vec{\psi}_{out}^{(1)}[n] = -\beta_n^* \cdot \vec{\psi}_{in}^{(0)}[n] + \alpha_n^* \cdot \vec{\psi}_{in}^{(1)}[n] $$

---

## 4. Pure Continuous-Variable (CV) Gates

Pure CV gates act exclusively on the physical tensors (Qumodes) without altering the HDD branch topology. The CPU extracts unique tensor IDs and dispatches them to the GPU for in-place updates.

### 4.1 Gaussian Path (SET Track)
In the Symbolic Execution Track, a CV state is represented by a displacement vector $d \in \mathbb{R}^{2M}$ and covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{2M \times 2M}$.

*   **Symplectic Update Formulas:** A Gaussian gate $G$ (such as $D, S, BS, R$) applies a symplectic transformation $(S_g, d_g)$:
    $$ \mathbf{\Sigma}_{new} = S_g \mathbf{\Sigma}_{old} S_g^T $$
    $$ d_{new} = S_g d_{old} + d_g $$
*   **Fidelity-Based Path Merging:** Used for HDD deduplication:
    $$ F = \frac{2^M}{\sqrt{\det(\mathbf{\Sigma}_1 + \mathbf{\Sigma}_2)}} \exp \left( -\frac{1}{2} (d_1 - d_2)^T (\mathbf{\Sigma}_1 + \mathbf{\Sigma}_2)^{-1} (d_1 - d_2) \right) $$

### 4.2 Fock-ELL Path (TET Track)
In the Tensor Execution Track, operations are performed directly on the truncated Fock space tensors (dimension $D$).

#### 4.2.1 Diagonal Gates (Level 0)
*   **Phase-Space Rotation (R):** $R(\theta) = \exp[-i\theta \hat{n}]$
    $$ \vec{\psi}_{out}[idx] = \vec{\psi}_{in}[idx] \cdot \exp(-i \cdot \theta \cdot n_k) $$
*   **Kerr Gate (Non-Gaussian):** $K(\chi) = \exp[-i\chi \hat{n}^2]$
    $$ \vec{\psi}_{out}[idx] = \vec{\psi}_{in}[idx] \cdot \exp(-i \cdot \chi \cdot n_k^2) $$

#### 4.2.2 Ladder/Shift Gates (Level 1)
*   **Photon Creation ($\hat{a}^\dagger$):** $\vec{\psi}_{out}[n] = \sqrt{n} \cdot \vec{\psi}_{in}[n-1]$
*   **Photon Annihilation ($\hat{a}$):** $\vec{\psi}_{out}[n] = \sqrt{n+1} \cdot \vec{\psi}_{in}[n+1]$

#### 4.2.3 General Single-Mode Banded Gates (Level 2)
Gaussian gates act as banded matrices in the Fock basis, calculated via Sparse Matrix-Vector Multiplication (SpMV) using **Compact Strided-ELL**.
*   **Displacement (D):** $D(\alpha) = \exp[\alpha \hat{a}^\dagger - \alpha^* \hat{a}]$
*   **Squeezing (S):** $S(\xi) = \exp[\frac{1}{2}(\xi^* \hat{a}^2 - \xi \hat{a}^{\dagger 2})]$
    *(Note: Squeezing conserves parity, thus utilizing a "Stride-2" checkerboard ELL format.)*
    $$ \vec{\psi}_{out}[n] = \sum_{k=0}^{K_{eff}-1} \text{ELL\_Val}[n][k] \cdot \vec{\psi}_{in}[ \text{ELL\_Col}[n][k] ] $$

#### 4.2.4 Two-Mode Mixing Gates (Level 3)
*   **Beam-Splitter (BS):** $BS(\theta, \varphi) = \exp[-i \frac{\theta}{2} (e^{i\varphi} \hat{a}^\dagger \hat{b} + e^{-i\varphi} \hat{a} \hat{b}^\dagger)]$
    Conserves total photon number $N = n_a + n_b$. The space is decomposed into block-diagonal invariant subspaces $L$. For each sub-block $U_{BS}^{(L)}$:
    $$ \vec{\psi}_{out}^{(L)}[row] = \sum_{col=0}^{L} U^{(L)}_{row, col} \cdot \vec{\psi}_{in}^{(L)}[col] $$
    For small $\theta$ (Trotterization), it acts as a 2D Stencil shifting amplitudes between $|n_a, n_b\rangle$ and $|n_a \pm 1, n_b \mp 1\rangle$.

---

## 5. Non-Gaussian Operations and Cross-Track Evolution

### 5.1 Mixture Gaussian Decomposition (Non-Gaussian on SET)
When a weak non-Gaussian gate (e.g., Cubic Phase, small Kerr) is applied on the SET, it is decomposed to prevent collapsing to the TET:
$$ U_{NG} \approx \sum_{k=1}^K c_k G_k $$
The parent HDD node is logically split into $K$ new branches, maintaining the SET symplectic representation. Each branch updates via standard Gaussian arithmetic, and its CPU HDD weight is scaled by $c_k$.

### 5.2 Analytical Ejection (SET to TET)
When a Gaussian state must be materialized into a Fock tensor (due to strong non-Gaussianity or measurement), an **Analytical Ejection** occurs directly on the GPU. The state $|d, \mathbf{\Sigma}\rangle$ is projected to $c_n = \langle n | \psi \rangle$:
$$ c_n = \frac{1}{\sqrt{n!}} \exp(A) \cdot B^n \cdot H_n(C) $$
Computed using a stabilized parallel recurrence relation:
$$ \sqrt{n+1} \cdot c_{n+1} = f_1(A, B, C) \cdot c_n + f_2(A, B, C) \cdot c_{n-1} $$
