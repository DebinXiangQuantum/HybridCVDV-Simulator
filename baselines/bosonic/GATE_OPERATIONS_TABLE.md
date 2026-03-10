# operators_gpu.py 实现的量子门操作表

## 📊 完整门操作列表

### 基础量子比特门（Qubit Gates）

| 门名称 | 符号 | 矩阵维度 | 描述 | 实现方式 |
|--------|------|----------|------|----------|
| Pauli X | X | 2×2 | 比特翻转门 | 预定义矩阵 |
| Pauli Y | Y | 2×2 | Y 旋转门 | 预定义矩阵 |
| Pauli Z | Z | 2×2 | 相位翻转门 | 预定义矩阵 |
| S+ (PLUS) | σ+ | 2×2 | 升算符 | 预定义矩阵 |
| S- (MINUS) | σ- | 2×2 | 降算符 | 预定义矩阵 |
| P0 | \|0⟩⟨0\| | 2×2 | 投影到 \|0⟩ | 预定义矩阵 |
| P1 | \|1⟩⟨1\| | 2×2 | 投影到 \|1⟩ | 预定义矩阵 |

### 基础连续变量算符（CV Operators）

| 算符名称 | 符号 | 矩阵维度 | 描述 | 函数名 | 参数 |
|----------|------|----------|------|--------|------|
| 单位算符 | I | cutoff×cutoff | 单位矩阵 | `get_eye(dim)` | dim: 维度 |
| 湮灭算符 | a | cutoff×cutoff | 降低光子数 | `get_a(cutoff)` | cutoff: 截断维度 |
| 产生算符 | a† | cutoff×cutoff | 增加光子数 | `get_ad(cutoff)` | cutoff: 截断维度 |
| 数算符 | n | cutoff×cutoff | 光子数算符 | `get_N(cutoff)` | cutoff: 截断维度 |
| 投影算符 | \|n⟩⟨n\| | cutoff×cutoff | 投影到 Fock 态 n | `get_projector(n, cutoff)` | n: Fock 态, cutoff: 截断维度 |

### 单模量子门（Single-Mode Gates）

| 门名称 | 符号 | 描述 | 函数名 | 参数 | 性能测量 |
|--------|------|------|--------|------|----------|
| 相位旋转 | R(θ) | 相空间旋转 | `r(theta, cutoff)` | theta: 旋转角度 | ✅ |
| 位移 | D(α) | 相空间位移 | `d(alpha, cutoff)` | alpha: 复数位移 | ✅ |
| 单模压缩 | S(θ) | 单模压缩 | `s(theta, cutoff)` | theta: 复数压缩参数 | ✅ |

### 双模量子门（Two-Mode Gates）

| 门名称 | 符号 | 描述 | 函数名 | 参数 | 性能测量 |
|--------|------|------|--------|------|----------|
| 双模压缩 | S₂(θ) | 双模压缩 | `s2(theta, cutoff_a, cutoff_b)` | theta: 复数参数 | ✅ |
| 分束器 | BS(θ) | 光束分裂器 | `bs(theta, cutoff_a, cutoff_b)` | theta: 复数参数 | ✅ |
| 指数 SWAP | eSWAP(θ) | 指数交换门 | `eswap(theta, cutoff_a, cutoff_b)` | theta: 角度 | ✅ |
| 双模求和 | SUM(s) | 双模求和门 | `sum(scale, cutoff_a, cutoff_b)` | scale: 缩放因子 | ✅ |

### 三模量子门（Three-Mode Gates）

| 门名称 | 符号 | 描述 | 函数名 | 参数 | 性能测量 |
|--------|------|------|--------|------|----------|
| 三模压缩 | S₃(θ) | 三模压缩 | `s3(theta, cutoff_a, cutoff_b, cutoff_c)` | theta: 复数参数 | ✅ |

### 受控量子门（Controlled Gates）

| 门名称 | 符号 | 描述 | 函数名 | 参数 | 性能测量 |
|--------|------|------|--------|------|----------|
| 受控相位旋转 | CR(θ) | 受控 Z 旋转 | `cr(theta, cutoff)` | theta: 角度 | ✅ |
| 受控 X 旋转 | CRX(θ) | 受控 X 旋转 | `crx(theta, cutoff)` | theta: 角度 | ✅ |
| 受控 Y 旋转 | CRY(θ) | 受控 Y 旋转 | `cry(theta, cutoff)` | theta: 角度 | ✅ |
| 受控位移 | CD(α,β) | 受控位移 | `cd(alpha, beta, cutoff)` | alpha, beta: 复数 | ✅ |
| 回声受控位移 | ECD(θ) | 回声受控位移 | `ecd(theta, cutoff)` | theta: 复数 | ✅ |
| 受控分束器 | CBS(θ) | 受控分束器 | `cbs(theta, cutoff_a, cutoff_b)` | theta: 复数 | ✅ |
| 受控压缩 | CSQ(θ) | 受控单模压缩 | `csq(theta, cutoff)` | theta: 复数 | ✅ |
| 受控求和 | CSUM(s) | 受控双模求和 | `csum(scale, cutoff_a, cutoff_b)` | scale: 缩放因子 | ✅ |

### SNAP 类门（Selective Number-dependent Arbitrary Phase）

| 门名称 | 符号 | 描述 | 函数名 | 参数 | 性能测量 |
|--------|------|------|--------|------|----------|
| SNAP | SNAP(θ,n) | 单 Fock 态选择性相位 | `snap(theta, n, cutoff)` | theta: 相位, n: Fock 态 | ✅ |
| 受控 SNAP | CSNAP(θ,n) | 受控单 Fock 态相位 | `csnap(theta, n, cutoff)` | theta: 相位, n: Fock 态 | ✅ |
| 多态 SNAP | MultiSNAP | 多 Fock 态选择性相位 | `multisnap(phase_map, cutoff)` | phase_map: {n: θ} 字典 | ✅ |
| 受控多态 SNAP | CMultiSNAP | 受控多 Fock 态相位 | `multicsnap(phase_map, cutoff)` | phase_map: {n: θ} 字典 | ✅ |

### 特殊量子门（Special Gates）

| 门名称 | 符号 | 描述 | 函数名 | 参数 | 性能测量 |
|--------|------|------|--------|------|----------|
| SQR 门 | SQR(θ) | 平方门 | `sqr(theta, cutoff)` | theta: 角度 | ✅ |
| PNR 门 | PNR(max) | 光子数读出支持门 | `pnr(max, cutoff)` | max: 最大光子数 | ✅ |
| Jaynes-Cummings | JC(θ,φ) | JC 相互作用 | `jc(theta, phi, cutoff)` | theta, phi: 角度 | ✅ |
| 反 JC | AJC(θ,φ) | 反 JC 相互作用 | `ajc(theta, phi, cutoff)` | theta, phi: 角度 | ✅ |
| Rabi 门 | RB(θ) | Rabi 相互作用 | `rb(theta, cutoff)` | theta: 复数 | ✅ |
| Schwinger 门 | CSchwinger | 受控 Schwinger 门 | `cschwinger(beta, θ₁, φ₁, θ₂, φ₂, cutoff_a, cutoff_b)` | 多个角度参数 | ✅ |
| 多玻色子采样 | CMultiboson | 多玻色子采样支持 | `c_multiboson_sampling(max, cutoff)` | max: 最大数 | ✅ |

## 📈 性能测量指标

所有标记为 ✅ 的门操作都包含以下性能测量：

1. **单个门延迟** (ms) - 执行单个门操作的时间
2. **批处理吞吐量** (gates/s) - 每秒可执行的门操作数
3. **内存效率** - 非零元素密度（nnz / total_elements）

## 🔧 实现细节

### 矩阵格式
- **稀疏矩阵**: 使用 CuPy 的 CSC (Compressed Sparse Column) 格式
- **密集矩阵**: 用于小型矩阵和特殊操作

### 计算方式
- **矩阵指数**: 使用 SciPy 的 `expm` 函数（在 CPU 上计算后转回 GPU）
- **张量积**: 使用 CuPy 的 `kron` 函数
- **矩阵乘法**: 使用 CuPy 的稀疏矩阵乘法

### GPU 加速
- **内存管理**: 自动管理 GPU 内存池
- **批处理**: 支持批量门操作测量
- **性能监控**: 实时测量延迟和吞吐量

## 📊 门操作统计

### 按类别统计

| 类别 | 门数量 | 占比 |
|------|--------|------|
| 基础量子比特门 | 7 | 18.9% |
| 基础 CV 算符 | 5 | 13.5% |
| 单模门 | 3 | 8.1% |
| 双模门 | 4 | 10.8% |
| 三模门 | 1 | 2.7% |
| 受控门 | 8 | 21.6% |
| SNAP 类门 | 4 | 10.8% |
| 特殊门 | 5 | 13.5% |
| **总计** | **37** | **100%** |

### 按参数类型统计

| 参数类型 | 门数量 | 示例 |
|----------|--------|------|
| 实数角度 | 8 | r, cr, crx, cry, jc, ajc |
| 复数参数 | 12 | d, s, s2, s3, bs, cd, ecd, cbs, csq, rb |
| 整数参数 | 4 | snap, csnap, pnr, c_multiboson_sampling |
| 字典参数 | 2 | multisnap, multicsnap |
| 多参数 | 1 | cschwinger (5 个参数) |

### 按模式数统计

| 模式数 | 门数量 | 占比 |
|--------|--------|------|
| 单模 (qubit + qumode) | 15 | 40.5% |
| 双模 | 12 | 32.4% |
| 三模 | 1 | 2.7% |
| 纯量子比特 | 7 | 18.9% |
| 纯算符 | 5 | 13.5% |

## 🎯 使用示例

### 基本用法
```python
from operators_gpu import CVOperatorsGPU

# 创建实例
cv_ops = CVOperatorsGPU()
cutoff = 16

# 单模门
r_gate = cv_ops.r(0.5, cutoff)           # 相位旋转
d_gate = cv_ops.d(1.0 + 0.5j, cutoff)    # 位移
s_gate = cv_ops.s(0.3 + 0.2j, cutoff)    # 压缩

# 双模门
bs_gate = cv_ops.bs(0.5, cutoff, cutoff)           # 分束器
s2_gate = cv_ops.s2(0.3 + 0.2j, cutoff, cutoff)    # 双模压缩

# 受控门
cr_gate = cv_ops.cr(0.5, cutoff)                   # 受控旋转
cd_gate = cv_ops.cd(1.0+0.5j, 0.5-0.3j, cutoff)   # 受控位移

# SNAP 门
snap_gate = cv_ops.snap(0.5, 3, cutoff)            # 单态 SNAP
multi_snap = cv_ops.multisnap({0: 0.1, 1: 0.2}, cutoff)  # 多态 SNAP
```

### 性能测量
```python
from operators_gpu import print_performance_summary, save_performance_to_json

# 执行门操作（自动测量性能）
cv_ops.r(0.5, cutoff)
cv_ops.d(1.0 + 0.5j, cutoff)

# 打印性能总结
print_performance_summary()

# 保存到 JSON
save_performance_to_json("performance.json")
```

## 📝 注意事项

1. **内存管理**: 所有门操作后会自动清理 GPU 内存
2. **性能测量**: 使用 `@measure_performance()` 装饰器自动测量
3. **矩阵格式**: 返回 CSC 格式的稀疏矩阵
4. **CPU 回退**: 矩阵指数计算在 CPU 上进行（SciPy expm）
5. **批处理**: 默认批处理大小为 10，可在装饰器中修改

## 🔗 相关文件

- **实现文件**: `operators_gpu.py`
- **C++ 版本**: `gpu/operators.cu`
- **性能对比**: `compare_performance.py`
- **测试文件**: `test_operators_gpu.py`

## 📚 参考文献

- arXiv:2407.10381 - 连续变量量子计算
- Strawberry Fields 文档
- CuPy 稀疏矩阵文档

---

**生成日期**: 2026-01-29  
**版本**: 1.0  
**总门数**: 37 个
