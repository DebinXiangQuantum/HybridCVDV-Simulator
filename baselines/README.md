# 性能对比测试说明

本目录包含 HybridCVDV-Simulator 与 Strawberry Fields 的性能对比测试。

## 📋 目录结构

```
baselines/
├── README.md                          # 本文件
├── requirements.txt                   # Python依赖
├── test_strawberryfields.py          # Strawberry Fields 基准测试脚本
├── compare_results.py                 # 结果对比脚本
├── strawberryfields_results.json     # SF测试结果（运行后生成）
├── comparison_results.json            # 对比结果（运行后生成）
├── performance_comparison.png         # 性能对比图（运行后生成）
└── accuracy_comparison.png            # 精度对比图（运行后生成）
```

## 🚀 快速开始

### 方法1: 使用自动化脚本（推荐）

在项目根目录运行：

```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

这个脚本会自动完成以下步骤：
1. 安装 Python 依赖
2. 运行 Strawberry Fields 基准测试
3. 编译 C++ 基准测试
4. 运行 HybridCVDV-Simulator 基准测试
5. 对比并生成报告

### 方法2: 手动运行

#### 步骤1: 安装依赖

```bash
# 使用 uv 安装 Python 依赖
uv pip install -r baselines/requirements.txt
```

#### 步骤2: 运行 Strawberry Fields 测试

```bash
uv run baselines/test_strawberryfields.py
```

这会生成 `baselines/strawberryfields_results.json`

#### 步骤3: 编译并运行 C++ 基准测试

```bash
# 在项目根目录
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make benchmark_cv_gates -j$(nproc)
cd ..

# 运行基准测试
./build/benchmark_cv_gates
```

这会生成 `tests/hybridcvdv_results.json`

#### 步骤4: 对比结果

```bash
uv run baselines/compare_results.py
```

这会生成：
- `baselines/comparison_results.json` - 详细对比数据
- `baselines/performance_comparison.png` - 性能对比图
- `baselines/accuracy_comparison.png` - 精度对比图

## 📊 测试内容

### 测试的门操作

1. **位移门 (Displacement Gate)** - `D(α)`
   - 参数: α = 0.5 + 0.2i
   - 测试次数: 10次

2. **挤压门 (Squeezing Gate)** - `S(r, φ)`
   - 参数: r = 0.5, φ = 0.0
   - 测试次数: 10次

3. **光束分裂器 (Beam Splitter)** - `BS(θ, φ)`
   - 参数: θ = π/4, φ = 0.0
   - 测试次数: 10次

4. **相位旋转门 (Phase Rotation)** - `R(θ)`
   - 参数: θ = π/4
   - 测试次数: 10次

5. **Kerr门** - `K(κ)`
   - 参数: κ = 0.5
   - 测试次数: 10次

6. **复杂电路 (Complex Circuit)**
   - 多个门的组合
   - 测试次数: 10次

### 测试参数

- **Fock空间截断维度**: 16
- **量子模式数量**: 1-2个
- **每个测试的运行次数**: 10次

## 📈 评估指标

### 性能指标

1. **平均执行时间** (ms)
2. **标准差** (ms)
3. **最小/最大执行时间** (ms)
4. **加速比** = SF时间 / HybridCVDV时间

### 精度指标

1. **保真度 (Fidelity)**: F = |⟨ψ₁|ψ₂⟩|²
   - 范围: [0, 1]
   - 越接近1越好

2. **迹距离 (Trace Distance)**: D = √(1 - F)
   - 范围: [0, 1]
   - 越接近0越好

3. **L2误差**: ||ψ₁ - ψ₂||₂
   - 越小越好

## 🔧 自定义测试

### 修改测试参数

编辑 `test_strawberryfields.py` 中的参数：

```python
# 修改截断维度
benchmark = StrawberryFieldsBenchmark(cutoff_dim=32, num_modes=2)

# 修改测试参数
benchmark.test_displacement(alpha=1.0+0.5j, num_runs=20)
```

编辑 `tests/benchmark_cv_gates.cpp` 中的对应参数：

```cpp
CVBenchmark benchmark(32, 64);  // cutoff_dim=32, max_states=64
benchmark.test_displacement(1.0, 0.5, 20);  // alpha_real, alpha_imag, num_runs
```

### 添加新的测试

1. 在 `test_strawberryfields.py` 中添加新的测试方法
2. 在 `tests/benchmark_cv_gates.cpp` 中添加对应的C++测试
3. 确保测试名称 (`test_name`) 一致

## 📝 结果解读

### 性能对比

- **加速比 > 1**: HybridCVDV-Simulator 更快
- **加速比 < 1**: Strawberry Fields 更快
- **加速比 ≈ 1**: 性能相当

### 精度对比

- **保真度 > 0.9999**: 高精度，结果几乎完全一致
- **保真度 > 0.99**: 良好精度
- **保真度 < 0.99**: 可能存在数值误差

## 🐛 故障排除

### 问题1: CUDA错误

```
错误: CUDA not found
```

**解决方案**: 确保已安装CUDA并正确配置环境变量

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 问题2: Strawberry Fields安装失败

```
错误: Could not find a version that satisfies the requirement strawberryfields
```

**解决方案**: 使用 uv 安装

```bash
uv pip install strawberryfields
```

### 问题3: 编译错误

```
错误: undefined reference to `apply_displacement`
```

**解决方案**: 确保GPU源文件已正确编译

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make clean
make benchmark_cv_gates -j$(nproc)
```

### 问题4: 结果文件不存在

```
错误: 找不到结果文件
```

**解决方案**: 按顺序运行测试

```bash
# 1. 先运行 SF 测试
uv run baselines/test_strawberryfields.py

# 2. 再运行 C++ 测试
./build/benchmark_cv_gates

# 3. 最后对比结果
uv run baselines/compare_results.py
```

## 📚 参考资料

- [Strawberry Fields 文档](https://strawberryfields.ai/)
- [HybridCVDV-Simulator 文档](../README.md)
- [连续变量量子计算](https://en.wikipedia.org/wiki/Continuous-variable_quantum_information)

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
