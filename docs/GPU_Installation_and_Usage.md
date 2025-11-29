# HybridCVDV-Simulator GPU版本安装、运行和测试指南

## 概述

HybridCVDV-Simulator是一个混合CPU+GPU架构的量子模拟器，支持连续变量和离散变量量子计算。本文档介绍如何安装、运行和测试GPU版本的代码。

## 系统要求

### 硬件要求
- **NVIDIA GPU**: 支持CUDA计算能力6.0及以上的GPU（如GTX 10系列、RTX 20/30系列、A100等）
- **内存**: 至少8GB RAM，推荐16GB+
- **存储**: 至少2GB可用磁盘空间

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+, CentOS 7+, 或其他兼容的Linux发行版)
- **CUDA Toolkit**: 11.0+ (推荐11.8+)
- **GCC**: 9.0+ (与CUDA兼容的版本)
- **CMake**: 3.16+
- **Git**: 用于代码获取

## 安装步骤

### 1. 安装CUDA Toolkit

```bash
# 下载CUDA 11.8 (或最新版本)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装CUDA (需要root权限)
sudo sh cuda_11.8.0_520.61.05_linux.run

# 安装过程中选择:
# - CUDA Toolkit
# - CUDA Samples
# - CUDA Documentation
# - CUDA Visual Studio Integration (如果需要)
```

### 2. 配置CUDA环境变量

```bash
# 添加到~/.bashrc或/etc/environment
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 重新加载环境变量
source ~/.bashrc
```

### 3. 安装系统依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git gcc-9 g++-9

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 git gcc gcc-c++
```

### 4. 验证CUDA安装

```bash
# 检查CUDA版本
nvcc --version

# 检查GPU信息
nvidia-smi

# 运行CUDA示例
cd /usr/local/cuda-11.8/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

## 编译项目

### 1. 下载源代码

```bash
git clone https://github.com/your-repo/HybridCVDV-Simulator.git
cd HybridCVDV-Simulator
```

### 2. 创建构建目录

```bash
mkdir build && cd build
```

### 3. 配置CMake

```bash
# 自动检测CUDA
cmake ..

# 或手动指定CUDA路径
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### 4. 编译项目

```bash
# 编译所有组件
make -j$(nproc)

# 或者分步编译
make HybridCVDV-Simulator_core
make HybridCVDV-Simulator_gpu
make HybridCVDV-Simulator_main
make HybridCVDV-Simulator_gpu_validation
```

## 运行程序

### 1. GPU验证程序

GPU验证程序用于测试CUDA可用性和GPU计算精度：

```bash
# 运行GPU验证
./HybridCVDV-Simulator_gpu_validation

# 预期输出示例：
# ========================================
#   Hybrid Tensor-DD 量子模拟器 GPU验证
# ========================================
# 发现CUDA设备: NVIDIA A100
# GPU内存: 40GB
# CUDA版本: 11.8
#
# 1. GPU相位旋转测试 ✓
# 2. GPU Kerr门测试 ✓
# 3. GPU创建算符测试 ✓
# 4. GPU湮灭算符测试 ✓
# 5. GPU位移门测试 ✓
# 6. GPU挤压门测试 ✓
#
# 所有GPU测试通过，精度符合要求！
```

### 2. 主程序

主程序演示混合CPU+GPU量子计算：

```bash
# 运行主程序
./HybridCVDV-Simulator_main

# 预期输出示例：
# ========================================
#   Hybrid Tensor-DD 量子模拟器
# ========================================
# 版本: v1.5
# 架构: Hybrid CV-DV Quantum Systems
# 硬件: CPU (逻辑控制) + NVIDIA GPU (张量计算)
#
# 示例1: 连续变量位移和挤压
# 执行时间: 0.535264 秒
# 电路统计: 2 个门, 1 个活跃状态
```

### 3. 性能测试

```bash
# 使用time命令测量性能
time ./HybridCVDV-Simulator_gpu_validation

# 或使用nvprof进行详细分析
nvprof ./HybridCVDV-Simulator_main
```

## 测试说明

### GPU功能测试

项目包含全面的GPU功能测试：

#### 1. CUDA可用性测试
- 检测CUDA设备
- 检查GPU内存
- 验证CUDA运行时

#### 2. 精度测试
- 所有GPU操作的误差必须小于10^-6
- 与CPU参考实现进行比较
- 自动验证数值精度

#### 3. 功能测试
- **Level 0门**: 相位旋转、Kerr门、条件奇偶校验
- **Level 1门**: 创建算符、湮灭算符
- **Level 2门**: 位移门、单模门
- **Level 3门**: 光束分裂器
- **Level 4门**: 受控位移、受控挤压

### 运行测试

```bash
# 快速GPU功能测试
./HybridCVDV-Simulator_gpu_validation

# 详细性能分析
nvprof --print-gpu-trace ./HybridCVDV-Simulator_gpu_validation

# 内存使用分析
cuda-memcheck ./HybridCVDV-Simulator_main
```

## 故障排除

### 常见问题

#### 1. CUDA编译错误

**问题**: `nvcc fatal: Unsupported gpu architecture 'compute_80'`
**解决**: 更新CMakeLists.txt中的CUDA架构设置

```cmake
# 在CMakeLists.txt中
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
```

#### 2. 链接错误

**问题**: `undefined reference to cuda* functions`
**解决**: 确保正确链接CUDA库

```cmake
target_link_libraries(your_target CUDA::cudart CUDA::cublas)
```

#### 3. 内存访问错误

**问题**: `an illegal memory access was encountered`
**解决**:
- 检查GPU内存分配
- 确保指针正确传递
- 添加CUDA设备同步

#### 4. GCC版本兼容性

**问题**: `redefinition of 'constexpr const _Tp std::integral_constant<_Tp, __v>::value'`
**解决**: 使用GCC 9

```bash
sudo apt install gcc-9 g++-9
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
```

### 调试技巧

#### 启用详细输出

```bash
# 设置CUDA调试环境变量
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# 运行程序
./HybridCVDV-Simulator_gpu_validation
```

#### GPU内存检查

```bash
# 使用cuda-memcheck检测内存错误
cuda-memcheck ./HybridCVDV-Simulator_main
```

#### 性能分析

```bash
# 使用nvprof分析GPU性能
nvprof --print-gpu-summary ./HybridCVDV-Simulator_gpu_validation

# 生成详细的时间线
nvprof --print-gpu-trace --csv ./HybridCVDV-Simulator_main > profile.csv
```

## 性能优化

### GPU优化建议

1. **内存管理**
   - 使用统一内存（CUDA 6.0+）
   - 优化数据传输
   - 减少主机-设备内存拷贝

2. **并行优化**
   - 调整block和grid尺寸
   - 使用共享内存
   - 优化内存访问模式

3. **精度配置**
   - 根据需要选择单/双精度
   - 平衡精度和性能

### 示例性能数据

在NVIDIA A100 GPU上测试结果：

| 操作类型 | 执行时间 | 精度误差 |
|---------|---------|---------|
| 相位旋转 | < 0.1ms | < 10^-8 |
| 位移门 | < 0.5ms | < 10^-7 |
| 挤压门 | < 1.0ms | < 10^-6 |
| 批量操作 | < 10ms | < 10^-6 |

## 高级配置

### 自定义CUDA架构

```cmake
# 在CMakeLists.txt中设置目标架构
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 86)

# 为不同GPU编译多个版本
set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 80-real 86-real)
```

### 多GPU支持

```bash
# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=0,1

# 在代码中选择GPU设备
cudaSetDevice(device_id);
```

### 内存优化

```cpp
// 使用统一内存
cudaMallocManaged(&ptr, size);

// 异步内存拷贝
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

## 技术支持

如遇到问题，请：

1. 检查系统要求是否满足
2. 运行GPU验证程序确认CUDA环境正常
3. 查看CMake和make的错误输出
4. 使用cuda-memcheck和nvprof进行调试

## 版本信息

- **项目版本**: v1.5
- **CUDA要求**: 11.0+
- **GCC要求**: 9.0+
- **CMake要求**: 3.16+

---

本文档持续更新，如有问题请提交Issue或Pull Request。
