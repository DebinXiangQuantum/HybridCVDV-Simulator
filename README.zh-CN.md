# HybridCVDV-Simulator 中文入口

项目当前的主 README 已切换为英文维护，以避免中文文档长期滞后于代码实现。

- **英文主文档**: [README.md](README.md)
- **架构说明**: [docs/architecture.md](docs/architecture.md)
- **实验与 benchmark 说明**: [experiments/README.md](experiments/README.md)

## 当前项目内容

HybridCVDV-Simulator 是一个面向混合 DV-CV 量子线路的 C++/CUDA 模拟器代码库，核心方向包括：

- CPU 侧基于 HDD 的离散分支控制
- GPU 侧连续变量态演化与门执行
- Gaussian symbolic / exact Fock / mixture approximation 等执行路径
- `experiments/` 下的单 GPU benchmark 与基线对比工具
- `src/noisy/` 下正在演进中的噪声模拟子系统

如需最新的安装、构建、运行方式，请直接查看英文主文档：[README.md](README.md)。
