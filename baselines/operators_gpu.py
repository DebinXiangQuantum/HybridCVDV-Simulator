import cupy as cp
import cupyx.scipy.sparse as sp
from cupy.typing import ArrayLike, NDArray
import scipy.linalg as splinalg
import time
import psutil
import os
import json

# 导入 Qiskit 相关模块（在 GPU 版本中可能不需要，或者需要特殊处理）


# 性能测量数据存储
gate_performance_data = {}

# 性能测量装饰器
def measure_performance(batch_size=10):
    """
    测量门操作的性能指标：
    - 单个门延迟 (毫秒)
    - 批处理吞吐量 (门/秒)
    - 内存效率 (非零元素密度)
    
    参数:
        batch_size: 批处理测量的门数量
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # 获取当前进程的内存使用情况
            process = psutil.Process(os.getpid())
            
            # 清理GPU内存池，确保测量准确性
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # 测量单个门延迟
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            single_gate_latency_ms = (end_time - start_time) * 1000.0  # 转换为毫秒
            
            # 计算内存效率（非零元素密度，类似C++的active_states / cv_truncation）
            # 获取结果矩阵的非零元素数量和总元素数量
            if hasattr(result, 'nnz') and hasattr(result, 'shape'):
                total_elements = result.shape[0] * result.shape[1]
                non_zero_elements = result.nnz
                memory_efficiency = non_zero_elements / total_elements if total_elements > 0 else 0.0
            else:
                # 如果结果不是稀疏矩阵，使用默认值
                memory_efficiency = 1.0
            
            # 清理内存以准备批处理测量
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # 测量批处理吞吐量
            start_time_batch = time.time()
            for _ in range(batch_size):
                gate_result = func(self, *args, **kwargs)
                # 立即释放不再需要的中间结果
                del gate_result
                # 显式释放GPU内存
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            
            end_time_batch = time.time()
            batch_processing_time = end_time_batch - start_time_batch
            throughput = batch_size / batch_processing_time  # 门/秒
            
            # 获取门名称
            gate_name = func.__name__
            
            # 存储性能数据
            gate_performance_data[gate_name] = {
                "gate_name": gate_name,
                "single_gate_latency_ms": single_gate_latency_ms,
                "batch_throughput": throughput,
                "memory_efficiency": memory_efficiency
            }
            
            return result
        return wrapper
    return decorator


class CVOperatorsGPU:
    """
    基于 Python 的 GPU 加速实现的连续变量量子算子类
    """

    def __init__(self):
        self.cutoff: int = 0
        # 基本量子比特算子（作用于 0 和 1 维）
        self.X = sp.csc_matrix(cp.array([[0, 1], [1, 0]], dtype=cp.complex128))
        self.Y = sp.csc_matrix(cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128))
        self.Z = sp.csc_matrix(cp.array([[1, 0], [0, -1]], dtype=cp.complex128))
        self.PLUS = sp.csc_matrix(cp.array([[0, 1], [0, 0]], dtype=cp.complex128))
        self.MINUS = sp.csc_matrix(cp.array([[0, 0], [1, 0]], dtype=cp.complex128))
        self.SPLIT = sp.csc_matrix(cp.array([[0, 1], [0, 0]], dtype=cp.complex128))
        self.SMINUS = sp.csc_matrix(cp.array([[0, 0], [1, 0]], dtype=cp.complex128))
        self.P0 = sp.csc_matrix(cp.array([[1, 0], [0, 0]], dtype=cp.complex128))
        self.P1 = sp.csc_matrix(cp.array([[0, 0], [0, 1]], dtype=cp.complex128))
        self.PLUS = sp.csc_matrix(cp.array([[0, 1], [0, 0]], dtype=cp.complex128))
        self.MINUS = sp.csc_matrix(cp.array([[0, 0], [1, 0]], dtype=cp.complex128))
        self.SPLIT = sp.csc_matrix(cp.array([[0, 1], [0, 0]], dtype=cp.complex128))
        self.SMINUS = sp.csc_matrix(cp.array([[0, 0], [1, 0]], dtype=cp.complex128))
        self.P0 = sp.csc_matrix(cp.array([[1, 0], [0, 0]], dtype=cp.complex128))
        self.P1 = sp.csc_matrix(cp.array([[0, 0], [0, 1]], dtype=cp.complex128))

    def get_eye(self, dim: int) -> sp.csc_matrix:
        """Returns identity operator of dimension dim"""
        return sp.eye(dim, dtype=cp.complex128)

    def get_N(self, cutoff: int) -> sp.csc_matrix:
        """Returns number operator"""
        # 使用 coo_matrix 来构建数量算符
        rows = cp.arange(cutoff)
        cols = cp.arange(cutoff)
        data = cp.arange(cutoff, dtype=cp.complex128)
        n = sp.coo_matrix((data, (rows, cols)), shape=(cutoff, cutoff), dtype=cp.complex128)
        return n.tocsc()

    def get_a(self, cutoff: int) -> sp.csc_matrix:
        """Returns annihilation operator"""
        # 使用 coo_matrix 来构建湮灭算符
        rows = cp.arange(1, cutoff)
        cols = cp.arange(cutoff - 1)
        data = cp.sqrt(cp.arange(1, cutoff), dtype=cp.complex128)
        a = sp.coo_matrix((data, (rows, cols)), shape=(cutoff, cutoff), dtype=cp.complex128)
        return a.tocsc()

    def get_projector(self, n: int, cutoff: int) -> sp.csc_matrix:
        """Returns projector onto number state n"""
        # 使用 coo_matrix 来构建投影算符
        rows = cp.array([n])
        cols = cp.array([n])
        data = cp.array([1], dtype=cp.complex128)
        proj = sp.coo_matrix((data, (rows, cols)), shape=(cutoff, cutoff), dtype=cp.complex128)
        return proj.tocsc()

    def get_ad(self, cutoff: int) -> sp.csc_matrix:
        """Returns creation operator"""
        return self.get_a(cutoff).conj().transpose().tocsc()

    def get_op(self, name: str, *cutoffs: int) -> sp.csc_matrix:
        """
        Returns operator based on name and cutoffs.

        Example for a single mode:
        name="a" -> annihilation operator
        name="ad" -> creation operator
        name="n" -> number operator

        Example for two modes:
        name="a0 a1" -> a0 ⊗ a1
        name="ad0 n1" -> ad0 ⊗ n1
        """
        # 解析操作符名称
        if len(name.split()) == 1:
            # 单模操作符
            if name == "a":
                return self.get_a(cutoffs[0])
            elif name == "ad":
                return self.get_ad(cutoffs[0])
            elif name == "n":
                return self.get_N(cutoffs[0])
            else:
                raise ValueError(f"Unknown operator name: {name}")
        else:
            # 多模操作符（张量积）
            ops = name.split()
            result = None
            for op in ops:
                # 解析模式索引
                mode = int(op[-1])
                op_type = op[:-1]

                # 获取该模式的操作符
                if op_type == "a":
                    current_op = self.get_a(cutoffs[mode])
                elif op_type == "ad":
                    current_op = self.get_ad(cutoffs[mode])
                elif op_type == "n":
                    current_op = self.get_N(cutoffs[mode])
                else:
                    raise ValueError(f"Unknown operator type: {op_type}")

                # 计算张量积
                if result is None:
                    result = current_op
                else:
                    result = sp.kron(result, current_op, format="csc")

            return result

    @measure_performance()
    def r(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Phase space rotation operator"""
        arg = 1j * theta * self.get_N(cutoff)
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def d(self, alpha: complex, cutoff: int) -> sp.csc_matrix:
        """Displacement operator"""
        arg = alpha * self.get_ad(cutoff)
        hc = arg.conjugate().transpose()
        return sp.csc_matrix(cp.array(splinalg.expm((arg - hc).todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def s(self, theta: complex, cutoff: int) -> sp.csc_matrix:
        """Single-mode squeezing operator"""
        ad = self.get_ad(cutoff)
        ad2 = ad @ ad
        arg = 0.5 * cp.conjugate(theta) * ad2
        hc = arg.conjugate().transpose()
        return sp.csc_matrix(cp.array(splinalg.expm((arg - hc).todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def s2(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Two-mode squeezing operator"""
        r, phi = cp.abs(theta), cp.angle(theta)

        # eq. 183 in arXiv:2407.10381
        arg = r * cp.exp(1j * phi) * self.get_op("ad0 ad1", cutoff_a, cutoff_b)
        hc = arg.conjugate().transpose()
        return sp.csc_matrix(cp.array(splinalg.expm((arg - hc).todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def s3(
        self, theta: complex, cutoff_a: int, cutoff_b: int, cutoff_c: int
    ) -> sp.csc_matrix:
        """Three-mode squeezing operator"""
        r, phi = cp.abs(theta), cp.angle(theta)

        arg = (
            r
            * cp.exp(1j * phi)
            * self.get_op("ad0 ad1 ad2", cutoff_a, cutoff_b, cutoff_c)
        )
        hc = arg.conjugate().transpose()
        return sp.csc_matrix(cp.array(splinalg.expm((arg - hc).todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def bs(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Two-mode beam splitter operator"""
        arg = theta * self.get_op("ad0 a1", cutoff_a, cutoff_b)
        hc = arg.conjugate().transpose()
        return sp.csc_matrix(cp.array(splinalg.expm((arg - hc).todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def cr(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Controlled phase space rotation operator"""
        arg = theta * 1j * sp.kron(self.Z, self.get_N(cutoff), format="csc")
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def crx(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Controlled phase space rotation operator around sigma^x"""
        arg = theta * 1j * sp.kron(self.X, self.get_N(cutoff), format="csc")
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def cry(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Controlled phase space rotation operator around sigma^y"""
        arg = theta * 1j * sp.kron(self.Y, self.get_N(cutoff), format="csc")
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def cd(self, alpha: complex, beta: complex, cutoff: int) -> sp.csc_matrix:
        """Controlled displacement operator"""
        displace0 = self.d(alpha, cutoff)
        displace1 = self.d(beta, cutoff)
        res = sp.kron(self.P0, displace0) + sp.kron(self.P1, displace1)
        return res.tocsc()

    @measure_performance()
    def ecd(self, theta: complex, cutoff: int) -> sp.csc_matrix:
        """Echoed controlled displacement operator"""
        return self.cd(theta, -theta, cutoff)

    @measure_performance()
    def cbs(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Controlled phase two-mode beam splitter operator"""
        arg = self.bs(theta, cutoff_a, cutoff_b)
        res = sp.kron(self.P0, arg) + sp.kron(self.P1, arg.conjugate().transpose())
        return res.tocsc()

    @measure_performance()
    def snap(self, theta: float, n: int, cutoff: int) -> sp.csc_matrix:
        """Single Fock state selective phase operation"""
        # 使用密集矩阵来构建 SNAP 算符
        eye = cp.eye(cutoff, dtype=cp.complex128)
        eye[n, n] = cp.exp(1j * theta)
        return sp.csc_matrix(eye)

    @measure_performance()
    def csnap(self, theta: float, n: int, cutoff: int) -> sp.csc_matrix:
        """Controlled single Fock state selective phase operation"""
        # 使用密集矩阵来构建受控 SNAP 算符
        eye = cp.eye(cutoff, dtype=cp.complex128)
        op = cp.eye(cutoff, dtype=cp.complex128)
        op[n, n] = cp.exp(1j * theta)
        # Tensor product with the qubit control (apply op when qubit is 1)
        return sp.kron(self.P0, sp.csc_matrix(eye)) + sp.kron(self.P1, sp.csc_matrix(op))

    @measure_performance()
    def multisnap(self, phase_map, cutoff: int) -> sp.csc_matrix:
        """Multi Fock state selective phase operation"""
        # 使用密集矩阵来构建多态 SNAP 算符
        eye = cp.eye(cutoff, dtype=cp.complex128)
        for n, theta in phase_map.items():
            eye[n, n] = cp.exp(1j * theta)
        return sp.csc_matrix(eye)

    @measure_performance()
    def multicsnap(self, phase_map, cutoff: int) -> sp.csc_matrix:
        """Controlled multi Fock state selective phase operation"""
        # 使用密集矩阵来构建受控多态 SNAP 算符
        eye = cp.eye(cutoff, dtype=cp.complex128)
        op = cp.eye(cutoff, dtype=cp.complex128)
        for n, theta in phase_map.items():
            op[n, n] = cp.exp(1j * theta)
        # Tensor product with the qubit control (apply op when qubit is 1)
        return sp.kron(self.P0, sp.csc_matrix(eye)) + sp.kron(self.P1, sp.csc_matrix(op))

    @measure_performance()
    def sqr(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Squared gate (SQR gate)"""
        # Build block diagonal matrix for SQR gate
        dim = 2 * cutoff
        block_dim = 2  # Each block is 2x2
        num_blocks = dim // block_dim
        # 使用密集矩阵来构建块对角矩阵
        blocks = cp.eye(dim, dtype=cp.complex128)

        # Fill blocks with RGate for each position
        for i in range(num_blocks):
            # Create RGate for this block
            rgate = cp.array([[cp.cos(theta / 2), -cp.sin(theta / 2)], 
                             [cp.sin(theta / 2), cp.cos(theta / 2)]], dtype=cp.complex128)
            # Insert block into the appropriate position
            blocks[i*block_dim:(i+1)*block_dim, i*block_dim:(i+1)*block_dim] = rgate

        # Swap qubits and qumodes to get the correct order
        # This is done by reordering rows and columns
        # The new order interleaves qubits and qumodes differently
        row_indices = []
        col_indices = []
        for i in range(cutoff):
            row_indices.append(i * 2)
            row_indices.append(cutoff + i)
            col_indices.append(i * 2)
            col_indices.append(cutoff + i)

        # Apply permutation to the blocks matrix
        permutation = cp.eye(dim, dtype=cp.complex128)
        permutation = permutation[:, row_indices]
        permutation = permutation[col_indices, :]
        result = permutation @ blocks @ permutation

        return sp.csc_matrix(result)

    @measure_performance()
    def pnr(self, max: int, cutoff: int) -> sp.csc_matrix:
        """Support gate for photon number readout"""
        # 使用 coo_matrix 来构建投影器
        rows = []
        cols = []
        data = []
        # binary search
        for j in range(max // 2):
            for i in range(j, cutoff, max):
                # fill from right to left
                n = cutoff - (i + 1)
                rows.append(n)
                cols.append(n)
                data.append(1.0)
        
        # 创建投影器
        projector = sp.coo_matrix((cp.array(data), (cp.array(rows), cp.array(cols))), 
                                shape=(cutoff, cutoff), dtype=cp.complex128).tocsc()

        # Flip qubit if there is a boson present in any of the modes addressed by the projector
        arg = 1j * (-cp.pi / 2) * sp.kron(self.X, projector).tocsc()
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def eswap(self, theta, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Exponential SWAP operator"""
        dim = cutoff_a * cutoff_b
        m = cp.arange(cutoff_a)[:, None]
        n = cp.arange(cutoff_b)[None, :]

        row_indices = n + (m * cutoff_a)
        col_indices = (n * cutoff_b) + m

        data = cp.ones(dim)
        swap = sp.coo_matrix(
            (data, (row_indices.flatten(), col_indices.flatten())), shape=(dim, dim), dtype=cp.complex128
        )
        return sp.csc_matrix(cp.array(splinalg.expm(1j * theta * swap.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def csq(self, theta: complex, cutoff: int) -> sp.csc_matrix:
        """Single-mode squeezing operator"""
        a = self.get_a(cutoff)
        a2 = a @ a
        arg = 0.5 * cp.conj(theta) * a2
        hc = arg.conjugate().transpose()
        arg = sp.kron(self.Z, arg - hc)

        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def c_multiboson_sampling(self, max: int, cutoff: int) -> sp.csc_matrix:
        """SNAP gate creation for multiboson sampling purposes."""
        return self.get_eye(cutoff)

    @measure_performance()
    def sum(self, scale: float, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Two-qumode sum gate"""
        a_mat = self.get_a(cutoff_a) + self.get_ad(cutoff_a)
        b_mat = self.get_ad(cutoff_b) - self.get_a(cutoff_b)
        arg = (scale / 2) * sp.kron(a_mat, b_mat)
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def csum(self, scale: float, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Conditional two-qumode sum gate"""
        a_mat = self.get_a(cutoff_a) + self.get_ad(cutoff_a)
        b_mat = self.get_ad(cutoff_b) - self.get_a(cutoff_b)
        arg = (scale / 2) * sp.kron(a_mat, b_mat)
        arg = sp.kron(self.Z, arg)
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def jc(self, theta: float, phi: float, cutoff: int) -> sp.csc_matrix:
        """Jaynes-Cummings gate"""
        arg = cp.exp(1j * phi) * sp.kron(self.SMINUS, self.get_ad(cutoff))
        hc = arg.conjugate().transpose()
        arg = -1j * theta * (arg + hc)
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def ajc(self, theta: float, phi: float, cutoff: int) -> sp.csc_matrix:
        """Anti-Jaynes-Cummings gate"""
        arg = cp.exp(1j * phi) * sp.kron(self.SPLIT, self.get_ad(cutoff))
        hc = arg.conjugate().transpose()
        arg = -1j * theta * (arg + hc)
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def rb(self, theta: complex, cutoff: int) -> sp.csc_matrix:
        """Rabi interaction gate"""
        arg = theta * self.get_ad(cutoff)
        hc = arg.conjugate().transpose()
        arg = sp.kron(self.X, -1j * (arg + hc)).tocsc()
        return sp.csc_matrix(cp.array(splinalg.expm(arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU

    @measure_performance()
    def cschwinger(
        self,
        beta: float,
        theta_1: float,
        phi_1: float,
        theta_2: float,
        phi_2: float,
        cutoff_a: int,
        cutoff_b: int,
    ) -> sp.csc_matrix:
        """General form of a controlled Schwinger gate"""
        # Sx = (self.get_op("a0 ad1", cutoff_a, cutoff_b) + self.get_op("ad0 a1", cutoff_a, cutoff_b)) / 2
        Sx = self.get_op("a0 ad1", cutoff_a, cutoff_b)
        Sx = (Sx + Sx.conjugate().transpose()) / 2

        # Sy = (self.get_op("a0 ad1", cutoff_a, cutoff_b) - self.get_op("ad0 a1", cutoff_a, cutoff_b)) / (2 * 1j)
        Sy = self.get_op("a0 ad1", cutoff_a, cutoff_b)
        Sy = (Sy - Sy.conjugate().transpose()) / 2j

        # Sz = (self.get_N(cutoff_b) - self.get_N(cutoff_a)) / 2
        Sz = (sp.kron(self.get_eye(cutoff_a), self.get_N(cutoff_b)) - sp.kron(self.get_N(cutoff_a), self.get_eye(cutoff_b))) / 2

        sigma = (
            cp.sin(theta_1) * cp.cos(phi_1) * self.X
            + cp.sin(theta_1) * cp.sin(phi_1) * self.Y
            + cp.cos(theta_1) * self.Z
        )
        S = (
            cp.sin(theta_2) * cp.cos(phi_2) * Sx
            + cp.sin(theta_2) * cp.sin(phi_2) * Sy
            + cp.cos(theta_2) * Sz
        )
        arg = sp.kron(sigma, S).tocsc()

        return sp.csc_matrix(cp.array(splinalg.expm(-1j * beta * arg.todense().get())))  # 转换到 CPU 计算，再转回 GPU


def print_performance_summary():
    """
    打印所有门操作的性能测量结果，以表格形式展示
    """
    print("=" * 85)
    print("         门操作性能测量结果")
    print("=" * 85)
    print(f"{'门类型':<30} {'单个门延迟(ms)':<20} {'批处理吞吐量(门/秒)':<25} {'内存效率':<15}")
    print("-" * 85)
    
    for gate_name, metrics in gate_performance_data.items():
        latency = metrics["single_gate_latency_ms"]
        throughput = metrics["batch_throughput"]
        efficiency = metrics["memory_efficiency"]
        
        print(f"{gate_name:<30} {latency:<20.6f} {throughput:<25.2f} {efficiency:<15.4f}")
    
    print("=" * 85)


def save_performance_to_json(filename="gate_performance_data.json"):
    """
    将性能测量结果保存到JSON文件
    
    参数:
        filename: 保存JSON数据的文件名
    """
    try:
        # 构建与C++性能测试文件格式一致的JSON结构
        results = []
        for gate_name, metrics in gate_performance_data.items():
            gate_data = {
                "gate_name": gate_name,
                "single_gate_latency_ms": float(metrics["single_gate_latency_ms"]),
                "batch_throughput": float(metrics["batch_throughput"]),
                "memory_efficiency": float(metrics["memory_efficiency"])
            }
            results.append(gate_data)
        
        # 构建完整的JSON对象
        performance_data = {
            "simulator": "CVOperatorsGPU",
            "truncation_dimension": 16,  # 使用与C++相同的截断维度
            "timestamp": int(time.time()),
            "results": results
        }
        
        # 保存到JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, ensure_ascii=False, indent=4)
        
        print(f"性能数据已成功保存到文件: {filename}")
    except Exception as e:
        print(f"保存性能数据到JSON文件时出错: {e}")


# 示例用法
if __name__ == "__main__":
    # 创建 CVOperatorsGPU 实例
    cv_ops = CVOperatorsGPU()
    
    # 设置截断参数
    cutoff = 16
    
    # 测试各种门操作
    print("测试单个门操作...")
    
    # 单模门
    print("\n测试单模门...")
    cv_ops.r(0.5, cutoff)
    cv_ops.d(1.0 + 0.5j, cutoff)
    cv_ops.s(0.3 + 0.2j, cutoff)
    
    # 双模门
    print("\n测试双模门...")
    cv_ops.bs(0.5, cutoff, cutoff)
    cv_ops.s2(0.3 + 0.2j, cutoff, cutoff)
    cv_ops.eswap(0.5, cutoff, cutoff)
    
    # 三模门
    print("\n测试三模门...")
    cv_ops.s3(0.2 + 0.1j, cutoff, cutoff, cutoff)
    
    # 受控门
    print("\n测试受控门...")
    cv_ops.cr(0.5, cutoff)
    cv_ops.crx(0.5, cutoff)
    cv_ops.cry(0.5, cutoff)
    cv_ops.cd(1.0 + 0.5j, 0.5 - 0.3j, cutoff)
    cv_ops.ecd(1.0 + 0.5j, cutoff)
    cv_ops.cbs(0.5, cutoff, cutoff)
    
    # SNAP 相关门
    print("\n测试SNAP相关门...")
    cv_ops.snap(0.5, 3, cutoff)
    cv_ops.csnap(0.5, 3, cutoff)
    cv_ops.multisnap({0: 0.1, 1: 0.2, 2: 0.3}, cutoff)
    cv_ops.multicsnap({0: 0.1, 1: 0.2, 2: 0.3}, cutoff)
    
    # 其他门
    print("\n测试其他门...")
    cv_ops.sqr(0.5, cutoff)
    cv_ops.pnr(8, cutoff)
    cv_ops.csq(0.3 + 0.2j, cutoff)
    cv_ops.c_multiboson_sampling(8, cutoff)
    cv_ops.sum(0.5, cutoff, cutoff)
    cv_ops.csum(0.5, cutoff, cutoff)
    cv_ops.jc(0.5, 0.3, cutoff)
    cv_ops.ajc(0.5, 0.3, cutoff)
    cv_ops.rb(0.3 + 0.2j, cutoff)
    cv_ops.cschwinger(0.5, 0.3, 0.2, 0.4, 0.1, cutoff, cutoff)
    
    # 打印性能总结
    print("\n性能测量结果:")
    print_performance_summary()
    
    # 保存性能数据到JSON文件
    save_performance_to_json()