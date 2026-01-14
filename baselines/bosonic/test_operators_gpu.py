#!/usr/bin/env python3
import cupy as cp
from operators_gpu import CVOperatorsGPU

# 创建 CVOperatorsGPU 实例
operators = CVOperatorsGPU()

# 测试基本算子
def test_basic_operators():
    print("Testing basic operators...")
    cutoff = 5
    
    # 测试单位矩阵
    eye = operators.get_eye(cutoff)
    print(f"Identity matrix shape: {eye.shape}")
    
    # 测试湮灭算子
    a = operators.get_a(cutoff)
    print(f"Annihilation operator shape: {a.shape}")
    
    # 测试产生算子
    ad = operators.get_ad(cutoff)
    print(f"Creation operator shape: {ad.shape}")
    
    # 测试数量算子
    N = operators.get_N(cutoff)
    print(f"Number operator shape: {N.shape}")
    
    print("Basic operators test completed successfully!")

# 测试位移算子
def test_displacement():
    print("Testing displacement operator...")
    cutoff = 5
    alpha = 0.5
    
    d = operators.d(alpha, cutoff)
    print(f"Displacement operator shape: {d.shape}")
    
    # 测试受控位移算子
    cd = operators.cd(alpha, -alpha, cutoff)
    print(f"Controlled displacement operator shape: {cd.shape}")
    
    print("Displacement operator test completed successfully!")

# 测试压缩算子
def test_squeezing():
    print("Testing squeezing operator...")
    cutoff = 5
    theta = 0.1 + 0.2j
    
    # 测试单模压缩算子
    s = operators.s(theta, cutoff)
    print(f"Single-mode squeezing operator shape: {s.shape}")
    
    # 测试双模压缩算子
    s2 = operators.s2(theta, cutoff, cutoff)
    print(f"Two-mode squeezing operator shape: {s2.shape}")
    
    print("Squeezing operator test completed successfully!")

# 测试分束器算子
def test_beamsplitter():
    print("Testing beamsplitter operator...")
    cutoff = 5
    theta = 0.1
    
    # 测试分束器算子
    bs = operators.bs(theta, cutoff, cutoff)
    print(f"Beamsplitter operator shape: {bs.shape}")
    
    print("Beamsplitter operator test completed successfully!")

# 测试 SNAP 算子
def test_snap():
    print("Testing SNAP operator...")
    cutoff = 5
    theta = 0.1
    n = 2
    
    # 测试 SNAP 算子
    snap = operators.snap(theta, n, cutoff)
    print(f"SNAP operator shape: {snap.shape}")
    
    print("SNAP operator test completed successfully!")

# 测试符号算子
def test_symbolic():
    print("Testing symbolic operator construction...")
    cutoff = 3
    
    # 测试符号算子构造
    op = operators.get_op("ad0 a1", cutoff, cutoff)
    print(f"Symbolic operator shape: {op.shape}")
    
    print("Symbolic operator test completed successfully!")

# 运行所有测试
def run_all_tests():
    print("Starting GPU operators tests...")
    print("=" * 50)
    
    test_basic_operators()
    print()
    
    test_displacement()
    print()
    
    test_squeezing()
    print()
    
    test_beamsplitter()
    print()
    
    test_snap()
    print()
    
    test_symbolic()
    print()
    
    print("=" * 50)
    print("All tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
