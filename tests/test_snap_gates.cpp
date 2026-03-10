#include <gtest/gtest.h>
#include "cv_state_pool.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// 声明外部函数
extern void apply_snap(CVStatePool* pool, const int* targets, int batch_size, 
                      double theta, int target_fock_state);
extern void apply_multisnap(CVStatePool* pool, const int* targets, int batch_size,
                           const std::vector<double>& phase_map);
extern void apply_csnap(CVStatePool* pool, const int* targets, int batch_size,
                       double theta, int target_fock_state, int cutoff);
extern void apply_cmultisnap(CVStatePool* pool, const int* targets, int batch_size,
                            const std::vector<double>& phase_map, int cutoff);

// 辅助宏
#define ALLOC_DEVICE_TARGETS(d_ptr, host_array, size) \
    do { \
        cudaMalloc(&(d_ptr), (size) * sizeof(int)); \
        cudaMemcpy((d_ptr), (host_array), (size) * sizeof(int), cudaMemcpyHostToDevice); \
    } while(0)

#define FREE_DEVICE_TARGETS(d_ptr) \
    do { \
        cudaFree(d_ptr); \
        (d_ptr) = nullptr; \
    } while(0)

/**
 * SNAP 门操作单元测试
 */
class SNAPGatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        d_trunc = 8;
        max_states = 4;
        pool = new CVStatePool(d_trunc, max_states, 1);

        // 初始化测试状态
        state_id = pool->allocate_state();
    }

    void TearDown() override {
        delete pool;
    }

    int d_trunc;
    int max_states;
    CVStatePool* pool;
    int state_id;

    // 辅助函数：设置 Fock 态 |n⟩
    void set_fock_state(int n) {
        std::vector<cuDoubleComplex> state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
        if (n < d_trunc) {
            state[n] = make_cuDoubleComplex(1.0, 0.0);
        }
        pool->upload_state(state_id, state);
    }

    // 辅助函数：设置叠加态
    void set_superposition_state() {
        std::vector<cuDoubleComplex> state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
        double norm = 1.0 / sqrt(3.0);
        state[0] = make_cuDoubleComplex(norm, 0.0);
        state[1] = make_cuDoubleComplex(norm, 0.0);
        state[2] = make_cuDoubleComplex(norm, 0.0);
        pool->upload_state(state_id, state);
    }

    // 辅助函数：验证归一化
    void expect_normalized(const std::vector<cuDoubleComplex>& state, double tolerance = 1e-10) {
        double norm_sq = 0.0;
        for (const auto& amp : state) {
            double real = cuCreal(amp);
            double imag = cuCimag(amp);
            norm_sq += real * real + imag * imag;
        }
        EXPECT_NEAR(sqrt(norm_sq), 1.0, tolerance);
    }
};

TEST_F(SNAPGatesTest, SNAP_TargetFockState) {
    // 测试 SNAP 门对目标 Fock 态施加相位
    // SNAP(π, 2)|2⟩ = exp(iπ)|2⟩ = -|2⟩
    set_fock_state(2);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_snap(pool, d_targets, 1, M_PI, 2);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |2⟩ 应该变成 -|2⟩
    EXPECT_NEAR(cuCreal(result[2]), -1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[2]), 0.0, 1e-10);

    // 其他分量应该为 0
    for (int i = 0; i < d_trunc; ++i) {
        if (i != 2) {
            EXPECT_NEAR(cuCreal(result[i]), 0.0, 1e-10);
            EXPECT_NEAR(cuCimag(result[i]), 0.0, 1e-10);
        }
    }

    expect_normalized(result);
}

TEST_F(SNAPGatesTest, SNAP_NonTargetFockState) {
    // 测试 SNAP 门不影响非目标 Fock 态
    // SNAP(π, 2)|1⟩ = |1⟩
    set_fock_state(1);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_snap(pool, d_targets, 1, M_PI, 2);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |1⟩ 应该保持不变
    EXPECT_NEAR(cuCreal(result[1]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);

    expect_normalized(result);
}

TEST_F(SNAPGatesTest, SNAP_PiHalfPhase) {
    // 测试 π/2 相位
    // SNAP(π/2, 0)|0⟩ = exp(iπ/2)|0⟩ = i|0⟩
    set_fock_state(0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_snap(pool, d_targets, 1, M_PI / 2.0, 0);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |0⟩ 应该变成 i|0⟩
    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 1.0, 1e-10);

    expect_normalized(result);
}

TEST_F(SNAPGatesTest, SNAP_Superposition) {
    // 测试 SNAP 门作用在叠加态上
    // SNAP(π, 1)(|0⟩ + |1⟩ + |2⟩)/√3 = (|0⟩ - |1⟩ + |2⟩)/√3
    set_superposition_state();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_snap(pool, d_targets, 1, M_PI, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    double norm = 1.0 / sqrt(3.0);
    EXPECT_NEAR(cuCreal(result[0]), norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[1]), -norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
    EXPECT_NEAR(cuCreal(result[2]), norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[2]), 0.0, 1e-10);

    expect_normalized(result);
}

TEST_F(SNAPGatesTest, MultiSNAP_MultipleFockStates) {
    // 测试 Multi-SNAP 门对多个 Fock 态施加不同相位
    set_superposition_state();

    std::vector<double> phase_map(d_trunc, 0.0);
    phase_map[0] = 0.0;        // |0⟩ 不变
    phase_map[1] = M_PI;       // |1⟩ → -|1⟩
    phase_map[2] = M_PI / 2.0; // |2⟩ → i|2⟩

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_multisnap(pool, d_targets, 1, phase_map);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    double norm = 1.0 / sqrt(3.0);
    // |0⟩ 不变
    EXPECT_NEAR(cuCreal(result[0]), norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    // |1⟩ → -|1⟩
    EXPECT_NEAR(cuCreal(result[1]), -norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
    // |2⟩ → i|2⟩
    EXPECT_NEAR(cuCreal(result[2]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[2]), norm, 1e-10);

    expect_normalized(result);
}

TEST_F(SNAPGatesTest, MultiSNAP_ZeroPhases) {
    // 测试所有相位为 0 的 Multi-SNAP（应该是恒等操作）
    set_superposition_state();

    std::vector<cuDoubleComplex> original;
    pool->download_state(state_id, original);

    std::vector<double> phase_map(d_trunc, 0.0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_multisnap(pool, d_targets, 1, phase_map);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // 状态应该保持不变
    for (int i = 0; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), cuCreal(original[i]), 1e-10);
        EXPECT_NEAR(cuCimag(result[i]), cuCimag(original[i]), 1e-10);
    }
}

TEST_F(SNAPGatesTest, CSNAP_ControlQubit0) {
    // 测试受控 SNAP，控制比特为 |0⟩
    // 状态：|0, 2⟩（qubit=0, qumode=2）
    // CSNAP 不应该影响
    int cutoff = 4;
    int hybrid_dim = 2 * cutoff; // qubit ⊗ qumode
    
    // 重新分配状态池以支持混合维度
    delete pool;
    pool = new CVStatePool(hybrid_dim, max_states, 1);
    state_id = pool->allocate_state();

    std::vector<cuDoubleComplex> state(hybrid_dim, make_cuDoubleComplex(0.0, 0.0));
    state[2] = make_cuDoubleComplex(1.0, 0.0); // |0, 2⟩

    pool->upload_state(state_id, state);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_csnap(pool, d_targets, 1, M_PI, 2, cutoff);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |0, 2⟩ 应该保持不变
    EXPECT_NEAR(cuCreal(result[2]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[2]), 0.0, 1e-10);
}

TEST_F(SNAPGatesTest, CSNAP_ControlQubit1) {
    // 测试受控 SNAP，控制比特为 |1⟩
    // 状态：|1, 2⟩（qubit=1, qumode=2）
    // CSNAP(π, 2) 应该施加相位
    int cutoff = 4;
    int hybrid_dim = 2 * cutoff;
    
    delete pool;
    pool = new CVStatePool(hybrid_dim, max_states, 1);
    state_id = pool->allocate_state();

    std::vector<cuDoubleComplex> state(hybrid_dim, make_cuDoubleComplex(0.0, 0.0));
    state[cutoff + 2] = make_cuDoubleComplex(1.0, 0.0); // |1, 2⟩

    pool->upload_state(state_id, state);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_csnap(pool, d_targets, 1, M_PI, 2, cutoff);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |1, 2⟩ 应该变成 -|1, 2⟩
    EXPECT_NEAR(cuCreal(result[cutoff + 2]), -1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[cutoff + 2]), 0.0, 1e-10);
}

TEST_F(SNAPGatesTest, CSNAP_NonTargetQumode) {
    // 测试受控 SNAP 不影响非目标 qumode
    // 状态：|1, 1⟩，目标是 |1, 2⟩
    int cutoff = 4;
    int hybrid_dim = 2 * cutoff;
    
    delete pool;
    pool = new CVStatePool(hybrid_dim, max_states, 1);
    state_id = pool->allocate_state();

    std::vector<cuDoubleComplex> state(hybrid_dim, make_cuDoubleComplex(0.0, 0.0));
    state[cutoff + 1] = make_cuDoubleComplex(1.0, 0.0); // |1, 1⟩

    pool->upload_state(state_id, state);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_csnap(pool, d_targets, 1, M_PI, 2, cutoff);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |1, 1⟩ 应该保持不变
    EXPECT_NEAR(cuCreal(result[cutoff + 1]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[cutoff + 1]), 0.0, 1e-10);
}

TEST_F(SNAPGatesTest, CMultiSNAP_HybridSuperposition) {
    // 测试受控 Multi-SNAP 在混合叠加态上
    int cutoff = 4;
    int hybrid_dim = 2 * cutoff;
    
    delete pool;
    pool = new CVStatePool(hybrid_dim, max_states, 1);
    state_id = pool->allocate_state();

    // 创建叠加态：(|1,0⟩ + |1,1⟩ + |1,2⟩)/√3
    std::vector<cuDoubleComplex> state(hybrid_dim, make_cuDoubleComplex(0.0, 0.0));
    double norm = 1.0 / sqrt(3.0);
    state[cutoff + 0] = make_cuDoubleComplex(norm, 0.0);
    state[cutoff + 1] = make_cuDoubleComplex(norm, 0.0);
    state[cutoff + 2] = make_cuDoubleComplex(norm, 0.0);

    pool->upload_state(state_id, state);

    std::vector<double> phase_map(cutoff, 0.0);
    phase_map[0] = 0.0;
    phase_map[1] = M_PI;
    phase_map[2] = M_PI / 2.0;

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_cmultisnap(pool, d_targets, 1, phase_map, cutoff);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |1,0⟩ 不变
    EXPECT_NEAR(cuCreal(result[cutoff + 0]), norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[cutoff + 0]), 0.0, 1e-10);
    // |1,1⟩ → -|1,1⟩
    EXPECT_NEAR(cuCreal(result[cutoff + 1]), -norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[cutoff + 1]), 0.0, 1e-10);
    // |1,2⟩ → i|1,2⟩
    EXPECT_NEAR(cuCreal(result[cutoff + 2]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[cutoff + 2]), norm, 1e-10);
}

TEST_F(SNAPGatesTest, BatchProcessing) {
    // 测试批处理多个状态
    int state_id2 = pool->allocate_state();
    
    set_fock_state(1);
    
    std::vector<cuDoubleComplex> state2(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    state2[2] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id2, state2);

    int host_targets[] = {state_id, state_id2};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 2);

    apply_snap(pool, d_targets, 2, M_PI, 1);

    FREE_DEVICE_TARGETS(d_targets);

    // 验证第一个状态: SNAP(π, 1)|1⟩ = -|1⟩
    std::vector<cuDoubleComplex> result1;
    pool->download_state(state_id, result1);
    EXPECT_NEAR(cuCreal(result1[1]), -1.0, 1e-10);

    // 验证第二个状态: SNAP(π, 1)|2⟩ = |2⟩ (不受影响)
    std::vector<cuDoubleComplex> result2;
    pool->download_state(state_id2, result2);
    EXPECT_NEAR(cuCreal(result2[2]), 1.0, 1e-10);

    pool->free_state(state_id2);
}
