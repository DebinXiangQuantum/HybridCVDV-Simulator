#include <gtest/gtest.h>
#include "cv_state_pool.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// 声明外部函数
extern void apply_number_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_creation_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_annihilation_operator(CVStatePool* pool, const int* targets, int batch_size);

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
 * 数算符单元测试
 */
class NumberOperatorTest : public ::testing::Test {
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

TEST_F(NumberOperatorTest, VacuumState) {
    // 测试数算符作用在真空态上: n|0⟩ = 0|0⟩ = 0
    set_fock_state(0);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_number_operator(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |0⟩ 应该变成 0|0⟩ = 0
    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);

    // 其他分量也应该为 0
    for (int i = 1; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), 0.0, 1e-10);
        EXPECT_NEAR(cuCimag(result[i]), 0.0, 1e-10);
    }
}

TEST_F(NumberOperatorTest, SinglePhotonState) {
    // 测试数算符作用在单光子态上: n|1⟩ = 1|1⟩ = |1⟩
    set_fock_state(1);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_number_operator(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |1⟩ 应该变成 1|1⟩ = |1⟩
    EXPECT_NEAR(cuCreal(result[1]), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);

    // 其他分量应该为 0
    for (int i = 0; i < d_trunc; ++i) {
        if (i != 1) {
            EXPECT_NEAR(cuCreal(result[i]), 0.0, 1e-10);
            EXPECT_NEAR(cuCimag(result[i]), 0.0, 1e-10);
        }
    }

    expect_normalized(result);
}

TEST_F(NumberOperatorTest, TwoPhotonState) {
    // 测试数算符作用在双光子态上: n|2⟩ = 2|2⟩
    set_fock_state(2);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_number_operator(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |2⟩ 应该变成 2|2⟩
    EXPECT_NEAR(cuCreal(result[2]), 2.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[2]), 0.0, 1e-10);

    // 其他分量应该为 0
    for (int i = 0; i < d_trunc; ++i) {
        if (i != 2) {
            EXPECT_NEAR(cuCreal(result[i]), 0.0, 1e-10);
            EXPECT_NEAR(cuCimag(result[i]), 0.0, 1e-10);
        }
    }
}

TEST_F(NumberOperatorTest, HighPhotonNumberState) {
    // 测试数算符作用在高光子数态上: n|5⟩ = 5|5⟩
    set_fock_state(5);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_number_operator(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |5⟩ 应该变成 5|5⟩
    EXPECT_NEAR(cuCreal(result[5]), 5.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[5]), 0.0, 1e-10);

    // 其他分量应该为 0
    for (int i = 0; i < d_trunc; ++i) {
        if (i != 5) {
            EXPECT_NEAR(cuCreal(result[i]), 0.0, 1e-10);
            EXPECT_NEAR(cuCimag(result[i]), 0.0, 1e-10);
        }
    }
}

TEST_F(NumberOperatorTest, SuperpositionState) {
    // 测试数算符作用在叠加态上
    // n(|0⟩ + |1⟩ + |2⟩)/√3 = (0|0⟩ + 1|1⟩ + 2|2⟩)/√3
    set_superposition_state();

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_number_operator(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    double norm = 1.0 / sqrt(3.0);
    // |0⟩ → 0|0⟩ = 0
    EXPECT_NEAR(cuCreal(result[0]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[0]), 0.0, 1e-10);
    // |1⟩ → 1|1⟩
    EXPECT_NEAR(cuCreal(result[1]), norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
    // |2⟩ → 2|2⟩
    EXPECT_NEAR(cuCreal(result[2]), 2.0 * norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[2]), 0.0, 1e-10);

    // 其他分量应该为 0
    for (int i = 3; i < d_trunc; ++i) {
        EXPECT_NEAR(cuCreal(result[i]), 0.0, 1e-10);
        EXPECT_NEAR(cuCimag(result[i]), 0.0, 1e-10);
    }
}

TEST_F(NumberOperatorTest, ComplexSuperposition) {
    // 测试数算符作用在复数叠加态上
    std::vector<cuDoubleComplex> state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    double norm = 1.0 / sqrt(2.0);
    state[1] = make_cuDoubleComplex(norm, 0.0);
    state[3] = make_cuDoubleComplex(0.0, norm);  // i|3⟩
    pool->upload_state(state_id, state);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    apply_number_operator(pool, d_targets, 1);

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result;
    pool->download_state(state_id, result);

    // |1⟩ → 1|1⟩
    EXPECT_NEAR(cuCreal(result[1]), norm, 1e-10);
    EXPECT_NEAR(cuCimag(result[1]), 0.0, 1e-10);
    // i|3⟩ → 3i|3⟩
    EXPECT_NEAR(cuCreal(result[3]), 0.0, 1e-10);
    EXPECT_NEAR(cuCimag(result[3]), 3.0 * norm, 1e-10);
}

TEST_F(NumberOperatorTest, CommutationWithCreation) {
    // 测试对易关系: [n, a†] = a†
    // 即 n a† - a† n = a†
    // 验证: n a†|2⟩ = a† n|2⟩ + a†|2⟩
    
    set_fock_state(2);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    // 先应用 a†，再应用 n
    apply_creation_operator(pool, d_targets, 1);  // a†|2⟩ = √3|3⟩
    apply_number_operator(pool, d_targets, 1);    // n(√3|3⟩) = 3√3|3⟩

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result_na_dag;
    pool->download_state(state_id, result_na_dag);

    // 重置状态
    set_fock_state(2);
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    // 先应用 n，再应用 a†
    apply_number_operator(pool, d_targets, 1);    // n|2⟩ = 2|2⟩
    apply_creation_operator(pool, d_targets, 1);  // a†(2|2⟩) = 2√3|3⟩

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result_a_dag_n;
    pool->download_state(state_id, result_a_dag_n);

    // 验证 n a† - a† n = a†
    // n a†|2⟩ = 3√3|3⟩
    // a† n|2⟩ = 2√3|3⟩
    // 差值应该是 √3|3⟩ = a†|2⟩
    double sqrt3 = sqrt(3.0);
    EXPECT_NEAR(cuCreal(result_na_dag[3]) - cuCreal(result_a_dag_n[3]), sqrt3, 1e-10);
}

TEST_F(NumberOperatorTest, CommutationWithAnnihilation) {
    // 测试对易关系: [n, a] = -a
    // 即 n a - a n = -a
    // 验证: n a|2⟩ = a n|2⟩ - a|2⟩
    
    set_fock_state(2);

    int host_targets[] = {state_id};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    // 先应用 a，再应用 n
    apply_annihilation_operator(pool, d_targets, 1);  // a|2⟩ = √2|1⟩
    apply_number_operator(pool, d_targets, 1);        // n(√2|1⟩) = √2|1⟩

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result_na;
    pool->download_state(state_id, result_na);

    // 重置状态
    set_fock_state(2);
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

    // 先应用 n，再应用 a
    apply_number_operator(pool, d_targets, 1);        // n|2⟩ = 2|2⟩
    apply_annihilation_operator(pool, d_targets, 1);  // a(2|2⟩) = 2√2|1⟩

    FREE_DEVICE_TARGETS(d_targets);

    std::vector<cuDoubleComplex> result_an;
    pool->download_state(state_id, result_an);

    // 验证 n a - a n = -a
    // n a|2⟩ = √2|1⟩
    // a n|2⟩ = 2√2|1⟩
    // 差值应该是 -√2|1⟩ = -a|2⟩
    double sqrt2 = sqrt(2.0);
    EXPECT_NEAR(cuCreal(result_na[1]) - cuCreal(result_an[1]), -sqrt2, 1e-10);
}

TEST_F(NumberOperatorTest, ExpectationValue) {
    // 测试期望值计算
    // 对于 Fock 态 |n⟩，⟨n|n̂|n⟩ = n
    
    for (int n = 0; n < 5; ++n) {
        set_fock_state(n);

        std::vector<cuDoubleComplex> original;
        pool->download_state(state_id, original);

        int host_targets[] = {state_id};
        int* d_targets = nullptr;
        ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

        apply_number_operator(pool, d_targets, 1);

        FREE_DEVICE_TARGETS(d_targets);

        std::vector<cuDoubleComplex> result;
        pool->download_state(state_id, result);

        // 计算期望值 ⟨ψ|n̂|ψ⟩
        cuDoubleComplex expectation = make_cuDoubleComplex(0.0, 0.0);
        for (int i = 0; i < d_trunc; ++i) {
            // ⟨ψ|n̂|ψ⟩ = Σ ψ*[i] · (n̂ψ)[i]
            cuDoubleComplex conj_orig = cuConj(original[i]);
            expectation = cuCadd(expectation, cuCmul(conj_orig, result[i]));
        }

        // 期望值应该等于 n
        EXPECT_NEAR(cuCreal(expectation), static_cast<double>(n), 1e-10);
        EXPECT_NEAR(cuCimag(expectation), 0.0, 1e-10);
    }
}

TEST_F(NumberOperatorTest, BatchProcessing) {
    // 测试批处理多个状态
    int state_id2 = pool->allocate_state();
    
    set_fock_state(1);
    
    std::vector<cuDoubleComplex> state2(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    state2[3] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id2, state2);

    int host_targets[] = {state_id, state_id2};
    int* d_targets = nullptr;
    ALLOC_DEVICE_TARGETS(d_targets, host_targets, 2);

    apply_number_operator(pool, d_targets, 2);

    FREE_DEVICE_TARGETS(d_targets);

    // 验证第一个状态: n|1⟩ = |1⟩
    std::vector<cuDoubleComplex> result1;
    pool->download_state(state_id, result1);
    EXPECT_NEAR(cuCreal(result1[1]), 1.0, 1e-10);

    // 验证第二个状态: n|3⟩ = 3|3⟩
    std::vector<cuDoubleComplex> result2;
    pool->download_state(state_id2, result2);
    EXPECT_NEAR(cuCreal(result2[3]), 3.0, 1e-10);

    pool->free_state(state_id2);
}

TEST_F(NumberOperatorTest, IdentityRelation) {
    // 测试恒等关系: n = a† a
    // 验证: n|n⟩ = a† a|n⟩
    
    for (int n = 0; n < 5; ++n) {
        // 测试 n|n⟩
        set_fock_state(n);

        int host_targets[] = {state_id};
        int* d_targets = nullptr;
        ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

        apply_number_operator(pool, d_targets, 1);

        FREE_DEVICE_TARGETS(d_targets);

        std::vector<cuDoubleComplex> result_n;
        pool->download_state(state_id, result_n);

        // 测试 a† a|n⟩
        set_fock_state(n);
        ALLOC_DEVICE_TARGETS(d_targets, host_targets, 1);

        apply_annihilation_operator(pool, d_targets, 1);  // a|n⟩
        apply_creation_operator(pool, d_targets, 1);      // a†(a|n⟩)

        FREE_DEVICE_TARGETS(d_targets);

        std::vector<cuDoubleComplex> result_a_dag_a;
        pool->download_state(state_id, result_a_dag_a);

        // 两个结果应该相同
        for (int i = 0; i < d_trunc; ++i) {
            EXPECT_NEAR(cuCreal(result_n[i]), cuCreal(result_a_dag_a[i]), 1e-10);
            EXPECT_NEAR(cuCimag(result_n[i]), cuCimag(result_a_dag_a[i]), 1e-10);
        }
    }
}
