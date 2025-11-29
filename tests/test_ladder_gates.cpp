#include <gtest/gtest.h>
#include "cv_state_pool.h"
#include "reference_gates.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// 声明外部函数
extern void apply_creation_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_annihilation_operator(CVStatePool* pool, const int* targets, int batch_size);

/**
 * 梯算符门操作单元测试
 */
class LadderGatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        d_trunc = 8;
        max_states = 4;
        pool = new CVStatePool(d_trunc, max_states);

        // 初始化测试状态
        state_id = pool->allocate_state();
        std::vector<cuDoubleComplex> initial_state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
        initial_state[0] = make_cuDoubleComplex(1.0, 0.0);  // |0⟩状态
        pool->upload_state(state_id, initial_state);
    }

    void TearDown() override {
        delete pool;
    }

    int d_trunc;
    int max_states;
    CVStatePool* pool;
    int state_id;

    /**
     * 辅助函数：将cuDoubleComplex向量转换为std::complex<double>向量
     */
    Reference::Vector to_reference_vector(const std::vector<cuDoubleComplex>& cuda_vec) {
        Reference::Vector ref_vec;
        for (const auto& val : cuda_vec) {
            ref_vec.emplace_back(cuCreal(val), cuCimag(val));
        }
        return ref_vec;
    }

    /**
     * 辅助函数：对比GPU实现与参考实现的误差
     */
    void compare_with_reference(const std::vector<cuDoubleComplex>& gpu_result,
                               const Reference::Vector& ref_result,
                               double max_allowed_error = 1e-10) {
        auto gpu_as_ref = to_reference_vector(gpu_result);
        auto error_metrics = Reference::compute_error_metrics(ref_result, gpu_as_ref);

        std::cout << "误差指标:" << std::endl;
        std::cout << "  L2误差: " << error_metrics.l2_error << std::endl;
        std::cout << "  最大误差: " << error_metrics.max_error << std::endl;
        std::cout << "  相对误差: " << error_metrics.relative_error << std::endl;
        std::cout << "  保真度偏差: " << error_metrics.fidelity_deviation << std::endl;

        EXPECT_LT(error_metrics.l2_error, max_allowed_error);
        EXPECT_LT(error_metrics.max_error, max_allowed_error);
        EXPECT_LT(error_metrics.relative_error, max_allowed_error);
        EXPECT_LT(error_metrics.fidelity_deviation, max_allowed_error);
    }

    /**
     * 辅助函数：调用GPU门函数，处理target_indices的GPU内存分配
     */
    template <typename Func, typename... Args>
    void call_gpu_gate(Func func, const std::vector<int>& targets, Args... args) {
        int* d_targets;
        cudaError_t err = cudaMalloc(&d_targets, targets.size() * sizeof(int));
        if (err != cudaSuccess) {
            FAIL() << "Failed to allocate device memory for targets: " << cudaGetErrorString(err);
            return;
        }
        
        err = cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_targets);
            FAIL() << "Failed to copy targets to device: " << cudaGetErrorString(err);
            return;
        }
        
        func(pool, d_targets, targets.size(), args...);
        
        // 同步并检查错误
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_targets);
            FAIL() << "GPU kernel execution failed: " << cudaGetErrorString(err);
        }
        
        cudaFree(d_targets);
    }
};

TEST_F(LadderGatesTest, CreationOperatorVacuum) {
    // 测试创建算符作用在真空上: a†|0⟩ = |1⟩
    std::vector<cuDoubleComplex> input_state;
    pool->download_state(state_id, input_state);

    // 参考实现
    auto ref_input = to_reference_vector(input_state);
    auto ref_result = Reference::LadderGates::apply_creation_operator(ref_input);

    // GPU实现
    call_gpu_gate(apply_creation_operator, {state_id});

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    compare_with_reference(gpu_result, ref_result);
}

TEST_F(LadderGatesTest, CreationOperatorFockState) {
    // 测试创建算符作用在Fock态上: a†|2⟩ = √3|3⟩
    std::vector<cuDoubleComplex> fock_state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    fock_state[2] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id, fock_state);

    // 参考实现
    auto ref_input = to_reference_vector(fock_state);
    auto ref_result = Reference::LadderGates::apply_creation_operator(ref_input);

    // GPU实现
    call_gpu_gate(apply_creation_operator, {state_id});

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    compare_with_reference(gpu_result, ref_result);
}

TEST_F(LadderGatesTest, AnnihilationOperatorVacuum) {
    // 测试湮灭算符作用在真空上: a|0⟩ = 0
    std::vector<cuDoubleComplex> input_state;
    pool->download_state(state_id, input_state);

    // 参考实现
    auto ref_input = to_reference_vector(input_state);
    auto ref_result = Reference::LadderGates::apply_annihilation_operator(ref_input);

    // GPU实现
    call_gpu_gate(apply_annihilation_operator, {state_id});

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    // 对于真空态湮灭，结果应为零向量，保真度未定义或为0，因此跳过保真度检查
    // 仅检查L2误差（确保结果为零向量）
    auto ref_result_vec = to_reference_vector(gpu_result);
    auto error_metrics = Reference::compute_error_metrics(ref_result, ref_result_vec);
    
    std::cout << "误差指标:" << std::endl;
    std::cout << "  L2误差: " << error_metrics.l2_error << std::endl;
    std::cout << "  最大误差: " << error_metrics.max_error << std::endl;
    
    EXPECT_LT(error_metrics.l2_error, 1e-10);
    EXPECT_LT(error_metrics.max_error, 1e-10);
}

TEST_F(LadderGatesTest, AnnihilationOperatorFockState) {
    // 测试湮灭算符作用在Fock态上: a|2⟩ = √2|1⟩
    std::vector<cuDoubleComplex> fock_state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    fock_state[2] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id, fock_state);

    // 参考实现
    auto ref_input = to_reference_vector(fock_state);
    auto ref_result = Reference::LadderGates::apply_annihilation_operator(ref_input);

    // GPU实现
    call_gpu_gate(apply_annihilation_operator, {state_id});

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 对比误差
    compare_with_reference(gpu_result, ref_result);
}

TEST_F(LadderGatesTest, CreationAnnihilationCommutation) {
    // 测试对易关系: [a, a†] = 1
    // a†a|2⟩ = 2|2⟩, aa†|2⟩ = 3|2⟩

    std::vector<cuDoubleComplex> fock_state(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    fock_state[2] = make_cuDoubleComplex(1.0, 0.0);
    pool->upload_state(state_id, fock_state);

    // 先应用a，再应用a†
    // a|2⟩ = √2|1⟩
    call_gpu_gate(apply_annihilation_operator, {state_id});

    // a†(a|2⟩) = a†(√2|1⟩) = √2 * √2 |2⟩ = 2|2⟩
    call_gpu_gate(apply_creation_operator, {state_id});

    std::vector<cuDoubleComplex> gpu_result;
    pool->download_state(state_id, gpu_result);

    // 结果应该是 2|2⟩
    EXPECT_NEAR(cuCreal(gpu_result[2]), 2.0, 1e-10);
    EXPECT_NEAR(cuCimag(gpu_result[2]), 0.0, 1e-10);

    // 其他分量应该为0
    for (int i = 0; i < d_trunc; ++i) {
        if (i != 2) {
            EXPECT_NEAR(cuCreal(gpu_result[i]), 0.0, 1e-10);
            EXPECT_NEAR(cuCimag(gpu_result[i]), 0.0, 1e-10);
        }
    }
}

TEST_F(LadderGatesTest, BatchProcessing) {
    // 测试批处理多个状态
    int state_id2 = pool->allocate_state();
    std::vector<cuDoubleComplex> state2(d_trunc, make_cuDoubleComplex(0.0, 0.0));
    state2[1] = make_cuDoubleComplex(1.0, 0.0);  // |1⟩状态
    pool->upload_state(state_id2, state2);

    call_gpu_gate(apply_creation_operator, {state_id, state_id2});

    // 验证第一个状态: a†|0⟩ = |1⟩
    std::vector<cuDoubleComplex> result1;
    pool->download_state(state_id, result1);
    EXPECT_NEAR(cuCreal(result1[1]), 1.0, 1e-10);

    // 验证第二个状态: a†|1⟩ = √2|2⟩
    std::vector<cuDoubleComplex> result2;
    pool->download_state(state_id2, result2);
    EXPECT_NEAR(cuCreal(result2[2]), std::sqrt(2.0), 1e-10);

    pool->free_state(state_id2);
}
