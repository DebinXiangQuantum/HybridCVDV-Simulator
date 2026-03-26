#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include "cv_state_pool.h"
#include "reference_gates.h"

// 声明GPU函数
extern void apply_phase_rotation(CVStatePool* pool, const int* targets, int batch_size, double theta);
extern void apply_kerr_gate(CVStatePool* pool, const int* targets, int batch_size, double chi);
extern void apply_creation_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_annihilation_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_displacement_gate(CVStatePool* pool, const int* targets, int batch_size, cuDoubleComplex alpha, cudaStream_t stream = nullptr, bool synchronize = true);

// Wrapper for use with call_gpu_function template (defaults don't apply to function pointers)
inline void apply_displacement_gate_default(CVStatePool* pool, const int* targets, int batch_size, cuDoubleComplex alpha) {
    apply_displacement_gate(pool, targets, batch_size, alpha, nullptr, true);
}

// 辅助函数：调用GPU函数（处理target_indices的GPU内存分配）
template<typename Func, typename... Args>
void call_gpu_function(Func gpu_func, CVStatePool* state_pool, int state_id, Args... args) {
    int* d_target_indices = nullptr;
    cudaMalloc(&d_target_indices, sizeof(int));
    cudaMemcpy(d_target_indices, &state_id, sizeof(int), cudaMemcpyHostToDevice);

    gpu_func(state_pool, d_target_indices, 1, args...);

    cudaFree(d_target_indices);
}

/**
 * GPU验证测试类
 * 对比GPU实现与CPU参考实现的误差，确保GPU版本精度符合要求
 */
class GpuValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 16;  // 截断维度
        max_states = 10;  // 最大状态数
        
        // 创建测试状态
        vacuum_state = Reference::Vector(dim, Reference::Complex(0.0, 0.0));
        vacuum_state[0] = Reference::Complex(1.0, 0.0);
        
        fock1_state = Reference::Vector(dim, Reference::Complex(0.0, 0.0));
        fock1_state[1] = Reference::Complex(1.0, 0.0);
        
        fock2_state = Reference::Vector(dim, Reference::Complex(0.0, 0.0));
        fock2_state[2] = Reference::Complex(1.0, 0.0);
        
        // 检查CUDA可用性
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        cuda_available = (err == cudaSuccess && device_count > 0);
        
        if (cuda_available) {
            // 重置CUDA设备以确保干净的状态
            cudaDeviceReset();
            cudaSetDevice(0);
            
            // 创建状态池
            state_pool = new CVStatePool(dim, max_states);
            cuda_available = (state_pool->data != nullptr);
        }
    }
    
    void TearDown() override {
        if (state_pool) {
            delete state_pool;
        }
    }
    
    int dim;
    int max_states;
    bool cuda_available;
    CVStatePool* state_pool = nullptr;
    Reference::Vector vacuum_state;
    Reference::Vector fock1_state;
    Reference::Vector fock2_state;
};

// 测试CUDA可用性
TEST_F(GpuValidationTest, CudaAvailability) {
    if (!cuda_available) {
        GTEST_SKIP() << "CUDA不可用，跳过GPU测试";
    }
    EXPECT_TRUE(cuda_available);
}

// 测试GPU状态池管理
TEST_F(GpuValidationTest, GpuStatePoolManagement) {
    if (!cuda_available) {
        GTEST_SKIP() << "CUDA不可用，跳过GPU测试";
    }
    
    // 分配GPU状态
    int gpu_state_id = state_pool->allocate_state();
    EXPECT_GE(gpu_state_id, 0);
    
    // 上传状态到GPU
    std::vector<cuDoubleComplex> cuda_state;
    for (const auto& val : vacuum_state) {
        cuda_state.emplace_back(cuDoubleComplex{val.real(), val.imag()});
    }
    
    state_pool->upload_state(gpu_state_id, cuda_state);
    
    // 下载状态验证
    std::vector<cuDoubleComplex> downloaded_state(dim);
    state_pool->download_state(gpu_state_id, downloaded_state);
    
    // 计算误差
    double error = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double real_diff = vacuum_state[i].real() - downloaded_state[i].x;
        double imag_diff = vacuum_state[i].imag() - downloaded_state[i].y;
        error += real_diff * real_diff + imag_diff * imag_diff;
    }
    error = std::sqrt(error);
    
    EXPECT_NEAR(error, 0.0, 1e-12);
    
    // 释放状态
    state_pool->free_state(gpu_state_id);
}

// 测试GPU相位旋转门
TEST_F(GpuValidationTest, GpuPhaseRotationGate) {
    if (!cuda_available) {
        GTEST_SKIP() << "CUDA不可用，跳过GPU测试";
    }
    
    // CPU参考结果
    auto cpu_result = Reference::DiagonalGates::apply_phase_rotation(vacuum_state, M_PI/4.0);
    
    // GPU测试
    int gpu_state_id = state_pool->allocate_state();
    std::vector<cuDoubleComplex> cuda_state;
    for (const auto& val : vacuum_state) {
        cuda_state.emplace_back(cuDoubleComplex{val.real(), val.imag()});
    }
    state_pool->upload_state(gpu_state_id, cuda_state);
    
    // 调用GPU函数
    call_gpu_function(apply_phase_rotation, state_pool, gpu_state_id, M_PI/4.0);
    
    // 同步GPU操作
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaGetLastError();
    EXPECT_EQ(sync_err, cudaSuccess);
    
    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(dim);
    state_pool->download_state(gpu_state_id, gpu_result);
    
    // 计算误差
    double error = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double real_diff = cpu_result[i].real() - gpu_result[i].x;
        double imag_diff = cpu_result[i].imag() - gpu_result[i].y;
        error += real_diff * real_diff + imag_diff * imag_diff;
    }
    error = std::sqrt(error);
    
    EXPECT_NEAR(error, 0.0, 1e-6);
    
    state_pool->free_state(gpu_state_id);
}

// 测试GPU Kerr门
TEST_F(GpuValidationTest, GpuKerrGate) {
    if (!cuda_available) {
        GTEST_SKIP() << "CUDA不可用，跳过GPU测试";
    }
    
    // CPU参考结果
    auto cpu_result = Reference::DiagonalGates::apply_kerr_gate(vacuum_state, 0.1);
    
    // GPU测试
    int gpu_state_id = state_pool->allocate_state();
    std::vector<cuDoubleComplex> cuda_state;
    for (const auto& val : vacuum_state) {
        cuda_state.emplace_back(cuDoubleComplex{val.real(), val.imag()});
    }
    state_pool->upload_state(gpu_state_id, cuda_state);
    
    // 调用GPU函数
    call_gpu_function(apply_kerr_gate, state_pool, gpu_state_id, 0.1);
    
    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(dim);
    state_pool->download_state(gpu_state_id, gpu_result);
    
    // 计算误差
    double error = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double real_diff = cpu_result[i].real() - gpu_result[i].x;
        double imag_diff = cpu_result[i].imag() - gpu_result[i].y;
        error += real_diff * real_diff + imag_diff * imag_diff;
    }
    error = std::sqrt(error);
    
    EXPECT_NEAR(error, 0.0, 1e-6);
    
    state_pool->free_state(gpu_state_id);
}

// 测试GPU创建算符
TEST_F(GpuValidationTest, GpuCreationOperator) {
    if (!cuda_available) {
        GTEST_SKIP() << "CUDA不可用，跳过GPU测试";
    }
    
    // CPU参考结果
    auto cpu_result = Reference::LadderGates::apply_creation_operator(vacuum_state);
    
    // GPU测试
    int gpu_state_id = state_pool->allocate_state();
    std::vector<cuDoubleComplex> cuda_state;
    for (const auto& val : vacuum_state) {
        cuda_state.emplace_back(cuDoubleComplex{val.real(), val.imag()});
    }
    state_pool->upload_state(gpu_state_id, cuda_state);
    
    // 调用GPU函数
    call_gpu_function(apply_creation_operator, state_pool, gpu_state_id);
    
    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(dim);
    state_pool->download_state(gpu_state_id, gpu_result);
    
    // 计算误差
    double error = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double real_diff = cpu_result[i].real() - gpu_result[i].x;
        double imag_diff = cpu_result[i].imag() - gpu_result[i].y;
        error += real_diff * real_diff + imag_diff * imag_diff;
    }
    error = std::sqrt(error);
    
    EXPECT_NEAR(error, 0.0, 1e-6);
    
    state_pool->free_state(gpu_state_id);
}

// 测试GPU湮灭算符
TEST_F(GpuValidationTest, GpuAnnihilationOperator) {
    if (!cuda_available) {
        GTEST_SKIP() << "CUDA不可用，跳过GPU测试";
    }
    
    // CPU参考结果
    auto cpu_result = Reference::LadderGates::apply_annihilation_operator(fock1_state);
    
    // GPU测试
    int gpu_state_id = state_pool->allocate_state();
    std::vector<cuDoubleComplex> cuda_state;
    for (const auto& val : fock1_state) {
        cuda_state.emplace_back(cuDoubleComplex{val.real(), val.imag()});
    }
    state_pool->upload_state(gpu_state_id, cuda_state);
    
    // 调用GPU函数
    call_gpu_function(apply_annihilation_operator, state_pool, gpu_state_id);
    
    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(dim);
    state_pool->download_state(gpu_state_id, gpu_result);
    
    // 计算误差
    double error = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double real_diff = cpu_result[i].real() - gpu_result[i].x;
        double imag_diff = cpu_result[i].imag() - gpu_result[i].y;
        error += real_diff * real_diff + imag_diff * imag_diff;
    }
    error = std::sqrt(error);
    
    EXPECT_NEAR(error, 0.0, 1e-6);
    
    state_pool->free_state(gpu_state_id);
}

// 测试GPU位移门
TEST_F(GpuValidationTest, GpuDisplacementGate) {
    if (!cuda_available) {
        GTEST_SKIP() << "CUDA不可用，跳过GPU测试";
    }
    
    // CPU参考结果
    Reference::Complex alpha(0.1, 0.05);
    auto cpu_result = Reference::SingleModeGates::apply_displacement_gate(vacuum_state, alpha);
    
    // GPU测试
    int gpu_state_id = state_pool->allocate_state();
    std::vector<cuDoubleComplex> cuda_state;
    for (const auto& val : vacuum_state) {
        cuda_state.emplace_back(cuDoubleComplex{val.real(), val.imag()});
    }
    state_pool->upload_state(gpu_state_id, cuda_state);
    
    // 调用GPU函数
    cuDoubleComplex cuda_alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
    call_gpu_function(apply_displacement_gate_default, state_pool, gpu_state_id, cuda_alpha);
    
    // 下载GPU结果
    std::vector<cuDoubleComplex> gpu_result(dim);
    state_pool->download_state(gpu_state_id, gpu_result);
    
    // 计算误差
    double error = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double real_diff = cpu_result[i].real() - gpu_result[i].x;
        double imag_diff = cpu_result[i].imag() - gpu_result[i].y;
        error += real_diff * real_diff + imag_diff * imag_diff;
    }
    error = std::sqrt(error);
    
    EXPECT_NEAR(error, 0.0, 1e-6);
    
    state_pool->free_state(gpu_state_id);
}
