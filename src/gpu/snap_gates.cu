#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "cv_state_pool.h"

namespace {

int compute_mode_right_stride(int trunc_dim, int target_qumode, int num_qumodes) {
    if (trunc_dim <= 0) {
        throw std::invalid_argument("truncation dimension must be positive");
    }
    if (num_qumodes <= 0) {
        throw std::invalid_argument("number of qumodes must be positive");
    }
    if (target_qumode < 0 || target_qumode >= num_qumodes) {
        throw std::out_of_range("target qumode is out of range");
    }

    int right_stride = 1;
    for (int mode = target_qumode + 1; mode < num_qumodes; ++mode) {
        right_stride *= trunc_dim;
    }
    return right_stride;
}

}  // namespace

/**
 * SNAP 门 GPU 实现
 * 
 * SNAP (Selective Number-dependent Arbitrary Phase) 门
 * 对特定的 Fock 态施加相位
 * 
 * 实现的门：
 * - SNAP(θ, n): 对单个 Fock 态 |n⟩ 施加相位 θ
 * - CSNAP(θ, n): 受控 SNAP
 * - MultiSNAP: 对多个 Fock 态施加不同相位
 * - CMultiSNAP: 受控 MultiSNAP
 */

/**
 * SNAP 门内核
 * 对 Fock 态 |n⟩ 施加相位 exp(iθ)
 * 
 * SNAP(θ, n)|m⟩ = exp(iθ) |m⟩  if m == n
 *                = |m⟩         otherwise
 */
__global__ void apply_snap_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double theta,
    int target_fock_state,
    int target_mode_dim,
    int target_mode_right_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];
    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const int target_n = static_cast<int>((flat_index / right_stride) % target_mode_dim);

    // 只对目标 Fock 态施加相位
    if (target_n == target_fock_state) {
        // 相位因子: exp(iθ)
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_theta, sin_theta);
        
        cuDoubleComplex current_val = psi[flat_index];
        psi[flat_index] = cuCmul(current_val, phase_factor);
    }
    // 其他 Fock 态保持不变
}

/**
 * Multi-SNAP 门内核
 * 对多个 Fock 态施加不同的相位
 * 
 * @param phase_map 设备端数组，存储每个 Fock 态的相位
 *                  phase_map[n] = θ_n，如果 phase_map[n] == 0 则不施加相位
 */
__global__ void apply_multisnap_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    const double* phase_map,
    int map_size,
    int target_mode_dim,
    int target_mode_right_stride
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int current_dim = state_dims[state_idx];
    if (flat_index >= static_cast<size_t>(current_dim)) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];
    const size_t right_stride = static_cast<size_t>(target_mode_right_stride);
    const int target_n = static_cast<int>((flat_index / right_stride) % target_mode_dim);

    // 检查是否需要对该 Fock 态施加相位
    if (target_n < map_size) {
        double theta = phase_map[target_n];
        
        // 如果相位不为 0，施加相位
        if (fabs(theta) > 1e-15) {
            double cos_theta = cos(theta);
            double sin_theta = sin(theta);
            cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_theta, sin_theta);
            
            cuDoubleComplex current_val = psi[flat_index];
            psi[flat_index] = cuCmul(current_val, phase_factor);
        }
    }
}

/**
 * Controlled SNAP 门内核
 * 作用于 qubit ⊗ qumode 系统
 * 
 * CSNAP 只在 qubit 处于 |1⟩ 时对 qumode 施加 SNAP
 * 
 * 状态排列：|qubit, qumode⟩
 * - |0, n⟩: 索引 n
 * - |1, n⟩: 索引 cutoff + n
 */
__global__ void apply_csnap_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    double theta,
    int target_fock_state,
    int cutoff
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= cutoff) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // 只对 |1, target_fock_state⟩ 施加相位
    if (n == target_fock_state) {
        int idx_1n = cutoff + n;  // |1, n⟩ 的索引
        
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_theta, sin_theta);
        
        cuDoubleComplex current_val = psi[idx_1n];
        psi[idx_1n] = cuCmul(current_val, phase_factor);
    }
    // |0, n⟩ 保持不变
}

/**
 * Controlled Multi-SNAP 门内核
 * 作用于 qubit ⊗ qumode 系统
 */
__global__ void apply_cmultisnap_kernel(
    cuDoubleComplex* state_data,
    const size_t* state_offsets,
    const int* state_dims,
    const int* target_indices,
    int batch_size,
    const double* phase_map,
    int map_size,
    int cutoff
) {
    int batch_id = blockIdx.y;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= cutoff) return;

    size_t offset = state_offsets[state_idx];
    cuDoubleComplex* psi = &state_data[offset];

    // 对 |1, n⟩ 施加相位
    if (n < map_size) {
        double theta = phase_map[n];
        
        if (fabs(theta) > 1e-15) {
            int idx_1n = cutoff + n;
            
            double cos_theta = cos(theta);
            double sin_theta = sin(theta);
            cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_theta, sin_theta);
            
            cuDoubleComplex current_val = psi[idx_1n];
            psi[idx_1n] = cuCmul(current_val, phase_factor);
        }
    }
}

// ==================== 主机端接口 ====================

/**
 * 应用 SNAP 门
 * 
 * @param state_pool 状态池
 * @param target_indices 目标状态索引数组（设备端）
 * @param batch_size 批次大小
 * @param theta 相位角度
 * @param target_fock_state 目标 Fock 态
 */
void apply_snap_on_mode(CVStatePool* state_pool, const int* target_indices,
                        int batch_size, double theta, int target_fock_state,
                        int target_qumode, int num_qumodes) {
    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_snap_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        theta, target_fock_state,
        state_pool->d_trunc, target_mode_right_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("SNAP kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("SNAP kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void apply_snap(CVStatePool* state_pool, const int* target_indices,
                int batch_size, double theta, int target_fock_state) {
    apply_snap_on_mode(state_pool, target_indices, batch_size, theta, target_fock_state, 0, 1);
}

/**
 * 应用 Multi-SNAP 门
 * 
 * @param state_pool 状态池
 * @param target_indices 目标状态索引数组（设备端）
 * @param batch_size 批次大小
 * @param phase_map 相位映射（主机端 vector）
 */
void apply_multisnap_on_mode(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, const std::vector<double>& phase_map,
                             int target_qumode, int num_qumodes) {
    if (phase_map.empty()) {
        return;
    }

    const int target_mode_right_stride =
        compute_mode_right_stride(state_pool->d_trunc, target_qumode, num_qumodes);

    // 将 phase_map 上传到设备
    double* d_phase_map = nullptr;
    cudaMalloc(&d_phase_map, phase_map.size() * sizeof(double));
    cudaMemcpy(d_phase_map, phase_map.data(), 
               phase_map.size() * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->max_total_dim + block_dim.x - 1) / block_dim.x, batch_size);

    apply_multisnap_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        d_phase_map, phase_map.size(),
        state_pool->d_trunc, target_mode_right_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_phase_map);
        throw std::runtime_error("Multi-SNAP kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    cudaFree(d_phase_map);
    
    if (err != cudaSuccess) {
        throw std::runtime_error("Multi-SNAP kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void apply_multisnap(CVStatePool* state_pool, const int* target_indices,
                     int batch_size, const std::vector<double>& phase_map) {
    apply_multisnap_on_mode(state_pool, target_indices, batch_size, phase_map, 0, 1);
}

/**
 * 应用 Controlled SNAP 门
 * 
 * @param state_pool 状态池
 * @param target_indices 目标状态索引数组（设备端）
 * @param batch_size 批次大小
 * @param theta 相位角度
 * @param target_fock_state 目标 Fock 态
 * @param cutoff qumode 的截断维度
 */
void apply_csnap(CVStatePool* state_pool, const int* target_indices,
                 int batch_size, double theta, int target_fock_state, int cutoff) {
    dim3 block_dim(256);
    dim3 grid_dim((cutoff + block_dim.x - 1) / block_dim.x, batch_size);

    apply_csnap_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        theta, target_fock_state, cutoff
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CSNAP kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CSNAP kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 应用 Controlled Multi-SNAP 门
 * 
 * @param state_pool 状态池
 * @param target_indices 目标状态索引数组（设备端）
 * @param batch_size 批次大小
 * @param phase_map 相位映射（主机端 vector）
 * @param cutoff qumode 的截断维度
 */
void apply_cmultisnap(CVStatePool* state_pool, const int* target_indices,
                      int batch_size, const std::vector<double>& phase_map, int cutoff) {
    // 将 phase_map 上传到设备
    double* d_phase_map = nullptr;
    cudaMalloc(&d_phase_map, phase_map.size() * sizeof(double));
    cudaMemcpy(d_phase_map, phase_map.data(), 
               phase_map.size() * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_dim((cutoff + block_dim.x - 1) / block_dim.x, batch_size);

    apply_cmultisnap_kernel<<<grid_dim, block_dim>>>(
        state_pool->data,
        state_pool->state_offsets,
        state_pool->state_dims,
        target_indices, batch_size,
        d_phase_map, phase_map.size(), cutoff
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_phase_map);
        throw std::runtime_error("CMulti-SNAP kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    cudaFree(d_phase_map);
    
    if (err != cudaSuccess) {
        throw std::runtime_error("CMulti-SNAP kernel synchronization failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
