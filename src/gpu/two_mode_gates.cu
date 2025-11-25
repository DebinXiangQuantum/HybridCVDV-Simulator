#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include "cv_state_pool.h"

/**
 * Level 3: 双模混合门 (Two-Mode Mixing Gates) GPU内核
 *
 * 特性：作用于两个Qumode，矩阵巨大(D²×D²)，但光子数守恒
 * 典型门：Beam Splitter BS(θ,φ)
 *
 * 物理性质：光子数守恒(n₁ + n₂ = N_total = const)
 * 矩阵分解为多个子空间：对于总光子数k，状态向量长度为k+1
 */

/**
 * Beam Splitter矩阵计算函数
 * 计算BS(θ,φ)在子空间k中的矩阵元素
 *
 * BS(θ,φ) = exp[iφ/2] * exp[-iθ(a†b - ab†)]
 * 对于固定光子数k，矩阵大小为(k+1)×(k+1)
 */
__host__ __device__ cuDoubleComplex compute_bs_matrix_element(int k, int i, int j, double theta, double phi) {
    if (i + j != k) return make_cuDoubleComplex(0.0, 0.0);

    // 计算矩阵元素 <k-i, i| BS |k-j, j>
    // 这里i和j表示两个mode的光子数

    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    double phase = phi / 2.0;

    // 对于简单的BS门，矩阵元素可以通过递推关系计算
    // 这里使用简化的计算公式

    double elem_real = 0.0;
    double elem_imag = 0.0;

    if (i == j) {
        // 对角元素
        elem_real = cos_theta;
        elem_imag = 0.0;
    } else if (i == j + 1) {
        // 次对角线
        double coeff = sin_theta * sqrt((double)(k - j) * (j + 1));
        elem_real = coeff;
        elem_imag = 0.0;
    } else if (i == j - 1) {
        // 另一条次对角线
        double coeff = -sin_theta * sqrt((double)(k - j + 1) * j);
        elem_real = coeff;
        elem_imag = 0.0;
    }

    // 应用全局相位
    double cos_phase = cos(phase);
    double sin_phase = sin(phase);

    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_phase, sin_phase);
    cuDoubleComplex elem = make_cuDoubleComplex(elem_real, elem_imag);

    return cuCmul(phase_factor, elem);
}

/**
 * 双模混合门内核 - Block per Subspace 版本
 * 每个CUDA Block处理一个光子数子空间
 */
__global__ void apply_two_mode_gate_kernel(
    CVStatePool* state_pool,
    const int* target_indices,
    int batch_size,
    int max_photon_number,  // 最大总光子数
    double param1,          // 门参数1 (θ for BS)
    double param2           // 门参数2 (φ for BS)
) {
    // 共享内存用于存储子空间矩阵和向量
    extern __shared__ cuDoubleComplex shared_mem[];
    cuDoubleComplex* sub_matrix = shared_mem;
    cuDoubleComplex* sub_vec_in = &shared_mem[max_photon_number * max_photon_number];
    cuDoubleComplex* sub_vec_out = &sub_vec_in[max_photon_number];

    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    cuDoubleComplex* psi = &state_pool->data[state_idx * state_pool->d_trunc];

    // 计算当前block处理的子空间
    int subspace_k = blockIdx.x;  // 总光子数k
    int block_size = blockDim.x * blockDim.y;

    if (subspace_k >= max_photon_number) return;

    int sub_dim = subspace_k + 1;  // 子空间维度 (k+1)

    // 线程组织：每个线程负责矩阵的一个元素
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 加载子空间矩阵到共享内存
    if (tid < sub_dim * sub_dim) {
        int row = tid / sub_dim;
        int col = tid % sub_dim;
        // 计算Beam Splitter矩阵元素
        sub_matrix[row * sub_dim + col] = compute_bs_matrix_element(
            subspace_k, row, col, param1, param2);
    }

    // 加载输入向量到共享内存
    if (tid < sub_dim) {
        // 计算全局索引：对于总光子数k，状态索引为 k*(k+1)/2 + i
        // 这里简化为：状态按光子数分组存储
        int global_idx = subspace_k * (subspace_k + 1) / 2 + tid;
        if (global_idx < state_pool->d_trunc) {
            sub_vec_in[tid] = psi[global_idx];
        } else {
            sub_vec_in[tid] = make_cuDoubleComplex(0.0, 0.0);
        }
    }

    __syncthreads();

    // 执行稠密矩阵向量乘法
    if (tid < sub_dim) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        for (int j = 0; j < sub_dim; ++j) {
            cuDoubleComplex matrix_elem = sub_matrix[tid * sub_dim + j];
            cuDoubleComplex vec_elem = sub_vec_in[j];
            sum = cuCadd(sum, cuCmul(matrix_elem, vec_elem));
        }

        sub_vec_out[tid] = sum;
    }

    __syncthreads();

    // 将结果写回全局内存
    if (tid < sub_dim) {
        int global_idx = subspace_k * (subspace_k + 1) / 2 + tid;
        if (global_idx < state_pool->d_trunc) {
            psi[global_idx] = sub_vec_out[tid];
        }
    }
}

/**
 * 优化版本：预计算Beam Splitter矩阵
 * 对于常用的BS参数，可以预计算矩阵存储在常量内存中
 */
#define MAX_SUBSPACE_DIM 16
__constant__ cuDoubleComplex bs_matrix_const[MAX_SUBSPACE_DIM * MAX_SUBSPACE_DIM];

__global__ void apply_two_mode_gate_fast_kernel(
    CVStatePool* state_pool,
    const int* target_indices,
    int batch_size,
    int max_photon_number
) {
    extern __shared__ cuDoubleComplex shared_vec[];

    int batch_id = blockIdx.z;
    if (batch_id >= batch_size) return;

    int state_idx = target_indices[batch_id];
    cuDoubleComplex* psi = &state_pool->data[state_idx * state_pool->d_trunc];

    int subspace_k = blockIdx.x;
    if (subspace_k >= max_photon_number) return;

    int sub_dim = subspace_k + 1;
    int tid = threadIdx.x;

    // 加载输入向量到共享内存
    if (tid < sub_dim) {
        int global_idx = subspace_k * (subspace_k + 1) / 2 + tid;
        shared_vec[tid] = (global_idx < state_pool->d_trunc) ? psi[global_idx] :
                         make_cuDoubleComplex(0.0, 0.0);
        shared_vec[sub_dim + tid] = make_cuDoubleComplex(0.0, 0.0);  // 输出向量
    }

    __syncthreads();

    // 执行矩阵向量乘法 (使用常量内存中的矩阵)
    if (tid < sub_dim) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        for (int j = 0; j < sub_dim; ++j) {
            cuDoubleComplex matrix_elem = bs_matrix_const[tid * MAX_SUBSPACE_DIM + j];
            cuDoubleComplex vec_elem = shared_vec[j];
            sum = cuCadd(sum, cuCmul(matrix_elem, vec_elem));
        }

        shared_vec[sub_dim + tid] = sum;
    }

    __syncthreads();

    // 写回结果
    if (tid < sub_dim) {
        int global_idx = subspace_k * (subspace_k + 1) / 2 + tid;
        if (global_idx < state_pool->d_trunc) {
            psi[global_idx] = shared_vec[sub_dim + tid];
        }
    }
}

/**
 * 主机端：预计算Beam Splitter矩阵到常量内存
 */
void prepare_bs_matrix(double theta, double phi, int max_k) {
    int max_dim = max_k + 1;
    if (max_dim > MAX_SUBSPACE_DIM) {
        throw std::runtime_error("子空间维度超过常量内存限制");
    }

    std::vector<cuDoubleComplex> host_matrix(MAX_SUBSPACE_DIM * MAX_SUBSPACE_DIM,
                                           make_cuDoubleComplex(0.0, 0.0));

    // 计算所有子空间的BS矩阵
    for (int k = 0; k <= max_k; ++k) {
        int sub_dim = k + 1;
        int offset = k * (k + 1) / 2;  // 子空间在全局矩阵中的偏移

        for (int i = 0; i < sub_dim; ++i) {
            for (int j = 0; j < sub_dim; ++j) {
                cuDoubleComplex elem = compute_bs_matrix_element(k, i, j, theta, phi);
                int global_row = offset + i;
                int global_col = offset + j;
                host_matrix[global_row * MAX_SUBSPACE_DIM + global_col] = elem;
            }
        }
    }

    // 复制到常量内存
    cudaMemcpyToSymbol(bs_matrix_const, host_matrix.data(),
                      MAX_SUBSPACE_DIM * MAX_SUBSPACE_DIM * sizeof(cuDoubleComplex));
}

/**
 * 主机端接口：应用Beam Splitter门 BS(θ,φ)
 */
void apply_beam_splitter(CVStatePool* state_pool, const int* target_indices,
                        int batch_size, double theta, double phi, int max_photon_number) {
    // 计算共享内存大小
    int max_sub_dim = max_photon_number + 1;
    size_t shared_mem_size = max_sub_dim * max_sub_dim * sizeof(cuDoubleComplex) +  // 矩阵
                           2 * max_sub_dim * sizeof(cuDoubleComplex);  // 输入输出向量

    dim3 block_dim(16, 16);  // 16x16 = 256 threads per block
    dim3 grid_dim(max_photon_number, 1, batch_size);

    // 使用Block per Subspace版本
    apply_two_mode_gate_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        state_pool, target_indices, batch_size, max_photon_number, theta, phi
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Beam Splitter kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    // 确保GPU操作完成
    cudaDeviceSynchronize();
}

/**
 * 主机端接口：应用优化版Beam Splitter门 (使用常量内存)
 */
void apply_beam_splitter_fast(CVStatePool* state_pool, const int* target_indices,
                             int batch_size, int max_photon_number) {
    int max_sub_dim = max_photon_number + 1;
    size_t shared_mem_size = 2 * max_sub_dim * sizeof(cuDoubleComplex);

    dim3 block_dim(256);
    dim3 grid_dim(max_photon_number, 1, batch_size);

    apply_two_mode_gate_fast_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        state_pool, target_indices, batch_size, max_photon_number
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Fast Beam Splitter kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：应用通用双模混合门
 */
void apply_two_mode_gate(CVStatePool& state_pool, const int* target_indices,
                        int batch_size, double param1, double param2, int max_photon_number) {
    apply_beam_splitter(&state_pool, target_indices, batch_size,
                       param1, param2, max_photon_number);
}
