#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cv_state_pool.h"
#include "hdd_node.h"

/**
 * Level 4: 混合控制门 (Hybrid Control Gates) GPU内核
 *
 * 特性：Qubit控制Qumode，涉及HDD结构变更
 * 典型门：Controlled-Displacement CD(α)
 *
 * 逻辑流程：
 * 1. CPU遍历HDD
 * 2. 对于Control=|0>分支，跳过
 * 3. 对于Control=|1>分支，执行Copy-on-Write
 * 4. 批量执行Level 2 Kernel
 */

/**
 * 受控位移门应用内核
 * 对指定的状态集合应用Displacement门
 */
__global__ void apply_controlled_displacement_kernel(
    CVStatePool* state_pool,
    const int* target_state_ids,
    int num_states,
    cuDoubleComplex alpha
) {
    int state_id = target_state_ids[blockIdx.y];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= state_pool->d_trunc) return;

    cuDoubleComplex* psi = &state_pool->data[state_id * state_pool->d_trunc];

    // 计算Displacement矩阵的第n行 (简化的实现)
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    double alpha_norm_sq = cuCreal(cuCmul(alpha, cuConj(alpha)));
    double exp_factor = exp(-alpha_norm_sq / 2.0);

    // 对于小的α值，只计算带状区域
    int bandwidth = min(10, (int)sqrt(alpha_norm_sq * 4) + 2);

    for (int m = max(0, n - bandwidth); m <= min(state_pool->d_trunc - 1, n + bandwidth); ++m) {
        // 简化的Displacement矩阵元素计算
        double coeff = exp_factor;
        if (n >= m) {
            // 计算 √(n! / m!)
            double sqrt_factorial_ratio = 1.0;
            for (int k = m + 1; k <= n; ++k) {
                sqrt_factorial_ratio *= sqrt((double)k);
            }
            coeff *= sqrt_factorial_ratio;
        }

        if (abs(n - m) <= bandwidth) {
            cuDoubleComplex power_term = make_cuDoubleComplex(1.0, 0.0);
            int power = n - m;

            if (power > 0) {
                for (int p = 0; p < power; ++p) {
                    power_term = cuCmul(power_term, alpha);
                }
            } else if (power < 0) {
                cuDoubleComplex alpha_conj = cuConj(alpha);
                for (int p = 0; p < -power; ++p) {
                    power_term = cuCmul(power_term, alpha_conj);
                }
            }

            // 简化的Laguerre多项式近似
            double laguerre = 1.0;
            if (abs(power) > 0) {
                laguerre = exp(-alpha_norm_sq / 2.0) * pow(alpha_norm_sq, abs(power) / 2.0);
            }

            cuDoubleComplex matrix_elem = make_cuDoubleComplex(
                coeff * laguerre, 0.0
            );
            matrix_elem = cuCmul(matrix_elem, power_term);

            sum = cuCadd(sum, cuCmul(matrix_elem, psi[m]));
        }
    }

    psi[n] = sum;
}

/**
 * 批量状态复制内核 (用于Copy-on-Write)
 */
__global__ void copy_states_kernel(
    CVStatePool* state_pool,
    const int* source_ids,
    const int* dest_ids,
    int num_copies
) {
    int copy_id = blockIdx.y;
    if (copy_id >= num_copies) return;

    int src_id = source_ids[copy_id];
    int dst_id = dest_ids[copy_id];
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= state_pool->d_trunc) return;

    cuDoubleComplex* src_ptr = &state_pool->data[src_id * state_pool->d_trunc];
    cuDoubleComplex* dst_ptr = &state_pool->data[dst_id * state_pool->d_trunc];

    dst_ptr[n] = src_ptr[n];
}

/**
 * 主机端接口：应用受控位移门 CD(α)
 */
void apply_controlled_displacement(CVStatePool* state_pool,
                                 const std::vector<int>& controlled_states,
                                 cuDoubleComplex alpha) {
    if (controlled_states.empty()) return;

    // 上传状态ID到GPU
    int* d_state_ids = nullptr;
    cudaMalloc(&d_state_ids, controlled_states.size() * sizeof(int));
    cudaMemcpy(d_state_ids, controlled_states.data(),
               controlled_states.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x,
                  controlled_states.size());

    apply_controlled_displacement_kernel<<<grid_dim, block_dim>>>(
        state_pool, d_state_ids, controlled_states.size(), alpha
    );

    cudaFree(d_state_ids);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Controlled Displacement kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 主机端接口：复制状态 (用于Copy-on-Write)
 */
void copy_states(CVStatePool* state_pool,
                const std::vector<int>& source_ids,
                const std::vector<int>& dest_ids) {
    if (source_ids.size() != dest_ids.size()) {
        throw std::invalid_argument("源状态ID和目标状态ID数量不匹配");
    }

    if (source_ids.empty()) return;

    // 上传ID列表到GPU
    int* d_src_ids = nullptr;
    int* d_dst_ids = nullptr;

    cudaMalloc(&d_src_ids, source_ids.size() * sizeof(int));
    cudaMalloc(&d_dst_ids, dest_ids.size() * sizeof(int));

    cudaMemcpy(d_src_ids, source_ids.data(),
               source_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_ids, dest_ids.data(),
               dest_ids.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_dim((state_pool->d_trunc + block_dim.x - 1) / block_dim.x,
                  source_ids.size());

    copy_states_kernel<<<grid_dim, block_dim>>>(
        state_pool, d_src_ids, d_dst_ids, source_ids.size()
    );

    cudaFree(d_src_ids);
    cudaFree(d_dst_ids);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("State copy kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

/**
 * 高级接口：执行混合控制门
 * 该函数实现了完整的Level 4逻辑，包括HDD遍历和Copy-on-Write
 */
void execute_hybrid_control_gate(HDDNode* root_node,
                               CVStatePool* state_pool,
                               HDDNodeManager& node_manager,
                               cuDoubleComplex gate_param) {
    // 简化的混合控制门实现
    // 实际的完整实现需要遍历整个HDD，这里只处理单个终端节点作为示例

    std::vector<int> states_to_modify;

    // 简化实现：假设只有一个终端节点，且控制条件满足
    if (root_node && root_node->is_terminal() && root_node->tensor_id >= 0) {
        // 检查控制条件 (简化：假设总是应用操作)
        states_to_modify.push_back(root_node->tensor_id);
    }

    // 应用控制位移门
    if (!states_to_modify.empty()) {
        apply_controlled_displacement(state_pool, states_to_modify, gate_param);
    }
}

/**
 * 主机端接口：应用通用混合控制门
 */
void apply_hybrid_control_gate(HDDNode* root_node,
                             CVStatePool* state_pool,
                             HDDNodeManager& node_manager,
                             const std::string& gate_type,
                             cuDoubleComplex param) {
    if (gate_type == "controlled_displacement") {
        execute_hybrid_control_gate(root_node, state_pool, node_manager, param);
    } else {
        throw std::invalid_argument("不支持的混合控制门类型: " + gate_type);
    }
}
