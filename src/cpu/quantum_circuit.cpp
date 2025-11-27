#include "quantum_circuit.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>

// CUDA错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA错误: " + std::string(cudaGetErrorString(err))); \
        } \
    } while (0)

// 包含GPU内核头文件
// Level 0
void apply_phase_rotation(CVStatePool* pool, const int* targets, int batch_size, double theta);
void apply_kerr_gate(CVStatePool* pool, const int* targets, int batch_size, double chi);
void apply_conditional_parity(CVStatePool* pool, const int* targets, int batch_size, double parity);

// Level 1
void apply_creation_operator(CVStatePool* pool, const int* targets, int batch_size);
void apply_annihilation_operator(CVStatePool* pool, const int* targets, int batch_size);

// Level 2
void apply_single_mode_gate(CVStatePool* pool, FockELLOperator* ell_op,
                           const int* targets, int batch_size);
void apply_displacement_gate(CVStatePool* pool, const int* targets, int batch_size,
                            cuDoubleComplex alpha);

// Level 3
void apply_beam_splitter(CVStatePool* pool, const int* targets, int batch_size,
                        double theta, double phi, int max_photon);

// Level 4
void apply_hybrid_control_gate(HDDNode* root_node, CVStatePool* state_pool,
                              HDDNodeManager& node_manager,
                              const std::string& gate_type,
                              cuDoubleComplex param);

// 状态加法函数
void add_states(CVStatePool* state_pool,
                const int* src1_indices,
                const cuDoubleComplex* weights1,
                const int* src2_indices,
                const cuDoubleComplex* weights2,
                const int* dst_indices,
                int batch_size);

/**
 * HDD节点加法: result = w1 * n1 + w2 * n2
 */
HDDNode* QuantumCircuit::hdd_add(HDDNode* n1, std::complex<double> w1, HDDNode* n2, std::complex<double> w2) {
    // 处理零权重
    if (std::abs(w1) < 1e-14) {
        if (std::abs(w2) < 1e-14) {
            // 两者都为零：返回"零"节点
            int zero_id = state_pool_.allocate_state();
            std::vector<cuDoubleComplex> zeros(cv_truncation_, make_cuDoubleComplex(0.0, 0.0));
            state_pool_.upload_state(zero_id, zeros);
            return node_manager_.create_terminal_node(zero_id);
        }
        // 只有 w2，递归处理
        return hdd_add(n2, w2, n1, w1);
    }
    if (std::abs(w2) < 1e-14) {
        // 只有 w1，递归处理
        return hdd_add(n1, w1, n2, w2);
    }

    // 基本情况：终端节点
    if (n1->is_terminal() && n2->is_terminal()) {
        int id1 = n1->tensor_id;
        int id2 = n2->tensor_id;
        int new_id = state_pool_.allocate_state();
        
        // 准备设备端数据
        int* d_src1 = nullptr;
        int* d_src2 = nullptr;
        int* d_dst = nullptr;
        cuDoubleComplex* d_w1 = nullptr;
        cuDoubleComplex* d_w2 = nullptr;
        
        cudaMalloc(&d_src1, sizeof(int));
        cudaMalloc(&d_src2, sizeof(int));
        cudaMalloc(&d_dst, sizeof(int));
        cudaMalloc(&d_w1, sizeof(cuDoubleComplex));
        cudaMalloc(&d_w2, sizeof(cuDoubleComplex));
        
        cudaMemcpy(d_src1, &id1, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_src2, &id2, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dst, &new_id, sizeof(int), cudaMemcpyHostToDevice);
        
        cuDoubleComplex w1_cu = make_cuDoubleComplex(w1.real(), w1.imag());
        cuDoubleComplex w2_cu = make_cuDoubleComplex(w2.real(), w2.imag());
        cudaMemcpy(d_w1, &w1_cu, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2, &w2_cu, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        
        // 调用GPU内核
        add_states(&state_pool_, d_src1, d_w1, d_src2, d_w2, d_dst, 1);
        
        // 清理
        cudaFree(d_src1);
        cudaFree(d_src2);
        cudaFree(d_dst);
        cudaFree(d_w1);
        cudaFree(d_w2);
        
        return node_manager_.create_terminal_node(new_id);
    }
    
    // 递归步骤
    int level1 = n1->is_terminal() ? -1 : n1->qubit_level;
    int level2 = n2->is_terminal() ? -1 : n2->qubit_level;
    
    if (level1 != level2) {
        if (level1 > level2) {
            HDDNode* new_low = hdd_add(n1->low, w1 * n1->w_low, n2, w2);
            HDDNode* new_high = hdd_add(n1->high, w1 * n1->w_high, n2, w2);
            return node_manager_.get_or_create_node(level1, new_low, new_high, 1.0, 1.0);
        } else {
            // level2 > level1
            HDDNode* new_low = hdd_add(n1, w1, n2->low, w2 * n2->w_low);
            HDDNode* new_high = hdd_add(n1, w1, n2->high, w2 * n2->w_high);
            return node_manager_.get_or_create_node(level2, new_low, new_high, 1.0, 1.0);
        }
    }
    
    // 相同层级
    int level = level1;
    HDDNode* new_low = hdd_add(n1->low, w1 * n1->w_low, n2->low, w2 * n2->w_low);
    HDDNode* new_high = hdd_add(n1->high, w1 * n1->w_high, n2->high, w2 * n2->w_high);
    
    return node_manager_.get_or_create_node(level, new_low, new_high, 1.0, 1.0);
}

HDDNode* QuantumCircuit::apply_single_qubit_gate_recursive(HDDNode* node, int target_qubit, const std::vector<std::complex<double>>& u) {
    if (node->is_terminal()) {
        return node;
    }
    
    if (node->qubit_level == target_qubit) {
        // 应用矩阵
        // u: [u00, u01, u10, u11]
        HDDNode* L = node->low;
        HDDNode* H = node->high;
        
        std::complex<double> wL = node->w_low;
        std::complex<double> wH = node->w_high;
        
        // New L = u00 * (wL * L) + u01 * (wH * H)
        HDDNode* new_L = hdd_add(L, u[0] * wL, H, u[1] * wH);
        
        // New H = u10 * (wL * L) + u11 * (wH * H)
        HDDNode* new_H = hdd_add(L, u[2] * wL, H, u[3] * wH);
        
        return node_manager_.get_or_create_node(node->qubit_level, new_L, new_H, 1.0, 1.0);
    } else if (node->qubit_level > target_qubit) {
        HDDNode* new_low = apply_single_qubit_gate_recursive(node->low, target_qubit, u);
        HDDNode* new_high = apply_single_qubit_gate_recursive(node->high, target_qubit, u);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    } else {
        return node;
    }
}

HDDNode* QuantumCircuit::apply_cnot_recursive(HDDNode* node, int control, int target) {
    if (node->is_terminal()) return node;
    
    if (node->qubit_level == control) {
        // 控制节点
        HDDNode* new_low = node->low; // 恒等
        
        // 高分支：对目标应用X
        std::vector<std::complex<double>> px = {0.0, 1.0, 1.0, 0.0};
        HDDNode* new_high = apply_single_qubit_gate_recursive(node->high, target, px);
        
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    } else if (node->qubit_level > control) {
        HDDNode* new_low = apply_cnot_recursive(node->low, control, target);
        HDDNode* new_high = apply_cnot_recursive(node->high, control, target);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    }
    
    return node;
}

void QuantumCircuit::execute_qubit_gate(const GateParams& gate) {
    if (gate.target_qubits.empty()) return;
    
    HDDNode* new_root = nullptr;
    
    if (gate.type == GateType::CNOT) {
        if (gate.target_qubits.size() < 2) return;
        int c = gate.target_qubits[0];
        int t = gate.target_qubits[1];
        if (c < t) {
            throw std::runtime_error("Currently only supports Control > Target (higher index controls lower index)");
        }
        new_root = apply_cnot_recursive(root_node_, c, t);
    } else {
        int target = gate.target_qubits[0];
        std::vector<std::complex<double>> mat(4);
        
        double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        
        switch (gate.type) {
            case GateType::HADAMARD:
                mat = {inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2};
                break;
            case GateType::PAULI_X:
                mat = {0.0, 1.0, 1.0, 0.0};
                break;
            case GateType::PAULI_Y:
                mat = {0.0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0.0};
                break;
            case GateType::PAULI_Z:
                mat = {1.0, 0.0, 0.0, -1.0};
                break;
            case GateType::ROTATION_X: {
                double theta = gate.params[0].real();
                double c = std::cos(theta/2);
                double s = std::sin(theta/2);
                mat = {c, std::complex<double>(0, -s), std::complex<double>(0, -s), c};
                break;
            }
            case GateType::ROTATION_Y: {
                double theta = gate.params[0].real();
                double c = std::cos(theta/2);
                double s = std::sin(theta/2);
                mat = {c, -s, s, c};
                break;
            }
            case GateType::ROTATION_Z: {
                double theta = gate.params[0].real();
                double c = std::cos(theta/2);
                double s = std::sin(theta/2);
                mat = {std::complex<double>(c, -s), 0.0, 0.0, std::complex<double>(c, s)}; // exp(-i t/2 Z)
                break;
            }
            default:
                return;
        }
        
        new_root = apply_single_qubit_gate_recursive(root_node_, target, mat);
    }
    
    if (new_root != root_node_) {
        node_manager_.release_node(root_node_);
        root_node_ = new_root;
    }
}

/**
 * QuantumCircuit 构造函数
 */
QuantumCircuit::QuantumCircuit(int num_qubits, int num_qumodes, int cv_truncation, int max_states)
    : num_qubits_(num_qubits), num_qumodes_(num_qumodes), cv_truncation_(cv_truncation),
      root_node_(nullptr), state_pool_(cv_truncation, max_states),
      is_built_(false), is_executed_(false) {

    if (num_qubits <= 0 || num_qumodes <= 0 || cv_truncation <= 0) {
        throw std::invalid_argument("Qubit数量、Qumode数量和截断维度必须为正数");
    }

    std::cout << "创建量子电路: " << num_qubits << " qubits, "
              << num_qumodes << " qumodes, 截断维度=" << cv_truncation << std::endl;
}

/**
 * QuantumCircuit 析构函数
 */
QuantumCircuit::~QuantumCircuit() {
    reset();
}

/**
 * 添加门操作到线路
 */
void QuantumCircuit::add_gate(const GateParams& gate) {
    if (is_built_) {
        throw std::runtime_error("不能在构建后添加门操作");
    }
    gate_sequence_.push_back(gate);
}

/**
 * 批量添加门操作
 */
void QuantumCircuit::add_gates(const std::vector<GateParams>& gates) {
    if (is_built_) {
        throw std::runtime_error("不能在构建后添加门操作");
    }
    gate_sequence_.insert(gate_sequence_.end(), gates.begin(), gates.end());
}

/**
 * 构建量子线路
 */
void QuantumCircuit::build() {
    if (is_built_) return;

    std::cout << "构建量子线路..." << std::endl;

    // 初始化HDD结构
    initialize_hdd();

    is_built_ = true;
    std::cout << "量子线路构建完成" << std::endl;
}

/**
 * 执行量子线路
 */
void QuantumCircuit::execute() {
    if (!is_built_) {
        throw std::runtime_error("必须先构建量子线路");
    }

    if (is_executed_) {
        std::cout << "线路已执行，跳过重复执行" << std::endl;
        return;
    }

    std::cout << "执行量子线路..." << std::endl;

    // 执行所有门操作
    for (const auto& gate : gate_sequence_) {
        execute_gate(gate);
    }

    is_executed_ = true;
    std::cout << "量子线路执行完成" << std::endl;
}

/**
 * 重置量子线路状态
 */
void QuantumCircuit::reset() {
    // 同步所有GPU操作，确保在重置前所有操作完成
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess && sync_err != cudaErrorNotReady) {
        // 如果之前的操作有错误，尝试清除错误状态
        std::cerr << "警告：重置前检测到GPU错误: " << cudaGetErrorString(sync_err) << std::endl;
        // 清除CUDA错误状态，允许后续操作继续
        cudaGetLastError(); // 清除错误标志
    }

    if (root_node_) {
        node_manager_.release_node(root_node_);
        root_node_ = nullptr;
    }

    node_manager_.clear();
    gate_sequence_.clear();
    is_built_ = false;
    is_executed_ = false;

    // 重新初始化状态池 (可选)
    // 这里保持状态池不变，以允许重新使用
}

/**
 * 初始化HDD结构
 */
void QuantumCircuit::initialize_hdd() {
    // 创建初始状态 |00...0> ⊗ |vacuum>
    // 首先分配一个CV状态用于真空态
    int vacuum_state_id = state_pool_.allocate_state();

    // 初始化真空态 (第一个元素为1，其他为0)
    std::vector<cuDoubleComplex> vacuum_state(cv_truncation_, make_cuDoubleComplex(0.0, 0.0));
    vacuum_state[0] = make_cuDoubleComplex(1.0, 0.0);
    state_pool_.upload_state(vacuum_state_id, vacuum_state);

    // 创建HDD根节点 (所有qubits为|0>)
    root_node_ = node_manager_.create_terminal_node(vacuum_state_id);
}

/**
 * 执行单个门操作
 */
void QuantumCircuit::execute_gate(const GateParams& gate) {
    switch (gate.type) {
        case GateType::PHASE_ROTATION:
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
            execute_level0_gate(gate);
            break;

        case GateType::CREATION_OPERATOR:
        case GateType::ANNIHILATION_OPERATOR:
            execute_level1_gate(gate);
            break;

        case GateType::DISPLACEMENT:
        case GateType::SQUEEZING:
            execute_level2_gate(gate);
            break;

        case GateType::BEAM_SPLITTER:
            execute_level3_gate(gate);
            break;

        case GateType::CONTROLLED_DISPLACEMENT:
        case GateType::CONTROLLED_SQUEEZING:
            execute_level4_gate(gate);
            break;

        default:
            throw std::runtime_error("不支持的门类型");
    }
}

/**
 * 执行Level 0门 (对角门)
 */
void QuantumCircuit::execute_level0_gate(const GateParams& gate) {
    auto target_states = collect_target_states(gate);

    if (target_states.empty()) return;

    // 上传状态ID到GPU
    int* d_target_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_target_ids, target_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
               target_states.size() * sizeof(int), cudaMemcpyHostToDevice));

    double param = gate.params.empty() ? 0.0 : gate.params[0].real();

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            apply_phase_rotation(&state_pool_, d_target_ids, target_states.size(), param);
            break;
        case GateType::KERR_GATE:
            apply_kerr_gate(&state_pool_, d_target_ids, target_states.size(), param);
            break;
        case GateType::CONDITIONAL_PARITY:
            apply_conditional_parity(&state_pool_, d_target_ids, target_states.size(), param);
            break;
        default:
            break;
    }

    // 检查GPU内核执行错误
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_target_ids));
}

/**
 * 执行Level 1门 (梯算符门)
 */
void QuantumCircuit::execute_level1_gate(const GateParams& gate) {
    auto target_states = collect_target_states(gate);

    if (target_states.empty()) return;

    int* d_target_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_target_ids, target_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
               target_states.size() * sizeof(int), cudaMemcpyHostToDevice));

    switch (gate.type) {
        case GateType::CREATION_OPERATOR:
            apply_creation_operator(&state_pool_, d_target_ids, target_states.size());
            break;
        case GateType::ANNIHILATION_OPERATOR:
            apply_annihilation_operator(&state_pool_, d_target_ids, target_states.size());
            break;
        default:
            break;
    }

    // 检查GPU内核执行错误
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_target_ids));
}

/**
 * 执行Level 2门 (单模门)
 */
void QuantumCircuit::execute_level2_gate(const GateParams& gate) {
    auto target_states = collect_target_states(gate);

    if (target_states.empty()) return;

    int* d_target_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_target_ids, target_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
               target_states.size() * sizeof(int), cudaMemcpyHostToDevice));

    if (gate.type == GateType::DISPLACEMENT && !gate.params.empty()) {
        cuDoubleComplex alpha = make_cuDoubleComplex(gate.params[0].real(), gate.params[0].imag());
        apply_displacement_gate(&state_pool_, d_target_ids, target_states.size(), alpha);
        // 检查GPU内核执行错误
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        // 使用ELL格式的通用实现
        FockELLOperator* ell_op = prepare_ell_operator(gate);
        if (ell_op) {
            apply_single_mode_gate(&state_pool_, ell_op, d_target_ids, target_states.size());
            // 在删除ELL操作符之前，确保GPU操作完成
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            delete ell_op;
        }
    }

    CHECK_CUDA(cudaFree(d_target_ids));
}

/**
 * 执行Level 3门 (双模门)
 */
void QuantumCircuit::execute_level3_gate(const GateParams& gate) {
    auto target_states = collect_target_states(gate);

    if (target_states.empty()) return;

    int* d_target_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_target_ids, target_states.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_target_ids, target_states.data(),
               target_states.size() * sizeof(int), cudaMemcpyHostToDevice));

    if (gate.type == GateType::BEAM_SPLITTER && gate.params.size() >= 2) {
        double theta = gate.params[0].real();
        double phi = gate.params[1].real();
        int max_photon = cv_truncation_ - 1;  // 最大光子数

        // Beam splitter operates on the current state (state 0)
        // The target qumodes are specified in gate.targets
        int state_id = 0;  // Currently we only have one state
        int* d_state_id = nullptr;
        CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));

        apply_beam_splitter(&state_pool_, d_state_id, 1, theta, phi, max_photon);

        // 检查GPU内核执行错误
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaFree(d_state_id));
    }

    CHECK_CUDA(cudaFree(d_target_ids));
}

/**
 * 执行Level 4门 (混合控制门)
 */
void QuantumCircuit::execute_level4_gate(const GateParams& gate) {
    std::string gate_type;
    cuDoubleComplex param = make_cuDoubleComplex(0.0, 0.0);

    if (gate.type == GateType::CONTROLLED_DISPLACEMENT) {
        gate_type = "controlled_displacement";
        if (!gate.params.empty()) {
            param = make_cuDoubleComplex(gate.params[0].real(), gate.params[0].imag());
        }
    } else if (gate.type == GateType::CONTROLLED_SQUEEZING) {
        gate_type = "controlled_squeezing";
        if (!gate.params.empty()) {
            param = make_cuDoubleComplex(gate.params[0].real(), gate.params[0].imag());
        }
    }

    // 对于混合控制门，目前实现是占位符
    // 如果gate_type是controlled_displacement，我们需要实际实现它
    if (gate_type == "controlled_displacement" && root_node_ != nullptr) {
        // 收集需要应用控制位移的状态
        // 从状态池获取所有活跃的状态ID
        std::vector<int> controlled_states = state_pool_.get_active_state_ids();
        
        if (!controlled_states.empty()) {
            // 使用hybrid_gates.cu中的apply_controlled_displacement函数
            // 但需要先声明它
            extern void apply_controlled_displacement(CVStatePool* state_pool,
                                                     const std::vector<int>& controlled_states,
                                                     cuDoubleComplex alpha);
            apply_controlled_displacement(&state_pool_, controlled_states, param);
            // apply_controlled_displacement内部已经包含错误检查和同步
        }
    } else {
        // 调用占位符函数（目前为空实现）
        apply_hybrid_control_gate(root_node_, &state_pool_, node_manager_, gate_type, param);
        // 即使占位符也可能有GPU操作，检查错误
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

/**
 * 收集需要更新的状态ID
 */
std::vector<int> QuantumCircuit::collect_target_states(const GateParams& gate) {
    // 从状态池获取所有活跃的状态ID
    // 这些是实际分配的状态ID，而不是简单的索引
    return state_pool_.get_active_state_ids();
}

/**
 * 准备ELL算符
 */
FockELLOperator* QuantumCircuit::prepare_ell_operator(const GateParams& gate) {
    // 简化的实现：创建基本的ELL算符
    // 在实际实现中，需要根据具体的门参数构建正确的矩阵

    FockELLOperator* ell_op = new FockELLOperator(cv_truncation_, 10);  // 假设带宽为10

    // 这里应该根据门类型填充ELL格式数据
    // 暂时返回空算符

    return ell_op;
}

/**
 * 获取状态振幅
 */
std::complex<double> QuantumCircuit::get_amplitude(
    const std::vector<int>& qubit_states,
    const std::vector<std::vector<std::complex<double>>>& qumode_states) {

    // 简化的实现：只返回真空态振幅
    // 在实际实现中，需要遍历HDD找到对应的终端节点

    if (root_node_ && root_node_->is_terminal() && root_node_->tensor_id >= 0) {
        std::vector<cuDoubleComplex> state_data;
        state_pool_.download_state(root_node_->tensor_id, state_data);

        // 返回真空态分量
        return std::complex<double>(cuCreal(state_data[0]), cuCimag(state_data[0]));
    }

    return std::complex<double>(0.0, 0.0);
}

/**
 * 获取线路统计信息
 */
QuantumCircuit::CircuitStats QuantumCircuit::get_stats() const {
    return {
        num_qubits_,
        num_qumodes_,
        cv_truncation_,
        static_cast<int>(gate_sequence_.size()),
        state_pool_.active_count,
        node_manager_.get_cache_size()
    };
}

// ===== 门操作便捷函数实现 =====

namespace Gates {
    GateParams PhaseRotation(int qubit, double theta) {
        return GateParams(GateType::PHASE_ROTATION, {qubit}, {}, {theta});
    }

    GateParams KerrGate(int qumode, double chi) {
        return GateParams(GateType::KERR_GATE, {}, {qumode}, {chi});
    }

    GateParams ConditionalParity(int qumode, double parity) {
        return GateParams(GateType::CONDITIONAL_PARITY, {}, {qumode}, {parity});
    }

    GateParams CreationOperator(int qumode) {
        return GateParams(GateType::CREATION_OPERATOR, {}, {qumode});
    }

    GateParams AnnihilationOperator(int qumode) {
        return GateParams(GateType::ANNIHILATION_OPERATOR, {}, {qumode});
    }

    GateParams Displacement(int qumode, std::complex<double> alpha) {
        return GateParams(GateType::DISPLACEMENT, {}, {qumode}, {alpha});
    }

    GateParams Squeezing(int qumode, std::complex<double> xi) {
        return GateParams(GateType::SQUEEZING, {}, {qumode}, {xi});
    }

    GateParams BeamSplitter(int qumode1, int qumode2, double theta, double phi) {
        return GateParams(GateType::BEAM_SPLITTER, {}, {qumode1, qumode2}, {theta, phi});
    }

    GateParams ControlledDisplacement(int control_qubit, int target_qumode, std::complex<double> alpha) {
        return GateParams(GateType::CONTROLLED_DISPLACEMENT, {control_qubit}, {target_qumode}, {alpha});
    }

    GateParams ControlledSqueezing(int control_qubit, int target_qumode, std::complex<double> xi) {
        return GateParams(GateType::CONTROLLED_SQUEEZING, {control_qubit}, {target_qumode}, {xi});
    }
}
