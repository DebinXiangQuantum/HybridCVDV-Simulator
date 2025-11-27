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

/**
 * 递归应用CZ门
 */
HDDNode* QuantumCircuit::apply_cz_recursive(HDDNode* node, int control, int target) {
    if (node->is_terminal()) return node;

    if (node->qubit_level == control) {
        // 控制节点
        HDDNode* new_low = node->low; // 恒等

        // 高分支：对目标应用Z
        std::vector<std::complex<double>> pz = {1.0, 0.0, 0.0, -1.0};
        HDDNode* new_high = apply_single_qubit_gate_recursive(node->high, target, pz);

        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    } else if (node->qubit_level > control) {
        HDDNode* new_low = apply_cz_recursive(node->low, control, target);
        HDDNode* new_high = apply_cz_recursive(node->high, control, target);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high, node->w_low, node->w_high);
    }

    return node;
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
        // CPU端纯Qubit门
        case GateType::HADAMARD:
        case GateType::PAULI_X:
        case GateType::PAULI_Y:
        case GateType::PAULI_Z:
        case GateType::ROTATION_X:
        case GateType::ROTATION_Y:
        case GateType::ROTATION_Z:
        case GateType::PHASE_GATE_S:
        case GateType::PHASE_GATE_T:
        case GateType::CNOT:
        case GateType::CZ:
            execute_qubit_gate(gate);
            break;

        // GPU端纯Qumode门
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

        // CPU+GPU混合门
        case GateType::CONDITIONAL_DISPLACEMENT:
        case GateType::CONDITIONAL_SQUEEZING:
        case GateType::CONDITIONAL_BEAM_SPLITTER:
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
        case GateType::CONDITIONAL_SUM:
        case GateType::RABI_INTERACTION:
        case GateType::JAYNES_CUMMINGS:
        case GateType::ANTI_JAYNES_CUMMINGS:
        case GateType::SELECTIVE_QUBIT_ROTATION:
            execute_hybrid_gate(gate);
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

    if (gate.type == GateType::CONDITIONAL_DISPLACEMENT) {
        gate_type = "controlled_displacement";
        if (!gate.params.empty()) {
            param = make_cuDoubleComplex(gate.params[0].real(), gate.params[0].imag());
        }
    } else if (gate.type == GateType::CONDITIONAL_SQUEEZING) {
        gate_type = "controlled_squeezing";
        if (!gate.params.empty()) {
            param = make_cuDoubleComplex(gate.params[0].real(), gate.params[0].imag());
        }
    }

    // 对于混合控制门，目前实现是占位符
    // 调用混合控制门执行函数
    apply_hybrid_control_gate(root_node_, &state_pool_, node_manager_, gate_type, param);
    // 检查GPU错误
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

/**
 * 执行Qubit门操作
 */
void QuantumCircuit::execute_qubit_gate(const GateParams& gate) {
    if (gate.target_qubits.empty()) {
        throw std::runtime_error("Qubit门需要指定目标Qubit");
    }

    int target_qubit = gate.target_qubits[0];
    if (target_qubit >= num_qubits_) {
        throw std::runtime_error("目标Qubit索引超出范围");
    }

    // 构造对应的单比特门矩阵U
    std::vector<std::complex<double>> u(4, std::complex<double>(0.0, 0.0));

    switch (gate.type) {
        case GateType::PAULI_X: {
            // X = [[0, 1], [1, 0]]
            u[1] = 1.0;  // (0,1)
            u[2] = 1.0;  // (1,0)
            break;
        }
        case GateType::PAULI_Y: {
            // Y = [[0, -i], [i, 0]]
            u[1] = std::complex<double>(0.0, -1.0);  // (0,1)
            u[2] = std::complex<double>(0.0, 1.0);   // (1,0)
            break;
        }
        case GateType::PAULI_Z: {
            // Z = [[1, 0], [0, -1]]
            u[0] = 1.0;   // (0,0)
            u[3] = -1.0;  // (1,1)
            break;
        }
        case GateType::HADAMARD: {
            // H = 1/√2 * [[1, 1], [1, -1]]
            double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            u[0] = inv_sqrt2;  // (0,0)
            u[1] = inv_sqrt2;  // (0,1)
            u[2] = inv_sqrt2;  // (1,0)
            u[3] = -inv_sqrt2; // (1,1)
            break;
        }
        case GateType::ROTATION_X: {
            if (gate.params.empty()) {
                throw std::runtime_error("Rx门需要角度参数");
            }
            double theta = gate.params[0].real();
            double cos_half = std::cos(theta / 2.0);
            double sin_half = std::sin(theta / 2.0);
            // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
            u[0] = cos_half;                           // (0,0)
            u[1] = std::complex<double>(0.0, -sin_half); // (0,1)
            u[2] = std::complex<double>(0.0, -sin_half); // (1,0)
            u[3] = cos_half;                           // (1,1)
            break;
        }
        case GateType::ROTATION_Y: {
            if (gate.params.empty()) {
                throw std::runtime_error("Ry门需要角度参数");
            }
            double theta = gate.params[0].real();
            double cos_half = std::cos(theta / 2.0);
            double sin_half = std::sin(theta / 2.0);
            // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
            u[0] = cos_half;     // (0,0)
            u[1] = -sin_half;    // (0,1)
            u[2] = sin_half;     // (1,0)
            u[3] = cos_half;     // (1,1)
            break;
        }
        case GateType::ROTATION_Z: {
            if (gate.params.empty()) {
                throw std::runtime_error("Rz门需要角度参数");
            }
            double theta = gate.params[0].real();
            double cos_half = std::cos(theta / 2.0);
            double sin_half = std::sin(theta / 2.0);
            // Rz(θ) = [[cos(θ/2)-i*sin(θ/2), 0], [0, cos(θ/2)+i*sin(θ/2)]]
            // = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
            u[0] = std::complex<double>(cos_half, -sin_half); // (0,0)
            u[3] = std::complex<double>(cos_half, sin_half);  // (1,1)
            break;
        }
        case GateType::PHASE_GATE_S: {
            // S = [[1, 0], [0, i]]
            u[0] = 1.0;  // (0,0)
            u[3] = std::complex<double>(0.0, 1.0); // (1,1)
            break;
        }
        case GateType::PHASE_GATE_T: {
            // T = [[1, 0], [0, e^(iπ/4)]]
            u[0] = 1.0;  // (0,0)
            u[3] = std::complex<double>(std::cos(M_PI/4.0), std::sin(M_PI/4.0)); // (1,1)
            break;
        }
        case GateType::CNOT: {
            // CNOT是双比特门，需要控制位和目标位
            if (gate.target_qubits.size() < 2) {
                throw std::runtime_error("CNOT门需要控制位和目标位");
            }
            int control = gate.target_qubits[0];
            int target = gate.target_qubits[1];
            root_node_ = apply_cnot_recursive(root_node_, control, target);
            return;  // CNOT特殊处理，不走单比特门逻辑
        }
        case GateType::CZ: {
            // CZ也是双比特门
            if (gate.target_qubits.size() < 2) {
                throw std::runtime_error("CZ门需要控制位和目标位");
            }
            int control = gate.target_qubits[0];
            int target = gate.target_qubits[1];
            // CZ可以通过修改CNOT来实现，或者直接实现
            // 这里暂时用CNOT的逻辑，后面可以优化
            root_node_ = apply_cz_recursive(root_node_, control, target);
            return;  // CZ特殊处理
        }
        default:
            throw std::runtime_error("不支持的Qubit门类型");
    }

    // 应用单比特门到HDD
    root_node_ = apply_single_qubit_gate_recursive(root_node_, target_qubit, u);
}

/**
 * 执行混合门操作 (CPU+GPU)
 */
void QuantumCircuit::execute_hybrid_gate(const GateParams& gate) {
    // 混合门需要同时操作Qubit和Qumode
    // 根据gatemath.md，这些门分为分离型和混合型

    switch (gate.type) {
        // 分离型受控门
        case GateType::CONDITIONAL_DISPLACEMENT: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("CD门需要控制Qubit和目标Qumode");
            }
            // 实现CD门：当Qubit为|1⟩时应用Displacement
            execute_conditional_displacement(gate);
            break;
        }
        case GateType::CONDITIONAL_SQUEEZING: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("CS门需要控制Qubit和目标Qumode");
            }
            execute_conditional_squeezing(gate);
            break;
        }
        case GateType::CONDITIONAL_BEAM_SPLITTER: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 2) {
                throw std::runtime_error("CBS门需要控制Qubit和两个目标Qumode");
            }
            execute_conditional_beam_splitter(gate);
            break;
        }

        // 混合型相互作用门
        case GateType::RABI_INTERACTION: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("RB门需要控制Qubit和目标Qumode");
            }
            execute_rabi_interaction(gate);
            break;
        }
        case GateType::JAYNES_CUMMINGS: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("JC门需要控制Qubit和目标Qumode");
            }
            execute_jaynes_cummings(gate);
            break;
        }
        case GateType::ANTI_JAYNES_CUMMINGS: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("AJC门需要控制Qubit和目标Qumode");
            }
            execute_anti_jaynes_cummings(gate);
            break;
        }
        case GateType::SELECTIVE_QUBIT_ROTATION: {
            if (gate.target_qubits.size() < 1 || gate.target_qumodes.size() < 1) {
                throw std::runtime_error("SQR门需要目标Qubit和控制Qumode");
            }
            execute_selective_qubit_rotation(gate);
            break;
        }

        default:
            throw std::runtime_error("未知的混合门类型: " + std::to_string(static_cast<int>(gate.type)));
    }
}

/**
 * 执行受控位移门 CD(α)
 * CD(α) = exp[σ_z ⊗ (α a† - α* a)]
 * 分离型：控制qubit为|0⟩时应用D(+α)，为|1⟩时应用D(-α)
 */
void QuantumCircuit::execute_conditional_displacement(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    std::complex<double> alpha = gate.params.empty() ? std::complex<double>(0.0, 0.0) : gate.params[0];

    // 递归遍历HDD，为每个分支应用相应的位移
    root_node_ = apply_conditional_displacement_recursive(root_node_, control_qubit, target_qumode, alpha);
}

/**
 * 递归应用条件位移门
 */
HDDNode* QuantumCircuit::apply_conditional_displacement_recursive(
    HDDNode* node, int control_qubit, int target_qumode, std::complex<double> alpha) {

    if (node->is_terminal()) {
        // 终端节点：应用位移门
        int state_id = node->tensor_id;
        apply_displacement_to_state(state_id, alpha);
        return node;
    }

    if (node->qubit_level == control_qubit) {
        // 控制qubit所在层级
        // 分支0 (|0⟩): 应用D(+α)
        HDDNode* low_branch = apply_conditional_displacement_recursive(
            node->low, control_qubit, target_qumode, alpha);

        // 分支1 (|1⟩): 应用D(-α)
        HDDNode* high_branch = apply_conditional_displacement_recursive(
            node->high, control_qubit, target_qumode, -alpha);

        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        // 递归到更深层级
        HDDNode* new_low = apply_conditional_displacement_recursive(
            node->low, control_qubit, target_qumode, alpha);
        HDDNode* new_high = apply_conditional_displacement_recursive(
            node->high, control_qubit, target_qumode, alpha);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        // 不需要处理的层级
        return node;
    }
}

/**
 * 对单个状态应用位移门
 */
void QuantumCircuit::apply_displacement_to_state(int state_id, std::complex<double> alpha) {
    // 上传位移参数到GPU
    cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.real(), alpha.imag());
    int* d_state_id = nullptr;
    cuDoubleComplex* d_alpha = nullptr;

    CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha_cu, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // 调用GPU位移门内核
    apply_displacement_gate(&state_pool_, d_state_id, 1, alpha_cu);

    CHECK_CUDA(cudaFree(d_state_id));
    CHECK_CUDA(cudaFree(d_alpha));
}

/**
 * 执行受控挤压门 CS(ξ)
 * CS(ξ) = exp[σ_z ⊗ (ξ* a²/2 - ξ* (a†)²/2)]
 * 分离型：控制qubit为|0⟩时应用S(+ξ)，为|1⟩时应用S(-ξ)
 */
void QuantumCircuit::execute_conditional_squeezing(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    std::complex<double> xi = gate.params.empty() ? std::complex<double>(0.0, 0.0) : gate.params[0];

    // 递归遍历HDD，为每个分支应用相应的挤压
    root_node_ = apply_conditional_squeezing_recursive(root_node_, control_qubit, target_qumode, xi);
}

/**
 * 递归应用条件挤压门
 */
HDDNode* QuantumCircuit::apply_conditional_squeezing_recursive(
    HDDNode* node, int control_qubit, int target_qumode, std::complex<double> xi) {

    if (node->is_terminal()) {
        // 终端节点：应用挤压门
        int state_id = node->tensor_id;
        apply_squeezing_to_state(state_id, xi);
        return node;
    }

    if (node->qubit_level == control_qubit) {
        // 分支0 (|0⟩): S(+ξ)
        HDDNode* low_branch = apply_conditional_squeezing_recursive(
            node->low, control_qubit, target_qumode, xi);

        // 分支1 (|1⟩): S(-ξ)
        HDDNode* high_branch = apply_conditional_squeezing_recursive(
            node->high, control_qubit, target_qumode, -xi);

        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_conditional_squeezing_recursive(
            node->low, control_qubit, target_qumode, xi);
        HDDNode* new_high = apply_conditional_squeezing_recursive(
            node->high, control_qubit, target_qumode, xi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用挤压门
 */
void QuantumCircuit::apply_squeezing_to_state(int state_id, std::complex<double> xi) {
    // 注意：目前single_mode_gates.cu中的挤压门实现可能不完整
    // 这里调用通用单模门，需要ELL算符
    FockELLOperator* ell_op = prepare_squeezing_ell_operator(xi);
    if (ell_op) {
        int* d_state_id = nullptr;
        CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));

        apply_single_mode_gate(&state_pool_, ell_op, d_state_id, 1);

        CHECK_CUDA(cudaFree(d_state_id));
        delete ell_op;
    }
}

/**
 * 执行受控光束分裂器 CBS(θ,φ)
 * CBS(θ,φ) = exp[-i θ/2 σ_z ⊗ (e^{iφ} a† b + e^{-iφ} a b†)]
 * 分离型：控制qubit为|0⟩时应用BS(+θ,φ)，为|1⟩时应用BS(-θ,φ)
 */
void QuantumCircuit::execute_conditional_beam_splitter(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode1 = gate.target_qumodes[0];
    int target_qumode2 = gate.target_qumodes[1];
    double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;

    // 递归遍历HDD，为每个分支应用相应的光束分裂器
    root_node_ = apply_conditional_beam_splitter_recursive(root_node_, control_qubit, target_qumode1, target_qumode2, theta, phi);
}

/**
 * 递归应用条件光束分裂器
 */
HDDNode* QuantumCircuit::apply_conditional_beam_splitter_recursive(
    HDDNode* node, int control_qubit, int qumode1, int qumode2, double theta, double phi) {

    if (node->is_terminal()) {
        // 终端节点：应用光束分裂器
        int state_id = node->tensor_id;
        apply_beam_splitter_to_state(state_id, theta, phi, num_qumodes_);
        return node;
    }

    if (node->qubit_level == control_qubit) {
        // 分支0 (|0⟩): BS(+θ, φ)
        HDDNode* low_branch = apply_conditional_beam_splitter_recursive(
            node->low, control_qubit, qumode1, qumode2, theta, phi);

        // 分支1 (|1⟩): BS(-θ, φ)
        HDDNode* high_branch = apply_conditional_beam_splitter_recursive(
            node->high, control_qubit, qumode1, qumode2, -theta, phi);

        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_conditional_beam_splitter_recursive(
            node->low, control_qubit, qumode1, qumode2, theta, phi);
        HDDNode* new_high = apply_conditional_beam_splitter_recursive(
            node->high, control_qubit, qumode1, qumode2, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用光束分裂器
 */
void QuantumCircuit::apply_beam_splitter_to_state(int state_id, double theta, double phi, int max_photon) {
    // 调用GPU光束分裂器内核
    int* d_state_id = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state_id, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_state_id, &state_id, sizeof(int), cudaMemcpyHostToDevice));

    apply_beam_splitter(&state_pool_, d_state_id, 1, theta, phi, max_photon);

    CHECK_CUDA(cudaFree(d_state_id));
}

/**
 * 执行受控双模挤压 CTMS(ξ)
 * CTMS(ξ) = exp[1/2 σ_z ⊗ (ξ* a† b† - ξ a b)]
 * 分离型：控制qubit为|0⟩时应用TMS(+ξ)，为|1⟩时应用TMS(-ξ)
 */
void QuantumCircuit::execute_conditional_two_mode_squeezing(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode1 = gate.target_qumodes[0];
    int target_qumode2 = gate.target_qumodes[1];
    std::complex<double> xi = gate.params.empty() ? std::complex<double>(0.0, 0.0) : gate.params[0];

    // 递归遍历HDD
    root_node_ = apply_conditional_two_mode_squeezing_recursive(root_node_, control_qubit, target_qumode1, target_qumode2, xi);
}

/**
 * 递归应用条件双模挤压
 */
HDDNode* QuantumCircuit::apply_conditional_two_mode_squeezing_recursive(
    HDDNode* node, int control_qubit, int qumode1, int qumode2, std::complex<double> xi) {

    if (node->is_terminal()) {
        // 应用双模挤压门（需要实现TMS内核）
        int state_id = node->tensor_id;
        apply_two_mode_squeezing_to_state(state_id, xi);
        return node;
    }

    if (node->qubit_level == control_qubit) {
        HDDNode* low_branch = apply_conditional_two_mode_squeezing_recursive(
            node->low, control_qubit, qumode1, qumode2, xi);
        HDDNode* high_branch = apply_conditional_two_mode_squeezing_recursive(
            node->high, control_qubit, qumode1, qumode2, -xi);
        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_conditional_two_mode_squeezing_recursive(
            node->low, control_qubit, qumode1, qumode2, xi);
        HDDNode* new_high = apply_conditional_two_mode_squeezing_recursive(
            node->high, control_qubit, qumode1, qumode2, xi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用双模挤压门
 */
void QuantumCircuit::apply_two_mode_squeezing_to_state(int state_id, std::complex<double> xi) {
    // TMS门需要新的GPU内核，目前使用占位符
    // TODO: 实现TMS GPU内核
    std::cout << "TMS门应用到状态 " << state_id << "，参数 ξ = " << xi << std::endl;
}

/**
 * 执行受控SUM门
 */
void QuantumCircuit::execute_conditional_sum(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode1 = gate.target_qumodes[0];
    int target_qumode2 = gate.target_qumodes[1];
    double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;

    root_node_ = apply_conditional_sum_recursive(root_node_, control_qubit, target_qumode1, target_qumode2, theta, phi);
}

/**
 * 递归应用条件SUM门
 */
HDDNode* QuantumCircuit::apply_conditional_sum_recursive(
    HDDNode* node, int control_qubit, int qumode1, int qumode2, double theta, double phi) {

    if (node->is_terminal()) {
        int state_id = node->tensor_id;
        apply_sum_to_state(state_id, theta, phi);
        return node;
    }

    if (node->qubit_level == control_qubit) {
        HDDNode* low_branch = apply_conditional_sum_recursive(
            node->low, control_qubit, qumode1, qumode2, theta, phi);
        HDDNode* high_branch = apply_conditional_sum_recursive(
            node->high, control_qubit, qumode1, qumode2, -theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_conditional_sum_recursive(
            node->low, control_qubit, qumode1, qumode2, theta, phi);
        HDDNode* new_high = apply_conditional_sum_recursive(
            node->high, control_qubit, qumode1, qumode2, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用SUM门
 */
void QuantumCircuit::apply_sum_to_state(int state_id, double theta, double phi) {
    // SUM门需要新的GPU内核，目前使用占位符
    std::cout << "SUM门应用到状态 " << state_id << "，参数 θ = " << theta << ", φ = " << phi << std::endl;
}

// ===== 混合型相互作用门实现 =====

/**
 * 执行Rabi振荡门 RB(θ)
 * RB(θ) = exp[-i θ σ_x ⊗ (a + a†)]
 * 混合型：Qubit和Qumode相互作用
 */
void QuantumCircuit::execute_rabi_interaction(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    double theta = gate.params.empty() ? 0.0 : gate.params[0].real();

    // Rabi相互作用需要同时处理qubit和qumode状态
    // 这是一个复杂的混合操作，需要扩展HDD处理逻辑
    root_node_ = apply_rabi_interaction_recursive(root_node_, control_qubit, target_qumode, theta);
}

/**
 * 递归应用Rabi相互作用
 */
HDDNode* QuantumCircuit::apply_rabi_interaction_recursive(
    HDDNode* node, int control_qubit, int target_qumode, double theta) {

    if (node->is_terminal()) {
        // 对于终端节点，需要应用完整的Rabi相互作用
        // RB(θ) = exp[-i θ σ_x ⊗ (a + a†)]
        // 这会混合qubit和qumode状态
        int state_id = node->tensor_id;
        apply_rabi_to_state(state_id, theta);
        return node;
    }

    if (node->qubit_level == control_qubit) {
        // 在控制qubit处，需要应用σ_x操作
        // RB分解为：对于每个分支，应用相应的旋转和qumode操作

        // 简化的实现：递归处理子节点
        HDDNode* low_branch = apply_rabi_interaction_recursive(
            node->low, control_qubit, target_qumode, theta);
        HDDNode* high_branch = apply_rabi_interaction_recursive(
            node->high, control_qubit, target_qumode, theta);

        // 对于Rabi门，需要更复杂的权重计算
        // 这里使用简化的实现
        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_rabi_interaction_recursive(
            node->low, control_qubit, target_qumode, theta);
        HDDNode* new_high = apply_rabi_interaction_recursive(
            node->high, control_qubit, target_qumode, theta);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用Rabi相互作用
 */
void QuantumCircuit::apply_rabi_to_state(int state_id, double theta) {
    // Rabi相互作用需要新的GPU内核
    // RB(θ)涉及σ_x ⊗ (a + a†)，需要同时处理qubit和qumode
    std::cout << "Rabi相互作用应用到状态 " << state_id << "，参数 θ = " << theta << std::endl;
}

/**
 * 执行Jaynes-Cummings相互作用 JC(θ,φ)
 * JC(θ,φ) = exp[-iθ(e^{iφ} σ- a† + e^{-iφ} σ+ a)]
 */
void QuantumCircuit::execute_jaynes_cummings(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;

    root_node_ = apply_jaynes_cummings_recursive(root_node_, control_qubit, target_qumode, theta, phi);
}

/**
 * 递归应用Jaynes-Cummings相互作用
 */
HDDNode* QuantumCircuit::apply_jaynes_cummings_recursive(
    HDDNode* node, int control_qubit, int target_qumode, double theta, double phi) {

    if (node->is_terminal()) {
        int state_id = node->tensor_id;
        apply_jaynes_cummings_to_state(state_id, theta, phi);
        return node;
    }

    if (node->qubit_level == control_qubit) {
        // JC门会混合qubit状态，需要更复杂的处理
        HDDNode* low_branch = apply_jaynes_cummings_recursive(
            node->low, control_qubit, target_qumode, theta, phi);
        HDDNode* high_branch = apply_jaynes_cummings_recursive(
            node->high, control_qubit, target_qumode, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_jaynes_cummings_recursive(
            node->low, control_qubit, target_qumode, theta, phi);
        HDDNode* new_high = apply_jaynes_cummings_recursive(
            node->high, control_qubit, target_qumode, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用Jaynes-Cummings相互作用
 */
void QuantumCircuit::apply_jaynes_cummings_to_state(int state_id, double theta, double phi) {
    // JC相互作用需要新的GPU内核
    std::cout << "JC相互作用应用到状态 " << state_id << "，参数 θ = " << theta << ", φ = " << phi << std::endl;
}

/**
 * 执行Anti-Jaynes-Cummings相互作用 AJC(θ,φ)
 */
void QuantumCircuit::execute_anti_jaynes_cummings(const GateParams& gate) {
    int control_qubit = gate.target_qubits[0];
    int target_qumode = gate.target_qumodes[0];
    double theta = gate.params.size() > 0 ? gate.params[0].real() : 0.0;
    double phi = gate.params.size() > 1 ? gate.params[1].real() : 0.0;

    root_node_ = apply_anti_jaynes_cummings_recursive(root_node_, control_qubit, target_qumode, theta, phi);
}

/**
 * 递归应用Anti-Jaynes-Cummings相互作用
 */
HDDNode* QuantumCircuit::apply_anti_jaynes_cummings_recursive(
    HDDNode* node, int control_qubit, int target_qumode, double theta, double phi) {

    if (node->is_terminal()) {
        int state_id = node->tensor_id;
        apply_anti_jaynes_cummings_to_state(state_id, theta, phi);
        return node;
    }

    if (node->qubit_level == control_qubit) {
        HDDNode* low_branch = apply_anti_jaynes_cummings_recursive(
            node->low, control_qubit, target_qumode, theta, phi);
        HDDNode* high_branch = apply_anti_jaynes_cummings_recursive(
            node->high, control_qubit, target_qumode, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > control_qubit) {
        HDDNode* new_low = apply_anti_jaynes_cummings_recursive(
            node->low, control_qubit, target_qumode, theta, phi);
        HDDNode* new_high = apply_anti_jaynes_cummings_recursive(
            node->high, control_qubit, target_qumode, theta, phi);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用Anti-Jaynes-Cummings相互作用
 */
void QuantumCircuit::apply_anti_jaynes_cummings_to_state(int state_id, double theta, double phi) {
    // AJC相互作用需要新的GPU内核
    std::cout << "AJC相互作用应用到状态 " << state_id << "，参数 θ = " << theta << ", φ = " << phi << std::endl;
}

/**
 * 执行选择性Qubit旋转 SQR(θ,φ)
 */
void QuantumCircuit::execute_selective_qubit_rotation(const GateParams& gate) {
    int target_qubit = gate.target_qubits[0];
    int control_qumode = gate.target_qumodes[0];
    std::vector<double> theta_vec, phi_vec;

    // 从参数中提取θ和φ向量
    size_t num_params = gate.params.size();
    for (size_t i = 0; i < num_params / 2; ++i) {
        theta_vec.push_back(gate.params[2 * i].real());
        phi_vec.push_back(gate.params[2 * i + 1].real());
    }

    root_node_ = apply_selective_qubit_rotation_recursive(root_node_, target_qubit, control_qumode, theta_vec, phi_vec);
}

/**
 * 递归应用选择性Qubit旋转
 */
HDDNode* QuantumCircuit::apply_selective_qubit_rotation_recursive(
    HDDNode* node, int target_qubit, int control_qumode,
    const std::vector<double>& theta_vec, const std::vector<double>& phi_vec) {

    if (node->is_terminal()) {
        int state_id = node->tensor_id;
        apply_selective_qubit_rotation_to_state(state_id, theta_vec, phi_vec);
        return node;
    }

    if (node->qubit_level == target_qubit) {
        // SQR根据qumode的光子数选择性旋转qubit
        // 需要从qumode状态推断光子数，这里简化为应用到所有分支
        HDDNode* low_branch = apply_selective_qubit_rotation_recursive(
            node->low, target_qubit, control_qumode, theta_vec, phi_vec);
        HDDNode* high_branch = apply_selective_qubit_rotation_recursive(
            node->high, target_qubit, control_qumode, theta_vec, phi_vec);
        return node_manager_.get_or_create_node(node->qubit_level, low_branch, high_branch,
                                                 node->w_low, node->w_high);
    } else if (node->qubit_level > target_qubit) {
        HDDNode* new_low = apply_selective_qubit_rotation_recursive(
            node->low, target_qubit, control_qumode, theta_vec, phi_vec);
        HDDNode* new_high = apply_selective_qubit_rotation_recursive(
            node->high, target_qubit, control_qumode, theta_vec, phi_vec);
        return node_manager_.get_or_create_node(node->qubit_level, new_low, new_high,
                                                 node->w_low, node->w_high);
    } else {
        return node;
    }
}

/**
 * 对单个状态应用选择性Qubit旋转
 */
void QuantumCircuit::apply_selective_qubit_rotation_to_state(
    int state_id, const std::vector<double>& theta_vec, const std::vector<double>& phi_vec) {
    // SQR需要根据光子数选择旋转角度
    // 这里需要访问qumode状态来确定光子数
    std::cout << "SQR应用到状态 " << state_id << std::endl;
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
 * 准备挤压门的ELL算符
 */
FockELLOperator* QuantumCircuit::prepare_squeezing_ell_operator(std::complex<double> xi) {
    // 简化的实现：返回基本的ELL算符
    // 在实际实现中，需要构建正确的挤压矩阵
    FockELLOperator* ell_op = new FockELLOperator(cv_truncation_, 10);
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

    GateParams ConditionalDisplacement(int control_qubit, int target_qumode, std::complex<double> alpha) {
        return GateParams(GateType::CONDITIONAL_DISPLACEMENT, {control_qubit}, {target_qumode}, {alpha});
    }

    GateParams ConditionalSqueezing(int control_qubit, int target_qumode, std::complex<double> xi) {
        return GateParams(GateType::CONDITIONAL_SQUEEZING, {control_qubit}, {target_qumode}, {xi});
    }

    GateParams ConditionalBeamSplitter(int control_qubit, int target_qumode1, int target_qumode2, double theta, double phi) {
        return GateParams(GateType::CONDITIONAL_BEAM_SPLITTER, {control_qubit}, {target_qumode1, target_qumode2}, {theta, phi});
    }

    GateParams ConditionalTwoModeSqueezing(int control_qubit, int target_qumode1, int target_qumode2, std::complex<double> xi) {
        return GateParams(GateType::CONDITIONAL_TWO_MODE_SQUEEZING, {control_qubit}, {target_qumode1, target_qumode2}, {xi});
    }

    GateParams ConditionalSUM(int control_qubit, int target_qumode1, int target_qumode2, double theta, double phi) {
        return GateParams(GateType::CONDITIONAL_SUM, {control_qubit}, {target_qumode1, target_qumode2}, {theta, phi});
    }

    GateParams RabiInteraction(int control_qubit, int target_qumode, double theta) {
        return GateParams(GateType::RABI_INTERACTION, {control_qubit}, {target_qumode}, {theta});
    }

    GateParams JaynesCummings(int control_qubit, int target_qumode, double theta, double phi) {
        return GateParams(GateType::JAYNES_CUMMINGS, {control_qubit}, {target_qumode}, {theta, phi});
    }

    GateParams AntiJaynesCummings(int control_qubit, int target_qumode, double theta, double phi) {
        return GateParams(GateType::ANTI_JAYNES_CUMMINGS, {control_qubit}, {target_qumode}, {theta, phi});
    }

    GateParams SelectiveQubitRotation(int target_qubit, int control_qumode, const std::vector<double>& theta_vec, const std::vector<double>& phi_vec) {
        std::vector<std::complex<double>> params;
        params.reserve(theta_vec.size() + phi_vec.size());
        for (double theta : theta_vec) params.push_back(theta);
        for (double phi : phi_vec) params.push_back(phi);
        return GateParams(GateType::SELECTIVE_QUBIT_ROTATION, {target_qubit}, {control_qumode}, params);
    }

    // Qubit门
    GateParams Hadamard(int qubit) {
        return GateParams(GateType::HADAMARD, {qubit});
    }

    GateParams PauliX(int qubit) {
        return GateParams(GateType::PAULI_X, {qubit});
    }

    GateParams PauliY(int qubit) {
        return GateParams(GateType::PAULI_Y, {qubit});
    }

    GateParams PauliZ(int qubit) {
        return GateParams(GateType::PAULI_Z, {qubit});
    }

    GateParams RotationX(int qubit, double theta) {
        return GateParams(GateType::ROTATION_X, {qubit}, {}, {theta});
    }

    GateParams RotationY(int qubit, double theta) {
        return GateParams(GateType::ROTATION_Y, {qubit}, {}, {theta});
    }

    GateParams RotationZ(int qubit, double theta) {
        return GateParams(GateType::ROTATION_Z, {qubit}, {}, {theta});
    }

    GateParams PhaseGateS(int qubit) {
        return GateParams(GateType::PHASE_GATE_S, {qubit});
    }

    GateParams PhaseGateT(int qubit) {
        return GateParams(GateType::PHASE_GATE_T, {qubit});
    }

    GateParams CNOT(int control, int target) {
        return GateParams(GateType::CNOT, {control, target});
    }

    GateParams CZ(int control, int target) {
        return GateParams(GateType::CZ, {control, target});
    }
}
