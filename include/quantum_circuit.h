#pragma once

#include <vector>
#include <complex>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include "gaussian_mixture.h"
#include "symplectic_math.h"
#include "cv_state_pool.h"
#include "fock_ell_operator.h"
#include "hdd_node.h"

class GaussianStatePool;

/**
 * 门操作类型枚举
 */
enum class GateType {
    // CPU端纯Qubit门
    HADAMARD,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    ROTATION_X,
    ROTATION_Y,
    ROTATION_Z,
    PHASE_GATE_S,    // S门
    PHASE_GATE_T,    // T门
    CNOT,
    CZ,

    // GPU端纯Qumode门 (Level 0-3)
    PHASE_ROTATION,  // R(θ) - 相位旋转
    KERR_GATE,       // Kerr门
    CONDITIONAL_PARITY, // 条件奇偶
    SNAP_GATE,       // SNAP(θ, n)
    MULTI_SNAP_GATE, // Multi-SNAP({θ_n})
    CROSS_KERR_GATE, // Cross-Kerr / CK(κ)
    CREATION_OPERATOR,
    ANNIHILATION_OPERATOR,
    DISPLACEMENT,    // D(α) - 位移门
    SQUEEZING,       // S(ξ) - 挤压门
    BEAM_SPLITTER,   // BS(θ,φ) - 光束分裂器

    // CPU+GPU混合门 (分离型)
    CONDITIONAL_DISPLACEMENT,     // CD(α)
    CONDITIONAL_SQUEEZING,        // CS(ξ)
    CONDITIONAL_BEAM_SPLITTER,    // CBS(θ,φ)
    CONDITIONAL_TWO_MODE_SQUEEZING, // CTMS
    CONDITIONAL_SUM,              // SUM门

    // CPU+GPU混合门 (混合型)
    RABI_INTERACTION,             // RB(θ)
    JAYNES_CUMMINGS,              // JC(θ,φ)
    ANTI_JAYNES_CUMMINGS,         // AJC(θ,φ)
    SELECTIVE_QUBIT_ROTATION      // SQR(θ,φ)
};

/**
 * 门操作参数结构体
 */
struct GateParams {
    GateType type;
    std::vector<int> target_qubits;     // 目标Qubit索引
    std::vector<int> target_qumodes;    // 目标Qumode索引
    std::vector<std::complex<double>> params;  // 门参数

    // 构造函数
    GateParams(GateType t) : type(t) {}
    GateParams(GateType t, const std::vector<int>& qubits,
               const std::vector<int>& qumodes = {},
               const std::vector<std::complex<double>>& p = {})
        : type(t), target_qubits(qubits), target_qumodes(qumodes), params(p) {}
};

/**
 * 量子线路类
 * 管理量子门的序列和执行
 */
class QuantumCircuit {
private:
    int num_qubits_;              // Qubit数量
    int num_qumodes_;             // Qumode数量
    int cv_truncation_;           // CV状态截断维度

    HDDNode* root_node_;          // HDD根节点
    HDDNodeManager node_manager_; // HDD节点管理器
    CVStatePool state_pool_;      // CV状态池
    std::unique_ptr<GaussianStatePool> gaussian_state_pool_; // 持久Gaussian branch状态池

    std::vector<GateParams> gate_sequence_;  // 门操作序列

    // 内部状态
    bool is_built_;               // 是否已构建
    bool is_executed_;            // 是否已执行
    int shared_zero_state_id_;    // HDD共享零分支对应的隐藏零态

    // 时间统计
    double total_time_;           // 总执行时间 (毫秒)
    double transfer_time_;        // 传输时延 (毫秒)
    double computation_time_;     // 计算时延 (毫秒)
    double planning_time_;        // 块级编译/规划时间 (毫秒)
    int gaussian_symbolic_mode_limit_; // Gaussian symbolic路径的活跃mode上限
    int next_symbolic_terminal_id_; // 负ID用于symbolic terminal sidecar

public:
    /**
     * 构造函数
     * @param num_qubits Qubit数量
     * @param num_qumodes Qumode数量
     * @param cv_truncation CV状态截断维度
     * @param max_states 最大状态数量
     */
    QuantumCircuit(int num_qubits, int num_qumodes, int cv_truncation, int max_states = 1024);

    /**
     * 析构函数
     */
    ~QuantumCircuit();

    /**
     * 添加门操作到线路
     */
    void add_gate(const GateParams& gate);

    /**
     * 批量添加门操作
     */
    void add_gates(const std::vector<GateParams>& gates);

    /**
     * 构建量子线路 (准备执行)
     */
    void build();

    /**
     * 执行量子线路
     */
    void execute();

    /**
     * 重置量子线路状态
     */
    void reset();

    /**
     * 设置Gaussian symbolic执行的活跃mode上限
     */
    void set_gaussian_symbolic_mode_limit(int limit);

    /**
     * 获取Gaussian symbolic执行的活跃mode上限
     */
    int get_gaussian_symbolic_mode_limit() const { return gaussian_symbolic_mode_limit_; }

    /**
     * 获取最终状态的振幅
     * @param qubit_states Qubit状态向量 (0或1)
     * @param qumode_states Qumode状态向量
     * @return 状态振幅
     */
    std::complex<double> get_amplitude(const std::vector<int>& qubit_states,
                                     const std::vector<std::vector<std::complex<double>>>& qumode_states);

    /**
     * 获取状态池引用 (用于高级操作)
     */
    CVStatePool& get_state_pool() { return state_pool_; }
    const CVStatePool& get_state_pool() const { return state_pool_; }

    // 动态张量积管理
    struct StateConfig {
        std::vector<int> state_ids; // 每个mode对应的物理状态ID
    };
    std::map<int, StateConfig> state_configs_;
    int next_config_id_ = 0;

    int register_state_config(const std::vector<int>& ids);
    StateConfig get_state_config(int config_id);
    
    // 辅助：获取某mode所在的物理状态ID和该状态包含的modes
    struct PhysicalStateInfo {
        int physical_id;
        std::vector<int> contained_modes; // 包含的mode索引列表 (0..M-1)
    };
    PhysicalStateInfo get_physical_state_info(int config_id, int qumode_index);

    /**
     * 获取HDD根节点
     */
    HDDNode* get_root_node() { return root_node_; }

    /**
     * 获取节点管理器引用
     */
    HDDNodeManager& get_node_manager() { return node_manager_; }

    /**
     * 时间统计信息
     */
    struct TimeStats {
        double total_time;          // 总执行时间 (毫秒)
        double transfer_time;       // 传输时延 (毫秒)
        double computation_time;    // 计算时延 (毫秒)
        double planning_time;       // 块级编译/规划时间 (毫秒)
    };

    /**
     * 获取线路统计信息
     */
    struct CircuitStats {
        int num_qubits;
        int num_qumodes;
        int cv_truncation;
        int num_gates;
        int active_states;
        size_t hdd_nodes;
    };
    CircuitStats get_stats() const;

    /**
     * 获取时间统计信息
     */
    TimeStats get_time_stats() const;

private:
    enum class ExecutionBlockKind {
        Gaussian,
        DiagonalNonGaussian,
        QubitOnly,
        Other
    };

    struct ExecutionBlock {
        ExecutionBlockKind kind;
        size_t begin = 0;
        size_t end = 0;
    };

    struct CompiledExecutionBlock {
        ExecutionBlockKind kind = ExecutionBlockKind::Other;
        size_t begin = 0;
        size_t end = 0;
        std::vector<GateParams> gates;
        std::vector<SymplecticGate> gaussian_updates;
        std::vector<GaussianMixtureApproximation> diagonal_mixture_updates;
        double downstream_non_gaussianity = 0.0;
        double compile_time_ms = 0.0;
        double estimated_diagonal_l2_error = 0.0;
        double diagonal_fidelity_lower_bound = 1.0;
        size_t mixture_branch_count = 0;
        bool gaussian_ready = false;
        bool diagonal_mixture_ready = false;
        std::string compile_error;
    };

    struct SymbolicGaussianBranch {
        int gaussian_state_id = -1;
        std::complex<double> weight{1.0, 0.0};
        std::vector<GateParams> replay_gates;
    };

    struct SymbolicTerminalState {
        std::vector<SymbolicGaussianBranch> branches;
    };

    std::unordered_map<int, SymbolicTerminalState> symbolic_terminal_states_;

    /**
     * 初始化HDD结构
     */
    void initialize_hdd();

    /**
     * 执行单个门操作
     */
    void execute_gate(const GateParams& gate);

    /**
     * 规范化执行门序列，合并并重排可交换的纯CV对角门窗口
     */
    std::vector<GateParams> canonicalize_gate_sequence_for_execution() const;

    /**
     * 将规范化后的门序列划分为最大执行块
     */
    std::vector<ExecutionBlock> partition_execution_blocks(
        const std::vector<GateParams>& execution_sequence) const;

    /**
     * 编译单个执行块，预先生成Gaussian symplectic更新和执行策略元数据
     */
    CompiledExecutionBlock compile_execution_block(
        const std::vector<GateParams>& execution_sequence,
        const std::vector<ExecutionBlock>& execution_blocks,
        size_t block_index) const;

    /**
     * 尝试用Gaussian symbolic track执行某个Gaussian块，并在需要时切回Fock张量态
     * @return 是否成功完成块级加速切换
     */
    bool try_execute_gaussian_block_with_ede(
        const CompiledExecutionBlock& compiled_block);

    /**
     * 尝试用Gaussian Mixture近似执行纯CV对角非高斯块
     * @return 是否成功启用mixture路径
     */
    bool try_execute_diagonal_non_gaussian_block_with_mixture(
        const CompiledExecutionBlock& compiled_block);

    /**
     * 在GPU上对单个Fock终端态应用当前支持的Gaussian mixture近似
     */
    bool apply_gaussian_mixture_approximation_on_gpu(
        int state_id,
        const GaussianMixtureApproximation& approximation);

    bool has_symbolic_terminals() const;
    bool is_symbolic_terminal_id(int terminal_id) const;
    void ensure_gaussian_state_pool();
    int allocate_symbolic_terminal_id();
    void release_symbolic_terminal(int terminal_id);
    void clear_symbolic_terminals();
    std::vector<int> collect_symbolic_terminal_ids(HDDNode* root) const;
    void initialize_gaussian_vacuum_state(int gaussian_state_id);
    int duplicate_gaussian_state(int gaussian_state_id);
    void apply_symplectic_update_to_gaussian_states(
        const std::vector<int>& gaussian_state_ids,
        const SymplecticGate& gate);
    void apply_replayable_gaussian_gate_to_state(int state_id, const GateParams& gate);
    int project_symbolic_terminal_to_fock_state(int terminal_id);
    bool materialize_symbolic_terminals_to_fock();

    /**
     * 替换当前HDD根节点并回收旧分支引用的状态
     */
    void replace_root_node(HDDNode* new_root);

    /**
     * 执行Level 0门 (对角门)
     */
    void execute_level0_gate(const GateParams& gate);

    /**
     * 执行Level 1门 (梯算符门)
     */
    void execute_level1_gate(const GateParams& gate);

    /**
     * 执行Level 2门 (单模门)
     */
    void execute_level2_gate(const GateParams& gate);

    /**
     * 执行Level 3门 (双模门)
     */
    void execute_level3_gate(const GateParams& gate);

    /**
     * 执行Level 4门 (混合控制门)
     */
    void execute_level4_gate(const GateParams& gate);

    /**
     * 工具函数：准备ELL算符
     */
    FockELLOperator* prepare_ell_operator(const GateParams& gate);

    /**
     * 工具函数：收集需要更新的状态ID
     */
    std::vector<int> collect_target_states(const GateParams& gate);

    /**
     * 对单个状态应用位移门
     */
    void apply_displacement_to_state(int state_id, std::complex<double> alpha, int target_qumode = 0);

    /**
     * 对单个状态应用挤压门
     */
    void apply_squeezing_to_state(int state_id, std::complex<double> xi, int target_qumode = 0);

    /**
     * 对单个状态应用光束分裂器
     */
    void apply_beam_splitter_to_state(int state_id, double theta, double phi, int qumode1, int qumode2);

    /**
     * 对单个状态应用双模挤压门
     */
    void apply_two_mode_squeezing_to_state(int state_id, int qumode1, int qumode2, std::complex<double> xi);

    /**
     * 对单个状态应用SUM门
     */
    void apply_sum_to_state(int state_id, int qumode1, int qumode2, double theta, double phi);

    /**
     * 对单个状态应用Rabi相互作用
     */
    void apply_rabi_to_state(int state_id, double theta);

    /**
     * 对单个状态应用Jaynes-Cummings相互作用
     */
    void apply_jaynes_cummings_to_state(int state_id, double theta, double phi);

    /**
     * 对单个状态应用Anti-Jaynes-Cummings相互作用
     */
    void apply_anti_jaynes_cummings_to_state(int state_id, double theta, double phi);

    /**
     * 对单个状态应用选择性Qubit旋转
     */
    void apply_selective_qubit_rotation_to_state(int state_id, const std::vector<double>& theta_vec, const std::vector<double>& phi_vec);

    /**
     * 准备挤压门的ELL算符
     */
    FockELLOperator* prepare_squeezing_ell_operator(std::complex<double> xi);

    /**
     * HDD节点加法: result = w1 * n1 + w2 * n2
     */
    HDDNode* hdd_add(HDDNode* n1, std::complex<double> w1, HDDNode* n2, std::complex<double> w2);

    /**
     * HDD节点数乘: result = weight * node
     */
    HDDNode* scale_hdd_node(HDDNode* node, std::complex<double> weight);

    /**
     * 复制并缩放终端节点状态
     */
    HDDNode* duplicate_scaled_terminal_node(HDDNode* terminal_node, std::complex<double> weight);

    /**
     * 递归应用单qubit门
     */
    HDDNode* apply_single_qubit_gate_recursive(HDDNode* node, int target_qubit, const std::vector<std::complex<double>>& u);

    /**
     * 递归应用CNOT门
     */
    HDDNode* apply_cnot_recursive(HDDNode* node, int control, int target);

    /**
     * 递归应用CZ门
     */
    HDDNode* apply_cz_recursive(HDDNode* node, int control, int target);

    /**
     * 递归应用条件位移门
     */
    HDDNode* apply_conditional_displacement_recursive(HDDNode* node, int control_qubit, int target_qumode, std::complex<double> alpha);

    /**
     * 递归应用条件挤压门
     */
    HDDNode* apply_conditional_squeezing_recursive(HDDNode* node, int control_qubit, int target_qumode, std::complex<double> xi);

    /**
     * 递归应用条件光束分裂器
     */
    HDDNode* apply_conditional_beam_splitter_recursive(HDDNode* node, int control_qubit, int qumode1, int qumode2, double theta, double phi);

    /**
     * 递归应用条件双模挤压
     */
    HDDNode* apply_conditional_two_mode_squeezing_recursive(HDDNode* node, int control_qubit, int qumode1, int qumode2, std::complex<double> xi);

    /**
     * 递归应用条件SUM门
     */
    HDDNode* apply_conditional_sum_recursive(HDDNode* node, int control_qubit, int qumode1, int qumode2, double theta, double phi);

    /**
     * 递归应用Rabi相互作用
     */
    HDDNode* apply_rabi_interaction_recursive(HDDNode* node, int control_qubit, int target_qumode, double theta);

    /**
     * 对控制分支的低/高子树递归施加Rabi耦合
     */
    std::pair<HDDNode*, HDDNode*> apply_rabi_pair_recursive(
        HDDNode* low_node,
        std::complex<double> low_weight,
        HDDNode* high_node,
        std::complex<double> high_weight,
        int target_qumode,
        double theta);

    /**
     * 递归应用Jaynes-Cummings相互作用
     */
    HDDNode* apply_jaynes_cummings_recursive(HDDNode* node, int control_qubit, int target_qumode, double theta, double phi);

    /**
     * 对控制分支的低/高子树递归施加JC/AJC耦合
     */
    std::pair<HDDNode*, HDDNode*> apply_jc_like_pair_recursive(
        HDDNode* low_node,
        std::complex<double> low_weight,
        HDDNode* high_node,
        std::complex<double> high_weight,
        int target_qumode,
        double theta,
        double phi,
        bool anti_jaynes_cummings);

    /**
     * 递归应用Anti-Jaynes-Cummings相互作用
     */
    HDDNode* apply_anti_jaynes_cummings_recursive(HDDNode* node, int control_qubit, int target_qumode, double theta, double phi);

    /**
     * 递归应用选择性Qubit旋转
     */
    HDDNode* apply_selective_qubit_rotation_recursive(HDDNode* node, int target_qubit, int control_qumode, const std::vector<double>& theta_vec, const std::vector<double>& phi_vec);

    /**
     * 执行qubit门操作
     */
    void execute_qubit_gate(const GateParams& gate);

    /**
     * 执行混合门操作 (CPU+GPU)
     */
    void execute_hybrid_gate(const GateParams& gate);

    /**
     * 执行条件位移门 CD(α)
     */
    void execute_conditional_displacement(const GateParams& gate);

    /**
     * 执行条件挤压门 CS(ξ)
     */
    void execute_conditional_squeezing(const GateParams& gate);

    /**
     * 执行条件光束分裂器 CBS(θ,φ)
     */
    void execute_conditional_beam_splitter(const GateParams& gate);

    /**
     * 执行条件双模挤压 CTMS(ξ)
     */
    void execute_conditional_two_mode_squeezing(const GateParams& gate);

    /**
     * 执行条件SUM门
     */
    void execute_conditional_sum(const GateParams& gate);

    /**
     * 执行Rabi相互作用 RB(θ)
     */
    void execute_rabi_interaction(const GateParams& gate);

    /**
     * 执行Jaynes-Cummings相互作用 JC(θ,φ)
     */
    void execute_jaynes_cummings(const GateParams& gate);

    /**
     * 执行Anti-Jaynes-Cummings相互作用 AJC(θ,φ)
     */
    void execute_anti_jaynes_cummings(const GateParams& gate);

    /**
     * 执行选择性Qubit旋转 SQR(θ,φ)
     */
    void execute_selective_qubit_rotation(const GateParams& gate);

    // 禁用拷贝
    QuantumCircuit(const QuantumCircuit&) = delete;
    QuantumCircuit& operator=(const QuantumCircuit&) = delete;
};

/**
 * 便捷函数：创建常用门操作
 */
namespace Gates {
    // CPU端纯Qubit门
    GateParams Hadamard(int qubit);
    GateParams PauliX(int qubit);
    GateParams PauliY(int qubit);
    GateParams PauliZ(int qubit);
    GateParams RotationX(int qubit, double theta);
    GateParams RotationY(int qubit, double theta);
    GateParams RotationZ(int qubit, double theta);
    GateParams PhaseGateS(int qubit);
    GateParams PhaseGateT(int qubit);
    GateParams CNOT(int control, int target);
    GateParams CZ(int control, int target);

    // GPU端纯Qumode门
    GateParams PhaseRotation(int qumode, double theta);
    GateParams KerrGate(int qumode, double chi);
    GateParams ConditionalParity(int qumode, double parity);
    GateParams Snap(int qumode, double theta, int target_fock_state);
    GateParams MultiSNAP(int qumode, const std::vector<double>& phase_map);
    GateParams CrossKerr(int qumode1, int qumode2, double kappa);
    GateParams CreationOperator(int qumode);
    GateParams AnnihilationOperator(int qumode);
    GateParams Displacement(int qumode, std::complex<double> alpha);
    GateParams Squeezing(int qumode, std::complex<double> xi);
    GateParams BeamSplitter(int qumode1, int qumode2, double theta, double phi = 0.0);

    // CPU+GPU混合门 (分离型)
    GateParams ConditionalDisplacement(int control_qubit, int target_qumode, std::complex<double> alpha);
    GateParams ConditionalSqueezing(int control_qubit, int target_qumode, std::complex<double> xi);
    GateParams ConditionalBeamSplitter(int control_qubit, int target_qumode1, int target_qumode2, double theta, double phi = 0.0);
    GateParams ConditionalTwoModeSqueezing(int control_qubit, int target_qumode1, int target_qumode2, std::complex<double> xi);
    GateParams ConditionalSUM(int control_qubit, int target_qumode1, int target_qumode2, double theta, double phi = 0.0);

    // CPU+GPU混合门 (混合型)
    GateParams RabiInteraction(int control_qubit, int target_qumode, double theta);
    GateParams JaynesCummings(int control_qubit, int target_qumode, double theta, double phi = 0.0);
    GateParams AntiJaynesCummings(int control_qubit, int target_qumode, double theta, double phi = 0.0);
    GateParams SelectiveQubitRotation(int target_qubit, int control_qumode, const std::vector<double>& theta_vec, const std::vector<double>& phi_vec);
}
