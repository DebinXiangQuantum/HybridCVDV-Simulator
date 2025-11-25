// 这个文件包含指令融合的额外实现
// 主要逻辑已在 batch_scheduler.cpp 中实现

#include "batch_scheduler.h"

// 额外的融合规则可以在这里添加
// 例如：连续的旋转门融合、相似门的重排序等

/**
 * 检查指令序列是否可以进行更复杂的融合
 */
bool InstructionFusion::can_fuse_sequence(const std::vector<GateParams>& sequence) const {
    if (sequence.size() < 2) return false;

    // 检查是否所有指令都是相同类型的
    GateType first_type = sequence.front().type;
    for (const auto& gate : sequence) {
        if (gate.type != first_type) return false;
    }

    // 检查是否作用在相同的量子比特/模上
    const auto& first_targets = sequence.front().target_qumodes;
    for (const auto& gate : sequence) {
        if (gate.target_qumodes != first_targets) return false;
        if (gate.target_qubits != sequence.front().target_qubits) return false;
    }

    return true;
}
