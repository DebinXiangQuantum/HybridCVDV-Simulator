// 这个文件包含垃圾回收器的额外实现
// 主要逻辑已在 memory_pool.cpp 中实现

#include "memory_pool.h"
#include "cv_state_pool.h"
#include "hdd_node.h"

/**
 * 深度优先遍历HDD，更新状态引用
 * 这是一个复杂的操作，需要递归遍历整个HDD树
 */
void GarbageCollector::update_hdd_references(HDDNode* node, int old_state_id, int new_state_id) {
    if (!node) return;

    if (node->is_terminal()) {
        if (node->tensor_id == old_state_id) {
            node->tensor_id = new_state_id;
        }
    } else {
        // 递归遍历子节点
        update_hdd_references(node->low, old_state_id, new_state_id);
        update_hdd_references(node->high, old_state_id, new_state_id);
    }
}
