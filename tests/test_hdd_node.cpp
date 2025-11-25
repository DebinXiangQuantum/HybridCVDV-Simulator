#include <gtest/gtest.h>
#include "hdd_node.h"
#include <memory>

/**
 * HDDNode 单元测试
 */
class HDDNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        node_manager = new HDDNodeManager();
    }

    void TearDown() override {
        delete node_manager;
    }

    HDDNodeManager* node_manager;
};

TEST_F(HDDNodeTest, TerminalNodeCreation) {
    int cv_state_id = 42;
    HDDNode* terminal_node = node_manager->create_terminal_node(cv_state_id);

    ASSERT_NE(terminal_node, nullptr);
    EXPECT_TRUE(terminal_node->is_terminal());
    EXPECT_EQ(terminal_node->qubit_level, -1);
    EXPECT_EQ(terminal_node->tensor_id, cv_state_id);
    EXPECT_EQ(terminal_node->get_ref_count(), 1);
}

TEST_F(HDDNodeTest, InternalNodeCreation) {
    HDDNode* low_child = node_manager->create_terminal_node(1);
    HDDNode* high_child = node_manager->create_terminal_node(2);

    std::complex<double> w_low(1.0, 0.0);
    std::complex<double> w_high(0.0, 1.0);

    HDDNode* internal_node = node_manager->get_or_create_node(0, low_child, high_child, w_low, w_high);

    ASSERT_NE(internal_node, nullptr);
    EXPECT_FALSE(internal_node->is_terminal());
    EXPECT_EQ(internal_node->qubit_level, 0);
    EXPECT_EQ(internal_node->low, low_child);
    EXPECT_EQ(internal_node->high, high_child);
    EXPECT_EQ(internal_node->w_low, w_low);
    EXPECT_EQ(internal_node->w_high, w_high);
    EXPECT_EQ(internal_node->get_ref_count(), 1);
}

TEST_F(HDDNodeTest, NodeDeduplication) {
    // 创建两个相同的终端节点
    HDDNode* terminal1 = node_manager->create_terminal_node(100);
    HDDNode* terminal2 = node_manager->create_terminal_node(100);

    // 应该返回相同的节点 (去重)
    EXPECT_EQ(terminal1, terminal2);
    EXPECT_EQ(terminal1->get_ref_count(), 2);
    EXPECT_EQ(node_manager->get_cache_size(), 1);

    // 创建相同的内部节点
    HDDNode* internal1 = node_manager->get_or_create_node(1, terminal1, terminal2);
    HDDNode* internal2 = node_manager->get_or_create_node(1, terminal1, terminal2);

    EXPECT_EQ(internal1, internal2);
    EXPECT_EQ(internal1->get_ref_count(), 2);
    EXPECT_EQ(node_manager->get_cache_size(), 2);  // 终端节点 + 内部节点
}

TEST_F(HDDNodeTest, ReferenceCounting) {
    HDDNode* terminal_node = node_manager->create_terminal_node(1);
    EXPECT_EQ(terminal_node->get_ref_count(), 1);

    // 增加引用计数
    terminal_node->increment_ref();
    EXPECT_EQ(terminal_node->get_ref_count(), 2);

    // 减少引用计数
    int new_count = terminal_node->decrement_ref();
    EXPECT_EQ(new_count, 1);
    EXPECT_EQ(terminal_node->get_ref_count(), 1);
}

TEST_F(HDDNodeTest, NodeRelease) {
    HDDNode* terminal_node = node_manager->create_terminal_node(1);
    EXPECT_EQ(node_manager->get_cache_size(), 1);

    // 释放节点 (引用计数减为0)
    node_manager->release_node(terminal_node);
    EXPECT_EQ(node_manager->get_cache_size(), 0);
}

TEST_F(HDDNodeTest, DeepCopy) {
    HDDNode* original = node_manager->create_terminal_node(5);
    HDDNode* copy = original->deep_copy();

    ASSERT_NE(copy, nullptr);
    EXPECT_TRUE(copy->is_terminal());
    EXPECT_EQ(copy->tensor_id, original->tensor_id);
    EXPECT_EQ(copy->get_ref_count(), 1);  // 新节点独立引用计数
    EXPECT_NE(copy->get_unique_id(), original->get_unique_id());  // 不同ID
}

TEST_F(HDDNodeTest, HashConsistency) {
    HDDNode* node1 = node_manager->create_terminal_node(10);
    HDDNode* node2 = node_manager->create_terminal_node(10);

    // 相同内容的节点应该有相同的哈希
    EXPECT_EQ(node1->get_unique_id(), node2->get_unique_id());

    // 创建不同内容的节点
    HDDNode* node3 = node_manager->create_terminal_node(20);
    EXPECT_NE(node1->get_unique_id(), node3->get_unique_id());
}

TEST_F(HDDNodeTest, GarbageCollection) {
    // 创建一些节点
    HDDNode* node1 = node_manager->create_terminal_node(1);
    HDDNode* node2 = node_manager->create_terminal_node(2);

    EXPECT_EQ(node_manager->get_cache_size(), 2);

    // 手动释放一个节点
    node_manager->release_node(node1);
    EXPECT_EQ(node_manager->get_cache_size(), 1);

    // 执行垃圾回收
    node_manager->garbage_collect();
    EXPECT_EQ(node_manager->get_cache_size(), 1);  // 只剩下node2
}

TEST_F(HDDNodeTest, ComplexNodeStructure) {
    // 创建一个小的HDD树结构
    // 层级 1
    HDDNode* leaf00 = node_manager->create_terminal_node(0);
    HDDNode* leaf01 = node_manager->create_terminal_node(1);
    HDDNode* level1_node0 = node_manager->get_or_create_node(1, leaf00, leaf01);

    HDDNode* leaf10 = node_manager->create_terminal_node(2);
    HDDNode* leaf11 = node_manager->create_terminal_node(3);
    HDDNode* level1_node1 = node_manager->get_or_create_node(1, leaf10, leaf11);

    // 层级 0 (根节点)
    HDDNode* root = node_manager->get_or_create_node(0, level1_node0, level1_node1);

    // 验证结构
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->qubit_level, 0);
    EXPECT_EQ(root->low, level1_node0);
    EXPECT_EQ(root->high, level1_node1);

    // 验证引用计数
    EXPECT_EQ(leaf00->get_ref_count(), 2);  // 被level1_node0和root引用
    EXPECT_EQ(level1_node0->get_ref_count(), 2);  // 被root引用
    EXPECT_EQ(root->get_ref_count(), 1);  // 根节点
}

TEST_F(HDDNodeTest, NodeManagerClear) {
    // 创建一些节点
    node_manager->create_terminal_node(1);
    node_manager->create_terminal_node(2);
    node_manager->get_or_create_node(0,
        node_manager->create_terminal_node(3),
        node_manager->create_terminal_node(4));

    EXPECT_GT(node_manager->get_cache_size(), 0);

    // 清空管理器
    node_manager->clear();
    EXPECT_EQ(node_manager->get_cache_size(), 0);
}

TEST_F(HDDNodeTest, InvalidOperations) {
    // 测试无效操作
    HDDNode* terminal = node_manager->create_terminal_node(1);

    // 尝试创建无效的内部节点 (负层级)
    EXPECT_THROW({
        HDDNode invalid_node(-2, terminal, terminal);
    }, std::invalid_argument);

    // 清理
    node_manager->release_node(terminal);
}
