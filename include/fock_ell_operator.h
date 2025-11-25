#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <memory>

/**
 * Fock-ELL 算符存储格式
 *
 * 针对Displacement等带状矩阵的优化存储格式
 * 使用ELL (ELLpack) 格式存储稀疏矩阵
 */
struct FockELLOperator {
    // 存储非零元素的值
    // 维度: [dim, max_bandwidth] (Row-Major)
    cuDoubleComplex* ell_val = nullptr;

    // 存储对应的列索引
    // 维度: [dim, max_bandwidth]
    // 值 -1 表示该位置为 Padding
    int* ell_col = nullptr;

    int max_bandwidth = 0; // K_eff (有效带宽)
    int dim = 0;           // D (矩阵维度)

    // 主机端副本
    std::vector<cuDoubleComplex> host_ell_val;
    std::vector<int> host_ell_col;

    /**
     * 构造函数
     * @param matrix_dim 矩阵维度D
     * @param bandwidth 最大带宽K
     */
    FockELLOperator(int matrix_dim, int bandwidth);

    /**
     * 析构函数
     */
    ~FockELLOperator();

    /**
     * 从稠密矩阵构建ELL格式
     * @param dense_matrix 稠密矩阵 (dim x dim)
     * @param tolerance 非零元素阈值
     */
    void build_from_dense(const std::vector<cuDoubleComplex>& dense_matrix, double tolerance = 1e-12);

    /**
     * 从带状矩阵构建ELL格式 (更高效)
     * @param diagonals 对角线数据，每行是一个对角线
     * @param offsets 对角线偏移量
     */
    void build_from_diagonals(const std::vector<std::vector<cuDoubleComplex>>& diagonals,
                             const std::vector<int>& offsets);

    /**
     * 获取矩阵元素 (主机端)
     * @param row 行索引
     * @param col 列索引
     * @return 矩阵元素值
     */
    cuDoubleComplex get_element(int row, int col) const;

    /**
     * 设置矩阵元素 (主机端)
     * @param row 行索引
     * @param col 列索引
     * @param value 值
     */
    void set_element(int row, int col, cuDoubleComplex value);

    /**
     * 上传数据到GPU
     */
    void upload_to_gpu();

    /**
     * 从GPU下载数据到主机
     */
    void download_from_gpu();

    /**
     * 检查矩阵是否为空
     */
    bool is_empty() const { return dim == 0 || max_bandwidth == 0; }

    /**
     * 获取非零元素数量
     */
    int get_nnz() const;

private:
    // 禁用拷贝构造和赋值
    FockELLOperator(const FockELLOperator&) = delete;
    FockELLOperator& operator=(const FockELLOperator&) = delete;

    /**
     * 重新分配GPU内存
     */
    void realloc_gpu_memory();
};
