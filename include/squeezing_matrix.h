#pragma once

#include <vector>
#include <complex>

/**
 * 生成挤压门矩阵
 * 
 * @param r 挤压幅度
 * @param theta 挤压角度
 * @param cutoff Fock空间截断维度
 * @return 挤压门矩阵 (row-major, size = cutoff * cutoff)
 */
std::vector<std::complex<double>> generate_squeezing_matrix(double r, double theta, int cutoff);

/**
 * 将稠密矩阵转换为ELL格式
 */
int convert_to_ell_format(
    const std::vector<std::complex<double>>& dense_matrix,
    int cutoff,
    int max_nnz_per_row,
    double threshold,
    std::vector<std::complex<double>>& ell_values,
    std::vector<int>& ell_indices
);

/**
 * 应用挤压门到状态向量 (CPU版本)
 */
void apply_squeezing_cpu(
    const std::vector<std::complex<double>>& state_in,
    std::vector<std::complex<double>>& state_out,
    const std::vector<std::complex<double>>& squeezing_matrix,
    int cutoff
);
