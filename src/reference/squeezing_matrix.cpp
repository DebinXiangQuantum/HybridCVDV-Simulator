/**
 * Squeezing Gate Matrix Generation
 * 使用递推关系计算挤压门矩阵元素
 * 
 * 参考: Strawberry Fields implementation
 */

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

/**
 * 计算挤压门矩阵 S(r, theta)
 * 
 * 使用递推关系，利用奇偶性守恒特性
 * 
 * @param r 挤压幅度
 * @param theta 挤压角度
 * @param cutoff Fock空间截断维度
 * @return 挤压门矩阵 (cutoff x cutoff)
 */
std::vector<std::complex<double>> generate_squeezing_matrix(double r, double theta, int cutoff) {
    std::vector<std::complex<double>> S(cutoff * cutoff, std::complex<double>(0.0, 0.0));
    
    // 预计算平方根
    std::vector<double> sqrt_n(cutoff);
    for (int i = 0; i < cutoff; ++i) {
        sqrt_n[i] = std::sqrt(static_cast<double>(i));
    }
    
    // 计算参数
    std::complex<double> eitheta_tanhr = std::exp(std::complex<double>(0.0, theta)) * std::tanh(r);
    double sechr = 1.0 / std::cosh(r);
    
    // 递推矩阵 R
    std::complex<double> R[2][2];
    R[0][0] = -eitheta_tanhr;
    R[0][1] = sechr;
    R[1][0] = sechr;
    R[1][1] = std::conj(eitheta_tanhr);
    
    // 初始化 S[0,0]
    S[0 * cutoff + 0] = std::sqrt(sechr);
    
    // 递推计算第一列 (偶数行)
    for (int m = 2; m < cutoff; m += 2) {
        S[m * cutoff + 0] = sqrt_n[m - 1] / sqrt_n[m] * R[0][0] * S[(m - 2) * cutoff + 0];
    }
    
    // 递推计算所有其他元素
    for (int m = 0; m < cutoff; ++m) {
        for (int n = 1; n < cutoff; ++n) {
            // 奇偶性守恒: 只有 (m + n) 为偶数时才有非零元素
            if ((m + n) % 2 == 0) {
                std::complex<double> term1(0.0, 0.0);
                std::complex<double> term2(0.0, 0.0);
                
                // 第一项: sqrt[n-1]/sqrt[n] * R[1,1] * S[m, n-2]
                if (n >= 2) {
                    term1 = sqrt_n[n - 1] / sqrt_n[n] * R[1][1] * S[m * cutoff + (n - 2)];
                }
                
                // 第二项: sqrt[m]/sqrt[n] * R[0,1] * S[m-1, n-1]
                if (m >= 1 && n >= 1) {
                    term2 = sqrt_n[m] / sqrt_n[n] * R[0][1] * S[(m - 1) * cutoff + (n - 1)];
                }
                
                S[m * cutoff + n] = term1 + term2;
            }
        }
    }
    
    return S;
}

/**
 * 将稠密矩阵转换为ELL格式
 * 
 * @param dense_matrix 稠密矩阵 (row-major, size = cutoff * cutoff)
 * @param cutoff 矩阵维度
 * @param max_nnz_per_row 每行最大非零元素数
 * @param threshold 阈值，小于此值的元素视为零
 * @param ell_values 输出: ELL格式的值数组
 * @param ell_indices 输出: ELL格式的列索引数组
 * @return 实际使用的每行非零元素数
 */
int convert_to_ell_format(
    const std::vector<std::complex<double>>& dense_matrix,
    int cutoff,
    int max_nnz_per_row,
    double threshold,
    std::vector<std::complex<double>>& ell_values,
    std::vector<int>& ell_indices
) {
    ell_values.resize(cutoff * max_nnz_per_row, std::complex<double>(0.0, 0.0));
    ell_indices.resize(cutoff * max_nnz_per_row, -1);
    
    int actual_max_nnz = 0;
    
    for (int row = 0; row < cutoff; ++row) {
        // 收集该行的非零元素
        std::vector<std::pair<double, int>> nonzeros; // (magnitude, col_index)
        
        for (int col = 0; col < cutoff; ++col) {
            std::complex<double> val = dense_matrix[row * cutoff + col];
            double mag = std::abs(val);
            
            if (mag > threshold) {
                nonzeros.push_back({mag, col});
            }
        }
        
        // 按幅度排序，保留最大的 max_nnz_per_row 个
        std::sort(nonzeros.begin(), nonzeros.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        int nnz_count = std::min(static_cast<int>(nonzeros.size()), max_nnz_per_row);
        actual_max_nnz = std::max(actual_max_nnz, nnz_count);
        
        // 按列索引排序（ELL格式要求）
        std::sort(nonzeros.begin(), nonzeros.begin() + nnz_count,
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // 填充ELL数组
        for (int k = 0; k < nnz_count; ++k) {
            int col = nonzeros[k].second;
            ell_values[row * max_nnz_per_row + k] = dense_matrix[row * cutoff + col];
            ell_indices[row * max_nnz_per_row + k] = col;
        }
    }
    
    return actual_max_nnz;
}

/**
 * 应用挤压门到状态向量
 * 
 * @param state_in 输入状态向量
 * @param state_out 输出状态向量
 * @param squeezing_matrix 挤压门矩阵
 * @param cutoff 维度
 */
void apply_squeezing_cpu(
    const std::vector<std::complex<double>>& state_in,
    std::vector<std::complex<double>>& state_out,
    const std::vector<std::complex<double>>& squeezing_matrix,
    int cutoff
) {
    state_out.resize(cutoff, std::complex<double>(0.0, 0.0));
    
    for (int m = 0; m < cutoff; ++m) {
        std::complex<double> sum(0.0, 0.0);
        for (int n = 0; n < cutoff; ++n) {
            sum += squeezing_matrix[m * cutoff + n] * state_in[n];
        }
        state_out[m] = sum;
    }
}
