#include "fock_ell_operator.h"
#include <iostream>
#include <algorithm>
#include <cmath>

/**
 * FockELLOperator 构造函数
 */
FockELLOperator::FockELLOperator(int matrix_dim, int bandwidth)
    : dim(matrix_dim), max_bandwidth(bandwidth) {

    if (dim <= 0 || max_bandwidth <= 0) {
        throw std::invalid_argument("矩阵维度和带宽必须为正数");
    }

    // 初始化主机端存储
    size_t total_elements = static_cast<size_t>(dim) * max_bandwidth;
    host_ell_val.assign(total_elements, make_cuDoubleComplex(0.0, 0.0));
    host_ell_col.assign(total_elements, -1);

    // 分配GPU内存
    realloc_gpu_memory();
    
    // 上传初始数据（零矩阵）到GPU
    upload_to_gpu();

    std::cout << "FockELLOperator 初始化完成: 维度=" << dim
              << ", 带宽=" << max_bandwidth << std::endl;
}

/**
 * FockELLOperator 析构函数
 */
FockELLOperator::~FockELLOperator() {
    if (ell_val) {
        cudaFree(ell_val);
        ell_val = nullptr;
    }
    if (ell_col) {
        cudaFree(ell_col);
        ell_col = nullptr;
    }

    std::cout << "FockELLOperator 销毁完成" << std::endl;
}

/**
 * 从稠密矩阵构建ELL格式
 */
void FockELLOperator::build_from_dense(const std::vector<cuDoubleComplex>& dense_matrix, double tolerance) {
    if (dense_matrix.size() != static_cast<size_t>(dim) * dim) {
        throw std::invalid_argument("稠密矩阵大小不匹配");
    }

    // 清空当前数据
    std::fill(host_ell_val.begin(), host_ell_val.end(), make_cuDoubleComplex(0.0, 0.0));
    std::fill(host_ell_col.begin(), host_ell_col.end(), -1);

    // 为每一行找到非零元素
    for (int row = 0; row < dim; ++row) {
        std::vector<std::pair<int, cuDoubleComplex>> row_elements;

        // 收集该行的非零元素
        for (int col = 0; col < dim; ++col) {
            cuDoubleComplex val = dense_matrix[row * dim + col];
            double magnitude = sqrt(val.x * val.x + val.y * val.y);

            if (magnitude > tolerance) {
                row_elements.emplace_back(col, val);
            }
        }

        // 如果非零元素超过最大带宽，保留最大的那些
        if (row_elements.size() > static_cast<size_t>(max_bandwidth)) {
            std::sort(row_elements.begin(), row_elements.end(),
                     [tolerance](const auto& a, const auto& b) {
                         double mag_a = sqrt(a.second.x * a.second.x + a.second.y * a.second.y);
                         double mag_b = sqrt(b.second.x * b.second.x + b.second.y * b.second.y);
                         return mag_a > mag_b;
                     });
            row_elements.resize(max_bandwidth);
        }

        // 存储到ELL格式
        for (size_t k = 0; k < row_elements.size(); ++k) {
            size_t idx = row * max_bandwidth + k;
            host_ell_col[idx] = row_elements[k].first;
            host_ell_val[idx] = row_elements[k].second;
        }
    }

    // 上传到GPU
    upload_to_gpu();
}

/**
 * 从带状矩阵构建ELL格式 (更高效)
 */
void FockELLOperator::build_from_diagonals(const std::vector<std::vector<cuDoubleComplex>>& diagonals,
                                          const std::vector<int>& offsets) {
    if (diagonals.size() != offsets.size()) {
        throw std::invalid_argument("对角线数量和偏移量数量不匹配");
    }

    // 清空当前数据
    std::fill(host_ell_val.begin(), host_ell_val.end(), make_cuDoubleComplex(0.0, 0.0));
    std::fill(host_ell_col.begin(), host_ell_col.end(), -1);

    // 为每一行构建ELL格式
    for (int row = 0; row < dim; ++row) {
        int col_idx = 0;

        // 遍历所有对角线
        for (size_t d = 0; d < diagonals.size() && col_idx < max_bandwidth; ++d) {
            int col = row + offsets[d];

            if (col >= 0 && col < dim) {
                // 找到对应的对角线元素
                int diag_idx = (offsets[d] >= 0) ? row : col;
                if (diag_idx >= 0 && diag_idx < static_cast<int>(diagonals[d].size())) {
                    cuDoubleComplex val = diagonals[d][diag_idx];

                    // 检查是否为非零元素
                    double magnitude = sqrt(val.x * val.x + val.y * val.y);
                    if (magnitude > 1e-12) {
                        host_ell_col[row * max_bandwidth + col_idx] = col;
                        host_ell_val[row * max_bandwidth + col_idx] = val;
                        col_idx++;
                    }
                }
            }
        }
    }

    // 上传到GPU
    upload_to_gpu();
}

/**
 * 获取矩阵元素 (主机端)
 */
cuDoubleComplex FockELLOperator::get_element(int row, int col) const {
    if (row < 0 || row >= dim || col < 0 || col >= dim) {
        return make_cuDoubleComplex(0.0, 0.0);
    }

    for (int k = 0; k < max_bandwidth; ++k) {
        size_t idx = row * max_bandwidth + k;
        if (host_ell_col[idx] == col) {
            return host_ell_val[idx];
        }
        if (host_ell_col[idx] == -1) {
            break; // 该行到此结束
        }
    }

    return make_cuDoubleComplex(0.0, 0.0);
}

/**
 * 设置矩阵元素 (主机端)
 */
void FockELLOperator::set_element(int row, int col, cuDoubleComplex value) {
    if (row < 0 || row >= dim || col < 0 || col >= dim) {
        return;
    }

    // 查找是否已存在
    for (int k = 0; k < max_bandwidth; ++k) {
        size_t idx = row * max_bandwidth + k;
        if (host_ell_col[idx] == col) {
            host_ell_val[idx] = value;
            return;
        }
        if (host_ell_col[idx] == -1) {
            // 找到空位置，插入新元素
            host_ell_col[idx] = col;
            host_ell_val[idx] = value;
            return;
        }
    }

    // 如果没有空间，替换最后一个元素
    std::cerr << "警告：行 " << row << " 的非零元素超过最大带宽，替换最后一个元素" << std::endl;
    size_t idx = row * max_bandwidth + max_bandwidth - 1;
    host_ell_col[idx] = col;
    host_ell_val[idx] = value;
}

/**
 * 上传数据到GPU
 */
void FockELLOperator::upload_to_gpu() {
    if (!ell_val || !ell_col) {
        realloc_gpu_memory();
    }

    cudaError_t err;

    err = cudaMemcpy(ell_val, host_ell_val.data(),
                     host_ell_val.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法上传ELL值到GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(ell_col, host_ell_col.data(),
                     host_ell_col.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法上传ELL列索引到GPU: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * 从GPU下载数据到主机
 */
void FockELLOperator::download_from_gpu() {
    if (!ell_val || !ell_col) {
        return;
    }

    cudaError_t err;

    err = cudaMemcpy(host_ell_val.data(), ell_val,
                     host_ell_val.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法从GPU下载ELL值: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(host_ell_col.data(), ell_col,
                     host_ell_col.size() * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法从GPU下载ELL列索引: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * 获取非零元素数量
 */
int FockELLOperator::get_nnz() const {
    int nnz = 0;
    for (int col : host_ell_col) {
        if (col != -1) {
            nnz++;
        }
    }
    return nnz;
}

/**
 * 重新分配GPU内存
 */
void FockELLOperator::realloc_gpu_memory() {
    // 同步所有GPU操作，确保在释放内存前所有操作完成
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess && sync_err != cudaErrorNotReady) {
        // 如果之前的操作有错误，记录但不抛出异常
        std::cerr << "警告：重新分配GPU内存前检测到GPU错误: " << cudaGetErrorString(sync_err) << std::endl;
    }

    // 释放现有内存
    if (ell_val) {
        cudaError_t err = cudaFree(ell_val);
        if (err != cudaSuccess) {
            std::cerr << "警告：释放ELL值内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        ell_val = nullptr;
    }
    if (ell_col) {
        cudaError_t err = cudaFree(ell_col);
        if (err != cudaSuccess) {
            std::cerr << "警告：释放ELL列索引内存失败: " << cudaGetErrorString(err) << std::endl;
        }
        ell_col = nullptr;
    }

    size_t val_size = static_cast<size_t>(dim) * max_bandwidth * sizeof(cuDoubleComplex);
    size_t col_size = static_cast<size_t>(dim) * max_bandwidth * sizeof(int);

    cudaError_t err;

    err = cudaMalloc(&ell_val, val_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("无法分配GPU内存用于ELL值: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&ell_col, col_size);
    if (err != cudaSuccess) {
        cudaFree(ell_val);
        ell_val = nullptr;
        throw std::runtime_error("无法分配GPU内存用于ELL列索引: " + std::string(cudaGetErrorString(err)));
    }
}
