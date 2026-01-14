#include "operators.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <regex>
#include <algorithm>

// 使用gpu命名空间
using namespace gpu;

// CUDA错误检查宏
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << " " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CUBLAS错误检查宏
#define CUBLAS_CHECK(err) { \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << " " << __LINE__ << ": " << err << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA核函数定义
__global__ void conjugate_transpose_kernel(Complex* in_values, Complex* out_values, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        out_values[idx] = Complex(in_values[idx].real, -in_values[idx].imag);
    }
}

__global__ void scale_kernel(Complex* values, Complex scalar, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        double real = values[idx].real * scalar.real - values[idx].imag * scalar.imag;
        double imag = values[idx].real * scalar.imag + values[idx].imag * scalar.real;
        values[idx].real = real;
        values[idx].imag = imag;
    }
}

// CUDASparseMatrix 实现
CUDASparseMatrix::CUDASparseMatrix() : rows_(0), cols_(0) {
    device_data_.values = nullptr;
    device_data_.row_indices = nullptr;
    device_data_.col_indices = nullptr;
    device_data_.nnz = 0;
}

CUDASparseMatrix::CUDASparseMatrix(int rows, int cols) : rows_(rows), cols_(cols) {
    device_data_.values = nullptr;
    device_data_.row_indices = nullptr;
    device_data_.col_indices = nullptr;
    device_data_.nnz = 0;
}

CUDASparseMatrix::~CUDASparseMatrix() {
    if (device_data_.values != nullptr) {
        CUDA_CHECK(cudaFree(device_data_.values));
    }
    if (device_data_.row_indices != nullptr) {
        CUDA_CHECK(cudaFree(device_data_.row_indices));
    }
    if (device_data_.col_indices != nullptr) {
        CUDA_CHECK(cudaFree(device_data_.col_indices));
    }
}

void CUDASparseMatrix::set(int row, int col, Complex value) {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    
    if (value != Complex(0)) {
        temp_values_.push_back(value);
        temp_row_indices_.push_back(row);
        temp_col_indices_.push_back(col);
    }
}

Complex CUDASparseMatrix::get(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    
    // 为了获取单个元素，我们需要下载整个矩阵到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    non_const_this->downloadFromDevice();
    
    // 线性搜索查找元素
    for (int i = 0; i < temp_values_.size(); ++i) {
        if (temp_row_indices_[i] == row && temp_col_indices_[i] == col) {
            return temp_values_[i];
        }
    }
    
    return Complex(0);
}

void CUDASparseMatrix::uploadToDevice() {
    // 释放旧的设备内存
    if (device_data_.values != nullptr) {
        CUDA_CHECK(cudaFree(device_data_.values));
    }
    if (device_data_.row_indices != nullptr) {
        CUDA_CHECK(cudaFree(device_data_.row_indices));
    }
    if (device_data_.col_indices != nullptr) {
        CUDA_CHECK(cudaFree(device_data_.col_indices));
    }
    
    // 更新非零元素数量
    device_data_.nnz = temp_values_.size();
    
    // 分配新的设备内存
    if (device_data_.nnz > 0) {
        CUDA_CHECK(cudaMalloc(&device_data_.values, device_data_.nnz * sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&device_data_.row_indices, device_data_.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device_data_.col_indices, device_data_.nnz * sizeof(int)));
        
        // 复制数据到设备
        CUDA_CHECK(cudaMemcpy(device_data_.values, temp_values_.data(), device_data_.nnz * sizeof(Complex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_data_.row_indices, temp_row_indices_.data(), device_data_.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_data_.col_indices, temp_col_indices_.data(), device_data_.nnz * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void CUDASparseMatrix::downloadFromDevice() const {
    if (device_data_.nnz > 0) {
        CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
        
        non_const_this->temp_values_.resize(device_data_.nnz);
        non_const_this->temp_row_indices_.resize(device_data_.nnz);
        non_const_this->temp_col_indices_.resize(device_data_.nnz);
        
        CUDA_CHECK(cudaMemcpy(non_const_this->temp_values_.data(), device_data_.values, device_data_.nnz * sizeof(Complex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(non_const_this->temp_row_indices_.data(), device_data_.row_indices, device_data_.nnz * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(non_const_this->temp_col_indices_.data(), device_data_.col_indices, device_data_.nnz * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

CUDASparseMatrix CUDASparseMatrix::conjugateTranspose() const {
    CUDASparseMatrix result(cols_, rows_);
    
    // 下载数据到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    non_const_this->downloadFromDevice();
    
    // 在CPU上构建转置矩阵
    for (int i = 0; i < temp_values_.size(); ++i) {
        // 将值取共轭
        Complex value = conj(temp_values_[i]);
        result.set(temp_col_indices_[i], temp_row_indices_[i], value);
    }
    
    // 上传到设备
    result.uploadToDevice();
    
    return result;
}



CUDASparseMatrix CUDASparseMatrix::multiply(const CUDASparseMatrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    
    // 对于稀疏矩阵乘法，我们暂时在CPU上实现，因为CUDA的稀疏矩阵乘法需要更复杂的库支持
    // 下载两个矩阵的数据到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    CUDASparseMatrix* non_const_other = const_cast<CUDASparseMatrix*>(&other);
    
    non_const_this->downloadFromDevice();
    non_const_other->downloadFromDevice();
    
    // 在CPU上执行稀疏矩阵乘法
    CUDASparseMatrix result(rows_, other.cols_);
    
    // 为了提高效率，我们可以将other矩阵转换为按列存储的结构
    std::map<int, std::map<int, Complex>> other_col_data;
    for (int i = 0; i < non_const_other->temp_values_.size(); ++i) {
        int row = non_const_other->temp_row_indices_[i];
        int col = non_const_other->temp_col_indices_[i];
        Complex value = non_const_other->temp_values_[i];
        other_col_data[col][row] = value;
    }
    
    // 执行矩阵乘法
    for (int i = 0; i < temp_values_.size(); ++i) {
        int row1 = temp_row_indices_[i];
        int col1 = temp_col_indices_[i];
        Complex val1 = temp_values_[i];
        
        // 检查other矩阵中是否有col1列
        auto col_it = other_col_data.find(col1);
        if (col_it != other_col_data.end()) {
            // 遍历other矩阵中col1列的所有元素
            for (const auto& entry : col_it->second) {
                int row2 = entry.first;
                Complex val2 = entry.second;
                
                // 更新结果矩阵
                Complex current = result.get(row1, row2);
                result.set(row1, row2, current + val1 * val2);
            }
        }
    }
    
    // 上传结果到设备
    result.uploadToDevice();
    
    return result;
}

CUDASparseMatrix CUDASparseMatrix::add(const CUDASparseMatrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }
    
    // 下载两个矩阵的数据到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    CUDASparseMatrix* non_const_other = const_cast<CUDASparseMatrix*>(&other);
    
    non_const_this->downloadFromDevice();
    non_const_other->downloadFromDevice();
    
    // 在CPU上执行稀疏矩阵加法
    CUDASparseMatrix result(rows_, cols_);
    
    // 复制当前矩阵的数据
    for (int i = 0; i < temp_values_.size(); ++i) {
        result.set(temp_row_indices_[i], temp_col_indices_[i], temp_values_[i]);
    }
    
    // 添加other矩阵的数据
    for (int i = 0; i < non_const_other->temp_values_.size(); ++i) {
        int row = non_const_other->temp_row_indices_[i];
        int col = non_const_other->temp_col_indices_[i];
        Complex value = non_const_other->temp_values_[i];
        
        Complex current = result.get(row, col);
        result.set(row, col, current + value);
    }
    
    // 上传结果到设备
    result.uploadToDevice();
    
    return result;
}

CUDASparseMatrix CUDASparseMatrix::subtract(const CUDASparseMatrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
    }
    
    // 下载两个矩阵的数据到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    CUDASparseMatrix* non_const_other = const_cast<CUDASparseMatrix*>(&other);
    
    non_const_this->downloadFromDevice();
    non_const_other->downloadFromDevice();
    
    // 在CPU上执行稀疏矩阵减法
    CUDASparseMatrix result(rows_, cols_);
    
    // 复制当前矩阵的数据
    for (int i = 0; i < temp_values_.size(); ++i) {
        result.set(temp_row_indices_[i], temp_col_indices_[i], temp_values_[i]);
    }
    
    // 减去other矩阵的数据
    for (int i = 0; i < non_const_other->temp_values_.size(); ++i) {
        int row = non_const_other->temp_row_indices_[i];
        int col = non_const_other->temp_col_indices_[i];
        Complex value = non_const_other->temp_values_[i];
        
        Complex current = result.get(row, col);
        result.set(row, col, current - value);
    }
    
    // 上传结果到设备
    result.uploadToDevice();
    
    return result;
}



CUDASparseMatrix CUDASparseMatrix::scale(Complex scalar) const {
    CUDASparseMatrix result(rows_, cols_);
    
    // 如果标量为0，返回空矩阵
    if (scalar == Complex(0)) {
        return result;
    }
    
    // 下载数据到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    non_const_this->downloadFromDevice();
    
    // 在CPU上缩放矩阵
    for (int i = 0; i < temp_values_.size(); ++i) {
        result.set(temp_row_indices_[i], temp_col_indices_[i], temp_values_[i] * scalar);
    }
    
    // 上传结果到设备
    result.uploadToDevice();
    
    return result;
}

CUDASparseMatrix CUDASparseMatrix::exp() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square to compute exponential");
    }
    
    // 下载矩阵到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    non_const_this->downloadFromDevice();
    
    // 转换为密集矩阵
    auto dense = toDense();
    
    // 创建结果矩阵（初始化为单位矩阵）
    std::vector<std::vector<Complex>> result(rows_, std::vector<Complex>(cols_, Complex(0)));
    for (int i = 0; i < rows_; ++i) {
        result[i][i] = Complex(1);
    }
    
    // 创建当前项矩阵（初始化为输入矩阵）
    std::vector<std::vector<Complex>> current = dense;
    
    // 项的系数（初始化为1）
    Complex coeff = Complex(1);
    
    // 计算前20项的泰勒展开
    for (int n = 1; n <= 20; ++n) {
        // 更新系数
        coeff /= Complex(n);
        
        // 将current缩放并加到result
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[i][j] += coeff * current[i][j];
            }
        }
        
        // 计算下一项的矩阵（current = current * dense）
        std::vector<std::vector<Complex>> next(rows_, std::vector<Complex>(cols_, Complex(0)));
        for (int i = 0; i < rows_; ++i) {
            for (int k = 0; k < cols_; ++k) {
                if (current[i][k] != Complex(0)) {
                    for (int j = 0; j < cols_; ++j) {
                        if (dense[k][j] != Complex(0)) {
                            next[i][j] += current[i][k] * dense[k][j];
                        }
                    }
                }
            }
        }
        
        current = next;
    }
    
    // 转换回稀疏矩阵
    CUDASparseMatrix exp_result(rows_, cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            if (result[i][j] != Complex(0)) {
                exp_result.set(i, j, result[i][j]);
            }
        }
    }
    
    // 上传结果到设备
    exp_result.uploadToDevice();
    
    return exp_result;
}

CUDASparseMatrix CUDASparseMatrix::kroneckerProduct(const CUDASparseMatrix& other) const {
    int new_rows = rows_ * other.rows_;
    int new_cols = cols_ * other.cols_;
    
    CUDASparseMatrix result(new_rows, new_cols);
    
    // 下载两个矩阵的数据到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    CUDASparseMatrix* non_const_other = const_cast<CUDASparseMatrix*>(&other);
    
    non_const_this->downloadFromDevice();
    non_const_other->downloadFromDevice();
    
    // 在CPU上计算克罗内克积
    for (int i = 0; i < temp_values_.size(); ++i) {
        int row1 = temp_row_indices_[i];
        int col1 = temp_col_indices_[i];
        Complex val1 = temp_values_[i];
        
        for (int j = 0; j < non_const_other->temp_values_.size(); ++j) {
            int row2 = non_const_other->temp_row_indices_[j];
            int col2 = non_const_other->temp_col_indices_[j];
            Complex val2 = non_const_other->temp_values_[j];
            
            int new_row = row1 * other.rows_ + row2;
            int new_col = col1 * other.cols_ + col2;
            
            result.set(new_row, new_col, val1 * val2);
        }
    }
    
    // 上传结果到设备
    result.uploadToDevice();
    
    return result;
}

std::vector<std::vector<Complex>> CUDASparseMatrix::toDense() const {
    std::vector<std::vector<Complex>> dense(rows_, std::vector<Complex>(cols_, Complex(0)));
    
    // 下载数据到CPU
    CUDASparseMatrix* non_const_this = const_cast<CUDASparseMatrix*>(this);
    non_const_this->downloadFromDevice();
    
    for (int i = 0; i < temp_values_.size(); ++i) {
        dense[temp_row_indices_[i]][temp_col_indices_[i]] = temp_values_[i];
    }
    
    return dense;
}

// QubitGatesGPU 实现
CUDASparseMatrix QubitGatesGPU::I() {
    CUDASparseMatrix mat(2, 2);
    mat.set(0, 0, Complex(1, 0));
    mat.set(1, 1, Complex(1, 0));
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix QubitGatesGPU::X() {
    CUDASparseMatrix mat(2, 2);
    mat.set(0, 1, Complex(1, 0));
    mat.set(1, 0, Complex(1, 0));
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix QubitGatesGPU::Y() {
    CUDASparseMatrix mat(2, 2);
    mat.set(0, 1, Complex(0, -1));
    mat.set(1, 0, Complex(0, 1));
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix QubitGatesGPU::Z() {
    CUDASparseMatrix mat(2, 2);
    mat.set(0, 0, Complex(1, 0));
    mat.set(1, 1, Complex(-1, 0));
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix QubitGatesGPU::SPLUS() {
    CUDASparseMatrix x = X();
    CUDASparseMatrix y = Y();
    CUDASparseMatrix result = x.add(y.scale(Complex(0, 1))).scale(Complex(0.5, 0));
    return result;
}

CUDASparseMatrix QubitGatesGPU::SMINUS() {
    CUDASparseMatrix x = X();
    CUDASparseMatrix y = Y();
    CUDASparseMatrix result = x.subtract(y.scale(Complex(0, 1))).scale(Complex(0.5, 0));
    return result;
}

CUDASparseMatrix QubitGatesGPU::P0() {
    CUDASparseMatrix i = I();
    CUDASparseMatrix z = Z();
    CUDASparseMatrix result = i.add(z).scale(Complex(0.5, 0));
    return result;
}

CUDASparseMatrix QubitGatesGPU::P1() {
    CUDASparseMatrix i = I();
    CUDASparseMatrix z = Z();
    CUDASparseMatrix result = i.subtract(z).scale(Complex(0.5, 0));
    return result;
}

// CVOperatorsGPU 实现

// 解析操作符表达式
void CVOperatorsGPU::parseOpExpr(const std::string& expr, std::map<int, std::string>& ops) const {
    std::regex pattern(R"((\w+)(\d+))");
    std::smatch match;
    
    std::string::const_iterator searchStart(expr.cbegin());
    while (std::regex_search(searchStart, expr.cend(), match, pattern)) {
        std::string op = match[1];
        int qumode_idx = std::stoi(match[2]);
        ops[qumode_idx] = op;
        searchStart = match.suffix().first;
    }
}

CUDASparseMatrix CVOperatorsGPU::getEye(int cutoff) const {
    CUDASparseMatrix mat(cutoff, cutoff);
    for (int i = 0; i < cutoff; ++i) {
        mat.set(i, i, Complex(1, 0));
    }
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix CVOperatorsGPU::getA(int cutoff) const {
    CUDASparseMatrix mat(cutoff, cutoff);
    for (int i = 0; i < cutoff - 1; ++i) {
        double sqrt_i_plus_1 = std::sqrt(i + 1);
        mat.set(i, i + 1, Complex(sqrt_i_plus_1, 0));
    }
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix CVOperatorsGPU::getAd(int cutoff) const {
    return getA(cutoff).conjugateTranspose();
}

CUDASparseMatrix CVOperatorsGPU::getN(int cutoff) const {
    CUDASparseMatrix a = getA(cutoff);
    CUDASparseMatrix ad = getAd(cutoff);
    return ad.multiply(a);
}

CUDASparseMatrix CVOperatorsGPU::getProjector(int n, int cutoff) const {
    CUDASparseMatrix mat(cutoff, cutoff);
    mat.set(n, n, Complex(1, 0));
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix CVOperatorsGPU::getOp(const std::string& expr, const std::vector<int>& cutoffs) const {
    std::map<int, std::string> ops;
    parseOpExpr(expr, ops);
    
    if (ops.empty()) {
        throw std::invalid_argument("No operators found in expression");
    }
    
    // 找到最大的qumode索引
    int max_qumode = 0;
    for (const auto& op : ops) {
        if (op.first > max_qumode) {
            max_qumode = op.first;
        }
    }
    
    if (max_qumode >= cutoffs.size()) {
        throw std::invalid_argument("Qumode index out of bounds: " + std::to_string(max_qumode));
    }
    
    // 直接计算张量积，避免使用vector
    CUDASparseMatrix result;
    bool first_matrix = true;
    
    for (int i = 0; i <= max_qumode; ++i) {
        int cutoff = cutoffs[i];
        CUDASparseMatrix current_matrix;
        
        auto it = ops.find(i);
        if (it != ops.end()) {
            const std::string& op = it->second;
            if (op == "a") {
                current_matrix = getA(cutoff);
            } else if (op == "ad") {
                current_matrix = getAd(cutoff);
            } else if (op == "n") {
                current_matrix = getN(cutoff);
            } else {
                throw std::invalid_argument("Unrecognized operator: " + op);
            }
        } else {
            current_matrix = getEye(cutoff);
        }
        
        if (first_matrix) {
            result = current_matrix;
            first_matrix = false;
        } else {
            result = result.kroneckerProduct(current_matrix);
        }
    }
    
    return result;
}

CUDASparseMatrix CVOperatorsGPU::r(double theta, int cutoff) const {
    CUDASparseMatrix arg = getN(cutoff).scale(Complex(0, 1) * Complex(theta));
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::d(Complex alpha, int cutoff) const {
    CUDASparseMatrix ad = getAd(cutoff);
    CUDASparseMatrix a = getA(cutoff);
    
    CUDASparseMatrix arg = ad.scale(alpha).subtract(a.scale(conj(alpha)));
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::s(Complex theta, int cutoff) const {
    CUDASparseMatrix ad = getAd(cutoff);
    CUDASparseMatrix ad2 = ad.multiply(ad);
    
    CUDASparseMatrix arg = ad2.scale(conj(theta) * Complex(0.5));
    CUDASparseMatrix hc = arg.conjugateTranspose();
    
    return arg.subtract(hc).exp();
}

CUDASparseMatrix CVOperatorsGPU::s2(Complex theta, int cutoff_a, int cutoff_b) const {
    double r = abs(theta);
    double phi = arg(theta);
    
    std::vector<int> cutoffs = {cutoff_a, cutoff_b};
    CUDASparseMatrix op = getOp("ad0 ad1", cutoffs);
    CUDASparseMatrix arg = op.scale(r * exp(Complex(0, 1) * phi));
    CUDASparseMatrix hc = arg.conjugateTranspose();
    
    return arg.subtract(hc).exp();
}

CUDASparseMatrix CVOperatorsGPU::s3(Complex theta, int cutoff_a, int cutoff_b, int cutoff_c) const {
    double r = abs(theta);
    double phi = arg(theta);
    
    std::vector<int> cutoffs = {cutoff_a, cutoff_b, cutoff_c};
    CUDASparseMatrix op = getOp("ad0 ad1 ad2", cutoffs);
    CUDASparseMatrix arg = op.scale(r * exp(Complex(0, 1) * phi));
    CUDASparseMatrix hc = arg.conjugateTranspose();
    
    return arg.subtract(hc).exp();
}

CUDASparseMatrix CVOperatorsGPU::bs(Complex theta, int cutoff_a, int cutoff_b) const {
    std::vector<int> cutoffs = {cutoff_a, cutoff_b};
    CUDASparseMatrix op = getOp("ad0 a1", cutoffs);
    CUDASparseMatrix arg = op.scale(theta);
    CUDASparseMatrix hc = arg.conjugateTranspose();
    
    return arg.subtract(hc).exp();
}

CUDASparseMatrix CVOperatorsGPU::cr(double theta, int cutoff) const {
    CUDASparseMatrix z = QubitGatesGPU::Z();
    CUDASparseMatrix n = getN(cutoff);
    CUDASparseMatrix arg = z.kroneckerProduct(n).scale(Complex(0, 1) * theta);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::crx(double theta, int cutoff) const {
    CUDASparseMatrix x = QubitGatesGPU::X();
    CUDASparseMatrix n = getN(cutoff);
    CUDASparseMatrix arg = x.kroneckerProduct(n).scale(Complex(0, 1) * theta);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::cry(double theta, int cutoff) const {
    CUDASparseMatrix y = QubitGatesGPU::Y();
    CUDASparseMatrix n = getN(cutoff);
    CUDASparseMatrix arg = y.kroneckerProduct(n).scale(Complex(0, 1) * theta);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::cd(Complex alpha, Complex* beta, int cutoff) const {
    CUDASparseMatrix displace0 = d(alpha, cutoff);
    Complex beta_val = (beta != nullptr) ? *beta : -alpha;
    CUDASparseMatrix displace1 = d(beta_val, cutoff);
    
    CUDASparseMatrix p0 = QubitGatesGPU::P0();
    CUDASparseMatrix p1 = QubitGatesGPU::P1();
    
    return p0.kroneckerProduct(displace0).add(p1.kroneckerProduct(displace1));
}

CUDASparseMatrix CVOperatorsGPU::ecd(Complex theta, int cutoff) const {
    Complex beta = -theta;
    return cd(theta, &beta, cutoff);
}

CUDASparseMatrix CVOperatorsGPU::cbs(Complex theta, int cutoff_a, int cutoff_b) const {
    CUDASparseMatrix arg = bs(theta, cutoff_a, cutoff_b);
    CUDASparseMatrix p0 = QubitGatesGPU::P0();
    CUDASparseMatrix p1 = QubitGatesGPU::P1();
    
    return p0.kroneckerProduct(arg).add(p1.kroneckerProduct(arg.conjugateTranspose()));
}

CUDASparseMatrix CVOperatorsGPU::cschwinger(
    double beta, double theta_1, double phi_1,
    double theta_2, double phi_2,
    int cutoff_a, int cutoff_b) const {
    std::vector<int> cutoffs = {cutoff_a, cutoff_b};
    
    // 计算Sx
    CUDASparseMatrix sx = getOp("a0 ad1", cutoffs);
    sx = sx.add(sx.conjugateTranspose()).scale(Complex(0.5, 0));
    
    // 计算Sy
    CUDASparseMatrix sy = getOp("a0 ad1", cutoffs);
    sy = sy.subtract(sy.conjugateTranspose()).scale(Complex(0, -0.5));
    
    // 计算Sz
    CUDASparseMatrix sz = getOp("n1", cutoffs).subtract(getOp("n0", cutoffs)).scale(Complex(0.5, 0));
    
    // 计算sigma
    CUDASparseMatrix sigma = QubitGatesGPU::X().scale(std::sin(theta_1) * std::cos(phi_1))
                            .add(QubitGatesGPU::Y().scale(std::sin(theta_1) * std::sin(phi_1)))
                            .add(QubitGatesGPU::Z().scale(std::cos(theta_1)));
    
    // 计算S
    CUDASparseMatrix s = sx.scale(std::sin(theta_2) * std::cos(phi_2))
                        .add(sy.scale(std::sin(theta_2) * std::sin(phi_2)))
                        .add(sz.scale(std::cos(theta_2)));
    
    // 计算指数
    CUDASparseMatrix arg = sigma.kroneckerProduct(s).scale(Complex(0, -1) * beta);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::snap(double theta, int n, int cutoff) const {
    std::vector<double> params = {theta, static_cast<double>(n)};
    return multisnap(params, cutoff);
}

CUDASparseMatrix CVOperatorsGPU::csnap(double theta, int n, int cutoff) const {
    std::vector<double> params = {theta, static_cast<double>(n)};
    return multicsnap(params, cutoff);
}

CUDASparseMatrix CVOperatorsGPU::multisnap(const std::vector<double>& params, int cutoff) const {
    if (params.size() % 2 != 0) {
        throw std::invalid_argument("Params must have even number of elements");
    }
    
    int num_pairs = params.size() / 2;
    std::vector<double> thetas(params.begin(), params.begin() + num_pairs);
    std::vector<int> ns;
    for (int i = num_pairs; i < params.size(); ++i) {
        ns.push_back(static_cast<int>(params[i]));
    }
    
    CUDASparseMatrix mat(cutoff, cutoff);
    for (int i = 0; i < cutoff; ++i) {
        Complex value(1, 0);
        for (int j = 0; j < num_pairs; ++j) {
            if (i == ns[j]) {
                value = exp(Complex(0, 1) * thetas[j]);
                break;
            }
        }
        mat.set(i, i, value);
    }
    
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix CVOperatorsGPU::multicsnap(const std::vector<double>& params, int cutoff) const {
    if (params.size() % 2 != 0) {
        throw std::invalid_argument("Params must have even number of elements");
    }
    
    int num_pairs = params.size() / 2;
    std::vector<double> thetas(params.begin(), params.begin() + num_pairs);
    std::vector<int> ns;
    for (int i = num_pairs; i < params.size(); ++i) {
        ns.push_back(static_cast<int>(params[i]));
    }
    
    CUDASparseMatrix mat(2 * cutoff, 2 * cutoff);
    for (int i = 0; i < cutoff; ++i) {
        Complex value(1, 0);
        for (int j = 0; j < num_pairs; ++j) {
            if (i == ns[j]) {
                value = exp(Complex(0, 1) * thetas[j]);
                break;
            }
        }
        mat.set(i, i, value);
        mat.set(i + cutoff, i + cutoff, conj(value));
    }
    
    mat.uploadToDevice();
    return mat;
}

CUDASparseMatrix CVOperatorsGPU::sqr(const std::vector<double>& params, int cutoff) const {
    if (params.size() % 3 != 0) {
        throw std::invalid_argument("Params must have multiple of 3 elements");
    }
    
    int num_blocks = params.size() / 3;
    
    // 存储需要修改的块的索引和对应的矩阵
    std::map<int, CUDASparseMatrix> modified_blocks;
    
    for (int i = 0; i < num_blocks; ++i) {
        double theta = params[3 * i];
        double phi = params[3 * i + 1];
        int n = static_cast<int>(params[3 * i + 2]);
        
        if (n < 0 || n >= cutoff) {
            throw std::invalid_argument("Fock state out of bounds");
        }
        
        // 创建R门矩阵
        CUDASparseMatrix r_gate(2, 2);
        r_gate.set(0, 0, exp(Complex(0, 1) * phi / 2.0));  // 将2改为2.0
        r_gate.set(0, 1, Complex(0, 0));
        r_gate.set(1, 0, Complex(0, 0));
        r_gate.set(1, 1, exp(Complex(0, -1) * phi / 2.0));  // 将2改为2.0
        r_gate.uploadToDevice();
        
        modified_blocks[n] = r_gate;
    }
    
    // 创建块对角矩阵
    int total_size = 2 * cutoff;
    CUDASparseMatrix out(total_size, total_size);
    
    for (int i = 0; i < cutoff; ++i) {
        // 获取当前块矩阵
        CUDASparseMatrix block;
        auto it = modified_blocks.find(i);
        if (it != modified_blocks.end()) {
            block = it->second;
        } else {
            block = QubitGatesGPU::I();
        }
        
        // 下载当前块的数据
        CUDASparseMatrix* non_const_block = const_cast<CUDASparseMatrix*>(&block);
        non_const_block->downloadFromDevice();
        
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                Complex value = non_const_block->get(j, k);
                if (value != Complex(0, 0)) {
                    out.set(2 * i + j, 2 * i + k, value);
                }
            }
        }
    }
    
    // 上传out矩阵到设备
    out.uploadToDevice();
    
    // 重新排列以作用于Qubit x Qumode
    CUDASparseMatrix result(total_size, total_size);
    
    // 下载out矩阵的数据
    CUDASparseMatrix* non_const_out = const_cast<CUDASparseMatrix*>(&out);
    non_const_out->downloadFromDevice();
    
    for (int i = 0; i < cutoff; ++i) {
        for (int j = 0; j < 2; ++j) {
            int old_index = 2 * i + j;
            int new_index = j * cutoff + i;
            
            for (int k = 0; k < total_size; ++k) {
                Complex value = non_const_out->get(old_index, k);
                if (value != Complex(0, 0)) {
                    // 计算k的新位置
                    int k_qubit = k / cutoff;
                    int k_qumode = k % cutoff;
                    int new_k = k_qumode * 2 + k_qubit;
                    
                    result.set(new_index, new_k, value);
                }
            }
        }
    }
    
    // 上传结果到设备
    result.uploadToDevice();
    
    return result;
}

CUDASparseMatrix CVOperatorsGPU::pnr(int max, int cutoff) const {
    CUDASparseMatrix projector(cutoff, cutoff);
    
    // 创建投影器
    for (int j = 0; j < max / 2; ++j) {
        for (int i = j; i < cutoff; i += max) {
            projector = projector.add(getProjector(cutoff - (i + 1), cutoff));
        }
    }
    
    CUDASparseMatrix x = QubitGatesGPU::X();
    CUDASparseMatrix arg = x.kroneckerProduct(projector).scale(Complex(0, -M_PI / 2));
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::eswap(double theta, int cutoff_a, int cutoff_b) const {
    int dim = cutoff_a * cutoff_b;
    CUDASparseMatrix swap(dim, dim);
    
    for (int m = 0; m < cutoff_a; ++m) {
        for (int n = 0; n < cutoff_b; ++n) {
            int row_index = n + m * cutoff_a;
            int col_index = m + n * cutoff_b;
            swap.set(row_index, col_index, Complex(1, 0));
        }
    }
    
    swap.uploadToDevice();
    
    CUDASparseMatrix arg = swap.scale(Complex(0, 1) * theta);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::csq(Complex theta, int cutoff) const {
    CUDASparseMatrix a = getA(cutoff);
    CUDASparseMatrix a2 = a.multiply(a);
    
    CUDASparseMatrix arg = a2.scale(conj(theta) * Complex(0.5));
    CUDASparseMatrix hc = arg.conjugateTranspose();
    
    CUDASparseMatrix z = QubitGatesGPU::Z();
    
    return z.kroneckerProduct(arg.subtract(hc)).exp();
}

CUDASparseMatrix CVOperatorsGPU::cMultibosonSampling(int max, int cutoff) const {
    return getEye(cutoff);
}

CUDASparseMatrix CVOperatorsGPU::gateFromMatrix(const std::vector<std::vector<Complex>>& matrix) const {
    int rows = matrix.size();
    if (rows == 0) {
        throw std::invalid_argument("Matrix is empty");
    }
    
    int cols = matrix[0].size();
    CUDASparseMatrix result(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        if (matrix[i].size() != cols) {
            throw std::invalid_argument("Matrix rows have different lengths");
        }
        
        for (int j = 0; j < cols; ++j) {
            if (matrix[i][j] != Complex(0, 0)) {
                result.set(i, j, matrix[i][j]);
            }
        }
    }
    
    result.uploadToDevice();
    return result;
}

CUDASparseMatrix CVOperatorsGPU::sum(double scale, int cutoff_a, int cutoff_b) const {
    CUDASparseMatrix a_mat = getA(cutoff_a).add(getAd(cutoff_a));
    CUDASparseMatrix b_mat = getAd(cutoff_b).subtract(getA(cutoff_b));
    
    CUDASparseMatrix arg = a_mat.kroneckerProduct(b_mat).scale(scale / 2);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::csum(double scale, int cutoff_a, int cutoff_b) const {
    CUDASparseMatrix a_mat = getA(cutoff_a).add(getAd(cutoff_a));
    CUDASparseMatrix b_mat = getAd(cutoff_b).subtract(getA(cutoff_b));
    
    CUDASparseMatrix z = QubitGatesGPU::Z();
    CUDASparseMatrix arg = z.kroneckerProduct(a_mat.kroneckerProduct(b_mat)).scale(scale / 2);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::jc(double theta, double phi, int cutoff) const {
    CUDASparseMatrix sminus = QubitGatesGPU::SMINUS();
    CUDASparseMatrix ad = getAd(cutoff);
    
    Complex exp_phi = exp(Complex(0, 1) * phi);
    CUDASparseMatrix arg = sminus.kroneckerProduct(ad).scale(exp_phi);
    CUDASparseMatrix hc = arg.conjugateTranspose();
    
    arg = arg.add(hc).scale(Complex(0, -1) * theta);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::ajc(double theta, double phi, int cutoff) const {
    CUDASparseMatrix splus = QubitGatesGPU::SPLUS();
    CUDASparseMatrix ad = getAd(cutoff);
    
    Complex exp_phi = exp(Complex(0, 1) * phi);
    CUDASparseMatrix arg = splus.kroneckerProduct(ad).scale(exp_phi);
    CUDASparseMatrix hc = arg.conjugateTranspose();
    
    arg = arg.add(hc).scale(Complex(0, -1) * theta);
    
    return arg.exp();
}

CUDASparseMatrix CVOperatorsGPU::rb(Complex theta, int cutoff) const {
    CUDASparseMatrix x = QubitGatesGPU::X();
    CUDASparseMatrix ad = getAd(cutoff);
    CUDASparseMatrix a = getA(cutoff);
    
    CUDASparseMatrix arg = ad.add(a).scale(theta);
    arg = x.kroneckerProduct(arg).scale(Complex(0, -1));
    
    return arg.exp();
}





// 拷贝构造函数
CUDASparseMatrix::CUDASparseMatrix(const CUDASparseMatrix& other) : 
    rows_(other.rows_), 
    cols_(other.cols_),
    temp_values_(other.temp_values_),
    temp_row_indices_(other.temp_row_indices_),
    temp_col_indices_(other.temp_col_indices_)
{
    // 初始化设备数据指针为nullptr
    device_data_.values = nullptr;
    device_data_.row_indices = nullptr;
    device_data_.col_indices = nullptr;
    device_data_.nnz = 0;
    
    // 如果other有设备数据，上传临时数据到设备
    if (other.device_data_.nnz > 0) {
        uploadToDevice();
    }
}

// 拷贝赋值运算符
CUDASparseMatrix& CUDASparseMatrix::operator=(const CUDASparseMatrix& other) {
    // 检查自赋值
    if (this != &other) {
        // 释放当前的设备数据
        if (device_data_.values != nullptr) {
            CUDA_CHECK(cudaFree(device_data_.values));
            device_data_.values = nullptr;
        }
        if (device_data_.row_indices != nullptr) {
            CUDA_CHECK(cudaFree(device_data_.row_indices));
            device_data_.row_indices = nullptr;
        }
        if (device_data_.col_indices != nullptr) {
            CUDA_CHECK(cudaFree(device_data_.col_indices));
            device_data_.col_indices = nullptr;
        }
        
        // 复制基本属性
        rows_ = other.rows_;
        cols_ = other.cols_;
        
        // 复制临时数据
        temp_values_ = other.temp_values_;
        temp_row_indices_ = other.temp_row_indices_;
        temp_col_indices_ = other.temp_col_indices_;
        
        // 重新初始化设备数据指针
        device_data_.nnz = 0;
        
        // 如果other有设备数据，上传临时数据到设备
        if (other.device_data_.nnz > 0) {
            uploadToDevice();
        }
    }
    
    return *this;
}

// 移动构造函数
CUDASparseMatrix::CUDASparseMatrix(CUDASparseMatrix&& other) : 
    rows_(other.rows_), 
    cols_(other.cols_),
    temp_values_(std::move(other.temp_values_)),
    temp_row_indices_(std::move(other.temp_row_indices_)),
    temp_col_indices_(std::move(other.temp_col_indices_))
{
    // 接管other的设备数据指针
    device_data_.values = other.device_data_.values;
    device_data_.row_indices = other.device_data_.row_indices;
    device_data_.col_indices = other.device_data_.col_indices;
    device_data_.nnz = other.device_data_.nnz;
    
    // 将other的设备数据指针设置为nullptr，避免双重释放
    other.device_data_.values = nullptr;
    other.device_data_.row_indices = nullptr;
    other.device_data_.col_indices = nullptr;
    other.device_data_.nnz = 0;
    
    // 重置other的基本属性
    other.rows_ = 0;
    other.cols_ = 0;
}

// 移动赋值运算符
CUDASparseMatrix& CUDASparseMatrix::operator=(CUDASparseMatrix&& other) {
    // 检查自赋值
    if (this != &other) {
        // 释放当前的设备数据
        if (device_data_.values != nullptr) {
            CUDA_CHECK(cudaFree(device_data_.values));
        }
        if (device_data_.row_indices != nullptr) {
            CUDA_CHECK(cudaFree(device_data_.row_indices));
        }
        if (device_data_.col_indices != nullptr) {
            CUDA_CHECK(cudaFree(device_data_.col_indices));
        }
        
        // 接管other的基本属性
        rows_ = other.rows_;
        cols_ = other.cols_;
        
        // 接管other的设备数据指针
        device_data_.values = other.device_data_.values;
        device_data_.row_indices = other.device_data_.row_indices;
        device_data_.col_indices = other.device_data_.col_indices;
        device_data_.nnz = other.device_data_.nnz;
        
        // 接管other的临时数据
        temp_values_ = std::move(other.temp_values_);
        temp_row_indices_ = std::move(other.temp_row_indices_);
        temp_col_indices_ = std::move(other.temp_col_indices_);
        
        // 将other的设备数据指针设置为nullptr，避免双重释放
        other.device_data_.values = nullptr;
        other.device_data_.row_indices = nullptr;
        other.device_data_.col_indices = nullptr;
        other.device_data_.nnz = 0;
        
        // 重置other的基本属性
        other.rows_ = 0;
        other.cols_ = 0;
    }
    
    return *this;
}