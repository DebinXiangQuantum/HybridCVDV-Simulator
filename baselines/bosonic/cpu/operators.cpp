#include "operators.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <regex>

// SparseMatrix 实现
template <typename T>
SparseMatrix<T>::SparseMatrix(int rows, int cols) : rows_(rows), cols_(cols) {
}

template <typename T>
SparseMatrix<T>::~SparseMatrix() {
}

template <typename T>
void SparseMatrix<T>::set(int row, int col, T value) {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    
    if (value != T(0)) {
        data_[std::make_pair(row, col)] = value;
    } else {
        auto it = data_.find(std::make_pair(row, col));
        if (it != data_.end()) {
            data_.erase(it);
        }
    }
}

template <typename T>
T SparseMatrix<T>::get(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    
    auto it = data_.find(std::make_pair(row, col));
    if (it != data_.end()) {
        return it->second;
    }
    return T(0);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::conjugateTranspose() const {
    SparseMatrix<T> result(cols_, rows_);
    
    for (const auto& entry : data_) {
        int row = entry.first.first;
        int col = entry.first.second;
        T value = conj(entry.second);
        result.set(col, row, value);
    }
    
    return result;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::multiply(const SparseMatrix<T>& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    
    SparseMatrix<T> result(rows_, other.cols_);
    
    // 为了提高效率，我们可以将other矩阵转换为按列存储的结构
    std::map<int, std::map<int, T>> other_col_data;
    for (const auto& entry : other.data_) {
        int row = entry.first.first;
        int col = entry.first.second;
        other_col_data[col][row] = entry.second;
    }
    
    // 执行矩阵乘法
    for (const auto& entry1 : data_) {
        int row1 = entry1.first.first;
        int col1 = entry1.first.second;
        T val1 = entry1.second;
        
        // 检查other矩阵中是否有col1列
        auto col_it = other_col_data.find(col1);
        if (col_it != other_col_data.end()) {
            // 遍历other矩阵中col1列的所有元素
            for (const auto& entry2 : col_it->second) {
                int row2 = entry2.first;
                int col2 = col1;
                T val2 = entry2.second;
                
                // 更新结果矩阵
                T current = result.get(row1, row2);
                result.set(row1, row2, current + val1 * val2);
            }
        }
    }
    
    return result;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::add(const SparseMatrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }
    
    SparseMatrix<T> result(rows_, cols_);
    
    // 复制当前矩阵的数据
    for (const auto& entry : data_) {
        result.set(entry.first.first, entry.first.second, entry.second);
    }
    
    // 添加other矩阵的数据
    for (const auto& entry : other.data_) {
        int row = entry.first.first;
        int col = entry.first.second;
        T current = result.get(row, col);
        result.set(row, col, current + entry.second);
    }
    
    return result;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::subtract(const SparseMatrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
    }
    
    SparseMatrix<T> result(rows_, cols_);
    
    // 复制当前矩阵的数据
    for (const auto& entry : data_) {
        result.set(entry.first.first, entry.first.second, entry.second);
    }
    
    // 减去other矩阵的数据
    for (const auto& entry : other.data_) {
        int row = entry.first.first;
        int col = entry.first.second;
        T current = result.get(row, col);
        result.set(row, col, current - entry.second);
    }
    
    return result;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::scale(T scalar) const {
    SparseMatrix<T> result(rows_, cols_);
    
    for (const auto& entry : data_) {
        result.set(entry.first.first, entry.first.second, entry.second * scalar);
    }
    
    return result;
}

// 矩阵指数的简单实现（实际应用中应使用更高效的算法）
template <typename T>
SparseMatrix<T> SparseMatrix<T>::exp() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square to compute exponential");
    }
    
    // 转换为密集矩阵
    auto dense = toDense();
    
    // 创建结果矩阵（初始化为单位矩阵）
    std::vector<std::vector<T>> result(rows_, std::vector<T>(cols_, T(0)));
    for (int i = 0; i < rows_; ++i) {
        result[i][i] = T(1);
    }
    
    // 创建当前项矩阵（初始化为输入矩阵）
    std::vector<std::vector<T>> current = dense;
    
    // 项的系数（初始化为1）
    T coeff = T(1);
    
    // 计算前20项的泰勒展开
    for (int n = 1; n <= 20; ++n) {
        // 更新系数
        coeff /= T(n);
        
        // 将current缩放并加到result
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[i][j] += coeff * current[i][j];
            }
        }
        
        // 计算下一项的矩阵（current = current * dense）
        std::vector<std::vector<T>> next(rows_, std::vector<T>(cols_, T(0)));
        for (int i = 0; i < rows_; ++i) {
            for (int k = 0; k < cols_; ++k) {
                if (current[i][k] != T(0)) {
                    for (int j = 0; j < cols_; ++j) {
                        if (dense[k][j] != T(0)) {
                            next[i][j] += current[i][k] * dense[k][j];
                        }
                    }
                }
            }
        }
        
        current = next;
    }
    
    // 转换回稀疏矩阵
    SparseMatrix<T> exp_result(rows_, cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            if (result[i][j] != T(0)) {
                exp_result.set(i, j, result[i][j]);
            }
        }
    }
    
    return exp_result;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::kroneckerProduct(const SparseMatrix<T>& other) const {
    int new_rows = rows_ * other.rows_;
    int new_cols = cols_ * other.cols_;
    
    SparseMatrix<T> result(new_rows, new_cols);
    
    for (const auto& entry1 : data_) {
        int row1 = entry1.first.first;
        int col1 = entry1.first.second;
        T val1 = entry1.second;
        
        for (const auto& entry2 : other.data_) {
            int row2 = entry2.first.first;
            int col2 = entry2.first.second;
            T val2 = entry2.second;
            
            int new_row = row1 * other.rows_ + row2;
            int new_col = col1 * other.cols_ + col2;
            
            result.set(new_row, new_col, val1 * val2);
        }
    }
    
    return result;
}

template <typename T>
std::vector<std::vector<T>> SparseMatrix<T>::toDense() const {
    std::vector<std::vector<T>> dense(rows_, std::vector<T>(cols_, T(0)));
    
    for (const auto& entry : data_) {
        dense[entry.first.first][entry.first.second] = entry.second;
    }
    
    return dense;
}

// 显式实例化SparseMatrix<Complex>
template class SparseMatrix<Complex>;

// QubitGates 实现
SparseMatrix<Complex> QubitGates::I() {
    SparseMatrix<Complex> mat(2, 2);
    mat.set(0, 0, Complex(1, 0));
    mat.set(1, 1, Complex(1, 0));
    return mat;
}

SparseMatrix<Complex> QubitGates::X() {
    SparseMatrix<Complex> mat(2, 2);
    mat.set(0, 1, Complex(1, 0));
    mat.set(1, 0, Complex(1, 0));
    return mat;
}

SparseMatrix<Complex> QubitGates::Y() {
    SparseMatrix<Complex> mat(2, 2);
    mat.set(0, 1, Complex(0, -1));
    mat.set(1, 0, Complex(0, 1));
    return mat;
}

SparseMatrix<Complex> QubitGates::Z() {
    SparseMatrix<Complex> mat(2, 2);
    mat.set(0, 0, Complex(1, 0));
    mat.set(1, 1, Complex(-1, 0));
    return mat;
}

SparseMatrix<Complex> QubitGates::SPLUS() {
    return (X().add(Y().scale(Complex(0, 1)))).scale(Complex(0.5, 0));
}

SparseMatrix<Complex> QubitGates::SMINUS() {
    return (X().subtract(Y().scale(Complex(0, 1)))).scale(Complex(0.5, 0));
}

SparseMatrix<Complex> QubitGates::P0() {
    return (I().add(Z())).scale(Complex(0.5, 0));
}

SparseMatrix<Complex> QubitGates::P1() {
    return (I().subtract(Z())).scale(Complex(0.5, 0));
}

// CVOperators 实现
SparseMatrix<Complex> CVOperatorsCPU::getEye(int cutoff) const {
    SparseMatrix<Complex> mat(cutoff, cutoff);
    for (int i = 0; i < cutoff; ++i) {
        mat.set(i, i, Complex(1, 0));
    }
    return mat;
}

SparseMatrix<Complex> CVOperatorsCPU::getA(int cutoff) const {
    SparseMatrix<Complex> mat(cutoff, cutoff);
    for (int i = 0; i < cutoff - 1; ++i) {
        double sqrt_i_plus_1 = std::sqrt(i + 1);
        mat.set(i, i + 1, Complex(sqrt_i_plus_1, 0));
    }
    return mat;
}

SparseMatrix<Complex> CVOperatorsCPU::getAd(int cutoff) const {
    return getA(cutoff).conjugateTranspose();
}

SparseMatrix<Complex> CVOperatorsCPU::getN(int cutoff) const {
    SparseMatrix<Complex> a = getA(cutoff);
    SparseMatrix<Complex> ad = getAd(cutoff);
    return ad.multiply(a);
}

SparseMatrix<Complex> CVOperatorsCPU::getProjector(int n, int cutoff) const {
    SparseMatrix<Complex> mat(cutoff, cutoff);
    mat.set(n, n, Complex(1, 0));
    return mat;
}

// 解析操作符表达式
void CVOperatorsCPU::parseOpExpr(const std::string& expr, std::map<int, std::string>& ops) const {
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

SparseMatrix<Complex> CVOperatorsCPU::getOp(const std::string& expr, const std::vector<int>& cutoffs) const {
    std::map<int, std::string> ops;
    parseOpExpr(expr, ops);
    
    if (ops.empty()) {
        throw std::invalid_argument("No operators found in expression");
    }
    
    // 确定最大的qumode索引
    int max_qumode = 0;
    for (const auto& op : ops) {
        max_qumode = std::max(max_qumode, op.first);
    }
    
    if (max_qumode >= cutoffs.size()) {
        throw std::invalid_argument("Not enough cutoffs provided");
    }
    
    // 创建所有qumode的矩阵列表
    std::vector<SparseMatrix<Complex>> mats;
    for (int i = 0; i <= max_qumode; ++i) {
        int cutoff = cutoffs[i];
        auto it = ops.find(i);
        if (it != ops.end()) {
            const std::string& op = it->second;
            if (op == "a") {
                mats.push_back(getA(cutoff));
            } else if (op == "ad") {
                mats.push_back(getAd(cutoff));
            } else if (op == "n") {
                mats.push_back(getN(cutoff));
            } else {
                throw std::invalid_argument("Unrecognized operator: " + op);
            }
        } else {
            mats.push_back(getEye(cutoff));
        }
    }
    
    // 计算所有矩阵的张量积
    SparseMatrix<Complex> result = mats[0];
    for (size_t i = 1; i < mats.size(); ++i) {
        result = result.kroneckerProduct(mats[i]);
    }
    
    return result;
}

SparseMatrix<Complex> CVOperatorsCPU::r(double theta, int cutoff) const {
    SparseMatrix<Complex> arg = getN(cutoff).scale(Complex(0, 1) * Complex(theta));
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::d(Complex alpha, int cutoff) const {
    SparseMatrix<Complex> ad = getAd(cutoff);
    SparseMatrix<Complex> a = getA(cutoff);
    
    SparseMatrix<Complex> arg = ad.scale(alpha).subtract(a.scale(conj(alpha)));
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::s(Complex theta, int cutoff) const {
    SparseMatrix<Complex> ad = getAd(cutoff);
    SparseMatrix<Complex> ad2 = ad.multiply(ad);
    
    SparseMatrix<Complex> arg = ad2.scale(conj(theta) * Complex(0.5));
    SparseMatrix<Complex> hc = arg.conjugateTranspose();
    
    return arg.subtract(hc).exp();
}

SparseMatrix<Complex> CVOperatorsCPU::s2(Complex theta, int cutoff_a, int cutoff_b) const {
    double r = abs(theta);
    double phi = arg(theta);
    
    std::vector<int> cutoffs = {cutoff_a, cutoff_b};
    SparseMatrix<Complex> op = getOp("ad0 ad1", cutoffs);
    SparseMatrix<Complex> arg = op.scale(r * exp(Complex(0, 1) * phi));
    SparseMatrix<Complex> hc = arg.conjugateTranspose();
    
    return arg.subtract(hc).exp();
}

SparseMatrix<Complex> CVOperatorsCPU::s3(Complex theta, int cutoff_a, int cutoff_b, int cutoff_c) const {
    double r = abs(theta);
    double phi = arg(theta);
    
    std::vector<int> cutoffs = {cutoff_a, cutoff_b, cutoff_c};
    SparseMatrix<Complex> op = getOp("ad0 ad1 ad2", cutoffs);
    SparseMatrix<Complex> arg = op.scale(r * exp(Complex(0, 1) * phi));
    SparseMatrix<Complex> hc = arg.conjugateTranspose();
    
    return arg.subtract(hc).exp();
}

SparseMatrix<Complex> CVOperatorsCPU::bs(Complex theta, int cutoff_a, int cutoff_b) const {
    std::vector<int> cutoffs = {cutoff_a, cutoff_b};
    SparseMatrix<Complex> op = getOp("ad0 a1", cutoffs);
    SparseMatrix<Complex> arg = op.scale(theta);
    SparseMatrix<Complex> hc = arg.conjugateTranspose();
    
    return arg.subtract(hc).exp();
}

SparseMatrix<Complex> CVOperatorsCPU::cr(double theta, int cutoff) const {
    SparseMatrix<Complex> z = QubitGates::Z();
    SparseMatrix<Complex> n = getN(cutoff);
    SparseMatrix<Complex> arg = z.kroneckerProduct(n).scale(Complex(0, 1) * theta);
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::crx(double theta, int cutoff) const {
    SparseMatrix<Complex> x = QubitGates::X();
    SparseMatrix<Complex> n = getN(cutoff);
    SparseMatrix<Complex> arg = x.kroneckerProduct(n).scale(Complex(0, 1) * theta);
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::cry(double theta, int cutoff) const {
    SparseMatrix<Complex> y = QubitGates::Y();
    SparseMatrix<Complex> n = getN(cutoff);
    SparseMatrix<Complex> arg = y.kroneckerProduct(n).scale(Complex(0, 1) * theta);
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::cd(Complex alpha, Complex* beta, int cutoff) const {
    SparseMatrix<Complex> displace0 = d(alpha, cutoff);
    Complex beta_val = (beta != nullptr) ? *beta : -alpha;
    SparseMatrix<Complex> displace1 = d(beta_val, cutoff);
    
    SparseMatrix<Complex> p0 = QubitGates::P0();
    SparseMatrix<Complex> p1 = QubitGates::P1();
    
    return p0.kroneckerProduct(displace0).add(p1.kroneckerProduct(displace1));
}

SparseMatrix<Complex> CVOperatorsCPU::ecd(Complex theta, int cutoff) const {
    Complex neg_theta = -theta;  // 创建临时变量存储-theta
    return cd(theta, &neg_theta, cutoff);
}

SparseMatrix<Complex> CVOperatorsCPU::cbs(Complex theta, int cutoff_a, int cutoff_b) const {
    SparseMatrix<Complex> arg = bs(theta, cutoff_a, cutoff_b);
    SparseMatrix<Complex> p0 = QubitGates::P0();
    SparseMatrix<Complex> p1 = QubitGates::P1();
    
    return p0.kroneckerProduct(arg).add(p1.kroneckerProduct(arg.conjugateTranspose()));
}

SparseMatrix<Complex> CVOperatorsCPU::cschwinger(
    double beta, double theta_1, double phi_1,
    double theta_2, double phi_2,
    int cutoff_a, int cutoff_b) const {
    std::vector<int> cutoffs = {cutoff_a, cutoff_b};
    
    // 计算Sx
    SparseMatrix<Complex> sx = getOp("a0 ad1", cutoffs);
    sx = sx.add(sx.conjugateTranspose()).scale(Complex(0.5, 0));
    
    // 计算Sy
    SparseMatrix<Complex> sy = getOp("a0 ad1", cutoffs);
    sy = sy.subtract(sy.conjugateTranspose()).scale(Complex(0, -0.5));
    
    // 计算Sz
    SparseMatrix<Complex> sz = getOp("n1", cutoffs).subtract(getOp("n0", cutoffs)).scale(Complex(0.5, 0));
    
    // 计算sigma
    SparseMatrix<Complex> sigma = QubitGates::X().scale(std::sin(theta_1) * std::cos(phi_1))
                                .add(QubitGates::Y().scale(std::sin(theta_1) * std::sin(phi_1)))
                                .add(QubitGates::Z().scale(std::cos(theta_1)));
    
    // 计算S
    SparseMatrix<Complex> s = sx.scale(std::sin(theta_2) * std::cos(phi_2))
                            .add(sy.scale(std::sin(theta_2) * std::sin(phi_2)))
                            .add(sz.scale(std::cos(theta_2)));
    
    // 计算指数
    SparseMatrix<Complex> arg = sigma.kroneckerProduct(s).scale(Complex(0, -1) * beta);
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::snap(double theta, int n, int cutoff) const {
    std::vector<double> params{theta, static_cast<double>(n)};
    return multisnap(params, cutoff);
}

SparseMatrix<Complex> CVOperatorsCPU::csnap(double theta, int n, int cutoff) const {
    std::vector<double> params{theta, static_cast<double>(n)};
    return multicsnap(params, cutoff);
}

SparseMatrix<Complex> CVOperatorsCPU::multisnap(const std::vector<double>& params, int cutoff) const {
    if (params.size() % 2 != 0) {
        throw std::invalid_argument("Params must have even number of elements");
    }
    
    int num_pairs = params.size() / 2;
    std::vector<double> thetas(params.begin(), params.begin() + num_pairs);
    std::vector<int> ns;
    for (int i = num_pairs; i < params.size(); ++i) {
        ns.push_back(static_cast<int>(params[i]));
    }
    
    SparseMatrix<Complex> mat(cutoff, cutoff);
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
    
    return mat;
}

SparseMatrix<Complex> CVOperatorsCPU::multicsnap(const std::vector<double>& params, int cutoff) const {
    if (params.size() % 2 != 0) {
        throw std::invalid_argument("Params must have even number of elements");
    }
    
    int num_pairs = params.size() / 2;
    std::vector<double> thetas(params.begin(), params.begin() + num_pairs);
    std::vector<int> ns;
    for (int i = num_pairs; i < params.size(); ++i) {
        ns.push_back(static_cast<int>(params[i]));
    }
    
    SparseMatrix<Complex> mat(2 * cutoff, 2 * cutoff);
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
    
    return mat;
}

SparseMatrix<Complex> CVOperatorsCPU::sqr(const std::vector<double>& params, int cutoff) const {
    if (params.size() % 3 != 0) {
        throw std::invalid_argument("Params must have multiple of 3 elements");
    }
    
    int num_blocks = params.size() / 3;
    
    // 创建块矩阵列表
    std::vector<SparseMatrix<Complex>> blocks;
    for (int i = 0; i < cutoff; ++i) {
        blocks.push_back(QubitGates::I());
    }
    
    for (int i = 0; i < num_blocks; ++i) {
        double theta = params[3 * i];
        double phi = params[3 * i + 1];
        int n = static_cast<int>(params[3 * i + 2]);
        
        if (n < 0 || n >= cutoff) {
            throw std::invalid_argument("Fock state out of bounds");
        }
        
        // 创建R门矩阵
        SparseMatrix<Complex> r_gate(2, 2);  // 显式指定模板参数
        r_gate.set(0, 0, exp(Complex(0, 1) * phi / 2.0));
        r_gate.set(0, 1, Complex(0, 0));
        r_gate.set(1, 0, Complex(0, 0));
        r_gate.set(1, 1, exp(Complex(0, -1) * phi / 2.0));
        
        blocks[n] = r_gate;
    }
    
    // 创建块对角矩阵
    int total_size = 2 * cutoff;
    SparseMatrix<Complex> out(total_size, total_size);
    
    for (int i = 0; i < cutoff; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                Complex value = blocks[i].get(j, k);
                if (value != Complex(0, 0)) {
                    out.set(2 * i + j, 2 * i + k, value);
                }
            }
        }
    }
    
    // 重新排列以作用于Qubit x Qumode
    SparseMatrix<Complex> result(total_size, total_size);
    for (int i = 0; i < cutoff; ++i) {
        for (int j = 0; j < 2; ++j) {
            int old_index = 2 * i + j;
            int new_index = j * cutoff + i;
            
            for (int k = 0; k < total_size; ++k) {
                Complex value = out.get(old_index, k);
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
    
    return result;
}

SparseMatrix<Complex> CVOperatorsCPU::pnr(int max, int cutoff) const {
    SparseMatrix<Complex> projector(cutoff, cutoff);
    
    // 创建投影器
    for (int j = 0; j < max / 2; ++j) {
        for (int i = j; i < cutoff; i += max) {
            projector = projector.add(getProjector(cutoff - (i + 1), cutoff));
        }
    }
    
    SparseMatrix<Complex> x = QubitGates::X();
    SparseMatrix<Complex> arg = x.kroneckerProduct(projector).scale(Complex(0, -M_PI / 2));
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::eswap(double theta, int cutoff_a, int cutoff_b) const {
    int dim = cutoff_a * cutoff_b;
    SparseMatrix<Complex> swap(dim, dim);
    
    for (int m = 0; m < cutoff_a; ++m) {
        for (int n = 0; n < cutoff_b; ++n) {
            int row_index = n + m * cutoff_a;
            int col_index = m + n * cutoff_b;
            swap.set(row_index, col_index, Complex(1, 0));
        }
    }
    
    SparseMatrix<Complex> arg = swap.scale(Complex(0, 1) * theta);
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::csq(Complex theta, int cutoff) const {
    SparseMatrix<Complex> a = getA(cutoff);
    SparseMatrix<Complex> a2 = a.multiply(a);
    
    SparseMatrix<Complex> arg = a2.scale(conj(theta) * Complex(0.5));
    SparseMatrix<Complex> hc = arg.conjugateTranspose();
    
    SparseMatrix<Complex> z = QubitGates::Z();
    
    return z.kroneckerProduct(arg.subtract(hc)).exp();
}

SparseMatrix<Complex> CVOperatorsCPU::cMultibosonSampling(int max, int cutoff) const {
    return getEye(cutoff);
}

SparseMatrix<Complex> CVOperatorsCPU::gateFromMatrix(const std::vector<std::vector<Complex>>& matrix) const {
    int rows = matrix.size();
    if (rows == 0) {
        throw std::invalid_argument("Matrix is empty");
    }
    
    int cols = matrix[0].size();
    SparseMatrix<Complex> result(rows, cols);
    
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
    
    return result;
}

SparseMatrix<Complex> CVOperatorsCPU::sum(double scale, int cutoff_a, int cutoff_b) const {
    SparseMatrix<Complex> a_mat = getA(cutoff_a).add(getAd(cutoff_a));
    SparseMatrix<Complex> b_mat = getAd(cutoff_b).subtract(getA(cutoff_b));
    
    SparseMatrix<Complex> arg = a_mat.kroneckerProduct(b_mat).scale(scale / 2);
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::csum(double scale, int cutoff_a, int cutoff_b) const {
    SparseMatrix<Complex> a_mat = getA(cutoff_a).add(getAd(cutoff_a));
    SparseMatrix<Complex> b_mat = getAd(cutoff_b).subtract(getA(cutoff_b));
    
    SparseMatrix<Complex> z = QubitGates::Z();
    SparseMatrix<Complex> arg = z.kroneckerProduct(a_mat.kroneckerProduct(b_mat)).scale(scale / 2);
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::jc(double theta, double phi, int cutoff) const {
    SparseMatrix<Complex> sminus = QubitGates::SMINUS();
    SparseMatrix<Complex> ad = getAd(cutoff);
    
    Complex exp_phi = exp(Complex(0, 1) * phi);
    SparseMatrix<Complex> arg = sminus.kroneckerProduct(ad).scale(exp_phi);
    SparseMatrix<Complex> hc = arg.conjugateTranspose();
    
    arg = arg.add(hc).scale(Complex(0, -1) * theta);
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::ajc(double theta, double phi, int cutoff) const {
    SparseMatrix<Complex> splus = QubitGates::SPLUS();
    SparseMatrix<Complex> ad = getAd(cutoff);
    
    Complex exp_phi = exp(Complex(0, 1) * phi);
    SparseMatrix<Complex> arg = splus.kroneckerProduct(ad).scale(exp_phi);
    SparseMatrix<Complex> hc = arg.conjugateTranspose();
    
    arg = arg.add(hc).scale(Complex(0, -1) * theta);
    
    return arg.exp();
}

SparseMatrix<Complex> CVOperatorsCPU::rb(Complex theta, int cutoff) const {
    SparseMatrix<Complex> x = QubitGates::X();
    SparseMatrix<Complex> ad = getAd(cutoff);
    SparseMatrix<Complex> a = getA(cutoff);
    
    SparseMatrix<Complex> arg = ad.add(a).scale(theta);
    arg = x.kroneckerProduct(arg).scale(Complex(0, -1));
    
    return arg.exp();
}