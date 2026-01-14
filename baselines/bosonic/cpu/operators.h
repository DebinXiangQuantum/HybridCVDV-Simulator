#ifndef BOSONIC_OPERATORS_CPU_H
#define BOSONIC_OPERATORS_CPU_H

#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <map>

// 自定义复数结构体，与GPU版本保持一致
struct Complex {
    double real;
    double imag;
    
    // 构造函数
    Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}
    
    // 转换构造函数，用于从std::complex<double>转换
    Complex(const std::complex<double>& c) : real(c.real()), imag(c.imag()) {}
    
    // 转换运算符，用于转换为std::complex<double>
    operator std::complex<double>() const {
        return std::complex<double>(real, imag);
    }
    
    // 比较运算符
    bool operator==(const Complex& other) const {
        return real == other.real && imag == other.imag;
    }
    
    // 不等于运算符
    bool operator!=(const Complex& other) const {
        return !(*this == other);
    }
    
    // 加法运算符
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    // 减法运算符
    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    // 乘法运算符（复数与复数）
    Complex operator*(const Complex& other) const {
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }
    
    // 乘法运算符（复数与标量）
    Complex operator*(double scalar) const {
        return Complex(real * scalar, imag * scalar);
    }
    
    // 加法赋值运算符
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }
    
    // 乘法赋值运算符
    Complex& operator*=(const Complex& other) {
        double r = real * other.real - imag * other.imag;
        double i = real * other.imag + imag * other.real;
        real = r;
        imag = i;
        return *this;
    }
    
    // 乘法赋值运算符（标量）
    Complex& operator*=(double scalar) {
        real *= scalar;
        imag *= scalar;
        return *this;
    }
    
    // 除法赋值运算符（复数与复数）
    Complex& operator/=(const Complex& other) {
        // 复数除法：(a + bi)/(c + di) = [(a+bi)(c-di)] / (c² + d²)
        double denominator = other.real * other.real + other.imag * other.imag;
        double r = (real * other.real + imag * other.imag) / denominator;
        double i = (imag * other.real - real * other.imag) / denominator;
        real = r;
        imag = i;
        return *this;
    }
    
    // 除法赋值运算符（复数与标量）
    Complex& operator/=(double scalar) {
        real /= scalar;
        imag /= scalar;
        return *this;
    }
    
    // 除法运算符（复数与标量）
    Complex operator/(double scalar) const {
        return Complex(real / scalar, imag / scalar);
    }
    
    // 一元负号运算符
    Complex operator-() const {
        return Complex(-real, -imag);
    }
};

// 标量与复数的乘法（友元函数）
inline Complex operator*(double scalar, const Complex& c) {
    return Complex(c.real * scalar, c.imag * scalar);
}

// 共轭函数
inline Complex conj(const Complex& c) {
    return Complex(c.real, -c.imag);
}

// 指数函数
inline Complex exp(const Complex& c) {
    double mag = std::exp(c.real);
    return Complex(mag * std::cos(c.imag), mag * std::sin(c.imag));
}

// 绝对值函数
inline double abs(const Complex& c) {
    return std::sqrt(c.real * c.real + c.imag * c.imag);
}

// 辐角函数
inline double arg(const Complex& c) {
    return std::atan2(c.imag, c.real);
}

// 稀疏矩阵类
template <typename T>
class SparseMatrix {
public:
    SparseMatrix(int rows, int cols);
    ~SparseMatrix();
    
    void set(int row, int col, T value);
    T get(int row, int col) const;
    
    SparseMatrix<T> conjugateTranspose() const;
    SparseMatrix<T> multiply(const SparseMatrix<T>& other) const;
    SparseMatrix<T> add(const SparseMatrix<T>& other) const;
    SparseMatrix<T> subtract(const SparseMatrix<T>& other) const;
    SparseMatrix<T> scale(T scalar) const;
    
    SparseMatrix<T> exp() const;
    SparseMatrix<T> kroneckerProduct(const SparseMatrix<T>& other) const;
    
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    
    // 转换为密集矩阵
    std::vector<std::vector<T>> toDense() const;
    
private:
    int rows_;
    int cols_;
    std::map<std::pair<int, int>, T> data_;
};

// 定义常用的量子门矩阵
class QubitGates {
public:
    static SparseMatrix<Complex> I();
    static SparseMatrix<Complex> X();
    static SparseMatrix<Complex> Y();
    static SparseMatrix<Complex> Z();
    static SparseMatrix<Complex> SPLUS();
    static SparseMatrix<Complex> SMINUS();
    static SparseMatrix<Complex> P0();
    static SparseMatrix<Complex> P1();
};

// 连续变量操作符类
class CVOperatorsCPU {
public:
    // 单位矩阵
    SparseMatrix<Complex> getEye(int cutoff) const;
    
    // 湮灭算符
    SparseMatrix<Complex> getA(int cutoff) const;
    
    // 产生算符
    SparseMatrix<Complex> getAd(int cutoff) const;
    
    // 数算符
    SparseMatrix<Complex> getN(int cutoff) const;
    
    // 投影算符
    SparseMatrix<Complex> getProjector(int n, int cutoff) const;
    
    // 符号构造算符
    SparseMatrix<Complex> getOp(const std::string& expr, const std::vector<int>& cutoffs) const;
    
    // 相位空间旋转算符
    SparseMatrix<Complex> r(double theta, int cutoff) const;
    
    // 位移算符
    SparseMatrix<Complex> d(Complex alpha, int cutoff) const;
    
    // 单模压缩算符
    SparseMatrix<Complex> s(Complex theta, int cutoff) const;
    
    // 双模压缩算符
    SparseMatrix<Complex> s2(Complex theta, int cutoff_a, int cutoff_b) const;
    
    // 三模压缩算符
    SparseMatrix<Complex> s3(Complex theta, int cutoff_a, int cutoff_b, int cutoff_c) const;
    
    // 双模分束器算符
    SparseMatrix<Complex> bs(Complex theta, int cutoff_a, int cutoff_b) const;
    
    // 受控相位空间旋转算符
    SparseMatrix<Complex> cr(double theta, int cutoff) const;
    
    // 受控X轴相位空间旋转算符
    SparseMatrix<Complex> crx(double theta, int cutoff) const;
    
    // 受控Y轴相位空间旋转算符
    SparseMatrix<Complex> cry(double theta, int cutoff) const;
    
    // 受控位移算符
    SparseMatrix<Complex> cd(Complex alpha, Complex* beta, int cutoff) const;
    
    // 回波受控位移算符
    SparseMatrix<Complex> ecd(Complex theta, int cutoff) const;
    
    // 受控相位双模分束器算符
    SparseMatrix<Complex> cbs(Complex theta, int cutoff_a, int cutoff_b) const;
    
    // 受控Schwinger门
    SparseMatrix<Complex> cschwinger(
        double beta, double theta_1, double phi_1, 
        double theta_2, double phi_2, 
        int cutoff_a, int cutoff_b) const;
    
    // SNAP算符
    SparseMatrix<Complex> snap(double theta, int n, int cutoff) const;
    
    // csnap算符
    SparseMatrix<Complex> csnap(double theta, int n, int cutoff) const;
    
    // 多Fock态SNAP算符
    SparseMatrix<Complex> multisnap(const std::vector<double>& params, int cutoff) const;
    
    // 多Fock态csnap算符
    SparseMatrix<Complex> multicsnap(const std::vector<double>& params, int cutoff) const;
    
    // SQR门
    SparseMatrix<Complex> sqr(const std::vector<double>& params, int cutoff) const;
    
    // 光子数读出支持门
    SparseMatrix<Complex> pnr(int max, int cutoff) const;
    
    // 指数SWAP算符
    SparseMatrix<Complex> eswap(double theta, int cutoff_a, int cutoff_b) const;
    
    // 受控单模压缩算符
    SparseMatrix<Complex> csq(Complex theta, int cutoff) const;
    
    // 多玻色子采样SNAP门
    SparseMatrix<Complex> cMultibosonSampling(int max, int cutoff) const;
    
    // 从矩阵创建门
    SparseMatrix<Complex> gateFromMatrix(const std::vector<std::vector<Complex>>& matrix) const;
    
    // 双模求和门
    SparseMatrix<Complex> sum(double scale, int cutoff_a, int cutoff_b) const;
    
    // 条件双模求和门
    SparseMatrix<Complex> csum(double scale, int cutoff_a, int cutoff_b) const;
    
    // Jaynes-Cummings门
    SparseMatrix<Complex> jc(double theta, double phi, int cutoff) const;
    
    // 反Jaynes-Cummings门
    SparseMatrix<Complex> ajc(double theta, double phi, int cutoff) const;
    
    // Rabi相互作用门
    SparseMatrix<Complex> rb(Complex theta, int cutoff) const;
    
private:
    // 辅助方法
    void parseOpExpr(const std::string& expr, std::map<int, std::string>& ops) const;
};

#endif // BOSONIC_OPERATORS_CPU_H