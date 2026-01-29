#ifndef BOSONIC_OPERATORS_GPU_H
#define BOSONIC_OPERATORS_GPU_H

#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <map>

// 声明gpu命名空间以便在全局声明的函数中使用
namespace gpu {
struct Complex;
}

// CUDA核函数声明（全局命名空间）
#ifdef __CUDACC__
__global__ void conjugate_transpose_kernel(gpu::Complex* in_values, gpu::Complex* out_values, int nnz);
__global__ void scale_kernel(gpu::Complex* values, gpu::Complex scalar, int nnz);
#endif

// GPU命名空间，用于避免与CPU版本的函数冲突
namespace gpu {

// 自定义复数结构体，用于CUDA设备代码
struct Complex {
    double real;
    double imag;
    
    // 构造函数
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}
    
    // 转换构造函数，用于从std::complex<double>转换
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex(const std::complex<double>& c) : real(c.real()), imag(c.imag()) {}
    
    // 转换运算符，用于转换为std::complex<double>
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    operator std::complex<double>() const {
        return std::complex<double>(real, imag);
    }
    
    // 比较运算符
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    bool operator==(const Complex& other) const {
        return real == other.real && imag == other.imag;
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    bool operator!=(const Complex& other) const {
        return !(*this == other);
    }
    
    // 算术运算符
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex operator*(double scalar) const {
        return Complex(real * scalar, imag * scalar);
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex& operator*=(const Complex& other) {
        double new_real = real * other.real - imag * other.imag;
        double new_imag = real * other.imag + imag * other.real;
        real = new_real;
        imag = new_imag;
        return *this;
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex& operator*=(double scalar) {
        real *= scalar;
        imag *= scalar;
        return *this;
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex& operator/=(const Complex& other) {
        double denominator = other.real * other.real + other.imag * other.imag;
        double new_real = (real * other.real + imag * other.imag) / denominator;
        double new_imag = (imag * other.real - real * other.imag) / denominator;
        real = new_real;
        imag = new_imag;
        return *this;
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex& operator/=(double scalar) {
        real /= scalar;
        imag /= scalar;
        return *this;
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex operator/(double scalar) const {
        return Complex(real / scalar, imag / scalar);
    }
    
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Complex operator-() const {
        return Complex(-real, -imag);
    };
};

// 标量乘法运算符
#ifdef __CUDACC__
__host__ __device__
#endif
inline Complex operator*(double scalar, const Complex& c) {
    return Complex(scalar * c.real, scalar * c.imag);
}

// 共轭函数
#ifdef __CUDACC__
__host__ __device__
#endif
inline Complex conj(const Complex& c) {
    return Complex(c.real, -c.imag);
}

// 指数函数
#ifdef __CUDACC__
__host__ __device__
#endif
inline Complex exp(const Complex& c) {
    double e_real = std::exp(c.real);
    return Complex(e_real * std::cos(c.imag), e_real * std::sin(c.imag));
}

// 绝对值函数
#ifdef __CUDACC__
__host__ __device__
#endif
inline double abs(const Complex& c) {
    return std::sqrt(c.real * c.real + c.imag * c.imag);
}

// 辐角函数
#ifdef __CUDACC__
__host__ __device__
#endif
inline double arg(const Complex& c) {
    return std::atan2(c.imag, c.real);
}

// CUDA稀疏矩阵类
class CUDASparseMatrix {
public:
    // 默认构造函数
    CUDASparseMatrix();
    
    CUDASparseMatrix(int rows, int cols);
    ~CUDASparseMatrix();
    
    void set(int row, int col, Complex value);
    Complex get(int row, int col) const;
    
    CUDASparseMatrix conjugateTranspose() const;
    CUDASparseMatrix multiply(const CUDASparseMatrix& other) const;
    CUDASparseMatrix add(const CUDASparseMatrix& other) const;
    CUDASparseMatrix subtract(const CUDASparseMatrix& other) const;
    CUDASparseMatrix scale(Complex scalar) const;
    
    CUDASparseMatrix exp() const;
    CUDASparseMatrix kroneckerProduct(const CUDASparseMatrix& other) const;
    
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    
    // 转换为密集矩阵（CPU上）
    std::vector<std::vector<Complex>> toDense() const;
    
    // 将CPU临时数据上传到GPU
    void uploadToDevice();
    
    // 从GPU下载数据到CPU
    void downloadFromDevice() const;
    
    // 拷贝构造函数
    CUDASparseMatrix(const CUDASparseMatrix& other);
    
    // 拷贝赋值运算符
    CUDASparseMatrix& operator=(const CUDASparseMatrix& other);
    
    // 移动构造函数
    CUDASparseMatrix(CUDASparseMatrix&& other);
    
    // 移动赋值运算符
    CUDASparseMatrix& operator=(CUDASparseMatrix&& other);
    
private:
    int rows_;
    int cols_;
    
    // CUDA设备上的数据（COO 格式）
    struct DeviceData {
        Complex* values;
        int* row_indices;
        int* col_indices;
        int nnz;
    };
    
    DeviceData device_data_;
    
    // CSR 格式数据（用于 cuSPARSE SpGEMM）
    struct CSRData {
        Complex* values;
        int* col_indices;
        int* row_ptr;
        int nnz;
        bool allocated;
    };
    
    mutable CSRData csr_data_;
    
    // CPU上的临时数据，用于构建矩阵
    mutable std::vector<Complex> temp_values_;
    mutable std::vector<int> temp_row_indices_;
    mutable std::vector<int> temp_col_indices_;
    
    // 格式转换函数
    void convertCOOtoCSR() const;
    void freeCSRData() const;
};

// 定义常用的量子门矩阵（GPU版本）
class QubitGatesGPU {
public:
    static CUDASparseMatrix I();
    static CUDASparseMatrix X();
    static CUDASparseMatrix Y();
    static CUDASparseMatrix Z();
    static CUDASparseMatrix SPLUS();
    static CUDASparseMatrix SMINUS();
    static CUDASparseMatrix P0();
    static CUDASparseMatrix P1();
};

// 连续变量操作符类（GPU版本）
class CVOperatorsGPU {
public:
    // 单位矩阵
    CUDASparseMatrix getEye(int cutoff) const;
    
    // 湮灭算符
    CUDASparseMatrix getA(int cutoff) const;
    
    // 产生算符
    CUDASparseMatrix getAd(int cutoff) const;
    
    // 数算符
    CUDASparseMatrix getN(int cutoff) const;
    
    // 投影算符
    CUDASparseMatrix getProjector(int n, int cutoff) const;
    
    // 符号构造算符
    CUDASparseMatrix getOp(const std::string& expr, const std::vector<int>& cutoffs) const;
    
    // 相位空间旋转算符
    CUDASparseMatrix r(double theta, int cutoff) const;
    
    // 位移算符
    CUDASparseMatrix d(Complex alpha, int cutoff) const;
    
    // 单模压缩算符
    CUDASparseMatrix s(Complex theta, int cutoff) const;
    
    // 双模压缩算符
    CUDASparseMatrix s2(Complex theta, int cutoff_a, int cutoff_b) const;
    
    // 三模压缩算符
    CUDASparseMatrix s3(Complex theta, int cutoff_a, int cutoff_b, int cutoff_c) const;
    
    // 双模分束器算符
    CUDASparseMatrix bs(Complex theta, int cutoff_a, int cutoff_b) const;
    
    // 受控相位空间旋转算符
    CUDASparseMatrix cr(double theta, int cutoff) const;
    
    // 受控X轴相位空间旋转算符
    CUDASparseMatrix crx(double theta, int cutoff) const;
    
    // 受控Y轴相位空间旋转算符
    CUDASparseMatrix cry(double theta, int cutoff) const;
    
    // 受控位移算符
    CUDASparseMatrix cd(Complex alpha, Complex* beta, int cutoff) const;
    
    // 回波受控位移算符
    CUDASparseMatrix ecd(Complex theta, int cutoff) const;
    
    // 受控相位双模分束器算符
    CUDASparseMatrix cbs(Complex theta, int cutoff_a, int cutoff_b) const;
    
    // 受控Schwinger门
    CUDASparseMatrix cschwinger(
        double beta, double theta_1, double phi_1, 
        double theta_2, double phi_2, 
        int cutoff_a, int cutoff_b) const;
    
    // SNAP算符
    CUDASparseMatrix snap(double theta, int n, int cutoff) const;
    
    // csnap算符
    CUDASparseMatrix csnap(double theta, int n, int cutoff) const;
    
    // 多Fock态SNAP算符
    CUDASparseMatrix multisnap(const std::vector<double>& params, int cutoff) const;
    
    // 多Fock态csnap算符
    CUDASparseMatrix multicsnap(const std::vector<double>& params, int cutoff) const;
    
    // SQR门
    CUDASparseMatrix sqr(const std::vector<double>& params, int cutoff) const;
    
    // 光子数读出支持门
    CUDASparseMatrix pnr(int max, int cutoff) const;
    
    // 指数SWAP算符
    CUDASparseMatrix eswap(double theta, int cutoff_a, int cutoff_b) const;
    
    // 受控单模压缩算符
    CUDASparseMatrix csq(Complex theta, int cutoff) const;
    
    // 多玻色子采样SNAP门
    CUDASparseMatrix cMultibosonSampling(int max, int cutoff) const;
    
    // 从矩阵创建门
    CUDASparseMatrix gateFromMatrix(const std::vector<std::vector<Complex>>& matrix) const;
    
    // 双模求和门
    CUDASparseMatrix sum(double scale, int cutoff_a, int cutoff_b) const;
    
    // 条件双模求和门
    CUDASparseMatrix csum(double scale, int cutoff_a, int cutoff_b) const;
    
    // Jaynes-Cummings门
    CUDASparseMatrix jc(double theta, double phi, int cutoff) const;
    
    // 反Jaynes-Cummings门
    CUDASparseMatrix ajc(double theta, double phi, int cutoff) const;
    
    // Rabi相互作用门
    CUDASparseMatrix rb(Complex theta, int cutoff) const;
    
private:
    // 辅助方法
    void parseOpExpr(const std::string& expr, std::map<int, std::string>& ops) const;
};

} // namespace gpu

#endif // BOSONIC_OPERATORS_GPU_H