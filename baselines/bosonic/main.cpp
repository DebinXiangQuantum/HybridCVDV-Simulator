#include <iostream>
#include <complex>
#include <vector>
#include "cpu/operators.h"
#include "gpu/operators.cuh"

using namespace std;

int main() {
    cout << "Testing CPU version..." << endl;
    
    try {
        // 测试CPU版本的CVOperators类
        CVOperatorsCPU cv;
        int cutoff = 10;
        
        // 创建位移算符
        auto d = cv.d(complex<double>(1.0, 0.0), cutoff);
        cout << "Created displacement operator d." << endl;
        
        // 创建压缩算符
        auto s = cv.s(complex<double>(0.5, 0.0), cutoff);
        cout << "Created squeezing operator s." << endl;
        
        // 获取数算符
        auto n = cv.getN(cutoff);
        cout << "Created number operator N." << endl;
        
        // 获取湮灭算符
        auto a = cv.getA(cutoff);
        cout << "Created annihilation operator a." << endl;
        
        // 获取产生算符
        auto ad = cv.getAd(cutoff);
        cout << "Created creation operator ad." << endl;
        
        // 创建分束器算符
        auto bs = cv.bs(complex<double>(0.5, 0.0), cutoff, cutoff);
        cout << "Created beam splitter operator bs." << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    cout << "\nTesting GPU version..." << endl;
    
    try {
        // 测试GPU版本的CVOperators类
        gpu::CVOperatorsGPU cv_gpu;
        int cutoff = 10;
        
        // 创建位移算符
        auto d_gpu = cv_gpu.d(gpu::Complex(1.0, 0.0), cutoff);
        cout << "Created displacement operator d (GPU)." << endl;
        
        // 创建压缩算符
        auto s_gpu = cv_gpu.s(gpu::Complex(0.5, 0.0), cutoff);
        cout << "Created squeezing operator s (GPU)." << endl;
        
        // 获取数算符
        auto n_gpu = cv_gpu.getN(cutoff);
        cout << "Created number operator N (GPU)." << endl;
        
        // 获取湮灭算符
        auto a_gpu = cv_gpu.getA(cutoff);
        cout << "Created annihilation operator a (GPU)." << endl;
        
        // 获取产生算符
        auto ad_gpu = cv_gpu.getAd(cutoff);
        cout << "Created creation operator ad (GPU)." << endl;
        
        // 创建分束器算符
        auto bs_gpu = cv_gpu.bs(gpu::Complex(0.5, 0.0), cutoff, cutoff);
        cout << "Created beam splitter operator bs (GPU)." << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    cout << "\nAll tests completed!" << endl;
    return 0;
}