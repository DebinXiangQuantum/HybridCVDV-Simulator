#ifndef QUANTUM_KERNELS_CUH
#define QUANTUM_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace gpu {

struct Complex {
    double real;
    double imag;
    
    __host__ __device__ Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}
    
    __host__ __device__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    __host__ __device__ Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    __host__ __device__ Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag, 
                       real * other.imag + imag * other.real);
    }
    
    __host__ __device__ Complex operator*(double scalar) const {
        return Complex(real * scalar, imag * scalar);
    }
    
    __host__ __device__ friend Complex operator*(double scalar, const Complex& c) {
        return Complex(c.real * scalar, c.imag * scalar);
    }
};

__global__ void apply_hadamard_kernel(Complex* state, int qubit, int qubit_dim, int qumode_dim);
__global__ void apply_phase_s_kernel(Complex* state, int qubit, int qubit_dim, int qumode_dim);
__global__ void apply_rotation_z_kernel(Complex* state, int qubit, double angle, int qubit_dim, int qumode_dim);
__global__ void apply_rotation_x_kernel(Complex* state, int qubit, double angle, int qubit_dim, int qumode_dim);
__global__ void apply_pauli_z_kernel(Complex* state, int qubit, int qubit_dim, int qumode_dim);
__global__ void apply_pauli_x_kernel(Complex* state, int qubit, int qubit_dim, int qumode_dim);
__global__ void apply_phase_rotation_kernel(Complex* state, int qumode, double angle, int qubit_dim, int cutoff, int num_qumodes);
__global__ void apply_displacement_kernel(Complex* state, int qumode, Complex alpha, int cutoff, int qubit_dim, int num_qumodes);
__global__ void apply_squeezing_kernel(Complex* state, int qumode, Complex xi, int cutoff, int qubit_dim, int num_qumodes);
__global__ void apply_conditional_displacement_kernel(Complex* state, int qubit, int qumode, Complex alpha, int cutoff, int qubit_dim, int num_qumodes);
__global__ void apply_jaynes_cummings_kernel(Complex* state, int qubit, int qumode, double angle, int cutoff, int qubit_dim, int num_qumodes);
__global__ void apply_beam_splitter_kernel(Complex* state, int qumode1, int qumode2, double angle, int cutoff, int qubit_dim, int num_qumodes);

} // namespace gpu

#endif // QUANTUM_KERNELS_CUH