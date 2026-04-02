#include "quantum_kernels.cuh"
#include <math.h>
#include <cuda_runtime.h>

namespace gpu {

__global__ void apply_hadamard_kernel(Complex* state, int qubit, int qubit_dim, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int qubit_pair = idx & qubit_mask;
    int base_idx = idx & ~qubit_mask;
    
    if (qubit_pair == 0) {
        int pair_idx = base_idx | qubit_mask;
        if (pair_idx < total_dim) {
            Complex psi0 = state[idx];
            Complex psi1 = state[pair_idx];
            
            Complex new_psi0 = psi0 + psi1;
            Complex new_psi1 = psi0 - psi1;
            
            state[idx] = new_psi0 * 0.70710678118654752440; // 1/sqrt(2)
            state[pair_idx] = new_psi1 * 0.70710678118654752440;
        }
    }
}

__global__ void apply_phase_s_kernel(Complex* state, int qubit, int qubit_dim, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    if (idx & qubit_mask) {
        Complex i(0.0, 1.0);
        state[idx] = state[idx] * i;
    }
}

__global__ void apply_rotation_z_kernel(Complex* state, int qubit, double angle, int qubit_dim, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    if (idx & qubit_mask) {
        double cos_half = cos(angle / 2.0);
        double sin_half = sin(angle / 2.0);
        Complex phase(cos_half, -sin_half);
        state[idx] = state[idx] * phase;
    } else {
        double cos_half = cos(angle / 2.0);
        double sin_half = sin(angle / 2.0);
        Complex phase(cos_half, sin_half);
        state[idx] = state[idx] * phase;
    }
}

__global__ void apply_rotation_x_kernel(Complex* state, int qubit, double angle, int qubit_dim, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int qubit_pair = idx & qubit_mask;
    int base_idx = idx & ~qubit_mask;
    
    if (qubit_pair == 0) {
        int pair_idx = base_idx | qubit_mask;
        if (pair_idx < total_dim) {
            Complex psi0 = state[idx];
            Complex psi1 = state[pair_idx];
            
            double cos_half = cos(angle / 2.0);
            double sin_half = sin(angle / 2.0);
            
            Complex new_psi0 = psi0 * cos_half - psi1 * Complex(0.0, sin_half);
            Complex new_psi1 = psi0 * Complex(0.0, sin_half) + psi1 * cos_half;
            
            state[idx] = new_psi0;
            state[pair_idx] = new_psi1;
        }
    }
}

__global__ void apply_pauli_z_kernel(Complex* state, int qubit, int qubit_dim, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    if (idx & qubit_mask) {
        state[idx] = state[idx] * Complex(-1.0, 0.0);
    }
}

__global__ void apply_pauli_x_kernel(Complex* state, int qubit, int qubit_dim, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int qubit_pair = idx & qubit_mask;
    int base_idx = idx & ~qubit_mask;
    
    if (qubit_pair == 0) {
        int pair_idx = base_idx | qubit_mask;
        if (pair_idx < total_dim) {
            Complex temp = state[idx];
            state[idx] = state[pair_idx];
            state[pair_idx] = temp;
        }
    }
}

__global__ void apply_phase_rotation_kernel(Complex* state, int qumode, double angle, int cutoff, int qubit_dim, int num_qumodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    if (idx >= total_dim) return;
    
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    
    Complex phase(cos(angle * qumode_idx * qumode_idx), sin(angle * qumode_idx * qumode_idx));
    state[idx] = state[idx] * phase;
}

__global__ void apply_displacement_kernel(Complex* state, int qumode, Complex alpha, int cutoff, int qubit_dim, int num_qumodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    if (idx >= total_dim) return;
    
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    
    double alpha_real = alpha.real;
    double alpha_imag = alpha.imag;
    double alpha_norm_sq = alpha_real * alpha_real + alpha_imag * alpha_imag;
    double exp_factor = exp(-alpha_norm_sq / 2.0);
    
    Complex sum(0.0, 0.0);
    
    for (int m = 0; m < cutoff; ++m) {
        int base_idx = idx - qumode_idx * qumode_stride + m * qumode_stride;
        if (base_idx >= 0 && base_idx < total_dim) {
            int n = qumode_idx;
            
            double sqrt_fact_ratio = 1.0;
            if (n > m) {
                for (int k = m + 1; k <= n; ++k) sqrt_fact_ratio *= sqrt((double)k);
                sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
            } else if (m > n) {
                for (int k = n + 1; k <= m; ++k) sqrt_fact_ratio *= sqrt((double)k);
                sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
            }
            
            int lower = (n < m) ? n : m;
            int upper = (n > m) ? n : m;
            int k = upper - lower;
            
            double laguerre = 0.0;
            double term = 1.0;
            double binom = 1.0;
            for(int i = 1; i <= lower; ++i) binom = binom * (upper - i + 1) / i;
            term = binom;
            laguerre += term;
            
            for(int j = 1; j <= lower; ++j) {
                term = term * (-alpha_norm_sq) * (lower - j + 1) / ((k + j) * j);
                laguerre += term;
            }
            
            Complex power_val(1.0, 0.0);
            if (n >= m) {
                for(int p = 0; p < k; ++p) power_val = power_val * alpha;
            } else {
                Complex minus_alpha_conj(-alpha_real, alpha_imag);
                for(int p = 0; p < k; ++p) power_val = power_val * minus_alpha_conj;
            }
            
            Complex coeff = sqrt_fact_ratio * exp_factor * power_val * Complex(laguerre, 0.0);
            sum = sum + coeff * state[base_idx];
        }
    }
    
    state[idx] = sum;
}

__global__ void apply_squeezing_kernel(Complex* state, int qumode, Complex xi, int cutoff, int qubit_dim, int num_qumodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    if (idx >= total_dim) return;
    
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    
    double xi_real = xi.real;
    double xi_imag = xi.imag;
    double r = sqrt(xi_real * xi_real + xi_imag * xi_imag);
    double phi = atan2(xi_imag, xi_real);
    
    double cosh_r = cosh(r);
    double sinh_r = sinh(r);
    
    Complex sum(0.0, 0.0);
    
    for (int m = 0; m < cutoff; ++m) {
        int base_idx = idx - qumode_idx * qumode_stride + m * qumode_stride;
        if (base_idx >= 0 && base_idx < total_dim) {
            int n = qumode_idx;
            
            double sqrt_fact_ratio = 1.0;
            if (n > m) {
                for (int k = m + 1; k <= n; ++k) sqrt_fact_ratio *= sqrt((double)k);
                sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
            } else if (m > n) {
                for (int k = n + 1; k <= m; ++k) sqrt_fact_ratio *= sqrt((double)k);
                sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
            }
            
            int lower = (n < m) ? n : m;
            int upper = (n > m) ? n : m;
            int k = upper - lower;
            
            double hermite = 0.0;
            double term = 1.0;
            double binom = 1.0;
            for(int i = 1; i <= lower; ++i) binom = binom * (upper - i + 1) / i;
            term = binom;
            hermite += term;
            
            for(int j = 1; j <= lower; ++j) {
                term = term * (-1.0) * (lower - j + 1) / ((k + j) * j);
                hermite += term;
            }
            
            double coeff_val = sqrt_fact_ratio * hermite;
            if (k % 2 == 1) {
                coeff_val *= -sinh_r;
            } else {
                coeff_val *= cosh_r;
            }
            
            Complex phase(cos(phi * k), sin(phi * k));
            Complex coeff = coeff_val * phase;
            
            sum = sum + coeff * state[base_idx];
        }
    }
    
    state[idx] = sum * (1.0 / cosh_r);
}

__global__ void apply_conditional_displacement_kernel(Complex* state, int qubit, int qumode, Complex alpha, int cutoff, int qubit_dim, int num_qumodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int qubit_state = (idx / (total_dim / qubit_dim)) & qubit_mask;
    
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    
    if (qubit_state != 0) {
        Complex sum(0.0, 0.0);
        
        for (int m = 0; m < cutoff; ++m) {
            int base_idx = idx - qumode_idx * qumode_stride + m * qumode_stride;
            if (base_idx >= 0 && base_idx < total_dim) {
                int n = qumode_idx;
                
                double sqrt_fact_ratio = 1.0;
                if (n > m) {
                    for (int k = m + 1; k <= n; ++k) sqrt_fact_ratio *= sqrt((double)k);
                    sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
                } else if (m > n) {
                    for (int k = n + 1; k <= m; ++k) sqrt_fact_ratio *= sqrt((double)k);
                    sqrt_fact_ratio = 1.0 / sqrt_fact_ratio;
                }
                
                int lower = (n < m) ? n : m;
                int upper = (n > m) ? n : m;
                int k = upper - lower;
                
                double laguerre = 0.0;
                double term = 1.0;
                double binom = 1.0;
                for(int i = 1; i <= lower; ++i) binom = binom * (upper - i + 1) / i;
                term = binom;
                laguerre += term;
                
                for(int j = 1; j <= lower; ++j) {
                    term = term * (-alpha.real * alpha.real - alpha.imag * alpha.imag) * (lower - j + 1) / ((k + j) * j);
                    laguerre += term;
                }
                
                Complex power_val(1.0, 0.0);
                if (n >= m) {
                    for(int p = 0; p < k; ++p) power_val = power_val * alpha;
                } else {
                    Complex minus_alpha_conj(-alpha.real, alpha.imag);
                    for(int p = 0; p < k; ++p) power_val = power_val * minus_alpha_conj;
                }
                
                double alpha_norm_sq = alpha.real * alpha.real + alpha.imag * alpha.imag;
                double exp_factor = exp(-alpha_norm_sq / 2.0);
                
                Complex coeff = sqrt_fact_ratio * exp_factor * power_val * Complex(laguerre, 0.0);
                sum = sum + coeff * state[base_idx];
            }
        }
        
        state[idx] = sum;
    }
}

__global__ void apply_jaynes_cummings_kernel(Complex* state, int qubit, int qumode, double angle, int cutoff, int qubit_dim, int num_qumodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int qubit_state = (idx / (total_dim / qubit_dim)) & qubit_mask;
    
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    
    if (qubit_state == 0 && qumode_idx < cutoff - 1) {
        int excited_idx = idx | qubit_mask;
        int next_qumode_idx = idx + qumode_stride;
        
        if (excited_idx < total_dim && next_qumode_idx < total_dim) {
            Complex psi_gn = state[idx];
            Complex psi_en_minus_1 = state[excited_idx];
            
            double sqrt_val = sqrt((double)(qumode_idx + 1));
            double cos_angle = cos(angle * sqrt_val);
            double sin_angle = sin(angle * sqrt_val);
            
            Complex new_psi_gn = psi_gn * Complex(cos_angle, 0.0) - psi_en_minus_1 * Complex(0.0, sin_angle);
            Complex new_psi_en_minus_1 = psi_gn * Complex(0.0, sin_angle) + psi_en_minus_1 * Complex(cos_angle, 0.0);
            
            state[idx] = new_psi_gn;
            state[excited_idx] = new_psi_en_minus_1;
        }
    }
}

__global__ void apply_beam_splitter_kernel(Complex* state, int qumode1, int qumode2, double angle, int cutoff, int qubit_dim, int num_qumodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = qubit_dim;
    for (int i = 0; i < num_qumodes; ++i) {
        total_dim *= cutoff;
    }
    
    if (idx >= total_dim) return;
    
    int qumode1_stride = 1;
    for (int i = 0; i < qumode1; ++i) {
        qumode1_stride *= cutoff;
    }
    
    int qumode2_stride = 1;
    for (int i = 0; i < qumode2; ++i) {
        qumode2_stride *= cutoff;
    }
    
    int qumode1_idx = (idx / qumode1_stride) % cutoff;
    int qumode2_idx = (idx / qumode2_stride) % cutoff;
    
    Complex sum(0.0, 0.0);
    
    for (int m1 = 0; m1 < cutoff; ++m1) {
        for (int m2 = 0; m2 < cutoff; ++m2) {
            int base_idx = idx - qumode1_idx * qumode1_stride - qumode2_idx * qumode2_stride + m1 * qumode1_stride + m2 * qumode2_stride;
            if (base_idx >= 0 && base_idx < total_dim) {
                int n1 = qumode1_idx;
                int n2 = qumode2_idx;
                
                double coeff = 0.0;
                for (int k = max(0, m1 - n2); k <= min(n1, m2); ++k) {
                    double sqrt_fact = 1.0;
                    for (int i = 0; i < k; ++i) sqrt_fact *= sqrt((double)(n1 - i) * (m2 - i) / ((i + 1.0) * (n2 - m2 + i + 1.0)));
                    
                    double cos_pow = pow(cos(angle), n1 + m2 - 2 * k);
                    double sin_pow = pow(sin(angle), m1 + n2 - 2 * k);
                    
                    int sign = ((k % 2 == 0) ?1 : -1);
                    coeff += sign * sqrt_fact * cos_pow * sin_pow;
                }
                
                sum = sum + coeff * state[base_idx];
            }
        }
    }
    
    state[idx] = sum;
}

} // namespace gpu