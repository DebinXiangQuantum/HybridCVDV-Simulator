#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

__global__ void hadamard_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits) * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int partner_idx = idx ^ qubit_mask;
    
    if (idx < partner_idx) {
        cuDoubleComplex psi_0 = state[idx];
        cuDoubleComplex psi_1 = state[partner_idx];
        
        cuDoubleComplex inv_sqrt2 = make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0);
        
        state[idx] = cuCmul(inv_sqrt2, cuCadd(psi_0, psi_1));
        state[partner_idx] = cuCmul(inv_sqrt2, cuCadd(psi_0, cuCmul(make_cuDoubleComplex(-1.0, 0.0), psi_1)));
    }
}

__global__ void phase_s_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits) * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_value = (idx / qumode_dim) >> qubit & 1;
    
    if (qubit_value == 1) {
        cuDoubleComplex i = make_cuDoubleComplex(0.0, 1.0);
        state[idx] = cuCmul(i, state[idx]);
    }
}

__global__ void rotation_z_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim, double angle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits) * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_value = (idx / qumode_dim) >> qubit & 1;
    
    if (qubit_value == 1) {
        cuDoubleComplex phase = make_cuDoubleComplex(cos(angle / 2.0), sin(angle / 2.0));
        state[idx] = cuCmul(phase, state[idx]);
    }
}

__global__ void rotation_x_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim, double angle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits) * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int partner_idx = idx ^ qubit_mask;
    
    if (idx < partner_idx) {
        cuDoubleComplex psi_0 = state[idx];
        cuDoubleComplex psi_1 = state[partner_idx];
        
        double cos_half = cos(angle / 2.0);
        double sin_half = sin(angle / 2.0);
        
        cuDoubleComplex new_psi_0 = cuCadd(
            make_cuDoubleComplex(cos_half, 0.0),
            cuCmul(make_cuDoubleComplex(-sin_half, 0.0), psi_1)
        );
        
        cuDoubleComplex new_psi_1 = cuCadd(
            make_cuDoubleComplex(-sin_half, 0.0),
            cuCmul(make_cuDoubleComplex(cos_half, 0.0), psi_1)
        );
        
        state[idx] = new_psi_0;
        state[partner_idx] = new_psi_1;
    }
}

__global__ void pauli_x_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits) * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int partner_idx = idx ^ qubit_mask;
    
    if (idx < partner_idx) {
        cuDoubleComplex temp = state[idx];
        state[idx] = state[partner_idx];
        state[partner_idx] = temp;
    }
}

__global__ void pauli_z_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits) * qumode_dim;
    
    if (idx >= total_dim) return;
    
    int qubit_value = (idx / qumode_dim) >> qubit & 1;
    
    if (qubit_value == 1) {
        state[idx] = cuCmul(make_cuDoubleComplex(-1.0, 0.0), state[idx]);
    }
}

__global__ void phase_rotation_kernel(cuDoubleComplex* state, int qumode, int num_qubits, int num_qumodes, int cutoff, double angle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits);
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    if (idx >= total_dim * qumode_dim) return;
    
    int qumode_idx = idx % cutoff;
    int qumode_shift = qumode * cutoff;
    int base_idx = (idx / cutoff) * qumode_dim + qumode_shift + qumode_idx;
    
    cuDoubleComplex phase = make_cuDoubleComplex(cos(angle * qumode_idx), sin(angle * qumode_idx));
    state[base_idx] = cuCmul(phase, state[base_idx]);
}

__global__ void displacement_kernel(cuDoubleComplex* state, int qumode, int num_qubits, int num_qumodes, int cutoff, double alpha_real, double alpha_imag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits);
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    if (idx >= total_dim * qumode_dim) return;
    
    int qumode_idx = idx % cutoff;
    int qumode_shift = qumode * cutoff;
    int base_idx = (idx / cutoff) * qumode_dim + qumode_shift + qumode_idx;
    
    double exp_factor = exp(-(alpha_real * alpha_real + alpha_imag * alpha_imag) / 2.0);
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int m = 0; m < cutoff; ++m) {
        int m_idx = (idx / cutoff) * qumode_dim + qumode_shift + m;
        
        int min_nm = (qumode_idx < m) ? qumode_idx : m;
        int max_nm = (qumode_idx > m) ? qumode_idx : m;
        int diff = max_nm - min_nm;
        
        double sqrt_fact_ratio = 1.0;
        for (int k = min_nm + 1; k <= max_nm; ++k) {
            double sqrt_k = sqrt((double)k);
            sqrt_fact_ratio /= sqrt_k;
        }
        
        cuDoubleComplex power_term = make_cuDoubleComplex(1.0, 0.0);
        if (qumode_idx >= m) {
            for (int k = 0; k < diff; ++k) {
                power_term = cuCmul(power_term, make_cuDoubleComplex(alpha_real, alpha_imag));
            }
        } else {
            for (int k = 0; k < diff; ++k) {
                power_term = cuCmul(power_term, make_cuDoubleComplex(-alpha_real, alpha_imag));
            }
        }
        
        double laguerre = 0.0;
        double x = alpha_real * alpha_real + alpha_imag * alpha_imag;
        for (int j = 0; j <= min_nm; ++j) {
            double term = 1.0;
            double binom = 1.0;
            for (int i = 0; i < min_nm - j; ++i) {
                binom = binom * (max_nm - i) / (i + 1);
            }
            term = binom;
            
            double x_pow_j = 1.0;
            for (int k = 0; k < j; ++k) {
                x_pow_j *= x;
            }
            
            double fact_j = 1.0;
            for (int k = 1; k <= j; ++k) {
                fact_j *= k;
            }
            
            term *= x_pow_j / fact_j;
            if (j % 2 == 1) term = -term;
            
            laguerre += term;
        }
        
        cuDoubleComplex d_nm = make_cuDoubleComplex(
            exp_factor * sqrt_fact_ratio * laguerre * cuCreal(power_term),
            exp_factor * sqrt_fact_ratio * laguerre * cuCimag(power_term)
        );
        
        sum = cuCadd(sum, cuCmul(d_nm, state[m_idx]));
    }
    
    state[base_idx] = sum;
}

__global__ void squeezing_kernel(cuDoubleComplex* state, int qumode, int num_qubits, int num_qumodes, int cutoff, double r_real, double r_imag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits);
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    if (idx >= total_dim * qumode_dim) return;
    
    int qumode_idx = idx % cutoff;
    int qumode_shift = qumode * cutoff;
    int base_idx = (idx / cutoff) * qumode_dim + qumode_shift + qumode_idx;
    
    double r_mag = sqrt(r_real * r_real + r_imag * r_imag);
    double r_phase = atan2(r_imag, r_real);
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int m = 0; m < cutoff; ++m) {
        int m_idx = (idx / cutoff) * qumode_dim + qumode_shift + m;
        
        int min_nm = (qumode_idx < m) ? qumode_idx : m;
        int max_nm = (qumode_idx > m) ? qumode_idx : m;
        int diff = max_nm - min_nm;
        
        double sqrt_fact_ratio = 1.0;
        for (int k = min_nm + 1; k <= max_nm; ++k) {
            double sqrt_k = sqrt((double)k);
            sqrt_fact_ratio /= sqrt_k;
        }
        
        double tanh_r = tanh(r_mag);
        double sech_r = 1.0 / cosh(r_mag);
        
        double s_nm = 0.0;
        if ((min_nm + diff) % 2 == 0) {
            double term = sqrt_fact_ratio * pow(sech_r, min_nm + 1);
            
            double hermite = 0.0;
            for (int j = 0; j <= min_nm / 2; ++j) {
                double h_term = 1.0;
                double fact_j = 1.0;
                for (int k = 1; k <= j; ++k) {
                    fact_j *= k;
                }
                
                double fact_min_nm_2j = 1.0;
                for (int k = 1; k <= min_nm - 2 * j; ++k) {
                    fact_min_nm_2j *= k;
                }
                
                double sqrt_min_nm_plus_1 = sqrt((double)(min_nm + 1));
                h_term = pow(-1.0, j) * pow(2.0 * sqrt_min_nm_plus_1 * tanh_r, min_nm - 2 * j) / (fact_j * fact_min_nm_2j);
                hermite += h_term;
            }
            
            s_nm = term * hermite * pow(tanh_r, diff);
        }
        
        cuDoubleComplex phase = make_cuDoubleComplex(cos(r_phase * diff), sin(r_phase * diff));
        cuDoubleComplex s_nm_complex = make_cuDoubleComplex(s_nm * cuCreal(phase), s_nm * cuCimag(phase));
        
        sum = cuCadd(sum, cuCmul(s_nm_complex, state[m_idx]));
    }
    
    state[base_idx] = sum;
}

__global__ void jaynes_cummings_kernel(cuDoubleComplex* state, int qubit, int qumode, int num_qubits, int num_qumodes, int cutoff, double angle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits);
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    if (idx >= total_dim * qumode_dim) return;
    
    int qubit_value = (idx / qumode_dim) >> qubit & 1;
    int qumode_idx = idx % cutoff;
    int qumode_shift = qumode * cutoff;
    
    if (qubit_value == 1 && qumode_idx < cutoff - 1) {
        int excited_idx = (idx / cutoff) * qumode_dim + qumode_shift + qumode_idx;
        int ground_idx = (idx / cutoff) * qumode_dim + qumode_shift + qumode_idx + 1;
        
        cuDoubleComplex psi_g = state[ground_idx];
        
        double sqrt_qumode_idx_plus_1 = sqrt((double)(qumode_idx + 1));
        double cos_angle = cos(angle * sqrt_qumode_idx_plus_1);
        double sin_angle = sin(angle * sqrt_qumode_idx_plus_1);
        
        cuDoubleComplex new_psi_e = cuCadd(
            make_cuDoubleComplex(cos_angle, 0.0),
            cuCmul(make_cuDoubleComplex(-sin_angle, 0.0), psi_g)
        );
        
        cuDoubleComplex new_psi_g = cuCadd(
            make_cuDoubleComplex(sin_angle, 0.0),
            cuCmul(make_cuDoubleComplex(cos_angle, 0.0), psi_g)
        );
        
        state[excited_idx] = new_psi_e;
        state[ground_idx] = new_psi_g;
    }
}

__global__ void beam_splitter_kernel(cuDoubleComplex* state, int qumode1, int qumode2, int num_qubits, int num_qumodes, int cutoff, double angle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits);
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    if (idx >= total_dim * qumode_dim) return;
    
    int qumode1_idx = (idx / cutoff) % cutoff;
    int qumode2_idx = idx % cutoff;
    
    int qumode1_shift = qumode1 * cutoff;
    int qumode2_shift = qumode2 * cutoff;
    
    int base_idx = (idx / (cutoff * cutoff)) * qumode_dim + qumode1_shift + qumode1_idx + qumode2_shift + qumode2_idx;
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int n2 = 0; n2 < cutoff; ++n2) {
        int partner_idx = (idx / (cutoff * cutoff)) * qumode_dim + qumode1_shift + qumode1_idx + qumode2_shift + n2;
        
        double bs_coeff = 0.0;
        if (n2 <= qumode2_idx) {
            double fact_qumode1_idx = 1.0;
            for (int k = 1; k <= qumode1_idx; ++k) {
                fact_qumode1_idx *= k;
            }
            
            double fact_qumode2_idx = 1.0;
            for (int k = 1; k <= qumode2_idx; ++k) {
                fact_qumode2_idx *= k;
            }
            
            double fact_n2 = 1.0;
            for (int k = 1; k <= n2; ++k) {
                fact_n2 *= k;
            }
            
            double fact_qumode1_idx_plus_n2 = 1.0;
            for (int k = 1; k <= qumode1_idx + n2; ++k) {
                fact_qumode1_idx_plus_n2 *= k;
            }
            
            double fact_qumode2_idx_minus_n2 = 1.0;
            for (int k = 1; k <= qumode2_idx - n2; ++k) {
                fact_qumode2_idx_minus_n2 *= k;
            }
            
            double sqrt_arg = fact_qumode1_idx * fact_qumode2_idx / (fact_n2 * fact_qumode1_idx_plus_n2 * fact_qumode2_idx_minus_n2);
            double sqrt_term = sqrt(sqrt_arg);
            
            double cos_pow = pow(cos(angle), qumode1_idx + n2);
            double sin_pow = pow(sin(angle), qumode2_idx - n2);
            
            bs_coeff = sqrt_term * cos_pow * sin_pow;
            
            if ((qumode2_idx - n2) % 2 == 1) {
                bs_coeff = -bs_coeff;
            }
        }
        
        sum = cuCadd(sum, cuCmul(make_cuDoubleComplex(bs_coeff, 0.0), state[partner_idx]));
    }
    
    state[base_idx] = sum;
}

__global__ void conditional_displacement_kernel(cuDoubleComplex* state, int qubit, int qumode, int num_qubits, int num_qumodes, int cutoff, double alpha_real, double alpha_imag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = (1 << num_qubits);
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    if (idx >= total_dim * qumode_dim) return;
    
    int qubit_value = (idx / qumode_dim) >> qubit & 1;
    int qumode_idx = idx % cutoff;
    int qumode_shift = qumode * cutoff;
    int base_idx = (idx / cutoff) * qumode_dim + qumode_shift + qumode_idx;
    
    if (qubit_value == 1) {
        double exp_factor = exp(-(alpha_real * alpha_real + alpha_imag * alpha_imag) / 2.0);
        
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (int m = 0; m < cutoff; ++m) {
            int m_idx = (idx / cutoff) * qumode_dim + qumode_shift + m;
            
            int min_nm = (qumode_idx < m) ? qumode_idx : m;
            int max_nm = (qumode_idx > m) ? qumode_idx : m;
            int diff = max_nm - min_nm;
            
            double sqrt_fact_ratio = 1.0;
            for (int k = min_nm + 1; k <= max_nm; ++k) {
                double sqrt_k = sqrt((double)k);
                sqrt_fact_ratio /= sqrt_k;
            }
            
            cuDoubleComplex power_term = make_cuDoubleComplex(1.0, 0.0);
            if (qumode_idx >= m) {
                for (int k = 0; k < diff; ++k) {
                    power_term = cuCmul(power_term, make_cuDoubleComplex(alpha_real, alpha_imag));
                }
            } else {
                for (int k = 0; k < diff; ++k) {
                    power_term = cuCmul(power_term, make_cuDoubleComplex(-alpha_real, alpha_imag));
                }
            }
            
            double laguerre = 0.0;
            double x = alpha_real * alpha_real + alpha_imag * alpha_imag;
            for (int j = 0; j <= min_nm; ++j) {
                double term = 1.0;
                double binom = 1.0;
                for (int i = 0; i < min_nm - j; ++i) {
                    binom = binom * (max_nm - i) / (i + 1);
                }
                term = binom;
                
                double x_pow_j = 1.0;
                for (int k = 0; k < j; ++k) {
                    x_pow_j *= x;
                }
                
                double fact_j = 1.0;
                for (int k = 1; k <= j; ++k) {
                    fact_j *= k;
                }
                
                term *= x_pow_j / fact_j;
                if (j % 2 == 1) term = -term;
                
                laguerre += term;
            }
            
            cuDoubleComplex d_nm = make_cuDoubleComplex(
                exp_factor * sqrt_fact_ratio * laguerre * cuCreal(power_term),
                exp_factor * sqrt_fact_ratio * laguerre * cuCimag(power_term)
            );
            
            sum = cuCadd(sum, cuCmul(d_nm, state[m_idx]));
        }
        
        state[base_idx] = sum;
    }
}
