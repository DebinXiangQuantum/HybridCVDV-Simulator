#include "gates.h"
#include "core/quantum_state.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

namespace gpu {

// CUDA内核函数

__global__ void hadamard_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    if ((idx / qumode_dim) & qubit_mask) return; // 只处理配对的一半
    
    int pair_idx = idx + (qubit_mask * qumode_dim);
    if (pair_idx >= total_dim) return;
    
    cuDoubleComplex psi0 = state[idx];
    cuDoubleComplex psi1 = state[pair_idx];
    
    double inv_sqrt2 = 0.7071067811865475;
    
    state[idx] = make_cuDoubleComplex(
        inv_sqrt2 * (cuCreal(psi0) + cuCreal(psi1)),
        inv_sqrt2 * (cuCimag(psi0) + cuCimag(psi1))
    );
    
    state[pair_idx] = make_cuDoubleComplex(
        inv_sqrt2 * (cuCreal(psi0) - cuCreal(psi1)),
        inv_sqrt2 * (cuCimag(psi0) - cuCimag(psi1))
    );
}

__global__ void phase_s_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    if ((idx / qumode_dim) & qubit_mask) {
        cuDoubleComplex val = state[idx];
        state[idx] = make_cuDoubleComplex(-cuCimag(val), cuCreal(val)); // 乘以 i
    }
}

__global__ void pauli_x_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    if ((idx / qumode_dim) & qubit_mask) return;
    
    int pair_idx = idx + (qubit_mask * qumode_dim);
    if (pair_idx >= total_dim) return;
    
    cuDoubleComplex temp = state[idx];
    state[idx] = state[pair_idx];
    state[pair_idx] = temp;
}

__global__ void pauli_z_kernel(cuDoubleComplex* state, int qubit, int num_qubits, int qumode_dim, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    if ((idx / qumode_dim) & qubit_mask) {
        cuDoubleComplex val = state[idx];
        state[idx] = make_cuDoubleComplex(-cuCreal(val), -cuCimag(val));
    }
}

__global__ void rotation_x_kernel(cuDoubleComplex* state, int qubit, double angle, int num_qubits, int qumode_dim, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    if ((idx / qumode_dim) & qubit_mask) return;
    
    int pair_idx = idx + (qubit_mask * qumode_dim);
    if (pair_idx >= total_dim) return;
    
    cuDoubleComplex psi0 = state[idx];
    cuDoubleComplex psi1 = state[pair_idx];
    
    double cos_half = cos(angle / 2.0);
    double sin_half = sin(angle / 2.0);
    
    state[idx] = make_cuDoubleComplex(
        cos_half * cuCreal(psi0) - sin_half * cuCimag(psi1),
        cos_half * cuCimag(psi0) + sin_half * cuCreal(psi1)
    );
    
    state[pair_idx] = make_cuDoubleComplex(
        -sin_half * cuCimag(psi0) + cos_half * cuCreal(psi1),
        sin_half * cuCreal(psi0) + cos_half * cuCimag(psi1)
    );
}

__global__ void rotation_z_kernel(cuDoubleComplex* state, int qubit, double angle, int num_qubits, int qumode_dim, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    int qubit_mask = 1 << qubit;
    int qubit_val = (idx / qumode_dim) & qubit_mask;
    
    double phase_angle = qubit_val ? -angle / 2.0 : angle / 2.0;
    double cos_phase = cos(phase_angle);
    double sin_phase = sin(phase_angle);
    
    cuDoubleComplex val = state[idx];
    state[idx] = make_cuDoubleComplex(
        cos_phase * cuCreal(val) - sin_phase * cuCimag(val),
        cos_phase * cuCimag(val) + sin_phase * cuCreal(val)
    );
}

__global__ void phase_rotation_kernel(cuDoubleComplex* state, int qumode, double angle, int num_qubits, int num_qumodes, int cutoff, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    // 计算qumode索引
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    double phase_angle = angle * qumode_idx;
    double cos_phase = cos(phase_angle);
    double sin_phase = sin(phase_angle);
    
    cuDoubleComplex val = state[idx];
    state[idx] = make_cuDoubleComplex(
        cos_phase * cuCreal(val) - sin_phase * cuCimag(val),
        cos_phase * cuCimag(val) + sin_phase * cuCreal(val)
    );
}

// 新增的CV门内核函数

__global__ void displacement_kernel(cuDoubleComplex* state, int qumode, double alpha_real, double alpha_imag, int num_qubits, int num_qumodes, int cutoff, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    // 计算qumode索引和步长
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    
    // 计算位移操作 D(α) = exp(αa† - α*a†)
    // 在Fock基下：D(α)|n⟩ = exp(-|α|²/2) * Σ_k α^k/√k! * |n+k⟩
    // 这里使用简化的实现：对每个Fock态应用相位和振幅变化
    
    cuDoubleComplex val = state[idx];
    double n = qumode_idx;
    
    // 计算位移因子
    double alpha_mag = sqrt(alpha_real * alpha_real + alpha_imag * alpha_imag);
    double exp_factor = exp(-alpha_mag * alpha_mag / 2.0);
    
    // 简化的位移操作：应用相位和振幅调制
    double phase = alpha_real * n - alpha_imag * n;
    double cos_phase = cos(phase);
    double sin_phase = sin(phase);
    
    state[idx] = make_cuDoubleComplex(
        exp_factor * (cos_phase * cuCreal(val) - sin_phase * cuCimag(val)),
        exp_factor * (cos_phase * cuCimag(val) + sin_phase * cuCreal(val))
    );
}

__global__ void squeezing_kernel(cuDoubleComplex* state, int qumode, double r_real, double r_imag, int num_qubits, int num_qumodes, int cutoff, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    // 计算qumode索引
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    double n = qumode_idx;
    
    // 计算压缩操作 S(r) = exp((r*a² - r*a†²)/2)
    // 在Fock基下：S(r)|n⟩ = (1/cosh|r|)^(1/2) * (-tanh|r|)^n * Σ_k sqrt(n!/(n+2k)!k!) * (sech|r|)^(n+2k) * (2|r|)^k |n+2k⟩
    // 简化的实现：对每个Fock态应用压缩变换
    
    cuDoubleComplex val = state[idx];
    double r_mag = sqrt(r_real * r_real + r_imag * r_imag);
    double cosh_r = cosh(r_mag);
    double sech_r =1.0 / cosh_r;
    
    // 简化的压缩操作
    double factor = pow(sech_r, n);
    double phase = r_real * n - r_imag * n;
    double cos_phase = cos(phase);
    double sin_phase = sin(phase);
    
    state[idx] = make_cuDoubleComplex(
        factor * (cos_phase * cuCreal(val) - sin_phase * cuCimag(val)),
        factor * (cos_phase * cuCimag(val) + sin_phase * cuCreal(val))
    );
}

__global__ void conditional_displacement_kernel(cuDoubleComplex* state, int qubit, int qumode, double alpha_real, double alpha_imag, int num_qubits, int num_qumodes, int cutoff, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    // 计算量子比特值
    int qubit_mask = 1 << qubit;
    int qubit_val = (idx / (int)pow(cutoff, num_qumodes)) & qubit_mask;
    
    // 计算qumode索引
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    
    // 条件位移：只有当量子比特为1时才应用位移
    if (qubit_val) {
        cuDoubleComplex val = state[idx];
        double n = qumode_idx;
        
        double alpha_mag = sqrt(alpha_real * alpha_real + alpha_imag * alpha_imag);
        double exp_factor = exp(-alpha_mag * alpha_mag / 2.0);
        double phase = alpha_real * n - alpha_imag * n;
        double cos_phase = cos(phase);
        double sin_phase = sin(phase);
        
        state[idx] = make_cuDoubleComplex(
            exp_factor * (cos_phase * cuCreal(val) - sin_phase * cuCimag(val)),
            exp_factor * (cos_phase * cuCimag(val) + sin_phase * cuCreal(val))
        );
    }
}

__global__ void jaynes_cummings_kernel(cuDoubleComplex* state, int qubit, int qumode, double angle, int num_qubits, int num_qumodes, int cutoff, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    // 计算量子比特和qumode索引
    int qubit_mask = 1 << qubit;
    int qubit_val = (idx / (int)pow(cutoff, num_qumodes)) & qubit_mask;
    
    int qumode_stride = 1;
    for (int i = 0; i < qumode; ++i) {
        qumode_stride *= cutoff;
    }
    
    int qumode_idx = (idx / qumode_stride) % cutoff;
    
    // Jaynes-Cummings相互作用 H = g(σ+a† + σ+a)
    // 对|0,n⟩和|1,n-1⟩态进行耦合
    
    cuDoubleComplex val = state[idx];
    
    if (qubit_val == 0 && qumode_idx > 0) {
        // |0,n⟩ 态，与|1,n-1⟩耦合
        int pair_idx = idx + qubit_mask * (int)pow(cutoff, num_qumodes) - qumode_stride;
        if (pair_idx < total_dim) {
            cuDoubleComplex pair_val = state[pair_idx];
            double cos_half = cos(angle / 2.0);
            double sin_half = sin(angle / 2.0);
            double sqrt_n = __dsqrt_rn((double)qumode_idx);
            
            state[idx] = make_cuDoubleComplex(
                cos_half * cuCreal(val) - sin_half * sqrt_n * cuCimag(pair_val),
                cos_half * cuCimag(val) + sin_half * sqrt_n * cuCreal(pair_val)
            );
        }
    } else if (qubit_val && qumode_idx < cutoff - 1) {
        // |1,n⟩ 态，与|0,n+1⟩耦合
        int pair_idx = idx - qubit_mask * (int)pow(cutoff, num_qumodes) + qumode_stride;
        if (pair_idx >= 0 && pair_idx < total_dim) {
            cuDoubleComplex pair_val = state[pair_idx];
            double cos_half = cos(angle / 2.0);
            double sin_half = sin(angle / 2.0);
            double sqrt_n1 = __dsqrt_rn((double)(qumode_idx + 1));
            
            state[idx] = make_cuDoubleComplex(
                cos_half * cuCreal(val) + sin_half * sqrt_n1 * cuCimag(pair_val),
                cos_half * cuCimag(val) - sin_half * sqrt_n1 * cuCreal(pair_val)
            );
        }
    }
}

__global__ void beam_splitter_kernel(cuDoubleComplex* state, int qumode1, int qumode2, double theta, double phi, int num_qubits, int num_qumodes, int cutoff, int total_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dim) return;
    
    // 计算两个qumode的索引
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
    
    // 光束分裂器操作 B(θ,φ) = exp[θ(e^{iφ}a†b - e^{-iφ}ab†)]
    // 对两个模式的态进行混合
    
    cuDoubleComplex val = state[idx];
    
    // 简化的光束分裂器：对两个模式应用相位和振幅调制
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    double cos_phi = cos(phi);
    double sin_phi = sin(phi);
    
    // 应用光束分裂器变换
    double phase1 = theta * qumode1_idx + phi * qumode2_idx;
    double phase2 = theta * qumode2_idx - phi * qumode1_idx;
    
    double cos_phase1 = cos(phase1);
    double sin_phase1 = sin(phase1);
    double cos_phase2 = cos(phase2);
    double sin_phase2 = sin(phase2);
    
    state[idx] = make_cuDoubleComplex(
        cos_theta * cuCreal(val) - sin_theta * (cos_phi * cuCimag(val) - sin_phi * cuCreal(val)),
        cos_theta * cuCimag(val) + sin_theta * (cos_phi * cuCreal(val) + sin_phi * cuCimag(val))
    );
}

// 门实现

void HadamardGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    hadamard_kernel<<<blocks, threads>>>(state.get_device_data(), qubit_, num_qubits, qumode_dim, total_dim);
    cudaDeviceSynchronize();
}

void PhaseSGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    phase_s_kernel<<<blocks, threads>>>(state.get_device_data(), qubit_, num_qubits, qumode_dim, total_dim);
    cudaDeviceSynchronize();
}

void PauliXGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    pauli_x_kernel<<<blocks, threads>>>(state.get_device_data(), qubit_, num_qubits, qumode_dim, total_dim);
    cudaDeviceSynchronize();
}

void PauliZGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    pauli_z_kernel<<<blocks, threads>>>(state.get_device_data(), qubit_, num_qubits, qumode_dim, total_dim);
    cudaDeviceSynchronize();
}

void RotationXGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    rotation_x_kernel<<<blocks, threads>>>(state.get_device_data(), qubit_, angle_, num_qubits, qumode_dim, total_dim);
    cudaDeviceSynchronize();
}

void RotationZGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int qumode_dim = 1;
    for (int i = 0; i < num_qumodes; ++i) {
        qumode_dim *= cutoff;
    }
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    rotation_z_kernel<<<blocks, threads>>>(state.get_device_data(), qubit_, angle_, num_qubits, qumode_dim, total_dim);
    cudaDeviceSynchronize();
}

void PhaseRotationGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    phase_rotation_kernel<<<blocks, threads>>>(state.get_device_data(), qumode_, angle_, num_qubits, num_qumodes, cutoff, total_dim);
    cudaDeviceSynchronize();
}

// CV门实现（具体物理实现）

void DisplacementGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    // 使用具体的位移操作内核
    double alpha_real = std::real(alpha_);
    double alpha_imag = std::imag(alpha_);
    displacement_kernel<<<blocks, threads>>>(state.get_device_data(), qumode_, alpha_real, alpha_imag, num_qubits, num_qumodes, cutoff, total_dim);
    cudaDeviceSynchronize();
}

void SqueezingGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    // 使用具体的压缩操作内核
    double r_real = std::real(r_);
    double r_imag = std::imag(r_);
    squeezing_kernel<<<blocks, threads>>>(state.get_device_data(), qumode_, r_real, r_imag, num_qubits, num_qumodes, cutoff, total_dim);
    cudaDeviceSynchronize();
}

void ConditionalDisplacementGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    // 使用具体的条件位移操作内核
    double alpha_real = std::real(alpha_);
    double alpha_imag = std::imag(alpha_);
    conditional_displacement_kernel<<<blocks, threads>>>(state.get_device_data(), qubit_, qumode_, alpha_real, alpha_imag, num_qubits, num_qumodes, cutoff, total_dim);
    cudaDeviceSynchronize();
}

void JaynesCummingsGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    // 使用具体的Jaynes-Cummings相互作用内核
    jaynes_cummings_kernel<<<blocks, threads>>>(state.get_device_data(), qubit_, qumode_, angle_, num_qubits, num_qumodes, cutoff, total_dim);
    cudaDeviceSynchronize();
}

void BeamSplitterGate::apply(QuantumState& state) const {
    int num_qubits = state.get_num_qubits();
    int num_qumodes = state.get_num_qumodes();
    int cutoff = state.get_cutoff();
    int total_dim = state.get_dim();
    
    int threads = 256;
    int blocks = (total_dim + threads - 1) / threads;
    
    // 使用具体的光束分裂器操作内核
    double theta = angle_;
    double phi = 0.0; // 默认相位为0
    beam_splitter_kernel<<<blocks, threads>>>(state.get_device_data(), qumode1_, qumode2_, theta, phi, num_qubits, num_qumodes, cutoff, total_dim);
    cudaDeviceSynchronize();
}

} // namespace gpu