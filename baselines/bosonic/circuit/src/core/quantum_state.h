#ifndef QUANTUM_STATE_H
#define QUANTUM_STATE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <complex>
#include <atomic>

namespace gpu {

// 简单的GPU量子态类，使用密集向量存储
class QuantumState {
public:
    QuantumState(int num_qubits, int num_qumodes, int cutoff);
    ~QuantumState();

    void initialize_zero();
    size_t initialize_ground();

    size_t get_dim() const { return dim_; }
    int get_num_qubits() const { return num_qubits_; }
    int get_num_qumodes() const { return num_qumodes_; }
    int get_cutoff() const { return cutoff_; }

    cuDoubleComplex* get_device_data() { return d_data_; }
    const cuDoubleComplex* get_device_data() const { return d_data_; }

    static size_t get_process_gpu_memory() { return process_allocated_gpu_bytes_; }

    void upload(const std::vector<std::complex<double>>& host_data);
    void download(std::vector<std::complex<double>>& host_data) const;

private:
    int num_qubits_;
    int num_qumodes_;
    int cutoff_;
    size_t dim_;
    cuDoubleComplex* d_data_;

    static std::atomic<size_t> process_allocated_gpu_bytes_;
};

} // namespace gpu

#endif // QUANTUM_STATE_H
