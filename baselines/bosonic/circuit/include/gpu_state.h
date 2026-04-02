#ifndef GPU_STATE_H
#define GPU_STATE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <complex>
#include <stdexcept>

namespace gpu {

class GPUState {
public:
    GPUState(int dim);
    ~GPUState();
    
    int dim() const { return dim_; }
    
    void uploadFromHost(const std::vector<std::complex<double>>& host_data);
    void downloadToHost(std::vector<std::complex<double>>& host_data) const;
    
    cuDoubleComplex* device_data() { return d_data_; }
    const cuDoubleComplex* device_data() const { return d_data_; }
    
    void copyFrom(const GPUState& other);
    void setZero();
    
private:
    int dim_;
    cuDoubleComplex* d_data_;
};

} // namespace gpu

#endif // GPU_STATE_H
