#pragma once

#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>

/**
 * GaussianStatePool - Manages symbolic states on the GPU with individual allocations.
 */
class GaussianStatePool {
public:
    GaussianStatePool(int num_qumodes, int capacity);
    ~GaussianStatePool();

    int allocate_state();
    void free_state(int state_id);

    // Getters for GPU pointers (Array of pointers on device)
    double** get_d_ptrs_device() { return d_ptrs_dev_; }
    double** get_sig_ptrs_device() { return sig_ptrs_dev_; }

    // Helper to get individual state pointer (Host side)
    double* get_displacement_ptr(int state_id);
    double* get_covariance_ptr(int state_id);

    int get_capacity() const { return capacity_; }
    int get_num_qumodes() const { return num_qumodes_; }

    void upload_state(int state_id, const std::vector<double>& d, const std::vector<double>& sigma);
    void download_state(int state_id, std::vector<double>& d, std::vector<double>& sigma) const;

private:
    int num_qumodes_;
    int capacity_;
    int dim_phase_space_;
    
    std::vector<double*> h_d_ptrs_;
    std::vector<double*> h_sig_ptrs_;
    
    double** d_ptrs_dev_;   // Device-side array of pointers to d
    double** sig_ptrs_dev_; // Device-side array of pointers to sigma
    
    std::vector<int> free_list_;
    std::vector<bool> active_flags_;
};
