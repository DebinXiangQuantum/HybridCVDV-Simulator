#pragma once

#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>

/**
 * GaussianStatePool - GPU 上的 Gaussian 状态池，支持引用计数和 CoW 语义。
 *
 * 引用计数规则：
 *   - allocate_state() 返回 refcount=1 的新状态
 *   - add_ref(id) 增加引用计数（用于 MixtureGaussianState 共享状态）
 *   - release_ref(id) 减少引用计数，refcount=0 时自动释放 GPU 内存
 *   - cow_copy(id) 若 refcount>1 则复制并返回新 id（refcount=1），否则返回原 id
 *   - free_state(id) 等价于 release_ref(id)，向后兼容
 */
class GaussianStatePool {
public:
    GaussianStatePool(int num_qumodes, int capacity);
    ~GaussianStatePool();

    int allocate_state();
    void free_state(int state_id);

    // 引用计数操作
    void add_ref(int state_id);
    void release_ref(int state_id);
    int get_ref_count(int state_id) const;

    // Copy-on-Write: 若 refcount > 1 则复制出独立副本，否则返回原 id
    int cow_copy(int state_id);

    // Getters for GPU pointers (Array of pointers on device)
    double** get_d_ptrs_device() { return d_ptrs_dev_; }
    double** get_sig_ptrs_device() { return sig_ptrs_dev_; }

    // Helper to get individual state pointer (Host side)
    double* get_displacement_ptr(int state_id);
    double* get_covariance_ptr(int state_id);

    int get_capacity() const { return capacity_; }
    int get_num_qumodes() const { return num_qumodes_; }
    int get_dim_phase_space() const { return dim_phase_space_; }

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
    std::vector<int> ref_counts_;   // 引用计数

    void deallocate_gpu_memory(int state_id);
};
