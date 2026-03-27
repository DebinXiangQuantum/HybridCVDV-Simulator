#include "gaussian_state.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

GaussianStatePool::GaussianStatePool(int num_qumodes, int capacity)
    : num_qumodes_(num_qumodes), capacity_(capacity) {
    
    dim_phase_space_ = 2 * num_qumodes_;
    h_d_ptrs_.resize(capacity_, nullptr);
    h_sig_ptrs_.resize(capacity_, nullptr);
    active_flags_.assign(capacity_, false);
    ref_counts_.assign(capacity_, 0);
    for (int i = capacity_ - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }

    CHECK_CUDA(cudaMalloc(&d_ptrs_dev_, capacity_ * sizeof(double*)));
    CHECK_CUDA(cudaMalloc(&sig_ptrs_dev_, capacity_ * sizeof(double*)));
    
    CHECK_CUDA(cudaMemset(d_ptrs_dev_, 0, capacity_ * sizeof(double*)));
    CHECK_CUDA(cudaMemset(sig_ptrs_dev_, 0, capacity_ * sizeof(double*)));

    std::cout << "GaussianStatePool initialized: " << num_qumodes_ << " qumodes, " 
              << capacity_ << " capacity (refcount-enabled)." << std::endl;
}

GaussianStatePool::~GaussianStatePool() {
    for (int i = 0; i < capacity_; ++i) {
        if (h_d_ptrs_[i]) cudaFree(h_d_ptrs_[i]);
        if (h_sig_ptrs_[i]) cudaFree(h_sig_ptrs_[i]);
    }
    cudaFree(d_ptrs_dev_);
    cudaFree(sig_ptrs_dev_);
}

int GaussianStatePool::allocate_state() {
    if (free_list_.empty()) return -1;
    
    int id = free_list_.back();
    free_list_.pop_back();
    active_flags_[id] = true;
    ref_counts_[id] = 1;
    
    size_t d_size = dim_phase_space_ * sizeof(double);
    size_t sig_size = dim_phase_space_ * dim_phase_space_ * sizeof(double);
    
    CHECK_CUDA(cudaMalloc(&h_d_ptrs_[id], d_size));
    CHECK_CUDA(cudaMalloc(&h_sig_ptrs_[id], sig_size));
    
    CHECK_CUDA(cudaMemset(h_d_ptrs_[id], 0, d_size));
    CHECK_CUDA(cudaMemset(h_sig_ptrs_[id], 0, sig_size));

    CHECK_CUDA(cudaMemcpy(d_ptrs_dev_ + id, &h_d_ptrs_[id], sizeof(double*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sig_ptrs_dev_ + id, &h_sig_ptrs_[id], sizeof(double*), cudaMemcpyHostToDevice));
    
    return id;
}

void GaussianStatePool::deallocate_gpu_memory(int state_id) {
    if (h_d_ptrs_[state_id]) cudaFree(h_d_ptrs_[state_id]);
    if (h_sig_ptrs_[state_id]) cudaFree(h_sig_ptrs_[state_id]);
    h_d_ptrs_[state_id] = nullptr;
    h_sig_ptrs_[state_id] = nullptr;
    
    double* null_ptr = nullptr;
    cudaMemcpy(d_ptrs_dev_ + state_id, &null_ptr, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(sig_ptrs_dev_ + state_id, &null_ptr, sizeof(double*), cudaMemcpyHostToDevice);

    active_flags_[state_id] = false;
    ref_counts_[state_id] = 0;
    free_list_.push_back(state_id);
}

void GaussianStatePool::free_state(int state_id) {
    release_ref(state_id);
}

void GaussianStatePool::add_ref(int state_id) {
    if (state_id < 0 || state_id >= capacity_ || !active_flags_[state_id]) return;
    ++ref_counts_[state_id];
}

void GaussianStatePool::release_ref(int state_id) {
    if (state_id < 0 || state_id >= capacity_ || !active_flags_[state_id]) return;
    if (--ref_counts_[state_id] <= 0) {
        deallocate_gpu_memory(state_id);
    }
}

int GaussianStatePool::get_ref_count(int state_id) const {
    if (state_id < 0 || state_id >= capacity_ || !active_flags_[state_id]) return 0;
    return ref_counts_[state_id];
}

int GaussianStatePool::cow_copy(int state_id) {
    if (state_id < 0 || state_id >= capacity_ || !active_flags_[state_id]) return -1;
    if (ref_counts_[state_id] <= 1) {
        return state_id;  // sole owner — no copy needed
    }

    // Allocate a new state and copy GPU data
    int new_id = allocate_state();
    if (new_id < 0) return -1;

    size_t d_bytes = static_cast<size_t>(dim_phase_space_) * sizeof(double);
    size_t sig_bytes = static_cast<size_t>(dim_phase_space_) * dim_phase_space_ * sizeof(double);
    CHECK_CUDA(cudaMemcpy(h_d_ptrs_[new_id], h_d_ptrs_[state_id], d_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(h_sig_ptrs_[new_id], h_sig_ptrs_[state_id], sig_bytes, cudaMemcpyDeviceToDevice));

    // Release the shared reference
    release_ref(state_id);
    return new_id;
}

double* GaussianStatePool::get_displacement_ptr(int state_id) {
    if (state_id < 0 || state_id >= capacity_) return nullptr;
    return h_d_ptrs_[state_id];
}

double* GaussianStatePool::get_covariance_ptr(int state_id) {
    if (state_id < 0 || state_id >= capacity_) return nullptr;
    return h_sig_ptrs_[state_id];
}

void GaussianStatePool::upload_state(int state_id, const std::vector<double>& d, const std::vector<double>& sigma) {
    if (state_id < 0 || state_id >= capacity_ || !active_flags_[state_id]) {
        throw std::runtime_error("Invalid state ID for upload");
    }
    CHECK_CUDA(cudaMemcpy(h_d_ptrs_[state_id], d.data(), d.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_sig_ptrs_[state_id], sigma.data(), sigma.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void GaussianStatePool::download_state(int state_id, std::vector<double>& d, std::vector<double>& sigma) const {
    if (state_id < 0 || state_id >= capacity_ || !active_flags_[state_id]) {
        throw std::runtime_error("Invalid state ID for download");
    }
    d.resize(dim_phase_space_);
    sigma.resize(dim_phase_space_ * dim_phase_space_);
    CHECK_CUDA(cudaMemcpy(d.data(), h_d_ptrs_[state_id], d.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sigma.data(), h_sig_ptrs_[state_id], sigma.size() * sizeof(double), cudaMemcpyDeviceToHost));
}
