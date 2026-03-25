#ifndef TWO_MODE_GATES_H
#define TWO_MODE_GATES_H

#include "cv_state_pool.h"

// MZgate (Mach-Zehnder) - 马赫-曾德尔干涉仪
void apply_mzgate(CVStatePool* state_pool, const int* target_indices,
                 int batch_size, double phi_in, double phi_ex, int cutoff_a, int cutoff_b);

// CZgate (受控相位门)
void apply_czgate(CVStatePool* state_pool, const int* target_indices,
                 int batch_size, double s, int cutoff_a, int cutoff_b);

// CKgate (Cross-Kerr门)
void apply_ckgate(CVStatePool* state_pool, const int* target_indices,
                 int batch_size, double kappa, int cutoff_a, int cutoff_b);

// 任意多qumode全量态上的 Cross-Kerr
void apply_ckgate_on_modes(CVStatePool* state_pool, const int* target_indices,
                          int batch_size, double kappa,
                          int target_qumode1, int target_qumode2,
                          int num_qumodes);

// 递归光束分裂器
void apply_beam_splitter_recursive(CVStatePool* state_pool, const int* target_indices,
                                  int batch_size, double theta, double phi,
                                  int target_qumode1 = 0, int target_qumode2 = 1,
                                  int num_qumodes = 2,
                                  cudaStream_t stream = nullptr,
                                  bool synchronize = true);

// 双模挤压门 TMS(r, theta)
void apply_two_mode_squeezing_recursive(CVStatePool* state_pool, const int* target_indices,
                                       int batch_size, double r, double theta,
                                       int target_qumode1 = 0, int target_qumode2 = 1,
                                       int num_qumodes = 2,
                                       cudaStream_t stream = nullptr,
                                       bool synchronize = true);

// SUM 门
void apply_sum_gate(CVStatePool* state_pool, const int* target_indices,
                   int batch_size, double scale, int cutoff_a, int cutoff_b,
                   int target_qumode1 = 0, int target_qumode2 = 1,
                   int num_qumodes = 2,
                   cudaStream_t stream = nullptr,
                   bool synchronize = true);

#endif // TWO_MODE_GATES_H
