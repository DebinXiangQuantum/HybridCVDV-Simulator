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

// 递归光束分裂器
void apply_beam_splitter_recursive(CVStatePool* state_pool, const int* target_indices,
                                  int batch_size, double theta, double phi);

#endif // TWO_MODE_GATES_H
