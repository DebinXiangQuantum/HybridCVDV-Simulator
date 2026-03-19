#pragma once

#include <vector>

namespace hybridcvdv::noisy::internal {

std::vector<double> identity_matrix(int dim);
std::vector<double> zero_matrix(int dim);
std::vector<double> transpose_square(const std::vector<double>& matrix, int dim);
std::vector<double> multiply_square(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs,
    int dim);
std::vector<double> multiply_matrix_vector(
    const std::vector<double>& matrix,
    const std::vector<double>& vector,
    int dim);
std::vector<double> add_square(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs,
    int dim);
std::vector<double> subtract_square(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs,
    int dim);
std::vector<double> scale_square(
    const std::vector<double>& matrix,
    int dim,
    double scale);
std::vector<double> symplectic_form(int num_qumodes);
std::vector<double> embed_single_mode_matrix(
    const std::vector<double>& local_2x2,
    int num_qumodes,
    int target_qumode);
std::vector<double> embed_single_mode_vector(
    const std::vector<double>& local_2,
    int num_qumodes,
    int target_qumode);
bool all_finite(const std::vector<double>& values);
bool is_symmetric(const std::vector<double>& matrix, int dim, double tolerance);
double jacobi_min_eigenvalue(std::vector<double> matrix, int dim, double tolerance);
std::vector<int> merge_unique_targets(
    const std::vector<int>& lhs,
    const std::vector<int>& rhs);
double determinant_2x2(const std::vector<double>& matrix, int offset, int dim);
std::vector<double> inverse_2x2(const std::vector<double>& matrix, int offset, int dim);

}  // namespace hybridcvdv::noisy::internal
