#include "matrix_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hybridcvdv::noisy::internal {

std::vector<double> identity_matrix(int dim) {
    std::vector<double> matrix(static_cast<std::size_t>(dim) * dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        matrix[static_cast<std::size_t>(i) * dim + i] = 1.0;
    }
    return matrix;
}

std::vector<double> zero_matrix(int dim) {
    return std::vector<double>(static_cast<std::size_t>(dim) * dim, 0.0);
}

std::vector<double> transpose_square(const std::vector<double>& matrix, int dim) {
    std::vector<double> result(static_cast<std::size_t>(dim) * dim, 0.0);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            result[static_cast<std::size_t>(col) * dim + row] =
                matrix[static_cast<std::size_t>(row) * dim + col];
        }
    }
    return result;
}

std::vector<double> multiply_square(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs,
    int dim) {
    std::vector<double> result(static_cast<std::size_t>(dim) * dim, 0.0);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double sum = 0.0;
            for (int k = 0; k < dim; ++k) {
                sum += lhs[static_cast<std::size_t>(row) * dim + k] *
                       rhs[static_cast<std::size_t>(k) * dim + col];
            }
            result[static_cast<std::size_t>(row) * dim + col] = sum;
        }
    }
    return result;
}

std::vector<double> multiply_matrix_vector(
    const std::vector<double>& matrix,
    const std::vector<double>& vector,
    int dim) {
    std::vector<double> result(static_cast<std::size_t>(dim), 0.0);
    for (int row = 0; row < dim; ++row) {
        double sum = 0.0;
        for (int col = 0; col < dim; ++col) {
            sum += matrix[static_cast<std::size_t>(row) * dim + col] *
                   vector[static_cast<std::size_t>(col)];
        }
        result[static_cast<std::size_t>(row)] = sum;
    }
    return result;
}

std::vector<double> add_square(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs,
    int dim) {
    std::vector<double> result(static_cast<std::size_t>(dim) * dim, 0.0);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

std::vector<double> subtract_square(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs,
    int dim) {
    std::vector<double> result(static_cast<std::size_t>(dim) * dim, 0.0);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

std::vector<double> scale_square(
    const std::vector<double>& matrix,
    int dim,
    double scale) {
    std::vector<double> result(static_cast<std::size_t>(dim) * dim, 0.0);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = matrix[i] * scale;
    }
    return result;
}

std::vector<double> symplectic_form(int num_qumodes) {
    const int dim = 2 * num_qumodes;
    std::vector<double> omega(static_cast<std::size_t>(dim) * dim, 0.0);
    for (int mode = 0; mode < num_qumodes; ++mode) {
        const int row = 2 * mode;
        omega[static_cast<std::size_t>(row) * dim + row + 1] = 1.0;
        omega[static_cast<std::size_t>(row + 1) * dim + row] = -1.0;
    }
    return omega;
}

std::vector<double> embed_single_mode_matrix(
    const std::vector<double>& local_2x2,
    int num_qumodes,
    int target_qumode) {
    if (local_2x2.size() != 4) {
        throw std::invalid_argument("single-mode local matrix must be 2x2");
    }
    if (target_qumode < 0 || target_qumode >= num_qumodes) {
        throw std::out_of_range("target qumode out of range");
    }

    const int dim = 2 * num_qumodes;
    std::vector<double> embedded = identity_matrix(dim);
    const int row = 2 * target_qumode;
    embedded[static_cast<std::size_t>(row) * dim + row] = local_2x2[0];
    embedded[static_cast<std::size_t>(row) * dim + row + 1] = local_2x2[1];
    embedded[static_cast<std::size_t>(row + 1) * dim + row] = local_2x2[2];
    embedded[static_cast<std::size_t>(row + 1) * dim + row + 1] = local_2x2[3];
    return embedded;
}

std::vector<double> embed_single_mode_vector(
    const std::vector<double>& local_2,
    int num_qumodes,
    int target_qumode) {
    if (local_2.size() != 2) {
        throw std::invalid_argument("single-mode local vector must have length 2");
    }
    if (target_qumode < 0 || target_qumode >= num_qumodes) {
        throw std::out_of_range("target qumode out of range");
    }

    std::vector<double> embedded(static_cast<std::size_t>(2 * num_qumodes), 0.0);
    const int row = 2 * target_qumode;
    embedded[static_cast<std::size_t>(row)] = local_2[0];
    embedded[static_cast<std::size_t>(row + 1)] = local_2[1];
    return embedded;
}

bool all_finite(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

bool is_symmetric(const std::vector<double>& matrix, int dim, double tolerance) {
    for (int row = 0; row < dim; ++row) {
        for (int col = row + 1; col < dim; ++col) {
            const double lhs = matrix[static_cast<std::size_t>(row) * dim + col];
            const double rhs = matrix[static_cast<std::size_t>(col) * dim + row];
            if (std::abs(lhs - rhs) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

double jacobi_min_eigenvalue(std::vector<double> matrix, int dim, double tolerance) {
    if (dim <= 0) {
        throw std::invalid_argument("jacobi_min_eigenvalue requires positive dimension");
    }

    for (int sweep = 0; sweep < 64 * dim; ++sweep) {
        double max_off_diagonal = 0.0;
        int pivot_row = 0;
        int pivot_col = 1;

        for (int row = 0; row < dim; ++row) {
            for (int col = row + 1; col < dim; ++col) {
                const double value = std::abs(matrix[static_cast<std::size_t>(row) * dim + col]);
                if (value > max_off_diagonal) {
                    max_off_diagonal = value;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }

        if (max_off_diagonal <= tolerance) {
            break;
        }

        const double app = matrix[static_cast<std::size_t>(pivot_row) * dim + pivot_row];
        const double aqq = matrix[static_cast<std::size_t>(pivot_col) * dim + pivot_col];
        const double apq = matrix[static_cast<std::size_t>(pivot_row) * dim + pivot_col];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < dim; ++k) {
            if (k == pivot_row || k == pivot_col) {
                continue;
            }

            const double aik = matrix[static_cast<std::size_t>(pivot_row) * dim + k];
            const double aqk = matrix[static_cast<std::size_t>(pivot_col) * dim + k];
            const double new_aik = c * aik - s * aqk;
            const double new_aqk = s * aik + c * aqk;

            matrix[static_cast<std::size_t>(pivot_row) * dim + k] = new_aik;
            matrix[static_cast<std::size_t>(k) * dim + pivot_row] = new_aik;
            matrix[static_cast<std::size_t>(pivot_col) * dim + k] = new_aqk;
            matrix[static_cast<std::size_t>(k) * dim + pivot_col] = new_aqk;
        }

        const double new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        const double new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[static_cast<std::size_t>(pivot_row) * dim + pivot_row] = new_app;
        matrix[static_cast<std::size_t>(pivot_col) * dim + pivot_col] = new_aqq;
        matrix[static_cast<std::size_t>(pivot_row) * dim + pivot_col] = 0.0;
        matrix[static_cast<std::size_t>(pivot_col) * dim + pivot_row] = 0.0;
    }

    double min_eigenvalue = std::numeric_limits<double>::infinity();
    for (int i = 0; i < dim; ++i) {
        min_eigenvalue = std::min(
            min_eigenvalue,
            matrix[static_cast<std::size_t>(i) * dim + i]);
    }
    return min_eigenvalue;
}

std::vector<int> merge_unique_targets(
    const std::vector<int>& lhs,
    const std::vector<int>& rhs) {
    std::vector<int> merged = lhs;
    merged.insert(merged.end(), rhs.begin(), rhs.end());
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    return merged;
}

double determinant_2x2(const std::vector<double>& matrix, int offset, int dim) {
    const double a = matrix[static_cast<std::size_t>(offset) * dim + offset];
    const double b = matrix[static_cast<std::size_t>(offset) * dim + offset + 1];
    const double c = matrix[static_cast<std::size_t>(offset + 1) * dim + offset];
    const double d = matrix[static_cast<std::size_t>(offset + 1) * dim + offset + 1];
    return a * d - b * c;
}

std::vector<double> inverse_2x2(const std::vector<double>& matrix, int offset, int dim) {
    const double det = determinant_2x2(matrix, offset, dim);
    if (std::abs(det) <= 1e-15) {
        throw std::runtime_error("2x2 matrix is singular");
    }

    const double a = matrix[static_cast<std::size_t>(offset) * dim + offset];
    const double b = matrix[static_cast<std::size_t>(offset) * dim + offset + 1];
    const double c = matrix[static_cast<std::size_t>(offset + 1) * dim + offset];
    const double d = matrix[static_cast<std::size_t>(offset + 1) * dim + offset + 1];

    return {d / det, -b / det, -c / det, a / det};
}

}  // namespace hybridcvdv::noisy::internal
