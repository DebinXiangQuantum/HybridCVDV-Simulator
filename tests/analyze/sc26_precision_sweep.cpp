#include "quantum_circuit.h"
#include "reference_gates.h"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Complex = std::complex<double>;
using Vector = Reference::Vector;

struct DeviceMetadata {
    bool available = false;
    int device_index = -1;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int multiprocessor_count = 0;
    size_t total_global_mem_bytes = 0;
};

struct Metrics {
    double l2_error = 0.0;
    double max_error = 0.0;
    double relative_error = 0.0;
    double fidelity = 0.0;
    double fidelity_deviation = 0.0;
    double reference_norm = 0.0;
    double implementation_norm = 0.0;
};

struct TailMetrics {
    double norm_sq = 0.0;
    double tail_population = 0.0;
    double tail_fraction = 0.0;
    double boundary_population = 0.0;
    double boundary_fraction = 0.0;
    double mean_total_photon_number = 0.0;
};

struct GaussianDiagnostics {
    bool available = false;
    std::string reason;
    double vacuum_fidelity = 0.0;
    double vacuum_fidelity_raw = 0.0;
    double vacuum_fidelity_deviation = 1.0;
    double displacement_l2 = 0.0;
    double covariance_max_abs_delta = 0.0;
    double covariance_fro_delta = 0.0;
    double weight_abs = 0.0;
    double weight_abs_sq = 0.0;
    double overlap_determinant = 0.0;
    double overlap_quadratic = 0.0;
};

struct GpuRunOutput {
    Vector state;
    QuantumCircuit::CircuitStats stats{};
    GaussianDiagnostics gaussian;
    bool materialized_for_diagnostics = false;
};

struct Result {
    std::string name;
    std::string category;
    std::string status = "ok";
    std::string note;
    std::map<std::string, std::string> params;
    std::map<std::string, double> metrics;
};

struct Options {
    std::string suite = "all";
    std::string name_filter;
    size_t max_dense_dim = 1 << 20;
    int max_states = 256;
    bool disable_symbolic = true;
    bool force_dense_fock = false;
    fs::path output_path = "experiments/results/sc26_precision_sweep.json";
};

struct CircuitSpec {
    std::string name;
    std::string workload;
    int num_qubits = 0;
    int num_qumodes = 1;
    int cutoff = 16;
    int depth = 0;
    std::string initial_profile = "low";
    std::vector<GateParams> gates;
};

bool is_unconditional_gaussian_gate(const GateParams& gate) {
    switch (gate.type) {
        case GateType::PHASE_ROTATION:
        case GateType::DISPLACEMENT:
        case GateType::SQUEEZING:
        case GateType::BEAM_SPLITTER:
            return true;
        default:
            return false;
    }
}

bool is_gaussian_vacuum_track_candidate(const CircuitSpec& spec) {
    return spec.initial_profile == "vacuum" &&
           std::all_of(spec.gates.begin(), spec.gates.end(), is_unconditional_gaussian_gate);
}

SymplecticGate embed_single_mode_symplectic(const SymplecticGate& local_gate,
                                            int total_qumodes,
                                            int target_qumode) {
    if (local_gate.num_qumodes != 1) {
        throw std::invalid_argument("expected a single-mode symplectic gate");
    }
    if (target_qumode < 0 || target_qumode >= total_qumodes) {
        throw std::out_of_range("target qumode out of range for symplectic embedding");
    }

    SymplecticGate embedded(total_qumodes);
    const int dim = 2 * total_qumodes;
    const int target_row = 2 * target_qumode;
    embedded.S[static_cast<size_t>(target_row) * dim + target_row] = local_gate.S[0];
    embedded.S[static_cast<size_t>(target_row) * dim + target_row + 1] = local_gate.S[1];
    embedded.S[static_cast<size_t>(target_row + 1) * dim + target_row] = local_gate.S[2];
    embedded.S[static_cast<size_t>(target_row + 1) * dim + target_row + 1] = local_gate.S[3];
    embedded.d[static_cast<size_t>(target_row)] = local_gate.d[0];
    embedded.d[static_cast<size_t>(target_row + 1)] = local_gate.d[1];
    return embedded;
}

SymplecticGate gate_to_symplectic_for_harness(const GateParams& gate, int total_qumodes) {
    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            return embed_single_mode_symplectic(
                SymplecticFactory::Rotation(gate.params.at(0).real()),
                total_qumodes,
                gate.target_qumodes.at(0));
        case GateType::DISPLACEMENT:
            return embed_single_mode_symplectic(
                SymplecticFactory::Displacement(gate.params.at(0)),
                total_qumodes,
                gate.target_qumodes.at(0));
        case GateType::SQUEEZING:
            return embed_single_mode_symplectic(
                SymplecticFactory::Squeezing(std::abs(gate.params.at(0)), std::arg(gate.params.at(0))),
                total_qumodes,
                gate.target_qumodes.at(0));
        case GateType::BEAM_SPLITTER: {
            const double phi = gate.params.size() >= 2 ? gate.params[1].real() : 0.0;
            return SymplecticFactory::BeamSplitter(
                gate.params.at(0).real(),
                phi,
                total_qumodes,
                gate.target_qumodes.at(0),
                gate.target_qumodes.at(1));
        }
        default:
            throw std::invalid_argument("gate cannot be represented as an unconditional symplectic update");
    }
}

std::string now_utc_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_tm{};
#if defined(_WIN32)
    gmtime_s(&utc_tm, &now_time);
#else
    gmtime_r(&now_time, &utc_tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&utc_tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (char c : input) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << c; break;
        }
    }
    return oss.str();
}

std::string format_double(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream oss;
    oss << std::setprecision(15) << value;
    return oss.str();
}

bool matches_filter(const std::string& filter, const std::string& name) {
    return filter.empty() || name.find(filter) != std::string::npos;
}

DeviceMetadata query_device() {
    DeviceMetadata device;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        cudaGetLastError();
        return device;
    }

    int index = 0;
    if (cudaGetDevice(&index) != cudaSuccess) {
        cudaGetLastError();
        index = 0;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) {
        cudaGetLastError();
        return device;
    }

    device.available = true;
    device.device_index = index;
    device.name = prop.name;
    device.cc_major = prop.major;
    device.cc_minor = prop.minor;
    device.multiprocessor_count = prop.multiProcessorCount;
    device.total_global_mem_bytes = prop.totalGlobalMem;
    return device;
}

size_t checked_pow_size(size_t base, int exp) {
    size_t value = 1;
    for (int i = 0; i < exp; ++i) {
        if (base != 0 && value > std::numeric_limits<size_t>::max() / base) {
            return std::numeric_limits<size_t>::max();
        }
        value *= base;
    }
    return value;
}

size_t total_dimension(int num_qubits, int num_qumodes, int cutoff) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    if (num_qubits >= static_cast<int>(8 * sizeof(size_t))) {
        return std::numeric_limits<size_t>::max();
    }
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    if (cv_dim != 0 && dv_dim > std::numeric_limits<size_t>::max() / cv_dim) {
        return std::numeric_limits<size_t>::max();
    }
    return dv_dim * cv_dim;
}

Vector zero_state(size_t dim) {
    return Vector(dim, Complex(0.0, 0.0));
}

Vector initial_full_state(int num_qubits, int num_qumodes, int cutoff) {
    const size_t dim = total_dimension(num_qubits, num_qumodes, cutoff);
    Vector state(dim, Complex(0.0, 0.0));
    if (!state.empty()) {
        state[0] = Complex(1.0, 0.0);
    }
    return state;
}

std::vector<cuDoubleComplex> to_cuda_state(const Vector& state) {
    std::vector<cuDoubleComplex> out;
    out.reserve(state.size());
    for (const auto& amp : state) {
        out.push_back(make_cuDoubleComplex(amp.real(), amp.imag()));
    }
    return out;
}

Vector from_cuda_state(const std::vector<cuDoubleComplex>& state) {
    Vector out;
    out.reserve(state.size());
    for (const auto& amp : state) {
        out.emplace_back(cuCreal(amp), cuCimag(amp));
    }
    return out;
}

Metrics compute_metrics(const Vector& reference, const Vector& implementation) {
    Metrics m;
    if (reference.size() != implementation.size()) {
        m.l2_error = std::numeric_limits<double>::infinity();
        m.max_error = std::numeric_limits<double>::infinity();
        m.relative_error = std::numeric_limits<double>::infinity();
        m.fidelity = 0.0;
        m.fidelity_deviation = 1.0;
        return m;
    }
    double ref_norm_sq = 0.0;
    double impl_norm_sq = 0.0;
    Complex overlap(0.0, 0.0);
    for (size_t i = 0; i < reference.size(); ++i) {
        const Complex diff = reference[i] - implementation[i];
        const double abs_error = std::abs(diff);
        m.l2_error += abs_error * abs_error;
        m.max_error = std::max(m.max_error, abs_error);
        ref_norm_sq += std::norm(reference[i]);
        impl_norm_sq += std::norm(implementation[i]);
        overlap += std::conj(reference[i]) * implementation[i];
    }
    m.l2_error = std::sqrt(m.l2_error);
    m.reference_norm = std::sqrt(ref_norm_sq);
    m.implementation_norm = std::sqrt(impl_norm_sq);
    m.relative_error = m.reference_norm > 0.0 ? m.l2_error / m.reference_norm : 0.0;
    if (ref_norm_sq < 1e-30 && impl_norm_sq < 1e-30) {
        m.fidelity = 1.0;
    } else if (ref_norm_sq < 1e-30 || impl_norm_sq < 1e-30) {
        m.fidelity = 0.0;
    } else {
        m.fidelity = std::norm(overlap) / (ref_norm_sq * impl_norm_sq);
    }
    m.fidelity = std::max(0.0, std::min(1.0, m.fidelity));
    m.fidelity_deviation = 1.0 - m.fidelity;
    return m;
}

TailMetrics compute_tail_metrics(const Vector& state,
                                 int num_qubits,
                                 int num_qumodes,
                                 int cutoff) {
    TailMetrics metrics;
    (void)num_qubits;
    if (state.empty() || num_qumodes <= 0 || cutoff <= 0) {
        return metrics;
    }

    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    if (cv_dim == 0 || cv_dim == std::numeric_limits<size_t>::max()) {
        return metrics;
    }

    const int tail_start = std::max(0, cutoff - std::max(1, cutoff / 8));
    for (size_t idx = 0; idx < state.size(); ++idx) {
        const double prob = std::norm(state[idx]);
        metrics.norm_sq += prob;
        const size_t cv_index = idx % cv_dim;
        size_t rem = cv_index;
        bool in_tail = false;
        bool on_boundary = false;
        int photon_sum = 0;
        for (int mode = num_qumodes - 1; mode >= 0; --mode) {
            (void)mode;
            const int n = static_cast<int>(rem % static_cast<size_t>(cutoff));
            rem /= static_cast<size_t>(cutoff);
            photon_sum += n;
            in_tail = in_tail || n >= tail_start;
            on_boundary = on_boundary || n == cutoff - 1;
        }
        if (in_tail) {
            metrics.tail_population += prob;
        }
        if (on_boundary) {
            metrics.boundary_population += prob;
        }
        metrics.mean_total_photon_number += prob * static_cast<double>(photon_sum);
    }

    if (metrics.norm_sq > 0.0) {
        metrics.tail_fraction = metrics.tail_population / metrics.norm_sq;
        metrics.boundary_fraction = metrics.boundary_population / metrics.norm_sq;
        metrics.mean_total_photon_number /= metrics.norm_sq;
    }
    return metrics;
}

bool solve_linear_and_determinant(std::vector<double> matrix,
                                  const std::vector<double>& rhs,
                                  std::vector<double>* solution,
                                  double* determinant) {
    const size_t n = rhs.size();
    if (matrix.size() != n * n || !solution || !determinant) {
        return false;
    }
    solution->assign(n, 0.0);
    std::vector<double> b = rhs;
    double det = 1.0;
    int sign = 1;

    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        double pivot_abs = std::abs(matrix[col * n + col]);
        for (size_t row = col + 1; row < n; ++row) {
            const double candidate = std::abs(matrix[row * n + col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs < 1e-300 || !std::isfinite(pivot_abs)) {
            return false;
        }
        if (pivot != col) {
            for (size_t k = col; k < n; ++k) {
                std::swap(matrix[col * n + k], matrix[pivot * n + k]);
            }
            std::swap(b[col], b[pivot]);
            sign = -sign;
        }

        const double diag = matrix[col * n + col];
        det *= diag;
        for (size_t row = col + 1; row < n; ++row) {
            const double factor = matrix[row * n + col] / diag;
            matrix[row * n + col] = 0.0;
            for (size_t k = col + 1; k < n; ++k) {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }

    if (sign < 0) {
        det = -det;
    }
    if (!std::isfinite(det) || det <= 0.0) {
        return false;
    }

    for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
        double accum = b[static_cast<size_t>(row)];
        for (size_t col = static_cast<size_t>(row) + 1; col < n; ++col) {
            accum -= matrix[static_cast<size_t>(row) * n + col] * (*solution)[col];
        }
        (*solution)[static_cast<size_t>(row)] =
            accum / matrix[static_cast<size_t>(row) * n + static_cast<size_t>(row)];
    }

    *determinant = det;
    return true;
}

GaussianDiagnostics compute_gaussian_vacuum_diagnostics(
    const std::vector<double>& displacement,
    const std::vector<double>& covariance,
    Complex weight) {
    GaussianDiagnostics diagnostics;
    diagnostics.available = true;
    diagnostics.reason = "ok";
    diagnostics.weight_abs = std::abs(weight);
    diagnostics.weight_abs_sq = std::norm(weight);

    const size_t dim = displacement.size();
    if (dim == 0 || covariance.size() != dim * dim) {
        diagnostics.available = false;
        diagnostics.reason = "invalid Gaussian moment dimensions";
        return diagnostics;
    }

    std::vector<double> overlap_matrix = covariance;
    for (size_t row = 0; row < dim; ++row) {
        diagnostics.displacement_l2 += displacement[row] * displacement[row];
        for (size_t col = 0; col < dim; ++col) {
            const double vacuum_value = (row == col) ? 0.5 : 0.0;
            const double delta = covariance[row * dim + col] - vacuum_value;
            diagnostics.covariance_max_abs_delta =
                std::max(diagnostics.covariance_max_abs_delta, std::abs(delta));
            diagnostics.covariance_fro_delta += delta * delta;
        }
        overlap_matrix[row * dim + row] += 0.5;
    }
    diagnostics.displacement_l2 = std::sqrt(diagnostics.displacement_l2);
    diagnostics.covariance_fro_delta = std::sqrt(diagnostics.covariance_fro_delta);

    std::vector<double> solved;
    double det = 0.0;
    if (!solve_linear_and_determinant(overlap_matrix, displacement, &solved, &det)) {
        diagnostics.available = false;
        diagnostics.reason = "Gaussian overlap linear solve failed";
        return diagnostics;
    }

    double quadratic = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        quadratic += displacement[i] * solved[i];
    }
    diagnostics.overlap_determinant = det;
    diagnostics.overlap_quadratic = quadratic;
    diagnostics.vacuum_fidelity_raw =
        diagnostics.weight_abs_sq * std::exp(-0.5 * quadratic) / std::sqrt(det);
    diagnostics.vacuum_fidelity =
        std::max(0.0, std::min(1.0, diagnostics.vacuum_fidelity_raw));
    diagnostics.vacuum_fidelity_deviation = 1.0 - diagnostics.vacuum_fidelity;
    return diagnostics;
}

void apply_symplectic_to_moments(const SymplecticGate& gate,
                                 std::vector<double>* displacement,
                                 std::vector<double>* covariance) {
    const int dim = 2 * gate.num_qumodes;
    std::vector<double> new_displacement(static_cast<size_t>(dim), 0.0);
    for (int row = 0; row < dim; ++row) {
        double accum = gate.d[static_cast<size_t>(row)];
        for (int col = 0; col < dim; ++col) {
            accum += gate.S[static_cast<size_t>(row) * dim + col] *
                     (*displacement)[static_cast<size_t>(col)];
        }
        new_displacement[static_cast<size_t>(row)] = accum;
    }

    std::vector<double> temp(static_cast<size_t>(dim) * dim, 0.0);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double accum = 0.0;
            for (int k = 0; k < dim; ++k) {
                accum += gate.S[static_cast<size_t>(row) * dim + k] *
                         (*covariance)[static_cast<size_t>(k) * dim + col];
            }
            temp[static_cast<size_t>(row) * dim + col] = accum;
        }
    }

    std::vector<double> new_covariance(static_cast<size_t>(dim) * dim, 0.0);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double accum = 0.0;
            for (int k = 0; k < dim; ++k) {
                accum += temp[static_cast<size_t>(row) * dim + k] *
                         gate.S[static_cast<size_t>(col) * dim + k];
            }
            new_covariance[static_cast<size_t>(row) * dim + col] = accum;
        }
    }

    *displacement = std::move(new_displacement);
    *covariance = std::move(new_covariance);
}

GaussianDiagnostics run_cpu_gaussian_vacuum_track(const CircuitSpec& spec) {
    const int dim = 2 * spec.num_qumodes;
    std::vector<double> displacement(static_cast<size_t>(dim), 0.0);
    std::vector<double> covariance(static_cast<size_t>(dim) * dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        covariance[static_cast<size_t>(i) * dim + i] = 0.5;
    }
    for (const GateParams& gate : spec.gates) {
        apply_symplectic_to_moments(
            gate_to_symplectic_for_harness(gate, spec.num_qumodes),
            &displacement,
            &covariance);
    }
    return compute_gaussian_vacuum_diagnostics(
        displacement,
        covariance,
        Complex(1.0, 0.0));
}

size_t cv_index_from_digits(const std::vector<int>& digits, int cutoff) {
    size_t idx = 0;
    for (int d : digits) {
        idx = idx * static_cast<size_t>(cutoff) + static_cast<size_t>(d);
    }
    return idx;
}

void decode_cv_index(size_t idx, int num_qumodes, int cutoff, std::vector<int>* digits) {
    digits->assign(static_cast<size_t>(num_qumodes), 0);
    for (int mode = num_qumodes - 1; mode >= 0; --mode) {
        (*digits)[static_cast<size_t>(mode)] = static_cast<int>(idx % static_cast<size_t>(cutoff));
        idx /= static_cast<size_t>(cutoff);
    }
}

void add_branch_state_recursive(HDDNode* node,
                                Complex weight,
                                int num_qubits,
                                int num_qumodes,
                                int cutoff,
                                const CVStatePool& pool,
                                Vector* full_state) {
    if (!node || std::abs(weight) < 1e-300) {
        return;
    }
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    if (node->is_terminal()) {
        if (node->tensor_id < 0 || !pool.is_valid_state(node->tensor_id)) {
            return;
        }
        std::vector<cuDoubleComplex> raw;
        pool.download_state(node->tensor_id, raw);
        const Vector cv_state = from_cuda_state(raw);
        const size_t copy_dim = std::min(cv_dim, cv_state.size());
        for (size_t i = 0; i < copy_dim; ++i) {
            (*full_state)[i] += weight * cv_state[i];
        }
        return;
    }

    const int shift = num_qubits - 1 - node->qubit_level;
    const size_t bit_value = shift >= 0 ? (static_cast<size_t>(1) << shift) : 0;

    Vector low_accum(cv_dim, Complex(0.0, 0.0));
    Vector high_accum(cv_dim, Complex(0.0, 0.0));
    add_branch_state_recursive(
        node->low, weight * node->w_low, num_qubits, num_qumodes, cutoff, pool, &low_accum);
    add_branch_state_recursive(
        node->high, weight * node->w_high, num_qubits, num_qumodes, cutoff, pool, &high_accum);

    const size_t parent_stride = cv_dim;
    for (size_t branch = 0; branch < (static_cast<size_t>(1) << num_qubits); ++branch) {
        const bool branch_bit_set = (branch & bit_value) != 0;
        const Vector& src = branch_bit_set ? high_accum : low_accum;
        const size_t base = branch * parent_stride;
        for (size_t i = 0; i < cv_dim; ++i) {
            (*full_state)[base + i] += src[i];
        }
    }
}

void collect_full_state_recursive(HDDNode* node,
                                  Complex weight,
                                  size_t branch_bits,
                                  int num_qubits,
                                  int num_qumodes,
                                  int cutoff,
                                  const CVStatePool& pool,
                                  Vector* full_state) {
    if (!node || std::abs(weight) < 1e-300) {
        return;
    }
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    if (node->is_terminal()) {
        if (node->tensor_id < 0 || !pool.is_valid_state(node->tensor_id)) {
            return;
        }
        std::vector<cuDoubleComplex> raw;
        pool.download_state(node->tensor_id, raw);
        const Vector cv_state = from_cuda_state(raw);
        const size_t base = branch_bits * cv_dim;
        const size_t copy_dim = std::min(cv_dim, cv_state.size());
        for (size_t i = 0; i < copy_dim; ++i) {
            (*full_state)[base + i] += weight * cv_state[i];
        }
        return;
    }

    const int shift = num_qubits - 1 - node->qubit_level;
    const size_t bit = shift >= 0 ? (static_cast<size_t>(1) << shift) : 0;
    collect_full_state_recursive(
        node->low,
        weight * node->w_low,
        branch_bits & ~bit,
        num_qubits,
        num_qumodes,
        cutoff,
        pool,
        full_state);
    collect_full_state_recursive(
        node->high,
        weight * node->w_high,
        branch_bits | bit,
        num_qubits,
        num_qumodes,
        cutoff,
        pool,
        full_state);
}

Vector extract_full_state(QuantumCircuit& circuit,
                          int num_qubits,
                          int num_qumodes,
                          int cutoff) {
    const size_t dim = total_dimension(num_qubits, num_qumodes, cutoff);
    Vector full(dim, Complex(0.0, 0.0));
    collect_full_state_recursive(
        circuit.get_root_node(),
        Complex(1.0, 0.0),
        0,
        num_qubits,
        num_qumodes,
        cutoff,
        circuit.get_state_pool(),
        &full);
    return full;
}

HDDNode* find_all_zero_terminal(HDDNode* node) {
    HDDNode* current = node;
    while (current && !current->is_terminal()) {
        current = current->low;
    }
    return current;
}

void inject_zero_branch_cv_state(QuantumCircuit& circuit, const Vector& cv_state) {
    HDDNode* terminal = find_all_zero_terminal(circuit.get_root_node());
    if (!terminal || terminal->tensor_id < 0) {
        throw std::runtime_error("failed to find all-zero terminal for state injection");
    }
    circuit.get_state_pool().upload_state(terminal->tensor_id, to_cuda_state(cv_state));
}

GpuRunOutput run_gpu_circuit_output(const CircuitSpec& spec,
                                    const Vector* initial_full,
                                    const Options& options,
                                    bool materialize_for_state = true) {
    const int max_states = options.max_states;
    QuantumCircuit circuit(spec.num_qubits, spec.num_qumodes, spec.cutoff, max_states);
    circuit.set_force_dense_fock(options.force_dense_fock);
    circuit.set_symbolic_vacuum_projection_tolerance(1e-6);
    if (options.disable_symbolic) {
        circuit.set_gaussian_symbolic_enabled(false);
        circuit.set_diagonal_mixture_enabled(false);
        circuit.set_fused_diagonal_enabled(true);
        circuit.set_eager_symbolic_materialization_enabled(false);
    }
    for (const GateParams& gate : spec.gates) {
        circuit.add_gate(gate);
    }
    circuit.build();

    if (initial_full) {
        const size_t cv_dim = checked_pow_size(static_cast<size_t>(spec.cutoff), spec.num_qumodes);
        const size_t expected_dim = total_dimension(spec.num_qubits, spec.num_qumodes, spec.cutoff);
        if (initial_full->size() != expected_dim) {
            throw std::runtime_error("initial full-state dimension mismatch");
        }
        bool only_zero_branch = true;
        for (size_t i = cv_dim; i < initial_full->size(); ++i) {
            if (std::abs((*initial_full)[i]) > 1e-14) {
                only_zero_branch = false;
                break;
            }
        }
        if (!only_zero_branch) {
            throw std::runtime_error("custom injection currently supports all-zero qubit branch only");
        }
        Vector cv_state(initial_full->begin(), initial_full->begin() + static_cast<std::ptrdiff_t>(cv_dim));
        inject_zero_branch_cv_state(circuit, cv_state);
    }

    circuit.execute();
    GpuRunOutput output;
    output.stats = circuit.get_stats();

    std::vector<double> gaussian_displacement;
    std::vector<double> gaussian_covariance;
    Complex gaussian_weight(0.0, 0.0);
    std::string gaussian_reason;
    if (circuit.get_single_symbolic_gaussian_state(
            gaussian_displacement,
            gaussian_covariance,
            &gaussian_weight,
            &gaussian_reason)) {
        output.gaussian = compute_gaussian_vacuum_diagnostics(
            gaussian_displacement,
            gaussian_covariance,
            gaussian_weight);
    } else {
        output.gaussian.available = false;
        output.gaussian.reason = gaussian_reason;
    }

    if (materialize_for_state) {
        output.materialized_for_diagnostics =
            circuit.materialize_symbolic_terminals_for_diagnostics();
        output.state = extract_full_state(circuit, spec.num_qubits, spec.num_qumodes, spec.cutoff);
    }
    return output;
}

Vector run_gpu_circuit(const CircuitSpec& spec,
                       const Vector* initial_full,
                       const Options& options) {
    return run_gpu_circuit_output(spec, initial_full, options, true).state;
}

void add_gpu_run_metrics(Result* result,
                         const std::string& prefix,
                         const GpuRunOutput& output) {
    result->metrics[prefix + "gaussian_symbolic_blocks"] =
        static_cast<double>(output.stats.gaussian_symbolic_blocks);
    result->metrics[prefix + "symbolic_materializations"] =
        static_cast<double>(output.stats.symbolic_materializations);
    result->metrics[prefix + "exact_blocks"] =
        static_cast<double>(output.stats.exact_blocks);
    result->metrics[prefix + "active_states"] =
        static_cast<double>(output.stats.active_states);
    result->params[prefix + "gaussian_available"] =
        output.gaussian.available ? "true" : "false";
    if (!output.gaussian.reason.empty()) {
        result->params[prefix + "gaussian_reason"] = output.gaussian.reason;
    }

    if (!output.gaussian.available) {
        return;
    }
    result->metrics[prefix + "gaussian_vacuum_fidelity"] =
        output.gaussian.vacuum_fidelity;
    result->metrics[prefix + "gaussian_vacuum_fidelity_raw"] =
        output.gaussian.vacuum_fidelity_raw;
    result->metrics[prefix + "gaussian_vacuum_fidelity_deviation"] =
        output.gaussian.vacuum_fidelity_deviation;
    result->metrics[prefix + "gaussian_displacement_l2"] =
        output.gaussian.displacement_l2;
    result->metrics[prefix + "gaussian_covariance_max_abs_delta"] =
        output.gaussian.covariance_max_abs_delta;
    result->metrics[prefix + "gaussian_covariance_fro_delta"] =
        output.gaussian.covariance_fro_delta;
    result->metrics[prefix + "gaussian_weight_abs"] =
        output.gaussian.weight_abs;
    result->metrics[prefix + "gaussian_weight_abs_sq"] =
        output.gaussian.weight_abs_sq;
    result->metrics[prefix + "gaussian_overlap_determinant"] =
        output.gaussian.overlap_determinant;
    result->metrics[prefix + "gaussian_overlap_quadratic"] =
        output.gaussian.overlap_quadratic;
}

std::array<Complex, 4> single_qubit_matrix(const GateParams& gate) {
    std::array<Complex, 4> m{Complex(0.0, 0.0), Complex(0.0, 0.0),
                             Complex(0.0, 0.0), Complex(0.0, 0.0)};
    switch (gate.type) {
        case GateType::PAULI_X:
            m[1] = 1.0; m[2] = 1.0; break;
        case GateType::PAULI_Y:
            m[1] = Complex(0.0, -1.0); m[2] = Complex(0.0, 1.0); break;
        case GateType::PAULI_Z:
            m[0] = 1.0; m[3] = -1.0; break;
        case GateType::HADAMARD: {
            const double s = 1.0 / std::sqrt(2.0);
            m[0] = s; m[1] = s; m[2] = s; m[3] = -s; break;
        }
        case GateType::ROTATION_X: {
            const double theta = gate.params.at(0).real();
            const double c = std::cos(theta / 2.0);
            const double s = std::sin(theta / 2.0);
            m[0] = c; m[1] = Complex(0.0, -s); m[2] = Complex(0.0, -s); m[3] = c; break;
        }
        case GateType::ROTATION_Y: {
            const double theta = gate.params.at(0).real();
            const double c = std::cos(theta / 2.0);
            const double s = std::sin(theta / 2.0);
            m[0] = c; m[1] = -s; m[2] = s; m[3] = c; break;
        }
        case GateType::ROTATION_Z: {
            const double theta = gate.params.at(0).real();
            const double c = std::cos(theta / 2.0);
            const double s = std::sin(theta / 2.0);
            m[0] = Complex(c, -s); m[3] = Complex(c, s); break;
        }
        case GateType::PHASE_GATE_S:
            m[0] = 1.0; m[3] = Complex(0.0, 1.0); break;
        case GateType::PHASE_GATE_T:
            m[0] = 1.0; m[3] = Complex(std::cos(M_PI / 4.0), std::sin(M_PI / 4.0)); break;
        default:
            throw std::runtime_error("not a single-qubit gate");
    }
    return m;
}

void apply_single_qubit_cpu(Vector* state,
                            int num_qubits,
                            int num_qumodes,
                            int cutoff,
                            int target_qubit,
                            const std::array<Complex, 4>& m) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const int shift = num_qubits - 1 - target_qubit;
    const size_t bit = static_cast<size_t>(1) << shift;
    Vector out(state->size(), Complex(0.0, 0.0));
    for (size_t q = 0; q < dv_dim; ++q) {
        const size_t in_bit = (q & bit) ? 1 : 0;
        for (size_t out_bit = 0; out_bit < 2; ++out_bit) {
            const size_t q_out = out_bit ? (q | bit) : (q & ~bit);
            const Complex coeff = m[out_bit * 2 + in_bit];
            if (std::abs(coeff) == 0.0) {
                continue;
            }
            for (size_t cv = 0; cv < cv_dim; ++cv) {
                out[q_out * cv_dim + cv] += coeff * (*state)[q * cv_dim + cv];
            }
        }
    }
    *state = std::move(out);
}

void apply_cnot_cpu(Vector* state,
                    int num_qubits,
                    int num_qumodes,
                    int cutoff,
                    int control,
                    int target) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const size_t cbit = static_cast<size_t>(1) << (num_qubits - 1 - control);
    const size_t tbit = static_cast<size_t>(1) << (num_qubits - 1 - target);
    Vector out(state->size(), Complex(0.0, 0.0));
    for (size_t q = 0; q < dv_dim; ++q) {
        const size_t q_out = (q & cbit) ? (q ^ tbit) : q;
        for (size_t cv = 0; cv < cv_dim; ++cv) {
            out[q_out * cv_dim + cv] += (*state)[q * cv_dim + cv];
        }
    }
    *state = std::move(out);
}

void apply_cz_cpu(Vector* state,
                  int num_qubits,
                  int num_qumodes,
                  int cutoff,
                  int control,
                  int target) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const size_t cbit = static_cast<size_t>(1) << (num_qubits - 1 - control);
    const size_t tbit = static_cast<size_t>(1) << (num_qubits - 1 - target);
    for (size_t q = 0; q < dv_dim; ++q) {
        if ((q & cbit) && (q & tbit)) {
            for (size_t cv = 0; cv < cv_dim; ++cv) {
                (*state)[q * cv_dim + cv] *= -1.0;
            }
        }
    }
}

Reference::Matrix cv_single_mode_matrix(const GateParams& gate, int cutoff) {
    Reference::Matrix mat(cutoff, Vector(cutoff, Complex(0.0, 0.0)));
    switch (gate.type) {
        case GateType::PHASE_ROTATION: {
            const double theta = gate.params.at(0).real();
            for (int n = 0; n < cutoff; ++n) {
                const double phase = -theta * static_cast<double>(n);
                mat[n][n] = Complex(std::cos(phase), std::sin(phase));
            }
            break;
        }
        case GateType::KERR_GATE: {
            const double chi = gate.params.at(0).real();
            for (int n = 0; n < cutoff; ++n) {
                const double phase = chi * static_cast<double>(n * n);
                mat[n][n] = Complex(std::cos(phase), std::sin(phase));
            }
            break;
        }
        case GateType::CONDITIONAL_PARITY: {
            const double parity = gate.params.at(0).real();
            for (int n = 0; n < cutoff; ++n) {
                const double phase = -parity * M_PI * static_cast<double>(n % 2);
                mat[n][n] = Complex(std::cos(phase), std::sin(phase));
            }
            break;
        }
        case GateType::SNAP_GATE: {
            const double theta = gate.params.at(0).real();
            const int target = static_cast<int>(std::llround(gate.params.at(1).real()));
            for (int n = 0; n < cutoff; ++n) {
                const double phase = n == target ? theta : 0.0;
                mat[n][n] = Complex(std::cos(phase), std::sin(phase));
            }
            break;
        }
        case GateType::MULTI_SNAP_GATE: {
            for (int n = 0; n < cutoff; ++n) {
                const double phase = n < static_cast<int>(gate.params.size()) ? gate.params[n].real() : 0.0;
                mat[n][n] = Complex(std::cos(phase), std::sin(phase));
            }
            break;
        }
        case GateType::CREATION_OPERATOR: {
            for (int n = 1; n < cutoff; ++n) {
                mat[n][n - 1] = std::sqrt(static_cast<double>(n));
            }
            break;
        }
        case GateType::ANNIHILATION_OPERATOR: {
            for (int n = 0; n + 1 < cutoff; ++n) {
                mat[n][n + 1] = std::sqrt(static_cast<double>(n + 1));
            }
            break;
        }
        case GateType::DISPLACEMENT:
            mat = Reference::create_displacement_matrix(cutoff, gate.params.at(0));
            break;
        case GateType::SQUEEZING:
            mat = Reference::create_squeezing_matrix(cutoff, gate.params.at(0));
            break;
        default:
            throw std::runtime_error("not a supported single-mode CV gate");
    }
    return mat;
}

Reference::Matrix cv_two_mode_matrix(const GateParams& gate, int cutoff) {
    const int dim = cutoff * cutoff;
    Reference::Matrix mat(dim, Vector(dim, Complex(0.0, 0.0)));
    if (gate.type == GateType::BEAM_SPLITTER) {
        return Reference::create_beam_splitter_matrix(
            cutoff, cutoff, gate.params.at(0).real(), gate.params.at(1).real());
    }
    if (gate.type == GateType::CROSS_KERR_GATE) {
        const double kappa = gate.params.at(0).real();
        for (int n0 = 0; n0 < cutoff; ++n0) {
            for (int n1 = 0; n1 < cutoff; ++n1) {
                const int idx = n0 * cutoff + n1;
                const double phase = kappa * static_cast<double>(n0 * n1);
                mat[idx][idx] = Complex(std::cos(phase), std::sin(phase));
            }
        }
        return mat;
    }
    throw std::runtime_error("not a supported two-mode CV gate");
}

Reference::Matrix two_mode_squeezing_matrix(int cutoff, Complex xi) {
    const int dim = cutoff * cutoff;
    Reference::Matrix generator(dim, Vector(dim, Complex(0.0, 0.0)));
    for (int p = 0; p < cutoff; ++p) {
        for (int q = 0; q < cutoff; ++q) {
            const int in = p * cutoff + q;
            if (p + 1 < cutoff && q + 1 < cutoff) {
                const int out = (p + 1) * cutoff + (q + 1);
                generator[out][in] += xi * std::sqrt(static_cast<double>((p + 1) * (q + 1)));
            }
            if (p > 0 && q > 0) {
                const int out = (p - 1) * cutoff + (q - 1);
                generator[out][in] += -std::conj(xi) * std::sqrt(static_cast<double>(p * q));
            }
        }
    }

    Reference::Matrix result = Reference::create_identity_matrix(dim);
    Reference::Matrix term = Reference::create_identity_matrix(dim);
    for (int order = 1; order <= 80; ++order) {
        Reference::Matrix next(dim, Vector(dim, Complex(0.0, 0.0)));
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                Complex sum(0.0, 0.0);
                for (int k = 0; k < dim; ++k) {
                    sum += term[i][k] * generator[k][j];
                }
                next[i][j] = sum / static_cast<double>(order);
            }
        }
        double max_abs = 0.0;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                result[i][j] += next[i][j];
                max_abs = std::max(max_abs, std::abs(next[i][j]));
            }
        }
        term = std::move(next);
        if (max_abs < 1e-14) {
            break;
        }
    }
    return result;
}

void apply_single_mode_cv_cpu(Vector* state,
                              int num_qubits,
                              int num_qumodes,
                              int cutoff,
                              int target_mode,
                              const Reference::Matrix& mat) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const size_t stride = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes - target_mode - 1);
    Vector out(state->size(), Complex(0.0, 0.0));
    std::vector<int> digits;
    for (size_t q = 0; q < dv_dim; ++q) {
        const size_t base = q * cv_dim;
        for (size_t cv = 0; cv < cv_dim; ++cv) {
            const int in_n = static_cast<int>((cv / stride) % static_cast<size_t>(cutoff));
            const size_t cv_without = cv - static_cast<size_t>(in_n) * stride;
            const Complex amp = (*state)[base + cv];
            if (std::abs(amp) == 0.0) {
                continue;
            }
            for (int out_n = 0; out_n < cutoff; ++out_n) {
                const Complex coeff = mat[out_n][in_n];
                if (std::abs(coeff) == 0.0) {
                    continue;
                }
                out[base + cv_without + static_cast<size_t>(out_n) * stride] += coeff * amp;
            }
        }
    }
    *state = std::move(out);
}

void apply_two_mode_cv_cpu(Vector* state,
                           int num_qubits,
                           int num_qumodes,
                           int cutoff,
                           int mode0,
                           int mode1,
                           const Reference::Matrix& mat) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const size_t stride0 = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes - mode0 - 1);
    const size_t stride1 = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes - mode1 - 1);
    Vector out(state->size(), Complex(0.0, 0.0));
    for (size_t q = 0; q < dv_dim; ++q) {
        const size_t base = q * cv_dim;
        for (size_t cv = 0; cv < cv_dim; ++cv) {
            const int in0 = static_cast<int>((cv / stride0) % static_cast<size_t>(cutoff));
            const int in1 = static_cast<int>((cv / stride1) % static_cast<size_t>(cutoff));
            const size_t cv_without =
                cv - static_cast<size_t>(in0) * stride0 - static_cast<size_t>(in1) * stride1;
            const int in_pair = in0 * cutoff + in1;
            const Complex amp = (*state)[base + cv];
            if (std::abs(amp) == 0.0) {
                continue;
            }
            for (int out0 = 0; out0 < cutoff; ++out0) {
                for (int out1 = 0; out1 < cutoff; ++out1) {
                    const int out_pair = out0 * cutoff + out1;
                    const Complex coeff = mat[out_pair][in_pair];
                    if (std::abs(coeff) == 0.0) {
                        continue;
                    }
                    const size_t out_cv =
                        cv_without + static_cast<size_t>(out0) * stride0 +
                        static_cast<size_t>(out1) * stride1;
                    out[base + out_cv] += coeff * amp;
                }
            }
        }
    }
    *state = std::move(out);
}

void apply_controlled_single_mode_cv_cpu(Vector* state,
                                         int num_qubits,
                                         int num_qumodes,
                                         int cutoff,
                                         int control,
                                         int target_mode,
                                         const Reference::Matrix& low_matrix,
                                         const Reference::Matrix& high_matrix) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const size_t bit = static_cast<size_t>(1) << (num_qubits - 1 - control);
    const size_t stride = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes - target_mode - 1);
    Vector out(state->size(), Complex(0.0, 0.0));
    for (size_t q = 0; q < dv_dim; ++q) {
        const auto& mat = (q & bit) ? high_matrix : low_matrix;
        const size_t base = q * cv_dim;
        for (size_t cv = 0; cv < cv_dim; ++cv) {
            const int in_n = static_cast<int>((cv / stride) % static_cast<size_t>(cutoff));
            const size_t cv_without = cv - static_cast<size_t>(in_n) * stride;
            const Complex amp = (*state)[base + cv];
            if (std::abs(amp) == 0.0) {
                continue;
            }
            for (int out_n = 0; out_n < cutoff; ++out_n) {
                const Complex coeff = mat[out_n][in_n];
                if (std::abs(coeff) != 0.0) {
                    out[base + cv_without + static_cast<size_t>(out_n) * stride] += coeff * amp;
                }
            }
        }
    }
    *state = std::move(out);
}

void apply_controlled_two_mode_cv_cpu(Vector* state,
                                      int num_qubits,
                                      int num_qumodes,
                                      int cutoff,
                                      int control,
                                      int mode0,
                                      int mode1,
                                      const Reference::Matrix& low_matrix,
                                      const Reference::Matrix& high_matrix) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const size_t bit = static_cast<size_t>(1) << (num_qubits - 1 - control);
    const size_t stride0 = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes - mode0 - 1);
    const size_t stride1 = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes - mode1 - 1);
    Vector out(state->size(), Complex(0.0, 0.0));
    for (size_t q = 0; q < dv_dim; ++q) {
        const auto& mat = (q & bit) ? high_matrix : low_matrix;
        const size_t base = q * cv_dim;
        for (size_t cv = 0; cv < cv_dim; ++cv) {
            const int in0 = static_cast<int>((cv / stride0) % static_cast<size_t>(cutoff));
            const int in1 = static_cast<int>((cv / stride1) % static_cast<size_t>(cutoff));
            const size_t cv_without =
                cv - static_cast<size_t>(in0) * stride0 - static_cast<size_t>(in1) * stride1;
            const int in_pair = in0 * cutoff + in1;
            const Complex amp = (*state)[base + cv];
            if (std::abs(amp) == 0.0) {
                continue;
            }
            for (int out0 = 0; out0 < cutoff; ++out0) {
                for (int out1 = 0; out1 < cutoff; ++out1) {
                    const int out_pair = out0 * cutoff + out1;
                    const Complex coeff = mat[out_pair][in_pair];
                    if (std::abs(coeff) != 0.0) {
                        out[base + cv_without + static_cast<size_t>(out0) * stride0 +
                            static_cast<size_t>(out1) * stride1] += coeff * amp;
                    }
                }
            }
        }
    }
    *state = std::move(out);
}

void apply_jc_like_cpu(Vector* state,
                       int num_qubits,
                       int num_qumodes,
                       int cutoff,
                       int control,
                       int target_mode,
                       double theta,
                       double phi,
                       bool anti) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const size_t bit = static_cast<size_t>(1) << (num_qubits - 1 - control);
    const size_t stride = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes - target_mode - 1);
    Vector out = *state;
    const Complex factor01_base(-std::sin(phi), -std::cos(phi));
    const Complex factor10_base(std::sin(phi), -std::cos(phi));

    for (size_t q_low = 0; q_low < dv_dim; ++q_low) {
        if (q_low & bit) {
            continue;
        }
        const size_t q_high = q_low | bit;
        const size_t low_base = q_low * cv_dim;
        const size_t high_base = q_high * cv_dim;
        for (size_t cv = 0; cv < cv_dim; ++cv) {
            const int n = static_cast<int>((cv / stride) % static_cast<size_t>(cutoff));
            if (n + 1 >= cutoff) {
                continue;
            }
            const size_t cv_without = cv - static_cast<size_t>(n) * stride;
            const size_t low_cv = anti
                ? cv_without + static_cast<size_t>(n) * stride
                : cv_without + static_cast<size_t>(n + 1) * stride;
            const size_t high_cv = anti
                ? cv_without + static_cast<size_t>(n + 1) * stride
                : cv_without + static_cast<size_t>(n) * stride;
            const double omega = theta * std::sqrt(static_cast<double>(n + 1));
            const double c = std::cos(omega);
            const double s = std::sin(omega);
            const Complex c0 = (*state)[low_base + low_cv];
            const Complex c1 = (*state)[high_base + high_cv];
            out[low_base + low_cv] = c * c0 + (factor01_base * s) * c1;
            out[high_base + high_cv] = (factor10_base * s) * c0 + c * c1;
        }
    }
    *state = std::move(out);
}

void mix_control_branches_cpu(Vector* state,
                              int num_qubits,
                              int num_qumodes,
                              int cutoff,
                              int control) {
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(cutoff), num_qumodes);
    const size_t dv_dim = static_cast<size_t>(1) << num_qubits;
    const size_t bit = static_cast<size_t>(1) << (num_qubits - 1 - control);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    Vector out = *state;
    for (size_t q_low = 0; q_low < dv_dim; ++q_low) {
        if (q_low & bit) {
            continue;
        }
        const size_t q_high = q_low | bit;
        const size_t low_base = q_low * cv_dim;
        const size_t high_base = q_high * cv_dim;
        for (size_t cv = 0; cv < cv_dim; ++cv) {
            const Complex v0 = (*state)[low_base + cv];
            const Complex v1 = (*state)[high_base + cv];
            out[low_base + cv] = (v0 + v1) * inv_sqrt2;
            out[high_base + cv] = (v0 - v1) * inv_sqrt2;
        }
    }
    *state = std::move(out);
}

void apply_rabi_cpu(Vector* state,
                    int num_qubits,
                    int num_qumodes,
                    int cutoff,
                    int control,
                    int target_mode,
                    double theta) {
    mix_control_branches_cpu(state, num_qubits, num_qumodes, cutoff, control);
    GateParams low_gate = Gates::Displacement(target_mode, Complex(0.0, -theta));
    GateParams high_gate = Gates::Displacement(target_mode, Complex(0.0, theta));
    apply_controlled_single_mode_cv_cpu(
        state,
        num_qubits,
        num_qumodes,
        cutoff,
        control,
        target_mode,
        cv_single_mode_matrix(low_gate, cutoff),
        cv_single_mode_matrix(high_gate, cutoff));
    mix_control_branches_cpu(state, num_qubits, num_qumodes, cutoff, control);
}

bool apply_gate_cpu(Vector* state,
                    int num_qubits,
                    int num_qumodes,
                    int cutoff,
                    const GateParams& gate,
                    std::string* unsupported_reason = nullptr) {
    try {
        switch (gate.type) {
            case GateType::HADAMARD:
            case GateType::PAULI_X:
            case GateType::PAULI_Y:
            case GateType::PAULI_Z:
            case GateType::ROTATION_X:
            case GateType::ROTATION_Y:
            case GateType::ROTATION_Z:
            case GateType::PHASE_GATE_S:
            case GateType::PHASE_GATE_T:
                apply_single_qubit_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), single_qubit_matrix(gate));
                return true;
            case GateType::CNOT:
                apply_cnot_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qubits.at(1));
                return true;
            case GateType::CZ:
                apply_cz_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qubits.at(1));
                return true;
            case GateType::PHASE_ROTATION:
            case GateType::KERR_GATE:
            case GateType::CONDITIONAL_PARITY:
            case GateType::SNAP_GATE:
            case GateType::MULTI_SNAP_GATE:
            case GateType::CREATION_OPERATOR:
            case GateType::ANNIHILATION_OPERATOR:
            case GateType::DISPLACEMENT:
            case GateType::SQUEEZING:
                apply_single_mode_cv_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qumodes.at(0), cv_single_mode_matrix(gate, cutoff));
                return true;
            case GateType::BEAM_SPLITTER:
            case GateType::CROSS_KERR_GATE:
                apply_two_mode_cv_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qumodes.at(0), gate.target_qumodes.at(1),
                    cv_two_mode_matrix(gate, cutoff));
                return true;
            case GateType::CONDITIONAL_DISPLACEMENT: {
                GateParams low_gate = Gates::Displacement(gate.target_qumodes.at(0), gate.params.at(0));
                GateParams high_gate = Gates::Displacement(gate.target_qumodes.at(0), -gate.params.at(0));
                apply_controlled_single_mode_cv_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qumodes.at(0),
                    cv_single_mode_matrix(low_gate, cutoff),
                    cv_single_mode_matrix(high_gate, cutoff));
                return true;
            }
            case GateType::CONDITIONAL_SQUEEZING: {
                GateParams low_gate = Gates::Squeezing(gate.target_qumodes.at(0), gate.params.at(0));
                GateParams high_gate = Gates::Squeezing(gate.target_qumodes.at(0), -gate.params.at(0));
                apply_controlled_single_mode_cv_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qumodes.at(0),
                    cv_single_mode_matrix(low_gate, cutoff),
                    cv_single_mode_matrix(high_gate, cutoff));
                return true;
            }
            case GateType::CONDITIONAL_BEAM_SPLITTER: {
                GateParams low_gate = Gates::BeamSplitter(
                    gate.target_qumodes.at(0), gate.target_qumodes.at(1),
                    gate.params.at(0).real(), gate.params.size() > 1 ? gate.params.at(1).real() : 0.0);
                GateParams high_gate = Gates::BeamSplitter(
                    gate.target_qumodes.at(0), gate.target_qumodes.at(1),
                    -gate.params.at(0).real(), gate.params.size() > 1 ? gate.params.at(1).real() : 0.0);
                apply_controlled_two_mode_cv_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qumodes.at(0), gate.target_qumodes.at(1),
                    cv_two_mode_matrix(low_gate, cutoff),
                    cv_two_mode_matrix(high_gate, cutoff));
                return true;
            }
            case GateType::CONDITIONAL_TWO_MODE_SQUEEZING: {
                const Reference::Matrix low = two_mode_squeezing_matrix(cutoff, gate.params.at(0));
                const Reference::Matrix high = two_mode_squeezing_matrix(cutoff, -gate.params.at(0));
                apply_controlled_two_mode_cv_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qumodes.at(0), gate.target_qumodes.at(1),
                    low, high);
                return true;
            }
            case GateType::JAYNES_CUMMINGS:
                apply_jc_like_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qumodes.at(0),
                    gate.params.empty() ? 0.0 : gate.params.at(0).real(),
                    gate.params.size() > 1 ? gate.params.at(1).real() : 0.0,
                    false);
                return true;
            case GateType::ANTI_JAYNES_CUMMINGS:
                apply_jc_like_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qumodes.at(0),
                    gate.params.empty() ? 0.0 : gate.params.at(0).real(),
                    gate.params.size() > 1 ? gate.params.at(1).real() : 0.0,
                    true);
                return true;
            case GateType::RABI_INTERACTION:
                apply_rabi_cpu(
                    state, num_qubits, num_qumodes, cutoff,
                    gate.target_qubits.at(0), gate.target_qumodes.at(0),
                    gate.params.empty() ? 0.0 : gate.params.at(0).real());
                return true;
            default:
                if (unsupported_reason) {
                    *unsupported_reason = "CPU dense harness does not support this hybrid gate type";
                }
                return false;
        }
    } catch (const std::exception& e) {
        if (unsupported_reason) {
            *unsupported_reason = e.what();
        }
        return false;
    }
}

std::vector<GateParams> inverse_gates(const GateParams& gate) {
    GateParams inv = gate;
    switch (gate.type) {
        case GateType::HADAMARD:
        case GateType::PAULI_X:
        case GateType::PAULI_Y:
        case GateType::PAULI_Z:
            return {inv};
        case GateType::PHASE_GATE_S:
            return {
                Gates::PhaseGateS(gate.target_qubits.at(0)),
                Gates::PhaseGateS(gate.target_qubits.at(0)),
                Gates::PhaseGateS(gate.target_qubits.at(0))
            };
        case GateType::PHASE_GATE_T:
            return {
                Gates::PhaseGateT(gate.target_qubits.at(0)),
                Gates::PhaseGateT(gate.target_qubits.at(0)),
                Gates::PhaseGateT(gate.target_qubits.at(0)),
                Gates::PhaseGateT(gate.target_qubits.at(0)),
                Gates::PhaseGateT(gate.target_qubits.at(0)),
                Gates::PhaseGateT(gate.target_qubits.at(0)),
                Gates::PhaseGateT(gate.target_qubits.at(0))
            };
        case GateType::ROTATION_X:
        case GateType::ROTATION_Y:
        case GateType::ROTATION_Z:
        case GateType::PHASE_ROTATION:
        case GateType::KERR_GATE:
        case GateType::CONDITIONAL_PARITY:
        case GateType::CROSS_KERR_GATE:
        case GateType::DISPLACEMENT:
        case GateType::SQUEEZING:
        case GateType::BEAM_SPLITTER:
        case GateType::CONDITIONAL_DISPLACEMENT:
        case GateType::CONDITIONAL_SQUEEZING:
        case GateType::CONDITIONAL_BEAM_SPLITTER:
        case GateType::CONDITIONAL_TWO_MODE_SQUEEZING:
        case GateType::JAYNES_CUMMINGS:
        case GateType::ANTI_JAYNES_CUMMINGS:
        case GateType::RABI_INTERACTION:
            for (auto& p : inv.params) {
                p = -p;
            }
            return {inv};
        case GateType::SNAP_GATE:
            inv.params.at(0) = -inv.params.at(0);
            return {inv};
        case GateType::MULTI_SNAP_GATE:
            for (auto& p : inv.params) {
                p = -p;
            }
            return {inv};
        case GateType::CNOT:
        case GateType::CZ:
            return {inv};
        default:
            throw std::runtime_error("inverse not available for this gate type");
    }
}

bool append_inverse_gates(const std::vector<GateParams>& gates,
                          std::vector<GateParams>* out,
                          std::string* reason) {
    *out = gates;
    for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
        try {
            std::vector<GateParams> inv = inverse_gates(*it);
            out->insert(out->end(), inv.begin(), inv.end());
        } catch (const std::exception& e) {
            if (reason) {
                *reason = e.what();
            }
            return false;
        }
    }
    return true;
}

void add_cv_qaoa_gates(std::vector<GateParams>* gates,
                       int num_qumodes,
                       int layers,
                       double s = 0.5,
                       double a = 1.0) {
    std::vector<double> params(2 * layers, 0.0);
    for (int i = 0; i < 2 * layers; ++i) {
        params[i] = 2.0 * M_PI * static_cast<double>(i + 1) / static_cast<double>(2 * layers);
    }
    for (int qm = 0; qm < num_qumodes; ++qm) {
        gates->push_back(Gates::Squeezing(qm, Complex(s, 0.0)));
    }
    for (int i = 0; i < layers; ++i) {
        const double gamma = params[i];
        const double eta = params[layers + i];
        for (int qm = 0; qm < num_qumodes; ++qm) {
            gates->push_back(Gates::Displacement(qm, Complex(a * gamma, 0.0)));
        }
        for (int qm = 0; qm < num_qumodes; ++qm) {
            gates->push_back(Gates::Squeezing(qm, Complex(eta, 0.0)));
        }
    }
}

void add_cv_qaoa_low_r_gates(std::vector<GateParams>* gates,
                             int num_qumodes,
                             int layers,
                             double s = 0.5,
                             double a = 1.0,
                             double eta_max = 0.5) {
    std::vector<double> gammas(static_cast<size_t>(layers), 0.0);
    for (int i = 0; i < layers; ++i) {
        gammas[static_cast<size_t>(i)] =
            2.0 * M_PI * static_cast<double>(i + 1) / static_cast<double>(2 * layers);
    }
    for (int qm = 0; qm < num_qumodes; ++qm) {
        gates->push_back(Gates::Squeezing(qm, Complex(s, 0.0)));
    }
    for (int i = 0; i < layers; ++i) {
        const double gamma = gammas[static_cast<size_t>(i)];
        const double eta = eta_max * static_cast<double>(i + 1) / static_cast<double>(layers);
        for (int qm = 0; qm < num_qumodes; ++qm) {
            gates->push_back(Gates::Displacement(qm, Complex(a * gamma, 0.0)));
        }
        for (int qm = 0; qm < num_qumodes; ++qm) {
            gates->push_back(Gates::Squeezing(qm, Complex(eta, 0.0)));
        }
    }
}

void add_cat_gates(std::vector<GateParams>* gates, double alpha, int qumode_idx) {
    gates->push_back(Gates::Hadamard(0));
    gates->push_back(Gates::ConditionalDisplacement(0, qumode_idx, Complex(alpha / std::sqrt(2.0), 0.0)));
    gates->push_back(Gates::Hadamard(0));
    gates->push_back(Gates::PhaseGateS(0));
    gates->push_back(Gates::Hadamard(0));
    gates->push_back(Gates::ConditionalDisplacement(
        0, qumode_idx, Complex(0.0, M_PI / (8.0 * alpha * std::sqrt(2.0)))));
    gates->push_back(Gates::Hadamard(0));
    gates->push_back(Gates::PhaseGateS(0));
}

void add_gkp_gates(std::vector<GateParams>* gates, int rounds, double r, int qumode_idx) {
    const double alpha = std::sqrt(M_PI);
    gates->push_back(Gates::Squeezing(qumode_idx, Complex(r, 0.0)));
    for (int i = 1; i < rounds; ++i) {
        gates->push_back(Gates::Hadamard(0));
        gates->push_back(Gates::ConditionalDisplacement(
            0, qumode_idx, Complex(alpha / std::sqrt(2.0), 0.0)));
        gates->push_back(Gates::Hadamard(0));
        gates->push_back(Gates::PhaseGateS(0));
        gates->push_back(Gates::Hadamard(0));
        gates->push_back(Gates::ConditionalDisplacement(
            0, qumode_idx, Complex(0.0, M_PI / (8.0 * alpha * std::sqrt(2.0)))));
        gates->push_back(Gates::Hadamard(0));
        gates->push_back(Gates::PhaseGateS(0));
    }
}

void add_basis_gates(std::vector<GateParams>* gates, int num_qubits) {
    for (int i = 0; i < num_qubits; ++i) {
        gates->push_back(Gates::Hadamard(i));
        if (i % 3 == 0) {
            gates->push_back(Gates::PauliX(i));
            gates->push_back(Gates::PauliZ(i));
        } else if (i % 3 == 1) {
            gates->push_back(Gates::PauliZ(i));
        } else {
            gates->push_back(Gates::PauliX(i));
        }
    }
}

void add_basis_reverse_gates(std::vector<GateParams>* gates, int num_qubits) {
    for (int i = num_qubits - 1; i >= 0; --i) {
        if (i % 3 == 0) {
            gates->push_back(Gates::PauliZ(i));
            gates->push_back(Gates::PauliX(i));
            gates->push_back(Gates::Hadamard(i));
        } else if (i % 3 == 1) {
            gates->push_back(Gates::PauliZ(i));
            gates->push_back(Gates::Hadamard(i));
        } else {
            gates->push_back(Gates::PauliX(i));
            gates->push_back(Gates::Hadamard(i));
        }
    }
}

void add_state_transfer_cvtodv_gates(std::vector<GateParams>* gates,
                                     int num_qubits,
                                     int num_qumodes,
                                     double lambda,
                                     bool apply_basis) {
    for (int q = 0; q < num_qubits; ++q) {
        gates->push_back(Gates::RotationX(q, M_PI / 4.0));
    }
    for (int qm = 0; qm < num_qumodes; ++qm) {
        gates->push_back(Gates::Displacement(qm, Complex(lambda, 0.0)));
    }
    for (int q = 0; q < num_qubits; ++q) {
        gates->push_back(Gates::RotationZ(q, M_PI / 4.0));
    }
    for (int qm = 0; qm < num_qumodes; ++qm) {
        gates->push_back(Gates::Squeezing(qm, Complex(lambda, 0.0)));
    }
    if (apply_basis) {
        add_basis_gates(gates, num_qubits);
    }
}

void add_state_transfer_dvtocv_gates(std::vector<GateParams>* gates,
                                     int num_qubits,
                                     int num_qumodes,
                                     double lambda,
                                     bool apply_basis) {
    if (apply_basis) {
        add_basis_reverse_gates(gates, num_qubits);
    }
    for (int q = 0; q < num_qubits; ++q) {
        gates->push_back(Gates::RotationZ(q, M_PI / 4.0));
    }
    for (int qm = 0; qm < num_qumodes; ++qm) {
        gates->push_back(Gates::Squeezing(qm, Complex(lambda, 0.0)));
    }
    for (int q = 0; q < num_qubits; ++q) {
        gates->push_back(Gates::RotationX(q, M_PI / 4.0));
    }
    for (int qm = 0; qm < num_qumodes; ++qm) {
        gates->push_back(Gates::Displacement(qm, Complex(lambda, 0.0)));
    }
}

void add_jch_gates(std::vector<GateParams>* gates,
                   int nsites,
                   int nqubits,
                   double j,
                   double omega_r,
                   double omega_q,
                   double g,
                   double tau,
                   int timesteps) {
    for (int t = 0; t < timesteps; ++t) {
        for (int i = 0; i < nsites; ++i) {
            gates->push_back(Gates::PhaseRotation(i, omega_r * tau));
        }
        for (int i = 0; i < nqubits; ++i) {
            gates->push_back(Gates::RotationZ(i, omega_q * tau / 2.0));
        }
        for (int i = 0; i < std::min(nsites, nqubits); ++i) {
            gates->push_back(Gates::JaynesCummings(i, i, g * tau, 0.0));
        }
        for (int i = 0; i < nsites - 1; ++i) {
            gates->push_back(Gates::BeamSplitter(i, i + 1, j * tau, 0.0));
        }
    }
}

std::vector<double> make_vqe_parameters(int depth, int num_qubits, int num_qumodes) {
    const int params_per_layer = 2 * num_qumodes + 2 * num_qubits;
    std::vector<double> params;
    params.reserve(static_cast<size_t>(depth * params_per_layer));
    for (int d = 0; d < depth; ++d) {
        for (int qm = 0; qm < num_qumodes; ++qm) {
            params.push_back(0.4 + 0.1 * static_cast<double>(d + qm));
            params.push_back(M_PI / 6.0 + 0.05 * static_cast<double>(d + qm));
        }
        for (int q = 0; q < num_qubits; ++q) {
            params.push_back(M_PI / 7.0 + 0.03 * static_cast<double>(d + q));
            params.push_back(M_PI / 5.0 + 0.02 * static_cast<double>(d + q));
        }
    }
    return params;
}

void add_vqe_gates(std::vector<GateParams>* gates,
                   int num_qubits,
                   int num_qumodes,
                   int depth,
                   const std::vector<double>& params) {
    size_t idx = 0;
    for (int d = 0; d < depth; ++d) {
        for (int qm = 0; qm < num_qumodes; ++qm) {
            gates->push_back(Gates::Displacement(qm, Complex(params[idx++], 0.0)));
            gates->push_back(Gates::Squeezing(qm, Complex(params[idx++], 0.0)));
        }
        for (int q = 0; q < num_qubits; ++q) {
            gates->push_back(Gates::RotationX(q, params[idx++]));
            gates->push_back(Gates::RotationZ(q, params[idx++]));
        }
        for (int q = 0; q < num_qubits; ++q) {
            const int qm = q % std::max(1, num_qumodes);
            gates->push_back(Gates::JaynesCummings(q, qm, M_PI / 4.0, 0.0));
        }
    }
}

void add_qft_gates(std::vector<GateParams>* gates, int num_qubits, int n, int a, int append) {
    const int total = n + a + append;
    (void)num_qubits;
    for (int i = 0; i < a; ++i) {
        gates->push_back(Gates::Hadamard(i));
    }
    for (int q = 0; q < total; ++q) {
        gates->push_back(Gates::RotationX(q, M_PI / 4.0));
    }
    gates->push_back(Gates::Displacement(0, Complex(0.29, 0.0)));
    gates->push_back(Gates::PhaseRotation(0, M_PI / 2.0));
    for (int q = 0; q < total; ++q) {
        gates->push_back(Gates::RotationZ(q, M_PI / 4.0));
    }
    gates->push_back(Gates::Squeezing(0, Complex(0.29, 0.0)));
    for (int i = 0; i < n; ++i) {
        gates->push_back(Gates::Hadamard(a + i));
    }
}

void add_shors_gates(std::vector<GateParams>* gates, int num_qumodes) {
    for (int qm = 0; qm < std::min(num_qumodes, 2); ++qm) {
        gates->push_back(Gates::Squeezing(qm, Complex(0.222, 0.0)));
        for (int i = 0; i < 2; ++i) {
            gates->push_back(Gates::Hadamard(0));
            gates->push_back(Gates::ConditionalDisplacement(0, qm, Complex(0.5, 0.0)));
        }
    }
    for (int i = 0; i < 2; ++i) {
        gates->push_back(Gates::JaynesCummings(0, 0, M_PI / 4.0, 0.0));
        gates->push_back(Gates::JaynesCummings(0, 1, M_PI / 4.0, 0.0));
    }
}

CircuitSpec make_spec(const std::string& name,
                      const std::string& workload,
                      int num_qubits,
                      int num_qumodes,
                      int cutoff) {
    CircuitSpec spec;
    spec.name = name;
    spec.workload = workload;
    spec.num_qubits = num_qubits;
    spec.num_qumodes = num_qumodes;
    spec.cutoff = cutoff;

    if (workload == "cat_state_circuit") {
        spec.depth = 8;
        add_cat_gates(&spec.gates, 1.0, 0);
    } else if (workload == "gkp_state_circuit") {
        spec.depth = 9;
        add_gkp_gates(&spec.gates, 9, 0.222, 0);
    } else if (workload == "qaoa_circuit" || workload == "cv_qaoa_circuit") {
        spec.depth = 2;
        spec.initial_profile = "vacuum";
        add_cv_qaoa_gates(&spec.gates, num_qumodes, 2);
    } else if (workload == "qaoa_low_r_circuit") {
        spec.depth = 2;
        spec.initial_profile = "vacuum";
        add_cv_qaoa_low_r_gates(&spec.gates, num_qumodes, 2);
    } else if (workload == "qft_circuit") {
        spec.depth = 10;
        add_qft_gates(&spec.gates, num_qubits, 2, 1, std::max(0, num_qubits - 3));
    } else if (workload == "shors_circuit") {
        spec.depth = 10;
        add_shors_gates(&spec.gates, num_qumodes);
    } else if (workload == "state_transfer_CVtoDV_circuit") {
        spec.depth = 8;
        add_state_transfer_cvtodv_gates(&spec.gates, num_qubits, num_qumodes, 0.29, true);
    } else if (workload == "state_transfer_DVtoCV_circuit") {
        spec.depth = 8;
        add_state_transfer_dvtocv_gates(&spec.gates, num_qubits, num_qumodes, 0.29, true);
    } else if (workload == "jch_simulation_circuit") {
        spec.depth = 5;
        add_jch_gates(&spec.gates, num_qumodes, num_qubits, 1.0, 1.0, 1.0, 0.5, 0.1, 5);
    } else if (workload == "vqe_circuit") {
        spec.depth = 2;
        add_vqe_gates(&spec.gates, num_qubits, num_qumodes, 2,
                      make_vqe_parameters(2, num_qubits, num_qumodes));
    } else {
        throw std::runtime_error("unknown workload: " + workload);
    }
    return spec;
}

std::vector<CircuitSpec> make_sc26_specs() {
    std::vector<CircuitSpec> specs;
    for (int cutoff : {16, 32}) {
        specs.push_back(make_spec("sc26_cat_c" + std::to_string(cutoff),
                                  "cat_state_circuit", 1, 1, cutoff));
        specs.push_back(make_spec("sc26_gkp_c" + std::to_string(cutoff),
                                  "gkp_state_circuit", 1, 1, cutoff));
    }
    for (int nm : {1, 2, 4, 6, 7, 8}) {
        specs.push_back(make_spec("sc26_qaoa_nm" + std::to_string(nm) + "_c16",
                                  "qaoa_circuit", 1, nm, 16));
    }
    for (int nq : {3, 5, 7, 9}) {
        specs.push_back(make_spec("sc26_qft_nq" + std::to_string(nq) + "_c16",
                                  "qft_circuit", nq, 1, 16));
    }
    for (int cutoff : {8, 16}) {
        specs.push_back(make_spec("sc26_shors_c" + std::to_string(cutoff),
                                  "shors_circuit", 1, 3, cutoff));
    }
    for (int nq : {2, 4, 8, 16}) {
        specs.push_back(make_spec("sc26_transfer_CVtoDV_nq" + std::to_string(nq) + "_c16",
                                  "state_transfer_CVtoDV_circuit", nq, 1, 16));
        specs.push_back(make_spec("sc26_transfer_DVtoCV_nq" + std::to_string(nq) + "_c16",
                                  "state_transfer_DVtoCV_circuit", nq, 1, 16));
    }
    for (int cutoff : {4, 8, 16, 32}) {
        for (int nq : {3, 6, 10}) {
            for (int nm : {2, 4, 7}) {
                specs.push_back(make_spec("sc26_jch_nq" + std::to_string(nq) +
                                              "_nm" + std::to_string(nm) +
                                              "_c" + std::to_string(cutoff),
                                          "jch_simulation_circuit", nq, nm, cutoff));
                specs.push_back(make_spec("sc26_vqe_nq" + std::to_string(nq) +
                                              "_nm" + std::to_string(nm) +
                                              "_c" + std::to_string(cutoff),
                                          "vqe_circuit", nq, nm, cutoff));
            }
        }
    }
    return specs;
}

std::vector<CircuitSpec> make_convergence_specs() {
    std::vector<CircuitSpec> specs;
    for (int cutoff : {8, 16, 32, 64}) {
        specs.push_back(make_spec("conv_cat_c" + std::to_string(cutoff),
                                  "cat_state_circuit", 1, 1, cutoff));
        specs.push_back(make_spec("conv_gkp_c" + std::to_string(cutoff),
                                  "gkp_state_circuit", 1, 1, cutoff));
        specs.push_back(make_spec("conv_qaoa_nm1_c" + std::to_string(cutoff),
                                  "qaoa_circuit", 1, 1, cutoff));
        specs.push_back(make_spec("conv_qaoa_low_r_nm1_c" + std::to_string(cutoff),
                                  "qaoa_low_r_circuit", 1, 1, cutoff));
        specs.push_back(make_spec("conv_vqe_nq3_nm2_c" + std::to_string(cutoff),
                                  "vqe_circuit", 3, 2, cutoff));
        specs.push_back(make_spec("conv_jch_nq3_nm2_c" + std::to_string(cutoff),
                                  "jch_simulation_circuit", 3, 2, cutoff));
        specs.push_back(make_spec("conv_transfer_CVtoDV_nq4_c" + std::to_string(cutoff),
                                  "state_transfer_CVtoDV_circuit", 4, 1, cutoff));
        specs.push_back(make_spec("conv_transfer_DVtoCV_nq4_c" + std::to_string(cutoff),
                                  "state_transfer_DVtoCV_circuit", 4, 1, cutoff));
    }
    specs.push_back(make_spec("conv_gkp_c128", "gkp_state_circuit", 1, 1, 128));
    for (int cutoff : {96, 112, 128, 144, 160, 192, 224, 256}) {
        specs.push_back(make_spec("conv_vqe_nq3_nm2_c" + std::to_string(cutoff),
                                  "vqe_circuit", 3, 2, cutoff));
    }
    for (int cutoff : {4, 8, 16, 32}) {
        specs.push_back(make_spec("conv_qaoa_nm2_c" + std::to_string(cutoff),
                                  "qaoa_circuit", 1, 2, cutoff));
        specs.push_back(make_spec("conv_qaoa_low_r_nm2_c" + std::to_string(cutoff),
                                  "qaoa_low_r_circuit", 1, 2, cutoff));
    }
    for (int cutoff : {128}) {
        specs.push_back(make_spec("conv_qaoa_low_r_nm1_c" + std::to_string(cutoff),
                                  "qaoa_low_r_circuit", 1, 1, cutoff));
    }
    for (int cutoff : {64, 128}) {
        specs.push_back(make_spec("conv_qaoa_low_r_nm2_c" + std::to_string(cutoff),
                                  "qaoa_low_r_circuit", 1, 2, cutoff));
    }
    return specs;
}

std::vector<CircuitSpec> make_gate_specs() {
    std::vector<CircuitSpec> specs;
    auto one = [](const std::string& name, GateParams gate, int cutoff = 16) {
        CircuitSpec spec;
        spec.name = name;
        spec.workload = "gate_micro_precision";
        spec.num_qubits = 0;
        spec.num_qumodes = 1;
        spec.cutoff = cutoff;
        spec.depth = 1;
        spec.gates.push_back(std::move(gate));
        return spec;
    };
    specs.push_back(one("gate_phase_rotation", Gates::PhaseRotation(0, M_PI / 5.0)));
    specs.push_back(one("gate_kerr", Gates::KerrGate(0, 0.07)));
    specs.push_back(one("gate_conditional_parity", Gates::ConditionalParity(0, 0.37)));
    specs.push_back(one("gate_snap", Gates::Snap(0, 0.11, 3)));
    specs.push_back(one("gate_multisnap", Gates::MultiSNAP(0, {0.0, 0.02, -0.04, 0.07, -0.01})));
    specs.push_back(one("gate_creation", Gates::CreationOperator(0)));
    specs.push_back(one("gate_annihilation", Gates::AnnihilationOperator(0)));
    specs.push_back(one("gate_displacement", Gates::Displacement(0, Complex(0.15, -0.08))));
    specs.push_back(one("gate_squeezing", Gates::Squeezing(0, Complex(0.10, 0.02))));

    auto controlled_single = [](const std::string& name, GateParams gate, int cutoff = 12) {
        CircuitSpec spec;
        spec.name = name;
        spec.workload = "gate_micro_precision";
        spec.num_qubits = 1;
        spec.num_qumodes = 1;
        spec.cutoff = cutoff;
        spec.depth = 1;
        spec.gates.push_back(Gates::Hadamard(0));
        spec.gates.push_back(std::move(gate));
        return spec;
    };
    specs.push_back(controlled_single(
        "gate_conditional_displacement",
        Gates::ConditionalDisplacement(0, 0, Complex(0.12, -0.05))));
    specs.push_back(controlled_single(
        "gate_conditional_squeezing",
        Gates::ConditionalSqueezing(0, 0, Complex(0.08, 0.03))));

    CircuitSpec bs;
    bs.name = "gate_beam_splitter";
    bs.workload = "gate_micro_precision";
    bs.num_qubits = 0;
    bs.num_qumodes = 2;
    bs.cutoff = 12;
    bs.depth = 1;
    bs.gates.push_back(Gates::BeamSplitter(0, 1, M_PI / 6.0, 0.0));
    specs.push_back(bs);

    CircuitSpec ck = bs;
    ck.name = "gate_cross_kerr";
    ck.gates.clear();
    ck.gates.push_back(Gates::CrossKerr(0, 1, 0.09));
    specs.push_back(ck);

    auto controlled_two = [](const std::string& name, GateParams gate, int cutoff = 8) {
        CircuitSpec spec;
        spec.name = name;
        spec.workload = "gate_micro_precision";
        spec.num_qubits = 1;
        spec.num_qumodes = 2;
        spec.cutoff = cutoff;
        spec.depth = 1;
        spec.gates.push_back(Gates::Hadamard(0));
        spec.gates.push_back(std::move(gate));
        return spec;
    };
    specs.push_back(controlled_two(
        "gate_conditional_beam_splitter",
        Gates::ConditionalBeamSplitter(0, 0, 1, M_PI / 7.0, 0.0)));
    specs.push_back(controlled_two(
        "gate_conditional_two_mode_squeezing",
        Gates::ConditionalTwoModeSqueezing(0, 0, 1, Complex(0.05, 0.02))));

    auto hybrid_single = [](const std::string& name, GateParams gate, int cutoff = 12) {
        CircuitSpec spec;
        spec.name = name;
        spec.workload = "gate_micro_precision";
        spec.num_qubits = 1;
        spec.num_qumodes = 1;
        spec.cutoff = cutoff;
        spec.depth = 1;
        spec.gates.push_back(Gates::Hadamard(0));
        spec.gates.push_back(std::move(gate));
        return spec;
    };
    specs.push_back(hybrid_single(
        "gate_jaynes_cummings",
        Gates::JaynesCummings(0, 0, 0.21, 0.13)));
    specs.push_back(hybrid_single(
        "gate_anti_jaynes_cummings",
        Gates::AntiJaynesCummings(0, 0, 0.19, -0.11)));
    specs.push_back(hybrid_single(
        "gate_rabi_interaction",
        Gates::RabiInteraction(0, 0, 0.17)));

    for (int q = 0; q < 1; ++q) {
        (void)q;
        CircuitSpec h;
        h.name = "gate_hadamard";
        h.workload = "gate_micro_precision";
        h.num_qubits = 1;
        h.num_qumodes = 1;
        h.cutoff = 8;
        h.depth = 1;
        h.gates.push_back(Gates::Hadamard(0));
        specs.push_back(h);

        CircuitSpec rx = h;
        rx.name = "gate_rotation_x";
        rx.gates.clear();
        rx.gates.push_back(Gates::RotationX(0, 0.33));
        specs.push_back(rx);
    }
    return specs;
}

std::vector<CircuitSpec> make_high_cutoff_gate_specs() {
    std::vector<CircuitSpec> specs;
    auto one = [](const std::string& name, GateParams gate, int cutoff, const std::string& initial_profile) {
        CircuitSpec spec;
        spec.name = name + "_" + initial_profile + "_c" + std::to_string(cutoff);
        spec.workload = "high_cutoff_gate_precision";
        spec.num_qubits = 0;
        spec.num_qumodes = 1;
        spec.cutoff = cutoff;
        spec.depth = 1;
        spec.initial_profile = initial_profile;
        spec.gates.push_back(std::move(gate));
        return spec;
    };
    auto add_profiles = [&](const std::string& name, GateParams gate, int cutoff) {
        specs.push_back(one(name, gate, cutoff, "low"));
        specs.push_back(one(name, gate, cutoff, "stress"));
    };

    for (int cutoff : {32, 64, 128, 256}) {
        add_profiles("highcutoff_phase_rotation", Gates::PhaseRotation(0, M_PI / 5.0), cutoff);
        add_profiles("highcutoff_displacement_0p1",
                     Gates::Displacement(0, Complex(0.1, 0.0)), cutoff);
        add_profiles("highcutoff_displacement_0p5",
                     Gates::Displacement(0, Complex(0.5, 0.0)), cutoff);
        add_profiles("highcutoff_displacement_pi_over_2",
                     Gates::Displacement(0, Complex(M_PI / 2.0, 0.0)), cutoff);
        add_profiles("highcutoff_displacement_pi",
                     Gates::Displacement(0, Complex(M_PI, 0.0)), cutoff);
        add_profiles("highcutoff_squeezing_0p25",
                     Gates::Squeezing(0, Complex(0.25, 0.0)), cutoff);
        add_profiles("highcutoff_squeezing_0p5",
                     Gates::Squeezing(0, Complex(0.5, 0.0)), cutoff);
    }
    return specs;
}

Vector make_probe_initial_state(const CircuitSpec& spec) {
    Vector state = initial_full_state(spec.num_qubits, spec.num_qumodes, spec.cutoff);
    if (spec.initial_profile == "vacuum") {
        return state;
    }
    const size_t cv_dim = checked_pow_size(static_cast<size_t>(spec.cutoff), spec.num_qumodes);
    std::vector<size_t> populated_indices;
    const int low_populated = std::min<int>(static_cast<int>(cv_dim), 8);
    for (int i = 0; i < low_populated; ++i) {
        populated_indices.push_back(static_cast<size_t>(i));
    }
    if (spec.initial_profile == "stress" && spec.num_qubits == 0 && spec.num_qumodes == 1) {
        for (int idx : {spec.cutoff / 4, spec.cutoff / 2, (3 * spec.cutoff) / 4}) {
            if (idx >= 0 && static_cast<size_t>(idx) < cv_dim &&
                std::find(populated_indices.begin(), populated_indices.end(), static_cast<size_t>(idx)) ==
                    populated_indices.end()) {
                populated_indices.push_back(static_cast<size_t>(idx));
            }
        }
    }
    double norm_sq = 0.0;
    for (size_t i = 0; i < populated_indices.size(); ++i) {
        const size_t basis_index = populated_indices[i];
        const Complex amp(1.0 / static_cast<double>(i + 1), 0.07 * static_cast<double>(basis_index % 17));
        state[basis_index] = amp;
        norm_sq += std::norm(amp);
    }
    const double norm = std::sqrt(norm_sq);
    if (norm > 0.0) {
        for (auto& amp : state) {
            amp /= norm;
        }
    }
    return state;
}

Result run_cpu_gpu_precision_case(const CircuitSpec& spec,
                                  const Options& options,
                                  bool per_gate) {
    Result result;
    result.name = spec.name + (per_gate ? "_per_gate" : "_full_circuit");
    result.category = per_gate ? "gate_cpu_dense_vs_gpu" : "circuit_cpu_dense_vs_gpu";
    result.params["workload"] = spec.workload;
    result.params["num_qubits"] = std::to_string(spec.num_qubits);
    result.params["num_qumodes"] = std::to_string(spec.num_qumodes);
    result.params["cutoff"] = std::to_string(spec.cutoff);
    result.params["num_gates"] = std::to_string(spec.gates.size());
    result.params["initial_profile"] = spec.initial_profile;
    result.params["force_dense_fock"] = options.force_dense_fock ? "true" : "false";
    result.params["gaussian_symbolic_enabled"] = options.disable_symbolic ? "false" : "true";
    result.metrics["state_dim"] =
        static_cast<double>(total_dimension(spec.num_qubits, spec.num_qumodes, spec.cutoff));

    const bool gaussian_only =
        is_gaussian_vacuum_track_candidate(spec) &&
        !options.disable_symbolic &&
        !per_gate;
    if (gaussian_only && !per_gate) {
        const GaussianDiagnostics diagnostics = run_cpu_gaussian_vacuum_track(spec);
        if (diagnostics.available) {
            result.metrics["fidelity"] = diagnostics.vacuum_fidelity;
            result.metrics["gpu_norm"] = diagnostics.weight_abs;
            result.metrics["reference_norm"] = 1.0;
            result.metrics["max_l2_error"] = diagnostics.displacement_l2;
            result.metrics["max_abs_error"] = diagnostics.covariance_max_abs_delta;
            result.metrics["max_fidelity_deviation"] = diagnostics.vacuum_fidelity_deviation;
            result.metrics["mean_l2_error"] = diagnostics.displacement_l2;
            result.metrics["gpu_gaussian_vacuum_fidelity"] = diagnostics.vacuum_fidelity;
            result.metrics["gpu_gaussian_vacuum_fidelity_deviation"] =
                diagnostics.vacuum_fidelity_deviation;
            result.metrics["gpu_gaussian_displacement_l2"] = diagnostics.displacement_l2;
            result.metrics["gpu_gaussian_covariance_max_abs_delta"] =
                diagnostics.covariance_max_abs_delta;
            result.metrics["gpu_gaussian_covariance_fro_delta"] =
                diagnostics.covariance_fro_delta;
            result.metrics["gpu_gaussian_symbolic_blocks"] = 1.0;
            result.metrics["gpu_exact_blocks"] = 0.0;
        }
        result.params["gaussian_only_dense_state"] = "true";
        result.params["gaussian_backend"] = "cpu_gaussian_symplectic";
        result.note = "pure Gaussian vacuum-track run; CPU dense reference skipped";
        return result;
    }
    result.params["gaussian_only_dense_state"] = "false";
    if (!gaussian_only &&
        total_dimension(spec.num_qubits, spec.num_qumodes, spec.cutoff) > options.max_dense_dim) {
        result.status = "skipped";
        result.note = "state dimension exceeds --max-dense-dim";
        return result;
    }

    Vector cpu_state = make_probe_initial_state(spec);
    Vector gpu_initial = cpu_state;

    std::vector<double> l2_errors;
    std::vector<double> max_errors;
    std::vector<double> fidelity_devs;

    if (per_gate) {
        for (size_t i = 0; i < spec.gates.size(); ++i) {
            std::string reason;
            if (!apply_gate_cpu(&cpu_state, spec.num_qubits, spec.num_qumodes, spec.cutoff, spec.gates[i], &reason)) {
                result.status = "skipped";
                result.note = "unsupported CPU dense gate at index " + std::to_string(i) + ": " + reason;
                return result;
            }

            CircuitSpec prefix = spec;
            prefix.gates.assign(spec.gates.begin(), spec.gates.begin() + static_cast<std::ptrdiff_t>(i + 1));
            const GpuRunOutput gpu_output = run_gpu_circuit_output(prefix, &gpu_initial, options);
            const Metrics m = compute_metrics(cpu_state, gpu_output.state);
            l2_errors.push_back(m.l2_error);
            max_errors.push_back(m.max_error);
            fidelity_devs.push_back(m.fidelity_deviation);
            if (i + 1 == spec.gates.size()) {
                add_gpu_run_metrics(&result, "gpu_", gpu_output);
            }
        }
    } else {
        for (size_t i = 0; i < spec.gates.size(); ++i) {
            std::string reason;
            if (!apply_gate_cpu(&cpu_state, spec.num_qubits, spec.num_qumodes, spec.cutoff, spec.gates[i], &reason)) {
                result.status = "skipped";
                result.note = "unsupported CPU dense gate at index " + std::to_string(i) + ": " + reason;
                return result;
            }
        }
        const GpuRunOutput gpu_output = run_gpu_circuit_output(spec, &gpu_initial, options);
        const Metrics m = compute_metrics(cpu_state, gpu_output.state);
        l2_errors.push_back(m.l2_error);
        max_errors.push_back(m.max_error);
        fidelity_devs.push_back(m.fidelity_deviation);
        result.metrics["reference_norm"] = m.reference_norm;
        result.metrics["gpu_norm"] = m.implementation_norm;
        result.metrics["fidelity"] = m.fidelity;
        add_gpu_run_metrics(&result, "gpu_", gpu_output);
    }

    result.metrics["max_l2_error"] = l2_errors.empty() ? 0.0 : *std::max_element(l2_errors.begin(), l2_errors.end());
    result.metrics["max_abs_error"] = max_errors.empty() ? 0.0 : *std::max_element(max_errors.begin(), max_errors.end());
    result.metrics["max_fidelity_deviation"] =
        fidelity_devs.empty() ? 0.0 : *std::max_element(fidelity_devs.begin(), fidelity_devs.end());
    result.metrics["mean_l2_error"] =
        l2_errors.empty() ? 0.0 : std::accumulate(l2_errors.begin(), l2_errors.end(), 0.0) / l2_errors.size();
    return result;
}

Result run_identity_case(const CircuitSpec& spec, const Options& options) {
    Result result;
    result.name = spec.name + "_reverse_identity";
    result.category = "reverse_identity";
    result.params["workload"] = spec.workload;
    result.params["num_qubits"] = std::to_string(spec.num_qubits);
    result.params["num_qumodes"] = std::to_string(spec.num_qumodes);
    result.params["cutoff"] = std::to_string(spec.cutoff);
    result.params["forward_gates"] = std::to_string(spec.gates.size());
    result.params["initial_profile"] = spec.initial_profile;
    result.params["force_dense_fock"] = options.force_dense_fock ? "true" : "false";
    result.params["gaussian_symbolic_enabled"] = options.disable_symbolic ? "false" : "true";
    result.metrics["state_dim"] =
        static_cast<double>(total_dimension(spec.num_qubits, spec.num_qumodes, spec.cutoff));

    const bool gaussian_only =
        is_gaussian_vacuum_track_candidate(spec) &&
        !options.disable_symbolic &&
        total_dimension(spec.num_qubits, spec.num_qumodes, spec.cutoff) > options.max_dense_dim;
    if (!gaussian_only &&
        total_dimension(spec.num_qubits, spec.num_qumodes, spec.cutoff) > options.max_dense_dim) {
        result.status = "skipped";
        result.note = "state dimension exceeds --max-dense-dim";
        return result;
    }

    CircuitSpec id_spec = spec;
    std::string reason;
    if (!append_inverse_gates(spec.gates, &id_spec.gates, &reason)) {
        result.status = "skipped";
        result.note = "inverse circuit unsupported: " + reason;
        return result;
    }

    const Vector initial = gaussian_only ? Vector() : make_probe_initial_state(spec);
    const Vector* initial_ptr = gaussian_only ? nullptr : &initial;
    GpuRunOutput gpu_output;
    GaussianDiagnostics cpu_gaussian_diagnostics;
    if (gaussian_only) {
        cpu_gaussian_diagnostics = run_cpu_gaussian_vacuum_track(id_spec);
        gpu_output.gaussian = cpu_gaussian_diagnostics;
        gpu_output.stats.gaussian_symbolic_blocks = 1;
        gpu_output.stats.exact_blocks = 0;
    } else {
        gpu_output = run_gpu_circuit_output(id_spec, initial_ptr, options, true);
    }
    if (!gaussian_only) {
        const Metrics m = compute_metrics(initial, gpu_output.state);
        result.metrics["l2_error"] = m.l2_error;
        result.metrics["max_abs_error"] = m.max_error;
        result.metrics["relative_error"] = m.relative_error;
        result.metrics["fidelity"] = m.fidelity;
        result.metrics["fidelity_deviation"] = m.fidelity_deviation;
        result.metrics["initial_norm"] = m.reference_norm;
        result.metrics["gpu_norm"] = m.implementation_norm;
    } else if (gpu_output.gaussian.available) {
        result.metrics["fidelity"] = gpu_output.gaussian.vacuum_fidelity;
        result.metrics["fidelity_deviation"] = gpu_output.gaussian.vacuum_fidelity_deviation;
        result.metrics["l2_error"] = gpu_output.gaussian.displacement_l2;
        result.metrics["max_abs_error"] = gpu_output.gaussian.covariance_max_abs_delta;
        result.metrics["relative_error"] = gpu_output.gaussian.vacuum_fidelity_deviation;
        result.metrics["initial_norm"] = 1.0;
        result.metrics["gpu_norm"] = gpu_output.gaussian.weight_abs;
    }
    result.params["gaussian_only_dense_state"] = gaussian_only ? "true" : "false";
    if (gaussian_only) {
        result.params["gaussian_backend"] = "cpu_gaussian_symplectic";
    }
    add_gpu_run_metrics(&result, "reverse_", gpu_output);
    result.params["identity_gates"] = std::to_string(id_spec.gates.size());
    return result;
}

Result run_convergence_case(const CircuitSpec& spec, const Options& options) {
    Result result;
    result.name = spec.name + "_reverse_convergence";
    result.category = "reverse_convergence";
    result.params["workload"] = spec.workload;
    result.params["num_qubits"] = std::to_string(spec.num_qubits);
    result.params["num_qumodes"] = std::to_string(spec.num_qumodes);
    result.params["cutoff"] = std::to_string(spec.cutoff);
    result.params["forward_gates"] = std::to_string(spec.gates.size());
    result.params["initial_profile"] = spec.initial_profile;
    result.params["force_dense_fock"] = options.force_dense_fock ? "true" : "false";
    result.params["gaussian_symbolic_enabled"] = options.disable_symbolic ? "false" : "true";
    const size_t dim = total_dimension(spec.num_qubits, spec.num_qumodes, spec.cutoff);
    result.metrics["state_dim"] = static_cast<double>(dim);

    const bool gaussian_only =
        is_gaussian_vacuum_track_candidate(spec) &&
        !options.disable_symbolic &&
        dim > options.max_dense_dim;
    if (!gaussian_only && dim > options.max_dense_dim) {
        result.status = "skipped";
        result.note = "state dimension exceeds --max-dense-dim";
        return result;
    }

    CircuitSpec id_spec = spec;
    std::string reason;
    if (!append_inverse_gates(spec.gates, &id_spec.gates, &reason)) {
        result.status = "skipped";
        result.note = "inverse circuit unsupported: " + reason;
        return result;
    }

    const Vector initial = gaussian_only ? Vector() : make_probe_initial_state(spec);
    const Vector* initial_ptr = gaussian_only ? nullptr : &initial;
    GpuRunOutput forward_output;
    GpuRunOutput reverse_output;
    if (gaussian_only) {
        forward_output.gaussian = run_cpu_gaussian_vacuum_track(spec);
        forward_output.stats.gaussian_symbolic_blocks = 1;
        forward_output.stats.exact_blocks = 0;
        reverse_output.gaussian = run_cpu_gaussian_vacuum_track(id_spec);
        reverse_output.stats.gaussian_symbolic_blocks = 1;
        reverse_output.stats.exact_blocks = 0;
    } else {
        forward_output = run_gpu_circuit_output(spec, initial_ptr, options, true);
        reverse_output = run_gpu_circuit_output(id_spec, initial_ptr, options, true);
    }

    if (!gaussian_only) {
        const Metrics forward_delta = compute_metrics(initial, forward_output.state);
        const Metrics reverse_delta = compute_metrics(initial, reverse_output.state);
        const TailMetrics initial_tail = compute_tail_metrics(initial, spec.num_qubits, spec.num_qumodes, spec.cutoff);
        const TailMetrics forward_tail = compute_tail_metrics(forward_output.state, spec.num_qubits, spec.num_qumodes, spec.cutoff);
        const TailMetrics reverse_tail = compute_tail_metrics(reverse_output.state, spec.num_qubits, spec.num_qumodes, spec.cutoff);

        result.metrics["initial_norm"] = forward_delta.reference_norm;
        result.metrics["forward_norm"] = forward_delta.implementation_norm;
        result.metrics["forward_norm_loss"] = std::abs(1.0 - forward_delta.implementation_norm);
        result.metrics["forward_state_change_l2"] = forward_delta.l2_error;
        result.metrics["forward_state_change_fidelity"] = forward_delta.fidelity;
        result.metrics["forward_tail_fraction"] = forward_tail.tail_fraction;
        result.metrics["forward_boundary_fraction"] = forward_tail.boundary_fraction;
        result.metrics["forward_mean_total_photon_number"] = forward_tail.mean_total_photon_number;
        result.metrics["forward_tail_population"] = forward_tail.tail_population;
        result.metrics["forward_boundary_population"] = forward_tail.boundary_population;
        result.metrics["forward_norm_sq"] = forward_tail.norm_sq;

        result.metrics["reverse_fidelity"] = reverse_delta.fidelity;
        result.metrics["reverse_fidelity_deviation"] = reverse_delta.fidelity_deviation;
        result.metrics["reverse_l2_error"] = reverse_delta.l2_error;
        result.metrics["reverse_max_abs_error"] = reverse_delta.max_error;
        result.metrics["reverse_norm"] = reverse_delta.implementation_norm;
        result.metrics["reverse_norm_loss"] = std::abs(1.0 - reverse_delta.implementation_norm);
        result.metrics["reverse_tail_fraction"] = reverse_tail.tail_fraction;
        result.metrics["reverse_boundary_fraction"] = reverse_tail.boundary_fraction;
        result.metrics["reverse_mean_total_photon_number"] = reverse_tail.mean_total_photon_number;
        result.metrics["initial_tail_fraction"] = initial_tail.tail_fraction;
        result.metrics["initial_boundary_fraction"] = initial_tail.boundary_fraction;
    } else {
        result.metrics["initial_norm"] = 1.0;
        if (forward_output.gaussian.available) {
            result.metrics["forward_norm"] = forward_output.gaussian.weight_abs;
            result.metrics["forward_norm_loss"] = std::abs(1.0 - forward_output.gaussian.weight_abs);
            result.metrics["forward_state_change_l2"] = forward_output.gaussian.displacement_l2;
            result.metrics["forward_state_change_fidelity"] = forward_output.gaussian.vacuum_fidelity;
        }
        if (reverse_output.gaussian.available) {
            result.metrics["reverse_fidelity"] = reverse_output.gaussian.vacuum_fidelity;
            result.metrics["reverse_fidelity_deviation"] =
                reverse_output.gaussian.vacuum_fidelity_deviation;
            result.metrics["reverse_l2_error"] = reverse_output.gaussian.displacement_l2;
            result.metrics["reverse_max_abs_error"] =
                reverse_output.gaussian.covariance_max_abs_delta;
            result.metrics["reverse_norm"] = reverse_output.gaussian.weight_abs;
            result.metrics["reverse_norm_loss"] =
                std::abs(1.0 - reverse_output.gaussian.weight_abs);
        }
    }
    if (gaussian_only) {
        result.params["gaussian_backend"] = "cpu_gaussian_symplectic";
    }
    result.params["gaussian_only_dense_state"] = gaussian_only ? "true" : "false";
    add_gpu_run_metrics(&result, "forward_", forward_output);
    add_gpu_run_metrics(&result, "reverse_", reverse_output);
    result.params["identity_gates"] = std::to_string(id_spec.gates.size());
    return result;
}

void write_json(const fs::path& path,
                const Options& options,
                const DeviceMetadata& device,
                const std::vector<Result>& results) {
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output path: " + path.string());
    }

    out << "{\n";
    out << "  \"created_at\": \"" << now_utc_iso8601() << "\",\n";
    out << "  \"max_dense_dim\": " << options.max_dense_dim << ",\n";
    out << "  \"force_dense_fock\": " << (options.force_dense_fock ? "true" : "false") << ",\n";
    out << "  \"device\": {\n";
    out << "    \"available\": " << (device.available ? "true" : "false") << ",\n";
    out << "    \"index\": " << device.device_index << ",\n";
    out << "    \"name\": \"" << json_escape(device.name) << "\",\n";
    out << "    \"cc_major\": " << device.cc_major << ",\n";
    out << "    \"cc_minor\": " << device.cc_minor << ",\n";
    out << "    \"multiprocessor_count\": " << device.multiprocessor_count << ",\n";
    out << "    \"total_global_mem_bytes\": " << device.total_global_mem_bytes << "\n";
    out << "  },\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        out << "    {\n";
        out << "      \"name\": \"" << json_escape(r.name) << "\",\n";
        out << "      \"category\": \"" << json_escape(r.category) << "\",\n";
        out << "      \"status\": \"" << json_escape(r.status) << "\",\n";
        out << "      \"note\": \"" << json_escape(r.note) << "\",\n";
        out << "      \"params\": {";
        bool first = true;
        for (const auto& [key, value] : r.params) {
            out << (first ? "" : ", ") << "\"" << json_escape(key) << "\": \""
                << json_escape(value) << "\"";
            first = false;
        }
        out << "},\n";
        out << "      \"metrics\": {";
        first = true;
        for (const auto& [key, value] : r.metrics) {
            out << (first ? "" : ", ") << "\"" << json_escape(key) << "\": "
                << format_double(value);
            first = false;
        }
        out << "}\n";
        out << "    }" << (i + 1 == results.size() ? "" : ",") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void print_summary(const std::vector<Result>& results, const fs::path& output_path) {
    size_t ok = 0;
    size_t skipped = 0;
    size_t error = 0;
    std::map<std::string, double> max_by_category;
    for (const Result& r : results) {
        if (r.status == "ok") {
            ++ok;
            auto it = r.metrics.find("max_fidelity_deviation");
            if (it == r.metrics.end()) {
                it = r.metrics.find("fidelity_deviation");
            }
            if (it == r.metrics.end()) {
                it = r.metrics.find("reverse_fidelity_deviation");
            }
            if (it != r.metrics.end()) {
                max_by_category[r.category] = std::max(max_by_category[r.category], it->second);
            }
        } else if (r.status == "skipped") {
            ++skipped;
        } else {
            ++error;
        }
    }

    std::cout << "SC26 precision sweep complete\n";
    std::cout << "  output: " << output_path << "\n";
    std::cout << "  ok=" << ok << " skipped=" << skipped << " error=" << error << "\n";
    for (const auto& [category, value] : max_by_category) {
        std::cout << "  max fidelity deviation [" << category << "] = "
                  << std::setprecision(12) << value << "\n";
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--suite") {
            options.suite = need_value(arg);
        } else if (arg == "--name-filter") {
            options.name_filter = need_value(arg);
        } else if (arg == "--max-dense-dim") {
            options.max_dense_dim = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--max-states") {
            options.max_states = std::stoi(need_value(arg));
        } else if (arg == "--output") {
            options.output_path = need_value(arg);
        } else if (arg == "--enable-symbolic") {
            options.disable_symbolic = false;
        } else if (arg == "--force-dense-fock") {
            options.force_dense_fock = true;
        } else if (arg == "--help") {
            std::cout << "Usage: sc26_precision_sweep [--suite all|gates|sc26|identity|convergence|highcutoff] "
                         "[--name-filter SUBSTR] [--max-dense-dim N] [--max-states N] "
                         "[--output PATH] [--enable-symbolic] [--force-dense-fock]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const DeviceMetadata device = query_device();
        if (!device.available) {
            throw std::runtime_error("no CUDA device available");
        }

        std::vector<Result> results;
        auto append_result = [&](Result result) {
            std::cout << result.category << " " << result.name << " " << result.status;
            if (!result.note.empty()) {
                std::cout << " (" << result.note << ")";
            }
            std::cout << std::endl;
            results.push_back(std::move(result));
        };

        if (options.suite == "all" || options.suite == "gates") {
            for (const CircuitSpec& spec : make_gate_specs()) {
                if (!matches_filter(options.name_filter, spec.name)) {
                    continue;
                }
                try {
                    append_result(run_cpu_gpu_precision_case(spec, options, true));
                    append_result(run_identity_case(spec, options));
                } catch (const std::exception& e) {
                    Result r;
                    r.name = spec.name;
                    r.category = "gate_micro_precision";
                    r.status = "error";
                    r.note = e.what();
                    append_result(std::move(r));
                }
            }
        }

        if (options.suite == "all" || options.suite == "highcutoff") {
            for (const CircuitSpec& spec : make_high_cutoff_gate_specs()) {
                if (!matches_filter(options.name_filter, spec.name)) {
                    continue;
                }
                try {
                    append_result(run_cpu_gpu_precision_case(spec, options, true));
                    append_result(run_identity_case(spec, options));
                } catch (const std::exception& e) {
                    Result r;
                    r.name = spec.name;
                    r.category = "high_cutoff_gate_precision";
                    r.status = "error";
                    r.note = e.what();
                    append_result(std::move(r));
                }
            }
        }

        if (options.suite == "all" || options.suite == "sc26" || options.suite == "identity") {
            for (const CircuitSpec& spec : make_sc26_specs()) {
                if (!matches_filter(options.name_filter, spec.name)) {
                    continue;
                }
                try {
                    if (options.suite == "all" || options.suite == "sc26") {
                        append_result(run_cpu_gpu_precision_case(spec, options, false));
                    }
                    if (options.suite == "all" || options.suite == "identity") {
                        append_result(run_identity_case(spec, options));
                    }
                } catch (const std::exception& e) {
                    Result r;
                    r.name = spec.name;
                    r.category = "sc26_precision";
                    r.status = "error";
                    r.note = e.what();
                    append_result(std::move(r));
                }
            }
        }

        if (options.suite == "convergence") {
            for (const CircuitSpec& spec : make_convergence_specs()) {
                if (!matches_filter(options.name_filter, spec.name)) {
                    continue;
                }
                try {
                    append_result(run_convergence_case(spec, options));
                } catch (const std::exception& e) {
                    Result r;
                    r.name = spec.name;
                    r.category = "reverse_convergence";
                    r.status = "error";
                    r.note = e.what();
                    append_result(std::move(r));
                }
            }
        }

        write_json(options.output_path, options, device, results);
        print_summary(results, options.output_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sc26_precision_sweep failed: " << e.what() << std::endl;
        return 1;
    }
}
