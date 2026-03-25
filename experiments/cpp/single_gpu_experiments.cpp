#include "batch_scheduler.h"
#include "quantum_circuit.h"
#include "reference_gates.h"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
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
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::high_resolution_clock;
using Complex = std::complex<double>;

struct DeviceMetadata {
    bool available = false;
    int device_index = -1;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int multiprocessor_count = 0;
    size_t total_global_mem_bytes = 0;
    size_t shared_mem_per_block_bytes = 0;
};

struct ExperimentResult {
    std::string name;
    std::string category;
    std::string status = "ok";
    std::string note;
    std::map<std::string, std::string> params;
    std::map<std::string, double> metrics;
};

struct CircuitRunResult {
    bool ok = false;
    std::string error;
    QuantumCircuit::TimeStats time_stats{};
    QuantumCircuit::CircuitStats circuit_stats{};
    size_t memory_bytes = 0;
    Reference::Vector final_state;
};

struct OneQubitBranchStates {
    bool ok = false;
    std::string error;
    Reference::Vector low_state;
    Reference::Vector high_state;
};

struct BenchmarkSummary {
    bool ok = false;
    std::string error;
    double median_total_ms = 0.0;
    double median_transfer_ms = 0.0;
    double median_compute_ms = 0.0;
    double min_total_ms = 0.0;
    double max_total_ms = 0.0;
    double p25_total_ms = 0.0;
    double p75_total_ms = 0.0;
    double median_memory_bytes = 0.0;
    double median_active_states = 0.0;
    double median_hdd_nodes = 0.0;
    int warmup_runs = 0;
    int measured_runs = 0;
};

struct CliOptions {
    std::string suite = "all";
    std::string name_filter;
    int gaussian_symbolic_mode_limit = 4;
    bool use_interaction_picture = false;
    int max_states_override = 0; // 0 = use default per-circuit values
    fs::path output_path = fs::path("experiments/results/internal_single_gpu.json");
};

int g_gaussian_symbolic_mode_limit = 4;
bool g_use_interaction_picture = false;
int g_max_states_override = 0;
int g_scaling_warmup_runs_override = -1;
int g_scaling_measured_runs_override = -1;

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
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << c;
                break;
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

int parse_nonnegative_env_override(const char* name, int minimum_value = 0) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return -1;
    }

    const long parsed = std::strtol(raw, nullptr, 10);
    if (parsed < minimum_value) {
        throw std::invalid_argument(std::string(name) + " must be >= " + std::to_string(minimum_value));
    }
    if (parsed > std::numeric_limits<int>::max()) {
        throw std::overflow_error(std::string(name) + " exceeds int range");
    }
    return static_cast<int>(parsed);
}

DeviceMetadata query_device() {
    DeviceMetadata device;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return device;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return device;
    }

    device.available = true;
    device.device_index = 0;
    device.name = prop.name;
    device.cc_major = prop.major;
    device.cc_minor = prop.minor;
    device.multiprocessor_count = prop.multiProcessorCount;
    device.total_global_mem_bytes = prop.totalGlobalMem;
    device.shared_mem_per_block_bytes = prop.sharedMemPerBlock;
    return device;
}

Reference::Vector make_fock_state(int dim, int level) {
    Reference::Vector state(dim, Complex(0.0, 0.0));
    if (level >= 0 && level < dim) {
        state[level] = Complex(1.0, 0.0);
    }
    return state;
}

Reference::Vector make_vacuum_state(int dim) {
    return make_fock_state(dim, 0);
}

Reference::Vector make_coherent_state(int dim, double alpha) {
    Reference::Vector state(dim, Complex(0.0, 0.0));
    const double norm_factor = std::exp(-(alpha * alpha) / 2.0);
    for (int n = 0; n < dim; ++n) {
        const double coeff =
            norm_factor * std::pow(alpha, n) / std::sqrt(std::tgamma(static_cast<double>(n) + 1.0));
        state[n] = Complex(coeff, 0.0);
    }
    return state;
}

Reference::Vector make_two_mode_fock_state(int dim, int left_level, int right_level) {
    return Reference::tensor_product(make_fock_state(dim, left_level), make_fock_state(dim, right_level));
}

Reference::Vector make_two_mode_vacuum_state(int dim) {
    return make_two_mode_fock_state(dim, 0, 0);
}

std::vector<double> make_qaoa_angles(int p) {
    std::vector<double> params(2 * p, 0.0);
    for (int i = 0; i < 2 * p; ++i) {
        params[i] = 2.0 * M_PI * static_cast<double>(i + 1) / static_cast<double>(2 * p);
    }
    return params;
}

void add_cv_qaoa_circuit_gates(QuantumCircuit& circuit,
                               int num_qumodes,
                               const std::vector<double>& params,
                               double s,
                               double a,
                               int p) {
    for (int qm = 0; qm < num_qumodes; ++qm) {
        circuit.add_gate(Gates::Squeezing(qm, Complex(s, 0.0)));
    }

    for (int i = 0; i < p; ++i) {
        const double gamma = params[i];
        const double eta = params[p + i];

        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(Gates::Displacement(qm, Complex(a * gamma, 0.0)));
        }

        for (int qm = 0; qm < num_qumodes; ++qm) {
            circuit.add_gate(Gates::Squeezing(qm, Complex(eta, 0.0)));
        }
    }
}

void add_gkp_state_circuit_gates(QuantumCircuit& circuit,
                                 int rounds,
                                 double r,
                                 int qumode_idx) {
    const double alpha = std::sqrt(M_PI);
    circuit.add_gate(Gates::Squeezing(qumode_idx, Complex(r, 0.0)));

    for (int i = 1; i < rounds; ++i) {
        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::ConditionalDisplacement(0, qumode_idx, Complex(alpha / std::sqrt(2.0), 0.0)));
        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::PhaseGateS(0));
        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::ConditionalDisplacement(
            0, qumode_idx, Complex(0.0, M_PI / (8.0 * alpha * std::sqrt(2.0)))));
        circuit.add_gate(Gates::Hadamard(0));
        circuit.add_gate(Gates::PhaseGateS(0));
    }
}

void add_jch_simulation_circuit_gates(QuantumCircuit& circuit,
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
            circuit.add_gate(Gates::PhaseRotation(i, omega_r * tau));
        }

        for (int i = 0; i < nqubits; ++i) {
            circuit.add_gate(Gates::RotationZ(i, omega_q * tau / 2.0));
        }

        for (int i = 0; i < std::min(nsites, nqubits); ++i) {
            circuit.add_gate(Gates::JaynesCummings(i, i, g * tau));
        }

        for (int i = 0; i < nsites - 1; ++i) {
            circuit.add_gate(Gates::BeamSplitter(i, i + 1, j * tau));
        }
    }
}

void add_jch_photonic_chain_gates(QuantumCircuit& circuit,
                                  int nsites,
                                  double j,
                                  double omega_r,
                                  double tau,
                                  int timesteps) {
    for (int t = 0; t < timesteps; ++t) {
        for (int i = 0; i < nsites; ++i) {
            circuit.add_gate(Gates::PhaseRotation(i, omega_r * tau));
        }

        for (int i = 0; i < nsites - 1; ++i) {
            circuit.add_gate(Gates::BeamSplitter(i, i + 1, j * tau));
        }
    }
}

void add_cat_state_circuit_gates(QuantumCircuit& circuit, double alpha, int qumode_idx) {
    circuit.add_gate(Gates::Hadamard(0));
    circuit.add_gate(Gates::ConditionalDisplacement(
        0, qumode_idx, Complex(alpha / std::sqrt(2.0), 0.0)));
    circuit.add_gate(Gates::Hadamard(0));
    circuit.add_gate(Gates::PhaseGateS(0));
    circuit.add_gate(Gates::Hadamard(0));
    circuit.add_gate(Gates::ConditionalDisplacement(
        0, qumode_idx, Complex(0.0, M_PI / (8.0 * alpha * std::sqrt(2.0)))));
    circuit.add_gate(Gates::Hadamard(0));
    circuit.add_gate(Gates::PhaseGateS(0));
}

void add_basis_transformation_gates(QuantumCircuit& circuit, int num_qubits) {
    for (int i = 0; i < num_qubits; ++i) {
        circuit.add_gate(Gates::Hadamard(i));
        if (i % 3 == 0) {
            circuit.add_gate(Gates::PauliX(i));
            circuit.add_gate(Gates::PauliZ(i));
        } else if (i % 3 == 1) {
            circuit.add_gate(Gates::PauliZ(i));
        } else {
            circuit.add_gate(Gates::PauliX(i));
        }
    }
}

void add_basis_transformation_reverse_gates(QuantumCircuit& circuit, int num_qubits) {
    for (int i = num_qubits - 1; i >= 0; --i) {
        if (i % 3 == 0) {
            circuit.add_gate(Gates::PauliZ(i));
            circuit.add_gate(Gates::PauliX(i));
            circuit.add_gate(Gates::Hadamard(i));
        } else if (i % 3 == 1) {
            circuit.add_gate(Gates::PauliZ(i));
            circuit.add_gate(Gates::Hadamard(i));
        } else {
            circuit.add_gate(Gates::PauliX(i));
            circuit.add_gate(Gates::Hadamard(i));
        }
    }
}

void add_state_transfer_cvtodv_gates(QuantumCircuit& circuit,
                                     int num_qubits,
                                     int num_qumodes,
                                     double lambda,
                                     bool apply_basis) {
    for (int q = 0; q < num_qubits; ++q) {
        circuit.add_gate(Gates::RotationX(q, M_PI / 4.0));
    }

    for (int qm = 0; qm < num_qumodes; ++qm) {
        circuit.add_gate(Gates::Displacement(qm, Complex(lambda, 0.0)));
    }

    for (int q = 0; q < num_qubits; ++q) {
        circuit.add_gate(Gates::RotationZ(q, M_PI / 4.0));
    }

    for (int qm = 0; qm < num_qumodes; ++qm) {
        circuit.add_gate(Gates::Squeezing(qm, Complex(lambda, 0.0)));
    }

    if (apply_basis) {
        add_basis_transformation_gates(circuit, num_qubits);
    }
}

void add_state_transfer_dvtocv_gates(QuantumCircuit& circuit,
                                     int num_qubits,
                                     int num_qumodes,
                                     double lambda,
                                     bool apply_basis) {
    if (apply_basis) {
        add_basis_transformation_reverse_gates(circuit, num_qubits);
    }

    for (int q = 0; q < num_qubits; ++q) {
        circuit.add_gate(Gates::RotationZ(q, M_PI / 4.0));
    }

    for (int qm = 0; qm < num_qumodes; ++qm) {
        circuit.add_gate(Gates::Squeezing(qm, Complex(lambda, 0.0)));
    }

    for (int q = 0; q < num_qubits; ++q) {
        circuit.add_gate(Gates::RotationX(q, M_PI / 4.0));
    }

    for (int qm = 0; qm < num_qumodes; ++qm) {
        circuit.add_gate(Gates::Displacement(qm, Complex(lambda, 0.0)));
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
            params.push_back(M_PI / 5.0 + 0.07 * static_cast<double>(d + q));
            params.push_back(M_PI / 7.0 + 0.03 * static_cast<double>(d + q));
        }
    }
    return params;
}

void add_vqe_circuit_gates(QuantumCircuit& circuit,
                           int num_qubits,
                           int num_qumodes,
                           int depth,
                           const std::vector<double>& params) {
    size_t param_idx = 0;
    for (int d = 0; d < depth; ++d) {
        for (int qm = 0; qm < num_qumodes; ++qm) {
            const double beta_mag = params[param_idx++];
            const double beta_arg = params[param_idx++];
            circuit.add_gate(Gates::Displacement(
                qm, Complex(beta_mag * std::cos(beta_arg), beta_mag * std::sin(beta_arg))));
        }

        for (int q = 0; q < num_qubits; ++q) {
            const double theta = params[param_idx++];
            const double phi = params[param_idx++];
            circuit.add_gate(Gates::RotationX(q, theta));
            circuit.add_gate(Gates::RotationZ(q, phi));
        }

        for (int q = 0; q < num_qubits; ++q) {
            for (int qm = 0; qm < num_qumodes; ++qm) {
                circuit.add_gate(Gates::JaynesCummings(q, qm, M_PI / 4.0));
            }
        }
    }
}

std::vector<cuDoubleComplex> to_cuda_state(const Reference::Vector& state) {
    std::vector<cuDoubleComplex> cuda_state;
    cuda_state.reserve(state.size());
    for (const auto& value : state) {
        cuda_state.push_back(make_cuDoubleComplex(value.real(), value.imag()));
    }
    return cuda_state;
}

Reference::Vector from_cuda_state(const std::vector<cuDoubleComplex>& state) {
    Reference::Vector output(state.size(), Complex(0.0, 0.0));
    for (size_t i = 0; i < state.size(); ++i) {
        output[i] = Complex(cuCreal(state[i]), cuCimag(state[i]));
    }
    return output;
}

HDDNode* find_all_zero_qubit_terminal(HDDNode* node) {
    HDDNode* current = node;
    while (current && !current->is_terminal()) {
        current = current->low;
    }
    return current;
}

bool set_terminal_state(QuantumCircuit& circuit, const Reference::Vector& state) {
    HDDNode* terminal = find_all_zero_qubit_terminal(circuit.get_root_node());
    if (!terminal || terminal->tensor_id < 0) {
        return false;
    }
    circuit.get_state_pool().upload_state(terminal->tensor_id, to_cuda_state(state));
    return true;
}

Reference::Vector extract_terminal_state(QuantumCircuit& circuit) {
    HDDNode* terminal = find_all_zero_qubit_terminal(circuit.get_root_node());
    if (!terminal || terminal->tensor_id < 0) {
        return {};
    }
    std::vector<cuDoubleComplex> raw_state;
    circuit.get_state_pool().download_state(terminal->tensor_id, raw_state);
    return from_cuda_state(raw_state);
}

OneQubitBranchStates extract_one_qubit_branch_states(QuantumCircuit& circuit) {
    OneQubitBranchStates branches;
    HDDNode* root = circuit.get_root_node();
    if (!root || root->is_terminal()) {
        branches.error = "root is not a one-qubit branching node";
        return branches;
    }

    if (!root->low || !root->high || !root->low->is_terminal() || !root->high->is_terminal()) {
        branches.error = "root branches are not terminal states";
        return branches;
    }

    std::vector<cuDoubleComplex> raw_low;
    std::vector<cuDoubleComplex> raw_high;
    circuit.get_state_pool().download_state(root->low->tensor_id, raw_low);
    circuit.get_state_pool().download_state(root->high->tensor_id, raw_high);

    branches.ok = true;
    branches.low_state = from_cuda_state(raw_low);
    branches.high_state = from_cuda_state(raw_high);
    return branches;
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t low = static_cast<size_t>(std::floor(pos));
    const size_t high = static_cast<size_t>(std::ceil(pos));
    if (low == high) {
        return values[low];
    }
    const double frac = pos - static_cast<double>(low);
    return values[low] * (1.0 - frac) + values[high] * frac;
}

bool matches_name_filter(const std::string& name_filter, const std::string& case_name) {
    return name_filter.empty() || case_name.find(name_filter) != std::string::npos;
}

template <typename SetupFn>
CircuitRunResult run_circuit_once(int num_qubits,
                                  int num_qumodes,
                                  int cutoff,
                                  int max_states,
                                  const Reference::Vector* initial_state,
                                  SetupFn&& setup_fn) {
    CircuitRunResult result;
    try {
        QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, max_states);
        circuit.set_gaussian_symbolic_mode_limit(g_gaussian_symbolic_mode_limit);
        setup_fn(circuit);
        circuit.build();
        if (initial_state && !set_terminal_state(circuit, *initial_state)) {
            result.error = "failed to inject custom initial state into terminal root";
            return result;
        }
        if (g_use_interaction_picture) {
            circuit.execute_with_interaction_picture();
        } else {
            circuit.execute();
        }
        result.ok = true;
        result.time_stats = circuit.get_time_stats();
        result.circuit_stats = circuit.get_stats();
        result.memory_bytes = circuit.get_state_pool().get_memory_usage();
        result.final_state = extract_terminal_state(circuit);
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

template <typename SetupFn>
BenchmarkSummary benchmark_circuit_case(int num_qubits,
                                        int num_qumodes,
                                        int cutoff,
                                        int max_states,
                                        const Reference::Vector* initial_state,
                                        int warmup_runs,
                                        int measured_runs,
                                        SetupFn&& setup_fn) {
    BenchmarkSummary summary;
    summary.warmup_runs = warmup_runs;
    summary.measured_runs = measured_runs;

    for (int i = 0; i < warmup_runs; ++i) {
        CircuitRunResult warmup =
            run_circuit_once(num_qubits, num_qumodes, cutoff, max_states, initial_state, setup_fn);
        if (!warmup.ok) {
            summary.error = warmup.error;
            return summary;
        }
    }

    std::vector<double> total_times;
    std::vector<double> transfer_times;
    std::vector<double> compute_times;
    std::vector<double> memory_bytes;
    std::vector<double> active_states;
    std::vector<double> hdd_nodes;

    for (int i = 0; i < measured_runs; ++i) {
        CircuitRunResult run =
            run_circuit_once(num_qubits, num_qumodes, cutoff, max_states, initial_state, setup_fn);
        if (!run.ok) {
            summary.error = run.error;
            return summary;
        }
        total_times.push_back(run.time_stats.total_time);
        transfer_times.push_back(run.time_stats.transfer_time);
        compute_times.push_back(run.time_stats.computation_time);
        memory_bytes.push_back(static_cast<double>(run.memory_bytes));
        active_states.push_back(static_cast<double>(run.circuit_stats.active_states));
        hdd_nodes.push_back(static_cast<double>(run.circuit_stats.hdd_nodes));
    }

    summary.ok = true;
    summary.median_total_ms = percentile(total_times, 0.5);
    summary.median_transfer_ms = percentile(transfer_times, 0.5);
    summary.median_compute_ms = percentile(compute_times, 0.5);
    summary.min_total_ms = *std::min_element(total_times.begin(), total_times.end());
    summary.max_total_ms = *std::max_element(total_times.begin(), total_times.end());
    summary.p25_total_ms = percentile(total_times, 0.25);
    summary.p75_total_ms = percentile(total_times, 0.75);
    summary.median_memory_bytes = percentile(memory_bytes, 0.5);
    summary.median_active_states = percentile(active_states, 0.5);
    summary.median_hdd_nodes = percentile(hdd_nodes, 0.5);
    return summary;
}

std::vector<int> populate_state_pool(CVStatePool& pool, size_t num_states, const Reference::Vector& state) {
    std::vector<int> ids;
    ids.reserve(num_states);
    const std::vector<cuDoubleComplex> cuda_state = to_cuda_state(state);
    for (size_t i = 0; i < num_states; ++i) {
        const int state_id = pool.allocate_state();
        if (state_id < 0) {
            break;
        }
        pool.upload_state(state_id, cuda_state);
        ids.push_back(state_id);
    }
    return ids;
}

ExperimentResult make_unsupported_result(const std::string& name,
                                         const std::string& category,
                                         const std::string& note) {
    ExperimentResult result;
    result.name = name;
    result.category = category;
    result.status = "unsupported";
    result.note = note;
    return result;
}

ExperimentResult make_error_result(const std::string& name,
                                   const std::string& category,
                                   const std::string& error_message) {
    ExperimentResult result;
    result.name = name;
    result.category = category;
    result.status = "error";
    result.note = error_message;
    return result;
}

template <typename SetupFn, typename ReferenceFn>
ExperimentResult run_correctness_case(const std::string& name,
                                      int num_qumodes,
                                      int cutoff,
                                      int max_states,
                                      const Reference::Vector& initial_state,
                                      SetupFn&& setup_fn,
                                      ReferenceFn&& reference_fn,
                                      const std::string& note = std::string()) {
    ExperimentResult result;
    result.name = name;
    result.category = "correctness";
    result.note = note;
    result.params["cutoff"] = std::to_string(cutoff);
    result.params["num_qumodes"] = std::to_string(num_qumodes);
    result.params["max_states"] = std::to_string(max_states);

    CircuitRunResult run =
        run_circuit_once(1, num_qumodes, cutoff, max_states, &initial_state, std::forward<SetupFn>(setup_fn));
    if (!run.ok) {
        return make_error_result(name, "correctness", run.error);
    }
    if (run.final_state.empty()) {
        return make_error_result(name, "correctness", "final state extraction returned an empty vector");
    }

    const Reference::Vector reference_state = reference_fn(initial_state);
    if (reference_state.size() != run.final_state.size()) {
        return make_error_result(name, "correctness", "reference and implementation vector sizes differ");
    }

    const auto error_metrics = Reference::compute_error_metrics(reference_state, run.final_state);
    result.metrics["l2_error"] = error_metrics.l2_error;
    result.metrics["max_error"] = error_metrics.max_error;
    result.metrics["relative_error"] = error_metrics.relative_error;
    result.metrics["fidelity_deviation"] = error_metrics.fidelity_deviation;
    result.metrics["output_norm"] = Reference::vector_norm(run.final_state);
    result.metrics["reference_norm"] = Reference::vector_norm(reference_state);
    result.metrics["memory_bytes"] = static_cast<double>(run.memory_bytes);
    result.metrics["total_time_ms"] = run.time_stats.total_time;
    result.metrics["compute_time_ms"] = run.time_stats.computation_time;
    result.metrics["transfer_time_ms"] = run.time_stats.transfer_time;
    return result;
}

std::vector<ExperimentResult> run_correctness_suite() {
    const int cutoff = 16;
    const int max_states = 32;
    std::vector<ExperimentResult> results;
    results.push_back(run_correctness_case(
        "phase_rotation_fock1",
        1,
        cutoff,
        max_states,
        make_fock_state(cutoff, 1),
        [](QuantumCircuit& circuit) { circuit.add_gate(Gates::PhaseRotation(0, M_PI / 4.0)); },
        [](const Reference::Vector& input) {
            return Reference::DiagonalGates::apply_phase_rotation(input, M_PI / 4.0);
        }));

    results.push_back(run_correctness_case(
        "kerr_fock2",
        1,
        cutoff,
        max_states,
        make_fock_state(cutoff, 2),
        [](QuantumCircuit& circuit) { circuit.add_gate(Gates::KerrGate(0, 0.2)); },
        [](const Reference::Vector& input) {
            return Reference::DiagonalGates::apply_kerr_gate(input, 0.2);
        }));

    results.push_back(run_correctness_case(
        "creation_vacuum",
        1,
        cutoff,
        max_states,
        make_vacuum_state(cutoff),
        [](QuantumCircuit& circuit) { circuit.add_gate(Gates::CreationOperator(0)); },
        [](const Reference::Vector& input) {
            return Reference::LadderGates::apply_creation_operator(input);
        }));

    results.push_back(run_correctness_case(
        "annihilation_fock1",
        1,
        cutoff,
        max_states,
        make_fock_state(cutoff, 1),
        [](QuantumCircuit& circuit) { circuit.add_gate(Gates::AnnihilationOperator(0)); },
        [](const Reference::Vector& input) {
            return Reference::LadderGates::apply_annihilation_operator(input);
        }));

    results.push_back(run_correctness_case(
        "displacement_coherent",
        1,
        cutoff,
        max_states,
        make_coherent_state(cutoff, 0.2),
        [](QuantumCircuit& circuit) {
            circuit.add_gate(Gates::Displacement(0, Complex(0.15, 0.05)));
        },
        [](const Reference::Vector& input) {
            return Reference::SingleModeGates::apply_displacement_gate(input, Complex(0.15, 0.05));
        }));

    results.push_back(run_correctness_case(
        "squeezing_vacuum",
        1,
        cutoff,
        max_states,
        make_vacuum_state(cutoff),
        [](QuantumCircuit& circuit) {
            circuit.add_gate(Gates::Squeezing(0, Complex(0.1, 0.0)));
        },
        [](const Reference::Vector& input) {
            return Reference::SingleModeGates::apply_squeezing_gate(input, Complex(0.1, 0.0));
        }));

    results.push_back(run_correctness_case(
        "beamsplitter_fock10",
        2,
        8,
        max_states,
        make_two_mode_fock_state(8, 1, 0),
        [](QuantumCircuit& circuit) { circuit.add_gate(Gates::BeamSplitter(0, 1, M_PI / 6.0, 0.0)); },
        [](const Reference::Vector& input) {
            return Reference::TwoModeGates::apply_beam_splitter(input, M_PI / 6.0, 0.0, 4);
        }));

    {
        ExperimentResult result;
        result.name = "conditional_displacement_branching";
        result.category = "correctness";
        result.params["cutoff"] = std::to_string(cutoff);
        result.params["num_qumodes"] = "1";
        result.params["num_qubits"] = "1";

        try {
            const Complex alpha(0.1, 0.05);
            QuantumCircuit circuit(1, 1, cutoff, max_states);
            circuit.add_gate(Gates::Hadamard(0));
            circuit.add_gate(Gates::ConditionalDisplacement(0, 0, alpha));
            circuit.build();
            circuit.execute();

            const OneQubitBranchStates branches = extract_one_qubit_branch_states(circuit);
            if (!branches.ok) {
                results.push_back(make_error_result(result.name, result.category, branches.error));
            } else {
                Reference::Vector expected_low = make_vacuum_state(cutoff);
                Reference::Vector expected_high =
                    Reference::SingleModeGates::apply_displacement_gate(make_vacuum_state(cutoff), alpha);
                const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
                for (auto& amp : expected_low) {
                    amp *= inv_sqrt2;
                }
                for (auto& amp : expected_high) {
                    amp *= inv_sqrt2;
                }

                const Reference::ErrorMetrics low_metrics =
                    Reference::compute_error_metrics(expected_low, branches.low_state);
                const Reference::ErrorMetrics high_metrics =
                    Reference::compute_error_metrics(expected_high, branches.high_state);

                result.metrics["low_fidelity_deviation"] = low_metrics.fidelity_deviation;
                result.metrics["low_l2_error"] = low_metrics.l2_error;
                result.metrics["high_fidelity_deviation"] = high_metrics.fidelity_deviation;
                result.metrics["high_l2_error"] = high_metrics.l2_error;
                result.metrics["branch_probability_sum"] =
                    std::pow(Reference::vector_norm(branches.low_state), 2.0) +
                    std::pow(Reference::vector_norm(branches.high_state), 2.0);
                results.push_back(std::move(result));
            }
        } catch (const std::exception& e) {
            results.push_back(make_error_result(result.name, result.category, e.what()));
        }
    }

    {
        ExperimentResult result;
        result.name = "conditional_displacement_target_qumode1";
        result.category = "correctness";
        result.params["cutoff"] = std::to_string(cutoff);
        result.params["num_qumodes"] = "2";
        result.params["num_qubits"] = "1";
        result.params["target_qumode"] = "1";

        try {
            const Complex alpha(0.1, -0.04);
            QuantumCircuit circuit(1, 2, cutoff, max_states);
            circuit.add_gate(Gates::Hadamard(0));
            circuit.add_gate(Gates::ConditionalDisplacement(0, 1, alpha));
            circuit.build();
            circuit.execute();

            const OneQubitBranchStates branches = extract_one_qubit_branch_states(circuit);
            if (!branches.ok) {
                results.push_back(make_error_result(result.name, result.category, branches.error));
            } else {
                Reference::Vector expected_low = make_two_mode_vacuum_state(cutoff);
                Reference::Vector expected_high = Reference::tensor_product(
                    make_vacuum_state(cutoff),
                    Reference::SingleModeGates::apply_displacement_gate(
                        make_vacuum_state(cutoff), alpha));
                const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
                for (auto& amp : expected_low) {
                    amp *= inv_sqrt2;
                }
                for (auto& amp : expected_high) {
                    amp *= inv_sqrt2;
                }

                const Reference::ErrorMetrics low_metrics =
                    Reference::compute_error_metrics(expected_low, branches.low_state);
                const Reference::ErrorMetrics high_metrics =
                    Reference::compute_error_metrics(expected_high, branches.high_state);

                result.metrics["low_fidelity_deviation"] = low_metrics.fidelity_deviation;
                result.metrics["low_l2_error"] = low_metrics.l2_error;
                result.metrics["high_fidelity_deviation"] = high_metrics.fidelity_deviation;
                result.metrics["high_l2_error"] = high_metrics.l2_error;
                result.metrics["branch_probability_sum"] =
                    std::pow(Reference::vector_norm(branches.low_state), 2.0) +
                    std::pow(Reference::vector_norm(branches.high_state), 2.0);
                results.push_back(std::move(result));
            }
        } catch (const std::exception& e) {
            results.push_back(make_error_result(result.name, result.category, e.what()));
        }
    }

    {
        ExperimentResult result;
        result.name = "rabi_interaction_kernel";
        result.category = "correctness";
        result.params["cutoff"] = std::to_string(cutoff);
        result.params["num_qumodes"] = "1";
        result.params["num_qubits"] = "1";

        try {
            const double theta = 0.2;
            QuantumCircuit circuit(1, 1, cutoff, max_states);
            circuit.add_gate(Gates::RabiInteraction(0, 0, theta));
            circuit.build();
            circuit.execute();

            const OneQubitBranchStates branches = extract_one_qubit_branch_states(circuit);
            if (!branches.ok) {
                results.push_back(make_error_result(result.name, result.category, branches.error));
            } else {
                const Reference::Vector disp_minus =
                    Reference::SingleModeGates::apply_displacement_gate(
                        make_vacuum_state(cutoff), Complex(0.0, -theta));
                const Reference::Vector disp_plus =
                    Reference::SingleModeGates::apply_displacement_gate(
                        make_vacuum_state(cutoff), Complex(0.0, theta));

                Reference::Vector expected_low(cutoff, Complex(0.0, 0.0));
                Reference::Vector expected_high(cutoff, Complex(0.0, 0.0));
                for (int i = 0; i < cutoff; ++i) {
                    expected_low[i] = 0.5 * (disp_minus[i] + disp_plus[i]);
                    expected_high[i] = 0.5 * (disp_minus[i] - disp_plus[i]);
                }

                const Reference::ErrorMetrics low_metrics =
                    Reference::compute_error_metrics(expected_low, branches.low_state);
                const Reference::ErrorMetrics high_metrics =
                    Reference::compute_error_metrics(expected_high, branches.high_state);

                result.metrics["low_fidelity_deviation"] = low_metrics.fidelity_deviation;
                result.metrics["low_l2_error"] = low_metrics.l2_error;
                result.metrics["high_fidelity_deviation"] = high_metrics.fidelity_deviation;
                result.metrics["high_l2_error"] = high_metrics.l2_error;
                result.metrics["branch_probability_sum"] =
                    std::pow(Reference::vector_norm(branches.low_state), 2.0) +
                    std::pow(Reference::vector_norm(branches.high_state), 2.0);
                results.push_back(std::move(result));
            }
        } catch (const std::exception& e) {
            results.push_back(make_error_result(result.name, result.category, e.what()));
        }
    }

    {
        ExperimentResult result;
        result.name = "rabi_interaction_target_qumode1";
        result.category = "correctness";
        result.params["cutoff"] = std::to_string(cutoff);
        result.params["num_qumodes"] = "2";
        result.params["num_qubits"] = "1";
        result.params["target_qumode"] = "1";

        try {
            const double theta = 0.2;
            QuantumCircuit circuit(1, 2, cutoff, max_states);
            circuit.add_gate(Gates::RabiInteraction(0, 1, theta));
            circuit.build();
            circuit.execute();

            const OneQubitBranchStates branches = extract_one_qubit_branch_states(circuit);
            if (!branches.ok) {
                results.push_back(make_error_result(result.name, result.category, branches.error));
            } else {
                const Reference::Vector disp_minus =
                    Reference::SingleModeGates::apply_displacement_gate(
                        make_vacuum_state(cutoff), Complex(0.0, -theta));
                const Reference::Vector disp_plus =
                    Reference::SingleModeGates::apply_displacement_gate(
                        make_vacuum_state(cutoff), Complex(0.0, theta));

                Reference::Vector target_low(cutoff, Complex(0.0, 0.0));
                Reference::Vector target_high(cutoff, Complex(0.0, 0.0));
                for (int i = 0; i < cutoff; ++i) {
                    target_low[i] = 0.5 * (disp_minus[i] + disp_plus[i]);
                    target_high[i] = 0.5 * (disp_minus[i] - disp_plus[i]);
                }

                const Reference::Vector expected_low =
                    Reference::tensor_product(make_vacuum_state(cutoff), target_low);
                const Reference::Vector expected_high =
                    Reference::tensor_product(make_vacuum_state(cutoff), target_high);

                const Reference::ErrorMetrics low_metrics =
                    Reference::compute_error_metrics(expected_low, branches.low_state);
                const Reference::ErrorMetrics high_metrics =
                    Reference::compute_error_metrics(expected_high, branches.high_state);

                result.metrics["low_fidelity_deviation"] = low_metrics.fidelity_deviation;
                result.metrics["low_l2_error"] = low_metrics.l2_error;
                result.metrics["high_fidelity_deviation"] = high_metrics.fidelity_deviation;
                result.metrics["high_l2_error"] = high_metrics.l2_error;
                result.metrics["branch_probability_sum"] =
                    std::pow(Reference::vector_norm(branches.low_state), 2.0) +
                    std::pow(Reference::vector_norm(branches.high_state), 2.0);
                results.push_back(std::move(result));
            }
        } catch (const std::exception& e) {
            results.push_back(make_error_result(result.name, result.category, e.what()));
        }
    }

    {
        ExperimentResult result;
        result.name = "jaynes_cummings_kernel";
        result.category = "correctness";
        result.params["cutoff"] = std::to_string(cutoff);
        result.params["num_qumodes"] = "1";
        result.params["num_qubits"] = "1";

        try {
            const double theta = 0.3;
            QuantumCircuit circuit(1, 1, cutoff, max_states);
            circuit.add_gate(Gates::PauliX(0));
            circuit.add_gate(Gates::JaynesCummings(0, 0, theta));
            circuit.build();
            circuit.execute();

            const OneQubitBranchStates branches = extract_one_qubit_branch_states(circuit);
            if (!branches.ok) {
                results.push_back(make_error_result(result.name, result.category, branches.error));
            } else {
                Reference::Vector expected_low(cutoff, Complex(0.0, 0.0));
                Reference::Vector expected_high(cutoff, Complex(0.0, 0.0));
                if (cutoff > 1) {
                    expected_low[1] = Complex(0.0, -std::sin(theta));
                }
                expected_high[0] = Complex(std::cos(theta), 0.0);

                const Reference::ErrorMetrics low_metrics =
                    Reference::compute_error_metrics(expected_low, branches.low_state);
                const Reference::ErrorMetrics high_metrics =
                    Reference::compute_error_metrics(expected_high, branches.high_state);

                result.metrics["low_fidelity_deviation"] = low_metrics.fidelity_deviation;
                result.metrics["low_l2_error"] = low_metrics.l2_error;
                result.metrics["high_fidelity_deviation"] = high_metrics.fidelity_deviation;
                result.metrics["high_l2_error"] = high_metrics.l2_error;
                result.metrics["branch_probability_sum"] =
                    std::pow(Reference::vector_norm(branches.low_state), 2.0) +
                    std::pow(Reference::vector_norm(branches.high_state), 2.0);
                results.push_back(std::move(result));
            }
        } catch (const std::exception& e) {
            results.push_back(make_error_result(result.name, result.category, e.what()));
        }
    }

    {
        ExperimentResult result;
        result.name = "jaynes_cummings_target_qumode1";
        result.category = "correctness";
        result.params["cutoff"] = std::to_string(cutoff);
        result.params["num_qumodes"] = "2";
        result.params["num_qubits"] = "1";
        result.params["target_qumode"] = "1";

        try {
            const double theta = 0.3;
            QuantumCircuit circuit(1, 2, cutoff, max_states);
            circuit.add_gate(Gates::PauliX(0));
            circuit.add_gate(Gates::JaynesCummings(0, 1, theta));
            circuit.build();
            circuit.execute();

            const OneQubitBranchStates branches = extract_one_qubit_branch_states(circuit);
            if (!branches.ok) {
                results.push_back(make_error_result(result.name, result.category, branches.error));
            } else {
                Reference::Vector target_low(cutoff, Complex(0.0, 0.0));
                Reference::Vector target_high(cutoff, Complex(0.0, 0.0));
                if (cutoff > 1) {
                    target_low[1] = Complex(0.0, -std::sin(theta));
                }
                target_high[0] = Complex(std::cos(theta), 0.0);

                const Reference::Vector expected_low =
                    Reference::tensor_product(make_vacuum_state(cutoff), target_low);
                const Reference::Vector expected_high =
                    Reference::tensor_product(make_vacuum_state(cutoff), target_high);

                const Reference::ErrorMetrics low_metrics =
                    Reference::compute_error_metrics(expected_low, branches.low_state);
                const Reference::ErrorMetrics high_metrics =
                    Reference::compute_error_metrics(expected_high, branches.high_state);

                result.metrics["low_fidelity_deviation"] = low_metrics.fidelity_deviation;
                result.metrics["low_l2_error"] = low_metrics.l2_error;
                result.metrics["high_fidelity_deviation"] = high_metrics.fidelity_deviation;
                result.metrics["high_l2_error"] = high_metrics.l2_error;
                result.metrics["branch_probability_sum"] =
                    std::pow(Reference::vector_norm(branches.low_state), 2.0) +
                    std::pow(Reference::vector_norm(branches.high_state), 2.0);
                results.push_back(std::move(result));
            }
        } catch (const std::exception& e) {
            results.push_back(make_error_result(result.name, result.category, e.what()));
        }
    }

    {
        ExperimentResult result;
        result.name = "anti_jaynes_cummings_kernel";
        result.category = "correctness";
        result.params["cutoff"] = std::to_string(cutoff);
        result.params["num_qumodes"] = "1";
        result.params["num_qubits"] = "1";

        try {
            const double theta = 0.25;
            QuantumCircuit circuit(1, 1, cutoff, max_states);
            circuit.add_gate(Gates::AntiJaynesCummings(0, 0, theta));
            circuit.build();
            circuit.execute();

            const OneQubitBranchStates branches = extract_one_qubit_branch_states(circuit);
            if (!branches.ok) {
                results.push_back(make_error_result(result.name, result.category, branches.error));
            } else {
                Reference::Vector expected_low(cutoff, Complex(0.0, 0.0));
                Reference::Vector expected_high(cutoff, Complex(0.0, 0.0));
                expected_low[0] = Complex(std::cos(theta), 0.0);
                if (cutoff > 1) {
                    expected_high[1] = Complex(0.0, -std::sin(theta));
                }

                const Reference::ErrorMetrics low_metrics =
                    Reference::compute_error_metrics(expected_low, branches.low_state);
                const Reference::ErrorMetrics high_metrics =
                    Reference::compute_error_metrics(expected_high, branches.high_state);

                result.metrics["low_fidelity_deviation"] = low_metrics.fidelity_deviation;
                result.metrics["low_l2_error"] = low_metrics.l2_error;
                result.metrics["high_fidelity_deviation"] = high_metrics.fidelity_deviation;
                result.metrics["high_l2_error"] = high_metrics.l2_error;
                result.metrics["branch_probability_sum"] =
                    std::pow(Reference::vector_norm(branches.low_state), 2.0) +
                    std::pow(Reference::vector_norm(branches.high_state), 2.0);
                results.push_back(std::move(result));
            }
        } catch (const std::exception& e) {
            results.push_back(make_error_result(result.name, result.category, e.what()));
        }
    }

    {
        ExperimentResult result;
        result.name = "anti_jaynes_cummings_target_qumode1";
        result.category = "correctness";
        result.params["cutoff"] = std::to_string(cutoff);
        result.params["num_qumodes"] = "2";
        result.params["num_qubits"] = "1";
        result.params["target_qumode"] = "1";

        try {
            const double theta = 0.25;
            QuantumCircuit circuit(1, 2, cutoff, max_states);
            circuit.add_gate(Gates::AntiJaynesCummings(0, 1, theta));
            circuit.build();
            circuit.execute();

            const OneQubitBranchStates branches = extract_one_qubit_branch_states(circuit);
            if (!branches.ok) {
                results.push_back(make_error_result(result.name, result.category, branches.error));
            } else {
                Reference::Vector target_low(cutoff, Complex(0.0, 0.0));
                Reference::Vector target_high(cutoff, Complex(0.0, 0.0));
                target_low[0] = Complex(std::cos(theta), 0.0);
                if (cutoff > 1) {
                    target_high[1] = Complex(0.0, -std::sin(theta));
                }

                const Reference::Vector expected_low =
                    Reference::tensor_product(make_vacuum_state(cutoff), target_low);
                const Reference::Vector expected_high =
                    Reference::tensor_product(make_vacuum_state(cutoff), target_high);

                const Reference::ErrorMetrics low_metrics =
                    Reference::compute_error_metrics(expected_low, branches.low_state);
                const Reference::ErrorMetrics high_metrics =
                    Reference::compute_error_metrics(expected_high, branches.high_state);

                result.metrics["low_fidelity_deviation"] = low_metrics.fidelity_deviation;
                result.metrics["low_l2_error"] = low_metrics.l2_error;
                result.metrics["high_fidelity_deviation"] = high_metrics.fidelity_deviation;
                result.metrics["high_l2_error"] = high_metrics.l2_error;
                result.metrics["branch_probability_sum"] =
                    std::pow(Reference::vector_norm(branches.low_state), 2.0) +
                    std::pow(Reference::vector_norm(branches.high_state), 2.0);
                results.push_back(std::move(result));
            }
        } catch (const std::exception& e) {
            results.push_back(make_error_result(result.name, result.category, e.what()));
        }
    }

    return results;
}

template <typename SetupFn>
ExperimentResult run_microbenchmark_case(const std::string& name,
                                         int num_qumodes,
                                         int cutoff,
                                         int max_states,
                                         int warmup_runs,
                                         int measured_runs,
                                         const Reference::Vector* initial_state,
                                         SetupFn&& setup_fn) {
    ExperimentResult result;
    result.name = name;
    result.category = "microbench";
    result.params["cutoff"] = std::to_string(cutoff);
    result.params["num_qumodes"] = std::to_string(num_qumodes);
    result.params["warmup_runs"] = std::to_string(warmup_runs);
    result.params["measured_runs"] = std::to_string(measured_runs);

    const BenchmarkSummary summary = benchmark_circuit_case(
        1,
        num_qumodes,
        cutoff,
        max_states,
        initial_state,
        warmup_runs,
        measured_runs,
        std::forward<SetupFn>(setup_fn));

    if (!summary.ok) {
        return make_error_result(name, "microbench", summary.error);
    }

    result.metrics["median_total_ms"] = summary.median_total_ms;
    result.metrics["median_transfer_ms"] = summary.median_transfer_ms;
    result.metrics["median_compute_ms"] = summary.median_compute_ms;
    result.metrics["min_total_ms"] = summary.min_total_ms;
    result.metrics["max_total_ms"] = summary.max_total_ms;
    result.metrics["p25_total_ms"] = summary.p25_total_ms;
    result.metrics["p75_total_ms"] = summary.p75_total_ms;
    result.metrics["median_memory_bytes"] = summary.median_memory_bytes;
    result.metrics["median_active_states"] = summary.median_active_states;
    result.metrics["median_hdd_nodes"] = summary.median_hdd_nodes;
    result.metrics["throughput_ops_per_sec"] =
        summary.median_total_ms > 0.0 ? 1000.0 / summary.median_total_ms : 0.0;
    return result;
}

std::vector<ExperimentResult> run_microbenchmark_suite() {
    const int warmups = 3;
    const int measured = 20;
    const int max_states = 64;
    std::vector<ExperimentResult> results;

    for (int cutoff : {16, 32, 64}) {
        const Reference::Vector vacuum = make_vacuum_state(cutoff);
        const Reference::Vector fock = make_fock_state(cutoff, std::min(1, cutoff - 1));
        const Reference::Vector coh = make_coherent_state(cutoff, 0.25);

        results.push_back(run_microbenchmark_case(
            "phase_rotation_cutoff_" + std::to_string(cutoff),
            1,
            cutoff,
            max_states,
            warmups,
            measured,
            &vacuum,
            [](QuantumCircuit& circuit) { circuit.add_gate(Gates::PhaseRotation(0, 0.3)); }));

        results.push_back(run_microbenchmark_case(
            "creation_cutoff_" + std::to_string(cutoff),
            1,
            cutoff,
            max_states,
            warmups,
            measured,
            &fock,
            [](QuantumCircuit& circuit) { circuit.add_gate(Gates::CreationOperator(0)); }));

        results.push_back(run_microbenchmark_case(
            "displacement_cutoff_" + std::to_string(cutoff),
            1,
            cutoff,
            max_states,
            warmups,
            measured,
            &coh,
            [](QuantumCircuit& circuit) {
                circuit.add_gate(Gates::Displacement(0, Complex(0.15, 0.05)));
            }));
    }

    for (int cutoff : {16, 32, 64}) {
        const Reference::Vector input = make_two_mode_fock_state(cutoff, 1, 0);
        results.push_back(run_microbenchmark_case(
            "beamsplitter_cutoff_" + std::to_string(cutoff),
            2,
            cutoff,
            max_states,
            warmups,
            measured,
            &input,
            [](QuantumCircuit& circuit) { circuit.add_gate(Gates::BeamSplitter(0, 1, 0.4, 0.0)); }));
    }
    return results;
}

ExperimentResult run_batching_impact_case(size_t batch_size) {
    ExperimentResult result;
    result.name = "batching_phase_rotation_batch_" + std::to_string(batch_size);
    result.category = "runtime_ablation";
    result.params["batch_size"] = std::to_string(batch_size);
    result.params["gate"] = "phase_rotation";

    try {
        const int cutoff = 32;
        CVStatePool pool(cutoff, static_cast<int>(batch_size + 4), 1);
        const std::vector<int> state_ids =
            populate_state_pool(pool, batch_size, make_vacuum_state(cutoff));
        if (state_ids.size() != batch_size) {
            return make_error_result(result.name, result.category, "failed to populate enough active states");
        }

        BatchScheduler scheduler(&pool, 64);
        const int submitted_tasks = 32;
        const auto start = Clock::now();
        for (int i = 0; i < submitted_tasks; ++i) {
            scheduler.add_task(BatchTask(GateType::PHASE_ROTATION, state_ids, {Complex(0.2, 0.0)}));
        }
        scheduler.execute_pending_tasks();
        cudaDeviceSynchronize();
        const auto end = Clock::now();

        const auto stats = scheduler.get_stats();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double task_time_s = stats.total_time > 0.0 ? stats.total_time : wall_ms / 1000.0;

        result.metrics["submitted_tasks"] = static_cast<double>(submitted_tasks);
        result.metrics["processed_tasks"] = static_cast<double>(stats.total_tasks);
        result.metrics["wall_time_ms"] = wall_ms;
        result.metrics["scheduler_time_s"] = stats.total_time;
        result.metrics["avg_batch_size"] = stats.avg_batch_size;
        result.metrics["throughput_tasks_per_sec"] = stats.throughput;
        result.metrics["effective_state_updates_per_sec"] =
            task_time_s > 0.0 ? (static_cast<double>(batch_size) * static_cast<double>(stats.total_tasks)) / task_time_s
                              : 0.0;
        result.metrics["state_pool_memory_bytes"] = static_cast<double>(pool.get_memory_usage());
        result.metrics["active_states"] = static_cast<double>(pool.active_count);
        return result;
    } catch (const std::exception& e) {
        return make_error_result(result.name, result.category, e.what());
    }
}

ExperimentResult run_fusion_ablation_case(bool fusion_enabled) {
    ExperimentResult result;
    result.name = std::string("fusion_tiny_displacements_") + (fusion_enabled ? "on" : "off");
    result.category = "runtime_ablation";
    result.params["fusion_enabled"] = fusion_enabled ? "true" : "false";
    result.params["submitted_gates"] = "128";

    try {
        const int cutoff = 32;
        QuantumCircuit circuit(1, 1, cutoff, 64);
        circuit.build();

        // Add extra independent states so the scheduler sees a non-trivial batch.
        CVStatePool& pool = circuit.get_state_pool();
        populate_state_pool(pool, 15, make_vacuum_state(cutoff));

        RuntimeScheduler scheduler(&circuit, 64);
        scheduler.enable_fusion(fusion_enabled);
        scheduler.enable_auto_flush(false);

        const int submitted_gates = 128;
        const Complex tiny_alpha(1e-7, 0.0);
        const auto start = Clock::now();
        for (int i = 0; i < submitted_gates; ++i) {
            scheduler.schedule_gate(Gates::Displacement(0, tiny_alpha));
        }
        scheduler.execute_all();
        cudaDeviceSynchronize();
        const auto end = Clock::now();

        const auto stats = scheduler.get_stats();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double task_time_s =
            stats.batch_stats.total_time > 0.0 ? stats.batch_stats.total_time : wall_ms / 1000.0;

        result.metrics["submitted_gates"] = static_cast<double>(submitted_gates);
        result.metrics["executed_tasks"] = static_cast<double>(stats.batch_stats.total_tasks);
        result.metrics["wall_time_ms"] = wall_ms;
        result.metrics["scheduler_time_s"] = stats.batch_stats.total_time;
        result.metrics["avg_batch_size"] = stats.batch_stats.avg_batch_size;
        result.metrics["throughput_tasks_per_sec"] = stats.batch_stats.throughput;
        result.metrics["effective_state_updates_per_sec"] =
            task_time_s > 0.0 ? (static_cast<double>(pool.active_count) * static_cast<double>(stats.batch_stats.total_tasks)) / task_time_s
                              : 0.0;
        result.metrics["active_states"] = static_cast<double>(pool.active_count);
        return result;
    } catch (const std::exception& e) {
        return make_error_result(result.name, result.category, e.what());
    }
}

std::vector<ExperimentResult> run_runtime_ablation_suite() {
    std::vector<ExperimentResult> results;
    for (size_t batch_size : {1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL}) {
        results.push_back(run_batching_impact_case(batch_size));
    }
    results.push_back(run_fusion_ablation_case(false));
    results.push_back(run_fusion_ablation_case(true));
    results.push_back(make_unsupported_result(
        "fidelity_gc_ablation",
        "runtime_ablation",
        "GarbageCollector and MemoryManager are not wired into QuantumCircuit execution yet, so fidelity-aware GC cannot be benchmarked fairly"));
    return results;
}

template <typename SetupFn>
ExperimentResult run_scaling_case(const std::string& name,
                                  const std::string& workload,
                                  int num_qubits,
                                  int num_qumodes,
                                  int cutoff,
                                  int depth,
                                  int max_states,
                                  const Reference::Vector* initial_state,
                                  SetupFn&& setup_fn) {
    ExperimentResult result;
    result.name = name;
    result.category = "scaling";
    result.params["workload"] = workload;
    result.params["cutoff"] = std::to_string(cutoff);
    result.params["depth"] = std::to_string(depth);
    result.params["num_qubits"] = std::to_string(num_qubits);
    result.params["num_qumodes"] = std::to_string(num_qumodes);

    const int warmup_runs = g_scaling_warmup_runs_override >= 0 ? g_scaling_warmup_runs_override : 2;
    const int measured_runs = g_scaling_measured_runs_override >= 0 ? g_scaling_measured_runs_override : 10;
    result.params["warmup_runs"] = std::to_string(warmup_runs);
    result.params["measured_runs"] = std::to_string(measured_runs);

    const BenchmarkSummary summary = benchmark_circuit_case(
        num_qubits,
        num_qumodes,
        cutoff,
        max_states,
        initial_state,
        warmup_runs,
        measured_runs,
        std::forward<SetupFn>(setup_fn));

    if (!summary.ok) {
        return make_error_result(name, "scaling", summary.error);
    }

    result.metrics["median_total_ms"] = summary.median_total_ms;
    result.metrics["median_transfer_ms"] = summary.median_transfer_ms;
    result.metrics["median_compute_ms"] = summary.median_compute_ms;
    result.metrics["throughput_ops_per_sec"] =
        summary.median_total_ms > 0.0 ? 1000.0 / summary.median_total_ms : 0.0;
    result.metrics["median_memory_bytes"] = summary.median_memory_bytes;
    result.metrics["median_active_states"] = summary.median_active_states;
    result.metrics["median_hdd_nodes"] = summary.median_hdd_nodes;
    return result;
}

ExperimentResult run_hdd_vs_full_tensor_case(int num_qubits, int num_qumodes, int cutoff, int max_states) {
    ExperimentResult result;
    result.name = "hdd_vs_full_tensor_qubits_" + std::to_string(num_qubits);
    result.category = "scaling";
    result.params["workload"] = "hdd_vs_full_tensor";
    result.params["num_qubits"] = std::to_string(num_qubits);
    result.params["num_qumodes"] = std::to_string(num_qumodes);
    result.params["cutoff"] = std::to_string(cutoff);

    const Reference::Vector vacuum_input = make_vacuum_state(cutoff);
    const CircuitRunResult run = run_circuit_once(
        num_qubits,
        num_qumodes,
        cutoff,
        max_states,
        &vacuum_input,
        [](QuantumCircuit&) {});

    if (!run.ok) {
        return make_error_result(result.name, result.category, run.error);
    }

    const double cv_state_dim = std::pow(static_cast<double>(cutoff), num_qumodes);
    const double complex_bytes = static_cast<double>(sizeof(cuDoubleComplex));
    const double naive_full_tensor_bytes =
        std::pow(2.0, num_qubits) * cv_state_dim * complex_bytes;
    const double active_tensor_bytes =
        static_cast<double>(run.circuit_stats.active_states) * cv_state_dim * complex_bytes;
    const double hdd_node_bytes_estimate =
        static_cast<double>(run.circuit_stats.hdd_nodes) *
        static_cast<double>(sizeof(HDDNode));

    result.metrics["state_pool_reserved_bytes"] = static_cast<double>(run.memory_bytes);
    result.metrics["active_tensor_bytes_estimate"] = active_tensor_bytes;
    result.metrics["hdd_node_bytes_estimate"] = hdd_node_bytes_estimate;
    result.metrics["naive_full_tensor_bytes"] = naive_full_tensor_bytes;
    result.metrics["active_states"] = static_cast<double>(run.circuit_stats.active_states);
    result.metrics["hdd_nodes"] = static_cast<double>(run.circuit_stats.hdd_nodes);
    result.metrics["state_pool_vs_naive_ratio"] =
        run.memory_bytes > 0 ? naive_full_tensor_bytes / static_cast<double>(run.memory_bytes) : 0.0;
    result.metrics["active_tensor_vs_naive_ratio"] =
        active_tensor_bytes > 0.0 ? naive_full_tensor_bytes / active_tensor_bytes : 0.0;
    return result;
}

template <typename FactoryFn>
void append_filtered_scaling_case(std::vector<ExperimentResult>& results,
                                  const std::string& name_filter,
                                  const std::string& case_name,
                                  FactoryFn&& factory) {
    if (matches_name_filter(name_filter, case_name)) {
        results.push_back(factory());
    }
}


void add_qft_circuit_gates(QuantumCircuit& circuit, int num_qubits, int n, int a, int append) {
    int total = n + a + append;
    // 1. Initial Hadamards
    for (int i = 0; i < a; ++i) {
        circuit.add_gate(Gates::Hadamard(i));
    }
    
    // 2. DV-to-CV transfer proxy
    for (int q = 0; q < total; ++q) {
        circuit.add_gate(Gates::RotationX(q, M_PI / 4.0));
    }
    circuit.add_gate(Gates::Displacement(0, Complex(0.29, 0.0)));
    
    // 3. CV space QFT
    circuit.add_gate(Gates::PhaseRotation(0, M_PI / 2.0));
    
    // 4. CV-to-DV transfer proxy
    for (int q = 0; q < total; ++q) {
        circuit.add_gate(Gates::RotationZ(q, M_PI / 4.0));
    }
    circuit.add_gate(Gates::Squeezing(0, Complex(0.29, 0.0)));
    
    // 5. Measurement Hadamards
    for (int i = 0; i < n; ++i) {
        circuit.add_gate(Gates::Hadamard(a + i));
    }
}

void add_shors_circuit_gates(QuantumCircuit& circuit, int num_qubits, int num_qumodes, int N, int a) {
    // 1. GKP preparation proxies
    for (int qm = 0; qm < std::min(num_qumodes, 2); ++qm) {
        circuit.add_gate(Gates::Squeezing(qm, Complex(0.222, 0.0)));
        for (int i = 0; i < 2; ++i) { // Reduced rounds for benchmark
            circuit.add_gate(Gates::Hadamard(0));
            circuit.add_gate(Gates::ConditionalDisplacement(0, qm, Complex(0.5, 0.0)));
        }
    }
    
    // 2. Modular exponentiation proxies
    for (int i = 0; i < 2; ++i) {
        circuit.add_gate(Gates::JaynesCummings(0, 0, M_PI / 4.0));
        circuit.add_gate(Gates::JaynesCummings(0, 1, M_PI / 4.0));
    }
}
std::vector<ExperimentResult> run_scaling_suite(const std::string& name_filter) {
    std::vector<ExperimentResult> results;
    const int cat_max_states = g_max_states_override > 0 ? g_max_states_override : 64;
    const int qaoa_max_states = g_max_states_override > 0 ? g_max_states_override : 128;
    const int gkp_max_states = g_max_states_override > 0 ? g_max_states_override : 128;
    const int jch_max_states = g_max_states_override > 0 ? g_max_states_override : 64;
    const int gkp_rounds = 9;

    for (int cutoff : {16, 32, 64}) {
        const std::string cat_name =
            "cat_state_alpha_1.0_cutoff_" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, cat_name, [cutoff, cat_name, cat_max_states]() {
            const Reference::Vector input = make_vacuum_state(cutoff);
            ExperimentResult result = run_scaling_case(
                cat_name,
                "cat_state_circuit",
                1,
                1,
                cutoff,
                8,
                cat_max_states,
                &input,
                [](QuantumCircuit& circuit) { add_cat_state_circuit_gates(circuit, 1.0, 0); });
            result.params["alpha"] = "1.0";
            result.params["source_circuit"] = "circuit/src/cat_state_circuit.cpp";
            return result;
        });
    }

    for (const auto& spec : std::vector<std::tuple<int, int, int>>{
             {1, 2, 16}, {1, 2, 32}, {1, 2, 64}, {1, 2, 128}, {2, 4, 16}, {2, 4, 32}}) {
        const int num_modes = std::get<0>(spec);
        const int layers = std::get<1>(spec);
        const int cutoff = std::get<2>(spec);
        const std::string qaoa_name = "cv_qaoa_modes_" + std::to_string(num_modes) +
                                      "_layers_" + std::to_string(layers) +
                                      "_cutoff_" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, qaoa_name, [num_modes, layers, cutoff, qaoa_name, qaoa_max_states]() {
            const Reference::Vector input =
                num_modes == 1 ? make_vacuum_state(cutoff) : make_two_mode_vacuum_state(cutoff);
            const std::vector<double> qaoa_params = make_qaoa_angles(layers);
            ExperimentResult result = run_scaling_case(
                qaoa_name,
                "cv_qaoa_circuit",
                1,
                num_modes,
                cutoff,
                layers,
                qaoa_max_states,
                &input,
                [qaoa_params, num_modes, layers](QuantumCircuit& circuit) {
                    add_cv_qaoa_circuit_gates(circuit, num_modes, qaoa_params, 0.5, 1.0, layers);
                });
            result.params["dummy_qubits"] = "1";
            result.params["layers"] = std::to_string(layers);
            result.params["source_circuit"] = "circuit/src/qaoa_circuit.cpp";
            return result;
        });
    }

    for (const auto& spec : std::vector<std::tuple<int, int, int>>{
             {2, 5, 16}, {2, 5, 32}, {2, 5, 64}, {3, 8, 16}, {3, 8, 32}}) {
        const int num_modes = std::get<0>(spec);
        const int timesteps = std::get<1>(spec);
        const int cutoff = std::get<2>(spec);
        const std::string photonic_name = "jch_photonic_chain_modes_" + std::to_string(num_modes) +
                                          "_timesteps_" + std::to_string(timesteps) +
                                          "_cutoff_" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, photonic_name, [num_modes, timesteps, cutoff, photonic_name, qaoa_max_states]() {
            const Reference::Vector input =
                num_modes == 2 ? make_two_mode_vacuum_state(cutoff)
                               : Reference::tensor_product(make_two_mode_vacuum_state(cutoff), make_vacuum_state(cutoff));
            ExperimentResult result = run_scaling_case(
                photonic_name,
                "jch_photonic_chain",
                1,
                num_modes,
                cutoff,
                timesteps,
                qaoa_max_states,
                &input,
                [num_modes, timesteps](QuantumCircuit& circuit) {
                    add_jch_photonic_chain_gates(circuit, num_modes, 1.0, 1.0, 0.1, timesteps);
                });
            result.params["dummy_qubits"] = "1";
            result.params["timesteps"] = std::to_string(timesteps);
            result.params["source_circuit"] =
                "circuit/src/jch_simulation_circuit.cpp (bosonic terms only)";
            return result;
        });
    }

    for (int cutoff : {16, 32, 64}) {
        const std::string gkp_name =
            "gkp_rounds_9_cutoff_" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, gkp_name, [cutoff, gkp_name, gkp_rounds, gkp_max_states]() {
            const Reference::Vector input = make_vacuum_state(cutoff);
            ExperimentResult result = run_scaling_case(
                gkp_name,
                "gkp_state_circuit",
                1,
                1,
                cutoff,
                gkp_rounds,
                gkp_max_states,
                &input,
                [](QuantumCircuit& circuit) { add_gkp_state_circuit_gates(circuit, 9, 0.222, 0); });
            result.params["rounds"] = "9";
            result.params["source_circuit"] = "circuit/src/gkp_state_circuit.cpp";
            return result;
        });
    }

    for (int cutoff : {16, 32}) {
        const std::string cvtodv_name =
            "state_transfer_cvtodv_qubits_2_cutoff_" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, cvtodv_name, [cutoff, cvtodv_name, qaoa_max_states]() {
            const Reference::Vector input = make_two_mode_vacuum_state(cutoff);
            ExperimentResult result = run_scaling_case(
                cvtodv_name,
                "state_transfer_cvtodv",
                2,
                2,
                cutoff,
                8,
                qaoa_max_states,
                &input,
                [](QuantumCircuit& circuit) {
                    add_state_transfer_cvtodv_gates(circuit, 2, 2, 0.29, true);
                });
            result.params["source_circuit"] = "circuit/src/state_transfer_circuit.cpp";
            return result;
        });

        const std::string dvtocv_name =
            "state_transfer_dvtocv_qubits_2_cutoff_" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, dvtocv_name, [cutoff, dvtocv_name, qaoa_max_states]() {
            const Reference::Vector input = make_two_mode_vacuum_state(cutoff);
            ExperimentResult result = run_scaling_case(
                dvtocv_name,
                "state_transfer_dvtocv",
                2,
                2,
                cutoff,
                8,
                qaoa_max_states,
                &input,
                [](QuantumCircuit& circuit) {
                    add_state_transfer_dvtocv_gates(circuit, 2, 2, 0.29, true);
                });
            result.params["source_circuit"] = "circuit/src/state_transfer_circuit.cpp";
            return result;
        });
    }

    for (int cutoff : {8, 16, 32}) {
        const std::string jch_name =
            "jch_full_qubits_2_sites_2_timesteps_5_cutoff_" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, jch_name, [cutoff, jch_name, jch_max_states]() {
            const Reference::Vector input = make_two_mode_vacuum_state(cutoff);
            ExperimentResult result = run_scaling_case(
                jch_name,
                "jch_simulation_circuit",
                2,
                2,
                cutoff,
                5,
                jch_max_states,
                &input,
                [](QuantumCircuit& circuit) {
                    add_jch_simulation_circuit_gates(circuit, 2, 2, 1.0, 1.0, 1.0, 0.5, 0.1, 5);
                });
            result.params["source_circuit"] = "circuit/src/jch_simulation_circuit.cpp";
            return result;
        });
    }

    for (int cutoff : {16, 32}) {
        const std::string vqe_name =
            "vqe_qubits_2_qumodes_2_depth_2_cutoff_" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, vqe_name, [cutoff, vqe_name, qaoa_max_states]() {
            const Reference::Vector input = make_two_mode_vacuum_state(cutoff);
            const std::vector<double> params = make_vqe_parameters(2, 2, 2);
            ExperimentResult result = run_scaling_case(
                vqe_name,
                "vqe_circuit",
                2,
                2,
                cutoff,
                2,
                qaoa_max_states,
                &input,
                [params](QuantumCircuit& circuit) { add_vqe_circuit_gates(circuit, 2, 2, 2, params); });
            result.params["source_circuit"] = "circuit/src/vqe_circuit.cpp";
            return result;
        });
    }

    for (int num_qubits : {2, 4, 8, 12, 16, 20}) {
        const std::string hdd_name = "hdd_vs_full_tensor_qubits_" + std::to_string(num_qubits);
        append_filtered_scaling_case(results, name_filter, hdd_name, [num_qubits]() {
            return run_hdd_vs_full_tensor_case(num_qubits, 1, 16, std::max(8, num_qubits + 2));
        });
    }

    // SC26 requested scaling: qubits 3-10, qumodes 3-7
    for (int nq : {3, 4, 5, 6, 7, 8, 9, 10}) {
        for (int nm : {3, 4, 5, 6, 7}) {
            const int cutoff = 16;
            const std::string sc_vqe_name = "sc26_vqe_nq" + std::to_string(nq) + 
                                            "_nm" + std::to_string(nm) + "_c" + std::to_string(cutoff);
            append_filtered_scaling_case(results, name_filter, sc_vqe_name, [nq, nm, cutoff, sc_vqe_name, qaoa_max_states]() {
                const std::vector<double> params = make_vqe_parameters(2, nq, nm);
                ExperimentResult result = run_scaling_case(
                    sc_vqe_name,
                    "vqe_circuit",
                    nq,
                    nm,
                    cutoff,
                    2,
                    qaoa_max_states,
                    nullptr, // Use default vacuum
                    [nq, nm, params](QuantumCircuit& circuit) { add_vqe_circuit_gates(circuit, nq, nm, 2, params); });
                result.params["source_circuit"] = "circuit/src/vqe_circuit.cpp";
                return result;
            });

            const std::string sc_jch_name = "sc26_jch_nq" + std::to_string(nq) + 
                                            "_nm" + std::to_string(nm) + "_c" + std::to_string(cutoff);
            append_filtered_scaling_case(results, name_filter, sc_jch_name, [nq, nm, cutoff, sc_jch_name, jch_max_states]() {
                ExperimentResult result = run_scaling_case(
                    sc_jch_name,
                    "jch_simulation_circuit",
                    nq,
                    nm,
                    cutoff,
                    5,
                    jch_max_states,
                    nullptr, // Use default vacuum
                    [nq, nm](QuantumCircuit& circuit) {
                        add_jch_simulation_circuit_gates(circuit, nm, nq, 1.0, 1.0, 1.0, 0.5, 0.1, 5);
                    });
                result.params["source_circuit"] = "circuit/src/jch_simulation_circuit.cpp";
                return result;
            });
        }
    }

    // Pure CV benchmarks (0 qubits) — JCH lattice and CV-QAOA
    // nm=8 requires int64_t for max_total_dim (16^8 = 4.3B), needs ~68.7GB per state
    for (int nm : {5, 6, 7, 8}) {
        const int cutoff = 16;
        // Memory per state: cutoff^nm * 16 bytes (complex double)
        // nm=5: 16.8MB, nm=6: 268MB, nm=7: 4.29GB, nm=8: 68.7GB
        int cv_max = (nm <= 6) ? 64 : (nm == 7) ? 8 : 1;
        if (g_max_states_override > 0) cv_max = g_max_states_override;

        // Deeper circuits for larger nm to get meaningful compute timing
        const int jch_trotter = (nm >= 8) ? 20 : (nm >= 7) ? 10 : 5;
        const int qaoa_layers = (nm >= 8) ? 10 : (nm >= 7) ? 5 : 2;

        const std::string cv_jch_name = "sc26_cv_jch_nm" + std::to_string(nm) + "_c" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, cv_jch_name, [nm, cutoff, cv_jch_name, cv_max, jch_trotter]() {
            ExperimentResult result = run_scaling_case(
                cv_jch_name,
                "cv_jch_lattice",
                0, nm, cutoff, jch_trotter, cv_max,
                nullptr,
                [nm, jch_trotter](QuantumCircuit& circuit) {
                    add_jch_simulation_circuit_gates(circuit, nm, 0, 1.0, 1.0, 1.0, 0.5, 0.1, jch_trotter);
                });
            result.params["source_circuit"] = "circuit/src/jch_simulation_circuit.cpp";
            result.params["pure_cv"] = "true";
            return result;
        });

        const std::string cv_qaoa_name = "sc26_cv_qaoa_nm" + std::to_string(nm) + "_c" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, cv_qaoa_name, [nm, cutoff, cv_qaoa_name, cv_max, qaoa_layers]() {
            const std::vector<double> params = make_qaoa_angles(qaoa_layers);
            ExperimentResult result = run_scaling_case(
                cv_qaoa_name,
                "cv_qaoa_circuit",
                0, nm, cutoff, qaoa_layers, cv_max,
                nullptr,
                [nm, params, qaoa_layers](QuantumCircuit& circuit) {
                    add_cv_qaoa_circuit_gates(circuit, nm, params, 0.5, 1.0, qaoa_layers);
                });
            result.params["pure_cv"] = "true";
            return result;
        });
    }

    // Additional SC26 cases
    for (int cutoff : {16, 32}) {
        const std::string name = "sc26_cat_c" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, name, [cutoff, name, cat_max_states]() {
            return run_scaling_case(name, "cat_state_circuit", 1, 1, cutoff, 8, cat_max_states, nullptr,
                [](QuantumCircuit& circuit) { add_cat_state_circuit_gates(circuit, 1.0, 0); });
        });
    }

    for (int cutoff : {16, 32}) {
        const std::string name = "sc26_gkp_c" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, name, [cutoff, name, gkp_max_states]() {
            return run_scaling_case(name, "gkp_state_circuit", 1, 1, cutoff, 9, gkp_max_states, nullptr,
                [](QuantumCircuit& circuit) { add_gkp_state_circuit_gates(circuit, 9, 0.222, 0); });
        });
    }

    for (int nm : {1, 2, 4, 8}) {
        const int cutoff = 16;
        const std::string name = "sc26_qaoa_nm" + std::to_string(nm) + "_c" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, name, [nm, cutoff, name, qaoa_max_states]() {
            const std::vector<double> params = make_qaoa_angles(2);
            return run_scaling_case(name, "qaoa_circuit", 1, nm, cutoff, 2, qaoa_max_states, nullptr,
                [nm, params](QuantumCircuit& circuit) { add_cv_qaoa_circuit_gates(circuit, nm, params, 0.5, 1.0, 2); });
        });
    }

    for (int nq : {3, 5, 7, 9}) {
        for (int cutoff : {16, 32}) {
            if (cutoff == 32 && nq > 5) continue; // Skip large combinations
            const std::string name = "sc26_qft_nq" + std::to_string(nq) + "_c" + std::to_string(cutoff);
            append_filtered_scaling_case(results, name_filter, name, [nq, cutoff, name, qaoa_max_states]() {
                int n = nq / 2 + 1;
                int a = 1;
                int append = nq - n - a;
                return run_scaling_case(name, "qft_circuit", nq, 1, cutoff, 10, qaoa_max_states, nullptr,
                    [nq, n, a, append](QuantumCircuit& circuit) { add_qft_circuit_gates(circuit, nq, n, a, append); });
            });
        }
    }

    for (int cutoff : {8, 16}) {
        const std::string name = "sc26_shors_c" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, name, [cutoff, name, qaoa_max_states]() {
            return run_scaling_case(name, "shors_circuit", 1, 3, cutoff, 10, qaoa_max_states, nullptr,
                [](QuantumCircuit& circuit) { add_shors_circuit_gates(circuit, 1, 3, 15, 2); });
        });
    }

    for (int nq : {2, 4, 8, 16}) {
        const int cutoff = 16;
        const std::string cvtodv = "sc26_transfer_CVtoDV_nq" + std::to_string(nq) + "_c" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, cvtodv, [nq, cutoff, cvtodv, qaoa_max_states]() {
            return run_scaling_case(cvtodv, "state_transfer_CVtoDV_circuit", nq, 1, cutoff, 8, qaoa_max_states, nullptr,
                [nq](QuantumCircuit& circuit) { add_state_transfer_cvtodv_gates(circuit, nq, 1, 0.29, true); });
        });
        const std::string dvtocv = "sc26_transfer_DVtoCV_nq" + std::to_string(nq) + "_c" + std::to_string(cutoff);
        append_filtered_scaling_case(results, name_filter, dvtocv, [nq, cutoff, dvtocv, qaoa_max_states]() {
            return run_scaling_case(dvtocv, "state_transfer_DVtoCV_circuit", nq, 1, cutoff, 8, qaoa_max_states, nullptr,
                [nq](QuantumCircuit& circuit) { add_state_transfer_dvtocv_gates(circuit, nq, 1, 0.29, true); });
        });
    }
return results;
}

void append_results(std::vector<ExperimentResult>& target, const std::vector<ExperimentResult>& source) {
    target.insert(target.end(), source.begin(), source.end());
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--suite" && i + 1 < argc) {
            options.suite = argv[++i];
        } else if (arg == "--name-filter" && i + 1 < argc) {
            options.name_filter = argv[++i];
        } else if (arg == "--gaussian-symbolic-mode-limit" && i + 1 < argc) {
            options.gaussian_symbolic_mode_limit = std::stoi(argv[++i]);
        } else if (arg == "--use-interaction-picture") {
            options.use_interaction_picture = true;
        } else if (arg == "--output" && i + 1 < argc) {
            options.output_path = fs::path(argv[++i]);
        } else if (arg == "--max-states" && i + 1 < argc) {
            options.max_states_override = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: hybridcvdv_single_gpu_experiments "
                         "[--suite all|correctness|microbench|runtime_ablation|scaling] "
                         "[--name-filter substring] "
                         "[--gaussian-symbolic-mode-limit N] [--use-interaction-picture] "
                         "[--max-states N] [--output path]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.gaussian_symbolic_mode_limit <= 0) {
        throw std::invalid_argument("gaussian-symbolic-mode-limit must be positive");
    }
    return options;
}

void write_report(const fs::path& output_path,
                  const std::string& requested_suite,
                  int gaussian_symbolic_mode_limit,
                  bool use_interaction_picture,
                  const DeviceMetadata& device,
                  const std::vector<ExperimentResult>& results) {
    if (!output_path.parent_path().empty()) {
        fs::create_directories(output_path.parent_path());
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("failed to open output path: " + output_path.string());
    }

    out << "{\n";
    out << "  \"schema_version\": \"1.0\",\n";
    out << "  \"generated_at_utc\": \"" << json_escape(now_utc_iso8601()) << "\",\n";
    out << "  \"simulator\": \"HybridCVDV-Simulator\",\n";
    out << "  \"single_gpu_only\": true,\n";
    out << "  \"requested_suite\": \"" << json_escape(requested_suite) << "\",\n";
    out << "  \"gaussian_symbolic_mode_limit\": " << gaussian_symbolic_mode_limit << ",\n";
    out << "  \"use_interaction_picture\": " << (use_interaction_picture ? "true" : "false") << ",\n";
    out << "  \"device\": {\n";
    out << "    \"available\": " << (device.available ? "true" : "false") << ",\n";
    out << "    \"device_index\": " << device.device_index << ",\n";
    out << "    \"name\": \"" << json_escape(device.name) << "\",\n";
    out << "    \"compute_capability\": \"" << device.cc_major << "." << device.cc_minor << "\",\n";
    out << "    \"multiprocessor_count\": " << device.multiprocessor_count << ",\n";
    out << "    \"total_global_mem_bytes\": " << device.total_global_mem_bytes << ",\n";
    out << "    \"shared_mem_per_block_bytes\": " << device.shared_mem_per_block_bytes << "\n";
    out << "  },\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const ExperimentResult& result = results[i];
        out << "    {\n";
        out << "      \"name\": \"" << json_escape(result.name) << "\",\n";
        out << "      \"category\": \"" << json_escape(result.category) << "\",\n";
        out << "      \"status\": \"" << json_escape(result.status) << "\",\n";
        out << "      \"note\": \"" << json_escape(result.note) << "\",\n";
        out << "      \"params\": {\n";
        size_t param_index = 0;
        for (const auto& [key, value] : result.params) {
            out << "        \"" << json_escape(key) << "\": \"" << json_escape(value) << "\"";
            out << (++param_index < result.params.size() ? ",\n" : "\n");
        }
        out << "      },\n";
        out << "      \"metrics\": {\n";
        size_t metric_index = 0;
        for (const auto& [key, value] : result.metrics) {
            out << "        \"" << json_escape(key) << "\": " << format_double(value);
            out << (++metric_index < result.metrics.size() ? ",\n" : "\n");
        }
        out << "      }\n";
        out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        g_gaussian_symbolic_mode_limit = options.gaussian_symbolic_mode_limit;
        g_use_interaction_picture = options.use_interaction_picture;
        g_max_states_override = options.max_states_override;
        g_scaling_warmup_runs_override =
            parse_nonnegative_env_override("HYBRIDCVDV_SCALING_WARMUP_RUNS", 0);
        g_scaling_measured_runs_override =
            parse_nonnegative_env_override("HYBRIDCVDV_SCALING_MEASURED_RUNS", 1);
        const DeviceMetadata device = query_device();
        if (!device.available) {
            throw std::runtime_error("no CUDA device available for single-GPU experiments");
        }

        std::vector<ExperimentResult> results;
        if (options.suite == "all" || options.suite == "correctness") {
            append_results(results, run_correctness_suite());
        }
        if (options.suite == "all" || options.suite == "microbench") {
            append_results(results, run_microbenchmark_suite());
        }
        if (options.suite == "all" || options.suite == "runtime_ablation") {
            append_results(results, run_runtime_ablation_suite());
        }
        if (options.suite == "all" || options.suite == "scaling") {
            append_results(results, run_scaling_suite(options.name_filter));
        }

        if (results.empty()) {
            throw std::runtime_error("no experiments were selected");
        }

        write_report(options.output_path,
                     options.suite,
                     options.gaussian_symbolic_mode_limit,
                     options.use_interaction_picture,
                     device,
                     results);

        size_t ok_count = 0;
        size_t unsupported_count = 0;
        size_t error_count = 0;
        for (const auto& result : results) {
            if (result.status == "ok") {
                ++ok_count;
            } else if (result.status == "unsupported") {
                ++unsupported_count;
            } else {
                ++error_count;
            }
        }

        std::cout << "Wrote single-GPU experiment report to " << options.output_path << "\n";
        std::cout << "Results: ok=" << ok_count
                  << ", unsupported=" << unsupported_count
                  << ", error=" << error_count << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Experiment runner failed: " << e.what() << std::endl;
        return 1;
    }
}
