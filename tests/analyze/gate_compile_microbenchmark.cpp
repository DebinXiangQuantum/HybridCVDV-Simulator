#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#include <unistd.h>
#endif

// Analysis-only access to private compile/rewrite helpers. This benchmark does
// not change simulator behavior; it only times the existing internal paths.
#define private public
#include "quantum_circuit.h"
#undef private

namespace {

using Clock = std::chrono::steady_clock;

struct NullBuffer : std::streambuf {
    int overflow(int c) override { return c; }
};

class ScopedSilence {
public:
    ScopedSilence()
        : old_cout_(std::cout.rdbuf(&null_)),
          old_cerr_(std::cerr.rdbuf(&null_)) {}

    ~ScopedSilence() {
        std::cout.rdbuf(old_cout_);
        std::cerr.rdbuf(old_cerr_);
    }

private:
    NullBuffer null_;
    std::streambuf* old_cout_;
    std::streambuf* old_cerr_;
};

struct Options {
    int warmup = 64;
    int samples = 1000;
    int compile_repeats = 20000;
    int cutoff = 8;
    int max_states = 4096;
};

struct Summary {
    double mean_us = 0.0;
    double median_us = 0.0;
    double p10_us = 0.0;
    double p90_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

struct HostInfo {
    std::string cpu_model = "unknown";
    unsigned hardware_threads = 0;
    int affinity_cpus = -1;
    int process_threads = -1;
};

struct DDAddTerm {
    int input_basis = 0;
    std::complex<double> coefficient{0.0, 0.0};
};

struct GenericDDAddPlan {
    int dimension = 0;
    std::vector<std::vector<DDAddTerm>> rows;
    int nonzero_terms = 0;
};

struct Dense2QDDAddPlan {
    std::array<std::vector<DDAddTerm>, 4> rows;
    int nonzero_terms = 0;
};

struct WeightedNode {
    HDDNode* node = nullptr;
    std::complex<double> weight{1.0, 0.0};
};

double elapsed_us(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

Summary summarize(std::vector<double> values) {
    if (values.empty()) {
        return {};
    }
    std::sort(values.begin(), values.end());
    const auto pct = [&](double q) {
        const double pos = q * static_cast<double>(values.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        if (lo == hi) {
            return values[lo];
        }
        const double t = pos - static_cast<double>(lo);
        return values[lo] * (1.0 - t) + values[hi] * t;
    };

    Summary s;
    s.mean_us = std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size());
    s.median_us = pct(0.5);
    s.p10_us = pct(0.1);
    s.p90_us = pct(0.9);
    s.min_us = values.front();
    s.max_us = values.back();
    return s;
}

void print_result(const std::string& category,
                  const std::string& name,
                  int qubits,
                  int modes,
                  int ops,
                  const std::string& unit,
                  const Summary& s) {
    std::cout << std::left << std::setw(18) << category
              << std::setw(30) << name
              << std::right << std::setw(4) << qubits
              << std::setw(4) << modes
              << std::setw(6) << ops
              << std::setw(12) << unit
              << std::setw(12) << std::fixed << std::setprecision(3) << s.mean_us
              << std::setw(12) << s.median_us
              << std::setw(12) << s.p10_us
              << std::setw(12) << s.p90_us
              << std::setw(12) << s.min_us
              << std::setw(12) << s.max_us
              << '\n';
}

void synchronize_or_throw() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") +
                                 cudaGetErrorString(err));
    }
}

std::string gpu_name() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return "unknown";
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return "unknown";
    }
    return prop.name;
}

std::string trim_copy(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

HostInfo host_info() {
    HostInfo info;
    info.hardware_threads = std::thread::hardware_concurrency();

#if defined(__linux__)
    {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            const std::string key = "model name";
            if (line.rfind(key, 0) != 0) {
                continue;
            }
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                info.cpu_model = trim_copy(line.substr(colon + 1));
                break;
            }
        }
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        int count = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &mask)) {
                ++count;
            }
        }
        info.affinity_cpus = count;
    }

    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("Threads:", 0) != 0) {
                continue;
            }
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                info.process_threads = std::stoi(trim_copy(line.substr(colon + 1)));
                break;
            }
        }
    }
#endif

    return info;
}

int popcount_int(int value) {
    int count = 0;
    while (value != 0) {
        count += value & 1;
        value >>= 1;
    }
    return count;
}

std::vector<std::complex<double>> dft_unitary(int qubits) {
    constexpr double pi = 3.14159265358979323846;
    const int dim = 1 << qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<std::complex<double>> u(static_cast<size_t>(dim) * dim);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            const double angle =
                2.0 * pi * static_cast<double>(row * col) / static_cast<double>(dim);
            u[static_cast<size_t>(row) * dim + col] =
                scale * std::complex<double>(std::cos(angle), std::sin(angle));
        }
    }
    return u;
}

std::vector<std::complex<double>> walsh_hadamard_unitary(int qubits) {
    const int dim = 1 << qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<std::complex<double>> u(static_cast<size_t>(dim) * dim);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            const int parity = popcount_int(row & col) & 1;
            u[static_cast<size_t>(row) * dim + col] =
                std::complex<double>(parity == 0 ? scale : -scale, 0.0);
        }
    }
    return u;
}

GenericDDAddPlan lower_full_unitary_to_ddadd(
    const std::vector<std::complex<double>>& unitary,
    int dimension,
    double zero_tolerance = 1e-14) {
    if (dimension <= 0 ||
        unitary.size() != static_cast<size_t>(dimension) * dimension) {
        throw std::invalid_argument("unitary dimension mismatch");
    }

    GenericDDAddPlan plan;
    plan.dimension = dimension;
    plan.rows.assign(static_cast<size_t>(dimension), {});
    for (int output_basis = 0; output_basis < dimension; ++output_basis) {
        auto& row = plan.rows[static_cast<size_t>(output_basis)];
        row.reserve(static_cast<size_t>(dimension));
        for (int input_basis = 0; input_basis < dimension; ++input_basis) {
            const std::complex<double> coefficient =
                unitary[static_cast<size_t>(output_basis) * dimension + input_basis];
            if (std::abs(coefficient) <= zero_tolerance) {
                continue;
            }
            row.push_back({input_basis, coefficient});
            ++plan.nonzero_terms;
        }
    }
    return plan;
}

int lowering_repeats_for_dimension(const Options& options, int dimension) {
    const int terms = dimension * dimension;
    constexpr int kTargetTermsPerSample = 250000;
    return std::max(1, std::min(options.compile_repeats,
                                std::max(1, kTargetTermsPerSample / terms)));
}

Summary measure_full_unitary_lowering(const Options& options,
                                      const std::vector<std::complex<double>>& unitary,
                                      int dimension,
                                      int* terms_out,
                                      int* repeats_out) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.samples));
    const int repeats = lowering_repeats_for_dimension(options, dimension);
    int terms = 0;

    for (int sample = 0; sample < options.warmup + options.samples; ++sample) {
        const auto start = Clock::now();
        int guard = 0;
        for (int iter = 0; iter < repeats; ++iter) {
            const GenericDDAddPlan plan =
                lower_full_unitary_to_ddadd(unitary, dimension);
            guard += plan.nonzero_terms;
            terms = plan.nonzero_terms;
        }
        const auto end = Clock::now();
        if (guard <= 0) {
            std::abort();
        }
        if (sample >= options.warmup) {
            samples.push_back(elapsed_us(start, end) / static_cast<double>(repeats));
        }
    }

    if (terms_out) {
        *terms_out = terms;
    }
    if (repeats_out) {
        *repeats_out = repeats;
    }
    return summarize(std::move(samples));
}

std::array<std::complex<double>, 16> dft4_unitary() {
    const std::vector<std::complex<double>> generic = dft_unitary(2);
    std::array<std::complex<double>, 16> u{};
    std::copy(generic.begin(), generic.end(), u.begin());
    return u;
}

Dense2QDDAddPlan lower_dense_2q_unitary_to_ddadd(
    const std::array<std::complex<double>, 16>& unitary,
    double zero_tolerance = 1e-14) {
    Dense2QDDAddPlan plan;
    for (int output_basis = 0; output_basis < 4; ++output_basis) {
        auto& row = plan.rows[static_cast<size_t>(output_basis)];
        row.clear();
        row.reserve(4);
        for (int input_basis = 0; input_basis < 4; ++input_basis) {
            const std::complex<double> coefficient =
                unitary[static_cast<size_t>(output_basis * 4 + input_basis)];
            if (std::abs(coefficient) <= zero_tolerance) {
                continue;
            }
            row.push_back({input_basis, coefficient});
            ++plan.nonzero_terms;
        }
    }
    return plan;
}

WeightedNode extract_basis_branch(QuantumCircuit& circuit, int basis_index) {
    HDDNode* node = circuit.root_node_;
    std::complex<double> weight{1.0, 0.0};
    while (node && !node->is_terminal()) {
        const int bit = (basis_index >> node->qubit_level) & 1;
        if (bit == 0) {
            weight *= node->w_low;
            node = node->low;
        } else {
            weight *= node->w_high;
            node = node->high;
        }
    }
    return {node, weight};
}

void apply_dense_2q_plan_as_ddadd(QuantumCircuit& circuit,
                                  const Dense2QDDAddPlan& plan) {
    const std::array<WeightedNode, 4> inputs = {
        extract_basis_branch(circuit, 0),
        extract_basis_branch(circuit, 1),
        extract_basis_branch(circuit, 2),
        extract_basis_branch(circuit, 3)
    };
    HDDNode* zero = circuit.node_manager_.create_terminal_node(circuit.shared_zero_state_id_);
    std::array<HDDNode*, 4> outputs{};

    for (int output_basis = 0; output_basis < 4; ++output_basis) {
        HDDNode* acc = zero;
        for (const DDAddTerm& term : plan.rows[static_cast<size_t>(output_basis)]) {
            const WeightedNode& input = inputs[static_cast<size_t>(term.input_basis)];
            if (!input.node) {
                continue;
            }
            acc = circuit.hdd_add(acc,
                                  std::complex<double>(1.0, 0.0),
                                  input.node,
                                  term.coefficient * input.weight);
        }
        outputs[static_cast<size_t>(output_basis)] = acc;
    }

    HDDNode* low_q1 = circuit.node_manager_.get_or_create_node(
        0, outputs[0], outputs[1], 1.0, 1.0);
    HDDNode* high_q1 = circuit.node_manager_.get_or_create_node(
        0, outputs[2], outputs[3], 1.0, 1.0);
    HDDNode* new_root = circuit.node_manager_.get_or_create_node(
        1, low_q1, high_q1, 1.0, 1.0);
    circuit.state_pool_.synchronize_all_devices();
    circuit.replace_root_node(new_root);
}

Summary measure_unitary_lowering(const Options& options,
                                 const std::array<std::complex<double>, 16>& unitary,
                                 int* terms_out) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.samples));
    const int repeats = std::max(1, options.compile_repeats);
    int terms = 0;

    for (int sample = 0; sample < options.warmup + options.samples; ++sample) {
        const auto start = Clock::now();
        int guard = 0;
        for (int iter = 0; iter < repeats; ++iter) {
            const Dense2QDDAddPlan plan = lower_dense_2q_unitary_to_ddadd(unitary);
            guard += plan.nonzero_terms;
            terms = plan.nonzero_terms;
        }
        const auto end = Clock::now();
        if (guard <= 0) {
            std::abort();
        }
        if (sample >= options.warmup) {
            samples.push_back(elapsed_us(start, end) / static_cast<double>(repeats));
        }
    }

    if (terms_out) {
        *terms_out = terms;
    }
    return summarize(std::move(samples));
}

Summary measure_unitary_direct_ddadd(const Options& options,
                                     const std::array<std::complex<double>, 16>& unitary,
                                     bool include_lowering,
                                     int* hdd_nodes_out) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.samples));
    int last_hdd_nodes = 0;
    const Dense2QDDAddPlan reusable_plan = lower_dense_2q_unitary_to_ddadd(unitary);

    const int total = options.warmup + options.samples;
    for (int sample = 0; sample < total; ++sample) {
        QuantumCircuit circuit(2, 1, options.cutoff, options.max_states);
        circuit.build();
        circuit.execute_gate(Gates::Hadamard(0));
        circuit.execute_gate(Gates::Hadamard(1));
        synchronize_or_throw();

        const auto start = Clock::now();
        if (include_lowering) {
            const Dense2QDDAddPlan plan = lower_dense_2q_unitary_to_ddadd(unitary);
            apply_dense_2q_plan_as_ddadd(circuit, plan);
        } else {
            apply_dense_2q_plan_as_ddadd(circuit, reusable_plan);
        }
        synchronize_or_throw();
        const auto end = Clock::now();

        if (sample >= options.warmup) {
            samples.push_back(elapsed_us(start, end));
        }
        last_hdd_nodes = static_cast<int>(circuit.get_stats().hdd_nodes);
    }

    if (hdd_nodes_out) {
        *hdd_nodes_out = last_hdd_nodes;
    }
    return summarize(std::move(samples));
}

Summary measure_gaussian_block_compile(const Options& options,
                                       int qubits,
                                       int modes,
                                       const std::vector<GateParams>& gates) {
    QuantumCircuit circuit(qubits, modes, options.cutoff, options.max_states);
    circuit.add_gates(gates);
    const std::vector<GateParams> sequence =
        circuit.canonicalize_gate_sequence_for_execution();
    const std::vector<QuantumCircuit::ExecutionBlock> blocks =
        circuit.partition_execution_blocks(sequence);
    if (blocks.empty()) {
        throw std::runtime_error("no execution blocks to compile");
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.samples));
    const int repeats = std::max(1, options.compile_repeats);

    for (int sample = 0; sample < options.warmup + options.samples; ++sample) {
        const auto start = Clock::now();
        double guard = 0.0;
        for (int iter = 0; iter < repeats; ++iter) {
            const QuantumCircuit::CompiledExecutionBlock compiled =
                circuit.compile_execution_block(sequence, blocks, 0);
            guard += compiled.compile_time_ms;
        }
        const auto end = Clock::now();
        if (guard < 0.0) {
            std::abort();
        }
        if (sample >= options.warmup) {
            samples.push_back(elapsed_us(start, end) /
                              static_cast<double>(repeats * std::max<size_t>(1, gates.size())));
        }
    }
    return summarize(std::move(samples));
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(flag + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--warmup") {
            options.warmup = std::stoi(require_value(arg));
        } else if (arg == "--samples") {
            options.samples = std::stoi(require_value(arg));
        } else if (arg == "--compile-repeats") {
            options.compile_repeats = std::stoi(require_value(arg));
        } else if (arg == "--cutoff") {
            options.cutoff = std::stoi(require_value(arg));
        } else if (arg == "--max-states") {
            options.max_states = std::stoi(require_value(arg));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_args(argc, argv);

    std::vector<std::tuple<std::string, std::string, int, int, int, std::string, Summary>> results;
    std::vector<std::tuple<std::string, int>> node_counts;

    {
        ScopedSilence silence;
        for (int qubits = 2; qubits <= 6; ++qubits) {
            const int dimension = 1 << qubits;
            int terms = 0;
            int repeats = 0;

            results.emplace_back(
                "Unitary_lowering",
                "DFT_" + std::to_string(dimension) + "x" + std::to_string(dimension),
                qubits,
                0,
                dimension * dimension,
                "us/unitary",
                measure_full_unitary_lowering(
                    options,
                    dft_unitary(qubits),
                    dimension,
                    &terms,
                    &repeats));
            node_counts.emplace_back(
                "DFT_" + std::to_string(dimension) + "_terms", terms);
            node_counts.emplace_back(
                "DFT_" + std::to_string(dimension) + "_repeats_per_sample", repeats);

            results.emplace_back(
                "Unitary_lowering",
                "Hadamard_" + std::to_string(dimension) + "x" + std::to_string(dimension),
                qubits,
                0,
                dimension * dimension,
                "us/unitary",
                measure_full_unitary_lowering(
                    options,
                    walsh_hadamard_unitary(qubits),
                    dimension,
                    &terms,
                    &repeats));
            node_counts.emplace_back(
                "Hadamard_" + std::to_string(dimension) + "_terms", terms);
            node_counts.emplace_back(
                "Hadamard_" + std::to_string(dimension) + "_repeats_per_sample", repeats);
        }
    }

    std::cout << "Gantry full-unitary descriptor lowering microbenchmark\n";
    const HostInfo host = host_info();
    std::cout << "CPU: " << host.cpu_model << '\n';
    std::cout << "Host threads: benchmark driver=1"
              << " process_threads=" << host.process_threads
              << " affinity_cpus=" << host.affinity_cpus
              << " hardware_concurrency=" << host.hardware_threads << '\n';
    std::cout << "GPU: " << gpu_name() << " (queried only for host/runtime context)\n";
    std::cout << "samples=" << options.samples
              << " warmup=" << options.warmup
              << " cutoff=" << options.cutoff
              << " max_states=" << options.max_states
              << " compile_repeats=" << options.compile_repeats << "\n\n";

    std::cout << std::left << std::setw(18) << "category"
              << std::setw(30) << "case"
              << std::right << std::setw(4) << "nq"
              << std::setw(4) << "nm"
              << std::setw(6) << "terms"
              << std::setw(12) << "unit"
              << std::setw(12) << "mean_us"
              << std::setw(12) << "median_us"
              << std::setw(12) << "p10_us"
              << std::setw(12) << "p90_us"
              << std::setw(12) << "min_us"
              << std::setw(12) << "max_us"
              << '\n';

    for (const auto& [category, name, qubits, modes, ops, unit, summary] : results) {
        print_result(category, name, qubits, modes, ops, unit, summary);
    }

    std::cout << "\nextra counters:\n";
    for (const auto& [name, nodes] : node_counts) {
        std::cout << "  " << name << ": " << nodes << '\n';
    }

    return 0;
}
