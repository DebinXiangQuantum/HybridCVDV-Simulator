#include "quantum_circuit.h"

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace {

using Gates::AnnihilationOperator;
using Gates::BeamSplitter;
using Gates::ConditionalParity;
using Gates::CreationOperator;
using Gates::Displacement;
using Gates::KerrGate;
using Gates::PhaseRotation;
using Gates::Squeezing;

struct CliOptions {
    int num_qubits = 1;
    int num_qumodes = 3;
    int cutoff = 16;
    int max_states = 1024;
    int total_gates = 240;
    int checkpoint_step = 40;
    int warmup_runs = 1;
    int measured_runs = 5;
};

struct NullBuffer : std::streambuf {
    int overflow(int c) override { return c; }
};

class ScopedStreamSilencer {
public:
    explicit ScopedStreamSilencer(std::ostream& stream)
        : stream_(stream), old_buffer_(stream.rdbuf(&null_buffer_)) {}

    ~ScopedStreamSilencer() {
        stream_.rdbuf(old_buffer_);
    }

private:
    NullBuffer null_buffer_;
    std::ostream& stream_;
    std::streambuf* old_buffer_;
};

struct Sample {
    double total_ms = 0.0;
    double transfer_ms = 0.0;
    double compute_ms = 0.0;
    double memory_bytes = 0.0;
    double active_states = 0.0;
    double hdd_nodes = 0.0;
};

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t low = static_cast<size_t>(pos);
    const size_t high = std::min(values.size() - 1, low + 1);
    if (low == high) {
        return values[low];
    }

    const double frac = pos - static_cast<double>(low);
    return values[low] * (1.0 - frac) + values[high] * frac;
}

GateParams make_gaussian_gate(int gate_index, int num_qumodes) {
    const int mode = gate_index % num_qumodes;
    const int next_mode = (mode + 1) % num_qumodes;

    switch (gate_index % 4) {
        case 0:
            return PhaseRotation(mode, 0.11 + 0.01 * static_cast<double>(gate_index % 5));
        case 1:
            return Displacement(mode, std::complex<double>(0.07, -0.03));
        case 2:
            return Squeezing(mode, std::complex<double>(0.045, 0.0));
        default:
            return BeamSplitter(mode, next_mode, 0.18, 0.0);
    }
}

GateParams make_non_gaussian_gate(int gate_index, int num_qumodes) {
    const int mode = gate_index % num_qumodes;

    switch (gate_index % 4) {
        case 0:
            return KerrGate(mode, 0.02);
        case 1:
            return CreationOperator(mode);
        case 2:
            return AnnihilationOperator(mode);
        default:
            return ConditionalParity(mode, 1.0);
    }
}

std::vector<GateParams> build_gaussian_first_sequence(int total_gates, int num_qumodes) {
    const int gaussian_gates = total_gates / 2;
    const int non_gaussian_gates = total_gates - gaussian_gates;

    std::vector<GateParams> sequence;
    sequence.reserve(total_gates);

    for (int i = 0; i < gaussian_gates; ++i) {
        sequence.push_back(make_gaussian_gate(i, num_qumodes));
    }
    for (int i = 0; i < non_gaussian_gates; ++i) {
        sequence.push_back(make_non_gaussian_gate(i, num_qumodes));
    }

    return sequence;
}

std::vector<GateParams> build_non_gaussian_first_sequence(int total_gates, int num_qumodes) {
    const int gaussian_gates = total_gates / 2;
    const int non_gaussian_gates = total_gates - gaussian_gates;

    std::vector<GateParams> sequence;
    sequence.reserve(total_gates);

    for (int i = 0; i < non_gaussian_gates; ++i) {
        sequence.push_back(make_non_gaussian_gate(i, num_qumodes));
    }
    for (int i = 0; i < gaussian_gates; ++i) {
        sequence.push_back(make_gaussian_gate(i, num_qumodes));
    }

    return sequence;
}

std::vector<GateParams> build_alternating_sequence(int total_gates, int num_qumodes) {
    const int gaussian_gates = total_gates / 2;
    const int non_gaussian_gates = total_gates - gaussian_gates;

    std::vector<GateParams> sequence;
    sequence.reserve(total_gates);

    int gaussian_index = 0;
    int non_gaussian_index = 0;
    while (gaussian_index < gaussian_gates || non_gaussian_index < non_gaussian_gates) {
        if (gaussian_index < gaussian_gates) {
            sequence.push_back(make_gaussian_gate(gaussian_index++, num_qumodes));
        }
        if (non_gaussian_index < non_gaussian_gates) {
            sequence.push_back(make_non_gaussian_gate(non_gaussian_index++, num_qumodes));
        }
    }

    return sequence;
}

std::vector<int> build_checkpoints(int total_gates, int checkpoint_step) {
    std::vector<int> checkpoints;
    for (int depth = checkpoint_step; depth < total_gates; depth += checkpoint_step) {
        checkpoints.push_back(depth);
    }
    if (checkpoints.empty() || checkpoints.back() != total_gates) {
        checkpoints.push_back(total_gates);
    }
    return checkpoints;
}

Sample run_prefix_sample(const CliOptions& options,
                         const std::vector<GateParams>& full_sequence,
                         int prefix_length) {
    ScopedStreamSilencer silence_stdout(std::cout);

    QuantumCircuit circuit(options.num_qubits, options.num_qumodes, options.cutoff, options.max_states);
    for (int i = 0; i < prefix_length; ++i) {
        circuit.add_gate(full_sequence[static_cast<size_t>(i)]);
    }

    circuit.build();
    circuit.execute();

    const auto time_stats = circuit.get_time_stats();
    const auto circuit_stats = circuit.get_stats();

    return {
        time_stats.total_time,
        time_stats.transfer_time,
        time_stats.computation_time,
        static_cast<double>(circuit.get_state_pool().get_memory_usage()),
        static_cast<double>(circuit_stats.active_states),
        static_cast<double>(circuit_stats.hdd_nodes)
    };
}

Sample benchmark_prefix(const CliOptions& options,
                        const std::vector<GateParams>& full_sequence,
                        int prefix_length) {
    for (int i = 0; i < options.warmup_runs; ++i) {
        (void)run_prefix_sample(options, full_sequence, prefix_length);
    }

    std::vector<double> total_ms;
    std::vector<double> transfer_ms;
    std::vector<double> compute_ms;
    std::vector<double> memory_bytes;
    std::vector<double> active_states;
    std::vector<double> hdd_nodes;

    total_ms.reserve(options.measured_runs);
    transfer_ms.reserve(options.measured_runs);
    compute_ms.reserve(options.measured_runs);
    memory_bytes.reserve(options.measured_runs);
    active_states.reserve(options.measured_runs);
    hdd_nodes.reserve(options.measured_runs);

    for (int i = 0; i < options.measured_runs; ++i) {
        const Sample sample = run_prefix_sample(options, full_sequence, prefix_length);
        total_ms.push_back(sample.total_ms);
        transfer_ms.push_back(sample.transfer_ms);
        compute_ms.push_back(sample.compute_ms);
        memory_bytes.push_back(sample.memory_bytes);
        active_states.push_back(sample.active_states);
        hdd_nodes.push_back(sample.hdd_nodes);
    }

    return {
        percentile(total_ms, 0.5),
        percentile(transfer_ms, 0.5),
        percentile(compute_ms, 0.5),
        percentile(memory_bytes, 0.5),
        percentile(active_states, 0.5),
        percentile(hdd_nodes, 0.5)
    };
}

void print_case_header(const CliOptions& options) {
    std::cout << "config,num_qubits,num_qumodes,cutoff,total_gates,checkpoint,"
                 "memory_mb,total_ms,transfer_ms,compute_ms,overhead_ms,active_states,hdd_nodes\n";
    std::cout << std::fixed << std::setprecision(6);
}

void print_case_row(const std::string& config_name,
                    const CliOptions& options,
                    int checkpoint,
                    const Sample& sample) {
    const double memory_mb = sample.memory_bytes / (1024.0 * 1024.0);
    const double overhead_ms = sample.total_ms - sample.transfer_ms - sample.compute_ms;

    std::cout << config_name << ','
              << options.num_qubits << ','
              << options.num_qumodes << ','
              << options.cutoff << ','
              << options.total_gates << ','
              << checkpoint << ','
              << memory_mb << ','
              << sample.total_ms << ','
              << sample.transfer_ms << ','
              << sample.compute_ms << ','
              << overhead_ms << ','
              << sample.active_states << ','
              << sample.hdd_nodes << '\n';
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto read_int = [&](int& field) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            field = std::atoi(argv[++i]);
        };

        if (arg == "--num-qubits") {
            read_int(options.num_qubits);
        } else if (arg == "--num-qumodes") {
            read_int(options.num_qumodes);
        } else if (arg == "--cutoff") {
            read_int(options.cutoff);
        } else if (arg == "--max-states") {
            read_int(options.max_states);
        } else if (arg == "--total-gates") {
            read_int(options.total_gates);
        } else if (arg == "--checkpoint-step") {
            read_int(options.checkpoint_step);
        } else if (arg == "--warmup-runs") {
            read_int(options.warmup_runs);
        } else if (arg == "--measured-runs") {
            read_int(options.measured_runs);
        } else if (arg == "--help") {
            std::cout << "Usage: gaussian_order_benchmark [--num-qubits N] [--num-qumodes M]"
                         " [--cutoff D] [--max-states S] [--total-gates G]"
                         " [--checkpoint-step C] [--warmup-runs W] [--measured-runs R]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.num_qumodes < 2) {
        throw std::invalid_argument("gaussian order benchmark requires at least 2 qumodes");
    }
    if (options.total_gates <= 0 || options.checkpoint_step <= 0) {
        throw std::invalid_argument("total gates and checkpoint step must be positive");
    }
    if (options.measured_runs <= 0) {
        throw std::invalid_argument("measured runs must be positive");
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);
        const auto checkpoints = build_checkpoints(options.total_gates, options.checkpoint_step);

        const std::vector<std::pair<std::string, std::vector<GateParams>>> cases = {
            {"gaussian_first", build_gaussian_first_sequence(options.total_gates, options.num_qumodes)},
            {"alternating", build_alternating_sequence(options.total_gates, options.num_qumodes)},
            {"non_gaussian_first", build_non_gaussian_first_sequence(options.total_gates, options.num_qumodes)}
        };

        print_case_header(options);
        for (const auto& [name, sequence] : cases) {
            for (int checkpoint : checkpoints) {
                const Sample sample = benchmark_prefix(options, sequence, checkpoint);
                print_case_row(name, options, checkpoint, sample);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gaussian_order_benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
