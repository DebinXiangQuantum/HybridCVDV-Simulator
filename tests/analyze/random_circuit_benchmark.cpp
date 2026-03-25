#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <execinfo.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "quantum_circuit.h"

namespace {

void fatal_signal_handler(int signal_number) {
    const char* signal_name = strsignal(signal_number);
    std::cerr << "random_circuit_benchmark fatal signal: " << signal_number;
    if (signal_name) {
        std::cerr << " (" << signal_name << ")";
    }
    std::cerr << '\n';

    void* frames[64];
    const int frame_count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
    std::_Exit(128 + signal_number);
}

void install_fatal_handlers() {
    std::signal(SIGABRT, fatal_signal_handler);
    std::signal(SIGBUS, fatal_signal_handler);
    std::signal(SIGFPE, fatal_signal_handler);
    std::signal(SIGILL, fatal_signal_handler);
    std::signal(SIGSEGV, fatal_signal_handler);
    std::set_terminate([]() {
        std::cerr << "random_circuit_benchmark terminate handler invoked\n";
        if (const std::exception_ptr current = std::current_exception()) {
            try {
                std::rethrow_exception(current);
            } catch (const std::exception& e) {
                std::cerr << "terminate reason: " << e.what() << '\n';
            } catch (...) {
                std::cerr << "terminate reason: non-std exception\n";
            }
        }

        void* frames[64];
        const int frame_count = backtrace(frames, 64);
        backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
        std::_Exit(134);
    });
}

constexpr double kPi = 3.14159265358979323846;

enum class NonGaussianProfile {
    General,
    DiagonalOnly
};

struct BenchmarkOptions {
    int num_qubits = 2;
    int num_qumodes = 3;
    int cutoff = 8;
    int max_states = 4096;
    int gaussian_pool_capacity = 0;
    int symbolic_branch_limit = 0;
    int total_gates = 72;
    int max_blocks_per_run = 0;
    std::uint64_t seed = 20260315ULL;
    std::string checkpoint_file;
    bool resume_from_checkpoint = false;
    bool gaussian_only = false;
    int non_gaussian_gates = -1;
    NonGaussianProfile non_gaussian_profile = NonGaussianProfile::General;
    bool exclude_ladder_gates = false;
    bool exclude_advanced_hybrid_gates = false;
};

struct BuiltGate {
    GateParams gate;
    std::string name;
};

using GateBuilder = std::function<BuiltGate(std::mt19937_64&)>;

double peak_rss_mb() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return -1.0;
    }
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

int random_index(std::mt19937_64& rng, int upper_exclusive) {
    std::uniform_int_distribution<int> dist(0, upper_exclusive - 1);
    return dist(rng);
}

int random_distinct_index(std::mt19937_64& rng, int upper_exclusive, int forbidden) {
    int candidate = random_index(rng, upper_exclusive);
    while (candidate == forbidden) {
        candidate = random_index(rng, upper_exclusive);
    }
    return candidate;
}

double random_real(std::mt19937_64& rng, double low, double high) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(rng);
}

std::complex<double> random_complex(std::mt19937_64& rng, double radius_low, double radius_high) {
    const double radius = random_real(rng, radius_low, radius_high);
    const double phase = random_real(rng, -kPi, kPi);
    return std::polar(radius, phase);
}

NonGaussianProfile parse_non_gaussian_profile(const std::string& value) {
    if (value == "general") {
        return NonGaussianProfile::General;
    }
    if (value == "diagonal") {
        return NonGaussianProfile::DiagonalOnly;
    }
    throw std::invalid_argument("unknown non-gaussian-profile: " + value);
}

BenchmarkOptions parse_args(int argc, char** argv) {
    BenchmarkOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--num-qubits") {
            options.num_qubits = std::stoi(require_value("--num-qubits"));
        } else if (arg == "--num-qumodes") {
            options.num_qumodes = std::stoi(require_value("--num-qumodes"));
        } else if (arg == "--cutoff") {
            options.cutoff = std::stoi(require_value("--cutoff"));
        } else if (arg == "--max-states") {
            options.max_states = std::stoi(require_value("--max-states"));
        } else if (arg == "--gaussian-pool-capacity") {
            options.gaussian_pool_capacity =
                std::stoi(require_value("--gaussian-pool-capacity"));
        } else if (arg == "--symbolic-branch-limit") {
            options.symbolic_branch_limit =
                std::stoi(require_value("--symbolic-branch-limit"));
        } else if (arg == "--total-gates") {
            options.total_gates = std::stoi(require_value("--total-gates"));
        } else if (arg == "--max-blocks-per-run") {
            options.max_blocks_per_run = std::stoi(require_value("--max-blocks-per-run"));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint64_t>(std::stoull(require_value("--seed")));
        } else if (arg == "--checkpoint-file") {
            options.checkpoint_file = require_value("--checkpoint-file");
        } else if (arg == "--resume-from-checkpoint") {
            options.resume_from_checkpoint = true;
        } else if (arg == "--gaussian-only") {
            options.gaussian_only = true;
        } else if (arg == "--non-gaussian-gates") {
            options.non_gaussian_gates = std::stoi(require_value("--non-gaussian-gates"));
        } else if (arg == "--non-gaussian-profile") {
            options.non_gaussian_profile =
                parse_non_gaussian_profile(require_value("--non-gaussian-profile"));
        } else if (arg == "--exclude-ladder-gates") {
            options.exclude_ladder_gates = true;
        } else if (arg == "--exclude-advanced-hybrid-gates") {
            options.exclude_advanced_hybrid_gates = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.num_qubits < 2) {
        throw std::invalid_argument("random benchmark requires at least 2 qubits");
    }
    if (options.num_qumodes < 2) {
        throw std::invalid_argument("random benchmark requires at least 2 qumodes");
    }
    if (options.cutoff < 4) {
        throw std::invalid_argument("cutoff must be at least 4");
    }
    if (options.max_states <= 0 || options.total_gates <= 0) {
        throw std::invalid_argument("max_states and total_gates must be positive");
    }
    if (options.gaussian_pool_capacity < 0) {
        throw std::invalid_argument("gaussian-pool-capacity must be non-negative");
    }
    if (options.gaussian_only && options.non_gaussian_gates >= 0) {
        throw std::invalid_argument("gaussian-only and non-gaussian-gates cannot be used together");
    }
    if (options.non_gaussian_gates > options.total_gates) {
        throw std::invalid_argument("non-gaussian-gates cannot exceed total-gates");
    }
    if (options.non_gaussian_gates < -1) {
        throw std::invalid_argument("non-gaussian-gates must be -1 or a non-negative integer");
    }
    if (options.max_blocks_per_run < 0) {
        throw std::invalid_argument("max-blocks-per-run must be non-negative");
    }
    if (options.resume_from_checkpoint && options.checkpoint_file.empty()) {
        throw std::invalid_argument("resume-from-checkpoint requires --checkpoint-file");
    }

    return options;
}

BuiltGate make_random_qubit_rotation(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int qubit = random_index(rng, options.num_qubits);
    const double theta = random_real(rng, -0.9, 0.9);
    switch (random_index(rng, 3)) {
        case 0:
            return {Gates::RotationX(qubit, theta), "RotationX"};
        case 1:
            return {Gates::RotationY(qubit, theta), "RotationY"};
        default:
            return {Gates::RotationZ(qubit, theta), "RotationZ"};
    }
}

BuiltGate make_random_phase_family(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int qubit = random_index(rng, options.num_qubits);
    switch (random_index(rng, 4)) {
        case 0:
            return {Gates::Hadamard(qubit), "Hadamard"};
        case 1:
            return {Gates::PauliX(qubit), "PauliX"};
        case 2:
            return {Gates::PauliZ(qubit), "PauliZ"};
        default:
            return {Gates::PhaseGateS(qubit), "PhaseS"};
    }
}

BuiltGate make_random_two_qubit_gate(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int control = random_index(rng, options.num_qubits);
    const int target = random_distinct_index(rng, options.num_qubits, control);
    if (random_index(rng, 2) == 0) {
        return {Gates::CNOT(control, target), "CNOT"};
    }
    return {Gates::CZ(control, target), "CZ"};
}

BuiltGate make_random_single_mode_gaussian(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int qumode = random_index(rng, options.num_qumodes);
    switch (random_index(rng, 3)) {
        case 0:
            return {Gates::PhaseRotation(qumode, random_real(rng, -0.7, 0.7)), "PhaseRotation"};
        case 1:
            return {Gates::Displacement(qumode, random_complex(rng, 0.05, 0.22)), "Displacement"};
        default:
            return {Gates::Squeezing(qumode, random_complex(rng, 0.03, 0.16)), "Squeezing"};
    }
}

BuiltGate make_random_two_mode_gaussian(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int first = random_index(rng, options.num_qumodes);
    const int second = random_distinct_index(rng, options.num_qumodes, first);
    return {
        Gates::BeamSplitter(
            first,
            second,
            random_real(rng, 0.08, 0.45),
            random_real(rng, -0.6, 0.6)),
        "BeamSplitter"
    };
}

BuiltGate make_random_diagonal_non_gaussian(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int choice = random_index(rng, 5);
    if (choice == 0) {
        const int qumode = random_index(rng, options.num_qumodes);
        return {Gates::KerrGate(qumode, random_real(rng, -0.025, 0.025)), "Kerr"};
    }
    if (choice == 1) {
        const int qumode = random_index(rng, options.num_qumodes);
        return {Gates::ConditionalParity(qumode, random_index(rng, 2) == 0 ? 1.0 : -1.0), "ConditionalParity"};
    }
    if (choice == 2) {
        const int qumode = random_index(rng, options.num_qumodes);
        const int target_fock = std::min(options.cutoff - 1, 1 + random_index(rng, 4));
        return {Gates::Snap(qumode, random_real(rng, -0.5, 0.5), target_fock), "SNAP"};
    }
    if (choice == 3) {
        const int qumode = random_index(rng, options.num_qumodes);
        const int phase_count = std::min(options.cutoff, 4 + random_index(rng, 3));
        std::vector<double> phase_map(static_cast<size_t>(phase_count), 0.0);
        for (double& phase : phase_map) {
            phase = random_real(rng, -0.35, 0.35);
        }
        return {Gates::MultiSNAP(qumode, phase_map), "MultiSNAP"};
    }

    const int first = random_index(rng, options.num_qumodes);
    const int second = random_distinct_index(rng, options.num_qumodes, first);
    return {Gates::CrossKerr(first, second, random_real(rng, -0.02, 0.02)), "CrossKerr"};
}

BuiltGate make_random_ladder_gate(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int qumode = random_index(rng, options.num_qumodes);
    if (random_index(rng, 2) == 0) {
        return {Gates::CreationOperator(qumode), "Creation"};
    }
    return {Gates::AnnihilationOperator(qumode), "Annihilation"};
}

BuiltGate make_random_controlled_gaussian(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int control = random_index(rng, options.num_qubits);
    const int choice = random_index(rng, 3);
    if (choice == 0) {
        const int qumode = random_index(rng, options.num_qumodes);
        return {
            Gates::ConditionalDisplacement(control, qumode, random_complex(rng, 0.04, 0.18)),
            "ConditionalDisplacement"
        };
    }
    if (choice == 1) {
        const int qumode = random_index(rng, options.num_qumodes);
        return {
            Gates::ConditionalSqueezing(control, qumode, random_complex(rng, 0.03, 0.12)),
            "ConditionalSqueezing"
        };
    }

    const int first = random_index(rng, options.num_qumodes);
    const int second = random_distinct_index(rng, options.num_qumodes, first);
    return {
        Gates::ConditionalBeamSplitter(
            control,
            first,
            second,
            random_real(rng, 0.05, 0.28),
            random_real(rng, -0.4, 0.4)),
        "ConditionalBeamSplitter"
    };
}

BuiltGate make_random_two_mode_hybrid(std::mt19937_64& rng, const BenchmarkOptions& options) {
    const int control = random_index(rng, options.num_qubits);
    const int first = random_index(rng, options.num_qumodes);
    const int second = random_distinct_index(rng, options.num_qumodes, first);
    if (random_index(rng, 2) == 0) {
        return {
            Gates::ConditionalTwoModeSqueezing(
                control,
                first,
                second,
                random_complex(rng, 0.03, 0.12)),
            "ConditionalTwoModeSqueezing"
        };
    }
    return {
        Gates::ConditionalSUM(
            control,
            first,
            second,
            random_real(rng, 0.03, 0.16),
            0.0),
        "ConditionalSUM"
    };
}

std::vector<GateBuilder> make_gaussian_builders(const BenchmarkOptions& options) {
    return {
        [&](std::mt19937_64& local_rng) { return make_random_single_mode_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_two_mode_gaussian(local_rng, options); }
    };
}

std::vector<GateBuilder> make_non_gaussian_builders(const BenchmarkOptions& options) {
    if (options.non_gaussian_profile == NonGaussianProfile::DiagonalOnly) {
        return {
            [&](std::mt19937_64& local_rng) { return make_random_diagonal_non_gaussian(local_rng, options); }
        };
    }

    return {
        [&](std::mt19937_64& local_rng) { return make_random_diagonal_non_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_ladder_gate(local_rng, options); }
    };
}

void append_random_gates(std::vector<BuiltGate>* gates,
                         int count,
                         const std::vector<GateBuilder>& builders,
                         std::mt19937_64* rng,
                         bool ensure_coverage) {
    if (!gates || !rng || count <= 0 || builders.empty()) {
        return;
    }

    if (ensure_coverage) {
        const int coverage = std::min(count, static_cast<int>(builders.size()));
        for (int i = 0; i < coverage; ++i) {
            gates->push_back(builders[static_cast<size_t>(i)](*rng));
        }
    }

    while (static_cast<int>(gates->size()) < count) {
        const int family = random_index(*rng, static_cast<int>(builders.size()));
        gates->push_back(builders[static_cast<size_t>(family)](*rng));
    }
}

std::vector<BuiltGate> build_random_circuit(const BenchmarkOptions& options) {
    std::mt19937_64 rng(options.seed);
    std::vector<BuiltGate> gates;
    gates.reserve(static_cast<size_t>(options.total_gates));

    if (options.gaussian_only) {
        const std::vector<GateBuilder> gaussian_builders = make_gaussian_builders(options);
        append_random_gates(&gates, options.total_gates, gaussian_builders, &rng, true);
        std::shuffle(gates.begin(), gates.end(), rng);
        return gates;
    }

    if (options.non_gaussian_gates >= 0) {
        const int gaussian_gate_count = options.total_gates - options.non_gaussian_gates;
        std::vector<BuiltGate> gaussian_gates;
        std::vector<BuiltGate> non_gaussian_gates;
        gaussian_gates.reserve(static_cast<size_t>(gaussian_gate_count));
        non_gaussian_gates.reserve(static_cast<size_t>(options.non_gaussian_gates));

        const std::vector<GateBuilder> gaussian_builders = make_gaussian_builders(options);
        const std::vector<GateBuilder> non_gaussian_builders = make_non_gaussian_builders(options);
        append_random_gates(&gaussian_gates, gaussian_gate_count, gaussian_builders, &rng, true);
        append_random_gates(&non_gaussian_gates, options.non_gaussian_gates, non_gaussian_builders, &rng, true);

        gates.insert(gates.end(), gaussian_gates.begin(), gaussian_gates.end());
        gates.insert(gates.end(), non_gaussian_gates.begin(), non_gaussian_gates.end());
        std::shuffle(gates.begin(), gates.end(), rng);
        return gates;
    }

    gates.push_back({Gates::Hadamard(0), "Hadamard"});
    gates.push_back({Gates::Hadamard(1), "Hadamard"});

    std::vector<GateBuilder> coverage_builders = {
        [&](std::mt19937_64& local_rng) { return make_random_qubit_rotation(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_phase_family(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_two_qubit_gate(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_single_mode_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_two_mode_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_diagonal_non_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_controlled_gaussian(local_rng, options); }
    };
    if (!options.exclude_ladder_gates) {
        coverage_builders.insert(
            coverage_builders.begin() + 6,
            [&](std::mt19937_64& local_rng) { return make_random_ladder_gate(local_rng, options); });
    }
    if (!options.exclude_advanced_hybrid_gates) {
        coverage_builders.push_back(
            [&](std::mt19937_64& local_rng) { return make_random_two_mode_hybrid(local_rng, options); });
    }

    std::vector<BuiltGate> randomized_tail;
    randomized_tail.reserve(static_cast<size_t>(std::max(0, options.total_gates - 2)));

    for (const auto& builder : coverage_builders) {
        randomized_tail.push_back(builder(rng));
    }

    while (static_cast<int>(gates.size() + randomized_tail.size()) < options.total_gates) {
        const int family = random_index(rng, static_cast<int>(coverage_builders.size()));
        randomized_tail.push_back(coverage_builders[static_cast<size_t>(family)](rng));
    }

    std::shuffle(randomized_tail.begin(), randomized_tail.end(), rng);
    gates.insert(gates.end(), randomized_tail.begin(), randomized_tail.end());
    return gates;
}

void print_gate_histogram(const std::vector<BuiltGate>& gates) {
    std::map<std::string, int> histogram;
    for (const BuiltGate& gate : gates) {
        ++histogram[gate.name];
    }

    std::cout << "Gate histogram:\n";
    for (const auto& entry : histogram) {
        std::cout << "  " << std::setw(26) << std::left << entry.first
                  << entry.second << '\n';
    }
}

void print_gate_preview(const std::vector<BuiltGate>& gates) {
    constexpr size_t kPreviewCount = 20;
    std::cout << "Gate preview (first " << std::min(kPreviewCount, gates.size()) << "): ";
    for (size_t i = 0; i < gates.size() && i < kPreviewCount; ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << gates[i].name;
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        install_fatal_handlers();
        const BenchmarkOptions options = parse_args(argc, argv);
        const std::vector<BuiltGate> gates = build_random_circuit(options);

        std::cout << "=================================================\n";
        std::cout << " Random Circuit Benchmark\n";
        std::cout << "=================================================\n";
        std::cout << "seed=" << options.seed
                  << ", qubits=" << options.num_qubits
                  << ", qumodes=" << options.num_qumodes
                  << ", cutoff=" << options.cutoff
                  << ", max_states=" << options.max_states;
        if (options.gaussian_pool_capacity > 0) {
            std::cout << ", gaussian_pool_capacity=" << options.gaussian_pool_capacity;
        }
        if (options.symbolic_branch_limit > 0) {
            std::cout << ", symbolic_branch_limit=" << options.symbolic_branch_limit;
        }
        if (options.max_blocks_per_run > 0) {
            std::cout << ", max_blocks_per_run=" << options.max_blocks_per_run;
        }
        if (!options.checkpoint_file.empty()) {
            std::cout << ", checkpoint_file=" << options.checkpoint_file;
            if (options.resume_from_checkpoint) {
                std::cout << " (resume)";
            }
        }
        std::cout
                  << ", total_gates=" << options.total_gates;
        if (options.gaussian_only) {
            std::cout << ", workload=gaussian-only";
        } else if (options.non_gaussian_gates >= 0) {
            std::cout << ", workload=gaussian-plus-" << options.non_gaussian_gates << "-non-gaussian";
            std::cout << ", non_gaussian_profile="
                      << (options.non_gaussian_profile == NonGaussianProfile::DiagonalOnly
                              ? "diagonal"
                              : "general");
        } else {
            std::cout << ", workload=mixed";
            if (options.exclude_ladder_gates) {
                std::cout << "-no-ladder";
            }
            if (options.exclude_advanced_hybrid_gates) {
                std::cout << "-no-advanced-hybrid";
            }
        }
        std::cout << '\n';
        print_gate_histogram(gates);
        print_gate_preview(gates);
        std::cout << "=================================================\n";

        QuantumCircuit circuit(
            options.num_qubits,
            options.num_qumodes,
            options.cutoff,
            options.max_states);
        if (options.gaussian_pool_capacity > 0) {
            circuit.set_gaussian_state_pool_capacity(options.gaussian_pool_capacity);
        }
        if (options.symbolic_branch_limit > 0) {
            circuit.set_symbolic_branch_limit(options.symbolic_branch_limit);
        }

        for (const BuiltGate& built_gate : gates) {
            circuit.add_gate(built_gate.gate);
        }

        const auto wall_start = std::chrono::high_resolution_clock::now();

        const auto build_start = std::chrono::high_resolution_clock::now();
        circuit.build();
        const auto build_end = std::chrono::high_resolution_clock::now();

        const size_t total_blocks = circuit.get_execution_block_count();
        size_t checkpoint_total_blocks = total_blocks;
        size_t start_block = 0;
        if (options.resume_from_checkpoint) {
            start_block = circuit.load_exact_fock_checkpoint(
                options.checkpoint_file,
                &checkpoint_total_blocks);
            if (checkpoint_total_blocks != total_blocks) {
                throw std::runtime_error("checkpoint total block count does not match current circuit");
            }
        }

        const auto execute_start = std::chrono::high_resolution_clock::now();
        const size_t next_block = circuit.execute_range(
            start_block,
            options.max_blocks_per_run > 0
                ? static_cast<size_t>(options.max_blocks_per_run)
                : std::numeric_limits<size_t>::max());
        const auto execute_end = std::chrono::high_resolution_clock::now();

        const bool completed = (next_block == total_blocks);
        if (!completed) {
            if (options.checkpoint_file.empty()) {
                throw std::runtime_error(
                    "execution stopped before completion but no checkpoint file was provided");
            }
            circuit.save_exact_fock_checkpoint(options.checkpoint_file, next_block, total_blocks);
        } else if (!options.checkpoint_file.empty()) {
            std::remove(options.checkpoint_file.c_str());
        }

        const auto wall_end = std::chrono::high_resolution_clock::now();

        const double build_ms =
            std::chrono::duration<double, std::milli>(build_end - build_start).count();
        const double execute_wall_ms =
            std::chrono::duration<double, std::milli>(execute_end - execute_start).count();
        const double total_wall_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

        const QuantumCircuit::TimeStats time_stats = circuit.get_time_stats();
        const QuantumCircuit::CircuitStats stats = circuit.get_stats();
        const double pool_mb =
            static_cast<double>(circuit.get_state_pool().get_memory_usage()) / (1024.0 * 1024.0);
        const double peak_rss = peak_rss_mb();

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Build wall time (ms):     " << build_ms << '\n';
        std::cout << "Execute wall time (ms):   " << execute_wall_ms << '\n';
        std::cout << "End-to-end wall time (ms): " << total_wall_ms << '\n';
        std::cout << "Execution blocks total:   " << total_blocks << '\n';
        std::cout << "Execution start block:    " << start_block << '\n';
        std::cout << "Execution next block:     " << next_block << '\n';
        std::cout << "Execution completed:      " << (completed ? "yes" : "no") << '\n';
        std::cout << "Recorded total (ms):      " << time_stats.total_time << '\n';
        std::cout << "Transfer time (ms):       " << time_stats.transfer_time << '\n';
        std::cout << "Compute time (ms):        " << time_stats.computation_time << '\n';
        std::cout << "Planning time (ms):       " << time_stats.planning_time << '\n';
        std::cout << "Control overhead (ms):    "
                  << std::max(0.0, execute_wall_ms - time_stats.transfer_time - time_stats.computation_time)
                  << '\n';
        std::cout << "State pool alloc (MB):    " << pool_mb << '\n';
        std::cout << "Peak RSS (MB):            " << peak_rss << '\n';
        std::cout << "Active states:            " << stats.active_states << '\n';
        std::cout << "HDD nodes:                " << stats.hdd_nodes << '\n';
        if (!completed) {
            std::cout << "Checkpoint saved:         " << options.checkpoint_file << '\n';
        }
        std::cout << "=================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "random_circuit_benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
