#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

#include "quantum_circuit.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

struct BenchmarkOptions {
    int num_qubits = 2;
    int num_qumodes = 3;
    int cutoff = 8;
    int max_states = 4096;
    int total_gates = 72;
    std::uint64_t seed = 20260315ULL;
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
        } else if (arg == "--total-gates") {
            options.total_gates = std::stoi(require_value("--total-gates"));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint64_t>(std::stoull(require_value("--seed")));
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

std::vector<BuiltGate> build_random_circuit(const BenchmarkOptions& options) {
    std::mt19937_64 rng(options.seed);
    std::vector<BuiltGate> gates;
    gates.reserve(static_cast<size_t>(options.total_gates));

    gates.push_back({Gates::Hadamard(0), "Hadamard"});
    gates.push_back({Gates::Hadamard(1), "Hadamard"});

    std::vector<GateBuilder> coverage_builders = {
        [&](std::mt19937_64& local_rng) { return make_random_qubit_rotation(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_phase_family(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_two_qubit_gate(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_single_mode_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_two_mode_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_diagonal_non_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_ladder_gate(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_controlled_gaussian(local_rng, options); },
        [&](std::mt19937_64& local_rng) { return make_random_two_mode_hybrid(local_rng, options); }
    };

    std::vector<BuiltGate> randomized_tail;
    randomized_tail.reserve(static_cast<size_t>(std::max(0, options.total_gates - 2)));

    for (const auto& builder : coverage_builders) {
        randomized_tail.push_back(builder(rng));
    }

    while (static_cast<int>(gates.size() + randomized_tail.size()) < options.total_gates) {
        const int family = random_index(rng, static_cast<int>(coverage_builders.size()));
        randomized_tail.push_back(coverage_builders[family](rng));
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
        const BenchmarkOptions options = parse_args(argc, argv);
        const std::vector<BuiltGate> gates = build_random_circuit(options);

        std::cout << "=================================================\n";
        std::cout << " Random Circuit Benchmark\n";
        std::cout << "=================================================\n";
        std::cout << "seed=" << options.seed
                  << ", qubits=" << options.num_qubits
                  << ", qumodes=" << options.num_qumodes
                  << ", cutoff=" << options.cutoff
                  << ", max_states=" << options.max_states
                  << ", total_gates=" << options.total_gates << '\n';
        print_gate_histogram(gates);
        print_gate_preview(gates);
        std::cout << "=================================================\n";

        QuantumCircuit circuit(
            options.num_qubits,
            options.num_qumodes,
            options.cutoff,
            options.max_states);

        for (const BuiltGate& built_gate : gates) {
            circuit.add_gate(built_gate.gate);
        }

        const auto wall_start = std::chrono::high_resolution_clock::now();

        const auto build_start = std::chrono::high_resolution_clock::now();
        circuit.build();
        const auto build_end = std::chrono::high_resolution_clock::now();

        const auto execute_start = std::chrono::high_resolution_clock::now();
        circuit.execute();
        const auto execute_end = std::chrono::high_resolution_clock::now();

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
        std::cout << "=================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "random_circuit_benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
