#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include "quantum_circuit.h"
#include "batch_scheduler.h"

using namespace Gates;

void run_profiling(int num_qubits, int num_qumodes, int cutoff, int max_states, int num_gates) {
    std::cout << "=================================================" << std::endl;
    std::cout << " Profiling Test" << std::endl;
    std::cout << " Qubits: " << num_qubits << ", Qumodes: " << num_qumodes 
              << ", Cutoff: " << cutoff << ", Max States: " << max_states << std::endl;
    std::cout << " Gates: " << num_gates << std::endl;
    std::cout << "=================================================" << std::endl;

    QuantumCircuit circuit(num_qubits, num_qumodes, cutoff, max_states);
    
    // Add varied mix of operations
    for (int i = 0; i < num_gates; ++i) {
        // GPU Qumode operations (Level 0, 1, 2, 3)
        circuit.add_gate(Displacement(0, std::complex<double>(0.1, 0.0)));
        circuit.add_gate(PhaseRotation(0, M_PI / 4.0));
        circuit.add_gate(Squeezing(1, std::complex<double>(0.05, 0.0)));
        
        if (num_qumodes > 1) {
            circuit.add_gate(BeamSplitter(0, 1, M_PI / 4.0, 0.0));
        }

        // Qubit operations
        circuit.add_gate(Hadamard(0));
        if (num_qubits > 1) {
            circuit.add_gate(CNOT(0, 1));
        }

        // Hybrid operations
        circuit.add_gate(ConditionalDisplacement(0, 0, std::complex<double>(0.1, 0.0)));
    }
    
    circuit.build();
    
    auto start = std::chrono::high_resolution_clock::now();
    circuit.execute();
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_wall_time = std::chrono::duration<double, std::milli>(end - start).count();
    auto time_stats = circuit.get_time_stats();
    
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Execution Time Breakdown:" << std::endl;
    std::cout << "Total Wall Time:    " << std::fixed << std::setprecision(3) << total_wall_time << " ms" << std::endl;
    std::cout << "Recorded Total:     " << std::fixed << std::setprecision(3) << time_stats.total_time << " ms" << std::endl;
    std::cout << "Transfer Time:      " << std::fixed << std::setprecision(3) << time_stats.transfer_time << " ms" << std::endl;
    std::cout << "Computation Time:   " << std::fixed << std::setprecision(3) << time_stats.computation_time << " ms" << std::endl;
    std::cout << "Planning Time:      " << std::fixed << std::setprecision(3) << time_stats.planning_time << " ms" << std::endl;

    double overhead = total_wall_time - time_stats.transfer_time - time_stats.computation_time;
    std::cout << "Control Overhead:   " << std::fixed << std::setprecision(3) << overhead << " ms" << std::endl;
    std::cout << "Hidden Planning:    "
              << std::fixed << std::setprecision(3)
              << std::max(0.0, time_stats.planning_time - overhead) << " ms" << std::endl;

    double transfer_pct = (time_stats.transfer_time / total_wall_time) * 100.0;
    double compute_pct = (time_stats.computation_time / total_wall_time) * 100.0;
    double overhead_pct = (overhead / total_wall_time) * 100.0;
    double planning_pct = (time_stats.planning_time / total_wall_time) * 100.0;

    std::cout << "\nRelative Ratios:" << std::endl;
    std::cout << "Transfer %:         " << std::fixed << std::setprecision(1) << transfer_pct << "%" << std::endl;
    std::cout << "Compute %:          " << std::fixed << std::setprecision(1) << compute_pct << "%" << std::endl;
    std::cout << "Planning %:         " << std::fixed << std::setprecision(1) << planning_pct << "%" << std::endl;
    std::cout << "Control Overhead %: " << std::fixed << std::setprecision(1) << overhead_pct << "%" << std::endl;
    std::cout << "=================================================\n" << std::endl;
}

void run_large_diagonal_block_profiling(int cutoff, int max_states, int repetitions) {
    std::cout << "=================================================" << std::endl;
    std::cout << " Large Diagonal Non-Gaussian Block Profiling" << std::endl;
    std::cout << " Raw Diagonal Gates: " << (2 * repetitions)
              << ", Cutoff: " << cutoff
              << ", Max States: " << max_states << std::endl;
    std::cout << "=================================================" << std::endl;

    QuantumCircuit circuit(1, 2, cutoff, max_states);
    circuit.add_gate(Displacement(0, std::complex<double>(0.18, -0.03)));

    for (int i = 0; i < repetitions; ++i) {
        circuit.add_gate(KerrGate(0, 1e-4));
        circuit.add_gate(ConditionalParity(1, 1.0));
    }

    circuit.add_gate(PhaseRotation(0, 0.1));
    circuit.add_gate(CreationOperator(1));
    circuit.build();

    auto start = std::chrono::high_resolution_clock::now();
    circuit.execute();
    auto end = std::chrono::high_resolution_clock::now();

    const double total_wall_time = std::chrono::duration<double, std::milli>(end - start).count();
    const auto time_stats = circuit.get_time_stats();

    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Total Wall Time:    " << std::fixed << std::setprecision(3) << total_wall_time << " ms" << std::endl;
    std::cout << "Transfer Time:      " << std::fixed << std::setprecision(3) << time_stats.transfer_time << " ms" << std::endl;
    std::cout << "Computation Time:   " << std::fixed << std::setprecision(3) << time_stats.computation_time << " ms" << std::endl;
    std::cout << "Planning Time:      " << std::fixed << std::setprecision(3) << time_stats.planning_time << " ms" << std::endl;
    std::cout << "=================================================\n" << std::endl;
}

int main(int argc, char** argv) {
    try {
        // Small system, short circuit
        run_profiling(2, 2, 16, 64, 10);
        
        // Larger system, deeper circuit
        run_profiling(3, 3, 16, 128, 50);

        // Even deeper circuit
        run_profiling(4, 4, 16, 256, 100);

        // A representative large non-Gaussian diagonal window after canonicalization.
        run_large_diagonal_block_profiling(16, 256, 256);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
