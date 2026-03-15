#pragma once

#include "gaussian_state.h"
#include "symplectic_math.h"
#include <complex>
#include <vector>
#include <memory>

/**
 * GaussianCircuit - Orchestrates symbolic SET (Symbolic Execution Track)
 */
enum class ExecutionTrack {
    Symbolic,
    Tensor
};

struct ExecutionDecision {
    ExecutionTrack track = ExecutionTrack::Symbolic;
    int active_modes = 0;
    double non_gaussianity = 0.0;
    bool requires_projection = false;
};

class GaussianCircuit {
public:
    GaussianCircuit(int num_qumodes, int capacity);
    ~GaussianCircuit();

    void add_gate(const SymplecticGate& gate);
    void execute();
    ExecutionDecision execute_with_ede(int fock_cutoff, double non_gaussianity = 0.0);

    void set_symbolic_mode_limit(int limit) { symbolic_mode_limit_ = limit; }
    void set_non_gaussian_threshold(double threshold) { non_gaussian_threshold_ = threshold; }
    ExecutionDecision decide_execution_track(double non_gaussianity = 0.0) const;

    GaussianStatePool& get_pool() { return *pool_; }
    int get_root_state_id() const { return root_state_id_; }
    ExecutionTrack get_current_track() const { return current_track_; }
    const std::vector<std::complex<double>>& get_projected_fock_state() const {
        return projected_fock_state_;
    }

    std::vector<std::complex<double>> project_root_state_to_fock(int cutoff) const;

private:
    enum class RecordedGateType {
        NoOp,
        PhaseRotation,
        Displacement,
        Squeezing,
        BeamSplitter,
        Unsupported
    };

    struct RecordedGate {
        RecordedGateType type = RecordedGateType::NoOp;
        std::vector<int> target_qumodes;
        std::vector<double> real_params;
        std::vector<std::complex<double>> complex_params;
    };

    void execute_symbolic_sequence();
    int estimate_active_modes() const;
    RecordedGate record_gate(const SymplecticGate& gate) const;
    std::vector<std::complex<double>> project_executed_gaussian_sequence_to_fock(int cutoff) const;
    std::vector<std::complex<double>> project_via_bloch_messiah_to_fock(
        const std::vector<double>& d,
        const std::vector<double>& sigma,
        int cutoff) const;
    std::vector<std::complex<double>> project_single_mode_pure_gaussian_to_fock(
        const std::vector<double>& d,
        const std::vector<double>& sigma,
        int cutoff) const;

    int num_qumodes_;
    std::unique_ptr<GaussianStatePool> pool_;
    int root_state_id_;
    std::vector<SymplecticGate> gate_sequence_;
    int symbolic_mode_limit_;
    double non_gaussian_threshold_;
    ExecutionTrack current_track_;
    std::vector<std::complex<double>> projected_fock_state_;
    std::vector<RecordedGate> recorded_gate_history_;
    size_t executed_gate_count_;
};
