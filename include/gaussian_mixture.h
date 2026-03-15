#pragma once

#include "symplectic_math.h"

#include <complex>
#include <vector>

struct GaussianMixtureBranch {
    std::complex<double> weight;
    SymplecticGate gaussian_gate;
    std::vector<int> target_qumodes;
    std::vector<double> phase_rotation_thetas;
};

struct GaussianMixtureApproximation {
    std::vector<GaussianMixtureBranch> branches;
    std::vector<std::complex<double>> target_diagonal;
    std::vector<std::complex<double>> approximated_diagonal;
    double l2_diagonal_error = 0.0;
    double max_diagonal_error = 0.0;
    double conservative_fidelity_lower_bound = 0.0;
};

class GaussianMixtureDecomposer {
public:
    static GaussianMixtureApproximation approximate_single_mode_diagonal(
        int total_qumodes,
        int target_qumode,
        const std::vector<std::complex<double>>& diagonal,
        int max_branches);

    static GaussianMixtureApproximation approximate_kerr_gate(
        int total_qumodes,
        int target_qumode,
        double chi,
        int cutoff,
        int max_branches);

    static GaussianMixtureApproximation approximate_snap_gate(
        int total_qumodes,
        int target_qumode,
        double theta,
        int target_fock_state,
        int cutoff,
        int max_branches);

    static GaussianMixtureApproximation approximate_multisnap_gate(
        int total_qumodes,
        int target_qumode,
        const std::vector<double>& phase_map,
        int cutoff,
        int max_branches);

    static GaussianMixtureApproximation approximate_conditional_parity_gate(
        int total_qumodes,
        int target_qumode,
        double parity,
        int cutoff,
        int max_branches);

    static GaussianMixtureApproximation approximate_two_mode_diagonal(
        int total_qumodes,
        int first_target_qumode,
        int second_target_qumode,
        const std::vector<std::complex<double>>& diagonal,
        int cutoff,
        int max_branches);

    static GaussianMixtureApproximation approximate_cross_kerr_gate(
        int total_qumodes,
        int first_target_qumode,
        int second_target_qumode,
        double kappa,
        int cutoff,
        int max_branches);

    static GaussianMixtureApproximation approximate_cz_gate(
        int total_qumodes,
        int first_target_qumode,
        int second_target_qumode,
        double s,
        int cutoff,
        int max_branches);

    static std::vector<std::complex<double>> apply_to_fock_state(
        const std::vector<std::complex<double>>& input_state,
        int cutoff,
        int num_qumodes,
        const GaussianMixtureApproximation& approximation);
};
