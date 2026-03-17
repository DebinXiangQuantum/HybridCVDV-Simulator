#pragma once

#include <vector>
#include <cmath>
#include <complex>

/**
 * SymplecticGate - Represents a Gaussian operation in symplectic form
 */
struct SymplecticGate {
    std::vector<double> S; // (2M x 2M) Row-major
    std::vector<double> d; // (2M)
    int num_qumodes;

    SymplecticGate(int M) : num_qumodes(M) {
        S.assign(4 * M * M, 0.0);
        d.assign(2 * M, 0.0);
        // Initialize S as Identity
        for (int i = 0; i < 2 * M; ++i) S[i * 2 * M + i] = 1.0;
    }
};

namespace SymplecticFactory {
    // Single-mode Rotation
    SymplecticGate Rotation(double phi);
    
    // Single-mode Squeezing
    SymplecticGate Squeezing(double r, double theta);
    
    // Single-mode Displacement
    SymplecticGate Displacement(std::complex<double> alpha);
    
    // Two-mode Beam Splitter
    SymplecticGate BeamSplitter(double theta, double phi, int M, int mode1, int mode2);

    // Two-mode Squeezing
    SymplecticGate TwoModeSqueezing(std::complex<double> xi, int M, int mode1, int mode2);

    // SUM gate
    SymplecticGate SUM(double theta, double phi, int M, int mode1, int mode2);
}
