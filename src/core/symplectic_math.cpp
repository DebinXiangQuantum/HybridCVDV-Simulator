#include "symplectic_math.h"

namespace SymplecticFactory {

SymplecticGate Rotation(double phi) {
    SymplecticGate gate(1);
    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);
    
    gate.S[0] = cos_phi;  gate.S[1] = sin_phi;
    gate.S[2] = -sin_phi; gate.S[3] = cos_phi;
    
    return gate;
}

SymplecticGate Squeezing(double r, double theta) {
    SymplecticGate gate(1);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double cosh_r = std::cosh(r);
    double sinh_r = std::sinh(r);
    
    gate.S[0] = cosh_r - sinh_r * cos_theta;
    gate.S[1] = -sinh_r * sin_theta;
    gate.S[2] = -sinh_r * sin_theta;
    gate.S[3] = cosh_r + sinh_r * cos_theta;
    
    return gate;
}

SymplecticGate Displacement(std::complex<double> alpha) {
    SymplecticGate gate(1);
    // S is identity (already initialized)
    gate.d[0] = std::sqrt(2.0) * alpha.real();
    gate.d[1] = std::sqrt(2.0) * alpha.imag();
    
    return gate;
}

SymplecticGate BeamSplitter(double theta, double phi, int M, int mode1, int mode2) {
    SymplecticGate gate(M);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);
    
    // Sub-matrix elements
    // Mode 1 mapping: 2*mode1, 2*mode1+1
    // Mode 2 mapping: 2*mode2, 2*mode2+1
    
    int r1 = 2 * mode1;
    int r2 = 2 * mode2;
    int dim = 2 * M;
    
    // Identity part for other modes is already handled by constructor
    
    // Clear identity for interacting modes
    gate.S[r1 * dim + r1] = 0; gate.S[r1 * dim + r1 + 1] = 0;
    gate.S[(r1 + 1) * dim + r1] = 0; gate.S[(r1 + 1) * dim + r1 + 1] = 0;
    gate.S[r2 * dim + r2] = 0; gate.S[r2 * dim + r2 + 1] = 0;
    gate.S[(r2 + 1) * dim + r2] = 0; gate.S[(r2 + 1) * dim + r2 + 1] = 0;

    // Beam splitter coupling
    // Top-left (Mode 1 self)
    gate.S[r1 * dim + r1] = cos_theta;
    gate.S[(r1 + 1) * dim + r1 + 1] = cos_theta;
    
    // Top-right (Mode 1-2 coupling)
    gate.S[r1 * dim + r2] = sin_theta * cos_phi;
    gate.S[r1 * dim + r2 + 1] = sin_theta * sin_phi;
    gate.S[(r1 + 1) * dim + r2] = -sin_theta * sin_phi;
    gate.S[(r1 + 1) * dim + r2 + 1] = sin_theta * cos_phi;
    
    // Bottom-left (Mode 2-1 coupling)
    gate.S[r2 * dim + r1] = -sin_theta * cos_phi;
    gate.S[r2 * dim + r1 + 1] = sin_theta * sin_phi;
    gate.S[(r2 + 1) * dim + r1] = -sin_theta * sin_phi;
    gate.S[(r2 + 1) * dim + r1 + 1] = -sin_theta * cos_phi;
    
    // Bottom-right (Mode 2 self)
    gate.S[r2 * dim + r2] = cos_theta;
    gate.S[(r2 + 1) * dim + r2 + 1] = cos_theta;
    
    return gate;
}

} // namespace SymplecticFactory
