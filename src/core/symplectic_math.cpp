#include "symplectic_math.h"

namespace SymplecticFactory {

namespace {

void clear_two_mode_submatrix(SymplecticGate* gate, int mode1, int mode2) {
    const int dim = 2 * gate->num_qumodes;
    const int row1 = 2 * mode1;
    const int row2 = 2 * mode2;

    for (int local_row = 0; local_row < 2; ++local_row) {
        for (int local_col = 0; local_col < 2; ++local_col) {
            gate->S[(row1 + local_row) * dim + row1 + local_col] = 0.0;
            gate->S[(row1 + local_row) * dim + row2 + local_col] = 0.0;
            gate->S[(row2 + local_row) * dim + row1 + local_col] = 0.0;
            gate->S[(row2 + local_row) * dim + row2 + local_col] = 0.0;
        }
    }
}

void validate_two_mode_embedding(int M, int mode1, int mode2) {
    if (mode1 < 0 || mode1 >= M || mode2 < 0 || mode2 >= M) {
        throw std::out_of_range("two-mode symplectic gate target out of range");
    }
    if (mode1 == mode2) {
        throw std::invalid_argument("two-mode symplectic gate requires distinct modes");
    }
}

}  // namespace

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
    validate_two_mode_embedding(M, mode1, mode2);
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

SymplecticGate TwoModeSqueezing(std::complex<double> xi, int M, int mode1, int mode2) {
    validate_two_mode_embedding(M, mode1, mode2);

    SymplecticGate gate(M);
    clear_two_mode_submatrix(&gate, mode1, mode2);

    const int dim = 2 * M;
    const int row1 = 2 * mode1;
    const int row2 = 2 * mode2;

    // The exact Fock reference uses exp[(xi* a^\dagger b^\dagger - xi a b) / 2],
    // so the quadrature squeezing strength is |xi| / 2.
    const double squeeze = 0.5 * std::abs(xi);
    const double phase = std::arg(xi);
    const double cosh_s = std::cosh(squeeze);
    const double sinh_s = std::sinh(squeeze);
    const double cos_phase = std::cos(phase);
    const double sin_phase = std::sin(phase);

    // x1', p1'
    gate.S[row1 * dim + row1] = cosh_s;
    gate.S[(row1 + 1) * dim + row1 + 1] = cosh_s;
    gate.S[row1 * dim + row2] = sinh_s * cos_phase;
    gate.S[row1 * dim + row2 + 1] = -sinh_s * sin_phase;
    gate.S[(row1 + 1) * dim + row2] = -sinh_s * sin_phase;
    gate.S[(row1 + 1) * dim + row2 + 1] = -sinh_s * cos_phase;

    // x2', p2'
    gate.S[row2 * dim + row2] = cosh_s;
    gate.S[(row2 + 1) * dim + row2 + 1] = cosh_s;
    gate.S[row2 * dim + row1] = sinh_s * cos_phase;
    gate.S[row2 * dim + row1 + 1] = -sinh_s * sin_phase;
    gate.S[(row2 + 1) * dim + row1] = -sinh_s * sin_phase;
    gate.S[(row2 + 1) * dim + row1 + 1] = -sinh_s * cos_phase;

    return gate;
}

SymplecticGate SUM(double theta, double phi, int M, int mode1, int mode2) {
    validate_two_mode_embedding(M, mode1, mode2);
    if (std::abs(phi) > 1e-14) {
        throw std::invalid_argument("SUM symplectic gate currently supports phi = 0 only");
    }

    SymplecticGate gate(M);
    clear_two_mode_submatrix(&gate, mode1, mode2);

    const int dim = 2 * M;
    const int row1 = 2 * mode1;
    const int row2 = 2 * mode2;

    // U = exp[-i theta x_1 p_2], so x2 -> x2 + theta x1 and p1 -> p1 - theta p2.
    gate.S[row1 * dim + row1] = 1.0;
    gate.S[(row1 + 1) * dim + row1 + 1] = 1.0;
    gate.S[row2 * dim + row2] = 1.0;
    gate.S[(row2 + 1) * dim + row2 + 1] = 1.0;

    gate.S[(row1 + 1) * dim + row2 + 1] = -theta;
    gate.S[row2 * dim + row1] = theta;

    return gate;
}

} // namespace SymplecticFactory
