#ifndef GATES_H
#define GATES_H

#include "core/quantum_state.h"
#include <complex>
#include <memory>
#include <string>

namespace gpu {

// 门操作接口
class Gate {
public:
    virtual ~Gate() = default;
    virtual void apply(QuantumState& state) const = 0;
    virtual std::string name() const = 0;
};

// Hadamard门
class HadamardGate : public Gate {
    int qubit_;
public:
    HadamardGate(int qubit) : qubit_(qubit) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "Hadamard"; }
};

// Phase S门
class PhaseSGate : public Gate {
    int qubit_;
public:
    PhaseSGate(int qubit) : qubit_(qubit) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "PhaseS"; }
};

// Pauli X门
class PauliXGate : public Gate {
    int qubit_;
public:
    PauliXGate(int qubit) : qubit_(qubit) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "PauliX"; }
};

// Pauli Z门
class PauliZGate : public Gate {
    int qubit_;
public:
    PauliZGate(int qubit) : qubit_(qubit) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "PauliZ"; }
};

// Rotation X门
class RotationXGate : public Gate {
    int qubit_;
    double angle_;
public:
    RotationXGate(int qubit, double angle) : qubit_(qubit), angle_(angle) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "RotationX"; }
};

// Rotation Z门
class RotationZGate : public Gate {
    int qubit_;
    double angle_;
public:
    RotationZGate(int qubit, double angle) : qubit_(qubit), angle_(angle) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "RotationZ"; }
};

// 相位旋转门 (CV)
class PhaseRotationGate : public Gate {
    int qumode_;
    double angle_;
public:
    PhaseRotationGate(int qumode, double angle) : qumode_(qumode), angle_(angle) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "PhaseRotation"; }
};

// 位移门 (CV)
class DisplacementGate : public Gate {
    int qumode_;
    std::complex<double> alpha_;
public:
    DisplacementGate(int qumode, std::complex<double> alpha) : qumode_(qumode), alpha_(alpha) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "Displacement"; }
};

// 压缩门 (CV)
class SqueezingGate : public Gate {
    int qumode_;
    std::complex<double> r_;
public:
    SqueezingGate(int qumode, std::complex<double> r) : qumode_(qumode), r_(r) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "Squeezing"; }
};

// 条件位移门 (CV-DV混合)
class ConditionalDisplacementGate : public Gate {
    int qubit_;
    int qumode_;
    std::complex<double> alpha_;
public:
    ConditionalDisplacementGate(int qubit, int qumode, std::complex<double> alpha) 
        : qubit_(qubit), qumode_(qumode), alpha_(alpha) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "ConditionalDisplacement"; }
};

// Jaynes-Cummings门 (CV-DV混合)
class JaynesCummingsGate : public Gate {
    int qubit_;
    int qumode_;
    double angle_;
public:
    JaynesCummingsGate(int qubit, int qumode, double angle) 
        : qubit_(qubit), qumode_(qumode), angle_(angle) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "JaynesCummings"; }
};

// 光束分裂器门 (CV)
class BeamSplitterGate : public Gate {
    int qumode1_;
    int qumode2_;
    double angle_;
public:
    BeamSplitterGate(int qumode1, int qumode2, double angle) 
        : qumode1_(qumode1), qumode2_(qumode2), angle_(angle) {}
    void apply(QuantumState& state) const override;
    std::string name() const override { return "BeamSplitter"; }
};

// 门工厂
class Gates {
public:
    static std::unique_ptr<Gate> Hadamard(int qubit) {
        return std::make_unique<HadamardGate>(qubit);
    }
    static std::unique_ptr<Gate> PhaseS(int qubit) {
        return std::make_unique<PhaseSGate>(qubit);
    }
    static std::unique_ptr<Gate> PauliX(int qubit) {
        return std::make_unique<PauliXGate>(qubit);
    }
    static std::unique_ptr<Gate> PauliZ(int qubit) {
        return std::make_unique<PauliZGate>(qubit);
    }
    static std::unique_ptr<Gate> RotationX(int qubit, double angle) {
        return std::make_unique<RotationXGate>(qubit, angle);
    }
    static std::unique_ptr<Gate> RotationZ(int qubit, double angle) {
        return std::make_unique<RotationZGate>(qubit, angle);
    }
    static std::unique_ptr<Gate> PhaseRotation(int qumode, double angle) {
        return std::make_unique<PhaseRotationGate>(qumode, angle);
    }
    static std::unique_ptr<Gate> Displacement(int qumode, std::complex<double> alpha) {
        return std::make_unique<DisplacementGate>(qumode, alpha);
    }
    static std::unique_ptr<Gate> Squeezing(int qumode, std::complex<double> r) {
        return std::make_unique<SqueezingGate>(qumode, r);
    }
    static std::unique_ptr<Gate> ConditionalDisplacement(int qubit, int qumode, std::complex<double> alpha) {
        return std::make_unique<ConditionalDisplacementGate>(qubit, qumode, alpha);
    }
    static std::unique_ptr<Gate> JaynesCummings(int qubit, int qumode, double angle) {
        return std::make_unique<JaynesCummingsGate>(qubit, qumode, angle);
    }
    static std::unique_ptr<Gate> BeamSplitter(int qumode1, int qumode2, double angle) {
        return std::make_unique<BeamSplitterGate>(qumode1, qumode2, angle);
    }
};

} // namespace gpu

#endif // GATES_H