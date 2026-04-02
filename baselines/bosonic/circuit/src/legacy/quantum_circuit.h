#ifndef QUANTUM_CIRCUIT_GPU_H
#define QUANTUM_CIRCUIT_GPU_H

#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <memory>
#include "../../gpu/operators.cuh"

namespace gpu {

// 电路统计信息结构
struct CircuitStats {
    int num_gates;
    int active_states;
};

// 时间统计信息结构
struct TimeStats {
    double total_time;
    double transfer_time;
    double computation_time;
};

// 门操作基类
class Gate {
public:
    virtual ~Gate() = default;
    virtual CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const = 0;
    virtual std::string name() const = 0;
};

// 具体门操作实现
class HadamardGate : public Gate {
private:
    int qubit;
public:
    HadamardGate(int q) : qubit(q) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "Hadamard"; }
};

class PhaseGateS : public Gate {
private:
    int qubit;
public:
    PhaseGateS(int q) : qubit(q) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "PhaseGateS"; }
};

class ConditionalDisplacementGate : public Gate {
private:
    int qubit;
    int qumode;
    Complex displacement_param;
public:
    ConditionalDisplacementGate(int q, int qm, Complex param) 
        : qubit(q), qumode(qm), displacement_param(param) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "ConditionalDisplacement"; }
};

class SqueezingGate : public Gate {
private:
    int qumode;
    Complex squeezing_param;
public:
    SqueezingGate(int qm, Complex param) : qumode(qm), squeezing_param(param) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "Squeezing"; }
};

class PhaseRotationGate : public Gate {
private:
    int qumode;
    double angle;
public:
    PhaseRotationGate(int qm, double a) : qumode(qm), angle(a) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "PhaseRotation"; }
};

class RotationZGate : public Gate {
private:
    int qubit;
    double angle;
public:
    RotationZGate(int q, double a) : qubit(q), angle(a) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "RotationZ"; }
};

class JaynesCummingsGate : public Gate {
private:
    int qubit;
    int qumode;
    double angle;
public:
    JaynesCummingsGate(int q, int qm, double a) : qubit(q), qumode(qm), angle(a) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "JaynesCummings"; }
};

class BeamSplitterGate : public Gate {
private:
    int qumode1;
    int qumode2;
    double angle;
public:
    BeamSplitterGate(int qm1, int qm2, double a) : qumode1(qm1), qumode2(qm2), angle(a) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "BeamSplitter"; }
};

class DisplacementGate : public Gate {
private:
    int qumode;
    Complex displacement_param;
public:
    DisplacementGate(int qm, Complex param) : qumode(qm), displacement_param(param) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "Displacement"; }
};

class PauliZGate : public Gate {
private:
    int qubit;
public:
    PauliZGate(int q) : qubit(q) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "PauliZ"; }
};

class PauliXGate : public Gate {
private:
    int qubit;
public:
    PauliXGate(int q) : qubit(q) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "PauliX"; }
};

class RotationXGate : public Gate {
private:
    int qubit;
    double angle;
public:
    RotationXGate(int q, double a) : qubit(q), angle(a) {}
    CUDASparseMatrix apply(const CUDASparseMatrix& state, int cutoff) const override;
    std::string name() const override { return "RotationX"; }
};

// 门操作工厂类
class Gates {
public:
    static std::unique_ptr<Gate> Hadamard(int qubit) {
        return std::make_unique<::gpu::HadamardGate>(qubit);
    }
    
    static std::unique_ptr<Gate> PhaseGateS(int qubit) {
        return std::make_unique<::gpu::PhaseGateS>(qubit);
    }
    
    static std::unique_ptr<Gate> ConditionalDisplacement(int qubit, int qumode, Complex displacement_param) {
        return std::make_unique<::gpu::ConditionalDisplacementGate>(qubit, qumode, displacement_param);
    }
    
    static std::unique_ptr<Gate> Squeezing(int qumode, Complex squeezing_param) {
        return std::make_unique<::gpu::SqueezingGate>(qumode, squeezing_param);
    }
    
    static std::unique_ptr<Gate> PhaseRotation(int qumode, double angle) {
        return std::make_unique<::gpu::PhaseRotationGate>(qumode, angle);
    }
    
    static std::unique_ptr<Gate> RotationZ(int qubit, double angle) {
        return std::make_unique<::gpu::RotationZGate>(qubit, angle);
    }
    
    static std::unique_ptr<Gate> JaynesCummings(int qubit, int qumode, double angle) {
        return std::make_unique<::gpu::JaynesCummingsGate>(qubit, qumode, angle);
    }
    
    static std::unique_ptr<Gate> BeamSplitter(int qumode1, int qumode2, double angle) {
        return std::make_unique<::gpu::BeamSplitterGate>(qumode1, qumode2, angle);
    }
    
    static std::unique_ptr<Gate> Displacement(int qumode, Complex displacement_param) {
        return std::make_unique<::gpu::DisplacementGate>(qumode, displacement_param);
    }
    
    static std::unique_ptr<Gate> PauliZ(int qubit) {
        return std::make_unique<::gpu::PauliZGate>(qubit);
    }
    
    static std::unique_ptr<Gate> PauliX(int qubit) {
        return std::make_unique<::gpu::PauliXGate>(qubit);
    }
    
    static std::unique_ptr<Gate> RotationX(int qubit, double angle) {
        return std::make_unique<::gpu::RotationXGate>(qubit, angle);
    }
};

// 量子电路类
class QuantumCircuit {
private:
    int num_qubits;
    int num_qumodes;
    int cutoff;
    int max_active_states;
    std::vector<std::unique_ptr<Gate>> gates;
    CUDASparseMatrix state;
    CVOperatorsGPU cv_ops;
    double start_time;
    double end_time;
    double transfer_time;
    double computation_time;
    
public:
    QuantumCircuit(int n_qubits, int n_qumodes, int c, int max_states);
    
    void add_gate(std::unique_ptr<Gate> gate);
    
    void build();
    void execute();
    
    CircuitStats get_stats() const;
    TimeStats get_time_stats() const;
    
    const CUDASparseMatrix& get_state() const;
};

} // namespace gpu

#endif // QUANTUM_CIRCUIT_GPU_H
