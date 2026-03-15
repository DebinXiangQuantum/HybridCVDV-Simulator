#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cv_state_pool.h"
#include "reference_gates.h"

extern void apply_phase_rotation(CVStatePool* pool, const int* targets, int batch_size, double theta);
extern void apply_kerr_gate(CVStatePool* pool, const int* targets, int batch_size, double chi);
extern void apply_creation_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_annihilation_operator(CVStatePool* pool, const int* targets, int batch_size);
extern void apply_displacement_gate(CVStatePool* pool, const int* targets, int batch_size, cuDoubleComplex alpha);

namespace {

struct ErrorSummary {
    double l2_error = 0.0;
    double max_abs_error = 0.0;
    double fidelity = 0.0;
};

Reference::Vector normalize(Reference::Vector state) {
    double norm = 0.0;
    for (const auto& amplitude : state) {
        norm += std::norm(amplitude);
    }
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (auto& amplitude : state) {
            amplitude /= norm;
        }
    }
    return state;
}

std::vector<cuDoubleComplex> to_cuda_vector(const Reference::Vector& state) {
    std::vector<cuDoubleComplex> result;
    result.reserve(state.size());
    for (const auto& amplitude : state) {
        result.push_back(make_cuDoubleComplex(amplitude.real(), amplitude.imag()));
    }
    return result;
}

Reference::Vector to_reference_vector(const std::vector<cuDoubleComplex>& state) {
    Reference::Vector result;
    result.reserve(state.size());
    for (const auto& amplitude : state) {
        result.emplace_back(cuCreal(amplitude), cuCimag(amplitude));
    }
    return result;
}

ErrorSummary compute_error_summary(const Reference::Vector& reference,
                                   const Reference::Vector& gpu_result) {
    ErrorSummary summary;
    if (reference.size() != gpu_result.size()) {
        summary.l2_error = std::numeric_limits<double>::infinity();
        summary.max_abs_error = std::numeric_limits<double>::infinity();
        return summary;
    }

    for (size_t i = 0; i < reference.size(); ++i) {
        const double abs_error = std::abs(reference[i] - gpu_result[i]);
        summary.l2_error += abs_error * abs_error;
        summary.max_abs_error = std::max(summary.max_abs_error, abs_error);
    }
    summary.l2_error = std::sqrt(summary.l2_error);
    summary.fidelity = Reference::fidelity(reference, gpu_result);
    return summary;
}

template <typename Invoker>
ErrorSummary run_case(const std::string& name,
                      const Reference::Vector& input_state,
                      const Reference::Vector& reference_state,
                      Invoker&& invoker) {
    constexpr int kDim = 16;
    CVStatePool pool(kDim, 4, 1);
    const int state_id = pool.allocate_state();
    if (state_id < 0) {
        throw std::runtime_error("无法分配GPU状态");
    }

    const auto cuda_input = to_cuda_vector(input_state);
    pool.upload_state(state_id, cuda_input);

    int* d_target_indices = nullptr;
    cudaMalloc(&d_target_indices, sizeof(int));
    cudaMemcpy(d_target_indices, &state_id, sizeof(int), cudaMemcpyHostToDevice);

    invoker(&pool, d_target_indices);
    cudaDeviceSynchronize();

    std::vector<cuDoubleComplex> gpu_state;
    pool.download_state(state_id, gpu_state);
    cudaFree(d_target_indices);

    const auto gpu_reference = to_reference_vector(gpu_state);
    const auto summary = compute_error_summary(reference_state, gpu_reference);

    std::cout << std::fixed << std::setprecision(12)
              << name
              << " | l2_error=" << summary.l2_error
              << " | max_abs_error=" << summary.max_abs_error
              << " | fidelity=" << summary.fidelity
              << '\n';
    return summary;
}

}  // namespace

int main() {
    constexpr int kDim = 16;
    const double inv_norm = 1.0 / std::sqrt(1.0 + 0.5 * 0.5 + 0.25 * 0.25 + 0.1 * 0.1);

    Reference::Vector superposition(kDim, Reference::Complex(0.0, 0.0));
    superposition[0] = Reference::Complex(1.0 * inv_norm, 0.0);
    superposition[1] = Reference::Complex(0.5 * inv_norm, 0.0);
    superposition[2] = Reference::Complex(0.0, 0.25 * inv_norm);
    superposition[3] = Reference::Complex(0.1 * inv_norm, -0.05 * inv_norm);
    superposition = normalize(superposition);

    Reference::Vector vacuum(kDim, Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);

    bool ok = true;

    const auto phase_ref = Reference::DiagonalGates::apply_phase_rotation(superposition, M_PI / 5.0);
    const auto phase_summary = run_case(
        "phase_rotation", superposition, phase_ref,
        [](CVStatePool* pool, const int* targets) {
            apply_phase_rotation(pool, targets, 1, M_PI / 5.0);
        });
    ok = ok && phase_summary.l2_error < 1e-12;

    const auto kerr_ref = Reference::DiagonalGates::apply_kerr_gate(superposition, 0.07);
    const auto kerr_summary = run_case(
        "kerr_gate", superposition, kerr_ref,
        [](CVStatePool* pool, const int* targets) {
            apply_kerr_gate(pool, targets, 1, 0.07);
        });
    ok = ok && kerr_summary.l2_error < 1e-12;

    const auto creation_ref = Reference::LadderGates::apply_creation_operator(superposition);
    const auto creation_summary = run_case(
        "creation_operator", superposition, creation_ref,
        [](CVStatePool* pool, const int* targets) {
            apply_creation_operator(pool, targets, 1);
        });
    ok = ok && creation_summary.l2_error < 1e-12;

    const auto annihilation_ref = Reference::LadderGates::apply_annihilation_operator(superposition);
    const auto annihilation_summary = run_case(
        "annihilation_operator", superposition, annihilation_ref,
        [](CVStatePool* pool, const int* targets) {
            apply_annihilation_operator(pool, targets, 1);
        });
    ok = ok && annihilation_summary.l2_error < 1e-12;

    const Reference::Complex alpha(0.15, -0.08);
    const auto displacement_ref = Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha);
    const auto displacement_summary = run_case(
        "displacement_gate", vacuum, displacement_ref,
        [alpha](CVStatePool* pool, const int* targets) {
            apply_displacement_gate(pool, targets, 1, make_cuDoubleComplex(alpha.real(), alpha.imag()));
        });
    ok = ok && displacement_summary.l2_error < 1e-6;

    if (!ok) {
        std::cerr << "GPU precision regression detected" << std::endl;
        return 1;
    }

    return 0;
}
