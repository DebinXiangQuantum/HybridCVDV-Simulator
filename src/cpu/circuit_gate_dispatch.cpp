// circuit_gate_dispatch.cpp — Gate level dispatch (levels 0-4) and qubit gates

#include "quantum_circuit.h"
#include "circuit_internal.h"
#include "gaussian_kernels.h"
#include "gaussian_state.h"
#include "squeezing_gate_gpu.h"
#include "squeezing_matrix.h"
#include "two_mode_gates.h"
#include "dense_fock_baseline.h"

using namespace circuit_internal;

namespace {

using SingleQubitMatrix = std::array<std::complex<double>, 4>;

// Build a D×D dense diagonal matrix on host for a Level-0 gate (ablation baseline).
std::vector<cuDoubleComplex> build_dense_level0_matrix(
    GateType type, int D, double param, int target_fock = -1,
    const std::vector<std::complex<double>>* multi_params = nullptr) {
    std::vector<cuDoubleComplex> mat(static_cast<size_t>(D) * D,
                                     make_cuDoubleComplex(0.0, 0.0));
    for (int n = 0; n < D; ++n) {
        double phase = 0.0;
        switch (type) {
            case GateType::PHASE_ROTATION:
                phase = param * n;
                break;
            case GateType::KERR_GATE:
                phase = param * static_cast<double>(n) * static_cast<double>(n);
                break;
            case GateType::CONDITIONAL_PARITY:
                phase = param * (n % 2 == 0 ? 0.0 : M_PI);
                break;
            case GateType::SNAP_GATE:
                phase = (n == target_fock) ? param : 0.0;
                break;
            case GateType::MULTI_SNAP_GATE:
                phase = (multi_params && n < static_cast<int>(multi_params->size()))
                            ? (*multi_params)[n].real()
                            : 0.0;
                break;
            default:
                break;
        }
        mat[static_cast<size_t>(n) * D + n] =
            make_cuDoubleComplex(std::cos(phase), std::sin(phase));
    }
    return mat;
}

// Build a D×D dense matrix for creation operator â† (ablation baseline).
std::vector<cuDoubleComplex> build_dense_creation_matrix(int D) {
    std::vector<cuDoubleComplex> mat(static_cast<size_t>(D) * D,
                                     make_cuDoubleComplex(0.0, 0.0));
    for (int n = 1; n < D; ++n) {
        // â†|n-1⟩ = √n |n⟩  →  row n, col n-1
        mat[static_cast<size_t>(n) * D + (n - 1)] =
            make_cuDoubleComplex(std::sqrt(static_cast<double>(n)), 0.0);
    }
    return mat;
}

// Build a D×D dense matrix for annihilation operator â (ablation baseline).
std::vector<cuDoubleComplex> build_dense_annihilation_matrix(int D) {
    std::vector<cuDoubleComplex> mat(static_cast<size_t>(D) * D,
                                     make_cuDoubleComplex(0.0, 0.0));
    for (int n = 0; n + 1 < D; ++n) {
        // â|n+1⟩ = √(n+1) |n⟩  →  row n, col n+1
        mat[static_cast<size_t>(n) * D + (n + 1)] =
            make_cuDoubleComplex(std::sqrt(static_cast<double>(n + 1)), 0.0);
    }
    return mat;
}

// Upload a host matrix to device scratch_aux and return device pointer.
// Uses scratch_aux to avoid conflict with scratch_temp used by kernel temp buffers.
cuDoubleComplex* upload_dense_matrix_to_gpu(
    CVStatePool& pool, const std::vector<cuDoubleComplex>& host_mat) {
    const size_t bytes = host_mat.size() * sizeof(cuDoubleComplex);
    cuDoubleComplex* d_mat = static_cast<cuDoubleComplex*>(
        pool.scratch_aux.ensure(bytes));
    CHECK_CUDA(cudaMemcpy(d_mat, host_mat.data(), bytes, cudaMemcpyHostToDevice));
    return d_mat;
}

// Build a D×D dense displacement matrix <n|D(α)|m> on host.
std::vector<cuDoubleComplex> build_dense_displacement_matrix(int D, std::complex<double> alpha) {
    std::vector<cuDoubleComplex> mat(static_cast<size_t>(D) * D,
                                     make_cuDoubleComplex(0.0, 0.0));
    const double ar = alpha.real(), ai = alpha.imag();
    const double norm_sq = ar * ar + ai * ai;
    const double exp_factor = std::exp(-norm_sq / 2.0);

    for (int n = 0; n < D; ++n) {
        for (int m = 0; m < D; ++m) {
            int mn = std::min(n, m), mx = std::max(n, m);
            int diff = mx - mn;

            // sqrt(min!/max!) = prod_{k=mn+1}^{mx} 1/sqrt(k)
            double sqrt_fact_ratio = 1.0;
            for (int k = mn + 1; k <= mx; ++k)
                sqrt_fact_ratio /= std::sqrt(static_cast<double>(k));

            // alpha^{n-m} or (-conj(alpha))^{m-n}
            std::complex<double> power(1.0, 0.0);
            std::complex<double> base = (n >= m) ? alpha : std::complex<double>(-ar, ai);
            for (int k = 0; k < diff; ++k) power *= base;

            // Associated Laguerre L_{mn}^{|n-m|}(|alpha|^2)
            double laguerre = 0.0, x_pow = 1.0, fact_j = 1.0;
            for (int j = 0; j <= mn; ++j) {
                if (j > 0) { x_pow *= norm_sq; fact_j *= j; }
                double binom = 1.0;
                for (int k = 0; k < mn - j; ++k)
                    binom *= static_cast<double>(mx - k) / static_cast<double>(k + 1);
                laguerre += binom * ((j % 2 == 0) ? 1.0 : -1.0) * x_pow / fact_j;
            }

            double scale = exp_factor * sqrt_fact_ratio * laguerre;
            auto val = std::complex<double>(scale, 0.0) * power;
            mat[static_cast<size_t>(n) * D + m] =
                make_cuDoubleComplex(val.real(), val.imag());
        }
    }
    return mat;
}

// Build a D×D dense squeezing matrix on host using the existing trusted implementation.
std::vector<cuDoubleComplex> build_dense_squeezing_cucomplex(int D, double r, double theta) {
    auto cmat = generate_squeezing_matrix(r, theta, D);
    std::vector<cuDoubleComplex> mat(static_cast<size_t>(D) * D);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = make_cuDoubleComplex(cmat[i].real(), cmat[i].imag());
    }
    return mat;
}

// Build ELL format from a dense D×D matrix on CPU, upload to GPU scratch buffers.
// Returns (d_ell_val, d_ell_col, actual_bandwidth).
struct ELLOnGPU {
    cuDoubleComplex* d_val;
    int*             d_col;
    int              bandwidth;
};

ELLOnGPU build_and_upload_ell(
    CVStatePool& pool,
    const std::vector<cuDoubleComplex>& dense_matrix,
    int D, double tolerance = 1e-12)
{
    // Determine max non-zeros per row
    int max_nnz = 0;
    for (int row = 0; row < D; ++row) {
        int nnz = 0;
        for (int col = 0; col < D; ++col) {
            auto& v = dense_matrix[static_cast<size_t>(row) * D + col];
            if (std::sqrt(v.x * v.x + v.y * v.y) > tolerance) ++nnz;
        }
        max_nnz = std::max(max_nnz, nnz);
    }
    if (max_nnz == 0) max_nnz = 1;

    // Build ELL arrays on host
    const size_t total = static_cast<size_t>(D) * max_nnz;
    std::vector<cuDoubleComplex> h_val(total, make_cuDoubleComplex(0.0, 0.0));
    std::vector<int>             h_col(total, -1);

    for (int row = 0; row < D; ++row) {
        int k = 0;
        for (int col = 0; col < D && k < max_nnz; ++col) {
            auto& v = dense_matrix[static_cast<size_t>(row) * D + col];
            if (std::sqrt(v.x * v.x + v.y * v.y) > tolerance) {
                h_val[static_cast<size_t>(row) * max_nnz + k] = v;
                h_col[static_cast<size_t>(row) * max_nnz + k] = col;
                ++k;
            }
        }
    }

    // Upload to GPU — use scratch_aux (split in half for val and col)
    const size_t val_bytes = total * sizeof(cuDoubleComplex);
    const size_t col_bytes = total * sizeof(int);
    // Ensure scratch_aux has room for both arrays
    auto* base = static_cast<char*>(pool.scratch_aux.ensure(val_bytes + col_bytes));
    auto* d_val = reinterpret_cast<cuDoubleComplex*>(base);
    auto* d_col = reinterpret_cast<int*>(base + val_bytes);

    CHECK_CUDA(cudaMemcpy(d_val, h_val.data(), val_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col, h_col.data(), col_bytes, cudaMemcpyHostToDevice));

    return {d_val, d_col, max_nnz};
}

}  // namespace

void QuantumCircuit::execute_level0_gate(const GateParams& gate) {
    execute_level0_gate(gate, nullptr);
}

void QuantumCircuit::execute_level0_gate(const GateParams& gate,
                                         const ExactGateBatchContext* batch_context) {
    ScopedNvtxRange nvtx_range("qc::execute_level0_gate");
    const auto& target_states =
        batch_context && batch_context->target_states
            ? *batch_context->target_states
            : get_cached_target_states();

    if (target_states.empty()) return;
    if (state_pool_.spans_multiple_devices(target_states)) {
        synchronize_async_cv_pipeline();
        for (auto& [device_id, local_targets] : state_pool_.bucket_state_ids_by_device(target_states)) {
            CHECK_CUDA(cudaSetDevice(device_id));
            state_pool_.activate_device_view(device_id);
            ExactGateBatchContext local_context;
            local_context.target_states = &local_targets;
            local_context.batch_size = static_cast<int>(local_targets.size());
            execute_level0_gate(gate, &local_context);
        }
        return;
    }

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    const bool use_async_compute =
        async_cv_pipeline_enabled_ &&
        (gate.type == GateType::PHASE_ROTATION ||
         gate.type == GateType::KERR_GATE ||
         gate.type == GateType::CONDITIONAL_PARITY);
    if (!use_async_compute && async_cv_pipeline_enabled_) {
        synchronize_async_cv_pipeline();
    }

    size_t upload_slot = 0;
    int* d_target_ids = nullptr;
    if (batch_context && batch_context->d_target_ids) {
        d_target_ids = batch_context->d_target_ids;
    } else if (use_async_compute) {
        std::tie(d_target_ids, std::ignore) =
            upload_target_states_for_compute(target_states, &upload_slot);
    } else {
        d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
    }

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    double param = gate.params.empty() ? 0.0 : gate.params[0].real();
    const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];

    // 统计计算时延
    auto compute_start = std::chrono::high_resolution_clock::now();

    // Ablation: force dense D×D gemv for all Level-0 gates
    if (force_dense_fock_) {
        const int D = cv_truncation_;
        int target_fock = -1;
        if (gate.type == GateType::SNAP_GATE && gate.params.size() >= 2) {
            target_fock = static_cast<int>(std::llround(gate.params[1].real()));
        }
        const std::vector<std::complex<double>>* multi_params =
            (gate.type == GateType::MULTI_SNAP_GATE) ? &gate.params : nullptr;

        // Cross-Kerr is a two-mode diagonal gate — build D²×D² diagonal tensor
        if (gate.type == GateType::CROSS_KERR_GATE) {
            if (gate.target_qumodes.size() < 2) {
                throw std::runtime_error("Cross-Kerr门缺少两个目标qumode");
            }
            const int D2 = D * D;
            std::vector<cuDoubleComplex> tensor(
                static_cast<size_t>(D2) * D2, make_cuDoubleComplex(0.0, 0.0));
            for (int m = 0; m < D; ++m) {
                for (int n = 0; n < D; ++n) {
                    double phase = param * static_cast<double>(m) * static_cast<double>(n);
                    int idx = m * D + n;
                    tensor[static_cast<size_t>(idx) * D2 + idx] =
                        make_cuDoubleComplex(std::cos(phase), std::sin(phase));
                }
            }
            cuDoubleComplex* d_tensor = upload_dense_matrix_to_gpu(state_pool_, tensor);
            apply_dense_two_mode_gate_gpu(
                &state_pool_, d_target_ids, static_cast<int>(target_states.size()),
                d_tensor, D, gate.target_qumodes[0], gate.target_qumodes[1], num_qumodes_);
        } else {
            auto host_mat = build_dense_level0_matrix(
                gate.type, D, param, target_fock, multi_params);
            cuDoubleComplex* d_mat = upload_dense_matrix_to_gpu(state_pool_, host_mat);
            apply_dense_single_mode_gate_gpu(
                &state_pool_, d_target_ids, static_cast<int>(target_states.size()),
                d_mat, D, target_qumode, num_qumodes_);
        }
    } else {

    switch (gate.type) {
        case GateType::PHASE_ROTATION:
            apply_phase_rotation_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                         target_qumode, num_qumodes_,
                                         use_async_compute ? compute_stream_ : nullptr,
                                         !use_async_compute);
            break;
        case GateType::KERR_GATE:
            apply_kerr_gate_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                    target_qumode, num_qumodes_,
                                    use_async_compute ? compute_stream_ : nullptr,
                                    !use_async_compute);
            break;
        case GateType::CONDITIONAL_PARITY:
            apply_conditional_parity_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                                             target_qumode, num_qumodes_,
                                             use_async_compute ? compute_stream_ : nullptr,
                                             !use_async_compute);
            break;
        case GateType::SNAP_GATE: {
            if (gate.params.size() < 2) {
                throw std::runtime_error("SNAP门缺少目标Fock态参数");
            }
            const int target_fock_state = static_cast<int>(std::llround(gate.params[1].real()));
            apply_snap_on_mode(&state_pool_, d_target_ids, target_states.size(), param,
                               target_fock_state, target_qumode, num_qumodes_);
            break;
        }
        case GateType::MULTI_SNAP_GATE: {
            std::vector<double> phase_map;
            phase_map.reserve(gate.params.size());
            for (const auto& phase : gate.params) {
                phase_map.push_back(phase.real());
            }
            apply_multisnap_on_mode(&state_pool_, d_target_ids, target_states.size(), phase_map,
                                    target_qumode, num_qumodes_);
            break;
        }
        case GateType::CROSS_KERR_GATE: {
            if (gate.target_qumodes.size() < 2) {
                throw std::runtime_error("Cross-Kerr门缺少两个目标qumode");
            }
            apply_ckgate_on_modes(&state_pool_, d_target_ids, target_states.size(), param,
                                  gate.target_qumodes[0], gate.target_qumodes[1], num_qumodes_);
            break;
        }
        default:
            throw std::runtime_error("未实现的Level0门类型");
    }

    }  // end if/else force_dense_fock_

    // 检查GPU内核执行错误
    CHECK_CUDA(cudaGetLastError());
    if (use_async_compute && !(batch_context && batch_context->d_target_ids)) {
        mark_target_upload_slot_in_use(upload_slot);
    }

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行Level 1门 (梯算符门)
 */
void QuantumCircuit::execute_level1_gate(const GateParams& gate) {
    execute_level1_gate(gate, nullptr);
}

void QuantumCircuit::execute_level1_gate(const GateParams& gate,
                                         const ExactGateBatchContext* batch_context) {
    ScopedNvtxRange nvtx_range("qc::execute_level1_gate");
    const auto& target_states =
        batch_context && batch_context->target_states
            ? *batch_context->target_states
            : get_cached_target_states();

    if (target_states.empty()) return;
    if (state_pool_.spans_multiple_devices(target_states)) {
        synchronize_async_cv_pipeline();
        for (auto& [device_id, local_targets] : state_pool_.bucket_state_ids_by_device(target_states)) {
            CHECK_CUDA(cudaSetDevice(device_id));
            state_pool_.activate_device_view(device_id);
            ExactGateBatchContext local_context;
            local_context.target_states = &local_targets;
            local_context.batch_size = static_cast<int>(local_targets.size());
            execute_level1_gate(gate, &local_context);
        }
        return;
    }

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    size_t upload_slot = 0;
    int* d_target_ids = nullptr;
    if (batch_context && batch_context->d_target_ids) {
        d_target_ids = batch_context->d_target_ids;
    } else if (async_cv_pipeline_enabled_) {
        std::tie(d_target_ids, std::ignore) =
            upload_target_states_for_compute(target_states, &upload_slot);
    } else {
        d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
    }

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    // 统计计算时延
    auto compute_start = std::chrono::high_resolution_clock::now();
    const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];

    // Ablation: force dense D×D gemv for Level-1 (ladder) gates
    if (force_dense_fock_) {
        const int D = cv_truncation_;
        std::vector<cuDoubleComplex> host_mat;
        if (gate.type == GateType::CREATION_OPERATOR) {
            host_mat = build_dense_creation_matrix(D);
        } else if (gate.type == GateType::ANNIHILATION_OPERATOR) {
            host_mat = build_dense_annihilation_matrix(D);
        }
        if (!host_mat.empty()) {
            cuDoubleComplex* d_mat = upload_dense_matrix_to_gpu(state_pool_, host_mat);
            apply_dense_single_mode_gate_gpu(
                &state_pool_, d_target_ids, static_cast<int>(target_states.size()),
                d_mat, D, target_qumode, num_qumodes_);
        }
    } else {

    switch (gate.type) {
        case GateType::CREATION_OPERATOR:
            apply_creation_operator_on_mode(&state_pool_, d_target_ids, target_states.size(),
                                            target_qumode, num_qumodes_,
                                            async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                            !async_cv_pipeline_enabled_);
            break;
        case GateType::ANNIHILATION_OPERATOR:
            apply_annihilation_operator_on_mode(&state_pool_, d_target_ids, target_states.size(),
                                                target_qumode, num_qumodes_,
                                                async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                                !async_cv_pipeline_enabled_);
            break;
        default:
            break;
    }

    }  // end if/else force_dense_fock_

    // 检查GPU内核执行错误
    CHECK_CUDA(cudaGetLastError());
    if (async_cv_pipeline_enabled_ && !(batch_context && batch_context->d_target_ids)) {
        mark_target_upload_slot_in_use(upload_slot);
    }

    auto compute_end = std::chrono::high_resolution_clock::now();
    computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

/**
 * 执行Level 2门 (单模门)
 */
void QuantumCircuit::execute_level2_gate(const GateParams& gate) {
    execute_level2_gate(gate, nullptr);
}

void QuantumCircuit::execute_level2_gate(const GateParams& gate,
                                         const ExactGateBatchContext* batch_context) {
    ScopedNvtxRange nvtx_range("qc::execute_level2_gate");
    const auto& target_states =
        batch_context && batch_context->target_states
            ? *batch_context->target_states
            : get_cached_target_states();

    if (target_states.empty()) return;
    if (state_pool_.spans_multiple_devices(target_states)) {
        synchronize_async_cv_pipeline();
        for (auto& [device_id, local_targets] : state_pool_.bucket_state_ids_by_device(target_states)) {
            CHECK_CUDA(cudaSetDevice(device_id));
            state_pool_.activate_device_view(device_id);
            ExactGateBatchContext local_context;
            local_context.target_states = &local_targets;
            local_context.batch_size = static_cast<int>(local_targets.size());
            execute_level2_gate(gate, &local_context);
        }
        return;
    }

    const int target_qumode = gate.target_qumodes.empty() ? 0 : gate.target_qumodes[0];

    // === ELL sparse path (default) — O(K_eff × D) ===
    if (!force_dense_fock_) {
        // ELL path requires synchronous operation (CPU matrix build + upload)
        if (async_cv_pipeline_enabled_) {
            synchronize_async_cv_pipeline();
        }

        auto transfer_start = std::chrono::high_resolution_clock::now();
        int* d_target_ids = nullptr;
        if (batch_context && batch_context->d_target_ids) {
            d_target_ids = batch_context->d_target_ids;
        } else {
            d_target_ids = state_pool_.upload_vector_to_buffer(
                target_states, state_pool_.scratch_target_ids);
        }
        auto transfer_end = std::chrono::high_resolution_clock::now();
        transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

        auto compute_start = std::chrono::high_resolution_clock::now();
        const int D = cv_truncation_;

        if (gate.type == GateType::DISPLACEMENT && !gate.params.empty()) {
            std::complex<double> alpha = gate.params[0];
            auto dense_mat = build_dense_displacement_matrix(D, alpha);
            auto ell = build_and_upload_ell(state_pool_, dense_mat, D);
            apply_ell_gate_on_mode(
                &state_pool_, d_target_ids, static_cast<int>(target_states.size()),
                ell.d_val, ell.d_col, D, ell.bandwidth,
                target_qumode, num_qumodes_);
        } else if (gate.type == GateType::SQUEEZING && !gate.params.empty()) {
            double r = std::abs(gate.params[0]);
            double theta = std::arg(gate.params[0]);
            auto dense_mat = build_dense_squeezing_cucomplex(D, r, theta);
            auto ell = build_and_upload_ell(state_pool_, dense_mat, D);
            apply_ell_gate_on_mode(
                &state_pool_, d_target_ids, static_cast<int>(target_states.size()),
                ell.d_val, ell.d_col, D, ell.bandwidth,
                target_qumode, num_qumodes_);
        } else {
            throw std::runtime_error("Level 2 ELL path only supports displacement and squeezing gates");
        }

        CHECK_CUDA(cudaGetLastError());
        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
        return;
    }

    // === Dense path (force_dense_fock_ ablation) — O(D²) ===
    const bool displacement_uses_direct_kernel =
        gate.type == GateType::DISPLACEMENT &&
        num_qumodes_ == 1 &&
        target_qumode == 0;
    const bool use_async_compute =
        async_cv_pipeline_enabled_ &&
        ((gate.type == GateType::SQUEEZING && !gate.params.empty()) ||
         (gate.type == GateType::DISPLACEMENT &&
          !gate.params.empty() &&
          displacement_uses_direct_kernel));
    const bool needs_target_upload =
        gate.type != GateType::DISPLACEMENT || displacement_uses_direct_kernel;
    if (!use_async_compute && async_cv_pipeline_enabled_) {
        synchronize_async_cv_pipeline();
    }

    auto transfer_start = std::chrono::high_resolution_clock::now();
    size_t upload_slot = 0;
    int* d_target_ids = nullptr;
    if (batch_context && batch_context->d_target_ids) {
        d_target_ids = batch_context->d_target_ids;
    } else if (use_async_compute) {
        std::tie(d_target_ids, std::ignore) =
            upload_target_states_for_compute(target_states, &upload_slot);
    } else if (needs_target_upload) {
        d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
    }

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    if (gate.type == GateType::DISPLACEMENT && !gate.params.empty()) {
        cuDoubleComplex alpha = make_cuDoubleComplex(gate.params[0].real(), gate.params[0].imag());
        auto compute_start = std::chrono::high_resolution_clock::now();

        if (use_async_compute) {
            apply_displacement_gate(&state_pool_,
                                    d_target_ids,
                                    target_states.size(),
                                    alpha,
                                    compute_stream_,
                                    false);
        } else {
            apply_controlled_displacement_on_mode(&state_pool_,
                                                  target_states,
                                                  d_target_ids,
                                                  static_cast<int>(target_states.size()),
                                                  alpha,
                                                  target_qumode,
                                                  num_qumodes_,
                                                  async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                                  !async_cv_pipeline_enabled_);
        }

        CHECK_CUDA(cudaGetLastError());
        if (use_async_compute && !(batch_context && batch_context->d_target_ids)) {
            mark_target_upload_slot_in_use(upload_slot);
        }

        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    } else if (gate.type == GateType::SQUEEZING && !gate.params.empty()) {
        auto compute_start = std::chrono::high_resolution_clock::now();

        apply_squeezing_gate_gpu(&state_pool_,
                                 d_target_ids,
                                 static_cast<int>(target_states.size()),
                                 std::abs(gate.params[0]),
                                 std::arg(gate.params[0]),
                                 target_qumode,
                                 num_qumodes_,
                                 use_async_compute ? compute_stream_ : nullptr,
                                 !use_async_compute);

        CHECK_CUDA(cudaGetLastError());
        if (use_async_compute) {
            mark_target_upload_slot_in_use(upload_slot);
        }

        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    } else {
        throw std::runtime_error("Level 2 exact path only supports GPU-native displacement and squeezing gates");
    }
}

/**
 * 执行Level 3门 (双模门)
 */
void QuantumCircuit::execute_level3_gate(const GateParams& gate) {
    execute_level3_gate(gate, nullptr);
}

void QuantumCircuit::execute_level3_gate(const GateParams& gate,
                                         const ExactGateBatchContext* batch_context) {
    ScopedNvtxRange nvtx_range("qc::execute_level3_gate");
    const auto& target_states =
        batch_context && batch_context->target_states
            ? *batch_context->target_states
            : get_cached_target_states();

    if (target_states.empty()) return;
    if (state_pool_.spans_multiple_devices(target_states)) {
        synchronize_async_cv_pipeline();
        for (auto& [device_id, local_targets] : state_pool_.bucket_state_ids_by_device(target_states)) {
            CHECK_CUDA(cudaSetDevice(device_id));
            state_pool_.activate_device_view(device_id);
            ExactGateBatchContext local_context;
            local_context.target_states = &local_targets;
            local_context.batch_size = static_cast<int>(local_targets.size());
            execute_level3_gate(gate, &local_context);
        }
        return;
    }

    // 统计传输时延
    auto transfer_start = std::chrono::high_resolution_clock::now();
    size_t upload_slot = 0;
    int* d_target_ids = nullptr;
    if (batch_context && batch_context->d_target_ids) {
        d_target_ids = batch_context->d_target_ids;
    } else if (async_cv_pipeline_enabled_) {
        std::tie(d_target_ids, std::ignore) =
            upload_target_states_for_compute(target_states, &upload_slot);
    } else {
        d_target_ids = state_pool_.upload_vector_to_buffer(
            target_states, state_pool_.scratch_target_ids);
    }

    auto transfer_end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();

    if (gate.type == GateType::BEAM_SPLITTER && gate.params.size() >= 2) {
        double theta = gate.params[0].real();
        double phi = gate.params[1].real();
        const int target_qumode1 = gate.target_qumodes[0];
        const int target_qumode2 = gate.target_qumodes[1];

        // 统计计算时延
        auto compute_start = std::chrono::high_resolution_clock::now();

        // Ablation: force dense D⁴ tensor contraction for beam splitter
        if (force_dense_fock_) {
            const int D = cv_truncation_;
            std::vector<cuDoubleComplex> host_tensor;
            build_bs_matrix_recursive(host_tensor, D, theta, phi);
            const size_t bytes = host_tensor.size() * sizeof(cuDoubleComplex);
            cuDoubleComplex* d_tensor = static_cast<cuDoubleComplex*>(
                state_pool_.scratch_aux.ensure(bytes));
            CHECK_CUDA(cudaMemcpy(d_tensor, host_tensor.data(), bytes,
                                  cudaMemcpyHostToDevice));
            apply_dense_two_mode_gate_gpu(
                &state_pool_, d_target_ids, static_cast<int>(target_states.size()),
                d_tensor, D, target_qumode1, target_qumode2, num_qumodes_);
        } else {
            apply_beam_splitter_recursive(&state_pool_, d_target_ids, static_cast<int>(target_states.size()),
                                          theta, phi, target_qumode1, target_qumode2, num_qumodes_,
                                          async_cv_pipeline_enabled_ ? compute_stream_ : nullptr,
                                          !async_cv_pipeline_enabled_);
        }

        // 检查GPU内核执行错误
        CHECK_CUDA(cudaGetLastError());
        if (async_cv_pipeline_enabled_ && !(batch_context && batch_context->d_target_ids)) {
            mark_target_upload_slot_in_use(upload_slot);
        }

        auto compute_end = std::chrono::high_resolution_clock::now();
        computation_time_ += std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    }
}

/**
 * 执行Level 4门 (混合控制门)
 */
void QuantumCircuit::execute_level4_gate(const GateParams& gate) {
    execute_hybrid_gate(gate);
}

/**
 * 执行Qubit门操作
 */
bool QuantumCircuit::try_get_single_qubit_gate_matrix(
    const GateParams& gate,
    std::array<std::complex<double>, 4>* matrix) const {
    if (!matrix) {
        throw std::invalid_argument("single-qubit matrix output must not be null");
    }
    if (gate.target_qubits.size() != 1 || !gate.target_qumodes.empty()) {
        return false;
    }

    matrix->fill(std::complex<double>(0.0, 0.0));
    switch (gate.type) {
        case GateType::PAULI_X:
            (*matrix)[1] = 1.0;
            (*matrix)[2] = 1.0;
            return true;
        case GateType::PAULI_Y:
            (*matrix)[1] = std::complex<double>(0.0, -1.0);
            (*matrix)[2] = std::complex<double>(0.0, 1.0);
            return true;
        case GateType::PAULI_Z:
            (*matrix)[0] = 1.0;
            (*matrix)[3] = -1.0;
            return true;
        case GateType::HADAMARD: {
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            (*matrix)[0] = inv_sqrt2;
            (*matrix)[1] = inv_sqrt2;
            (*matrix)[2] = inv_sqrt2;
            (*matrix)[3] = -inv_sqrt2;
            return true;
        }
        case GateType::ROTATION_X: {
            if (gate.params.empty()) {
                throw std::runtime_error("Rx门需要角度参数");
            }
            const double theta = gate.params[0].real();
            const double cos_half = std::cos(theta / 2.0);
            const double sin_half = std::sin(theta / 2.0);
            (*matrix)[0] = cos_half;
            (*matrix)[1] = std::complex<double>(0.0, -sin_half);
            (*matrix)[2] = std::complex<double>(0.0, -sin_half);
            (*matrix)[3] = cos_half;
            return true;
        }
        case GateType::ROTATION_Y: {
            if (gate.params.empty()) {
                throw std::runtime_error("Ry门需要角度参数");
            }
            const double theta = gate.params[0].real();
            const double cos_half = std::cos(theta / 2.0);
            const double sin_half = std::sin(theta / 2.0);
            (*matrix)[0] = cos_half;
            (*matrix)[1] = -sin_half;
            (*matrix)[2] = sin_half;
            (*matrix)[3] = cos_half;
            return true;
        }
        case GateType::ROTATION_Z: {
            if (gate.params.empty()) {
                throw std::runtime_error("Rz门需要角度参数");
            }
            const double theta = gate.params[0].real();
            const double cos_half = std::cos(theta / 2.0);
            const double sin_half = std::sin(theta / 2.0);
            (*matrix)[0] = std::complex<double>(cos_half, -sin_half);
            (*matrix)[3] = std::complex<double>(cos_half, sin_half);
            return true;
        }
        case GateType::PHASE_GATE_S:
            (*matrix)[0] = 1.0;
            (*matrix)[3] = std::complex<double>(0.0, 1.0);
            return true;
        case GateType::PHASE_GATE_T:
            (*matrix)[0] = 1.0;
            (*matrix)[3] = std::complex<double>(std::cos(M_PI / 4.0), std::sin(M_PI / 4.0));
            return true;
        default:
            return false;
    }
}

void QuantumCircuit::apply_single_qubit_gate_matrix(
    int target_qubit,
    const std::array<std::complex<double>, 4>& matrix) {
    auto build_single_qubit_transform =
        [&](HDDNode* root,
            int single_target,
            const SingleQubitMatrix& single_qubit_matrix) -> HDDNode* {
            std::unordered_map<HDDNode*, HDDNode*> memo;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = memo.find(node);
                    if (memo_it != memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == single_target) {
                        HDDNode* low = node->low;
                        HDDNode* high = node->high;
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            hdd_add(low, single_qubit_matrix[0] * node->w_low,
                                    high, single_qubit_matrix[1] * node->w_high),
                            hdd_add(low, single_qubit_matrix[2] * node->w_low,
                                    high, single_qubit_matrix[3] * node->w_high),
                            1.0,
                            1.0);
                    } else if (node->qubit_level > single_target) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    memo.emplace(node, transformed);
                    return transformed;
                };
            return transform(root);
        };

    auto build_monomial_single_qubit_transform =
        [&](HDDNode* root,
            int single_target,
            const SingleQubitMatrix& single_qubit_matrix) -> HDDNode* {
            constexpr double kMatrixTolerance = 1e-14;
            const bool diagonal =
                std::abs(single_qubit_matrix[1]) < kMatrixTolerance &&
                std::abs(single_qubit_matrix[2]) < kMatrixTolerance;
            const bool anti_diagonal =
                std::abs(single_qubit_matrix[0]) < kMatrixTolerance &&
                std::abs(single_qubit_matrix[3]) < kMatrixTolerance;
            if (!diagonal && !anti_diagonal) {
                throw std::invalid_argument("monomial qubit transform requires diagonal or anti-diagonal matrix");
            }

            std::unordered_map<HDDNode*, HDDNode*> memo;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = memo.find(node);
                    if (memo_it != memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == single_target) {
                        if (diagonal) {
                            transformed = node_manager_.get_or_create_node(
                                node->qubit_level,
                                node->low,
                                node->high,
                                single_qubit_matrix[0] * node->w_low,
                                single_qubit_matrix[3] * node->w_high);
                        } else {
                            transformed = node_manager_.get_or_create_node(
                                node->qubit_level,
                                node->high,
                                node->low,
                                single_qubit_matrix[1] * node->w_high,
                                single_qubit_matrix[2] * node->w_low);
                        }
                    } else if (node->qubit_level > single_target) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    memo.emplace(node, transformed);
                    return transformed;
                };
            return transform(root);
        };

    constexpr double kMatrixTolerance = 1e-14;
    const bool use_monomial_single_qubit_transform =
        (std::abs(matrix[1]) < kMatrixTolerance &&
         std::abs(matrix[2]) < kMatrixTolerance) ||
        (std::abs(matrix[0]) < kMatrixTolerance &&
         std::abs(matrix[3]) < kMatrixTolerance);
    HDDNode* new_root = use_monomial_single_qubit_transform
        ? build_monomial_single_qubit_transform(root_node_, target_qubit, matrix)
        : build_single_qubit_transform(root_node_, target_qubit, matrix);
    if (use_monomial_single_qubit_transform) {
        replace_root_node_preserving_terminals(new_root);
        return;
    }

    state_pool_.synchronize_all_devices();
    replace_root_node(new_root);
}

void QuantumCircuit::execute_qubit_gate(const GateParams& gate) {
    ScopedNvtxRange nvtx_range("qc::execute_qubit_gate");
    if (gate.target_qubits.empty()) {
        throw std::runtime_error("Qubit门需要指定目标Qubit");
    }

    const int target_qubit = gate.target_qubits[0];
    if (target_qubit >= num_qubits_) {
        throw std::runtime_error("目标Qubit索引超出范围");
    }

    SingleQubitMatrix single_qubit_matrix;
    if (try_get_single_qubit_gate_matrix(gate, &single_qubit_matrix)) {
        apply_single_qubit_gate_matrix(target_qubit, single_qubit_matrix);
        return;
    }

    auto build_monomial_single_qubit_transform =
        [&](HDDNode* root,
            int single_target,
            const SingleQubitMatrix& single_qubit_matrix) -> HDDNode* {
            constexpr double kMatrixTolerance = 1e-14;
            const bool diagonal =
                std::abs(single_qubit_matrix[1]) < kMatrixTolerance &&
                std::abs(single_qubit_matrix[2]) < kMatrixTolerance;
            const bool anti_diagonal =
                std::abs(single_qubit_matrix[0]) < kMatrixTolerance &&
                std::abs(single_qubit_matrix[3]) < kMatrixTolerance;
            if (!diagonal && !anti_diagonal) {
                throw std::invalid_argument(
                    "monomial qubit transform requires diagonal or anti-diagonal matrix");
            }

            std::unordered_map<HDDNode*, HDDNode*> memo;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = memo.find(node);
                    if (memo_it != memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == single_target) {
                        if (diagonal) {
                            transformed = node_manager_.get_or_create_node(
                                node->qubit_level,
                                node->low,
                                node->high,
                                single_qubit_matrix[0] * node->w_low,
                                single_qubit_matrix[3] * node->w_high);
                        } else {
                            transformed = node_manager_.get_or_create_node(
                                node->qubit_level,
                                node->high,
                                node->low,
                                single_qubit_matrix[1] * node->w_high,
                                single_qubit_matrix[2] * node->w_low);
                        }
                    } else if (node->qubit_level > single_target) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    memo.emplace(node, transformed);
                    return transformed;
                };
            return transform(root);
        };

    switch (gate.type) {
        case GateType::CNOT: {
            if (gate.target_qubits.size() < 2) {
                throw std::runtime_error("CNOT门需要控制位和目标位");
            }
            const int control = gate.target_qubits[0];
            const int target = gate.target_qubits[1];
            const SingleQubitMatrix px = {0.0, 1.0, 1.0, 0.0};

            std::unordered_map<HDDNode*, HDDNode*> control_memo;
            HDDNode* new_root = nullptr;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = control_memo.find(node);
                    if (memo_it != control_memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == control) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            node->low,
                            build_monomial_single_qubit_transform(node->high, target, px),
                            node->w_low,
                            node->w_high);
                    } else if (node->qubit_level > control) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    control_memo.emplace(node, transformed);
                    return transformed;
                };

            new_root = transform(root_node_);
            replace_root_node_preserving_terminals(new_root);
            return;
        }
        case GateType::CZ: {
            if (gate.target_qubits.size() < 2) {
                throw std::runtime_error("CZ门需要控制位和目标位");
            }
            const int control = gate.target_qubits[0];
            const int target = gate.target_qubits[1];
            const SingleQubitMatrix pz = {1.0, 0.0, 0.0, -1.0};

            std::unordered_map<HDDNode*, HDDNode*> control_memo;
            HDDNode* new_root = nullptr;
            std::function<HDDNode*(HDDNode*)> transform =
                [&](HDDNode* node) -> HDDNode* {
                    if (!node || node->is_terminal()) {
                        return node;
                    }

                    const auto memo_it = control_memo.find(node);
                    if (memo_it != control_memo.end()) {
                        return memo_it->second;
                    }

                    HDDNode* transformed = nullptr;
                    if (node->qubit_level == control) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            node->low,
                            build_monomial_single_qubit_transform(node->high, target, pz),
                            node->w_low,
                            node->w_high);
                    } else if (node->qubit_level > control) {
                        transformed = node_manager_.get_or_create_node(
                            node->qubit_level,
                            transform(node->low),
                            transform(node->high),
                            node->w_low,
                            node->w_high);
                    } else {
                        transformed = node;
                    }

                    control_memo.emplace(node, transformed);
                    return transformed;
                };

            new_root = transform(root_node_);
            replace_root_node_preserving_terminals(new_root);
            return;
        }
        default:
            throw std::runtime_error("不支持的Qubit门类型");
    }
}

/**
 * 执行混合门操作 (CPU+GPU)
 */
