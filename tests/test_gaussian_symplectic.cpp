#include <gtest/gtest.h>
#include "gaussian_circuit.h"
#include "gaussian_state.h"
#include "symplectic_math.h"
#include "gaussian_kernels.h"
#include "reference_gates.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>

namespace {

size_t integer_power(size_t base, int exponent) {
    size_t result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

SymplecticGate embed_single_mode_gate(const SymplecticGate& local_gate,
                                      int total_qumodes,
                                      int target_qumode) {
    if (local_gate.num_qumodes != 1) {
        throw std::invalid_argument("test helper expects a single-mode symplectic gate");
    }

    SymplecticGate embedded(total_qumodes);
    const int dim = 2 * total_qumodes;
    const int row = 2 * target_qumode;

    embedded.S[row * dim + row] = local_gate.S[0];
    embedded.S[row * dim + row + 1] = local_gate.S[1];
    embedded.S[(row + 1) * dim + row] = local_gate.S[2];
    embedded.S[(row + 1) * dim + row + 1] = local_gate.S[3];
    embedded.d[row] = local_gate.d[0];
    embedded.d[row + 1] = local_gate.d[1];
    return embedded;
}

SymplecticGate compose_symplectic_gates(const SymplecticGate& first, const SymplecticGate& second) {
    if (first.num_qumodes != second.num_qumodes) {
        throw std::invalid_argument("cannot compose symplectic gates with different qumode counts");
    }

    SymplecticGate composed(first.num_qumodes);
    const int dim = 2 * first.num_qumodes;
    std::fill(composed.S.begin(), composed.S.end(), 0.0);
    std::fill(composed.d.begin(), composed.d.end(), 0.0);

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double sum = 0.0;
            for (int k = 0; k < dim; ++k) {
                sum += second.S[static_cast<size_t>(row) * dim + static_cast<size_t>(k)] *
                       first.S[static_cast<size_t>(k) * dim + static_cast<size_t>(col)];
            }
            composed.S[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] = sum;
        }

        double displacement = second.d[static_cast<size_t>(row)];
        for (int k = 0; k < dim; ++k) {
            displacement += second.S[static_cast<size_t>(row) * dim + static_cast<size_t>(k)] *
                            first.d[static_cast<size_t>(k)];
        }
        composed.d[static_cast<size_t>(row)] = displacement;
    }

    return composed;
}

template <typename Transform>
Reference::Vector apply_single_mode_transform(const Reference::Vector& state,
                                              int cutoff,
                                              int num_qumodes,
                                              int target_qumode,
                                              Transform&& transform) {
    const size_t stride = integer_power(static_cast<size_t>(cutoff), num_qumodes - target_qumode - 1);
    const size_t prefix_count = integer_power(static_cast<size_t>(cutoff), target_qumode);
    Reference::Vector result(state.size(), Reference::Complex(0.0, 0.0));

    for (size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const size_t block_base = prefix * static_cast<size_t>(cutoff) * stride;
        for (size_t suffix = 0; suffix < stride; ++suffix) {
            Reference::Vector slice(static_cast<size_t>(cutoff), Reference::Complex(0.0, 0.0));
            for (int photon = 0; photon < cutoff; ++photon) {
                slice[static_cast<size_t>(photon)] =
                    state[block_base + static_cast<size_t>(photon) * stride + suffix];
            }

            const Reference::Vector transformed = transform(slice);
            for (int photon = 0; photon < cutoff; ++photon) {
                result[block_base + static_cast<size_t>(photon) * stride + suffix] =
                    transformed[static_cast<size_t>(photon)];
            }
        }
    }

    return result;
}

template <typename Transform>
Reference::Vector apply_two_mode_transform(const Reference::Vector& state,
                                           int cutoff,
                                           int num_qumodes,
                                           int first_qumode,
                                           int second_qumode,
                                           Transform&& transform) {
    std::vector<size_t> strides(static_cast<size_t>(num_qumodes), 1);
    for (int mode = 0; mode < num_qumodes; ++mode) {
        strides[static_cast<size_t>(mode)] =
            integer_power(static_cast<size_t>(cutoff), num_qumodes - mode - 1);
    }

    std::vector<int> other_modes;
    for (int mode = 0; mode < num_qumodes; ++mode) {
        if (mode != first_qumode && mode != second_qumode) {
            other_modes.push_back(mode);
        }
    }

    const size_t outer_count = integer_power(static_cast<size_t>(cutoff), num_qumodes - 2);
    Reference::Vector result(state.size(), Reference::Complex(0.0, 0.0));

    for (size_t outer_index = 0; outer_index < outer_count; ++outer_index) {
        size_t residual = outer_index;
        size_t base_index = 0;
        for (int idx = static_cast<int>(other_modes.size()) - 1; idx >= 0; --idx) {
            const int digit = static_cast<int>(residual % static_cast<size_t>(cutoff));
            residual /= static_cast<size_t>(cutoff);
            base_index += static_cast<size_t>(digit) *
                          strides[static_cast<size_t>(other_modes[static_cast<size_t>(idx)])];
        }

        Reference::Vector local_state(static_cast<size_t>(cutoff * cutoff), Reference::Complex(0.0, 0.0));
        for (int photon_a = 0; photon_a < cutoff; ++photon_a) {
            for (int photon_b = 0; photon_b < cutoff; ++photon_b) {
                const size_t linear_index =
                    base_index +
                    static_cast<size_t>(photon_a) * strides[static_cast<size_t>(first_qumode)] +
                    static_cast<size_t>(photon_b) * strides[static_cast<size_t>(second_qumode)];
                local_state[static_cast<size_t>(photon_a) * cutoff + static_cast<size_t>(photon_b)] =
                    state[linear_index];
            }
        }

        const Reference::Vector transformed = transform(local_state);
        for (int photon_a = 0; photon_a < cutoff; ++photon_a) {
            for (int photon_b = 0; photon_b < cutoff; ++photon_b) {
                const size_t linear_index =
                    base_index +
                    static_cast<size_t>(photon_a) * strides[static_cast<size_t>(first_qumode)] +
                    static_cast<size_t>(photon_b) * strides[static_cast<size_t>(second_qumode)];
                result[linear_index] =
                    transformed[static_cast<size_t>(photon_a) * cutoff + static_cast<size_t>(photon_b)];
            }
        }
    }

    return result;
}

}  // namespace

class GaussianSymplecticTest : public ::testing::Test {
protected:
    void SetUp() override {
        num_qumodes = 1;
        dim = 2 * num_qumodes;
        capacity = 4;
        pool = new GaussianStatePool(num_qumodes, capacity);
        state_id = pool->allocate_state();
    }

    void TearDown() override {
        delete pool;
    }

    int num_qumodes;
    int dim;
    int capacity;
    GaussianStatePool* pool;
    int state_id;
};

TEST_F(GaussianSymplecticTest, SqueezingGateAccuracy) {
    // 1. Initial Vacuum State: d = [0, 0], Sigma = [[0.5, 0], [0, 0.5]]
    std::vector<double> d_init = {0.0, 0.0};
    std::vector<double> sigma_init = {0.5, 0.0, 0.0, 0.5};
    pool->upload_state(state_id, d_init, sigma_init);

    // 2. Prepare Squeezing gate: r=0.5, theta=0
    double r = 0.5;
    double theta = 0.0;
    SymplecticGate gate = SymplecticFactory::Squeezing(r, theta);

    // Upload gate parameters to GPU
    double *d_S, *d_dg;
    cudaMalloc(&d_S, gate.S.size() * sizeof(double));
    cudaMalloc(&d_dg, gate.d.size() * sizeof(double));
    cudaMemcpy(d_S, gate.S.data(), gate.S.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dg, gate.d.data(), gate.d.size() * sizeof(double), cudaMemcpyHostToDevice);

    // 3. Apply transformation on GPU
    int state_ids[] = {state_id};
    int* d_state_ids;
    cudaMalloc(&d_state_ids, sizeof(int));
    cudaMemcpy(d_state_ids, state_ids, sizeof(int), cudaMemcpyHostToDevice);
    
    std::cout << "Test: Applying Squeezing. state_id=" << state_id << std::endl;
    apply_batched_symplectic_update(pool, d_state_ids, 1, d_S, d_dg);

    // 4. Download and Verify
    std::vector<double> d_final, sigma_final;
    pool->download_state(state_id, d_final, sigma_final);

    double expected_s11 = 0.5 * std::exp(-2.0 * r);
    double expected_s22 = 0.5 * std::exp(2.0 * r);

    std::cout << "Final Sigma[0]=" << sigma_final[0] << ", Sigma[3]=" << sigma_final[3] << std::endl;

    EXPECT_NEAR(d_final[0], 0.0, 1e-10);
    EXPECT_NEAR(d_final[1], 0.0, 1e-10);
    EXPECT_NEAR(sigma_final[0], expected_s11, 1e-10);
    EXPECT_NEAR(sigma_final[3], expected_s22, 1e-10);
    EXPECT_NEAR(sigma_final[1], 0.0, 1e-10);
    EXPECT_NEAR(sigma_final[2], 0.0, 1e-10);

    cudaFree(d_S);
    cudaFree(d_dg);
    cudaFree(d_state_ids);
}

TEST_F(GaussianSymplecticTest, DisplacementGateAccuracy) {
    // 1. Initial Vacuum State
    std::vector<double> d_init = {0.0, 0.0};
    std::vector<double> sigma_init = {0.5, 0.0, 0.0, 0.5};
    pool->upload_state(state_id, d_init, sigma_init);

    // 2. Prepare Displacement gate: alpha = 0.5 + 0.5i
    std::complex<double> alpha(0.5, 0.5);
    SymplecticGate gate = SymplecticFactory::Displacement(alpha);

    double *d_S, *d_dg;
    cudaMalloc(&d_S, gate.S.size() * sizeof(double));
    cudaMalloc(&d_dg, gate.d.size() * sizeof(double));
    cudaMemcpy(d_S, gate.S.data(), gate.S.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dg, gate.d.data(), gate.d.size() * sizeof(double), cudaMemcpyHostToDevice);

    // 3. Apply transformation
    int state_ids[] = {state_id};
    int* d_state_ids;
    cudaMalloc(&d_state_ids, sizeof(int));
    cudaMemcpy(d_state_ids, state_ids, sizeof(int), cudaMemcpyHostToDevice);
    
    std::cout << "Test: Applying Displacement. state_id=" << state_id << std::endl;
    apply_batched_symplectic_update(pool, d_state_ids, 1, d_S, d_dg);

    // 4. Download and Verify
    std::vector<double> d_final, sigma_final;
    pool->download_state(state_id, d_final, sigma_final);

    double exp_dx = std::sqrt(2.0) * 0.5;
    double exp_dp = std::sqrt(2.0) * 0.5;

    std::cout << "Final d[0]=" << d_final[0] << ", d[1]=" << d_final[1] << std::endl;

    EXPECT_NEAR(d_final[0], exp_dx, 1e-10);
    EXPECT_NEAR(d_final[1], exp_dp, 1e-10);
    EXPECT_NEAR(sigma_final[0], 0.5, 1e-10);
    EXPECT_NEAR(sigma_final[3], 0.5, 1e-10);

    cudaFree(d_S);
    cudaFree(d_dg);
    cudaFree(d_state_ids);
}

TEST(GaussianCircuitTest, EDEKeepsGaussianTrackForSimpleGaussianEvolution) {
    GaussianCircuit circuit(1, 4);
    circuit.add_gate(SymplecticFactory::Squeezing(0.3, 0.0));

    ExecutionDecision decision = circuit.execute_with_ede(8, 0.0);

    EXPECT_EQ(decision.track, ExecutionTrack::Symbolic);
    EXPECT_EQ(circuit.get_current_track(), ExecutionTrack::Symbolic);
    EXPECT_TRUE(circuit.get_projected_fock_state().empty());
    EXPECT_GE(decision.active_modes, 1);
}

TEST(GaussianCircuitTest, GaussianToFockProjectionMatchesReference) {
    GaussianCircuit circuit(1, 4);
    circuit.set_symbolic_mode_limit(0);

    const std::complex<double> alpha(0.25, -0.1);
    const double r = 0.2;
    const double theta = 0.15;

    circuit.add_gate(SymplecticFactory::Squeezing(r, theta));
    circuit.add_gate(SymplecticFactory::Displacement(alpha));

    ExecutionDecision decision = circuit.execute_with_ede(8, 0.0);
    ASSERT_EQ(decision.track, ExecutionTrack::Tensor);

    const auto& projected = circuit.get_projected_fock_state();
    ASSERT_EQ(projected.size(), 8u);

    Reference::Vector vacuum(8, Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);
    Reference::Vector expected =
        Reference::SingleModeGates::apply_displacement_gate(
            Reference::SingleModeGates::apply_squeezing_gate(
                vacuum, std::polar(r, theta)),
            alpha);
    expected = Reference::normalize_vector(expected);

    double error_sq = 0.0;
    double norm_sq = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        error_sq += std::norm(projected[i] - expected[i]);
        norm_sq += std::norm(projected[i]);
    }

    EXPECT_NEAR(std::sqrt(norm_sq), 1.0, 1e-10);
    EXPECT_LT(std::sqrt(error_sq), 1e-8);
}

TEST(GaussianCircuitTest, CorrelatedMultiModeProjectionMatchesReference) {
    constexpr int cutoff = 8;

    GaussianCircuit circuit(2, 4);
    circuit.set_symbolic_mode_limit(0);

    const std::complex<double> alpha(0.12, -0.05);
    const double r = 0.22;
    const double sq_theta = 0.14;
    const double bs_theta = 0.31;
    const double bs_phi = 0.12;

    circuit.add_gate(embed_single_mode_gate(SymplecticFactory::Squeezing(r, sq_theta), 2, 0));
    circuit.add_gate(embed_single_mode_gate(SymplecticFactory::Displacement(alpha), 2, 1));
    circuit.add_gate(SymplecticFactory::BeamSplitter(bs_theta, bs_phi, 2, 0, 1));

    ExecutionDecision decision = circuit.execute_with_ede(cutoff, 0.0);
    ASSERT_EQ(decision.track, ExecutionTrack::Tensor);

    const auto& projected = circuit.get_projected_fock_state();
    ASSERT_EQ(projected.size(), static_cast<size_t>(cutoff * cutoff));

    Reference::Vector vacuum(cutoff, Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);

    Reference::Vector expected = Reference::tensor_product(
        Reference::SingleModeGates::apply_squeezing_gate(vacuum, std::polar(r, sq_theta)),
        Reference::SingleModeGates::apply_displacement_gate(vacuum, alpha));
    expected = Reference::TwoModeGates::apply_beam_splitter(expected, bs_theta, bs_phi);
    expected = Reference::normalize_vector(expected);

    double error_sq = 0.0;
    double norm_sq = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        error_sq += std::norm(projected[i] - expected[i]);
        norm_sq += std::norm(projected[i]);
    }

    EXPECT_NEAR(std::sqrt(norm_sq), 1.0, 1e-10);
    EXPECT_LT(std::sqrt(error_sq), 1e-8);
}

TEST(GaussianCircuitTest, HandwrittenGeneralSymplecticGateUsesBlochMessiahFallback) {
    constexpr int cutoff = 8;

    const double squeeze_r = 0.19;
    const double rotate_theta = 0.27;
    const double beam_theta = 0.31;
    const double beam_phi = 0.18;
    const std::complex<double> alpha(0.12, -0.07);

    SymplecticGate combined =
        compose_symplectic_gates(
            compose_symplectic_gates(
                compose_symplectic_gates(
                    embed_single_mode_gate(SymplecticFactory::Squeezing(squeeze_r, 0.0), 2, 0),
                    embed_single_mode_gate(SymplecticFactory::Rotation(rotate_theta), 2, 1)),
                SymplecticFactory::BeamSplitter(beam_theta, beam_phi, 2, 0, 1)),
            embed_single_mode_gate(SymplecticFactory::Displacement(alpha), 2, 1));

    GaussianCircuit circuit(2, 4);
    circuit.set_symbolic_mode_limit(0);
    circuit.add_gate(combined);

    const ExecutionDecision decision = circuit.execute_with_ede(cutoff, 0.0);
    ASSERT_EQ(decision.track, ExecutionTrack::Tensor);

    const auto& projected = circuit.get_projected_fock_state();
    ASSERT_EQ(projected.size(), static_cast<size_t>(cutoff * cutoff));

    Reference::Vector vacuum(cutoff, Reference::Complex(0.0, 0.0));
    vacuum[0] = Reference::Complex(1.0, 0.0);
    Reference::Vector expected = Reference::tensor_product(vacuum, vacuum);
    expected = apply_single_mode_transform(
        expected,
        cutoff,
        2,
        0,
        [squeeze_r](const Reference::Vector& local_state) {
            return Reference::SingleModeGates::apply_squeezing_gate(
                local_state, Reference::Complex(squeeze_r, 0.0));
        });
    expected = apply_single_mode_transform(
        expected,
        cutoff,
        2,
        1,
        [rotate_theta](const Reference::Vector& local_state) {
            return Reference::DiagonalGates::apply_phase_rotation(local_state, rotate_theta);
        });
    expected = apply_two_mode_transform(
        expected,
        cutoff,
        2,
        0,
        1,
        [beam_theta, beam_phi](const Reference::Vector& local_state) {
            return Reference::TwoModeGates::apply_beam_splitter(local_state, beam_theta, beam_phi);
        });
    expected = apply_single_mode_transform(
        expected,
        cutoff,
        2,
        1,
        [alpha](const Reference::Vector& local_state) {
            return Reference::SingleModeGates::apply_displacement_gate(local_state, alpha);
        });
    expected = Reference::normalize_vector(expected);

    double error_sq = 0.0;
    double norm_sq = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        error_sq += std::norm(projected[i] - expected[i]);
        norm_sq += std::norm(projected[i]);
    }

    EXPECT_NEAR(std::sqrt(norm_sq), 1.0, 1e-10);
    EXPECT_LT(std::sqrt(error_sq), 1e-6);
}
