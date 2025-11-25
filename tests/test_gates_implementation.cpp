#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>

#include "cv_state_pool.h"
#include "reference_gates.h"

// Forward declarations for GPU functions
void apply_rabi_interaction(CVStatePool* state_pool,
                          const std::vector<int>& qubit0_states,
                          const std::vector<int>& qubit1_states,
                          double theta);

void apply_jaynes_cummings(CVStatePool* state_pool,
                         const std::vector<int>& qubit0_states,
                         const std::vector<int>& qubit1_states,
                         double theta, double phi);

void apply_anti_jaynes_cummings(CVStatePool* state_pool,
                              const std::vector<int>& qubit0_states,
                              const std::vector<int>& qubit1_states,
                              double theta, double phi);

void apply_sqr(CVStatePool* state_pool,
               const std::vector<int>& qubit0_states,
               const std::vector<int>& qubit1_states,
               const std::vector<double>& thetas,
               const std::vector<double>& phis);

void apply_controlled_displacement(CVStatePool* state_pool,
                                 const std::vector<int>& controlled_states,
                                 cuDoubleComplex alpha);

// Helper to copy GPU to Host Vector
Reference::Vector gpu_to_host(CVStatePool& pool, int state_id) {
    int d = pool.d_trunc;
    std::vector<cuDoubleComplex> host_data(d);
    cudaMemcpy(host_data.data(), &pool.data[state_id * d], d * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    Reference::Vector result(d);
    for(int i=0; i<d; ++i) {
        result[i] = Reference::Complex(cuCreal(host_data[i]), cuCimag(host_data[i]));
    }
    return result;
}

// Helper to set GPU state
void host_to_gpu(CVStatePool& pool, int state_id, const Reference::Vector& vec) {
    int d = pool.d_trunc;
    std::vector<cuDoubleComplex> host_data(d);
    for(int i=0; i<d && i<vec.size(); ++i) {
        host_data[i] = make_cuDoubleComplex(vec[i].real(), vec[i].imag());
    }
    cudaMemcpy(&pool.data[state_id * d], host_data.data(), d * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
}

class GatesTest : public ::testing::Test {
protected:
    int d_trunc = 16;
    int capacity = 100;
    CVStatePool* pool;
    
    void SetUp() override {
        pool = new CVStatePool(d_trunc, capacity);
    }
    
    void TearDown() override {
        delete pool;
    }
};

TEST_F(GatesTest, RabiInteraction) {
    int id0 = 0;
    int id1 = 1;
    
    // Init: |0, 5>
    Reference::Vector v0(d_trunc, 0.0);
    Reference::Vector v1(d_trunc, 0.0);
    v0[5] = 1.0; 
    
    host_to_gpu(*pool, id0, v0);
    host_to_gpu(*pool, id1, v1);
    
    double theta = 0.1;
    apply_rabi_interaction(pool, {id0}, {id1}, theta);
    
    Reference::InteractionGates::apply_rabi(v0, v1, theta);
    
    Reference::Vector res0 = gpu_to_host(*pool, id0);
    Reference::Vector res1 = gpu_to_host(*pool, id1);
    
    auto err0 = Reference::compute_error_metrics(v0, res0);
    auto err1 = Reference::compute_error_metrics(v1, res1);
    
    EXPECT_LT(err0.max_error, 1e-5);
    EXPECT_LT(err1.max_error, 1e-5);
}

TEST_F(GatesTest, JaynesCummings) {
    int id0 = 0;
    int id1 = 1;
    
    // Init: |1, 5> -> v1 has |5>, v0 has 0
    Reference::Vector v0(d_trunc, 0.0);
    Reference::Vector v1(d_trunc, 0.0);
    v1[5] = 1.0;
    
    host_to_gpu(*pool, id0, v0);
    host_to_gpu(*pool, id1, v1);
    
    double theta = 0.1;
    double phi = 0.5;
    
    apply_jaynes_cummings(pool, {id0}, {id1}, theta, phi);
    Reference::InteractionGates::apply_jaynes_cummings(v0, v1, theta, phi);
    
    Reference::Vector res0 = gpu_to_host(*pool, id0);
    Reference::Vector res1 = gpu_to_host(*pool, id1);
    
    EXPECT_LT(Reference::compute_error_metrics(v0, res0).max_error, 1e-5);
    EXPECT_LT(Reference::compute_error_metrics(v1, res1).max_error, 1e-5);
}

TEST_F(GatesTest, AntiJaynesCummings) {
    int id0 = 0;
    int id1 = 1;
    
    // Init: |0, 5> -> v0 has |5>
    Reference::Vector v0(d_trunc, 0.0);
    Reference::Vector v1(d_trunc, 0.0);
    v0[5] = 1.0;
    
    host_to_gpu(*pool, id0, v0);
    host_to_gpu(*pool, id1, v1);
    
    double theta = 0.1;
    double phi = 0.2;
    
    apply_anti_jaynes_cummings(pool, {id0}, {id1}, theta, phi);
    Reference::InteractionGates::apply_anti_jaynes_cummings(v0, v1, theta, phi);
    
    EXPECT_LT(Reference::compute_error_metrics(v0, gpu_to_host(*pool, id0)).max_error, 1e-5);
    EXPECT_LT(Reference::compute_error_metrics(v1, gpu_to_host(*pool, id1)).max_error, 1e-5);
}

TEST_F(GatesTest, SQR) {
    int id0 = 0;
    int id1 = 1;
    
    Reference::Vector v0(d_trunc, 0.0);
    Reference::Vector v1(d_trunc, 0.0);
    // Superposition in mode space
    v0[0] = 0.6; v0[1] = 0.8;
    v1[0] = 0.0; v1[1] = 1.0; // |1, 1>
    
    std::vector<double> thetas(d_trunc);
    std::vector<double> phis(d_trunc);
    for(int i=0; i<d_trunc; ++i) {
        thetas[i] = 0.1 * i;
        phis[i] = 0.05 * i;
    }
    
    host_to_gpu(*pool, id0, v0);
    host_to_gpu(*pool, id1, v1);
    
    apply_sqr(pool, {id0}, {id1}, thetas, phis);
    Reference::SpecialGates::apply_sqr(v0, v1, thetas, phis);
    
    EXPECT_LT(Reference::compute_error_metrics(v0, gpu_to_host(*pool, id0)).max_error, 1e-5);
    EXPECT_LT(Reference::compute_error_metrics(v1, gpu_to_host(*pool, id1)).max_error, 1e-5);
}

TEST_F(GatesTest, ControlledDisplacement) {
    int id = 0;
    Reference::Vector v(d_trunc, 0.0);
    v[0] = 1.0; // Vacuum
    
    host_to_gpu(*pool, id, v);
    
    cuDoubleComplex alpha = make_cuDoubleComplex(0.5, 0.0);
    Reference::Complex r_alpha(0.5, 0.0);
    
    // In GPU implementation, apply_controlled_displacement just applies D(alpha) to the list.
    // It doesn't handle the "control=0" case implicitly (CPU does that).
    // So we compare with Reference D(alpha).
    
    apply_controlled_displacement(pool, {id}, alpha);
    v = Reference::SingleModeGates::apply_displacement_gate(v, r_alpha);
    
    EXPECT_LT(Reference::compute_error_metrics(v, gpu_to_host(*pool, id)).max_error, 1e-4);
}
