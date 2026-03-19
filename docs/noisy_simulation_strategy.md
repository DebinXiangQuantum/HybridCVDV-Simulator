# Noisy Simulation Strategy for Hybrid DV-CV

This note records a focused research conclusion for adding noisy simulation to the current Hybrid DV-CV simulator. It is not a general review. It is a recommendation for this repository's existing `HDD + Gaussian symbolic + truncated Fock` architecture.

## 1. Executive Conclusion

There is no single best method for all noisy CV simulation workloads.

For this codebase, the best overall strategy is:

1. Use Gaussian channels for Gaussian CV regions.
2. Use quantum trajectories for general Markovian noise on hybrid DV-CV circuits.
3. Add a phase-space Gaussian-mixture backend only if realistic bosonic-code states such as cat or GKP become a core target.
4. Avoid making the whole CV backend a density-matrix DD backend as the first implementation step.

In short:

- Best method for noisy Gaussian CV: moment propagation with Gaussian channels.
- Best method for general noisy hybrid DV-CV in this project: pure-state trajectories on top of the existing engine.
- Best method when non-Gaussian bosonic codes become central: linear combinations of Gaussian functions in phase space.

## 2. Why The Current Runtime Is Not Yet A Noisy Simulator

The current implementation is still fundamentally a pure-state engine:

- `GateType` contains unitary and non-Gaussian gates, but no noise-channel primitives.
- Symbolic Gaussian terminals store complex-amplitude branches, not classical probabilistic mixtures.
- The Gaussian symbolic path updates only first moments and covariance as
  - `d' = S d + d_g`
  - `Sigma' = S Sigma S^T`
- The symbolic-to-Fock projection currently assumes pure Gaussian covariance in the Bloch-Messiah path.

This means the current symbolic branch mechanism can represent coherent superpositions and approximate non-Gaussian unitary structure, but not a general CPTP noisy channel on the state.

## 3. Research Summary

### 3.1 Gaussian CV noise

For Gaussian states under Gaussian noise, the standard and most efficient representation is a Gaussian channel acting on moments. In the usual notation:

- `d' = X d + c`
- `Sigma' = X Sigma X^T + Y`

This is the right representation for:

- pure loss
- thermal loss
- additive Gaussian noise
- Gaussian amplifier noise

This approach is the most mature option because Gaussian states remain Gaussian under Gaussian channels, so the state complexity stays polynomial in the number of modes.

Primary source:

- C. Weedbrook et al., "Gaussian quantum information," Rev. Mod. Phys. 84, 621 (2012).
  https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621

Practical simulator reference:

- Strawberry Fields documents loss as a CPTP channel and exposes `LossChannel`.
  https://strawberryfields.ai/photonics/conventions/others.html

### 3.2 Non-Gaussian bosonic states under noise

When the state is not Gaussian, a plain Gaussian-moment backend is not enough. Among current practical approaches, one of the strongest directions is to represent bosonic states as linear combinations of Gaussian functions in phase space. This keeps Gaussian operations and Gaussian channels efficient while covering useful non-Gaussian resource states.

This is especially relevant for:

- cat states
- GKP states
- bosonic qubits
- moderate non-Gaussian structure with strong Gaussian processing and noise

Primary sources:

- J. E. Bourassa et al., "Fast simulation of bosonic qubits via Gaussian functions in phase space," PRX Quantum 2, 040315 (2021).
  https://doi.org/10.1103/PRXQuantum.2.040315
- Strawberry Fields bosonic backend introduction.
  https://strawberryfields.ai/photonics/demos/run_intro_bosonic.html
- Strawberry Fields GKP bosonic demo showing noisy loss studies with `LossChannel`.
  https://strawberryfields.ai/photonics/demos/run_GKP_bosonic.html

Useful follow-up result:

- N. C. Dias and R. F. Werner Konig, "Classical simulation of Gaussian superpositions in quantum circuits," Phys. Rev. A 110, 042402 (2024).
  https://doi.org/10.1103/PhysRevA.110.042402

### 3.3 Open-system simulation beyond pure Gaussian channels

For a general open quantum system with Lindblad noise, the two standard broad options are:

- evolve the density matrix directly
- sample pure-state quantum trajectories and average observables

For this repository, trajectories are the better fit.

Reason:

- each trajectory remains a pure state between stochastic jumps
- the existing HDD structure is already designed around pure-state branch amplitudes
- the existing Fock kernels also act on state vectors, not density operators
- direct density-matrix evolution would square the Hilbert-space dimension before any DD compression benefits are realized

Practical references:

- QuTiP master-equation solver documentation.
  https://qutip.org/docs/4.7/guide/dynamics/dynamics-intro.html
  https://qutip.org/docs/4.7/modules/qutip/mesolve.html
- QuTiP Monte Carlo trajectory solver documentation.
  https://qutip.org/docs/4.7/modules/qutip/mcsolve.html

### 3.4 Non-Markovian noise

If the target later expands from standard Lindblad noise to structured baths or memory effects, the current strong direction is pseudomode-based simulation.

Representative references:

- X. Luo et al., "A systematic and efficient pseudomode method to simulate open quantum systems under a bosonic environment," PRX Quantum 4, 030316 (2023).
  https://doi.org/10.1103/PRXQuantum.4.030316
- J. Zhou et al., "Quantum-Classical Decomposition of Gaussian Quantum Environments: A Stochastic Pseudomode Model," Phys. Rev. A 110, 022221 (2024).
  https://doi.org/10.1103/PhysRevA.110.022221

This is not the recommended first implementation target for this repository.

## 4. What Is "Best" By Scenario

### Scenario A: noisy Gaussian optics

Best method:

- Gaussian channels on moments

Why:

- exact for Gaussian state + Gaussian channel workloads
- cheapest state representation
- naturally matches the current symbolic Gaussian track

### Scenario B: general noisy hybrid DV-CV circuits

Best method:

- quantum trajectories over the current pure-state engine

Why:

- keeps the HDD pure-state semantics
- avoids immediate promotion to density operators
- allows DV noise and CV noise to share one runtime abstraction

### Scenario C: noisy bosonic-code simulation

Best method:

- phase-space Gaussian-mixture / bosonic backend style representation

Why:

- much better aligned with cat/GKP structure than a brute-force Fock density matrix
- Gaussian channels remain efficient
- has direct support precedent in Strawberry Fields

### Scenario D: exact mixed-state simulation for small systems only

Best method:

- density-matrix backend, possibly with DD compression on the DV side

Why:

- exact mixed-state semantics
- useful as a correctness oracle

Why it should not be the main backend:

- CV cutoff makes density matrices expensive very quickly
- the current implementation would need a major semantic rewrite

## 5. Recommendation For This Repository

The recommended roadmap is:

1. Add Gaussian channels to the symbolic track.
2. Add trajectory mode for general Lindblad-style noise.
3. Add observable-first APIs.
4. Add an optional small-system density-matrix reference backend for validation.
5. Only then consider a bosonic phase-space mixture backend if GKP/cat workloads become a primary target.

### 5.1 Phase 1: extend the symbolic Gaussian path

The current `SymplecticGate` abstraction should be generalized from a unitary Gaussian transform to a Gaussian channel transform:

- current:
  - `d' = S d + d_g`
  - `Sigma' = S Sigma S^T`
- target:
  - `d' = X d + c`
  - `Sigma' = X Sigma X^T + Y`

The first channels to implement should be:

- pure loss
- thermal loss
- additive Gaussian noise

This is the lowest-risk extension because it reuses the current symbolic infrastructure and stays exact on Gaussian workloads.

### 5.2 Phase 2: add trajectory execution

Add a trajectory executor above `QuantumCircuit`:

- each trajectory runs the existing pure-state engine
- noisy intervals sample jumps or no-jump evolution
- final observables are averaged across trajectories

This is the most natural path for:

- DV amplitude damping
- DV dephasing
- cavity loss
- hybrid dissipative interaction models

This phase should be the main path for noisy hybrid DV-CV simulation.

### 5.3 Phase 3: change the user-facing semantics

A noisy simulator should not treat state amplitudes as the primary public result.

The public API should shift toward:

- expectation values
- reduced statistics
- measurement samples
- fidelities
- parity and photon-number moments

`get_amplitude(...)` can remain for pure-state paths, but it should not be the central noisy-simulation API.

## 6. Why "DV Uses DD For Noise" Does Not Automatically Transfer To CV

For DV systems, DD methods can be effective for density matrices or superoperators because the local dimension is fixed and small.

For CV systems:

- the Hilbert space is infinite and must be truncated
- the local dimension grows with cutoff
- a density matrix squares that truncated dimension
- hybrid DV-CV systems then combine large local bosonic blocks with discrete branching structure

So while noisy DD methods are natural on the DV side, the same design is not automatically the best mainline representation on the CV side.

The better hybrid design is usually:

- DD or HDD structure for the DV-discrete branching logic
- Gaussian channels for Gaussian bosonic sectors
- trajectories for general noise
- optional phase-space Gaussian mixtures for realistic non-Gaussian bosonic codes

## 7. Final Recommendation

If the goal is to add noisy simulation to this repository without discarding the current architecture, the best technical direction is:

- Do not start with a full density-matrix CV backend.
- Start with Gaussian channels on the symbolic CV track.
- Use quantum trajectories as the main abstraction for general noisy hybrid DV-CV simulation.
- Add a bosonic phase-space mixture backend only if cat/GKP style workloads become a major use case.

That combination is the best match to both the current code structure and the current state of practical noisy CV simulation methods.
