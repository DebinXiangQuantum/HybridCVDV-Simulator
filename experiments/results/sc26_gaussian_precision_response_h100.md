# SC26 Gaussian / Symplectic Precision Audit on H100

Date: 2026-06-11

This note summarizes the H100 rerun after enabling the pure-Gaussian symplectic path for SC26 QAOA and pure-CV Gaussian benchmarks. Raw outputs are in:

- `experiments/results/sc26_qaoa_gaussian_symplectic_h100_20260611T092745Z/`
- `experiments/results/sc26_gaussian_path_audit_h100_20260611T0915Z/`

## Main Fix

The SC26 scaling runner now detects unconditional Gaussian vacuum circuits (`PHASE_ROTATION`, `DISPLACEMENT`, `SQUEEZING`, `BEAM_SPLITTER`) and runs them through a CPU Gaussian/symplectic moment backend instead of constructing a cutoff-dependent Fock state first.

This fixes the QAOA issue: the old path could allocate/materialize a large Fock state even though the circuit is pure Gaussian. The new path also bypasses the old Fock-memory case filter for pure Gaussian workloads, so large pure Gaussian QAOA cases such as `sc26_qaoa_nm8_c16` are no longer skipped.

## SC26 QAOA Full-Circuit Fidelity

Formal SC26 QAOA cases in `experiments/configs/sc26_scaling.json`:

| case | backend | forward vacuum overlap | reverse fidelity | reverse displacement L2 | reverse covariance max error |
|---|---|---:|---:|---:|---:|
| `sc26_qaoa_nm1_c16` | `cpu_gaussian_symplectic` | `2.03486382604756e-05` | `1` | `2.23314694542304e-13` | `1.11022302462516e-16` |
| `sc26_qaoa_nm2_c16` | `cpu_gaussian_symplectic` | `4.14067079055693e-10` | `1` | `3.15814669698932e-13` | `1.11022302462516e-16` |
| `sc26_qaoa_nm4_c16` | `cpu_gaussian_symplectic` | `1.71451545957714e-19` | `1` | `4.46629389084609e-13` | `1.11022302462516e-16` |
| `sc26_qaoa_nm6_c16` | `cpu_gaussian_symplectic` | `7.09924408342936e-29` | `1` | `5.47007053694133e-13` | `1.11022302462516e-16` |
| `sc26_qaoa_nm7_c16` | `cpu_gaussian_symplectic` | `1.44459949776526e-33` | `1` | `5.9083514586529e-13` | `1.11022302462516e-16` |
| `sc26_qaoa_nm8_c16` | `cpu_gaussian_symplectic` | `2.93956326112901e-38` | `1` | `6.31629339397864e-13` | `1.11022302462516e-16` |

Statistics over these 6 QAOA cases:

- Reverse fidelity: min `1`, max `1`, median `1`, std `0`
- Reverse fidelity deviation: min `0`, max `0`, median `0`, std `0`
- Reverse displacement L2: min `2.23314694542304e-13`, max `6.31629339397864e-13`, median `4.96818221389371e-13`, std `1.478338130992987e-13`
- Reverse covariance max error: min/max/median `1.11022302462516e-16`, std `0`
- Forward vacuum overlap: min `2.93956326112901e-38`, max `2.03486382604756e-05`, median `8.572577301435322e-20`, std `7.5834588720987795e-06`

Important: forward vacuum overlap is not an accuracy error. It is the physical overlap between the QAOA output state and vacuum. The identity/reverse precision metric is the reverse fidelity, which is exactly `1` within reported precision.

## Other Benchmark Path Audit

Representative H100 audit:

- Cases: 17
- OK: 15
- Errors/timeouts: 2 (`sc26_transfer_CVtoDV_nq16_c16`, `sc26_transfer_DVtoCV_nq16_c16`, each hit the 300s audit timeout)
- Pure Gaussian cases using `cpu_gaussian_symplectic`: 6
- Runtime cases with Gaussian symbolic blocks: 9
- Runtime cases requiring symbolic materialization: 5

Representative path findings:

| case | backend | Gaussian blocks | exact blocks | materializations |
|---|---|---:|---:|---:|
| `sc26_qaoa_nm8_c16` | `cpu_gaussian_symplectic` | 1 | 0 | 0 |
| `sc26_cv_qaoa_nm7_c16` | `cpu_gaussian_symplectic` | 1 | 0 | 0 |
| `sc26_cv_jch_nm7_c16` | `cpu_gaussian_symplectic` | 1 | 0 | 0 |
| `jch_photonic_chain_modes_3_timesteps_8_cutoff_32` | `cpu_gaussian_symplectic` | 1 | 0 | 0 |
| `sc26_qft_nq9_c16` | runtime | 2 | 0 | 0 |
| `sc26_cat_c16` | runtime | 1 | 1 | 0 |
| `sc26_gkp_c16` | runtime | 8 | 9 | 0 |
| `sc26_shors_c16` | runtime | 5 | 1 | 1 |
| `sc26_jch_nq3_nm2_c4` | runtime | 6 | 5 | 5 |
| `sc26_jch_nq4_nm5_c4` | runtime | 6 | 5 | 5 |
| `sc26_vqe_nq3_nm2_c4` | runtime | 1 | 3 | 1 |
| `sc26_diagonal_mix_c16` | runtime | 2 | 0 | 0 |
| `sc26_kerr_mix_nm4_c16` | runtime | 1 | 12 | 1 |

Conclusion: other benchmarks are not generally "missing Gaussian/symplectic". Mixed workloads already use Gaussian symbolic blocks where possible, but must materialize before non-Gaussian or exact components such as Jaynes-Cummings, Kerr/SNAP-style operations, or qubit-entangling branches.

The two `nq16` transfer audit cases timed out because the current mixed runtime expands controlled-Gaussian branch bookkeeping to a very large branch count (`2^16` scale). That is a separate scalability issue in the mixed branch representation, not the pure Gaussian QAOA bug.

## Reviewer-Facing Interpretation

For pure Gaussian SC26 QAOA, report Gaussian/symplectic reverse fidelity, not cutoff-projected Fock fidelity. Under this metric, the reverse circuit returns to vacuum with fidelity `1` across all six formal QAOA cases, with displacement residuals around `1e-13` and covariance residuals around machine precision.

For mixed benchmarks, report both:

- Gaussian symbolic block counts and materialization counts.
- Cutoff convergence metrics (`|O_D - O_2D|`, tail population near cutoff, norm loss) only on parts that are actually represented in truncated Fock space.

This separates physical cutoff error from implementation/numerical stability. Cutoff error is expected for Fock-projected non-Gaussian segments; norm explosion or failure to reverse a pure Gaussian circuit is not expected and should be treated as an implementation problem.
