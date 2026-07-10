# H100 high-cutoff single-gate stability diagnostic

Source JSON:
- experiments/results/sc26_highcutoff_gate_h100_default.json
- experiments/results/sc26_highcutoff_gate_h100_force_dense.json
- experiments/results/sc26_qaoa_low_r_c128_force_dense_h100.json

Setup: one-mode gates at D=32,64,128,256. Two initial profiles were used: low only populates the first 8 Fock states; stress also populates D/4, D/2, and 3D/4. Metrics are CPU dense vs GPU forward L2 and reverse identity U^-1 U fidelity/norm. The CPU dense displacement reference is not reliable for large alpha because the current Reference::create_displacement_matrix uses a fixed 10th-order Taylor expansion; reverse identity and norm are the main stability indicators.

## Summary stats

| metric | n | min | max | median | std |
|---|---:|---:|---:|---:|---:|
| default reverse fidelity deviation | 56 | 0 | 1.00000 | 2.220e-16 | 0.283552 |
| default reverse norm | 56 | 0.849290 | 1.226e+34 | 1.00000 | 1.624e+33 |
| default CPU/GPU L2 | 56 | 0 | 6.126e+15 | 1.748e-9 | 8.113e+14 |
| force reverse fidelity deviation | 56 | 0 | 1.00000 | 0 | 0.186961 |
| force reverse norm | 56 | 0.906894 | 7.298e+24 | 1.00000 | 9.665e+23 |
| force CPU/GPU L2 | 56 | 2.521e-16 | 5.719e+11 | 1.07790 | 7.578e+10 |

## Default path, low profile, D=256 reverse identity

| gate | fidelity | norm | L2 |
|---|---:|---:|---:|
| phase_rotation | 1.00000 | 1.00000 | 0 |
| displacement_0p1 | 1.00000 | 1.00000 | 4.611e-13 |
| displacement_0p5 | 1.00000 | 1.00000 | 6.816e-13 |
| displacement_pi_over_2 | 1.00000 | 1.00000 | 8.422e-13 |
| displacement_pi | 1.00000 | 1.00000 | 1.665e-5 |
| squeezing_0p25 | 1.00000 | 1.00000 | 1.033e-12 |
| squeezing_0p5 | 1.00000 | 1.00000 | 9.970e-13 |

## Default path, stress profile reverse identity

| gate | D | fidelity | norm | L2 |
|---|---:|---:|---:|---:|
| phase_rotation | 128 | 1.00000 | 1.00000 | 0 |
| phase_rotation | 256 | 1.00000 | 1.00000 | 0 |
| displacement_0p1 | 128 | 1.00000 | 1.00000 | 8.531e-13 |
| displacement_0p1 | 256 | 1.00000 | 1.00000 | 6.022e-13 |
| displacement_0p5 | 128 | 1.00000 | 1.00000 | 8.050e-13 |
| displacement_0p5 | 256 | 1.00000 | 1.00000 | 1.563e-12 |
| displacement_pi_over_2 | 128 | 0.990549 | 0.994358 | 0.0972184 |
| displacement_pi_over_2 | 256 | 0.000775823 | 47.1650 | 47.1477 |
| displacement_pi | 128 | 1.550e-5 | 9.559e+12 | 9.559e+12 |
| displacement_pi | 256 | 9.367e-6 | 1.226e+34 | 1.226e+34 |
| squeezing_0p25 | 128 | 0.951656 | 0.974850 | 0.219873 |
| squeezing_0p25 | 256 | 2.718e-6 | 596.086 | 596.085 |
| squeezing_0p5 | 128 | 0.870517 | 0.930292 | 0.359848 |
| squeezing_0p5 | 256 | 4.500e-13 | 2.989e+24 | 2.989e+24 |

## Force-dense/native path, stress profile selected reverse identity

| gate | D | fidelity | norm | L2 |
|---|---:|---:|---:|---:|
| displacement_pi_over_2 | 128 | 1.00000 | 1.00000 | 0 |
| displacement_pi_over_2 | 256 | 1.00000 | 1.00000 | 0 |
| displacement_pi | 128 | 1.00000 | 1.00000 | 0 |
| displacement_pi | 256 | 1.00000 | 1.00000 | 0 |
| squeezing_0p25 | 128 | 0.951656 | 0.974850 | 0.219873 |
| squeezing_0p25 | 256 | 9.811e-7 | 1028.22 | 1028.22 |
| squeezing_0p5 | 128 | 0.870731 | 0.930176 | 0.359552 |
| squeezing_0p5 | 256 | 4.178e-15 | 7.298e+24 | 7.298e+24 |

## QAOA low-r D=128 force-dense check

The low-r QAOA D=128 reverse run does not recover under force-dense/native execution: forward norm=6.674949, forward tail fraction=0.142494, forward boundary fraction=0.004951, reverse fidelity=3.464e-15, reverse norm=7.507e8. This matches the earlier default-path blow-up closely, so the QAOA D=128 instability is not explained solely by the default displacement ELL path.

## Interpretation

Low-energy single-gate inputs are stable up to D=256 on H100 for phase rotation, displacement, and squeezing under the default path. High-Fock stress inputs expose two issues: default displacement through the Level-2 ELL/Laguerre path becomes unstable for large alpha at D>=128, while squeezing reverse identity becomes non-unitary for high-Fock stress states at D=256 even when using the force-dense/native path. The CPU dense displacement L2 numbers for large alpha should not be used as truth because the reference implementation itself is a low-order Taylor approximation.
