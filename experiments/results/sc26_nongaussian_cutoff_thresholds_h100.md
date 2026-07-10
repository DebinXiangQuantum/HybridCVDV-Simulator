# SC26 Non-Gaussian Cutoff Thresholds, H100

This table reports empirical cutoff thresholds for the SC26 reverse-circuit
convergence tests. The criterion is `reverse_fidelity >= 0.999` on H100. This is
not a mathematical guarantee for arbitrary parameters; it is a benchmark-specific
cutoff recommendation for the tested circuits and initial states.

Primary inputs:

- `experiments/results/sc26_reverse_convergence_symbolic_h100_all.json`
- `experiments/results/sc26_reverse_convergence_gkp_c8_128_h100.json`
- `experiments/results/sc26_reverse_convergence_vqe_c8_256_h100.json`

QAOA is excluded here because the tested QAOA circuits are pure Gaussian and now
use the Gaussian/symplectic path. The table below focuses on circuits that still
exercise dense Fock or non-Gaussian hybrid paths.

| circuit | workload | q/m | cutoff grid | min D for F_rev >= 0.999 | F_rev at D | forward norm loss | forward tail frac. | reverse L2 | note |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| conv_cat | cat_state_circuit | 1/1 | 8,16,32,64 | 16 | 0.999998591581 | 8.56683e-06 | 0.000193606 | 0.00118687 | empirical threshold on H100 reverse test |
| conv_gkp | gkp_state_circuit | 1/1 | 8,16,32,64,128 | 128 | 0.999999535183 | 1.53325e-05 | 0.000145540 | 0.000682444 | empirical threshold on H100 reverse test |
| conv_jch_nq3_nm2 | jch_simulation_circuit | 3/2 | 8,16,32,64 | 8 | 1.000000000000 | 1.33227e-15 | 0.0114987 | 3.57446e-15 | empirical threshold on H100 reverse test |
| conv_transfer_CVtoDV_nq4 | state_transfer_CVtoDV_circuit | 4/1 | 8,16,32,64 | 32 | 0.999999999786 | 1.74784e-10 | 2.58353e-08 | 1.46351e-05 | empirical threshold on H100 reverse test |
| conv_transfer_DVtoCV_nq4 | state_transfer_DVtoCV_circuit | 4/1 | 8,16,32,64 | 32 | 0.999999999785 | 2.84286e-10 | 1.35005e-08 | 1.46738e-05 | empirical threshold on H100 reverse test |
| conv_vqe_nq3_nm2 | vqe_circuit | 3/2 | 8,16,32,64,96,112,128,144,160,192,224,256 | 112 | 0.999799485293 | 0.000114627 | 0.000939889 | 0.0141604 | D>=144 reverse path unstable; do not interpret as cutoff error |

VQE detail:

| cutoff | reverse fidelity | forward norm loss | forward tail frac. | reverse norm loss | reverse L2 |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.736214509002 | 0.169481 | 0.0338556 | 0.196109 | 0.516446 |
| 16 | 0.914023927819 | 0.0474732 | 0.00756269 | 0.0509794 | 0.293301 |
| 32 | 0.948778288434 | 0.0272297 | 0.00933501 | 0.0285103 | 0.226337 |
| 64 | 0.977870579732 | 0.0118646 | 0.0109269 | 0.0126020 | 0.148767 |
| 96 | 0.998691671973 | 0.000738752 | 0.00330002 | 0.000823118 | 0.0361712 |
| 112 | 0.999799485293 | 0.000114627 | 0.000939889 | 0.000128992 | 0.0141604 |
| 128 | 0.997843894530 | 1.47616e-05 | 0.000202150 | 0.00105024 | 0.0464827 |
| 144 | 3.06796971119e-05 | 1.81557e-06 | 3.59405e-05 | 179.540 | 180.537 |
| 160 | 7.28077975646e-15 | 2.25408e-07 | 5.62366e-06 | 1.17195e+07 | 1.17195e+07 |
| 192 | 1.43060927187e-46 | 2.84742e-09 | 1.59304e-07 | 8.36064e+22 | 8.36064e+22 |
| 224 | 5.74472930928e-75 | 2.57492e-11 | 3.11614e-09 | 1.31937e+37 | 1.31937e+37 |
| 256 | 1.31480472996e-106 | 7.81164e-11 | 2.03767e-10 | 8.72106e+52 | 8.72106e+52 |

Interpretation:

- `conv_gkp` requires D=128 in this benchmark. D=64 has significant cutoff
  loss, while D=128 reaches `F_rev=0.999999535`.
- `conv_vqe_nq3_nm2` reaches 0.999-level fidelity at D=112, but the high-cutoff
  reverse path becomes numerically unstable from D=144 onward. Since the forward
  norm loss and tail population continue to decrease at high D, this is an
  implementation stability issue rather than a truncation-convergence failure.
- For paper reporting, use this as an empirical threshold table and pair it with
  `|O_D - O_2D|`, tail population near cutoff, and norm-loss checks for each
  non-Gaussian workload.

