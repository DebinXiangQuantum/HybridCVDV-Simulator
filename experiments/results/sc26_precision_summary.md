# SC26 precision sweep summary

Generated locally from remote H100/A100 JSON results.

## H100 full SC26
device: NVIDIA H100 PCIe (cc 9.0), max_dense_dim=1048576, results=218
- circuit_cpu_dense_vs_gpu: ok=61 skipped=35 error=0; max_l2=1.29723064065e+31 (sc26_gkp_c32_full_circuit), max_abs=6.07662700784e+30, max_fid_dev=0.999626451931 (sc26_gkp_c32_full_circuit)
  - non-ok: 35 x state dimension exceeds --max-dense-dim
- gate_cpu_dense_vs_gpu: ok=13 skipped=0 error=0; max_l2=3.1485879522e-09 (gate_displacement_per_gate), max_abs=1.76845419532e-09, max_fid_dev=2.22044604925e-16 (gate_displacement_per_gate)
- reverse_identity: ok=72 skipped=37 error=0; max_l2=1 (sc26_qaoa_nm4_c16_reverse_identity), max_abs=0.672152601341, max_fid_dev=0.98942337998 (sc26_qaoa_nm4_c16_reverse_identity)
  - non-ok: 2 x inverse circuit unsupported: inverse not available for this gate type
  - non-ok: 35 x state dimension exceeds --max-dense-dim

## H100 extended gates
device: NVIDIA H100 PCIe (cc 9.0), max_dense_dim=1048576, results=40
- gate_cpu_dense_vs_gpu: ok=20 skipped=0 error=0; max_l2=1.1645425451e-05 (gate_rabi_interaction_per_gate), max_abs=8.20743954254e-06, max_fid_dev=1.35616406993e-10 (gate_rabi_interaction_per_gate)
- reverse_identity: ok=18 skipped=2 error=0; max_l2=0.0871368178609 (gate_jaynes_cummings_reverse_identity), max_abs=0.0393809039089, max_fid_dev=0.00756557528516 (gate_jaynes_cummings_reverse_identity)
  - non-ok: 2 x inverse circuit unsupported: inverse not available for this gate type

## A100 small SC26
device: NVIDIA A100 80GB PCIe (cc 8.0), max_dense_dim=262144, results=218
- circuit_cpu_dense_vs_gpu: ok=53 skipped=43 error=0; max_l2=1.29723064065e+31 (sc26_gkp_c32_full_circuit), max_abs=6.07662700784e+30, max_fid_dev=0.999626451931 (sc26_gkp_c32_full_circuit)
  - non-ok: 43 x state dimension exceeds --max-dense-dim
- gate_cpu_dense_vs_gpu: ok=13 skipped=0 error=0; max_l2=1.0094605324 (gate_kerr_per_gate), max_abs=0.672749253227, max_fid_dev=0.724587745278 (gate_kerr_per_gate)
- reverse_identity: ok=64 skipped=45 error=0; max_l2=1 (sc26_qaoa_nm4_c16_reverse_identity), max_abs=0.672152601341, max_fid_dev=0.98942337998 (sc26_qaoa_nm4_c16_reverse_identity)
  - non-ok: 2 x inverse circuit unsupported: inverse not available for this gate type
  - non-ok: 43 x state dimension exceeds --max-dense-dim

## A100 extended gates
device: NVIDIA A100 80GB PCIe (cc 8.0), max_dense_dim=1048576, results=40
- gate_cpu_dense_vs_gpu: ok=20 skipped=0 error=0; max_l2=1.15800122597 (gate_phase_rotation_per_gate), max_abs=0.672749253227, max_fid_dev=0.889901232929 (gate_phase_rotation_per_gate)
- reverse_identity: ok=18 skipped=2 error=0; max_l2=0.0871368178609 (gate_jaynes_cummings_reverse_identity), max_abs=0.0393809039089, max_fid_dev=0.00756557528516 (gate_jaynes_cummings_reverse_identity)
  - non-ok: 2 x inverse circuit unsupported: inverse not available for this gate type

## Top fidelity deviations: H100 full SC26
### gate_cpu_dense_vs_gpu
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| gate_displacement_per_gate | 16 | 3.1485879522e-09 | 1.76845419532e-09 | 2.22044604925e-16 |
| gate_phase_rotation_per_gate | 16 | 3.94148872134e-17 | 3.92523114671e-17 | 0 |
| gate_kerr_per_gate | 16 | 6.94106196954e-17 | 5.55111512313e-17 | 0 |
| gate_conditional_parity_per_gate | 16 | 6.93889390391e-18 | 6.93889390391e-18 | 0 |
| gate_snap_per_gate | 16 | 2.77555756156e-17 | 2.77555756156e-17 | 0 |
| gate_multisnap_per_gate | 16 | 6.20633538312e-17 | 5.55111512313e-17 | 0 |
| gate_creation_per_gate | 16 | 0 | 0 | 0 |
| gate_annihilation_per_gate | 16 | 0 | 0 | 0 |

### circuit_cpu_dense_vs_gpu
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| sc26_gkp_c32_full_circuit | 64 | 1.29723064065e+31 | 6.07662700784e+30 | 0.999626451931 |
| sc26_gkp_c16_full_circuit | 32 | 3.0377565757e+18 | 1.59515186795e+18 | 0.994325438134 |
| sc26_vqe_nq6_nm7_c4_full_circuit | 1048576 | 0.081204591623 | 0.00251620253844 | 0.326642994469 |
| sc26_vqe_nq3_nm7_c4_full_circuit | 131072 | 0.04726267469 | 0.00255096886225 | 0.239173933264 |
| sc26_vqe_nq6_nm4_c8_full_circuit | 262144 | 0.251447783582 | 0.0195138156652 | 0.0962934786522 |
| sc26_vqe_nq3_nm4_c8_full_circuit | 32768 | 0.241887876282 | 0.024618035787 | 0.0882002558313 |
| sc26_vqe_nq3_nm4_c4_full_circuit | 2048 | 0.113841503815 | 0.0159484034912 | 0.0634822199121 |
| sc26_vqe_nq6_nm4_c4_full_circuit | 16384 | 0.133533768722 | 0.0112584715735 | 0.0534201727685 |

### reverse_identity
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| sc26_qaoa_nm4_c16_reverse_identity | 131072 | 1 | 0.672152601341 | 0.98942337998 |
| sc26_vqe_nq3_nm7_c4_reverse_identity | 131072 | 0.991472685866 | 0.653064628143 | 0.945281881413 |
| sc26_vqe_nq6_nm7_c4_reverse_identity | 1048576 | 0.984025664299 | 0.637630184888 | 0.911381510425 |
| sc26_qaoa_nm2_c16_reverse_identity | 512 | 0.999999989049 | 0.672152591539 | 0.892674292307 |
| sc26_qaoa_nm1_c16_reverse_identity | 32 | 0.999890803008 | 0.67205486115 | 0.658113312619 |
| sc26_vqe_nq3_nm4_c4_reverse_identity | 2048 | 0.855272800589 | 0.412994835246 | 0.648064341792 |
| sc26_vqe_nq6_nm4_c4_reverse_identity | 16384 | 0.791974731934 | 0.345793693021 | 0.535689671245 |
| sc26_vqe_nq10_nm4_c4_reverse_identity | 262144 | 0.759473317774 | 0.33014833796 | 0.49525253514 |

## Top fidelity deviations: H100 extended gates
### gate_cpu_dense_vs_gpu
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| gate_rabi_interaction_per_gate | 24 | 1.1645425451e-05 | 8.20743954254e-06 | 1.35616406993e-10 |
| gate_conditional_displacement_per_gate | 24 | 2.38337734853e-06 | 1.74557932801e-06 | 5.68034508319e-12 |
| gate_displacement_per_gate | 16 | 3.1485879522e-09 | 1.76845419532e-09 | 2.22044604925e-16 |
| gate_phase_rotation_per_gate | 16 | 3.94148872134e-17 | 3.92523114671e-17 | 0 |
| gate_kerr_per_gate | 16 | 6.94106196954e-17 | 5.55111512313e-17 | 0 |
| gate_conditional_parity_per_gate | 16 | 6.93889390391e-18 | 6.93889390391e-18 | 0 |
| gate_snap_per_gate | 16 | 2.77555756156e-17 | 2.77555756156e-17 | 0 |
| gate_multisnap_per_gate | 16 | 6.20633538312e-17 | 5.55111512313e-17 | 0 |

### reverse_identity
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| gate_jaynes_cummings_reverse_identity | 24 | 0.0871368178609 | 0.0393809039089 | 0.00756557528516 |
| gate_anti_jaynes_cummings_reverse_identity | 24 | 0.0672469774716 | 0.030501586541 | 0.00450678611388 |
| gate_conditional_squeezing_reverse_identity | 24 | 0.00270948560405 | 0.0022542014193 | 7.34071987896e-06 |
| gate_squeezing_reverse_identity | 16 | 0.000179431660831 | 0.000144302109244 | 3.21957194238e-08 |
| gate_rabi_interaction_reverse_identity | 24 | 6.24614837201e-05 | 5.97810436008e-05 | 3.90143684026e-09 |
| gate_conditional_displacement_reverse_identity | 24 | 1.3405712618e-05 | 1.30632861935e-05 | 1.79713355308e-10 |
| gate_beam_splitter_reverse_identity | 144 | 1.81945036759e-16 | 7.24441791896e-17 | 2.22044604925e-16 |
| gate_phase_rotation_reverse_identity | 16 | 0 | 0 | 0 |

## Top fidelity deviations: A100 small SC26
### gate_cpu_dense_vs_gpu
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| gate_kerr_per_gate | 16 | 1.0094605324 | 0.672749253227 | 0.724587745278 |
| gate_beam_splitter_per_gate | 144 | 0.67826504854 | 0.21571766209 | 0.407133476102 |
| gate_displacement_per_gate | 16 | 3.1485879522e-09 | 1.76845419532e-09 | 2.22044604925e-16 |
| gate_phase_rotation_per_gate | 16 | 3.94148872134e-17 | 3.92523114671e-17 | 0 |
| gate_conditional_parity_per_gate | 16 | 6.93889390391e-18 | 6.93889390391e-18 | 0 |
| gate_snap_per_gate | 16 | 2.77555756156e-17 | 2.77555756156e-17 | 0 |
| gate_multisnap_per_gate | 16 | 6.20633538312e-17 | 5.55111512313e-17 | 0 |
| gate_creation_per_gate | 16 | 0 | 0 | 0 |

### circuit_cpu_dense_vs_gpu
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| sc26_gkp_c32_full_circuit | 64 | 1.29723064065e+31 | 6.07662700784e+30 | 0.999626451931 |
| sc26_gkp_c16_full_circuit | 32 | 3.0377565757e+18 | 1.59515186795e+18 | 0.994325438134 |
| sc26_vqe_nq3_nm7_c4_full_circuit | 131072 | 0.04726267469 | 0.00255096886225 | 0.239173933264 |
| sc26_vqe_nq6_nm4_c8_full_circuit | 262144 | 0.251447783582 | 0.0195138156652 | 0.0962934786522 |
| sc26_vqe_nq3_nm4_c8_full_circuit | 32768 | 0.241887876282 | 0.024618035787 | 0.0882002558313 |
| sc26_vqe_nq3_nm4_c4_full_circuit | 2048 | 0.113841503815 | 0.0159484034912 | 0.0634822199121 |
| sc26_vqe_nq6_nm4_c4_full_circuit | 16384 | 0.133533768722 | 0.0112584715735 | 0.0534201727685 |
| sc26_vqe_nq10_nm4_c4_full_circuit | 262144 | 0.151632798615 | 0.0138990323166 | 0.0472511968306 |

### reverse_identity
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| sc26_qaoa_nm4_c16_reverse_identity | 131072 | 1 | 0.672152601341 | 0.98942337998 |
| sc26_vqe_nq3_nm7_c4_reverse_identity | 131072 | 0.991472685866 | 0.653064628143 | 0.945281881413 |
| sc26_qaoa_nm2_c16_reverse_identity | 512 | 0.999999989049 | 0.672152591539 | 0.892674292307 |
| sc26_qaoa_nm1_c16_reverse_identity | 32 | 0.999890803008 | 0.67205486115 | 0.658113312619 |
| sc26_vqe_nq3_nm4_c4_reverse_identity | 2048 | 0.855272800589 | 0.412994835246 | 0.648064341792 |
| sc26_vqe_nq6_nm4_c4_reverse_identity | 16384 | 0.791974731934 | 0.345793693021 | 0.535689671245 |
| sc26_vqe_nq10_nm4_c4_reverse_identity | 262144 | 0.759473317774 | 0.33014833796 | 0.49525253514 |
| sc26_vqe_nq3_nm4_c8_reverse_identity | 32768 | 0.649055078296 | 0.340697456999 | 0.410590406823 |

## Top fidelity deviations: A100 extended gates
### gate_cpu_dense_vs_gpu
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| gate_phase_rotation_per_gate | 16 | 1.15800122597 | 0.567186335116 | 0.889901232929 |
| gate_kerr_per_gate | 16 | 1.0094605324 | 0.672749253227 | 0.724587745278 |
| gate_annihilation_per_gate | 16 | 1.0222464567 | 0.602468042687 | 0.439545440305 |
| gate_conditional_parity_per_gate | 16 | 0.646588910715 | 0.37322855867 | 0.273109192711 |
| gate_rabi_interaction_per_gate | 24 | 1.1645425451e-05 | 8.20743954254e-06 | 1.35616406993e-10 |
| gate_conditional_displacement_per_gate | 24 | 2.38337734853e-06 | 1.74557932801e-06 | 5.68034508319e-12 |
| gate_displacement_per_gate | 16 | 3.1485879522e-09 | 1.76845419532e-09 | 2.22044604925e-16 |
| gate_snap_per_gate | 16 | 2.77555756156e-17 | 2.77555756156e-17 | 0 |

### reverse_identity
| name | state_dim | l2 | max_abs | fidelity_deviation |
|---|---:|---:|---:|---:|
| gate_jaynes_cummings_reverse_identity | 24 | 0.0871368178609 | 0.0393809039089 | 0.00756557528516 |
| gate_anti_jaynes_cummings_reverse_identity | 24 | 0.0672469774716 | 0.030501586541 | 0.00450678611388 |
| gate_conditional_squeezing_reverse_identity | 24 | 0.00270948560405 | 0.0022542014193 | 7.34071987896e-06 |
| gate_squeezing_reverse_identity | 16 | 0.000179431660831 | 0.000144302109244 | 3.21957194238e-08 |
| gate_rabi_interaction_reverse_identity | 24 | 6.24614837201e-05 | 5.97810436008e-05 | 3.90143684026e-09 |
| gate_conditional_displacement_reverse_identity | 24 | 1.3405712618e-05 | 1.30632861935e-05 | 1.79713355308e-10 |
| gate_phase_rotation_reverse_identity | 16 | 0 | 0 | 0 |
| gate_kerr_reverse_identity | 16 | 0 | 0 | 0 |

## Notes
- H100 full SC26 used max_dense_dim=1048576; A100 small SC26 used max_dense_dim=262144 because the A100 was already heavily loaded.
- Extended gate suites were rerun after adding conditional and hybrid gates; the all-SC26 background jobs were started before that extension, so their embedded gate category contains the original 13 gate specs.
- A100 phase rotation single-gate error equals the error between CPU phase-rotated state and the original initial state, indicating the specialized phase-rotation path did not modify the state in that run while reverse identity can still cancel/appear exact.

## Requested statistics

`std` below is population standard deviation; sample std is included for reproducibility.

| metric | count | min | max | median | std | sample_std |
|---|---:|---:|---:|---:|---:|---:|
| H100 extended gate CPU-dense L2 error | 20 | 0 | 1.1645425451e-05 | 3.35852314145e-17 | 2.56370482355e-06 | 2.6303056501e-06 |
| H100 SC26 full-circuit fidelity | 61 | 0.000373548069483 | 1 | 0.99957926917 | 0.182124953751 | 0.183636390072 |
| H100 SC26 full-circuit fidelity deviation | 61 | 0 | 0.999626451931 | 0.000420730830457 | 0.182124953751 | 0.183636390072 |

Worst H100 extended gate L2 rows:

| gate | L2 error | fidelity deviation |
|---|---:|---:|
| gate_rabi_interaction_per_gate | 1.1645425451e-05 | 1.35616406993e-10 |
| gate_conditional_displacement_per_gate | 2.38337734853e-06 | 5.68034508319e-12 |
| gate_displacement_per_gate | 3.1485879522e-09 | 2.22044604925e-16 |
| gate_conditional_squeezing_per_gate | 1.3725234075e-16 | 0 |
| gate_anti_jaynes_cummings_per_gate | 7.18051514183e-17 | 0 |

Lowest H100 SC26 full-circuit fidelity rows:

| circuit | fidelity | fidelity deviation | L2 error |
|---|---:|---:|---:|
| sc26_gkp_c32_full_circuit | 0.000373548069483 | 0.999626451931 | 1.29723064065e+31 |
| sc26_gkp_c16_full_circuit | 0.00567456186636 | 0.994325438134 | 3.0377565757e+18 |
| sc26_vqe_nq6_nm7_c4_full_circuit | 0.673357005531 | 0.326642994469 | 0.081204591623 |
| sc26_vqe_nq3_nm7_c4_full_circuit | 0.760826066736 | 0.239173933264 | 0.04726267469 |
| sc26_vqe_nq6_nm4_c8_full_circuit | 0.903706521348 | 0.0962934786522 | 0.251447783582 |
| sc26_vqe_nq3_nm4_c8_full_circuit | 0.911799744169 | 0.0882002558313 | 0.241887876282 |
| sc26_vqe_nq3_nm4_c4_full_circuit | 0.936517780088 | 0.0634822199121 | 0.113841503815 |
| sc26_vqe_nq6_nm4_c4_full_circuit | 0.946579827232 | 0.0534201727685 | 0.133533768722 |


## H100 reverse cutoff convergence

Full convergence diagnostics are in `experiments/results/sc26_reverse_convergence_h100.md`. This suite uses GPU `U` and `U^{-1}U` only, not CPU dense full-circuit references. Tail fraction is probability mass with any qumode in the top 1/8 of Fock levels.

| circuit | max D | reverse fidelity | reverse norm loss | forward tail fraction | boundary fraction | interpretation |
|---|---:|---:|---:|---:|---:|---|
| conv_cat | 64 | 1 | 1.11022e-15 | 1.77545e-68 | 3.73387e-81 | well converged |
| conv_gkp | 64 | 0.902273 | 0.107791 | 0.0493612 | 0.000524402 | partially converged / needs larger D |
| conv_jch_nq3_nm2 | 64 | 1 | 3.10862e-15 | 0 | 0 | well converged |
| conv_qaoa_nm1 | 64 | 0.168339 | 0.999631 | 0.0649581 | 0.000126234 | not cutoff-stable; strong norm loss |
| conv_qaoa_nm2 | 32 | 0.0524047 | 1 | 0.126834 | 0.000212743 | not cutoff-stable; strong norm loss |
| conv_transfer_CVtoDV_nq4 | 64 | 1 | 1.11022e-15 | 9.60345e-22 | 3.18788e-25 | well converged |
| conv_transfer_DVtoCV_nq4 | 64 | 1 | 1.44329e-15 | 2.18069e-21 | 3.0425e-25 | well converged |
| conv_vqe_nq3_nm2 | 64 | 0.977871 | 0.012602 | 0.0109269 | 0.00227244 | usable but check observable convergence |
