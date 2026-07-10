# SC26 Gaussian Path Audit (H100)

cases: 17, ok: 15, error: 2

cpu_gaussian_symplectic: 6

runtime_with_gaussian_blocks: 9

runtime_with_materialization: 5

| case | status | backend | gs_blocks | exact | materialize | F_rev | cov_max |
|---|---:|---|---:|---:|---:|---:|---:|
| jch_photonic_chain_modes_3_timesteps_8_cutoff_32 | ok | cpu_gaussian_symplectic | 1 | 0 | 0 | 1 | 3.05311331771918e-15 |
| sc26_cat_c16 | ok | runtime | 1 | 1 | 0 | None | None |
| sc26_cv_jch_nm7_c16 | ok | cpu_gaussian_symplectic | 1 | 0 | 0 | 1 | 1.94289029309402e-15 |
| sc26_cv_qaoa_nm7_c16 | ok | cpu_gaussian_symplectic | 1 | 0 | 0 | 1 | 1.11022302462516e-16 |
| sc26_diagonal_mix_c16 | ok | runtime | 2 | 0 | 0 | None | None |
| sc26_gkp_c16 | ok | runtime | 8 | 9 | 0 | None | None |
| sc26_jch_nq3_nm2_c4 | ok | runtime | 6 | 5 | 5 | None | None |
| sc26_jch_nq4_nm5_c4 | ok | runtime | 6 | 5 | 5 | None | None |
| sc26_kerr_mix_nm4_c16 | ok | runtime | 1 | 12 | 1 | None | None |
| sc26_qaoa_nm1_c16 | ok | cpu_gaussian_symplectic | 1 | 0 | 0 | 1 | 1.11022302462516e-16 |
| sc26_qaoa_nm4_c16 | ok | cpu_gaussian_symplectic | 1 | 0 | 0 | 1 | 1.11022302462516e-16 |
| sc26_qaoa_nm8_c16 | ok | cpu_gaussian_symplectic | 1 | 0 | 0 | 1 | 1.11022302462516e-16 |
| sc26_qft_nq9_c16 | ok | runtime | 2 | 0 | 0 | None | None |
| sc26_shors_c16 | ok | runtime | 5 | 1 | 1 | None | None |
| sc26_transfer_CVtoDV_nq16_c16 | error | runtime | None | None | None | None | None |
| sc26_transfer_DVtoCV_nq16_c16 | error | runtime | None | None | None | None | None |
| sc26_vqe_nq3_nm2_c4 | ok | runtime | 1 | 3 | 1 | None | None |
