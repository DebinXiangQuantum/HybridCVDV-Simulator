# H100 QAOA symbolic reverse precision summary

- Gaussian reverse fidelity is computed from final Gaussian moments against vacuum.
- Fock reverse fidelity is reported where full Fock materialization was enabled; large SC26 QAOA identity cases use CPU symplectic-only diagnostics.

## identity
- fidelity: n=6, min=0.999999999882335, max=1, median=0.9999999999558755, std=5.217416301013142e-11
- fidelity_deviation: n=6, min=0, max=1.17664766818848e-10, median=4.412425980149235e-11, std=5.217400912974174e-11
- reverse_gaussian_vacuum_fidelity: n=6, min=0.999999999882335, max=0.999999999985292, median=0.9999999999264595, std=3.765853415277917e-11
- reverse_gaussian_vacuum_fidelity_deviation: n=6, min=1.47081236079316e-11, max=1.17664766818848e-10, median=7.354045150620435e-11, std=3.765833661608775e-11
- reverse_gaussian_displacement_l2: n=6, min=1.24791223958305e-08, max=3.52962882773946e-08, median=2.77628635495918e-08, std=8.26115026460559e-09
- reverse_gaussian_covariance_max_abs_delta: n=6, min=2.95719401928155e-07, max=2.95719401928155e-07, median=2.95719401928155e-07, std=0
### high_r_qaoa
- fidelity: n=6, min=0.999999999882335, max=1, median=0.9999999999558755, std=5.217416301013142e-11
- fidelity_deviation: n=6, min=0, max=1.17664766818848e-10, median=4.412425980149235e-11, std=5.217400912974174e-11
- reverse_gaussian_vacuum_fidelity: n=6, min=0.999999999882335, max=0.999999999985292, median=0.9999999999264595, std=3.765853415277917e-11
- reverse_gaussian_vacuum_fidelity_deviation: n=6, min=1.47081236079316e-11, max=1.17664766818848e-10, median=7.354045150620435e-11, std=3.765833661608775e-11
- reverse_gaussian_displacement_l2: n=6, min=1.24791223958305e-08, max=3.52962882773946e-08, median=2.77628635495918e-08, std=8.26115026460559e-09
- reverse_gaussian_covariance_max_abs_delta: n=6, min=2.95719401928155e-07, max=2.95719401928155e-07, median=2.95719401928155e-07, std=0
### low_r_qaoa

## convergence
- reverse_fidelity: n=19, min=1, max=1, median=1, std=0
- reverse_fidelity_deviation: n=19, min=0, max=0, median=0, std=0
- reverse_gaussian_vacuum_fidelity: n=19, min=0.999999999970584, max=1, median=1, std=1.18920408770214e-11
- reverse_gaussian_vacuum_fidelity_deviation: n=19, min=0, max=2.94161361935608e-11, median=0, std=1.189209108523097e-11
- reverse_gaussian_displacement_l2: n=19, min=2.93232067589019e-15, max=1.76481441386973e-08, median=4.14692766907094e-15, std=7.624071167892911e-09
- reverse_gaussian_covariance_max_abs_delta: n=19, min=4.9960036108132e-16, max=2.95719401928155e-07, median=4.9960036108132e-16, std=1.460049411522151e-07
- reverse_l2_error: n=19, min=0, max=0, median=0, std=0
- reverse_norm: n=19, min=1, max=1, median=1, std=0
### high_r_qaoa
- reverse_fidelity: n=8, min=1, max=1, median=1, std=0
- reverse_fidelity_deviation: n=8, min=0, max=0, median=0, std=0
- reverse_gaussian_vacuum_fidelity: n=8, min=0.999999999970584, max=0.999999999985292, median=0.999999999977938, std=7.354006292814574e-12
- reverse_gaussian_vacuum_fidelity_deviation: n=8, min=1.47081236079316e-11, max=2.94161361935608e-11, median=2.20621299007462e-11, std=7.3540062928146e-12
- reverse_gaussian_displacement_l2: n=8, min=1.24791223958305e-08, max=1.76481441386973e-08, median=1.50636332672639e-08, std=2.5845108714334e-09
- reverse_gaussian_covariance_max_abs_delta: n=8, min=2.95719401928155e-07, max=2.95719401928155e-07, median=2.95719401928155e-07, std=0
- reverse_l2_error: n=8, min=0, max=0, median=0, std=0
- reverse_norm: n=8, min=1, max=1, median=1, std=0
### low_r_qaoa
- reverse_fidelity: n=11, min=1, max=1, median=1, std=0
- reverse_fidelity_deviation: n=11, min=0, max=0, median=0, std=0
- reverse_gaussian_vacuum_fidelity: n=11, min=1, max=1, median=1, std=0
- reverse_gaussian_vacuum_fidelity_deviation: n=11, min=0, max=0, median=0, std=0
- reverse_gaussian_displacement_l2: n=11, min=2.93232067589019e-15, max=4.14692766907094e-15, median=4.14692766907094e-15, std=6.047887715169275e-16
- reverse_gaussian_covariance_max_abs_delta: n=11, min=4.9960036108132e-16, max=4.9960036108132e-16, median=4.9960036108132e-16, std=0
- reverse_l2_error: n=11, min=0, max=0, median=0, std=0
- reverse_norm: n=11, min=1, max=1, median=1, std=0
