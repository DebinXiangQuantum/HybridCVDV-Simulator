# SC26 GKP Reverse Convergence, H100, Cutoff 8-128

Source JSON: `experiments/results/sc26_reverse_convergence_gkp_c8_128_h100.json`

Run command:

```bash
./build/HybridCVDV-Simulator_sc26_precision_sweep \
  --suite convergence \
  --name-filter conv_gkp_c \
  --enable-symbolic \
  --max-dense-dim 1000000 \
  --output experiments/results/sc26_reverse_convergence_gkp_c8_128_h100.json
```

Device: NVIDIA H100 PCIe. Created at: 2026-06-11T12:23:16Z.

`conv_gkp` did not use the pure Gaussian symbolic path in this run:
`forward_gaussian_available=false`, `reverse_gaussian_available=false`,
`forward_gaussian_symbolic_blocks=0`, `reverse_gaussian_symbolic_blocks=0`.
The reported metrics are from the dense Fock path for the hybrid GKP circuit.

| cutoff | reverse fidelity | reverse fid. dev. | reverse L2 | reverse norm loss | forward norm loss | forward tail frac. | forward boundary frac. | forward mean photons |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.674322597256 | 0.325677402744 | 0.623562061610 | 0.430129619563 | 0.315923028519 | 0.0653853216094 | 0.0653853216094 | 2.34368774072 |
| 16 | 0.673376229776 | 0.326623770224 | 0.603227839501 | 0.372433594528 | 0.282380404766 | 0.0331872612684 | 0.0148617376877 | 3.27678727933 |
| 32 | 0.711623159511 | 0.288376840489 | 0.542145037278 | 0.230883138898 | 0.194512723693 | 0.0271610063778 | 0.00138977145706 | 8.09278147877 |
| 64 | 0.902273106940 | 0.0977268930601 | 0.317888267257 | 0.107791405968 | 0.0794069235531 | 0.0493611529331 | 0.000524401787418 | 15.6302473529 |
| 128 | 0.999999535183 | 4.64817299206e-07 | 0.000682443646400 | 3.04322683606e-05 | 1.53324525243e-05 | 0.000145539929552 | 5.69983848853e-07 | 24.5853110287 |

Absolute change from cutoff 64 to 128:

| metric | abs(delta) |
|---|---:|
| reverse fidelity | 0.0977264282428 |
| reverse fidelity deviation | 0.0977264282428 |
| reverse L2 error | 0.317205823611 |
| reverse norm loss | 0.107760973700 |
| forward norm loss | 0.0793915911006 |
| forward tail fraction | 0.0492156130036 |
| forward boundary fraction | 0.000523831803569 |
| forward mean total photon number | 8.95506367583 |

