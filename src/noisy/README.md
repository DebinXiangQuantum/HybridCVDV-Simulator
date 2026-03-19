# Noisy Runtime Sandbox

This directory is reserved for the new noisy-simulation subsystem.

Design rules:

- keep the new code isolated from the current `src/cpu`, `src/core`, and `src/gpu` runtime
- do not wire these files into the main `CMakeLists.txt` until the public API stabilizes
- reuse the current simulator through narrow interfaces rather than patching the current execution path

Current module split:

- `core/`
- `channels/`
- `executors/`
- `observables/`
- `runtime/`
- `internal/`

Facade headers kept at directory root:

- `types.h`
- `gaussian_channels.h`
- `runtime.h`

Implemented so far:

- Gaussian channel factories and validation
- host-side Gaussian moment state and channel executor
- noisy program container
- observable accumulator
- runtime mode selection

Partially implemented:

- trajectory executor scaffold

Not implemented yet:

- reference density-matrix backend
- phase-space Gaussian-mixture backend

Reference design:

- see `docs/noisy_implementation_plan.md`
