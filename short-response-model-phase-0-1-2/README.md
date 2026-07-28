# Short-Response Model — runnable mature phase pipeline

This repository implements the ordered Phase 0–8 program plus **Phase 4B** permanent structural reduction. The post-Phase-3 runners execute training, merging, pruning, evaluation, profiling, router training, benchmarking, sealed evaluation, and packaging; they do not merely print instructions.

## Start here

- `POST_PHASE_3_EXECUTION_GUIDE.md` — copy-paste execution order.
- `Phases.md` — phase boundaries and handoffs.
- Each Phase 4–8 folder contains `README.md`, `EXECUTION_MANUAL.md`, and `FILE_RESPONSIBILITIES.md`.
- `install_post_phase_3.ps1` — installs/verifies required packages after the correct CUDA PyTorch build is installed.
- `verify_post_phase_3.ps1` — compiles source, runs regression tests, checks all Phase 4–8 runner references, and optionally verifies dependency imports.

## Canonical handoffs

Phase 4 writes `artifacts/references/phase_4_reference.json`. Phase 4B writes `artifacts/references/selected_inference_reference.json`. Phases 5–8 resolve the latter automatically, preventing checkpoint drift.

## Production principle

A technique is shipped only when it creates measured value. The best release may use the dense Phase 4/4B reference, the context system, and short learned outputs while leaving custom sparsity disabled. Negative pruning/soup/sparse-runtime results are valid and retained.

Generated checkpoints and benchmark artifacts are not included in the source ZIP; they must be created on the target machine so the results reflect its actual model, CUDA stack, and hardware.
