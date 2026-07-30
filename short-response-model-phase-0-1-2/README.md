# Short-Response Model — runnable mature phase pipeline

This repository implements the ordered Phase 0–8 program plus two structural branches:

- **Phase 2B progressive reduction** tests every current layer, recovery-tunes promising deletions, and preserves full-quality/balanced/minimum-memory model profiles before DPO.
- **Phase 4B post-alignment structural reduction** revalidates or further refines the accepted Phase 4 architecture before context/sparsity calibration.

The post-Phase-3 runners execute training, merging, pruning, evaluation, profiling, router training, benchmarking, sealed evaluation, and packaging; they do not merely print instructions.

## Start here

- `Phases.md` — phase boundaries and handoffs.
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md` — every-layer/progressive deletion workflow.
- `POST_PHASE_3_EXECUTION_GUIDE.md` — Phase 4–8 execution order.
- Each Phase 4–8 folder contains `README.md`, `EXECUTION_MANUAL.md`, and `FILE_RESPONSIBILITIES.md`.
- `verify_post_phase_3.ps1` — compiles source, runs tests, and checks runner references.

## Canonical handoffs

Phase 2B writes `artifacts/phase_2_progressive_reduction/model_catalog.json`. Phase 3 may consume `full_quality`, `balanced`, or `min_memory`; reduced profiles require explicit human approval. Phase 4 writes `artifacts/references/phase_4_reference.json`. Phase 4B writes `artifacts/references/selected_inference_reference.json`. Phases 5–8 consume the latter automatically.

## Production principle

A technique ships only when it creates measured value. The best release may be the full dense model, a shallower recovered model, or a dense quantized model with context engineering while custom sparsity remains disabled. Negative selective-tuning, deletion, soup, and sparse-runtime results are valid and retained.

Generated checkpoints and benchmark artifacts are not included in the source ZIP; they must be created on the target machine so results reflect its actual model, CUDA stack, and hardware.
