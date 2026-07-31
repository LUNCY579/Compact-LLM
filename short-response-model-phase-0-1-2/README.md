# Short-Response Model — runnable mature phase pipeline

This repository implements the ordered Phase 0–8 program plus two structural branches and one targeted-recovery branch:

- **Phase 2B progressive reduction** tests every current layer, recovery-tunes promising deletions, and preserves full-quality/balanced/minimum-memory model profiles before DPO.
- **Phase 2C targeted recovery** ranks the surviving Phase 2B layers and trains the smallest selected-layer adapters that remain within strict quality floors of all-surviving-layer recovery.
- **Phase 4B post-alignment structural reduction** revalidates or further refines the accepted Phase 4 architecture before context/sparsity calibration.

The post-Phase-3 runners execute training, merging, pruning, evaluation, profiling, router training, benchmarking, sealed evaluation, and packaging; they do not merely print instructions.

## Start here

- `Phases.md` — phase boundaries and handoffs.
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md` — every-layer/progressive deletion workflow.
- `scripts/phase_2c_targeted_recovery/EXECUTION_MANUAL.md` — targeted recovery, review, resume, and handoff.
- `POST_PHASE_3_EXECUTION_GUIDE.md` — Phase 4–8 execution order.
- Each Phase 4–8 folder contains `README.md`, `EXECUTION_MANUAL.md`, and `FILE_RESPONSIBILITIES.md`.
- `verify_post_phase_3.ps1` — compiles source, runs tests, and checks runner references.

## Canonical handoffs

Phase 2B writes `artifacts/phase_2_progressive_reduction/model_catalog.json`. Phase 2C may start only from an approved reduced profile and writes `artifacts/phase_2c_targeted_recovery/model_catalog.json`. Phase 3 may consume either an approved Phase 2B profile or an approved Phase 2C model/adapter pair; the selectors are mutually exclusive. Phase 4 writes `artifacts/references/phase_4_reference.json`. Phase 4B writes `artifacts/references/selected_inference_reference.json`. Phases 5–8 consume the latter automatically.


## Response-control architecture

The repository now separates learned brevity from the deployable hard boundary:

- Phase 1 base/adapted evaluations use `eos_only` so weight-learning effects remain measurable.
- The smoke test and Phases 5–8 use `one_sentence`, which stops after model-native EOS or the first validated sentence boundary and retains `max_new_tokens` as a fail-safe.
- The controller protects decimals, versions, abbreviations, URLs/domains, Markdown code, ordered-list markers, internal ellipses, and unmatched quotes/brackets.
- Phase 8 refuses packaging unless sealed evaluation used strict mode with zero one-sentence violations.

See `RESPONSE_CONTROL_AND_EARLY_EXIT_AUDIT.md` and `scripts/phase_7_sparse_runtime/RESPONSE_CONTROL_CONTRACT.md`. This controller avoids decode work; it does not remove model weights or turn selective training into inference-time layer skipping.

## Production principle

A technique ships only when it creates measured value. The best release may be the full dense model, a shallower recovered model, or a dense quantized model with context engineering while custom sparsity remains disabled. Negative selective-tuning, deletion, soup, and sparse-runtime results are valid and retained.

Generated checkpoints and benchmark artifacts are not included in the source ZIP; they must be created on the target machine so results reflect its actual model, CUDA stack, and hardware.
