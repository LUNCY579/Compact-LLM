# Short-Response Model — runnable mature phase pipeline

This repository implements the ordered Phase 0–8 program plus two structural branches and one targeted-recovery branch:

- **Phase 2B progressive reduction** tests every current layer, recovery-tunes promising deletions, and preserves full-quality/balanced/minimum-memory model profiles before DPO.
- **Phase 2C targeted recovery** ranks surviving Phase 2B layers and trains selected-layer adapters whose proportional training-resource savings exceed their proportional quality opportunity loss.
- **Phase 4B post-alignment structural reduction** revalidates or further refines the accepted Phase 4 architecture before context/sparsity calibration.

The post-Phase-3 runners execute training, merging, pruning, evaluation, profiling, router training, benchmarking, sealed evaluation, and packaging; they do not merely print instructions.

## v1.3.2 semantic-integrity evaluation

Quality Evaluator v3.1 treats understandable conversational language as acceptable even when grammar or vocabulary is informal. References are examples, follow-up questions are permitted unless explicitly forbidden, and questions do not count as unsupported certainty. All official phases and the independent playground use one shared opportunity engine: the phase's proportional primary resource benefit must be strictly greater than combined proportional quality opportunity loss. Absolute score floors and zero-issue vetoes are prohibited. Technical invalidity and human gates remain binary prerequisites.

The historical blueprint targets are preserved and amended by `BLUEPRINT_AMENDMENT_RELATIVE_OPPORTUNITY_v1.0.md`. The authoritative implementation policy is `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md`.

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
- Phase 8 records the strict response-control mode and its observed violation rate for human release review; no absolute numerical violation floor independently promotes or rejects the model.

See `RESPONSE_CONTROL_AND_EARLY_EXIT_AUDIT.md` and `scripts/phase_7_sparse_runtime/RESPONSE_CONTROL_CONTRACT.md`. This controller avoids decode work; it does not remove model weights or turn selective training into inference-time layer skipping.

## Quality evaluation v3.1

All generation-based phases and the independent Phase 2B playground use `scripts/common/quality_evaluation.py` plus `scripts/common/opportunity_evaluation.py`. Case-level analysis is continuous across mechanical integrity, semantic coherence, prompt relevance, intent fulfillment, task adequacy, logic/factual consistency, safety, context consistency, completeness, brevity, conversational naturalness, and explicit content anchors. Severe corruption still receives a severe case cost, but candidate promotion is proportional rather than controlled by hidden absolute vetoes.

Existing generation evidence can be rescored with `scripts/common/rescore_quality_artifacts.py`; Phase 2B trees can be migrated with `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py`. Run `scripts/common/verify_quality_evaluation.py` and `scripts/common/audit_relative_opportunity_policy.py` after modifications. See `EVALUATION_V3_1_SEMANTIC_INTEGRITY_HOTFIX.md` for Phase 2B migration and replay commands.

## Production principle

A technique ships only when it creates measured value. The best release may be the full dense model, a shallower recovered model, or a dense quantized model with context engineering while custom sparsity remains disabled. Negative selective-tuning, deletion, soup, and sparse-runtime results are valid and retained.

Generated checkpoints and benchmark artifacts are not included in the source ZIP; they must be created on the target machine so results reflect its actual model, CUDA stack, and hardware.
