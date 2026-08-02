# Short-Response Model — runnable mature phase pipeline

This repository implements the ordered Phase 0–8 program plus two structural branches and one targeted-recovery branch:

- **Phase 2B progressive reduction** tests every current layer, recovery-tunes promising deletions, and preserves full-quality/balanced/minimum-memory model profiles before DPO.
- **Phase 2C targeted recovery** ranks surviving Phase 2B layers and trains selected-layer adapters whose proportional training-resource savings exceed their proportional quality opportunity loss.
- **Phase 4B post-alignment structural reduction** revalidates or further refines the accepted Phase 4 architecture before context/sparsity calibration.

The post-Phase-3 runners execute training, merging, pruning, evaluation, profiling, router training, benchmarking, sealed evaluation, and packaging; they do not merely print instructions.

## v1.4.4 focused prompt-adequacy evaluation

Quality Evaluator v4.4 removes category from contract inference, scoring weights, case decisions, and promotion. Categories remain only as reporting tags for benchmark balancing and regression analysis. Open conversational quality is estimated continuously from the actual system/user messages; strict facts and boundaries must be authored in the individual case contract.

The decoding-integrity detector now combines reproducible surface signals instead of relying on narrow single-pattern rules. The supplied 150 collapsed Phase 2B outputs are a mandatory regression corpus: all 150 receive zero quality and a critical concrete corruption result. False-positive guards cover multilingual text, technical identifiers, versions, measurements, paths, code, arithmetic, emoji, informal grammar, alternate replies, and valid one-word responses.

The calibration packet contains 620 cases covering all 31 reporting categories, all 150 official references, all 61 scored labels, all 150 real collapsed outputs, 150 human-adjudicated coherent outputs, and 25 targeted v4.4 regressions, including the final uncertainty, rejection, visibility, scope-limitation, update-fabrication, and compression-trade-off edge cases. Run `python .\scripts\common\verify_quality_evaluation_v4_4.py` before official rescoring.

The governing files are `EVALUATION_V4_4_FINAL_EDGE_REFINEMENT.md`, `configs/quality_issue_taxonomy_v4_4.json`, `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md`, and `BLUEPRINT_AMENDMENT_RELATIVE_OPPORTUNITY_v1.0.md`.

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

## Quality evaluation v4.4

All generation-based phases and the independent Phase 2B playground use `scripts/common/quality_evaluation.py`, `scripts/common/observable_quality.py`, and `scripts/common/opportunity_evaluation.py`.

The evaluator separates universal decoding/mechanical integrity, explicit case-level contract fulfillment, category-neutral prompt-function alignment, and proportional candidate comparison. Category statistics and review-only diagnostics never reduce the score or affect promotion. Existing generations can be rescored without model inference using `scripts/common/rescore_quality_artifacts.py`; complete Phase 2B trees can be migrated with `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_4.ps1`. The official contracted probes are `probe_50_v4_4.jsonl` and `probe_150_thorough_v4_4.jsonl`.

Verification commands:

```powershell
python .\scripts\common\verify_quality_evaluation_v4_4.py
pytest -q
python .\scripts\common\audit_relative_opportunity_policy.py
```

See `EVALUATION_V4_4_MIGRATION.md` for review-only rescoring, official in-place migration, corrected-frontier replay, and clean-run commands.

## Production principle

A technique ships only when it creates measured value. The best release may be the full dense model, a shallower recovered model, or a dense quantized model with context engineering while custom sparsity remains disabled. Negative selective-tuning, deletion, soup, and sparse-runtime results are valid and retained.

Generated checkpoints and benchmark artifacts are not included in the source ZIP; they must be created on the target machine so results reflect its actual model, CUDA stack, and hardware.
