# Changed files — Phase 2B progressive reduction integration

## Why this repository changed

The existing Phase 2A result remains negative: no static one-, two-, or four-layer
adapter matched the all-layer reference within its gate. Phase 3 and Phase 4 then
reached the same saturated 1.0 heuristic score, so another aggregate-score-only
soup comparison cannot expose small architectural deviations.

The repository now adds **Phase 2B** before DPO. It tests every current decoder
layer as a reversible physical deletion, records 50 generated examples plus
continuous target-NLL contrast, optionally recovery-tunes deletion branches, and
progressively retains reductions only while weight-memory savings exceed relative
heuristic-score loss and hard semantic/category/completion floors pass.

## Added files

### Phase 2B implementation

- `scripts/phase_2_layer_selection/progressive_reduction.py` — every-layer scan, focused/all-branch recovery, progressive deletion, frontier selection, profile catalog, reports, and stop gate.
- `scripts/phase_2_layer_selection/create_depth_candidate.py` — physically removes one current layer while preserving original layer lineage.
- `scripts/phase_2_layer_selection/build_structural_probe.py` — deterministic category-balanced 50-case development probe; never reads sealed data.
- `scripts/phase_2_layer_selection/evaluate_teacher_forcing.py` — assistant-target NLL/perplexity contrast for saturated generation scores.
- `scripts/phase_2_layer_selection/compare_outputs.py` — case-by-case reply changes, fuzzy similarity, score/token deltas, and newly introduced semantic flags.
- `scripts/phase_2_layer_selection/resolve_phase_2_profile.py` — gates `balanced` and `min_memory` handoffs behind explicit human approval.
- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1` — runnable Windows orchestration.
- `scripts/phase_2_layer_selection/build_phase_2_human_review.py` — supplies the previously missing Phase 2A review-packet executable referenced by `run_phase_2.ps1`.

### Shared structural utilities

- `scripts/common/depth_reduction.py` — shared physical layer removal and cache-index reindexing used by Phases 2B/4B.
- `scripts/common/materialize_model.py` — merges an optional adapter into a standalone reload-checked checkpoint while preserving the source.
- `scripts/common/measure_model_footprint.py` — reports parameters, loaded tensor bytes, buffers, dtype counts, layer count, and serialized directory bytes.

### Documentation, configuration, and tests

- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md`
- `scripts/phase_2_layer_selection/FILE_RESPONSIBILITIES.md`
- `configs/phase_2_progressive_reduction.yaml`
- `tests/test_phase_2_progressive_reduction.py`
- `IMPLEMENTATION_CHANGELOG_PHASE_2B_PROGRESSIVE_REDUCTION.md`
- `CHANGED_FILES_PHASE_2B_PROGRESSIVE_REDUCTION.md`

## Modified files

### Workflow and top-level documentation

- `README.md`
- `Phases.md`
- `POST_PHASE_3_EXECUTION_GUIDE.md`
- `verify_post_phase_3.ps1`

### Phase 2

- `scripts/phase_2_layer_selection/README.md`
- `scripts/phase_2_layer_selection/run_phase_2.ps1`
- `scripts/phase_2_layer_selection/run_static_layer_study.py` — corrected report columns.
- `scripts/phase_2_layer_selection/check_phase_2_gate.py` — corrected report columns.
- `scripts/common/check_human_gate.py` — supports named Phase 2B/4B gate reports and custom branch reports.

### Phase 3 handoff

- `scripts/phase_3_preference_alignment/run_phase_3.ps1`
- `scripts/phase_3_preference_alignment/train_dpo.py`
- `scripts/phase_3_preference_alignment/README.md`
- `scripts/phase_3_preference_alignment/EXECUTION_MANUAL.md`

Reduced profiles are materialized SFT checkpoints, so Phase 3 attaches a fresh
all-remaining-layer LoRA before DPO. The original full-quality branch continues
its accepted SFT adapter. Branch outputs and stop gates are isolated.

### Phase 4 and Phase 4B compatibility

- `scripts/phase_4_model_soups/run_phase_4.ps1`
- `scripts/phase_4_model_soups/build_greedy_soup.py`
- `scripts/phase_4_model_soups/README.md`
- `scripts/phase_4_model_soups/EXECUTION_MANUAL.md`
- `scripts/phase_4b_structural_reduction/run_phase_4b.ps1`
- `scripts/phase_4b_structural_reduction/reduce_depth.py`
- `scripts/phase_4b_structural_reduction/measure_checkpoint.py`
- `scripts/phase_4b_structural_reduction/README.md`

Named branches such as `balanced` automatically resolve matching Phase 3/4
adapters, gates, reference manifests, and isolated artifact roots. Soups never
mix incompatible depth profiles.

### Release evidence and regression tests

- `scripts/phase_8_release_candidate/collect_ablation_evidence.py` — includes the Phase 2B catalog in release evidence.
- `tests/test_post_phase3_pipeline.py` — verifies the new runnable Phase 2B orchestration with the Phase 4–8 runners.

## Preserved historical evidence

The existing Phase 1–4 reports were not rewritten. In particular:

- the successful dense-SFT result remains intact;
- Phase 2A remains a negative selective-tuning result under its approved deviation;
- the accepted Phase 3 result remains intact;
- the current Phase 4 best-individual/soup result remains intact.

Phase 2B therefore adds a new evidence branch without changing earlier conclusions.

## Main commands

Run only the new study after the completed Phase 2A result:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2.ps1 `
  -Mode StructuralOnly `
  -MaxDrops 6 `
  -RecoveryTopK 3 `
  -RecoverySteps 50
```

Recovery-train every one-layer deletion branch on Qwen3-0.6B:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2.ps1 `
  -Mode StructuralOnly `
  -MaxDrops 6 `
  -RecoveryTopK 28 `
  -RecoverySteps 50
```

Continue an approved balanced profile:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 -Phase2Profile balanced
.\scripts\phase_4_model_soups\run_phase_4.ps1 -BranchName balanced
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 -BranchName balanced -Mode Pilot
```

## Verification performed in the build environment

- Python source compilation: PASS.
- Unit/regression suite: **21 tests passed**.
- Literal Python executables referenced by every PowerShell runner: all present.
- Synthetic smoke test for the repaired Phase 2A human-review packet: PASS.
- Generated `__pycache__` and `.pyc` files: excluded from the returned ZIP.

PowerShell and GPU/model-training execution were not available in this Linux
build container. The new model experiments must therefore run on the target
Windows/CUDA machine before any quality, memory, or speed result can be claimed.
