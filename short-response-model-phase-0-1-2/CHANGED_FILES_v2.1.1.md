# Changed files — v2.1.1

## New files

- `configs/phase_9_procedural_recovery_v211.json` — preservation-first data/training/selection policy and conservative schedule.
- `scripts/phase_9_student_distillation/generate_procedural_recovery_v211.py` — deterministic 24-family real-world-reflected procedural corpus and preservation views.
- `scripts/phase_9_student_distillation/select_procedural_recovery_v211.py` — DEV-only baseline/Stage-A/B1/B2 preservation-adjusted selector with Pareto and sensitivity reporting.
- `scripts/phase_9_student_distillation/run_phase_9_procedural_recovery_v211.ps1` — complete v2.1.1 workflow with bounded micro-stage training, per-checkpoint evaluation, baseline fallback, and selected-candidate sealed test.
- `tests/test_phase_9_procedural_recovery_v211.py` — regression tests for corpus composition, schedule, runner, and selector behavior.
- `data/phase_9_student_distillation/procedural_recovery_v211/**` — generated 6,560-row canonical corpus, 24 procedure families, curriculum views, dataset README, manifests, and hashes.
- `PROCEDURAL_RECOVERY_V2_1_1.md` — rationale, workflow, commands, interpretation, and downstream decision rules.
- `RELEASE_VERIFICATION_v2.1.1.md` — package/data/test verification.
- `PACKAGE_MANIFEST_v2.1.1.json` — hashes and metadata for key v2.1.1 files plus the generated dataset manifest.
- `reports/PHASE_9_V211_DATASET_AUDIT.md` — generated corpus/curriculum composition audit.
- `reports/PHASE_9_V211_GENERATED_GOLD_AUDIT.json` — machine-readable clean-gold audit for generated v2.1.1 latency cases.
- `reports/PHASE_9_V21_OBSERVED_PROCEDURAL_RECOVERY_COMPARISON.md` — user-supplied v2.1 DEV result preserved as research input.
- `reports/PHASE_9_V21_ORIGINAL_DEV_PROCEDURAL_COLLISION_AUDIT.md` — user-supplied v2.1 collision audit preserved as research input.

## Updated files

- `README.md` — v2.1.1 becomes the top-level current recovery release; v2.1 remains documented for historical reproducibility.
- `Phases.md` — inserts the v2.1.1 preservation-first checkpoint before downstream Phase 9B/10/11 optimization when the v2.1 training path regresses.
- `scripts/phase_9_student_distillation/README.md` — adds the v2.1.1 preservation-first path and command.
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md` — adds the v2.1.1 checkpoint ladder and sealed-test procedure.
- `scripts/phase_9_student_distillation/FILE_RESPONSIBILITIES.md` — documents the new generator, selector, runner, and corpus boundaries.

## Intentionally unchanged

- The historical v2.1 corpus and runner remain present for reproducibility.
- The original `rag_reasoning_v3_large` training data is not rewritten.
- The 18 known historical latency-tie defects remain in the historical corpus; v2.1.1 excludes them only from preservation replay.
- `query_aware_v21` remains the retrieval-selection implementation because the v2.1 same-weight A/B showed it is the main original-path fix.
