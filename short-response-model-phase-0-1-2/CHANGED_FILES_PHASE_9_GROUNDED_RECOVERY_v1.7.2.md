# Changed files — Phase 9 grounded recovery v1.7.2

## Added

- `PHASE_9_GROUNDED_RECOVERY_v1.7.2.md` — governing experimental order and artifact contract.
- `RELEASE_VERIFICATION_PHASE_9_GROUNDED_RECOVERY_v1.7.2.md` — verification scope, evidence, and limitations.
- `PACKAGE_MANIFEST_PHASE_9_GROUNDED_RECOVERY_v1.7.2.json` — complete release-tree hashes; generated at packaging time.
- `scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1` — staged or one-command grounded recovery runner.
- `scripts/phase_9_student_distillation/rebalance_behavior_curriculum.py` — deterministic 45/25/10/10/5/5 training view.
- `scripts/phase_9_student_distillation/build_overfit_probe.py` — balanced deliberate overfit probe.
- `scripts/phase_9_student_distillation/phase9_training_weights.py` — observable token-level weighting.
- `scripts/phase_9_student_distillation/grounded_recovery_metrics.py` — Oracle capability metrics.
- `scripts/phase_9_student_distillation/check_grounded_recovery_ladder.py` — stage gate and unlocks.
- `scripts/phase_9_student_distillation/select_grounded_sft_candidate.py` — non-degenerate SFT selector.
- `tests/test_phase_9_grounded_recovery_v172.py` — regression coverage for the new plan.

## Modified

- `configs/phase_9_student_distillation.json` — schema 1.3, grounded ladder, compact sweep, mixture, weighting, and comparison policy.
- `scripts/phase_9_student_distillation/train_student.py` — weighted completion tokens, artifact hash, optimizer-step evidence, schema/revision update.
- `scripts/phase_9_student_distillation/evaluate_rag.py` — schema 1.3, procedure correctness/counts, and pipeline revision.
- `scripts/phase_9_student_distillation/generate_teacher_data.py` — rejects pre-v1.7.2 resume rows.
- `scripts/phase_9_student_distillation/check_phase_9_gate.py` — initialization references, four-way training comparisons, and explicit supported/abstention/conflict/procedure/social case evidence.
- `scripts/phase_9_student_distillation/phase9_metrics.py` — procedure/social task-contract costs and bounded behavior comparisons.
- `scripts/phase_9_student_distillation/audit_phase_9_results.py` — current schema guidance while preserving v1.7.1 diagnostics.
- `scripts/phase_9_student_distillation/run_phase_9.ps1` — balanced training view, initialization evaluation, tunable LR/rank/weights, and Hybrid lock.
- Phase 9 README/manual/responsibility docs, root README, `Phases.md`, and `STUDENT_DISTILLATION_PIPELINE.md` — v1.7.2 commands and phase order.
- Existing Phase 9 tests — superseding schema/revision assertions.
