# Changed files — Phase 9 Behavior Curriculum v2

Version 1.6.1 | August 2026

The latest uploaded repository was used as the base. Phases 10 and 11 were preserved. The changes below refine Phase 9 teaching, data validity, evaluation, and documentation.

## Root documentation

- `README.md` — added the v1.6.1 summary, corrected the Phase 9 learning sequence, and documented progressive KD and five-condition evaluation.
- `Phases.md` — updated the official Phase 9 responsibility and handoff.
- `POST_PHASE_3_EXECUTION_GUIDE.md` — added clean research-ready and smoke commands and updated candidate naming.
- `STUDENT_DISTILLATION_PIPELINE.md` — amended the master design to Behavior Curriculum v2.
- `PHASE_9_REFINEMENT_ARCHITECTURE_REVIEW.md` — added the full failure diagnosis and revised architecture.
- `RELEASE_VERIFICATION_PHASE_9_BEHAVIOR_CURRICULUM_V2.md` — added validation evidence and environment limitations.
- `PACKAGE_MANIFEST_PHASE_9_BEHAVIOR_CURRICULUM_V2.json` — records the base archive, changed-file inventory, hashes, and validation summary.
- `verify_post_phase_3.ps1` — added source checks for the Phase 9 curriculum/foundation/canary assets.

## Configuration

- `configs/phase_9_student_distillation.json` — advanced the schema/revision, increased the default serious-run budget, defined curriculum/foundation mixing, progressive KD, evaluation denominator policy, readiness rules, and filtered/passthrough adversarial conditions.

## New trusted training foundation

- `data/phase_9_student_distillation/foundation/README.md` — documents training-only fictional data and its limitations.
- `data/phase_9_student_distillation/foundation/knowledge.foundation.jsonl` — 64 fictional knowledge documents.
- `data/phase_9_student_distillation/foundation/procedures.foundation.jsonl` — curated reusable procedures.
- `data/phase_9_student_distillation/foundation/prompts.foundation.jsonl` — 336 authored behavior prompts and explicit contracts.

## New deterministic canary pack

- `data/phase_9_student_distillation/canary/README.md` — documents static/smoke validation use.
- `data/phase_9_student_distillation/canary/knowledge.canary.jsonl` — 16 canary knowledge documents.
- `data/phase_9_student_distillation/canary/procedures.canary.jsonl` — canary procedure source.
- `data/phase_9_student_distillation/canary/prompts.canary.jsonl` — 44 prompts spanning supported, unsupported, injection, procedure, and social behavior.

## New Phase 9 implementation files

- `scripts/phase_9_student_distillation/phase9_metrics.py` — explicit case classes, safe denominators, functional correctness, social usability, bounded comparisons, and readiness blockers.
- `scripts/phase_9_student_distillation/build_curriculum_dataset.py` — controlled transformations from approved targets only.
- `scripts/phase_9_student_distillation/build_foundation_dataset.py` — converts fictional authored prompts into training rows with validation and provenance.
- `scripts/phase_9_student_distillation/merge_training_data.py` — deterministic weighted mixing of real and foundation curricula.

## Modified Phase 9 implementation files

- `scripts/phase_9_student_distillation/run_phase_9.ps1` — rewrote the Pilot order to split → curriculum → foundation → mix → SFT → progressive KD; added hash-validated reuse safeguards, optional direct-KD ablation, and five-condition evaluation.
- `scripts/phase_9_student_distillation/train_student.py` — added completion-preserving examples, SFT-to-KD continuation, KD warmup/schedule, teacher-confidence filtering, EOS/end-of-turn weighting, richer manifests, and source/model/adapter fingerprints for safe reuse.
- `scripts/phase_9_student_distillation/evaluate_rag.py` — added explicit denominators, social separation, functional correctness, readiness evidence, source hashes, and adversarial passthrough evaluation.
- `scripts/phase_9_student_distillation/check_phase_9_gate.py` — added bounded comparisons, functional readiness blockers, and progressive/direct candidate support.
- `scripts/phase_9_student_distillation/build_pilot_dataset.py` — added research-ready minimums, class-aware manifests, and explicit smoke mode.
- `scripts/phase_9_student_distillation/calibrate_evidence_gate.py` — excludes social rows and requires real answerable/unanswerable denominators.
- `scripts/phase_9_student_distillation/phase9_common.py` — improved connected-component grouping and class-aware split balancing.
- `scripts/phase_9_student_distillation/generate_teacher_data.py` — surfaces authored authoritative targets as labeled candidates while preserving human review.
- `scripts/phase_9_student_distillation/create_student.py` — moved heavy imports behind guarded loading so static helper tests do not require Transformers.

## Phase 9 documentation

- `scripts/phase_9_student_distillation/README.md` — rewritten architecture and commands.
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md` — rewritten serious-run, smoke, hybrid, and interpretation workflow.
- `scripts/phase_9_student_distillation/FILE_RESPONSIBILITIES.md` — updated file ownership and artifact responsibilities.

## Tests

- `tests/test_phase_9_behavior_curriculum_v2.py` — added regression tests for case classification, zero denominators, functional contracts, bounded comparisons, readiness blockers, foundation coverage, deterministic curriculum/mixture behavior, and runner integration.

## Unchanged boundaries

- Existing Phase 0–8 teacher-development source remains unchanged.
- Existing Phase 10 architecture-reduction source remains unchanged.
- Existing Phase 11 production-RAG source remains unchanged.
- No generated model checkpoints or user artifacts are included in the source ZIP.
