# Changed Files — v1.7.5 Large Corrective Datasets

## New Phase 9 data and tools

- `data/phase_9_student_distillation/rag_reasoning_v3_large/**`
  - 5,900 canonical RAG/evidence-use rows
  - 6,000 knowledge documents
  - 200 trusted procedures
  - authored train/dev/test splits
  - preference and evaluator assets
  - corrective curriculum views
- `scripts/phase_9_student_distillation/generate_large_rag_curriculum_v175.py`
- `scripts/phase_9_student_distillation/audit_rag_retrieval_coverage.py`

## New Phase 9B data and tools

- `data/phase_9b_memory_reconstruction/reconstruction_corpus_v2_large/**`
  - 1,500 source-state memories
  - 3,600 canonical reconstruction rows
  - 600 counterfactual pairs
  - 10,800 curated preference pairs
  - 14,400 evaluator candidates
- `scripts/phase_9b_memory_reconstruction/generate_large_reconstruction_curriculum_v175.py`

## Modified Phase 9 pipeline

- `scripts/phase_9_student_distillation/build_pilot_dataset.py`
  - accepts deterministic `auto_verified` gold while preserving normal human-review semantics
  - v1.7.5 pipeline revision
- `scripts/phase_9_student_distillation/validate_rag_reasoning_corpus.py`
  - schema-v3 large-corpus support
  - manifest-driven counts
  - split/hash/leakage/procedure/trust validation
- `scripts/phase_9_student_distillation/rebalance_behavior_curriculum.py`
  - adds `support_procedure_recovery` profile (55/25/8/4/5/3)
- `scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1`
  - defaults to v3-large corpus
  - defaults to corrective behavior profile
  - runs support-retrieval audit before model evaluation
- `scripts/phase_9_student_distillation/run_phase_9.ps1`
  - defaults to v3-large corpus and corrective profile
- `scripts/phase_9_student_distillation/run_evidence_preference_refinement.ps1`
  - defaults to v3-large preference corpus
- The following Phase 9 helpers now identify `phase9_large_corrective_curriculum_v1_7_5` as the current revision while retaining explicit compatibility for older artifacts where required:
  - `build_overfit_probe.py`
  - `check_grounded_recovery_ladder.py`
  - `evaluate_rag.py`
  - `generate_teacher_data.py`
  - `merge_training_data.py`
  - `select_grounded_sft_candidate.py`
  - `train_student.py`

## Modified Phase 9B pipeline

- `scripts/phase_9b_memory_reconstruction/phase9b_common.py`
  - accepts deterministic auto-verified curated rows
  - v1.7.5 pipeline revision
- `scripts/phase_9b_memory_reconstruction/build_reconstruction_dataset.py`
  - consumes v2-large authored splits and curated preferences
- `scripts/phase_9b_memory_reconstruction/validate_memory_reconstruction_corpus.py`
  - defaults to v2-large corpus
- `scripts/phase_9b_memory_reconstruction/verify_memory_reconstruction_evaluator.py`
  - defaults to v2-large corpus
- `scripts/phase_9b_memory_reconstruction/run_phase_9b.ps1`
  - defaults to the v3-large Phase 9 corpus and v2-large reconstruction corpus

## Phase 10 / Phase 11 integration

- `scripts/phase_10_student_architecture/run_phase_10.ps1` defaults updated to v1.7.5 RAG/reconstruction assets and Phase 9 artifact root.
- `scripts/phase_11_production_rag/run_phase_11.ps1` defaults updated to v1.7.5 RAG/reconstruction assets and Phase 9 artifact root.

## Tests

- `tests/test_v175_large_corrective_datasets.py`
  - validates counts, distributions, corrective mix, reconstruction state coverage, decision coverage, and runner defaults.

## Documentation

- `PHASE_9_9B_LARGE_CORRECTIVE_DATASETS_v1.7.5.md`
- `CHANGED_FILES_v1.7.5.md`
- `RELEASE_VERIFICATION_v1.7.5.md`
- root and phase-data READMEs updated for v1.7.5.
