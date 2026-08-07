# Changed Files — Phase 9–11 RAG Reasoning Integration v1.7.3

This list is relative to the uploaded `short-response-v1.7.2-revised.zip`. Cache/bytecode files are intentionally excluded from the release.

## New integration documentation

- `PHASE_9_11_RAG_REASONING_INTEGRATION_v1.7.3.md`
- `CHANGED_FILES_PHASE_9_11_RAG_REASONING_v1.7.3.md`
- `RELEASE_VERIFICATION_PHASE_9_11_RAG_REASONING_v1.7.3.md`

## New canonical corpus

- `data/phase_9_student_distillation/rag_reasoning_v2/README.md`
- `data/phase_9_student_distillation/rag_reasoning_v2/knowledge.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/procedures.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/prompts.all.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/teacher_candidates.reviewed.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/splits/train.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/splits/dev.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/splits/test.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/reviewed_splits/train.reviewed.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/reviewed_splits/dev.reviewed.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/reviewed_splits/test.reviewed.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/preferences/train.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/preferences/dev.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/evaluator/calibration_train_dev.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/evaluator/candidate_judgments.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/evaluator/heldout_test_cases.jsonl`
- `data/phase_9_student_distillation/rag_reasoning_v2/manifests/dataset_manifest.json`
- `data/phase_9_student_distillation/rag_reasoning_v2/manifests/failure_taxonomy.json`
- `data/phase_9_student_distillation/rag_reasoning_v2/manifests/file_hashes.json`
- `data/phase_9_student_distillation/rag_reasoning_v2/manifests/validation_report.json`

## New corpus/integration utilities

- `scripts/phase_9_student_distillation/prepare_rag_reasoning_corpus.py`
  - deterministically normalizes the source corpus into the repository contract;
  - repairs the two objectively resolvable procedure rows;
  - enriches source/current-state metadata;
  - regenerates splits/preferences/evaluator assets and hashes.
- `scripts/phase_9_student_distillation/validate_rag_reasoning_corpus.py`
  - verifies hashes, counts, split disjointness, decision ontology, reviewed targets, and required files.
- `scripts/phase_9_student_distillation/run_evidence_preference_refinement.ps1`
  - optional DPO refinement after a proven SFT candidate; preference alternatives never become ordinary SFT labels.
- `tests/test_phase_9_11_rag_reasoning_v173.py`
  - cross-phase corpus, split, stale/conflict/procedure, reconstruction, and sealed-test regression tests.

## Modified Phase 9 configuration and workflow

- `configs/phase_9_student_distillation.json`
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`
- `scripts/phase_9_student_distillation/FILE_RESPONSIBILITIES.md`
- `scripts/phase_9_student_distillation/phase9_common.py`
- `scripts/phase_9_student_distillation/phase9_metrics.py`
- `scripts/phase_9_student_distillation/build_pilot_dataset.py`
- `scripts/phase_9_student_distillation/evaluate_rag.py`
- `scripts/phase_9_student_distillation/generate_teacher_data.py`
- `scripts/phase_9_student_distillation/merge_training_data.py`
- `scripts/phase_9_student_distillation/train_student.py`
- `scripts/phase_9_student_distillation/build_overfit_probe.py`
- `scripts/phase_9_student_distillation/check_grounded_recovery_ladder.py`
- `scripts/phase_9_student_distillation/rebalance_behavior_curriculum.py`
- `scripts/phase_9_student_distillation/select_grounded_sft_candidate.py`
- `scripts/phase_9_student_distillation/run_phase_9.ps1`
- `scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1`

Major effects:

- corpus defaults point to `rag_reasoning_v2`;
- authored 392/67/71 split can be preserved and validated;
- procedure answerability is normalized;
- safe conflict abstention is functionally valid;
- explicit conflict acknowledgment is retained as a diagnostic;
- superseded/untrusted sources are removed before conflict resolution;
- trust/freshness metadata is exposed to grounded prompts;
- foundation rehearsal is deduplicated against dev/test holdouts;
- expanded specialized metrics have real denominators;
- both the grounded-recovery and general Phase 9 runners are dev-first, and the test set is sealed until one final candidate is frozen;
- Phase 9 evaluation/training schema is versioned as v1.7.3.

## Modified Phase 9B memory reconstruction

- `configs/phase_9b_memory_reconstruction.json`
- `scripts/phase_9b_memory_reconstruction/EXECUTION_MANUAL.md`
- `scripts/phase_9b_memory_reconstruction/phase9b_common.py`
- `scripts/phase_9b_memory_reconstruction/build_reconstruction_candidates.py`
- `scripts/phase_9b_memory_reconstruction/build_reconstruction_dataset.py`
- `scripts/phase_9b_memory_reconstruction/run_phase_9b.ps1`
- `MEMORY_RECONSTRUCTION_PIPELINE.md`

Major effects:

- Phase 9B consumes the shared corpus by default;
- source-state applicability order is untrusted → superseded → validity → remaining conflict;
- stale/current cases become ADAPT where appropriate instead of false CONFLICT;
- authored split intent is preserved when building reconstruction datasets;
- human review remains mandatory before reconstruction training.

## Modified Phase 10

- `configs/phase_10_student_architecture.json`
- `scripts/phase_10_student_architecture/select_recovery_candidates.py`
- `scripts/phase_10_student_architecture/run_phase_10.ps1`
- `PHASE_10_11_WORKFLOW.md`

Major effects:

- v1.7.3 Phase 9 dataset/retrieval artifacts are default inputs;
- raw/recovered/challenger selection happens on dev;
- one final architecture is frozen on dev;
- only the reference and frozen candidate are evaluated on sealed test;
- test-time candidate swapping is prohibited by workflow contract.

## Modified Phase 11

- `configs/phase_11_production_rag.json`
- `scripts/phase_11_production_rag/calibrate_answerability.py`
- `scripts/phase_11_production_rag/benchmark_production_runtime.py`
- `scripts/phase_11_production_rag/run_phase_11.ps1`

Major effects:

- shared knowledge/procedure corpus is the default snapshot source;
- answerability calibration recognizes answerable procedures correctly;
- calibration uses dev and the final benchmark uses sealed test;
- production metrics use functional answer correctness rather than a generic score cutoff;
- safe conflict, procedure, evidence-calibration, stale-precedence, injection, social, and denominator metrics are reported separately.

## Modified project-level documentation

- `README.md`
- `Phases.md`
- `MEMORY_RECONSTRUCTION_PIPELINE.md`
- `PHASE_10_11_WORKFLOW.md`
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`
- `scripts/phase_9b_memory_reconstruction/EXECUTION_MANUAL.md`

## Updated existing regression tests

- `tests/test_phase_9_grounded_recovery_v172.py`
- `tests/test_phase_9_rag_integrity_hotfix.py`

The older tests remain as regression coverage but were updated to accept the v1.7.3 schema/revision contract.

## Removed from release

No source/data files from the uploaded v1.7.2 repository were intentionally removed. `.pytest_cache`, `__pycache__`, and `.pyc` artifacts are excluded from the packaged release.
