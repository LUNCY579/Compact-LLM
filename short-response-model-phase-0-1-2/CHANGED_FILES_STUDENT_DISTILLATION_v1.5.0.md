# Changed files — Phase 9 student distillation v1.5.0

This release appends a student-distillation and RAG-specialization workflow to the existing official Phase 0–8 teacher pipeline. It does not alter teacher checkpoints, prior artifacts, existing phase gates, or the governing relative-opportunity policy.

## Modified repository-level files

- `README.md` — announces Phase 9, documents the teacher/student boundary, and adds entry points.
- `Phases.md` — adds Phase 9 responsibility and handoff boundaries.
- `POST_PHASE_3_EXECUTION_GUIDE.md` — appends runnable Bootstrap and Pilot commands.
- `verify_post_phase_3.ps1` — verifies Evaluator v4.5 and includes the Phase 9 runner.

## Added master documentation

- `STUDENT_DISTILLATION_PIPELINE.md` — complete end-to-end workflow and research contract.
- `CHANGED_FILES_STUDENT_DISTILLATION_v1.5.0.md` — this file.
- `RELEASE_VERIFICATION_STUDENT_DISTILLATION_v1.5.0.md` — validation evidence and limitations.
- `HOTFIX_MANIFEST_STUDENT_DISTILLATION_v1.5.0.json` — hashes and install list for the hotfix package.

## Added configuration and sample data

- `configs/phase_9_student_distillation.json`
- `data/phase_9_student_distillation/README.md`
- `data/phase_9_student_distillation/knowledge.sample.jsonl`
- `data/phase_9_student_distillation/procedures.sample.jsonl`
- `data/phase_9_student_distillation/prompts.sample.jsonl`

## Added Phase 9 implementation

- `scripts/phase_9_student_distillation/__init__.py`
- `scripts/phase_9_student_distillation/README.md`
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`
- `scripts/phase_9_student_distillation/FILE_RESPONSIBILITIES.md`
- `scripts/phase_9_student_distillation/phase9_common.py`
- `scripts/phase_9_student_distillation/create_student.py`
- `scripts/phase_9_student_distillation/build_knowledge_index.py`
- `scripts/phase_9_student_distillation/generate_teacher_data.py`
- `scripts/phase_9_student_distillation/build_pilot_dataset.py`
- `scripts/phase_9_student_distillation/calibrate_evidence_gate.py`
- `scripts/phase_9_student_distillation/train_student.py`
- `scripts/phase_9_student_distillation/generate_on_policy_data.py`
- `scripts/phase_9_student_distillation/evaluate_rag.py`
- `scripts/phase_9_student_distillation/check_phase_9_gate.py`
- `scripts/phase_9_student_distillation/run_phase_9.ps1`

## Added tests

- `tests/test_phase_9_student_distillation.py`

## Manual installation from the hotfix ZIP

Extract the hotfix ZIP over the repository root while preserving paths. Existing Phase 0–8 files not listed in the manifest are unchanged. Re-run:

```powershell
python -m compileall -q .\scripts .\tests
python -m pytest -q
python .\scripts\common\verify_quality_evaluation_v4_5.py
python .\scripts\common\audit_relative_opportunity_policy.py
```

## Post-release Phase 9 response-contract compatibility correction

The repository version label remains unchanged. The following files were updated after a real Bootstrap run exposed a schema mismatch between the Phase 9 sample records and Evaluator v4.5:

- `scripts/phase_9_student_distillation/phase9_common.py` — normalizes legacy compact requirements and forbidden claims into explicit Evaluator-v4.5 objects.
- `scripts/phase_9_student_distillation/build_pilot_dataset.py` — reuses the normalizer and emits object-form abstention requirements.
- `data/phase_9_student_distillation/prompts.sample.jsonl` — migrated to canonical object-form response contracts.
- `tests/test_phase_9_student_distillation.py` — adds contract conversion and end-to-end evaluator regressions.
- `PHASE_9_RESPONSE_CONTRACT_SCHEMA_HOTFIX.md` — documents the failure, correction, and safe resume command.
- `HOTFIX_MANIFEST_PHASE9_RESPONSE_CONTRACT.json` — install list, hashes, and verification evidence.
