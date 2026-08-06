# Changed files — Phase 9 RAG integrity v1.7.1

## Added

- `scripts/phase_9_student_distillation/phase9_prompting.py` — token-aware evidence packing and prompt-integrity evidence.
- `scripts/phase_9_student_distillation/audit_phase_9_results.py` — raw-result and root-cause audit.
- `scripts/phase_9_student_distillation/run_phase_9_diagnostics.ps1` — audit wrapper for existing artifact trees.
- `tests/test_phase_9_rag_integrity_hotfix.py` — zero-reference, prompt-integrity, task-cost, and preflight regressions.
- `PHASE_9_RAG_INTEGRITY_HOTFIX_v1.7.1.md` — governing change description.
- `RELEASE_VERIFICATION_PHASE_9_RAG_INTEGRITY_v1.7.1.md` — verification record.
- `PACKAGE_MANIFEST_PHASE_9_RAG_INTEGRITY_v1.7.1.json` — file hashes for the complete release tree.

## Modified

- `scripts/common/opportunity_evaluation.py` — bounded near-zero comparison policy.
- `scripts/phase_9_student_distillation/evaluate_rag.py` — schema 1.2 prompt/decode evidence, prompted-support recall, and injection resistance.
- `scripts/phase_9_student_distillation/generate_teacher_data.py` — teacher prompt budgeting/provenance and rejection of legacy resume rows that lack the v1.7.1 prompt contract.
- `scripts/phase_9_student_distillation/train_student.py` — safe structured prompt rebuild; no tail truncation.
- `scripts/phase_9_student_distillation/phase9_metrics.py` — Phase 9 task-contract deficiency.
- `scripts/phase_9_student_distillation/check_phase_9_gate.py` — combined generic/functional opportunity decision and clearer safety-cap wording.
- `scripts/phase_9_student_distillation/run_phase_9.ps1` — schema invalidation, evidence-budget-aware cache keys, and mandatory teacher preflight.
- `configs/phase_9_student_distillation.json` — v1.2 policy declarations.
- `README.md`, `STUDENT_DISTILLATION_PIPELINE.md`, and `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md` — updated operation guidance.
