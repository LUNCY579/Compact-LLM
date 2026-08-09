# Changed Files — v2.1 Procedural Transfer Recovery

## New research/data assets

- `PROCEDURAL_RECOVERY_V2_1.md` — full postmortem, recovery rationale, experiment order, commands, and interpretation policy.
- `configs/phase_9_procedural_recovery_v21.json` — versioned recovery policy and fixed default recipe.
- `data/phase_9_student_distillation/procedural_recovery_v21/` — 6,120-row deterministic procedural recovery corpus with 4,648/736/736 topic-disjoint train/dev/test rows, 2,880 knowledge documents, 1,560 procedure records, hard-collision and preservation training views, hashes, and manifest.
- `reports/PHASE_9_PROCEDURAL_RETRIEVAL_COLLISION_AUDIT_v2.1.{json,md}` — model-free postmortem of the historical 150 answerable procedural test cases.
- `reports/PHASE_9_V175_TRAIN_PROCEDURAL_GOLD_AUDIT_v2.1.json` — records 18 historical v1.7.5 latency-tie gold inconsistencies without silently rewriting the old corpus.
- `reports/PHASE_9_V21_TRAIN_PROCEDURAL_GOLD_AUDIT.json` — verifies zero corresponding v2.1 violations.

## New Phase 9 tooling

- `scripts/phase_9_student_distillation/generate_procedural_recovery_v21.py`
  - generates 14 procedure families;
  - creates target + near-identical sibling evidence;
  - programmatically computes gold answers;
  - prevents implicit ties;
  - builds train-only non-procedure preservation mix;
  - writes hashes/manifests.
- `scripts/phase_9_student_distillation/audit_procedural_retrieval_collisions.py`
  - measures support rank, equivalent procedure recall, sibling collisions, and query-aware selection effects without model inference.
- `scripts/phase_9_student_distillation/audit_procedural_gold_consistency_v21.py`
  - detects the concrete historical latency-tie supervision defect and can fail closed on new recovery data.
- `scripts/phase_9_student_distillation/diagnose_procedural_transfer_gap.py`
  - pairs Oracle and Real outputs and classifies retrieval miss, prompt-budget loss, procedure miss, collision/evidence binding, Oracle skill failure, or residual generation/binding failure.
- `scripts/phase_9_student_distillation/compare_procedural_recovery_v21.py`
  - explicitly separates retrieval-policy gain from learned training gain and reports preservation cost.
- `scripts/phase_9_student_distillation/run_phase_9_procedural_recovery_v21.ps1`
  - end-to-end v2.1 DEV-first workflow;
  - same-weight legacy/query-aware A/B;
  - two-stage focused rank-16 recovery;
  - optional sealed test only after DEV review.

## Retrieval/runtime changes

- `scripts/phase_9_student_distillation/phase9_common.py`
  - adds explicit-locator extraction;
  - adds `select_retrieved_passages(..., policy="query_aware_v21")`;
  - suppresses explicit locator mismatches when exact matches exist;
  - deduplicates equivalent procedure rules;
  - keeps historical `legacy` policy available.
- `scripts/phase_9_student_distillation/evaluate_rag.py`
  - adds `--retrieval-selection-policy` and records it in evaluation summaries.
- `scripts/phase_9_student_distillation/calibrate_evidence_gate.py`
  - calibrates on the same selected evidence policy used at runtime.
- `scripts/phase_9_student_distillation/audit_rag_retrieval_coverage.py`
  - audits support recall after the selected evidence policy.
- `scripts/phase_11_production_rag/calibrate_answerability.py`
- `scripts/phase_11_production_rag/benchmark_production_runtime.py`
- `scripts/phase_11_production_rag/serve_production_rag.py`
  - propagate `query_aware_v21` so production does not reintroduce sibling collisions.
- `configs/phase_11_production_rag.json`
  - schema 1.3;
  - records `selection_policy: query_aware_v21` and recalibration requirement.
- `scripts/phase_9_student_distillation/run_phase_9.ps1`
- `scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1`
  - official Phase 9 wrappers now default to `query_aware_v21` while retaining an explicit `legacy` override;
  - reuse checks reject stale evaluations produced under a different retrieval-selection policy.
- `configs/phase_9_student_distillation.json`
  - records the v2.1 official-runner retrieval default.

## Tests/documentation

- `tests/test_phase_9_procedural_recovery_v21.py` — selector, deduplication, corpus validity, preservation isolation, transfer diagnosis, and Phase 11 policy regression tests.
- `README.md` — v2.1 entry point and recovery-first ordering.
- `Phases.md` — Phase 9 responsibility updated to include v2.1 transfer diagnosis/recovery before Phase 10/11 optimization.
- `RELEASE_VERIFICATION_v2.1.md` — package verification evidence.
- `PACKAGE_MANIFEST_v2.1.json` — package file inventory/hashes generated at release time.

## Deliberately not changed

- Historical v1.7.5 corpus rows and Phase 9 reports are not silently rewritten. Their observed defects remain reproducible research evidence.
- No broad new LoRA sweep is added.
- No automatic mass retraining of the narrow/other student tracks is added.
- v2.0 structural optimization remains downstream and should resume only after the repaired student is approved.
