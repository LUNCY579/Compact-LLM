# Changed files — Phase 9 hybrid + Phase 10/11 synthesis

## Merge strategy

`Output(10).zip` was treated as the authoritative Phase 0–9 base. The Phase 10/11 implementation was added on top of that base. No older Phase 9 script replaced the newer hybrid-suite implementation.

## Added Phase 10 files

- `configs/phase_10_student_architecture.json`
- `data/phase_10_student_architecture/README.md`
- `scripts/phase_10_student_architecture/__init__.py`
- `scripts/phase_10_student_architecture/README.md`
- `scripts/phase_10_student_architecture/EXECUTION_MANUAL.md`
- `scripts/phase_10_student_architecture/FILE_RESPONSIBILITIES.md`
- `scripts/phase_10_student_architecture/check_phase_10_gate.py`
- `scripts/phase_10_student_architecture/create_structural_candidate.py`
- `scripts/phase_10_student_architecture/finalize_phase_10_selection.py`
- `scripts/phase_10_student_architecture/finalize_phase_9_selection.py`
- `scripts/phase_10_student_architecture/phase10_common.py`
- `scripts/phase_10_student_architecture/profile_student_structure.py`
- `scripts/phase_10_student_architecture/resolve_phase_10_student.py`
- `scripts/phase_10_student_architecture/resolve_phase_9_student.py`
- `scripts/phase_10_student_architecture/run_phase_10.ps1`
- `scripts/phase_10_student_architecture/select_recovery_candidates.py`
- `tests/test_phase_10_student_architecture.py`

## Added Phase 11 files

- `configs/phase_11_production_rag.json`
- `data/phase_11_production_rag/README.md`
- `scripts/phase_11_production_rag/__init__.py`
- `scripts/phase_11_production_rag/README.md`
- `scripts/phase_11_production_rag/EXECUTION_MANUAL.md`
- `scripts/phase_11_production_rag/FILE_RESPONSIBILITIES.md`
- `scripts/phase_11_production_rag/benchmark_production_runtime.py`
- `scripts/phase_11_production_rag/build_knowledge_snapshot.py`
- `scripts/phase_11_production_rag/calibrate_answerability.py`
- `scripts/phase_11_production_rag/check_phase_11_gate.py`
- `scripts/phase_11_production_rag/package_phase_11_release.py`
- `scripts/phase_11_production_rag/phase11_common.py`
- `scripts/phase_11_production_rag/quantize_student.py`
- `scripts/phase_11_production_rag/run_phase_11.ps1`
- `scripts/phase_11_production_rag/serve_production_rag.py`
- `tests/test_phase_11_production_rag.py`

## Added repository documentation

- `PHASE_10_11_ARCHITECTURE_REVIEW.md`
- `PHASE_10_11_WORKFLOW.md`
- `CHANGED_FILES_PHASE_10_11.md`
- `RELEASE_VERIFICATION_PHASE_10_11.md`
- `PACKAGE_MANIFEST_PHASE_10_11.json`
- `CHANGED_FILES_PHASE_9_HYBRID_PHASE_10_11_MERGE.md`

## Modified during synthesis

- `README.md` — combines the hybrid Phase 9 suite with Phase 10/11 handoffs.
- `Phases.md` — defines Phase 9 hybrid assessment and official Phase 10/11 boundaries.
- `POST_PHASE_3_EXECUTION_GUIDE.md` — adds hybrid Phase 9 and exact candidate-key Phase 10 commands.
- `verify_post_phase_3.ps1` — verifies Phase 2B through Phase 11 runners.
- `scripts/phase_10_student_architecture/finalize_phase_9_selection.py` — removes the legacy fixed-name restriction and validates arbitrary assessment keys.
- `scripts/phase_10_student_architecture/README.md`
- `scripts/phase_10_student_architecture/EXECUTION_MANUAL.md`
- `scripts/phase_10_student_architecture/FILE_RESPONSIBILITIES.md`
- `PHASE_10_11_WORKFLOW.md`
- `PHASE_10_11_ARCHITECTURE_REVIEW.md`
- `tests/test_phase_10_student_architecture.py`
- `CHANGED_FILES_PHASE_10_11.md`
- `RELEASE_VERIFICATION_PHASE_10_11.md`
- `PACKAGE_MANIFEST_PHASE_10_11.json`

## Compatibility result

The Phase 10 handoff now supports both:

```text
Legacy single-suite keys:
  sft
  kd
  on_policy
  preference

Hybrid profile-qualified keys:
  sft_warm_344m
  kd_warm_344m
  sft_narrow_100m
  kd_narrow_100m
  ...and any other exact key recorded by Phase 9
```

Narrow random-initialized candidates remain research controls unless a human reviewer deliberately supplies the explicit promotion acknowledgement.
