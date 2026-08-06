# Phase 9 response-contract schema hotfix

## Failure fixed

Phase 9's original sample prompt records used compact response-contract entries:

```json
{
  "requirements": ["August 17, 2026"],
  "forbidden_claims": ["tomorrow"]
}
```

Evaluator v4.5 intentionally expects explicit requirement and prohibition objects.
During teacher-candidate generation, `validate_response_contract()` attempted to
call `.get()` on the string requirement and failed with:

```text
AttributeError: 'str' object has no attribute 'get'
```

The teacher checkpoint, generated response, and Phase 8 gate were not the cause.

## Corrections

- `phase9_common.py` now normalizes Phase 9 response contracts at the data-ingestion boundary.
- Legacy string requirements are converted into explicit Evaluator-v4.5 requirement objects.
- Legacy `forbidden_claims` strings are converted into explicit `prohibitions` objects.
- Explicit object-form contracts remain supported and are validated.
- `build_pilot_dataset.py` now uses the same normalizer and creates the deterministic abstention requirement in object form.
- `prompts.sample.jsonl` now uses the canonical Evaluator-v4.5 object schema directly.
- Regression tests cover legacy conversion, canonical sample records, abstention contracts, and full `evaluate_response()` execution.

The shared Evaluator v4.5 was not weakened or changed. Compatibility conversion is local to Phase 9.

## Resume after the failed Bootstrap

The student initialization and SQLite index were already completed successfully. Rerun with reuse enabled:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Bootstrap `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReuseExisting
```

`-ReuseExisting` skips the valid student checkpoint and knowledge index, then resumes teacher-data generation. If a partial `teacher_candidates.jsonl` exists, generation resumes by record ID.

## Verification

- Python compilation: PASS
- Full repository tests: 144 passed
- Evaluator v4.5 calibration: 639/639 passed
- Relative-opportunity policy audit: PASS
