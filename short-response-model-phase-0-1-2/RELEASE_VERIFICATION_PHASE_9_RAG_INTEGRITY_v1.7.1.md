# Release verification — Phase 9 RAG integrity v1.7.1

## Scope

This release updates the full repository extracted from `Output(20260806-210909).zip`. It corrects Phase 9 prompt construction, raw-result auditing, RAG-specific opportunity costs, zero-reference arithmetic, teacher preflight, and stale-artifact reuse behavior. It does not claim that a new student model has been trained or that Phase 9 has passed.

## Automated verification

Executed from the repository root on 2026-08-06:

```text
python -m compileall -q scripts tests
pytest -q
```

Result: `186 passed`.

```text
python scripts/common/verify_quality_evaluation_v4_5.py
```

Result: `639/639` calibration cases passed across 31 reporting categories and 63 scored labels; `150/150` real-collapse cases detected.

```text
python scripts/common/audit_relative_opportunity_policy.py
```

Result: policy audit passed; no absolute promotion floors were found.

The Phase 9 diagnostic CLI was also imported and its `--help` contract executed successfully. Regression tests cover bounded zero-reference comparisons, preservation of the grounding contract and final question, removal of tail truncation, full proportional cost for complete core-RAG failure, schema invalidation, evidence budgeting, and mandatory teacher preflight.

## Environment limitations

- PowerShell was not installed in the verification container, so the `.ps1` wrappers were statically inspected and regression-tested but not launched here.
- The container did not have the full GPU/Transformers training environment. No CUDA training, teacher generation, candidate evaluation, or model-quality claim was fabricated.
- The user must rerun schema-1.2 evaluations on their Windows/CUDA environment. Existing schema-1.1 outputs remain historical evidence but cannot verify prompt or decode integrity.

## Release status

Code/package status: **READY FOR USER-SIDE DIAGNOSTIC AND EVALUATION RERUN**.

Model-selection status: **PHASE 9 STILL BLOCKED** until a usable teacher path is demonstrated, corrected candidates are evaluated, and the human gate records an explicit PASS.
