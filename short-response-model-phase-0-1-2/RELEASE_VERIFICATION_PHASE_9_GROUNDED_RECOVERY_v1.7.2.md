# Release verification — Phase 9 grounded recovery v1.7.2

## Scope

This release starts from the complete v1.7.1 Phase 9 RAG-integrity repository and preserves its prompt budgeting, decode provenance, teacher preflight, bounded near-zero arithmetic, task-contract quality cost, and stale-artifact protections.

v1.7.2 changes the official recovery order. It first proves one inherited warm 344M student through teacher Oracle evaluation, an untrained initialization baseline, a tiny deliberate overfit wiring test, and a compact warm-SFT sweep. Real retrieval, progressive KD, and broad narrow-student comparison remain blocked until the preceding capability evidence exists.

It does not claim that a GPU training run has completed or that Phase 9 has passed.

## Automated verification

Executed from the repository root on 2026-08-06:

```text
python -m compileall -q scripts tests
pytest -q
```

Result: `194 passed`.

```text
python scripts/common/verify_quality_evaluation_v4_5.py
```

Result: `639/639` calibration cases passed across 31 reporting categories and 63 scored labels; `150/150` real-collapse cases detected.

```text
python scripts/common/audit_relative_opportunity_policy.py
```

Result: policy audit passed; no absolute promotion floors were found.

Additional checks passed:

- all referenced PowerShell script/config/data paths exist;
- quote-aware delimiter balance passed for both Phase 9 runners;
- all new Python CLI `--help` contracts executed;
- deterministic 45/25/10/10/5/5 curriculum rebalancing smoke path executed;
- 18-row six-behavior overfit-probe construction executed;
- SFT selector CLI selected the valid non-degenerate candidate in a synthetic packet;
- ladder CLI unlocked Real RAG/KD/architecture comparison only after overfit and Oracle improvement evidence;
- missing overfit behavior buckets are rejected;
- procedure correctness and numerator/denominator evidence are present in Phase 9 summaries and reports.

## Environment limitations

- PowerShell was not installed in the verification container. The `.ps1` wrappers were statically inspected, reference-checked, delimiter-checked, and covered by regression tests, but were not launched here.
- The container did not have the user's CUDA model checkpoints or full training environment. No teacher generation, SFT/KD training, student checkpoint, benchmark result, or human PASS was fabricated.
- The tiny-overfit numerical criteria are explicitly scoped to wiring verification; they are not universal production quality floors.

## Release status

Code/package status: **READY FOR USER-SIDE GROUNDED-RECOVERY EXECUTION**.

Model-selection status: **PHASE 9 REMAINS BLOCKED** until the teacher Oracle path, tiny overfit wiring proof, full Oracle SFT improvement, Real-RAG evaluation, and explicit human gate all succeed for a named artifact.
