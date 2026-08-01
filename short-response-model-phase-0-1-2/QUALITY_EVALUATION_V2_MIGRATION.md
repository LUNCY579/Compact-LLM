# Migrating Existing Runs to Quality Evaluation v2.0

## What must be repeated

Repeat **evaluation and gates**, not training, for every checkpoint that will be compared or promoted. A checkpoint's weights remain valid; its old aggregate evidence does not.

Recommended order:

1. Finish any currently active long-running job under one stable code version.
2. Back up its artifacts and reports.
3. Apply this update.
4. Re-evaluate the retained reference and finalist checkpoints with `evaluate_model.py`.
5. Re-run the relevant phase gate or playground report.
6. Continue downstream only from v2 evidence.

Do not mix legacy and v2 summaries in one ranking. Resume-enabled Phase 2B and playground runs detect stale generation summaries and rebuild them, but preserving a clean separate output root is the strongest research record.

## Read-only evidence audit

```powershell
python .\scripts\common\audit_quality_evidence.py `
  .\artifacts `
  .\playground_artifacts `
  --report .\reports\QUALITY_EVIDENCE_AUDIT.md `
  --strict
```

`--strict` exits nonzero when stale, incomplete, or malformed evidence is found.

## Evaluator smoke verification

```powershell
python .\scripts\common\verify_quality_evaluation.py `
  --report .\reports\QUALITY_EVALUATOR_VERIFICATION.md
```

This command needs no GPU or model. It checks that healthy examples pass and diverse collapse examples are rejected.

## Phase 1 reminder

Re-evaluating Phase 1 does not require retraining the dense SFT adapter. Use new output directories if you want to preserve the historical reports. Do not overwrite a previously human-approved gate unless you intend to review and approve the new gate.

## Playground reminder

The playground report is exploratory. Version 1.1 now generates automatic eligibility and reasons, but it still does not modify official Phase 2B gates or catalogs.
