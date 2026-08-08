# Phase 9 SFT Selection Gate Hotfix — v1.7.7

## Problem

The Phase 9 grounded-recovery ladder intentionally permits Stage B to be a weak untrained warm-student baseline. In affected v1.7.5/v1.7.6 runs, that baseline could contain prompt/decode-integrity failures while still providing a useful numeric Oracle capability reference.

The old selector combined Stage-B integrity and coverage issues into one `initialization_issues` list. Any item in that list caused every Stage-D candidate to receive `initialization_reference_invalid`. This created a policy mismatch: the ladder could proceed beyond Stage B, but the Stage-D selector could never choose a trained candidate, even when that candidate had clean integrity and strong Oracle performance.

## v1.7.7 policy

Stage-B diagnostics are split into two classes:

- `initialization_integrity_warnings`: prompt/decode-integrity findings from the untrained baseline. These are **non-blocking** for Stage-D selection.
- `initialization_blocking_issues`: behavior-coverage failures in the initialization reference. These remain blocking because missing required denominators make the capability comparison unreliable.

The compatibility field `initialization_issues` remains present as the aggregate diagnostic list, but it is no longer itself interpreted as a veto.

## Candidate checks that remain strict

A trained candidate is still invalid if any of the following holds:

- its own prompt/decode integrity fails;
- its own required behavior coverage is missing;
- it is degenerate on supported/abstention behavior;
- a required behavior has zero successful cases;
- its Oracle capability score does not improve over the initialization numeric baseline.

No quality rule on the trained candidate was weakened by this hotfix.

## Existing-run migration

`run_phase_9_grounded_recovery.ps1` now treats selection JSON/Markdown as derived artifacts. With `-ReuseExisting`, it re-runs the selector against existing Stage-D summaries instead of reusing stale selection output. The same rule is applied to the final DEV refinement selection.

Therefore an affected run does not need SFT retraining. Re-run the same Stage-D/Full command with the same output root and `-ReuseExisting`; the expensive training/evaluations are reused and the selection is refreshed under the v1.7.7 policy.

For a selection-only repair, run:

```powershell
python .\scripts\phase_9_student_distillation\reselect_existing_grounded_sft.py `
  --run-root .\artifacts\phase_9_grounded_recovery_v1_7_5_qwen
```

This rewrites only `warm_sft_sweep/selection.json` and `SELECTION_REPORT.md` from the existing evaluation summaries.

## Expected behavior for the reported v1.7.5 Qwen case

Given a Stage-B numeric baseline around 0.25 with decode-integrity warnings and clean Stage-D candidates around 0.66, 0.85, and 1.0, the integrity warnings alone no longer invalidate the frontier. Assuming the on-disk summaries match those stated metrics and contain no additional candidate/coverage defects, the selector should rank `lr1e4_s1200` first.
