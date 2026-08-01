# Evaluation v3 Migration Guide

## What changed

Evaluator v3 is conversationally permissive and continuous. Official phase decisions use `scripts/common/opportunity_evaluation.py` rather than fixed quality floors.

## Existing checkpoints

A scoring-policy change does not require retraining a checkpoint. Existing generated replies can be rescored with:

```powershell
python .\scripts\common\rescore_quality_artifacts.py `
  --benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  --results .\path\to\generation\results.jsonl `
  --output-dir .\path\to\generation_v3
```

The tool preserves generation text and recomputes case-level quality fields and `summary.json`.

## Phase 2B tree migration

First create non-destructive sibling evidence for inspection:

```powershell
python .\scripts\phase_2_layer_selection\rescore_phase_2b_artifacts.py `
  --run-root .\artifacts\phase_2b_quality_v2_thorough_20260731 `
  --benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl
```

When ready to continue the official run, migrate in place and prepare resume:

```powershell
python .\scripts\phase_2_layer_selection\rescore_phase_2b_artifacts.py `
  --run-root .\artifacts\phase_2b_quality_v2_thorough_20260731 `
  --benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  --in-place `
  --prepare-resume
```

The utility creates timestamped backups, rescoring stored replies without model
inference, rebuilds output-deviation evidence against the rescored full
reference, and archives the superseded derived catalog. Then rerun the exact
original Phase 2B command with `-Resume`. Model checkpoints, NLL, footprints,
and generations are reused; opportunity assessments and the progressive parent
are recalculated under Evaluator v3 before later stages continue.

## Resume safety

Do not use `-Resume` to mix v2 summaries with v3 decisions. The current code fingerprints evaluator version and requires current-schema summaries for relative comparison.
