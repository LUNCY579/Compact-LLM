> **Historical/superseded:** Evaluator v4.0 is authoritative. Do not use the commands or quality labels in this document for new official evidence. See `EVALUATION_V4_MIGRATION.md`.

# Evaluation v3.1 Migration Guide

## What changed

Evaluator v3.1 keeps the permissive conversational policy introduced in v3.0, but adds explicit semantic coherence, prompt relevance, intent fulfillment, and required-meaning preservation. Nonempty word salad can no longer inherit perfect task adequacy merely because it avoids a narrow mechanical corruption rule.

Official phase decisions still use `scripts/common/opportunity_evaluation.py` rather than fixed quality floors. The detailed design is in `EVALUATION_V3_1_SEMANTIC_INTEGRITY_HOTFIX.md`.

## Existing checkpoints

A scoring-policy change does not require checkpoint retraining. Existing generated replies can be rescored with:

```powershell
python .\scripts\common\rescore_quality_artifacts.py `
  --benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  --results .\path\to\generation\results.jsonl `
  --output-dir .\path\to\generation_quality_v3_1
```

The tool preserves generation text and runtime fields while recomputing case-level quality evidence and `summary.json`.

## Phase 2B tree migration

Non-destructive review:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v3_1.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  -ReviewOnly
```

Official in-place migration and replay preparation:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v3_1.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl
```

Equivalent Python command:

```powershell
python .\scripts\phase_2_layer_selection\rescore_phase_2b_artifacts.py `
  --run-root .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  --benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  --in-place `
  --prepare-resume
```

The utility creates timestamped backups, re-scores stored replies without model inference, rebuilds full-reference output deviations, writes evidence-lineage manifests, and archives superseded parent-relative decisions and catalogs. Then rerun the exact original Phase 2B command with `-Resume`.

Stored generations, teacher-forced NLL, footprint evidence, and compatible checkpoints are reused while the corrected frontier follows the same parent lineage. If v3.1 changes the accepted parent at a stage, only downstream work from that divergence must be regenerated.

## Resume safety

Do not mix 3.0 and 3.1 summaries in one official comparison. The current code fingerprints evaluator configuration, opportunity policy, benchmark, parent model, dropped layer, recovery settings, and retained layer lineage. Evidence with mismatched provenance is removed before reuse.
