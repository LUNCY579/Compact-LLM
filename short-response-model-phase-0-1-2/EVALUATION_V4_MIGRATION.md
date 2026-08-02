# Evaluator v4.0 Phase 2B migration and resume guide

## Purpose

This migration reuses stored generated replies, teacher-forced evidence, footprint measurements, and compatible checkpoints. It performs no model inference while rescoring. It backs up v3/v3.1 quality files, rebuilds v4 summaries and full-reference deviations, archives stale frontier decisions, and prepares the progressive runner to replay the frontier under the corrected policy.

A different v4 winner at Stage N changes the physical parent architecture. Work downstream of that divergence cannot be repaired by rescoring and is regenerated from Stage N onward. Matching earlier evidence is reused.

## 1. Stop the active run

Press:

```text
Ctrl+C
```

Do not delete the artifact directory.

## 2. Optional review-only preview

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4.jsonl `
  -ReviewOnly
```

This writes sibling `generation_quality_v4_0` folders and does not modify official evidence.

## 3. Official in-place migration

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4.jsonl
```

The command refuses an uncontracted benchmark. Existing files receive timestamped names similar to:

```text
results.jsonl.pre_quality_v4_0_20260802_004500
summary.json.pre_quality_v4_0_20260802_004500
model_catalog.json.pre_quality_v4_0_20260802_004500
candidate_registry.json.pre_quality_v4_0_20260802_004500
```

The migration audit is written to:

```text
artifacts\phase_2b_quality_v3_thorough_20260801\PHASE2B_QUALITY_V4_0_MIGRATION.json
```

## 4. Replay and continue the corrected frontier

Use the same v4 probe and the prior run settings:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Report .\reports\PHASE_2B_phase_2b_quality_v3_thorough_20260801_REPORT.md `
  -GateReport .\reports\PHASE_2B_phase_2b_quality_v3_thorough_20260801_STOP_GATE.md `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing `
  -Resume
```

The runner begins decision replay at Stage 1 but reuses matching v4 generation evidence. It continues from the corrected terminal frontier after replay. A changed parent invalidates only its downstream suffix.

## 5. Clean rerun option

A completely new run remains available by using a new output root and omitting `-Resume`:

```powershell
$runDate = Get-Date -Format "yyyyMMdd_HHmmss"

.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4.jsonl `
  -ProbeLimit 150 `
  -OutputRoot ".\artifacts\phase_2b_quality_v4_clean_$runDate" `
  -Report ".\reports\PHASE_2B_quality_v4_clean_${runDate}_REPORT.md" `
  -GateReport ".\reports\PHASE_2B_quality_v4_clean_${runDate}_STOP_GATE.md" `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

## 6. Approval boundary

Do not approve the Phase 2B gate or start Phase 2C until:

1. the v4 frontier replay finishes;
2. the accepted and nearest rejected raw replies are manually reviewed;
3. the migration manifest and provenance checks pass; and
4. a human explicitly approves the selected profile in the Phase 2B stop gate.
