# Evaluator v4.4 migration and Phase 2B resume guide

## Install

Extract the v4.4 hotfix into the repository root and allow included files to replace existing files. Keep the `artifacts` directory.

Verify the installation:

```powershell
python .\scripts\common\verify_quality_evaluation_v4_4.py
pytest -q
python .\scripts\common\audit_relative_opportunity_policy.py
```

Expected evaluator verification:

```text
Evaluator v4.4 verification passed: 620/620 cases,
31 reporting categories, 61 scored labels,
150/150 real collapses.
```

## Review-only rescore

This performs no model inference and does not replace official result files:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_4.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v4_full_20260802_140809 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_4.jsonl `
  -ReviewOnly
```

Use the actual path of your Phase 2B run if it differs.

## Official in-place rescore

After reviewing the preview:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_4.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v4_full_20260802_140809 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_4.jsonl
```

This command:

- reuses stored replies;
- performs no new generation;
- backs up prior result and summary files;
- rebuilds v4.4 quality summaries and output deviations;
- archives stale catalog, registry, and stage-decision evidence;
- writes `PHASE2B_QUALITY_V4_4_MIGRATION.json`;
- prepares the run for corrected frontier replay.

## Resume the existing run

Use the original run root and add `-Resume`:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_4.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2b_quality_v4_full_20260802_140809 `
  -Report .\reports\PHASE_2B_quality_v4_full_20260802_140809_REPORT.md `
  -GateReport .\reports\PHASE_2B_quality_v4_full_20260802_140809_STOP_GATE.md `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing `
  -Resume
```

Matching generation, teacher-forced NLL, footprint, and checkpoint evidence is reused. If v4.4 changes a winning parent, only the downstream branch after that divergence requires new model work.

## Clean run

Use a new output root and omit `-Resume`:

```powershell
$runDate = Get-Date -Format "yyyyMMdd_HHmmss"

.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_4.jsonl `
  -ProbeLimit 150 `
  -OutputRoot ".\artifacts\phase_2b_quality_v4_4_full_$runDate" `
  -Report ".\reports\PHASE_2B_quality_v4_4_full_${runDate}_REPORT.md" `
  -GateReport ".\reports\PHASE_2B_quality_v4_4_full_${runDate}_STOP_GATE.md" `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```
