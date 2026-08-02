# Evaluator v4.1 migration and Phase 2B resume guide

## Install

Extract the v4.1 hotfix ZIP into the repository root and replace included files. Stop any active Phase 2B process first, but keep its artifact directory.

## Verify the evaluator before touching artifacts

```powershell
python .\scripts\common\verify_quality_evaluation_v4_1.py
pytest -q
python .\scripts\common\audit_relative_opportunity_policy.py
```

## Review-only re-evaluation

This creates sibling `generation_quality_v4_1` directories and does not modify official evidence:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_1.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_1.jsonl `
  -ReviewOnly
```

## Official in-place re-evaluation

This performs no model inference. It creates timestamped backups, re-scores stored replies, rebuilds full-reference output deviations, archives stale frontier decisions, and prepares corrected-frontier replay:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_1.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_1.jsonl
```

Equivalent Python command:

```powershell
python .\scripts\phase_2_layer_selection\rescore_phase_2b_artifacts.py `
  --run-root .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  --benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_1.jsonl `
  --in-place `
  --prepare-resume
```

The migration audit is written as:

```text
<RunRoot>\PHASE2B_QUALITY_V4_1_MIGRATION.json
```

## Resume the corrected Phase 2B frontier

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_1.jsonl `
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

The runner reuses stored generation, target-NLL, footprint, and checkpoint evidence while the corrected frontier follows the same parent lineage. If v4.1 changes a stage winner, only evidence downstream of the first changed parent must be generated again.

## Start a clean v4.1 run instead

```powershell
$runDate = Get-Date -Format "yyyyMMdd_HHmmss"

.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_1.jsonl `
  -ProbeLimit 150 `
  -OutputRoot ".\artifacts\phase_2b_quality_v4_1_full_$runDate" `
  -Report ".\reports\PHASE_2B_quality_v4_1_full_${runDate}_REPORT.md" `
  -GateReport ".\reports\PHASE_2B_quality_v4_1_full_${runDate}_STOP_GATE.md" `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

Do not add `-Resume` to a clean run. A new output root preserves all earlier evidence.
