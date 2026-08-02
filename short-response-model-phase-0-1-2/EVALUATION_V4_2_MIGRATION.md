# Evaluator v4.2 migration and Phase 2B continuation

## Install

Extract the hotfix at the repository root and replace included files. Stop any active Phase 2B process first, but retain its artifact directory.

Verify the installation:

```powershell
python .\scripts\common\verify_quality_evaluation_v4_2.py
pytest -q
python .\scripts\common\audit_relative_opportunity_policy.py
```

## Review-only rescore

This performs no model inference and leaves official evidence unchanged:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_2.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_2.jsonl `
  -ReviewOnly
```

## Official in-place rescore

After reviewing the preview:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_2.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_2.jsonl
```

The migration backs up prior evaluator evidence, reuses stored replies, rebuilds summaries and full-reference deviations, archives stale decisions, and prepares corrected frontier replay. It does not regenerate responses.

## Resume Phase 2B

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_2.jsonl `
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

Evidence is reused while the corrected frontier follows the same parent lineage. Only downstream work after the first changed winner must be generated again.

## Clean rerun

```powershell
$runDate = Get-Date -Format "yyyyMMdd_HHmmss"

.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_2.jsonl `
  -ProbeLimit 150 `
  -OutputRoot ".\artifacts\phase_2b_quality_v4_2_full_$runDate" `
  -Report ".\reports\PHASE_2B_quality_v4_2_full_${runDate}_REPORT.md" `
  -GateReport ".\reports\PHASE_2B_quality_v4_2_full_${runDate}_STOP_GATE.md" `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

Do not add `-Resume` to the clean-run command.
