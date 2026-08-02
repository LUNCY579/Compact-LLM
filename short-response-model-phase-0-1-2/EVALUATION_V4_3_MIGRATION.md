# Evaluator v4.3 migration and Phase 2B continuation

## Install and verify

Extract the hotfix directly into the repository root and allow included files to be replaced. Keep the existing `artifacts` directory.

```powershell
python .\scripts\common\verify_quality_evaluation_v4_3.py
pytest -q
python .\scripts\common\audit_relative_opportunity_policy.py
```

Expected evaluator verification:

```text
Evaluator v4.3 verification passed: 611/611 cases,
31 reporting categories, 60 scored labels, 150/150 real collapses.
```

## Review-only rescore

This performs no model inference and does not replace official evidence:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_3.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_3.jsonl `
  -ReviewOnly
```

The preview writes separate v4.3 generation evidence for inspection.

## Official in-place rescore

After reviewing the preview:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_3.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_3.jsonl
```

This command performs no new generation. It backs up old evaluator evidence, re-scores stored replies, rebuilds summaries and full-reference deviations, archives stale catalogs and decisions, and writes:

```text
PHASE2B_QUALITY_V4_3_MIGRATION.json
```

## Resume corrected frontier reconstruction

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_3.jsonl `
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

Stored generation, teacher-forced NLL, footprint, and compatible checkpoints are reused while the corrected frontier follows the same parent lineage. If v4.3 chooses a different winner at a stage, only the downstream branch created from the superseded parent must be regenerated.

## Completely fresh v4.3 run

```powershell
$runDate = Get-Date -Format "yyyyMMdd_HHmmss"

.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_3.jsonl `
  -ProbeLimit 150 `
  -OutputRoot ".\artifacts\phase_2b_quality_v4_3_full_$runDate" `
  -Report ".\reports\PHASE_2B_quality_v4_3_full_${runDate}_REPORT.md" `
  -GateReport ".\reports\PHASE_2B_quality_v4_3_full_${runDate}_STOP_GATE.md" `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

Do not add `-Resume` to the clean-run command.

## Compatibility

Older v3.1/v4.0/v4.1/v4.2 rescore wrappers forward to v4.3 so saved commands do not silently run a superseded evaluator. Official evidence must use a v4.3 benchmark contract; mixed evaluator versions are rejected by provenance checks.
