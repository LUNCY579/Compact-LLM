# Evaluator v4.5 Migration and Phase 2B Resume Guide

## Installation

Extract the v4.5 hotfix ZIP directly into the repository root and allow included files to replace older files. Keep the existing `artifacts` directory.

Verify the installation:

```powershell
python .\scripts\common\verify_quality_evaluation_v4_5.py
pytest -q
python .\scripts\common\audit_relative_opportunity_policy.py
```

Expected evaluator verification includes:

```text
639/639 cases
31 reporting categories
63 scored labels
150/150 real collapses
```

## Preview preserved generations

This command performs no model inference and writes review-only sibling evidence:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_5.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v4_full_20260802_155652 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl `
  -ReviewOnly
```

## Official in-place rescore

After inspecting the preview:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_5.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v4_full_20260802_155652 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl
```

The migration:

1. backs up existing `results.jsonl` and `summary.json` files;
2. re-scores stored replies under v4.5;
3. rebuilds full-reference output-deviation evidence;
4. archives stale parent-relative decisions, catalogs, and registries;
5. records candidate lineage and evidence provenance;
6. writes `PHASE2B_QUALITY_V4_5_MIGRATION.json`;
7. prepares the run for corrected Stage-1 frontier replay.

No model generation is performed.

## Resume Phase 2B

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2b_quality_v4_full_20260802_155652 `
  -Report .\reports\PHASE_2B_quality_v4_full_20260802_155652_REPORT.md `
  -GateReport .\reports\PHASE_2B_quality_v4_full_20260802_155652_STOP_GATE.md `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing `
  -Resume
```

Existing generation, NLL, footprint, and model evidence is reused while the corrected winner follows the same parent lineage. If v4.5 changes a winning parent, only the downstream search suffix that depends on the superseded parent must be generated again.

## Clean run

Use a new output root and omit `-Resume`:

```powershell
$runDate = Get-Date -Format "yyyyMMdd_HHmmss"

.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl `
  -ProbeLimit 150 `
  -OutputRoot ".\artifacts\phase_2b_quality_v4_5_full_$runDate" `
  -Report ".\reports\PHASE_2B_quality_v4_5_full_${runDate}_REPORT.md" `
  -GateReport ".\reports\PHASE_2B_quality_v4_5_full_${runDate}_STOP_GATE.md" `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

## Compatibility

Older rescore wrappers forward to the v4.5 wrapper. Official Phase 2B selection requires a benchmark whose `response_contract.schema_version` matches Evaluator v4.5. Mixed v4.4/v4.5 decision evidence is rejected.
