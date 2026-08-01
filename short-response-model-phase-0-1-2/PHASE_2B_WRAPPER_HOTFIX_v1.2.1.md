# Phase 2B Wrapper Hotfix v1.2.1

This maintenance update synchronizes the Phase 2B PowerShell entry point with the already-correct Quality Evaluation v2.0 Python and YAML contracts. It changes orchestration and documentation only; no model weights, datasets, artifacts, phase decisions, or generated evidence are modified.

## Corrected discrepancies

- `MaxScoreLoss` default: `0.10` → `0.05`.
- `BalancedMaxScoreLoss` default: `0.03` → `0.02`.
- Adds and forwards `ProbeLimit`, preventing a 150-row probe from being described or created as a 50-case configuration.
- Exposes and forwards all remaining Quality Evaluation v2.0 thresholds instead of silently inheriting mixed defaults.
- Exposes phase benchmark, report, gate-report, probe seed, recovery seed, maximum response tokens, and minimum reduction advantage.
- Custom output roots now derive unique report names unless explicit report paths are supplied, preventing accidental report overwrite.
- Uses an explicitly supplied Python executable, a repository-local virtual environment when present, or the active `python` command in that order.
- Prints the resolved probe, output, report, gate, and key quality thresholds before the expensive run begins.
- Adds argument validation and a regression test that locks the wrapper to the v2.0 contract.

## Changed files

- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1`
- `scripts/phase_2_layer_selection/README.md`
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md`
- `tests/test_phase_2_progressive_reduction.py`

## Added file

- `PHASE_2B_WRAPPER_HOTFIX_v1.2.1.md`

## Recommended thorough command

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2_progressive_reduction_v1_2_thorough `
  -Report .\reports\PHASE_2B_V1_2_THOROUGH_REPORT.md `
  -GateReport .\reports\PHASE_2B_V1_2_THOROUGH_STOP_GATE.md `
  -MaxDrops 10 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

Add `-Resume` only when the code, evaluator, model, adapter, data, probe, seeds, and thresholds are unchanged.
