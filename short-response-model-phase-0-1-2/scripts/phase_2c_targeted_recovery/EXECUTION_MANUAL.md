# Phase 2C Execution Manual

## Prerequisites

1. Phase 1 is complete.
2. Phase 2B completed and `reports/PHASE_2B_STOP_GATE.md` contains an explicit human `PASS`.
3. The selected source profile (`balanced` or `min_memory`) is listed in Phase 2B `Approved profiles`.
4. CUDA PyTorch, Transformers, PEFT, bitsandbytes, and repository requirements are installed.
5. Keep the Phase 2B reduced checkpoint, catalog, candidate registry, and gate report in place.
6. Keep sufficient disk space for adapters, checkpoints, generated replies, NLL evidence, and repeated-seed controls.

## Normal run

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -CalibrationLimit 64 `
  -EvalLimit 180 `
  -MaxSteps 200 `
  -RepeatSeeds "23,37" `
  -RepeatTopK 2 `
  -LoraRank 16 `
  -LoraAlpha 32 `
  -LearningRate 0.0002 `
  -QLoRA `
  -GradientCheckpointing
```

The gate compares proportional trainable-state/training-memory savings with combined proportional quality opportunity loss. Critical and major issues, completion, categories, and target NLL contribute continuously; none is an independent absolute veto or floor.

## Alternate isolated run

Use unique output/report paths so the default run remains immutable:

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -OutputRoot .\artifacts\phase_2c_targeted_recovery_v2 `
  -Report .\reports\PHASE_2C_V2_REPORT.md `
  -GateReport .\reports\PHASE_2C_V2_STOP_GATE.md `
  -MaxSteps 300 `
  -RepeatSeeds "23,37,49" `
  -QLoRA `
  -GradientCheckpointing
```

## Resume

Use the exact original command and add `-Resume`. The runner fingerprints the source checkpoint and all decision-bearing files. It refuses a resume when a model, dataset, benchmark, seed, threshold, LoRA setting, or importance weight differs.

An incomplete training directory without a final `training_manifest.json` is deleted and regenerated. Completed generation, NLL, ranking, training, and comparison steps reuse their final sentinel files.

## Clean restart

Delete only the selected Phase 2C output root and its generated reports, not Phase 2B:

```powershell
Remove-Item -Recurse -Force .\artifacts\phase_2c_targeted_recovery -ErrorAction SilentlyContinue
Remove-Item -Force .\reports\PHASE_2C_TARGETED_RECOVERY_REPORT.md, .\reports\PHASE_2C_STOP_GATE.md -ErrorAction SilentlyContinue
```

Then rerun without `-Resume`.

## Review sequence

1. Inspect `run_config.json` and confirm the source-model fingerprint and settings.
2. Inspect `importance/combined_layer_ranking.json`; distinguish current reduced-layer IDs from original Qwen layer IDs.
3. Inspect every `candidate_result.json` and `OUTPUT_DEVIATION.md`.
4. Review malformed, empty, logic, clarification, context, factual, and safety cases.
5. Confirm each recommended profile has passing repeated-seed results and same-seed `all_remaining` controls.
6. Compare trainable parameters, peak training VRAM, runtime, and adapter bytes against `all_remaining`.
7. Inspect common ShortChatBench profile-validation outputs before approval.
8. Keep each adapter paired with the exact reduced model recorded in the catalog.

## Audit completed artifacts

```powershell
python .\scripts\phase_2c_targeted_recovery\audit_phase_2c_artifacts.py `
  --json-out .\artifacts\phase_2c_targeted_recovery\artifact_audit.json
```

## Approve profiles

Edit only the human fields in:

```text
reports\PHASE_2C_STOP_GATE.md
```

Example:

```text
- Human decision: `PASS`
- Approved profiles: `maximum_quality,balanced_training`
```

The resolver rejects profiles that are not human-approved, automated-eligible, and repeated-seed stable.

## Continue to Phase 3

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 `
  -Phase2CProfile balanced_training
```

The branch writes to `phase_3_dpo_phase2c_balanced_training` and does not overwrite legacy or Phase 2B-only results.
