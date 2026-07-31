# Phase 4 — Real model-soup execution

`run_phase_4.ps1` trains compatible additional DPO seeds, evaluates them, performs reconstructed-delta FP32 merges, evaluates each greedy trial, and materializes the accepted standalone checkpoint.

## Full-quality branch

```powershell
.\scripts\phase_4_model_soups\run_phase_4.ps1 `
  -PrimaryAdapter .\artifacts\phase_3_dpo\adapter
```

## Phase 2B profile branch

```powershell
.\scripts\phase_4_model_soups\run_phase_4.ps1 -BranchName balanced
```

## Phase 2C profile branch

```powershell
.\scripts\phase_4_model_soups\run_phase_4.ps1 -BranchName phase2c_balanced_training
```

Other valid Phase 2C branch names follow the Phase 3 output label:

- `phase2c_maximum_quality`
- `phase2c_balanced_training`
- `phase2c_minimum_training_memory`

For a named branch, the runner derives the matching Phase 3 adapter, branch stop gate, and isolated Phase 4 output tree unless explicitly overridden.

The runner reads the Phase 3 training manifest. A Phase 2B reduced profile with no predecessor adapter trains each ingredient as a fresh LoRA on the same materialized checkpoint. A Phase 2C branch continues the exact targeted SFT adapter recorded by Phase 3. Never mix full-quality, Phase 2B, or Phase 2C branches—or different Phase 2C placements—in one soup.

A best individual with score 1.0 remains a valid Phase 4 outcome. Soup contrast is limited when the benchmark saturates; NLL, category metrics, semantic flags, and raw output review remain necessary.
