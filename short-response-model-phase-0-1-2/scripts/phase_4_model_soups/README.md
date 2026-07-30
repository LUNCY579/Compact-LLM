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

For a named branch, the runner automatically selects the matching Phase 3
adapter, branch stop gate, and isolated Phase 4 output tree unless you explicitly
override them.

The runner reads the Phase 3 training manifest. When `sft_adapter` is null, it correctly trains every additional ingredient as a fresh LoRA on the same materialized reduced model. Never mix ingredients from full-quality, balanced, and minimum-memory architectures in one soup.

A best individual with score 1.0 is still a valid Phase 4 outcome. Soup contrast is limited when the benchmark saturates; Phase 2B's NLL and output-deviation evidence addresses structural contrast, while Phase 4 remains responsible for merge stability.
