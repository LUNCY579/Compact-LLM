# Phase 4 — Real model-soup execution

This phase no longer prints a checklist. `run_phase_4.ps1` trains compatible additional DPO seeds, evaluates them, performs real reconstructed-delta FP32 merges, evaluates each greedy trial, and materializes the accepted standalone Transformers checkpoint.

## Normal command

```powershell
.\scripts\phase_4_model_soups\run_phase_4.ps1 `
  -ModelId "Qwen/Qwen3-0.6B" `
  -PrimaryAdapter .\artifacts\phase_3_dpo\adapter
```

Use `-ReuseExisting` to resume valid artifacts or `-SkipTraining` only when all requested seed adapters already exist. Outputs include `accepted_model`, raw ingredient/trial evaluations, `MODEL_SOUP_REPORT.md`, `PHASE_4_STOP_GATE.md`, and `artifacts/references/phase_4_reference.json`.

The best individual is a valid result when no soup improves quality. See `FILE_RESPONSIBILITIES.md`.
