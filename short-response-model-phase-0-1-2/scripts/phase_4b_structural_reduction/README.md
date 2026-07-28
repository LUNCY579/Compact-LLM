# Phase 4B — Structural reduction and reference selection

Phase 4B tests permanent MLP pruning, optional grouped-query-attention pruning, and depth reduction after Phase 4 has finalized model weights. It always keeps the untouched dense reference as a candidate.

## Normal command

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 -Mode Pilot
```

The runner resolves the accepted Phase 4 model and generates long-form calibration data automatically when absent. `-Mode Full` expands the candidate sweep; `-EnableAttentionPruning` enables the riskier architecture-specific head/GQA experiment.

After reviewing `PHASE_4B_STOP_GATE.md`, finalize one candidate:

```powershell
python .\scripts\phase_4b_structural_reduction\finalize_reference_selection.py `
  --registry .\artifacts\phase_4b_structural_reduction\candidate_registry.json `
  --candidate dense_reference `
  --reason "Best practical checkpoint after review."
```

This writes `artifacts/references/selected_inference_reference.json`; Phases 5–8 consume it automatically. See `FILE_RESPONSIBILITIES.md`.
