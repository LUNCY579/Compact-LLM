# Project Scripts by Phase

The executable implementation is organized by blueprint phase.

## `phase_0_data/`

Dataset schema validation, leakage checks, and reproducible human-audit packet
creation.

## `phase_1_dense_sft/`

Base-model smoke inference, untouched baseline evaluation, all-layer QLoRA SFT,
adapter evaluation, and the formal Stop Gate 1 comparison.

Run the whole phase:

```powershell
.\scripts\phase_1_dense_sft\run_phase_1.ps1
```

## `phase_2_layer_selection/`

Controlled single-layer scan. The runner refuses to start unless
`reports/PHASE_1_STOP_GATE.md` contains an explicit human `PASS`.

## Compatibility entry points

The original root-level Python and PowerShell filenames remain as small wrappers,
so earlier commands still work. New work should call the phase-specific paths.
