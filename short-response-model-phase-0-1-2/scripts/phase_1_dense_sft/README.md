# Phase 1 — Dense SFT Baseline

This folder contains the complete Phase 1 workflow required before any
single-layer or sparse-inference experiment.

## Files

- `smoke_inference.py` — proves the base checkpoint loads and generates on CUDA.
- `evaluate_model.py` — evaluates either the untouched base or a PEFT adapter.
- `train_dense_sft.py` — trains the all-layer LoRA/QLoRA behavioral reference.
- `check_phase_1_gate.py` — compares base and adapter summaries and creates the
  formal Stop Gate 1 report.
- `001_smoke_inference.ps1` — optional one-prompt smoke test.
- `002_evaluate_base.ps1` — writes the untouched baseline artifacts.
- `003_train_all_layer_qlora.ps1` — trains the all-layer QLoRA reference.
- `004_evaluate_dense_sft.ps1` — evaluates the saved adapter.
- `005_check_stop_gate.ps1` — writes `reports/PHASE_1_STOP_GATE.md`.
- `run_phase_1.ps1` — runs all five steps in order.

## Run

From the repository root:

```powershell
.\scripts\phase_1_dense_sft\run_phase_1.ps1
```

The final automated status is not sufficient to start Phase 2. Review the
generated replies and record a human decision in `reports/PHASE_1_STOP_GATE.md`.

## Attention-versus-MLP adapter ablation

`train_dense_sft.py` now accepts `--target-module-group both|attention|mlp`. Use the same selected layers, seed, data, and training budget when comparing groups. This tests where concise behavior is easiest to learn; it does not authorize skipping other modules at inference.
