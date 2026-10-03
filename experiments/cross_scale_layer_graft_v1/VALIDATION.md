# Validation receipts

Environment: `C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe`,
PyTorch 2.12.1+cu126, Transformers 5.14.1. Execution date: 2026-09-05.

Before the model experiment:

```text
python -m pytest -q experiments/cross_scale_layer_graft_v1/tests
18 passed in 26.91s
```

These tests exercised native Qwen3 modules with different target/donor widths
and RoPE dimensions. They checked exact streamed-vs-monolithic raw layer states,
zero/identity output preservation, delta subtraction and bias cancellation,
future-token isolation, cache rejection, exception restoration, stem exclusion,
nonfinite rejection, affine-map generalization, and scenario-grouped holdout.

The audit tests were added after the model experiment started. They do not
modify registered model-run sources:

```text
python -m pytest -q experiments/cross_scale_layer_graft_v1/tests/test_audit.py
2 passed in 0.38s
```

These reject missing, duplicate or reordered cases, norm excursions, score loss,
timeouts, invalid output increases, and excessive KL in the independent gate.
Total: **20 tests passed** across these two invocations.

Compilation passed for experiment sources before model execution and for the
additional auditor and post-hoc diagnostic scripts afterward. The historical
ARMR source files are imported unchanged and hash-checked by the experiment.

The model-run manifest records an additional real Supa validation: compare one
streamed calibration state to the same post-L12 state from the canonical custom
Supa model, requiring bit identity before any graft evaluation.

The separate final audit recomputes scores, case coverage, primary and development
gates, data overlap, and historical A0/A1 identity from saved receipts. Its result
is recorded in an exclusive `evidence/audit_*/audit.json` directory.
