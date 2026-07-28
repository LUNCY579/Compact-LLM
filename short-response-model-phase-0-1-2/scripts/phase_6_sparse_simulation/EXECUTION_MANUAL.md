# Phase 6 Execution Manual

1. Freeze and hash the accepted dense reference.
2. Prepare a diverse non-sealed calibration JSONL.
3. Run `profile_activations.py` with decode-oriented `--positions last`.
4. Inspect the summary for sample count, block count, active ratio, and retained contribution in every layer.
5. Run `simulate_sparsity.py` for 100/80/65/50/35% active ratios.
6. Re-run with `--model` and `--benchmark` to capture masked generations.
7. Evaluate raw replies with the same deterministic generation settings used for the dense reference.
8. Record category regressions and confidence intervals.
9. Create `reports/PHASE_6_STOP_GATE.md`.
10. Record human `PASS` only if simulated sparsity skips a meaningful fraction without exceeding the accepted quality loss.

Do not interpret a high retained-contribution score as a runtime speedup. The simulation computes all gate/up channels before masking.
