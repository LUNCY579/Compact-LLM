# Verified execution manual — Phase 3

## Preflight

1. Activate the same environment used by Phase 2.
2. Choose exactly one profile for a comparable Phase 3/4 branch.
3. For `balanced` or `min_memory`, confirm `PHASE_2B_STOP_GATE.md` contains `Human decision: PASS` and lists that exact profile under `Approved profiles`.
4. Confirm the profile's ShortChatBench summary exists; Phase 3 must not compare DPO against the 50-case structural development probe.
5. Record artifact hashes before training.

## Execution

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 -Phase2Profile balanced
```

The runner resolves the model, predecessor adapter, and comparison summary from the Phase 2B catalog. Reduced branches attach a fresh LoRA; the full branch continues the accepted SFT adapter.

## Review

Inspect correctness, naturalness, context, clarification, safety, and the semantic flags. Do not approve a reduced branch merely because Phase 2B saved more weight memory than score. DPO must be reviewed on its own outputs.

## Recovery

Rerun evaluation without retraining when only the evaluator output is missing. Do not mix profile adapters or use a Phase 3 stop gate from another profile.
