# Verified Execution Manual — Phase 3

## Preflight

1. Activate the same environment used by Phase 2.
2. Choose exactly one upstream branch.
3. For `-Phase2Profile`, confirm the Phase 2B gate explicitly approves that profile.
4. For `-Phase2CProfile`, confirm the Phase 2C gate says `Human decision: PASS` and lists that exact profile under `Approved profiles`.
5. Confirm the resolved ShortChatBench summary exists; Phase 3 must not compare DPO against a structural/calibration probe.
6. Record artifact hashes before training.

## Phase 2B execution

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 -Phase2Profile balanced
```

## Phase 2C execution

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 -Phase2CProfile balanced_training
```

The Phase 2C branch writes:

```text
artifacts\phase_3_dpo_phase2c_balanced_training
artifacts\phase_3_dpo_phase2c_balanced_training_evaluation
reports\PHASE_3_phase2c_balanced_training_STOP_GATE.md
```

## Review

Inspect correctness, naturalness, context, clarification, safety, malformed outputs, and concrete defect and contract labels. Do not approve a branch merely because it uses fewer trainable parameters. DPO must be reviewed on its own outputs.

## Phase 4 handoff

For the example above:

```powershell
.\scripts\phase_4_model_soups\run_phase_4.ps1 -BranchName phase2c_balanced_training
```

This resolves the matching Phase 3 adapter and gate automatically.

## Recovery

Rerun evaluation without retraining only when the adapter exists and the evaluator output is missing. Never mix a model, SFT adapter, DPO adapter, or stop gate from a different profile.
