# Phase 4 — Selective Adapter Soups

**Blueprint reference:** Pages 16–17. This folder is an executable research scaffold plus a verification manual. Generated metrics are never assumed; artifacts remain outside source control.

## Prerequisite

The preceding phase report must contain an explicit human `PASS`. The PowerShell runner checks this before continuing.

## Step-by-step execution

1. Confirm Stop Gate 3. 2. Train 3–5 ingredients from the same base, tokenizer, selected layers, target modules, rank and objective; vary seed/data order first. 3. Validate compatibility. 4. Evaluate each ingredient. 5. Start greedy soup with the best. 6. Reconstruct each merged checkpoint and average FP32 deltas; never average quantized integers. 7. Evaluate every trial and reject any category regression. 8. Approve only if the accepted soup beats the best ingredient; otherwise keep the best individual. 9. Freeze the final dense reference and record hashes.

## Verification checklist

- [ ] Inputs, hashes, dataset versions, model/tokenizer versions and seed recorded.
- [ ] Development/calibration data used only for the intended phase.
- [ ] Sealed evaluation remains untouched until Phase 8.
- [ ] Commands and raw outputs preserved under `artifacts/`.
- [ ] Automated scores reviewed against raw generations/traces.
- [ ] No theoretical memory, active-parameter, or speed claim is presented as measured.
- [ ] Stop-gate report has evidence, risks, and a human decision.

## Files in this folder

Review every Python and PowerShell file before execution. Defaults are conservative examples; paths to winning adapters/models must be supplied from your actual reports. The scripts fail rather than fabricate missing results.

## Manual approval

Create or update `reports/PHASE_4_STOP_GATE.md` and use exactly `- Human decision: `PASS`` only after the checklist and blueprint gate are satisfied.
