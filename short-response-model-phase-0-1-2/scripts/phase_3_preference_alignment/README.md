# Phase 3 — Preference Alignment

**Blueprint reference:** Pages 15–16. This folder is an executable research scaffold plus a verification manual. Generated metrics are never assumed; artifacts remain outside source control.

## Prerequisite

The preceding phase report must contain an explicit human `PASS`. The PowerShell runner checks this before continuing.

## Step-by-step execution

1. Confirm Stop Gate 2 human PASS. 2. Build versioned chosen/rejected pairs covering correctness, brevity, naturalness, context, clarification, safety, and variety. 3. Validate pairs. 4. Train DPO from the winning static selective adapter; do not start RL yet. 5. Evaluate with exactly the same ShortChatBench decoding and preserved capability checks. 6. Blind-review SFT versus DPO. 7. Approve only when short-correct preference improves with zero critical regression. Optional RL is allowed only after a documented DPO failure.

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

Create or update `reports/PHASE_3_STOP_GATE.md` and use exactly `- Human decision: `PASS`` only after the checklist and blueprint gate are satisfied.
