# Phase 6 — Activation Profiling and Sparse Simulation

**Blueprint reference:** Pages 17–19. This folder is an executable research scaffold plus a verification manual. Generated metrics are never assumed; artifacts remain outside source control.

## Prerequisite

The preceding phase report must contain an explicit human `PASS`. The PowerShell runner checks this before continuing.

## Step-by-step execution

1. Confirm Phase 5. 2. Freeze the exact dense reference; any weight change invalidates traces. 3. Build a diverse calibration set including chat, facts, long-context memory, code/math controls, multilingual, safety and adversarial ambiguity. 4. Hook gated-MLP intermediate input to the down projection. 5. Compute block contribution proxies and save trace metadata. 6. Simulate 100/80/65/50/35% active ratios, uniform and layer-sensitive policies. 7. Re-run quality evaluation at each ratio. 8. Do not claim speed; masking still uses dense computation. 9. Approve only when meaningful blocks can be omitted inside quality limits.

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

Create or update `reports/PHASE_6_STOP_GATE.md` and use exactly `- Human decision: `PASS`` only after the checklist and blueprint gate are satisfied.
