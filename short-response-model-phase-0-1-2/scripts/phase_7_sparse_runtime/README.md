# Phase 7 — Sparse Runtime

**Blueprint reference:** Pages 19–21. This folder is an executable research scaffold plus a verification manual. Generated metrics are never assumed; artifacts remain outside source control.

## Prerequisite

The preceding phase report must contain an explicit human `PASS`. The PowerShell runner checks this before continuing.

## Step-by-step execution

1. Confirm Stop Gate 5/Phase 6. 2. Create oracle masks and router-training examples. 3. Train block routers with mask, reconstruction and budget objectives. 4. Calibrate confidence and enforce dense fallback. 5. Test layer-specific budgets without assuming training importance equals inference sensitivity. 6. Validate a PyTorch reference. 7. Prototype contiguous Triton/custom kernels. 8. Add look-ahead routing and only then hot/warm/cold placement. 9. Quantize after the model is frozen, then recalibrate. 10. Report router overhead, bytes, TTFT, decode time, total latency, VRAM/RAM and fallback rate. 11. Approve only with quality plus measured >=1.20x speed or >=20% lower peak VRAM.

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

Create or update `reports/PHASE_7_STOP_GATE.md` and use exactly `- Human decision: `PASS`` only after the checklist and blueprint gate are satisfied.
