# Phase 8 — Release Candidate

**Blueprint reference:** Pages 22–26. This folder is an executable research scaffold plus a verification manual. Generated metrics are never assumed; artifacts remain outside source control.

## Prerequisite

The preceding phase report must contain an explicit human `PASS`. The PowerShell runner checks this before continuing.

## Step-by-step execution

1. Confirm every previous human gate. 2. Run the required A–I ablation matrix. 3. Use sealed evaluation only now and preserve immutable results. 4. Conduct blinded randomized human review for correctness, naturalness, usefulness and preferred brevity. 5. Enforce category floors and zero critical safety regressions. 6. Hash model, tokenizer, adapters/soup recipe and manifests. 7. Record hardware, software, seeds and commands. 8. Audit risks and negative results. 9. Choose dense quantized, selective dense, soup, sparse feature flag, or larger-model routing based on evidence. 10. Human signs the final release decision.

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

Create or update `reports/PHASE_8_STOP_GATE.md` and use exactly `- Human decision: `PASS`` only after the checklist and blueprint gate are satisfied.
