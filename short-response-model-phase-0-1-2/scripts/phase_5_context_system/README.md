# Phase 5 — Context System

**Blueprint reference:** Pages 8–10 and 24. This folder is an executable research scaffold plus a verification manual. Generated metrics are never assumed; artifacts remain outside source control.

## Prerequisite

The preceding phase report must contain an explicit human `PASS`. The PowerShell runner checks this before continuing.

## Step-by-step execution

1. Confirm Stop Gate 4. 2. Implement structured memory with staleness and contradiction flags. 3. Retrieve using relevance plus recency/salience. 4. Assemble strict per-tier and total token budgets. 5. Deduplicate memory and recent turns. 6. Produce retrieval traces. 7. Implement a stable-prefix cache key containing model, tokenizer, exact prefix and context settings; actual KV reuse must be runtime-specific. 8. Add reply mode and escalation control. 9. Evaluate relevant-memory recall, selected tokens, contradictions, and latency. 10. Approve only when context quality and latency improve.

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

Create or update `reports/PHASE_5_STOP_GATE.md` and use exactly `- Human decision: `PASS`` only after the checklist and blueprint gate are satisfied.
