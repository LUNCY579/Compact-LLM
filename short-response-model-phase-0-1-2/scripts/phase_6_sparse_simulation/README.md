# Phase 6 — Activation profiling and masked quality frontier

Phase 6 automatically builds non-sealed calibration data, records real decoder-layer activations/oracle block labels, and runs dense-compute masked generations at 100%, 80%, 65%, 50%, and 35% active ratios.

```powershell
.\scripts\phase_6_sparse_simulation\run_phase_6.ps1
```

This phase measures quality tolerance only; it must not claim speed or VRAM savings because gate/up projections are still computed densely. When no sparse ratio passes, the gate records `DENSE_ONLY` rather than blocking the commercial pipeline. Outputs include `router_dataset.pt`, human-readable trace metadata, `frontier_with_generations.json`, raw replies, `SPARSITY_CALIBRATION_REPORT.md`, and `PHASE_6_STOP_GATE.md`. See `FILE_RESPONSIBILITIES.md`.

Dense and masked candidates use the same response-control mode (`one_sentence` by default), so a sparse candidate cannot appear better merely because it was allowed a different stopping rule. This phase still makes no runtime claim.

## Quality evaluator v3.1

This phase uses Quality Evaluator `3.1` and the shared relative-opportunity engine. Informal or mildly ungrammatical language is permitted when understandable; references are examples rather than unique answers; and behavioral restrictions are enforced only when explicitly present in the response contract. Promotion never depends on an absolute score, pass-rate, completion, category, NLL, or issue-count floor. The declared proportional resource benefit must be strictly greater than combined proportional quality opportunity loss. See `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` and `EVALUATION_V3_MIGRATION.md`.

