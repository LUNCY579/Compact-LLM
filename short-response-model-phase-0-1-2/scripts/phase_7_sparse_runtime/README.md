# Phase 7 — Measured routed runtime

Phase 7 first verifies the strict one-sentence response-control contract, then trains one confidence-calibrated router per decoder layer and benchmarks the real gathered PyTorch sparse-decode reference against dense decoding on identical prompts.

```powershell
.\scripts\phase_7_sparse_runtime\run_phase_7.ps1
```

The approved Phase 6 ratio is resolved automatically. If Phase 6 selected `DENSE_ONLY`, Phase 7 skips router training and measures the real dense runtime baseline. Otherwise router/model fingerprints must match and the benchmark includes router/gather overhead, latency, peak VRAM, quality, active ratio, and dense-fallback rate. A valid negative result selects `DENSE_RUNTIME`; a passing sparse result may select `SPARSE_FEATURE_FLAG`. See `FILE_RESPONSIBILITIES.md` and `KERNEL_CONTRACT.md`.

Both dense and sparse paths use the same incremental controller and report actual tokens, visible tokens, stop reasons, trim rate, and one-sentence violation rate. Phase 7 cannot pass if strict mode was not measured or violated. See `RESPONSE_CONTROL_CONTRACT.md`. Whole-layer early exit is not implemented here.

## Quality evaluator v3.0

This phase uses Quality Evaluator `3.0` and the shared relative-opportunity engine. Informal or mildly ungrammatical language is permitted when understandable; references are examples rather than unique answers; and behavioral restrictions are enforced only when explicitly present in the response contract. Promotion never depends on an absolute score, pass-rate, completion, category, NLL, or issue-count floor. The declared proportional resource benefit must be strictly greater than combined proportional quality opportunity loss. See `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` and `EVALUATION_V3_MIGRATION.md`.

