# Phase 7 — Measured routed runtime

Phase 7 first verifies the strict one-sentence response-control contract, then trains one confidence-calibrated router per decoder layer and benchmarks the real gathered PyTorch sparse-decode reference against dense decoding on identical prompts.

```powershell
.\scripts\phase_7_sparse_runtime\run_phase_7.ps1
```

The approved Phase 6 ratio is resolved automatically. If Phase 6 selected `DENSE_ONLY`, Phase 7 skips router training and measures the real dense runtime baseline. Otherwise router/model fingerprints must match and the benchmark includes router/gather overhead, latency, peak VRAM, quality, active ratio, and dense-fallback rate. A valid negative result selects `DENSE_RUNTIME`; a passing sparse result may select `SPARSE_FEATURE_FLAG`. See `FILE_RESPONSIBILITIES.md` and `KERNEL_CONTRACT.md`.

Both dense and sparse paths use the same incremental controller and report actual tokens, visible tokens, stop reasons, trim rate, and one-sentence violation rate. Phase 7 cannot pass if strict mode was not measured or violated. See `RESPONSE_CONTROL_CONTRACT.md`. Whole-layer early exit is not implemented here.

## Quality evaluator v2.0

This phase consumes or produces evaluator schema `2.0`. The main score is not a length/completion average: hard intelligibility collapse receives zero, and phase gates separately enforce tail quality, category behavior, semantic and completion floors, reference alignment, and critical/major issue limits. Evidence created by the legacy evaluator must be regenerated before this phase can automatically pass. See the repository-root `QUALITY_EVALUATION_V2_AUDIT.md` and `QUALITY_EVALUATION_V2_MIGRATION.md`.

