# Phase 7 — Measured routed runtime

Phase 7 trains one confidence-calibrated router per decoder layer and benchmarks the real gathered PyTorch sparse-decode reference against dense decoding on identical prompts.

```powershell
.\scripts\phase_7_sparse_runtime\run_phase_7.ps1
```

The approved Phase 6 ratio is resolved automatically. If Phase 6 selected `DENSE_ONLY`, Phase 7 skips router training and measures the real dense runtime baseline. Otherwise router/model fingerprints must match and the benchmark includes router/gather overhead, latency, peak VRAM, quality, active ratio, and dense-fallback rate. A valid negative result selects `DENSE_RUNTIME`; a passing sparse result may select `SPARSE_FEATURE_FLAG`. See `FILE_RESPONSIBILITIES.md` and `KERNEL_CONTRACT.md`.
