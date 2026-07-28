# Phase 7 - Routed Sparse Runtime

**Blueprint reference:** pages 19-21. This folder now provides a complete PyTorch correctness reference:

- one confidence-calibrated block router per decoder layer;
- dense fallback when router confidence is low;
- dense prefill plus gathered single-token decode;
- measured dense-versus-routed latency, peak VRAM, output agreement, active ratio, and fallback rate.

The gathered PyTorch path may be slower than dense GEMM. That is a valid result and the reason the blueprint requires measured wall-clock results before custom kernel claims.

## Prerequisite

Phase 6 must have a human `PASS`, and `router_dataset.pt` must come from the exact Phase 4B-selected reference being benchmarked. Normal execution resolves that model automatically.

## 1. Train per-layer routers

```powershell
python .\scripts\phase_7_sparse_runtime\train_router.py `
  --dataset artifacts\phase_6_sparse\router_dataset.pt `
  --target-ratio 0.65 `
  --minimum-weighted-recall 0.95 `
  --out artifacts\phase_7_router\routers.pt
```

The loss combines:

- binary oracle-mask prediction;
- a missed-contribution surrogate weighted by block importance;
- a soft active-ratio budget.

A validation split calibrates an entropy-based confidence threshold for dense fallback.

## 2. Benchmark the reference runtime

```powershell
python .\scripts\phase_7_sparse_runtime\benchmark_runtime.py `
  --model <Phase-4B-selected-model> `
  --routers artifacts\phase_7_router\routers.pt `
  --benchmark data\benchmark\shortchatbench_pilot.jsonl `
  --max-new-tokens 24 `
  --repeats 3 `
  --out artifacts\phase_7_router\runtime_comparison.json
```

The benchmark performs deterministic greedy decoding and reports:

- median prefill time;
- median decode milliseconds per token;
- total decode time;
- peak allocated VRAM;
- exact generated-token match rate;
- per-layer active-block and dense-fallback rates;
- whether the blueprint's provisional 1.20x speed or 20% peak-VRAM target passes.

## Implementation boundary

`RoutedSparseGatedMLP` gathers selected rows from gate/up projections and selected columns from down projection for batch=1, sequence=1 decode. Prefill and unsupported shapes use the original dense MLP.

This is deliberately a reference implementation. A production kernel still needs contiguous packed weights, fused routing/gather, asynchronous look-ahead, and hardware-specific profiling.

## Verification checklist

- [ ] Router dataset/model hashes match the benchmarked checkpoint.
- [ ] Router validation weighted recall meets the configured floor.
- [ ] Dense fallback is enabled and measured.
- [ ] Dense and sparse runs use identical prompts, precision, generation settings, and hardware.
- [ ] Raw replies are reviewed for quality regressions.
- [ ] Router overhead is included in measured decode time.
- [ ] A slower PyTorch reference is recorded honestly.
- [ ] Custom-kernel work starts only after correctness and quality pass.
