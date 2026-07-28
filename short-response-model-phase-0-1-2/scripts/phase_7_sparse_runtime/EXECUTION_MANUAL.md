# Phase 7 execution manual

1. Approve Phase 6 and keep its router dataset immutable.
2. Run `.\scripts\phase_7_sparse_runtime\run_phase_7.ps1`. The runner automatically chooses sparse benchmarking or a dense-only baseline from Phase 6 evidence.
3. When routers are trained, inspect per-layer validation recall and confidence thresholds.
4. Benchmark with no competing GPU workload; preserve hardware/software metadata.
5. Review dense/routed outputs as well as timing.
6. Select `SPARSE_FEATURE_FLAG` only if quality passes and measured speed or VRAM target passes; otherwise select `DENSE_RUNTIME`.
7. A dense selection is a successful practical Phase 7 outcome and allows Phase 8 to continue after human approval.
