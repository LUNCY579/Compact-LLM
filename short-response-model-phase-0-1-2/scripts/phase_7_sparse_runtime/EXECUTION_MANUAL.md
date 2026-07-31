# Phase 7 execution manual

1. Approve Phase 6 and keep its router dataset immutable.
2. The runner executes `verify_response_control.py`; inspect `response_control_verification.json` if it fails.
3. Run `.\scripts\phase_7_sparse_runtime\run_phase_7.ps1`. The runner automatically chooses sparse benchmarking or a dense-only baseline from Phase 6 evidence.
4. When routers are trained, inspect per-layer validation recall and confidence thresholds.
5. Benchmark with no competing GPU workload; preserve hardware/software metadata.
6. Review dense/routed outputs as well as timing.
7. Select `SPARSE_FEATURE_FLAG` only if quality passes and measured speed or VRAM target passes; otherwise select `DENSE_RUNTIME`.
8. A dense selection is a successful practical Phase 7 outcome and allows Phase 8 to continue after human approval.
9. Confirm both paths used `one_sentence` and report a zero one-sentence violation rate. Do not interpret this controller as weight pruning or whole-layer early exit.
