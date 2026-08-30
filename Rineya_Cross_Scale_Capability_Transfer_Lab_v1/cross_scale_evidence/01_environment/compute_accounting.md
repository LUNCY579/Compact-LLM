# Compute Accounting

- Frozen valid evaluation runs: 15
- Frozen valid evaluated rows: 869
- Frozen valid model-execution time: 0.444 CPU-hours wall-clock
- Frozen valid elapsed time including loads: 0.457 CPU-hours wall-clock
- Generated tokens: 5012
- Peak recorded post-run RSS: 3.059 GiB
- GPU hours: 0
- Final candidate: 596,049,920 parameters; 1,192,134,784 weight bytes; FP16 storage

## Instrumentation gaps

- Projection and merge wall time was not recorded in early manifests; commands completed in-session but no retrospective exact CPU-hour claim is made.
- Peak RAM is available only as process RSS after evaluation, not a sampled peak during projection or merge.
- TTFT was not separately instrumented; per-case end-to-end latency and aggregate tokens/s are preserved in predictions/summaries.
