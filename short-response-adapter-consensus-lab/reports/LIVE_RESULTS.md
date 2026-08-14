# Live Results

- Run: `phase2_cross_v212_full_k7`
- Completed example-method conditions: **3960/3960**
- Pending example-method conditions: **0**
- Status: **complete**
- Current best: `full_k7 / probability_mean`
- Correctness: **0.0303**
- Mean tokens: **7.24**
- Mean latency: **2.5110s**
- Mean agreement: **0.6874**

## Completed summaries

| Dataset | Committee | Method | K | N | Correct | Tokens | Latency | VRAM MiB | Agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| v212_test | full_k7 | calibrated_probability_mean | 7 | 1320 | 0.0015 | 5.48 | 2.1294 | 822.3 | 0.6039 |
| v212_test | full_k7 | hard_plurality | 7 | 1320 | 0.0008 | 5.59 | 1.7561 | 822.3 | 0.6259 |
| v212_test | full_k7 | probability_mean | 7 | 1320 | 0.0303 | 7.24 | 2.5110 | 822.3 | 0.6874 |

## Warnings

- Soft-fusion parameters were selected on DEV and were not retuned after sealed outcomes.
