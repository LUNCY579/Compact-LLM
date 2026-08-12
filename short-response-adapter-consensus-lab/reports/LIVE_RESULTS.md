# Live Results

- Run: `v22_frozen_finalists`
- Completed example-method conditions: **256/256**
- Pending example-method conditions: **0**
- Status: **complete**
- Current best: `abstention_k1 / hard_plurality`
- Correctness: **0.5000**
- Mean tokens: **11.00**
- Mean latency: **0.7081s**
- Mean agreement: **1.0000**
- Current-best minus backbone correctness: **+0.5000**

## Completed summaries

| Dataset | Committee | Method | K | N | Correct | Tokens | Latency | VRAM MiB | Agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| v22_sealed | abstention_k1 | hard_plurality | 1 | 32 | 0.5000 | 11.00 | 0.7081 | 790.8 | 1.0000 |
| v22_sealed | abstention_k1 | probability_mean | 1 | 32 | 0.5000 | 11.00 | 0.7491 | 790.8 | 1.0000 |
| v22_sealed | backbone_k1 | hard_plurality | 1 | 32 | 0.0000 | 6.72 | 0.5723 | 790.5 | 1.0000 |
| v22_sealed | backbone_k1 | probability_mean | 1 | 32 | 0.0000 | 6.72 | 0.3972 | 790.5 | 1.0000 |
| v22_sealed | full_k7 | hard_plurality | 7 | 32 | 0.2500 | 6.12 | 3.8317 | 818.2 | 0.7007 |
| v22_sealed | full_k7 | probability_mean | 7 | 32 | 0.3750 | 8.19 | 5.3660 | 818.2 | 0.7616 |
| v22_sealed | track_a_k1 | hard_plurality | 1 | 32 | 0.1562 | 4.34 | 0.2507 | 790.1 | 1.0000 |
| v22_sealed | track_a_k1 | probability_mean | 1 | 32 | 0.1562 | 4.34 | 0.1982 | 790.1 | 1.0000 |

## Warnings

- This is a bounded frozen sealed sample, not the full official TEST split.
- Soft-fusion parameters were selected on DEV and were not retuned after sealed outcomes.
- The aggregate-best condition is abstention-degenerate; inspect answerable correctness before selection.
