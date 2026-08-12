# Live Results

- Run: `v22_frozen_finalists`
- Completed example-method conditions: **32400/32400**
- Pending example-method conditions: **0**
- Status: **complete**
- Current best: `full_k7 / hard_plurality`
- Correctness: **0.1726**
- Mean tokens: **5.73**
- Mean latency: **1.4321s**
- Mean agreement: **0.6861**
- Current-best minus backbone correctness: **+0.1723**

## Completed summaries

| Dataset | Committee | Method | K | N | Correct | Tokens | Latency | VRAM MiB | Agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| v22_sealed | abstention_k1 | hard_plurality | 1 | 4050 | 0.1301 | 10.92 | 0.4023 | 793.9 | 1.0000 |
| v22_sealed | abstention_k1 | probability_mean | 1 | 4050 | 0.1301 | 10.92 | 0.4021 | 793.9 | 1.0000 |
| v22_sealed | backbone_k1 | hard_plurality | 1 | 4050 | 0.0002 | 6.96 | 0.3038 | 793.6 | 1.0000 |
| v22_sealed | backbone_k1 | probability_mean | 1 | 4050 | 0.0002 | 6.96 | 0.3001 | 793.6 | 1.0000 |
| v22_sealed | full_k7 | hard_plurality | 7 | 4050 | 0.1726 | 5.73 | 1.4321 | 818.2 | 0.6861 |
| v22_sealed | full_k7 | probability_mean | 7 | 4050 | 0.1410 | 7.50 | 1.8728 | 818.2 | 0.7380 |
| v22_sealed | track_a_k1 | hard_plurality | 1 | 4050 | 0.1459 | 4.03 | 0.1510 | 793.4 | 1.0000 |
| v22_sealed | track_a_k1 | probability_mean | 1 | 4050 | 0.1459 | 4.03 | 0.1521 | 793.4 | 1.0000 |

## Warnings

- This is a bounded frozen sealed sample, not the full official TEST split.
- Soft-fusion parameters were selected on DEV and were not retuned after sealed outcomes.
