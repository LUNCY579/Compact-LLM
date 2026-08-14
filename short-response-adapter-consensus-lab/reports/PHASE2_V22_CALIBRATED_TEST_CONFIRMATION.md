# v2.2 calibrated-probability TEST confirmation

Status: **V22_CALIBRATED_TEST_CONFIRMATION_COMPLETE**

The calibrated K=7 method and all temperatures were fixed from DEV before this output existed. Hard/raw anchors come from the verified cornerstone; their outcomes were already known, and the archived n=32 exposure caveat remains.

## Full TEST conditions

| Condition | Correct | Exact accuracy | Answerable | Abstention | Latency (s) | Forwards |
|---|---:|---:|---:|---:|---:|---:|
| hard_plurality | 699/4050 | 17.259% | 12.593% | 47.593% | 1.432 | 40.11 |
| probability_mean | 571/4050 | 14.099% | 7.949% | 54.074% | 1.873 | 52.47 |
| calibrated_probability_mean | 560/4050 | 13.827% | 10.912% | 32.778% | 1.980 | 43.67 |

## Predeclared paired family

| Comparison | Difference | Paired bootstrap 95% CI | Calibrated-only/Baseline-only | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|
| calibrated − hard_plurality | -3.432 pp | [-4.272, -2.617] | 81/220 | 5.363e-16 | 1.073e-15 |
| calibrated − probability_mean | -0.272 pp | [-0.988, +0.444] | 104/115 | 4.993e-01 | 4.993e-01 |

The two global comparisons constitute this confirmation family. Answerable/abstention cuts are descriptive, and v2.2 results are not aggregated with v2.1.2.

Cornerstone SHA-256: `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`.
Calibrated run SHA-256: `026a7e8dfe73b6fe7cae7f4bdc1f229abed95ad246ae5658869720776e3c4857`.
