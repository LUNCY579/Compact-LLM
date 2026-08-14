# DEV confidence conditioned on expert token correctness

Status: **DEV_CONFIDENCE_CONDITIONING_COMPLETE**

This cache-only supplement reports each expert's top-1 confidence separately for correct and incorrect gold-prefix token decisions. It does not change the frozen calibration fit or selected runs.

| Expert | Correct n | Incorrect n | Confidence if correct | Confidence if incorrect | Difference |
|---|---:|---:|---:|---:|---:|
| track_a | 15010 | 3750 | 94.09% | 38.28% | +55.82 pp |
| track_b | 14771 | 3989 | 87.60% | 29.75% | +57.85 pp |
| general | 14310 | 4450 | 92.13% | 31.62% | +60.51 pp |
| grounding | 11179 | 7581 | 87.59% | 51.93% | +35.66 pp |
| procedure | 10145 | 8615 | 87.71% | 45.99% | +41.72 pp |
| abstention | 8577 | 10183 | 97.31% | 52.45% | +44.85 pp |
| social | 7850 | 10910 | 88.52% | 23.96% | +64.56 pp |

Cache SHA-256: `4b192a736451e81fd9e3dcd212f3f3aa97cda0a67c440536802453187377ee13`.

Correctness-conditioned values are descriptive DEV token statistics, not independent sequence-level effects.
