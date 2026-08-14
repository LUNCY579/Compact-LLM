# Phase 2 Selected Free-Running DEV Analysis

Role: **DEV MECHANISM CONFIRMATION / SAME-SPLIT**. Methods and temperatures were frozen before these free-running outputs were inspected.

## Free-running results

| Method | Correct | Exact acc. | Answerable | Abstention | Latency (s) | Forwards |
|---|---:|---:|---:|---:|---:|---:|
| Hard plurality | 1025/4050 | 25.31% | 24.19% | 32.59% | 1.919 | 34.15 |
| Raw probability mean | 1077/4050 | 26.59% | 21.45% | 60.00% | 2.625 | 44.14 |
| Calibrated probability mean | 1085/4050 | 26.79% | 23.36% | 49.07% | 2.363 | 38.56 |
| Reciprocal rank top-5 | 742/4050 | 18.32% | 17.78% | 21.85% | 1.966 | 33.42 |

## Paired family versus hard plurality

| Difference | Effect | Paired bootstrap 95% CI | Other-only/Hard-only | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|
| Raw probability mean − Hard | +1.28 pp | [+0.42, +2.15] | 182/130 | 3.816e-03 | 3.816e-03 |
| Calibrated probability mean − Hard | +1.48 pp | [+0.74, +2.22] | 151/91 | 1.387e-04 | 2.773e-04 |
| Reciprocal rank top-5 − Hard | -6.99 pp | [-7.95, -6.02] | 58/341 | 7.298e-50 | 2.189e-49 |

## Teacher-forced versus free-running

| Method | Local token | Local all-token example | Free exact | Local example gap vs Hard | Free gap vs Hard |
|---|---:|---:|---:|---:|---:|
| Hard plurality | 78.33% | 25.26% | 25.31% | +0.00 pp | +0.00 pp |
| Raw probability mean | 78.91% | 26.62% | 26.59% | +1.36 pp | +1.28 pp |
| Calibrated probability mean | 79.23% | 26.77% | 26.79% | +1.51 pp | +1.48 pp |
| Reciprocal rank top-5 | 76.34% | 18.47% | 18.32% | -6.79 pp | -6.99 pp |

The two regimes use the same DEV examples but different estimands. The comparison tests whether the local ordering survives path construction; it is not a subtraction of commensurate accuracy metrics.

Exact provenance and detailed divergence summaries are in `results/phase2_mechanism/selected_dev_free_run_analysis.json`.
