# Phase 2 Final Synthesis Fact Packet

All numbers below are generated from the five strict long-run analysis JSON files. This packet is a transcription-safe source for the final narrative, not an independent statistical family.

## Same-split selected DEV methods

| Method | Correct/N | Free exact | Answerable | Abstention | Latency (s) | Forwards |
|---|---|---|---|---|---|---|
| hard_plurality | 1025/4050 | 25.309% | 24.188% | 32.593% | 1.9194 | 34.150 |
| probability_mean | 1077/4050 | 26.593% | 21.453% | 60.000% | 2.6251 | 44.136 |
| calibrated_probability_mean | 1085/4050 | 26.790% | 23.362% | 49.074% | 2.3634 | 38.560 |
| reciprocal_rank_top5 | 742/4050 | 18.321% | 17.778% | 21.852% | 1.9658 | 33.424 |

| Comparator minus hard | Difference | Paired 95% CI | Other-only/Hard-only | Holm p |
|---|---|---|---|---|
| probability_mean | +1.284 pp | +0.420 pp to +2.148 pp | 182/130 | 3.816e-03 |
| calibrated_probability_mean | +1.481 pp | +0.741 pp to +2.222 pp | 151/91 | 2.773e-04 |
| reciprocal_rank_top5 | -6.988 pp | -7.951 pp to -6.025 pp | 58/341 | 2.189e-49 |

| Method | Local token | Local all-token example | Free exact | Free gap vs hard |
|---|---|---|---|---|
| hard_plurality | 78.326% | 25.259% | 25.309% | +0.000 pp |
| probability_mean | 78.907% | 26.617% | 26.593% | +1.284 pp |
| calibrated_probability_mean | 79.232% | 26.765% | 26.790% | +1.481 pp |
| reciprocal_rank_top5 | 76.343% | 18.469% | 18.321% | -6.988 pp |

### Runtime provenance

- Stable partial top-k first raw line: **12239**; OOM retry telemetry first raw line: **12783**.
- Rows carrying retry telemetry: **3,418**; examples with retry: **0** (**0** total retries).

## Adaptive consultation

| Condition | Correct/N | Exact | Forwards | Forwards/token | Trigger/token | Latency (s) |
|---|---|---|---|---|---|---|
| hard_plurality | 1025/4050 | 25.309% | 34.150 | 7.000 | 0.000% | 1.9194 |
| adaptive_k3_to_k7 | 1017/4050 | 25.111% | 14.828 | 3.285 | 7.122% | 0.9278 |

Adaptive minus hard: **-0.198 pp** (paired 95% CI -0.765 pp to +0.370 pp; exact McNemar p=5.455e-01).

Expert-forward reduction versus hard: **56.579%**; latency ratio adaptive/hard: **0.4834**.

## Selected committee scaling and leave-one-out

| K | Committee | Exact | Balanced | Latency (s) | Forwards | 1000x acc./forward |
|---|---|---|---|---|---|---|
| 1 | best_k1_m1 | 26.099% | 21.638% | 0.2242 | 3.843 | 67.9131 |
| 2 | best_k2_m3 | 25.012% | 18.269% | 0.5757 | 7.820 | 31.9841 |
| 3 | best_k3_m35 | 27.185% | 41.303% | 0.9520 | 20.116 | 13.5145 |
| 4 | best_k4_m51 | 27.333% | 34.886% | 1.2699 | 22.551 | 12.1206 |
| 5 | best_k5_m103 | 25.877% | 38.433% | 1.2286 | 32.372 | 7.9936 |
| 6 | loo_without_grounding | 26.395% | 35.363% | 1.5463 | 34.596 | 7.6296 |
| 7 | full_k7 | 25.309% | 28.390% | 1.9194 | 34.150 | 7.4111 |

| Removed expert | Full minus K6 | Answerable delta | Abstention delta | Paired 95% CI | Full-only/K6-only | Holm p |
|---|---|---|---|---|---|---|
| abstention | +2.370 pp | -1.709 pp | +28.889 pp | +1.654 pp to +3.086 pp | 156/60 | 1.965e-10 |
| general | +4.617 pp | +7.009 pp | -10.926 pp | +3.704 pp to +5.531 pp | 278/91 | 1.940e-22 |
| grounding | -1.086 pp | +1.054 pp | -15.000 pp | -1.729 pp to -0.444 pp | 69/113 | 2.739e-03 |
| procedure | -0.099 pp | +1.595 pp | -11.111 pp | -0.716 pp to +0.519 pp | 78/82 | 8.126e-01 |
| social | +1.605 pp | -1.197 pp | +19.815 pp | +1.012 pp to +2.198 pp | 108/43 | 3.695e-07 |
| track_a | +8.420 pp | +10.256 pp | -3.519 pp | +7.383 pp to +9.481 pp | 422/81 | 8.392e-56 |
| track_b | +5.852 pp | +6.866 pp | -0.741 pp | +4.938 pp to +6.790 pp | 306/69 | 6.003e-36 |

## Selected calibrated probability on v2.2 TEST

| Condition | Correct/N | Exact | Answerable | Abstention | Latency (s) |
|---|---|---|---|---|---|
| hard_plurality | 699/4050 | 17.259% | 12.593% | 47.593% | 1.4321 |
| probability_mean | 571/4050 | 14.099% | 7.949% | 54.074% | 1.8728 |
| calibrated_probability_mean | 560/4050 | 13.827% | 10.912% | 32.778% | 1.9800 |

| Comparison | Difference | Paired 95% CI | Calibrated-only/Baseline-only | Holm p |
|---|---|---|---|---|
| calibrated_probability_mean_vs_hard_plurality | -3.432 pp | -4.272 pp to -2.617 pp | 81/220 | 1.073e-15 |
| calibrated_probability_mean_vs_probability_mean | -0.272 pp | -0.988 pp to +0.444 pp | 104/115 | 4.993e-01 |

## Fixed-method v2.1.2 transfer

| Condition | Correct/N | Exact | Answerable | Abstention | Latency (s) |
|---|---|---|---|---|---|
| backbone | 6/1320 | 0.455% | 0.000% | 5.000% | 0.2463 |
| track_a | 5/1320 | 0.379% | 0.417% | 0.000% | 0.1986 |
| abstention | 120/1320 | 9.091% | 0.000% | 100.000% | 0.4525 |
| hard_k7 | 1/1320 | 0.076% | 0.083% | 0.000% | 1.7561 |
| probability_k7 | 40/1320 | 3.030% | 0.000% | 33.333% | 2.5110 |
| calibrated_probability_k7 | 2/1320 | 0.152% | 0.167% | 0.000% | 2.1294 |

| Hard versus baseline | Difference | Paired 95% CI | Rescue/Harm | Holm p |
|---|---|---|---|---|
| backbone | -0.379 pp | -0.758 pp to +0.000 pp | 1/6 | 3.750e-01 |
| track_a | -0.303 pp | -0.682 pp to +0.000 pp | 1/5 | 4.375e-01 |
| abstention | -9.015 pp | -10.606 pp to -7.500 pp | 1/120 | 4.589e-34 |
| probability_k7 | -2.955 pp | -3.939 pp to -2.045 pp | 1/40 | 1.528e-10 |
| calibrated_probability_k7 | -0.076 pp | -0.379 pp to +0.152 pp | 1/2 | 1.000e+00 |

Claim boundary: fixed-method cross-corpus transfer only; the v2.1.2 corpus is not a globally untouched holdout, and no v2.2/v2.1.2 accuracy is pooled.

## Required-question and hypothesis routing

The machine-readable packet maps affected questions 6–7, 10–20 and hypotheses H1/H4/H6/H7/H8/H10 to exact nested fields. Questions not listed are adjudicated by the already-complete cornerstone and static mechanism analyses.

Exact nested statistics, subgroup records, caveats, and hashes remain in `results/phase2_mechanism/final_synthesis_facts.json` and its five source analyses.
