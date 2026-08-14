# Selected free-running DEV committee panel

Status: **DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE**

This targeted panel measures the teacher-forced best committee at each K and every K=6 leave-one-out under free-running hard plurality. It is same-split DEV model-selection evidence, not TEST confirmation.

## Teacher-forced-selected best-at-K trajectory and systems curve

| K | Committee | Correct | Exact acc. | Balanced answer-class | Latency (s) | Forwards | 1000×accuracy/forward |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | best_k1_m1 | 1057/4050 | 26.099% | 21.638% | 0.224 | 3.84 | 67.9131 |
| 2 | best_k2_m3 | 1013/4050 | 25.012% | 18.269% | 0.576 | 7.82 | 31.9841 |
| 3 | best_k3_m35 | 1101/4050 | 27.185% | 41.303% | 0.952 | 20.12 | 13.5145 |
| 4 | best_k4_m51 | 1107/4050 | 27.333% | 34.886% | 1.270 | 22.55 | 12.1206 |
| 5 | best_k5_m103 | 1048/4050 | 25.877% | 38.433% | 1.229 | 32.37 | 7.9936 |
| 6 | loo_without_grounding | 1069/4050 | 26.395% | 35.363% | 1.546 | 34.60 | 7.6296 |
| 7 | full_k7 | 1025/4050 | 25.309% | 28.390% | 1.919 | 34.15 | 7.4111 |

## Seven-comparison leave-one-out family

| Removed expert | Full minus K6 | Paired bootstrap 95% CI | Full-only/K6-only | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|
| abstention | +2.370 pp | [+1.654, +3.086] | 156/60 | 4.914e-11 | 1.965e-10 |
| general | +4.617 pp | [+3.704, +5.531] | 278/91 | 3.881e-23 | 1.940e-22 |
| grounding | -1.086 pp | [-1.729, -0.444] | 69/113 | 1.369e-03 | 2.739e-03 |
| procedure | -0.099 pp | [-0.716, +0.519] | 78/82 | 8.126e-01 | 8.126e-01 |
| social | +1.605 pp | [+1.012, +2.198] | 108/43 | 1.232e-07 | 3.695e-07 |
| track_a | +8.420 pp | [+7.383, +9.481] | 422/81 | 1.199e-56 | 8.392e-56 |
| track_b | +5.852 pp | [+4.938, +6.790] | 306/69 | 1.000e-36 | 6.003e-36 |

Answer-class and reasoning-family deltas are descriptive and stored in the machine artifact/strata CSV. Systems measurements are reference-decoder and hardware specific.
