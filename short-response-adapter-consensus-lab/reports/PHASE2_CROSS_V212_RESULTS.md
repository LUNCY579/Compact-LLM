# Phase 2 v2.1.2 Fixed-Method Cross-Corpus Results

Role: **FIXED-METHOD CROSS-CORPUS TRANSFER WITH PROVENANCE CAVEATS**. Fixed-method cross-corpus transfer only; not globally untouched or template-independent holdout evidence; do not aggregate with v2.2.

| Condition | Correct | Exact acc. | Answerable | Abstention | Latency (s) | Forwards |
|---|---:|---:|---:|---:|---:|---:|
| backbone | 6/1320 | 0.45% | 0.00% | 5.00% | 0.246 | 7.65 |
| track_a | 5/1320 | 0.38% | 0.42% | 0.00% | 0.199 | 4.52 |
| abstention | 120/1320 | 9.09% | 0.00% | 100.00% | 0.452 | 10.92 |
| hard_k7 | 1/1320 | 0.08% | 0.08% | 0.00% | 1.756 | 39.11 |
| probability_k7 | 40/1320 | 3.03% | 0.00% | 33.33% | 2.511 | 50.69 |
| calibrated_probability_k7 | 2/1320 | 0.15% | 0.17% | 0.00% | 2.129 | 38.33 |

| Comparison | Effect | Paired bootstrap 95% CI | Rescue/harm | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|
| Hard − backbone | -0.38 pp | [-0.76, +0.00] | 1/6 | 1.250e-01 | 3.750e-01 |
| Hard − track_a | -0.30 pp | [-0.68, +0.00] | 1/5 | 2.188e-01 | 4.375e-01 |
| Hard − abstention | -9.02 pp | [-10.61, -7.50] | 1/120 | 9.178e-35 | 4.589e-34 |
| Hard − probability_k7 | -2.95 pp | [-3.94, -2.05] | 1/40 | 3.820e-11 | 1.528e-10 |
| Hard − calibrated_probability_k7 | -0.08 pp | [-0.38, +0.15] | 1/2 | 1.000e+00 | 1.000e+00 |

## Mandatory caveats

- v2.1.2 TEST contains exact prompt-and-answer repetitions from v2.1.2 TRAIN; this does not leak into the seven v2.2-trained adapters, but prevents a template-independent holdout claim.
- The official project previously evaluated a different selected D_greedy dense model on v2.1.2 TEST; the split is therefore not globally untouched.

No v2.2 and v2.1.2 accuracy is pooled.
