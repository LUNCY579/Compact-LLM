# Phase 2 Teacher-Forced DEV Analysis (Archived Partial Checkpoint)

**Status: ARCHIVED / SUPERSEDED** by the 4,050-example analysis in
`reports/PHASE2_TEACHER_FORCED_DEV.md` and
`results/phase2_mechanism/teacher_forced_dev_analysis.json`.

This analysis uses **2,289 DEV examples / 10,638 gold next-token decisions**. All temperatures, committee choices, and comparator rankings here are `DEV MODEL-SELECTION / CALIBRATION`; they are not TEST confirmation.

## Expert calibration

| Expert | Fitted T | Top-1 acc. | Mean confidence | Conf.−acc. | ECE-15 | NLL | Brier | Entropy | Margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| track_a | 1.00 | 79.89% | 82.91% | +3.01 pp | 0.0415 | 0.9736 | 0.2487 | 0.7965 | 0.7849 |
| track_b | 1.00 | 78.49% | 75.48% | -3.01 pp | 0.0420 | 1.0777 | 0.2710 | 1.3755 | 0.7073 |
| general | 1.00 | 76.10% | 77.68% | +1.58 pp | 0.0254 | 1.4093 | 0.2932 | 1.2627 | 0.7308 |
| grounding | 1.25 | 59.77% | 72.97% | +13.20 pp | 0.1353 | 3.0786 | 0.5536 | 1.4692 | 0.6581 |
| procedure | 1.25 | 54.54% | 68.60% | +14.06 pp | 0.1406 | 3.6060 | 0.6271 | 1.8633 | 0.6271 |
| abstention | 2.00 | 45.54% | 72.84% | +27.30 pp | 0.2730 | 6.7736 | 0.7829 | 1.6018 | 0.6890 |
| social | 1.50 | 41.77% | 50.76% | +8.98 pp | 0.0898 | 5.9563 | 0.6727 | 3.0030 | 0.4431 |

The shared NLL-optimal grid temperature is **1.25**. Per-expert temperatures are fitted independently on the same DEV capture and must be frozen before any new TEST evaluation.

## Full K=7 local-decision comparators

| Method | Token acc. | All-token example acc. | First-token acc. | Answer-token acc. | EOS acc. | Candidate certification |
|---|---:|---:|---:|---:|---:|---:|
| hard_plurality | 78.38% | 25.78% | 26.29% | 72.52% | 99.79% | 100.00% |
| probability_mean | 78.97% | 27.12% | 27.63% | 73.24% | 99.87% | 100.00% |
| calibrated_probability_mean | 79.36% | 27.42% | 28.51% | 73.74% | 99.92% | 100.00% |
| shared_temperature_probability_mean | 78.28% | 23.98% | 25.62% | 72.36% | 99.87% | 100.00% |
| logit_mean | 73.34% | 18.08% | 22.77% | 66.43% | 98.53% | 99.99% |
| normalized_logit_mean | 73.63% | 18.54% | 22.73% | 66.73% | 98.83% | 100.00% |
| top2_approval | 73.25% | 14.78% | 17.41% | 66.71% | 97.15% | 100.00% |
| borda_top5 | 73.03% | 12.10% | 14.90% | 66.42% | 97.15% | 100.00% |
| reciprocal_rank_top5 | 76.63% | 19.67% | 20.47% | 70.31% | 99.71% | 100.00% |

Candidate certification proves that the selected probability/logit winner beats the maximum possible score of every token omitted from the top-k union. A rate below 100% marks positions whose soft/logit winner is approximate and requires a larger cache top-k before confirmation.

## Committee-size scaling

| Method | K | Committees | Mean example acc. | Best example acc. | Best members | Mean token acc. |
|---|---:|---:|---:|---:|---|---:|
| hard_plurality | 1 | 7 | 14.27% | 27.17% | track_a | 62.49% |
| hard_plurality | 2 | 21 | 14.26% | 25.45% | track_a, track_b | 67.20% |
| hard_plurality | 3 | 35 | 17.62% | 27.08% | track_a, track_b, abstention | 71.67% |
| hard_plurality | 4 | 35 | 18.70% | 27.54% | track_a, track_b, procedure, abstention | 74.10% |
| hard_plurality | 5 | 21 | 20.17% | 26.29% | track_a, track_b, general, abstention, social | 76.22% |
| hard_plurality | 6 | 7 | 23.13% | 27.17% | track_a, track_b, general, procedure, abstention, social | 77.39% |
| hard_plurality | 7 | 1 | 25.78% | 25.78% | track_a, track_b, general, grounding, procedure, abstention, social | 78.38% |
| probability_mean | 1 | 7 | 14.27% | 27.17% | track_a | 62.49% |
| probability_mean | 2 | 21 | 17.78% | 29.64% | track_a, social | 71.06% |
| probability_mean | 3 | 35 | 19.18% | 28.17% | track_a, track_b, social | 74.57% |
| probability_mean | 4 | 35 | 20.48% | 28.51% | track_a, track_b, procedure, abstention | 76.36% |
| probability_mean | 5 | 21 | 22.62% | 28.76% | track_a, track_b, general, procedure, abstention | 77.53% |
| probability_mean | 6 | 7 | 25.00% | 28.88% | track_a, track_b, general, procedure, abstention, social | 78.33% |
| probability_mean | 7 | 1 | 27.12% | 27.12% | track_a, track_b, general, grounding, procedure, abstention, social | 78.97% |
| calibrated_probability_mean | 1 | 7 | 14.27% | 27.17% | track_a | 62.49% |
| calibrated_probability_mean | 2 | 21 | 19.00% | 28.42% | track_a, social | 72.00% |
| calibrated_probability_mean | 3 | 35 | 21.91% | 29.64% | track_a, track_b, abstention | 75.95% |
| calibrated_probability_mean | 4 | 35 | 23.85% | 29.80% | track_a, track_b, procedure, abstention | 77.76% |
| calibrated_probability_mean | 5 | 21 | 25.38% | 29.76% | track_a, track_b, procedure, abstention, social | 78.67% |
| calibrated_probability_mean | 6 | 7 | 26.63% | 29.64% | track_a, track_b, general, procedure, abstention, social | 79.14% |
| calibrated_probability_mean | 7 | 1 | 27.42% | 27.42% | track_a, track_b, general, grounding, procedure, abstention, social | 79.36% |
| borda_top5 | 1 | 7 | 14.27% | 27.17% | track_a | 62.49% |
| borda_top5 | 2 | 21 | 12.10% | 25.83% | track_a, track_b | 64.80% |
| borda_top5 | 3 | 35 | 12.80% | 23.23% | track_a, track_b, general | 67.76% |
| borda_top5 | 4 | 35 | 13.66% | 22.73% | track_a, track_b, general, social | 69.97% |
| borda_top5 | 5 | 21 | 13.85% | 20.89% | track_a, track_b, general, procedure, abstention | 71.51% |
| borda_top5 | 6 | 7 | 13.11% | 15.74% | track_a, track_b, general, procedure, abstention, social | 72.44% |
| borda_top5 | 7 | 1 | 12.10% | 12.10% | track_a, track_b, general, grounding, procedure, abstention, social | 73.03% |

## Hard-K7 leave-one-out

| Removed expert | Full example acc. | K6 example acc. | Full−K6 | Full token acc. | K6 token acc. | Full−K6 token |
|---|---:|---:|---:|---:|---:|---:|
| track_a | 25.78% | 18.04% | +7.74 pp | 78.38% | 75.56% | +2.83 pp |
| track_b | 25.78% | 20.85% | +4.94 pp | 78.38% | 76.48% | +1.91 pp |
| general | 25.78% | 22.31% | +3.47 pp | 78.38% | 77.31% | +1.07 pp |
| grounding | 25.78% | 27.17% | -1.38 pp | 78.38% | 78.66% | -0.28 pp |
| procedure | 25.78% | 25.91% | -0.13 pp | 78.38% | 78.63% | -0.24 pp |
| abstention | 25.78% | 23.40% | +2.39 pp | 78.38% | 77.45% | +0.94 pp |
| social | 25.78% | 24.24% | +1.55 pp | 78.38% | 77.65% | +0.73 pp |

## Exact Shapley values on cached committees

| Expert | Hard standalone | Hard Shapley | Probability standalone | Probability Shapley |
|---|---:|---:|---:|---:|
| track_a | 27.17% | +10.54 pp | 27.17% | +11.15 pp |
| track_b | 23.73% | +8.01 pp | 23.73% | +8.10 pp |
| general | 19.55% | +6.49 pp | 19.55% | +6.29 pp |
| grounding | 7.58% | -0.83 pp | 7.58% | -0.52 pp |
| procedure | 8.29% | +0.90 pp | 8.29% | +0.56 pp |
| abstention | 13.56% | +1.98 pp | 13.56% | +1.27 pp |
| social | 0.00% | -1.30 pp | 0.00% | +0.27 pp |

## Mechanistic reading permitted by this DEV cache

- Hard K=7 local token accuracy: **78.38%**; raw probability mean: **78.97%**; calibrated probability mean: **79.36%**.
- Hard K=7 all-gold-token example accuracy: **25.78%**; raw probability mean: **27.12%**; calibrated probability mean: **27.42%**.
- Positive-temperature hard top-1 voting is mathematically invariant. The stored temperature interventions quantify how soft fusion changes when either all experts or one expert is rescaled.
- This cache establishes local next-token behavior only. A trajectory-correction conclusion still requires same-split free-running DEV evaluation of frozen methods.

## Provenance

- Cache SHA256: `9d1225ddbf3c58f5bb31918ffde0b96e5ad7cd0782ec65b6ab8b94450e371fad`
- Capture manifest SHA256: `1e78c6e438753aaa20b0ba0979dbfbfddd5aeb9f2d0fcc5e173d66448425bc5f`
- Analysis script SHA256: `ece69c51623c8fc9a0ceac3206bdec365195709643e826fd002ed9fd8984487e`
- Top-k: **64**
- Aggregated soft/logit decisions certified: **False**
