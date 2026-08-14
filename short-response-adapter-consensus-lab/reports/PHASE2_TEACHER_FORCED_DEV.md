# Phase 2 Teacher-Forced DEV Analysis

**Status: DEV_TEACHER_FORCED_ANALYSIS_COMPLETE_WITH_UNCERTIFIED_SOFT_POSITIONS**

This analysis uses **4,050 DEV examples / 18,760 gold next-token decisions**. All temperatures, committee choices, and comparator rankings here are `DEV MODEL-SELECTION / CALIBRATION`; they are not TEST confirmation.

## Expert calibration

| Expert | Fitted T | Top-1 acc. | Mean confidence | Conf.−acc. | ECE-15 | NLL | Brier | Entropy | Margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| track_a | 1.00 | 80.01% | 82.94% | +2.93 pp | 0.0422 | 0.9695 | 0.2470 | 0.8005 | 0.7856 |
| track_b | 1.00 | 78.74% | 75.30% | -3.44 pp | 0.0438 | 1.0654 | 0.2692 | 1.3886 | 0.7050 |
| general | 1.00 | 76.28% | 77.78% | +1.50 pp | 0.0250 | 1.4050 | 0.2906 | 1.2640 | 0.7319 |
| grounding | 1.25 | 59.59% | 73.18% | +13.59 pp | 0.1388 | 3.1083 | 0.5600 | 1.4619 | 0.6611 |
| procedure | 1.25 | 54.08% | 68.55% | +14.47 pp | 0.1447 | 3.6278 | 0.6304 | 1.8713 | 0.6274 |
| abstention | 2.00 | 45.72% | 72.96% | +27.24 pp | 0.2724 | 6.7605 | 0.7797 | 1.5972 | 0.6907 |
| social | 1.50 | 41.84% | 50.98% | +9.13 pp | 0.0913 | 5.9562 | 0.6712 | 2.9865 | 0.4454 |

The shared NLL-optimal grid temperature is **1.25**. Per-expert temperatures are fitted independently on the same DEV capture and must be frozen before any new TEST evaluation.

## Full K=7 local-decision comparators

| Method | Token acc. | All-token example acc. | First-token acc. | Answer-token acc. | EOS acc. | Candidate certification |
|---|---:|---:|---:|---:|---:|---:|
| hard_plurality | 78.33% | 25.26% | 25.78% | 72.39% | 99.88% | 100.00% |
| probability_mean | 78.91% | 26.62% | 26.99% | 73.12% | 99.93% | 100.00% |
| calibrated_probability_mean | 79.23% | 26.77% | 27.60% | 73.53% | 99.95% | 100.00% |
| shared_temperature_probability_mean | 78.24% | 23.85% | 25.21% | 72.26% | 99.93% | 100.00% |
| logit_mean | 73.07% | 17.51% | 22.15% | 66.08% | 98.44% | 99.99% |
| normalized_logit_mean | 73.33% | 17.80% | 21.88% | 66.29% | 98.89% | 100.00% |
| top2_approval | 73.01% | 13.98% | 16.72% | 66.32% | 97.28% | 100.00% |
| borda_top5 | 72.77% | 11.31% | 13.63% | 66.10% | 97.01% | 100.00% |
| reciprocal_rank_top5 | 76.34% | 18.47% | 19.19% | 69.91% | 99.70% | 100.00% |

Candidate certification proves that the selected probability/logit winner beats the maximum possible score of every token omitted from the top-k union. A rate below 100% marks positions whose soft/logit winner is approximate and requires a larger cache top-k before confirmation.

## Gold-prefix complementarity and error dependence

- Dynamic per-token oracle accuracy: **86.36%** (gain over best individual: **+6.35 pp**).
- Dynamic oracle all-token example accuracy: **48.12%**.
- All seven wrong at the same gold-prefix token: **13.64%**; all seven correct: **29.19%**.

| Expert pair | Top-1 disagreement | Complementary correctness | Double fault | Correctness phi | Q statistic |
|---|---:|---:|---:|---:|---:|
| procedure / abstention | 67.19% | 37.55% | 31.33% | 0.258 | 0.489 |
| procedure / social | 66.61% | 32.20% | 35.94% | 0.376 | 0.675 |
| grounding / social | 66.58% | 38.68% | 29.94% | 0.266 | 0.522 |
| grounding / abstention | 64.98% | 38.06% | 28.32% | 0.261 | 0.504 |
| general / social | 59.08% | 39.14% | 21.37% | 0.361 | 0.814 |
| track_a / social | 58.85% | 41.24% | 18.45% | 0.346 | 0.849 |
| track_b / social | 57.80% | 40.45% | 19.48% | 0.353 | 0.838 |
| general / abstention | 56.76% | 37.77% | 20.12% | 0.342 | 0.746 |
| track_a / abstention | 56.22% | 39.38% | 17.45% | 0.331 | 0.779 |
| track_b / abstention | 55.15% | 38.75% | 18.40% | 0.336 | 0.769 |
| abstention / social | 44.94% | 7.29% | 52.57% | 0.855 | 0.991 |
| track_b / procedure | 42.56% | 27.79% | 19.70% | 0.487 | 0.924 |
| general / procedure | 41.85% | 26.02% | 21.81% | 0.515 | 0.922 |
| track_a / procedure | 41.39% | 28.28% | 18.82% | 0.484 | 0.938 |
| grounding / procedure | 34.55% | 12.44% | 36.95% | 0.752 | 0.967 |
| general / grounding | 34.13% | 22.38% | 20.87% | 0.541 | 0.910 |
| track_b / grounding | 33.37% | 23.10% | 19.29% | 0.533 | 0.928 |
| track_a / grounding | 33.30% | 23.59% | 18.41% | 0.526 | 0.937 |
| track_b / general | 17.05% | 5.43% | 19.78% | 0.846 | 0.992 |
| track_a / general | 13.81% | 4.71% | 19.50% | 0.867 | 0.997 |
| track_a / track_b | 13.73% | 5.25% | 18.00% | 0.840 | 0.991 |

The dynamic oracle is an unattainable switching upper bound. Pairwise disagreement can reflect useful complementarity or shared low quality; committee-diversity correlations are therefore stored by fixed K and treated as exploratory, not as a causal specialization test.

## Committee-size scaling

| Method | K | Committees | Mean example acc. | Best example acc. | Best members | Mean token acc. |
|---|---:|---:|---:|---:|---|---:|
| hard_plurality | 1 | 7 | 13.92% | 26.07% | track_a | 62.32% |
| hard_plurality | 2 | 21 | 13.72% | 24.99% | track_a, track_b | 66.98% |
| hard_plurality | 3 | 35 | 17.09% | 27.23% | track_a, track_b, abstention | 71.49% |
| hard_plurality | 4 | 35 | 18.22% | 27.31% | track_a, track_b, procedure, abstention | 73.90% |
| hard_plurality | 5 | 21 | 19.39% | 25.80% | track_a, track_b, general, abstention, social | 76.04% |
| hard_plurality | 6 | 7 | 22.22% | 26.42% | track_a, track_b, general, procedure, abstention, social | 77.19% |
| hard_plurality | 7 | 1 | 25.26% | 25.26% | track_a, track_b, general, grounding, procedure, abstention, social | 78.33% |
| probability_mean | 1 | 7 | 13.92% | 26.07% | track_a | 62.32% |
| probability_mean | 2 | 21 | 17.33% | 29.04% | track_a, social | 70.92% |
| probability_mean | 3 | 35 | 18.72% | 27.90% | track_a, track_b, social | 74.47% |
| probability_mean | 4 | 35 | 19.95% | 28.32% | track_a, track_b, procedure, abstention | 76.22% |
| probability_mean | 5 | 21 | 22.11% | 28.69% | track_a, track_b, general, procedure, abstention | 77.41% |
| probability_mean | 6 | 7 | 24.45% | 28.67% | track_a, track_b, general, procedure, abstention, social | 78.20% |
| probability_mean | 7 | 1 | 26.62% | 26.62% | track_a, track_b, general, grounding, procedure, abstention, social | 78.91% |
| calibrated_probability_mean | 1 | 7 | 13.92% | 26.07% | track_a | 62.32% |
| calibrated_probability_mean | 2 | 21 | 18.60% | 27.90% | track_a, abstention | 71.86% |
| calibrated_probability_mean | 3 | 35 | 21.49% | 29.65% | track_a, track_b, abstention | 75.87% |
| calibrated_probability_mean | 4 | 35 | 23.38% | 30.02% | track_a, track_b, procedure, abstention | 77.66% |
| calibrated_probability_mean | 5 | 21 | 24.92% | 29.85% | track_a, track_b, procedure, abstention, social | 78.55% |
| calibrated_probability_mean | 6 | 7 | 26.09% | 29.28% | track_a, track_b, general, procedure, abstention, social | 79.00% |
| calibrated_probability_mean | 7 | 1 | 26.77% | 26.77% | track_a, track_b, general, grounding, procedure, abstention, social | 79.23% |
| borda_top5 | 1 | 7 | 13.92% | 26.07% | track_a | 62.32% |
| borda_top5 | 2 | 21 | 11.64% | 25.19% | track_a, track_b | 64.59% |
| borda_top5 | 3 | 35 | 12.39% | 22.84% | track_a, track_b, general | 67.56% |
| borda_top5 | 4 | 35 | 12.95% | 22.52% | track_a, track_b, general, social | 69.69% |
| borda_top5 | 5 | 21 | 13.02% | 19.70% | track_a, track_b, general, procedure, abstention | 71.23% |
| borda_top5 | 6 | 7 | 12.22% | 15.04% | track_a, track_b, general, procedure, abstention, social | 72.20% |
| borda_top5 | 7 | 1 | 11.31% | 11.31% | track_a, track_b, general, grounding, procedure, abstention, social | 72.77% |

## Hard-K7 leave-one-out

| Removed expert | Full example acc. | K6 example acc. | Full−K6 | Full token acc. | K6 token acc. | Full−K6 token |
|---|---:|---:|---:|---:|---:|---:|
| track_a | 25.26% | 16.99% | +8.27 pp | 78.33% | 75.28% | +3.05 pp |
| track_b | 25.26% | 19.36% | +5.90 pp | 78.33% | 76.11% | +2.21 pp |
| general | 25.26% | 20.79% | +4.47 pp | 78.33% | 76.95% | +1.38 pp |
| grounding | 25.26% | 26.42% | -1.16 pp | 78.33% | 78.40% | -0.07 pp |
| procedure | 25.26% | 25.38% | -0.12 pp | 78.33% | 78.51% | -0.19 pp |
| abstention | 25.26% | 22.91% | +2.35 pp | 78.33% | 77.49% | +0.83 pp |
| social | 25.26% | 23.70% | +1.56 pp | 78.33% | 77.61% | +0.71 pp |

## Exact Shapley values on cached committees

| Expert | Hard standalone | Hard Shapley | Probability standalone | Probability Shapley |
|---|---:|---:|---:|---:|
| track_a | 26.07% | +10.23 pp | 26.07% | +10.99 pp |
| track_b | 23.90% | +8.26 pp | 23.90% | +7.98 pp |
| general | 19.26% | +6.52 pp | 19.26% | +6.38 pp |
| grounding | 7.11% | -1.02 pp | 7.11% | -0.80 pp |
| procedure | 7.75% | +0.64 pp | 7.75% | +0.43 pp |
| abstention | 13.33% | +1.95 pp | 13.33% | +1.36 pp |
| social | 0.02% | -1.32 pp | 0.02% | +0.28 pp |

## Mechanistic reading permitted by this DEV cache

- Hard K=7 local token accuracy: **78.33%**; raw probability mean: **78.91%**; calibrated probability mean: **79.23%**.
- Hard K=7 all-gold-token example accuracy: **25.26%**; raw probability mean: **26.62%**; calibrated probability mean: **26.77%**.
- Positive-temperature hard top-1 voting is mathematically invariant. The stored temperature interventions quantify how soft fusion changes when either all experts or one expert is rescaled.
- This cache establishes local next-token behavior only. A trajectory-correction conclusion still requires same-split free-running DEV evaluation of frozen methods.

## Provenance

- Cache SHA256: `4b192a736451e81fd9e3dcd212f3f3aa97cda0a67c440536802453187377ee13`
- Capture manifest SHA256: `1e78c6e438753aaa20b0ba0979dbfbfddd5aeb9f2d0fcc5e173d66448425bc5f`
- Analysis script SHA256: `e6029c42fcd422ffe355ddbcb7c0cfd4f7b937138f8f4d5f2cca16f26db8547a`
- Top-k: **64**
- Aggregated soft/logit decisions certified: **False**
