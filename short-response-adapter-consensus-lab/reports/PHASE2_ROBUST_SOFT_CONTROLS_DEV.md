# Phase 2 robust soft-voting controls on DEV

Status: **DEV_ROBUST_SOFT_CONTROLS_COMPLETE**

These are cache-only, full-K=7, gold-prefix DEV robustness controls. They were added after DEV inspection and are not promoted into the frozen free-running selection.

## Definitions

- Entropy normalization means inverse-entropy expert weighting, followed by arithmetic probability pooling.
- Validation-weighted hard voting weights top-1 proposals by the frozen per-expert validation weights and retains the deployed confidence tie-break.
- Trimmed probability pooling drops the largest and smallest expert probability per candidate; median pooling takes the seven-expert median.
- Confidence clipping caps each expert's top-1 mass at 0.5, 0.7, or 0.9, exactly renormalizes that expert distribution, then arithmetic-pools experts.

## Results

| Method | Token accuracy | All-token example accuracy | First token | Answer token | EOS | Candidate certified |
|---|---:|---:|---:|---:|---:|---:|
| hard_plurality | 78.326% | 25.259% | 25.778% | 72.393% | 99.877% | 100.000% |
| probability_mean | 78.907% | 26.617% | 26.988% | 73.120% | 99.926% | 100.000% |
| validation_weighted_hard | 79.099% | 24.765% | 25.235% | 73.351% | 99.975% | 100.000% |
| entropy_normalized_probability_mean | 74.430% | 8.247% | 15.210% | 67.410% | 99.926% | 100.000% |
| trimmed_probability_mean | 78.886% | 26.198% | 26.593% | 73.093% | 99.926% | 100.000% |
| median_probability | 73.667% | 13.728% | 14.593% | 67.825% | 94.889% | 99.989% |
| confidence_clipped_probability_mean_0.5 | 78.843% | 26.444% | 26.963% | 73.039% | 99.926% | 100.000% |
| confidence_clipped_probability_mean_0.7 | 78.902% | 26.642% | 27.062% | 73.114% | 99.926% | 100.000% |
| confidence_clipped_probability_mean_0.9 | 78.907% | 26.617% | 26.988% | 73.120% | 99.926% | 100.000% |

The hard-plurality and raw-probability rows exactly reproduce the frozen full-committee analysis at both token and all-token-example levels. Every winner except two median-pooling positions is certified against tokens outside the cached top-64 union; the median row is therefore an explicitly approximate exploratory result.

## Paired all-token-example comparisons

| Reference | Other | Other-only | Reference-only | Difference | Exact McNemar p |
|---|---|---:|---:|---:|---:|
| hard_plurality | validation_weighted_hard | 105 | 125 | -0.494 pp | 0.210177 |
| probability_mean | validation_weighted_hard | 197 | 272 | -1.852 pp | 0.000618105 |
| hard_plurality | entropy_normalized_probability_mean | 150 | 839 | -17.012 pp | 1.03544e-116 |
| probability_mean | entropy_normalized_probability_mean | 76 | 820 | -18.370 pp | 1.98761e-158 |
| hard_plurality | trimmed_probability_mean | 115 | 77 | +0.938 pp | 0.00741866 |
| probability_mean | trimmed_probability_mean | 155 | 172 | -0.420 pp | 0.37629 |
| hard_plurality | median_probability | 126 | 593 | -11.531 pp | 2.9917e-73 |
| probability_mean | median_probability | 135 | 657 | -12.889 pp | 4.1269e-83 |
| hard_plurality | confidence_clipped_probability_mean_0.5 | 185 | 137 | +1.185 pp | 0.00871057 |
| probability_mean | confidence_clipped_probability_mean_0.5 | 12 | 19 | -0.173 pp | 0.281042 |
| hard_plurality | confidence_clipped_probability_mean_0.7 | 184 | 128 | +1.383 pp | 0.00180192 |
| probability_mean | confidence_clipped_probability_mean_0.7 | 2 | 1 | +0.025 pp | 1 |
| hard_plurality | confidence_clipped_probability_mean_0.9 | 183 | 128 | +1.358 pp | 0.00214772 |
| probability_mean | confidence_clipped_probability_mean_0.9 | 0 | 0 | +0.000 pp | 1 |

P-values are unadjusted descriptive diagnostics for exploratory DEV controls. No robust-control variant changes the locked free-running manifest.

## Provenance and limitations

- Frozen cache SHA-256: `4b192a736451e81fd9e3dcd212f3f3aa97cda0a67c440536802453187377ee13`.
- Frozen teacher-analysis SHA-256: `19fa2be96094f210d8f9e3006547bc649389a60383e594edf5051b6436b24b62`.
- No new model forwards were executed.
- Gold-prefix outcomes are not free-running sequence outcomes.
- Median pooling is uncertified at 2/18,760 positions and is not used for exact inference.
- The tested entropy, trimmed, median, validation-weighted, and clipping definitions are explicitly bounded; no post-hoc winner was promoted.
