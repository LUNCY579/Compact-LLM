# TGSA Phase 1 model-identifiability analysis

**Decision:** `inconclusive`

**Confirmatory configuration:** `True`  
**Registered confirmatory matrix:** `True`  
**Interval estimator:** `crossed_cluster_t`  
**Alignment falsifier:** `replay_derangement`

At most: controlled tiny-decoder Phase-1 evidence that an aligned finite-difference control signal adds capability beyond an information-matched sequential-history control on the registered synthetic task families. This analysis cannot establish Qwen/GQA, natural-language, long-context, efficiency, generalization, novelty, or paper-primary claims.

## Registered comparisons

| Comparison | Mean accuracy delta | Confidence interval | P(delta > 0) |
|---|---:|---:|---:|
| `learned_delta_minus_history_control` | +0.0014 | [-0.0021, +0.0048] | 0.7901 |
| `learned_delta_minus_cascade` | +0.0004 | [-0.0025, +0.0034] | 0.6191 |
| `aligned_delta_minus_replayed_delta` | +0.0019 | [-0.0003, +0.0040] | 0.9593 |
| `direct_lookup_delta_minus_history_control` | -0.0007 | [-0.0052, +0.0038] | 0.3767 |

## Advancement gates

- **full_registered_matrix:** PASS — selected arms=('cascade', 'history_control', 'learned_delta'); selected seeds=(101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020)
- **H1a_finite_difference_beats_information_matched_history:** FAIL — mean=+0.0014, lower=-0.0021, upper=+0.0048, powered_for=+0.0200, observed_at_or_above_powered_effect=False
- **H1b_finite_difference_beats_cascade:** FAIL — mean=+0.0004, lower=-0.0025, upper=+0.0034, powered_for=+0.0200, observed_at_or_above_powered_effect=False
- **H1c_trajectory_alignment_is_causal:** FAIL — aligned-minus-replayed mean=+0.0019, lower=-0.0003, upper=+0.0040, powered_for=+0.0150, observed_at_or_above_powered_effect=False
- **direct_lookup_noninferiority:** PASS — lower=-0.0052, margin=-0.0100
- **alignment_falsifier_coverage:** PASS — minimum per-seed length-matched derangement coverage=1.0000, threshold=0.9500
- **paired_initialization_and_artifact_integrity:** PASS — closed evidence chain verified: 2000/2000 checks passed

## Interpretation guardrail

A positive decision supports only the controlled Phase-1 statement that an aligned finite-difference control signal improved these trained synthetic tasks beyond an information-matched sequential-history control, and that replaying another same-length example's aligned control reduced the learned-delta model. It does not establish Qwen-scale, natural-language, long-context, efficiency, novelty, or publication-primary claims. A `smoke_only` decision is never evidence.
