# PHASE 2B — PROGRESSIVE STRUCTURAL REDUCTION REPORT

## Purpose

Test which complete layers can be removed, optionally recovered, and combined
progressively. This structural experiment remains distinct from selective-layer
training.

## Evidence boundary

- Probe cases: 150, selected deterministically from development data.
- Sealed evaluation used: **No**.
- Quality evaluator: **4.5**.
- Relative opportunity profile: **balanced**.
- Full-reference quality: 0.759.
- Full-reference target NLL: 4.8624.
- Every current layer is tested as a raw deletion at each progressive stage.
- Recovery first evaluates the strongest 6 raw candidates per stage.
- If focused recovery finds no eligible candidate, every remaining raw branch is recovery-tested once before the frontier is declared terminal.
- Resume continues directly from the last accepted frontier model; completed stages are not reconstructed.

## Stage 1 individual-layer results

| Dropped original layer | Variant | Quality | Weight-memory gain | Quality opportunity loss | Opportunity margin | Eval VRAM change | Generation latency change | Target NLL | Profile | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | raw deletion | 0.000 | 2.64% | 68.67% | -66.03% | +2.60% | -409.77% | 13.7890 | balanced | NOT WORTHWHILE |
| 1 | raw deletion | 0.010 | 2.64% | 68.76% | -66.12% | +2.60% | -186.73% | 11.5138 | balanced | NOT WORTHWHILE |
| 2 | raw deletion | 0.126 | 2.64% | 60.39% | -57.75% | +2.60% | -90.36% | 10.2266 | balanced | NOT WORTHWHILE |
| 3 | raw deletion | 0.685 | 2.64% | 8.82% | -6.18% | +2.60% | -12.52% | 5.2377 | balanced | NOT WORTHWHILE |
| 4 | raw deletion | 0.694 | 2.64% | 6.83% | -4.19% | +2.60% | -39.68% | 5.0194 | balanced | NOT WORTHWHILE |
| 5 | raw deletion | 0.713 | 2.64% | 5.95% | -3.31% | +2.60% | -29.84% | 5.0005 | balanced | NOT WORTHWHILE |
| 6 | raw deletion | 0.705 | 2.64% | 6.86% | -4.22% | +2.60% | -30.08% | 5.0072 | balanced | NOT WORTHWHILE |
| 7 | raw deletion | 0.712 | 2.64% | 4.82% | -2.18% | +2.60% | -42.01% | 4.7547 | balanced | NOT WORTHWHILE |
| 8 | raw deletion | 0.724 | 2.64% | 4.37% | -1.73% | +2.60% | -48.35% | 4.8559 | balanced | NOT WORTHWHILE |
| 9 | raw deletion | 0.717 | 2.64% | 4.68% | -2.04% | +2.60% | -16.84% | 4.8117 | balanced | NOT WORTHWHILE |
| 10 | raw deletion | 0.696 | 2.64% | 6.42% | -3.78% | +2.60% | -21.86% | 4.7447 | balanced | NOT WORTHWHILE |
| 11 | raw deletion | 0.729 | 2.64% | 2.63% | +0.01% | +2.60% | -57.40% | 4.9065 | balanced | WORTHWHILE |
| 12 | raw deletion | 0.720 | 2.64% | 3.88% | -1.24% | +2.60% | -28.07% | 4.4333 | balanced | NOT WORTHWHILE |
| 13 | raw deletion | 0.734 | 2.64% | 1.83% | +0.81% | +2.60% | -38.19% | 4.9410 | balanced | WORTHWHILE |
| 13 | deletion + recovery SFT | 0.803 | 2.64% | 0.08% | +2.56% | +0.00% | -193.45% | 3.1213 | balanced | WORTHWHILE |
| 14 | raw deletion | 0.788 | 2.64% | 0.00% | +2.64% | +2.60% | -34.33% | 4.5488 | balanced | WORTHWHILE |
| 14 | deletion + recovery SFT | 0.754 | 2.64% | 2.57% | +0.07% | +0.00% | -178.30% | 3.0788 | balanced | WORTHWHILE |
| 15 | raw deletion | 0.727 | 2.64% | 3.63% | -0.99% | +2.60% | -16.51% | 4.5838 | balanced | NOT WORTHWHILE |
| 16 | raw deletion | 0.731 | 2.64% | 3.30% | -0.66% | +2.60% | -50.27% | 4.7063 | balanced | NOT WORTHWHILE |
| 17 | raw deletion | 0.666 | 2.64% | 7.06% | -4.42% | +2.60% | -28.19% | 4.8067 | balanced | NOT WORTHWHILE |
| 18 | raw deletion | 0.801 | 2.64% | 2.53% | +0.11% | +2.60% | -34.37% | 4.6834 | balanced | WORTHWHILE |
| 18 | deletion + recovery SFT | 0.781 | 2.64% | 0.25% | +2.39% | +0.00% | -156.60% | 3.1359 | balanced | WORTHWHILE |
| 19 | raw deletion | 0.716 | 2.64% | 4.66% | -2.02% | +2.60% | -24.81% | 4.7932 | balanced | NOT WORTHWHILE |
| 20 | raw deletion | 0.711 | 2.64% | 4.54% | -1.90% | +2.60% | -49.13% | 4.9135 | balanced | NOT WORTHWHILE |
| 21 | raw deletion | 0.674 | 2.64% | 10.65% | -8.01% | +2.60% | -39.13% | 5.0163 | balanced | NOT WORTHWHILE |
| 22 | raw deletion | 0.748 | 2.64% | 1.16% | +1.48% | +2.60% | -48.13% | 4.8566 | balanced | WORTHWHILE |
| 22 | deletion + recovery SFT | 0.751 | 2.64% | 1.86% | +0.78% | +0.00% | -188.99% | 3.1491 | balanced | WORTHWHILE |
| 23 | raw deletion | 0.733 | 2.64% | 3.00% | -0.36% | +2.60% | -30.25% | 4.7903 | balanced | NOT WORTHWHILE |
| 24 | raw deletion | 0.743 | 2.64% | 1.61% | +1.02% | +2.60% | -37.92% | 4.7771 | balanced | WORTHWHILE |
| 24 | deletion + recovery SFT | 0.752 | 2.64% | 2.27% | +0.37% | +0.00% | -204.14% | 3.1721 | balanced | WORTHWHILE |
| 25 | raw deletion | 0.732 | 2.64% | 4.44% | -1.80% | +2.60% | -38.20% | 4.9712 | balanced | NOT WORTHWHILE |
| 26 | raw deletion | 0.745 | 2.64% | 2.62% | +0.02% | +2.60% | -38.93% | 4.7626 | balanced | WORTHWHILE |
| 26 | deletion + recovery SFT | 0.785 | 2.64% | 0.58% | +2.06% | +0.00% | -180.65% | 3.2942 | balanced | WORTHWHILE |
| 27 | raw deletion | 0.651 | 2.64% | 13.42% | -10.78% | +2.60% | -419.98% | 4.2455 | balanced | NOT WORTHWHILE |

## Accepted progressive frontier

| Stage | Original layers dropped so far | Remaining layers | Quality | Cumulative memory gain | Cumulative quality loss | Cumulative margin | Incremental memory gain | Incremental quality loss | Incremental margin | Eval VRAM change | Generation latency change | Selected form |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | [14] | 27 | 0.788 | 2.64% | 0.00% | +2.64% | 2.64% | 0.00% | +2.64% | +2.60% | -34.33% | raw deletion |
| 2 | [14, 24] | 26 | 0.761 | 5.28% | 0.22% | +5.06% | 2.71% | 2.61% | +0.10% | +5.21% | +3.90% | raw deletion |
| 3 | [14, 24, 6] | 25 | 0.757 | 7.92% | 1.65% | +6.27% | 2.79% | 1.83% | +0.96% | +4.90% | -94.09% | deletion + recovery SFT |
| 4 | [14, 24, 6, 23] | 24 | 0.757 | 10.56% | 2.42% | +8.14% | 2.87% | 0.98% | +1.88% | +7.62% | -83.24% | deletion + recovery SFT |
| 5 | [14, 24, 6, 23, 8] | 23 | 0.726 | 13.20% | 3.53% | +9.67% | 2.95% | 2.17% | +0.78% | +10.34% | -105.16% | deletion + recovery SFT |
| 6 | [14, 24, 6, 23, 8, 10] | 22 | 0.705 | 15.84% | 5.58% | +10.26% | 3.04% | 2.24% | +0.80% | +13.06% | -76.17% | deletion + recovery SFT |
| 7 | [14, 24, 6, 23, 8, 10, 18] | 21 | 0.686 | 18.47% | 6.90% | +11.58% | 3.14% | 2.25% | +0.89% | +15.78% | -57.28% | deletion + recovery SFT |
| 8 | [14, 24, 6, 23, 8, 10, 18, 9] | 20 | 0.662 | 21.11% | 9.55% | +11.56% | 3.24% | 3.03% | +0.21% | +18.50% | -93.01% | deletion + recovery SFT |
| 9 | [14, 24, 6, 23, 8, 10, 18, 9, 7] | 19 | 0.642 | 23.75% | 11.24% | +12.52% | 3.35% | 2.01% | +1.34% | +21.22% | -77.93% | deletion + recovery SFT |

## Terminal condition

- Reason: `no_marginally_worthwhile_deletion_after_exhaustive_recovery`
- Terminal stage attempted: `10`
- Accepted frontier stages: **9**
- Exhaustive recovery triggered at terminal/last stage: **True**

## Retained deployment/training profiles

| Profile | Remaining layers | Weight-memory gain | Quality opportunity loss | Opportunity margin | Model |
|---|---:|---:|---:|---:|---|
| full_quality | 28 | 0.00% | 0.00% | 0.00% | `artifacts\phase_2b_quality_v4_full_20260802_213049\reference\full_model` |
| balanced | 19 | 23.75% | 11.24% | +12.52% | `artifacts\phase_2b_quality_v4_full_20260802_213049\frontier\stage_09_drop_14_24_6_23_8_10_18_9_7\model` |
| min_memory | 19 | 23.75% | 11.24% | +12.52% | `artifacts\phase_2b_quality_v4_full_20260802_213049\frontier\stage_09_drop_14_24_6_23_8_10_18_9_7\model` |

## Decision rule

A technically comparable candidate enters the frontier only when **both** are true:

1. Its cumulative loaded weight-tensor memory reduction from the full reference is strictly greater than its cumulative combined quality opportunity loss.
2. Its incremental loaded weight-tensor memory reduction from the immediate accepted parent is strictly greater than its incremental combined quality opportunity loss.

The second condition prevents a later damaging deletion from being subsidized by
the large positive margin accumulated by earlier successful deletions. Mean
quality, tail quality, category behavior, comprehensibility, task adequacy,
logic, safety, context, completeness, content anchors, issue cost,
severe-collapse rate, and target NLL contribute continuously according to the
selected profile. None is an independent absolute floor or issue-count veto.
Latency and evaluation VRAM remain secondary measured evidence.

Human review and later phase controls remain mandatory. Technical invalidity—such
as missing evidence, a corrupt checkpoint, mismatched evaluator versions, or
non-reproducible provenance—still blocks comparison because no valid opportunity
calculation exists.

## Workflow handoff

- `full_quality` preserves the original all-layer SFT reference.
- `balanced` selects the eligible frontier point with the largest cumulative opportunity margin.
- `min_memory` selects the eligible frontier point with the largest weight-memory gain.
- Phase 2C should begin only after a reduced profile receives explicit human approval.
- Phase 4 soups must use ingredients from the same selected architecture profile.
