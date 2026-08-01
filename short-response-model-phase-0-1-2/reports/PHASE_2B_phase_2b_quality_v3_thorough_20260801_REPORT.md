# PHASE 2B — PROGRESSIVE STRUCTURAL REDUCTION REPORT

## Purpose

Test which complete layers can be removed, optionally recovered, and combined
progressively. This structural experiment remains distinct from selective-layer
training.

## Evidence boundary

- Probe cases: 150, selected deterministically from development data.
- Sealed evaluation used: **No**.
- Quality evaluator: **3.0**.
- Relative opportunity profile: **balanced**.
- Full-reference quality: 0.978.
- Full-reference target NLL: 4.8624.
- Every current layer is tested as a raw deletion at each progressive stage.
- Recovery fine-tuning is limited to the strongest 6 raw candidates per stage unless configured otherwise.

## Stage 1 individual-layer results

| Dropped original layer | Variant | Quality | Weight-memory gain | Quality opportunity loss | Opportunity margin | Eval VRAM change | Generation latency change | Target NLL | Profile | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | raw deletion | 0.515 | 2.64% | 39.30% | -36.66% | +2.60% | -269.71% | 13.7890 | balanced | NOT WORTHWHILE |
| 1 | raw deletion | 0.006 | 2.64% | 65.22% | -62.58% | +2.60% | -89.64% | 11.5138 | balanced | NOT WORTHWHILE |
| 2 | raw deletion | 0.000 | 2.64% | 65.46% | -62.82% | +2.60% | -32.82% | 10.2266 | balanced | NOT WORTHWHILE |
| 3 | raw deletion | 0.975 | 2.64% | 0.34% | +2.30% | +2.60% | +16.74% | 5.2377 | balanced | WORTHWHILE |
| 4 | raw deletion | 0.976 | 2.64% | 0.46% | +2.18% | +2.60% | +5.97% | 5.0194 | balanced | WORTHWHILE |
| 5 | raw deletion | 0.976 | 2.64% | 0.44% | +2.20% | +2.60% | +7.45% | 5.0005 | balanced | WORTHWHILE |
| 6 | raw deletion | 0.969 | 2.64% | 2.18% | +0.46% | +2.60% | +11.72% | 5.0072 | balanced | WORTHWHILE |
| 7 | raw deletion | 0.977 | 2.64% | 0.07% | +2.57% | +2.60% | +5.79% | 4.7547 | balanced | WORTHWHILE |
| 7 | deletion + recovery SFT | 0.976 | 2.64% | 0.13% | +2.51% | +0.00% | -77.52% | 3.1986 | balanced | WORTHWHILE |
| 8 | raw deletion | 0.977 | 2.64% | 0.12% | +2.52% | +2.60% | -1.44% | 4.8559 | balanced | WORTHWHILE |
| 8 | deletion + recovery SFT | 0.976 | 2.64% | 0.18% | +2.45% | +0.00% | -83.55% | 3.1168 | balanced | WORTHWHILE |
| 9 | raw deletion | 0.967 | 2.64% | 2.18% | +0.46% | +2.60% | +19.71% | 4.8117 | balanced | WORTHWHILE |
| 10 | raw deletion | 0.975 | 2.64% | 0.47% | +2.17% | +2.60% | +15.40% | 4.7447 | balanced | WORTHWHILE |
| 11 | raw deletion | 0.973 | 2.64% | 1.52% | +1.12% | +2.60% | -5.55% | 4.9065 | balanced | WORTHWHILE |
| 12 | raw deletion | 0.974 | 2.64% | 0.81% | +1.83% | +2.60% | +8.59% | 4.4333 | balanced | WORTHWHILE |
| 13 | raw deletion | 0.976 | 2.64% | 0.21% | +2.43% | +2.60% | +10.48% | 4.9410 | balanced | WORTHWHILE |
| 14 | raw deletion | 0.974 | 2.64% | 0.50% | +2.14% | +2.60% | +2.90% | 4.5488 | balanced | WORTHWHILE |
| 15 | raw deletion | 0.973 | 2.64% | 0.87% | +1.77% | +2.60% | +18.05% | 4.5838 | balanced | WORTHWHILE |
| 16 | raw deletion | 0.978 | 2.64% | 0.03% | +2.61% | +2.60% | -0.75% | 4.7063 | balanced | WORTHWHILE |
| 16 | deletion + recovery SFT | 0.977 | 2.64% | 0.36% | +2.28% | +0.00% | -91.04% | 3.1024 | balanced | WORTHWHILE |
| 17 | raw deletion | 0.970 | 2.64% | 1.91% | +0.73% | +2.60% | +16.78% | 4.8067 | balanced | WORTHWHILE |
| 18 | raw deletion | 0.972 | 2.64% | 1.74% | +0.90% | +2.60% | +4.97% | 4.6834 | balanced | WORTHWHILE |
| 19 | raw deletion | 0.976 | 2.64% | 0.51% | +2.13% | +2.60% | +9.21% | 4.7932 | balanced | WORTHWHILE |
| 20 | raw deletion | 0.978 | 2.64% | 0.15% | +2.49% | +2.60% | -8.36% | 4.9135 | balanced | WORTHWHILE |
| 21 | raw deletion | 0.972 | 2.64% | 0.99% | +1.65% | +2.60% | +6.35% | 5.0163 | balanced | WORTHWHILE |
| 22 | raw deletion | 0.971 | 2.64% | 1.94% | +0.70% | +2.60% | +2.36% | 4.8566 | balanced | WORTHWHILE |
| 23 | raw deletion | 0.977 | 2.64% | 0.12% | +2.52% | +2.60% | +6.92% | 4.7903 | balanced | WORTHWHILE |
| 23 | deletion + recovery SFT | 0.977 | 2.64% | 0.12% | +2.52% | +0.00% | -86.78% | 3.1429 | balanced | WORTHWHILE |
| 24 | raw deletion | 0.977 | 2.64% | 0.08% | +2.56% | +2.60% | +11.83% | 4.7771 | balanced | WORTHWHILE |
| 24 | deletion + recovery SFT | 0.976 | 2.64% | 0.19% | +2.45% | +0.00% | -101.64% | 3.1721 | balanced | WORTHWHILE |
| 25 | raw deletion | 0.942 | 2.64% | 8.59% | -5.95% | +2.60% | +6.69% | 4.9712 | balanced | NOT WORTHWHILE |
| 26 | raw deletion | 0.978 | 2.64% | 0.05% | +2.58% | +2.60% | +4.26% | 4.7626 | balanced | WORTHWHILE |
| 26 | deletion + recovery SFT | 0.977 | 2.64% | 0.10% | +2.54% | +0.00% | -89.17% | 3.2942 | balanced | WORTHWHILE |
| 27 | raw deletion | 0.923 | 2.64% | 8.42% | -5.79% | +2.60% | -238.82% | 4.2455 | balanced | NOT WORTHWHILE |

## Accepted progressive frontier

| Stage | Original layers dropped so far | Remaining layers | Quality | Weight-memory gain | Quality opportunity loss | Opportunity margin | Eval VRAM change | Generation latency change | Selected form |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | [16] | 27 | 0.978 | 2.64% | 0.03% | +2.61% | +2.60% | -0.75% | raw deletion |
| 2 | [16, 26] | 26 | 0.978 | 5.28% | 0.08% | +5.20% | +5.21% | -10.58% | raw deletion |
| 3 | [16, 26, 7] | 25 | 0.979 | 7.92% | 0.07% | +7.85% | +7.81% | -9.59% | raw deletion |
| 4 | [16, 26, 7, 21] | 24 | 0.974 | 10.56% | 0.69% | +9.87% | +10.41% | -17.58% | raw deletion |
| 5 | [16, 26, 7, 21, 17] | 23 | 0.973 | 13.20% | 0.75% | +12.45% | +13.02% | +0.43% | raw deletion |
| 6 | [16, 26, 7, 21, 17, 13] | 22 | 0.975 | 15.84% | 0.71% | +15.12% | +15.62% | -3.05% | raw deletion |
| 7 | [16, 26, 7, 21, 17, 13, 3] | 21 | 0.974 | 18.47% | 1.10% | +17.38% | +15.78% | -13.74% | deletion + recovery SFT |
| 8 | [16, 26, 7, 21, 17, 13, 3, 15] | 20 | 0.979 | 21.11% | 0.58% | +20.54% | +20.83% | +39.73% | raw deletion |
| 9 | [16, 26, 7, 21, 17, 13, 3, 15, 6] | 19 | 0.978 | 23.75% | 0.54% | +23.21% | +23.43% | +37.46% | raw deletion |
| 10 | [16, 26, 7, 21, 17, 13, 3, 15, 6, 11] | 18 | 0.973 | 26.39% | 1.01% | +25.38% | +23.94% | -33.15% | deletion + recovery SFT |

## Retained deployment/training profiles

| Profile | Remaining layers | Weight-memory gain | Quality opportunity loss | Opportunity margin | Model |
|---|---:|---:|---:|---:|---|
| full_quality | 28 | 0.00% | 0.00% | 0.00% | `artifacts\phase_2b_quality_v3_thorough_20260801\reference\full_model` |
| balanced | 18 | 26.39% | 1.01% | +25.38% | `artifacts\phase_2b_quality_v3_thorough_20260801\frontier\stage_10_drop_16_26_7_21_17_13_3_15_6_11\model` |
| min_memory | 18 | 26.39% | 1.01% | +25.38% | `artifacts\phase_2b_quality_v3_thorough_20260801\frontier\stage_10_drop_16_26_7_21_17_13_3_15_6_11\model` |

## Decision rule

A technically comparable candidate enters the frontier when its measured
proportional loaded weight-tensor memory reduction is **strictly greater** than
its combined proportional quality opportunity loss. Mean quality, tail quality,
category behavior, comprehensibility, task adequacy, logic, safety, context,
completeness, content anchors, issue cost, severe-collapse rate, and target NLL
contribute continuously according to the selected profile. None is an independent
absolute floor or issue-count veto. Latency and evaluation VRAM remain secondary
measured evidence and do not silently replace the declared primary resource.

Human review and later phase controls remain mandatory. Technical invalidity—such
as missing evidence, a corrupt checkpoint, mismatched evaluator versions, or
non-reproducible provenance—still blocks comparison because no valid opportunity
calculation exists.

## Workflow handoff

- `full_quality` preserves the original all-layer SFT reference.
- `balanced` selects the eligible frontier point with the largest opportunity margin.
- `min_memory` selects the eligible frontier point with the largest weight-memory gain.
- Phase 4 soups must use ingredients from the same selected architecture profile.
