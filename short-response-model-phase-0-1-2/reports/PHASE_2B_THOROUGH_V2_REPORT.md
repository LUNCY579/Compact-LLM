# PHASE 2B — PROGRESSIVE STRUCTURAL REDUCTION REPORT

## Purpose

Extend the negative Phase 2 selective-tuning result with a different question:
**which complete layers can be removed, optionally recovered, and combined progressively?**
This does not reinterpret the failed one-layer tuning gate as a success.

## Evidence boundary

- Probe cases: 150, selected deterministically from development data.
- Sealed evaluation used: **No**.
- Full-reference heuristic score: 1.000.
- Full-reference target NLL: 4.8624.
- Every current layer is tested as a raw deletion at each progressive stage.
- Recovery fine-tuning is limited to the strongest 6 raw candidates per stage unless configured otherwise.

The automated score previously saturated at 1.0, so this study also records target-token NLL and side-by-side generated-output deviation.

## Stage 1 individual-layer results

| Dropped original layer | Variant | Score | Score loss | Weight-memory reduction | Reduction advantage | Eval VRAM reduction | Generation latency reduction | Target NLL | Rule |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | raw deletion | 0.716 | 28.44% | 2.64% | -25.81% | +2.60% | -247.69% | 13.7890 | FAIL |
| 1 | raw deletion | 0.342 | 65.78% | 2.64% | -63.14% | +2.60% | -100.57% | 11.5138 | FAIL |
| 2 | raw deletion | 0.256 | 74.44% | 2.64% | -71.81% | +2.60% | -39.47% | 10.2266 | FAIL |
| 3 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +12.08% | 5.2377 | PASS |
| 4 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +0.55% | 5.0194 | PASS |
| 5 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +2.55% | 5.0005 | PASS |
| 6 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +5.96% | 5.0072 | PASS |
| 7 | raw deletion | 0.996 | 0.44% | 2.64% | +2.19% | +2.60% | +2.02% | 4.7547 | FAIL |
| 8 | raw deletion | 0.998 | 0.22% | 2.64% | +2.42% | +2.60% | -13.96% | 4.8559 | PASS |
| 9 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +15.30% | 4.8117 | PASS |
| 10 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +6.81% | 4.7447 | PASS |
| 11 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -10.29% | 4.9065 | PASS |
| 12 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +6.21% | 4.4333 | PASS |
| 12 | deletion + recovery SFT | 0.996 | 0.44% | 2.64% | +2.19% | +0.00% | -89.33% | 3.0992 | FAIL |
| 13 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +6.47% | 4.9410 | PASS |
| 14 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -1.07% | 4.5488 | PASS |
| 14 | deletion + recovery SFT | 0.996 | 0.44% | 2.64% | +2.19% | +0.00% | -84.68% | 3.1716 | FAIL |
| 15 | raw deletion | 0.998 | 0.22% | 2.64% | +2.42% | +2.60% | +14.41% | 4.5838 | PASS |
| 16 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -6.17% | 4.7063 | PASS |
| 16 | deletion + recovery SFT | 0.996 | 0.44% | 2.64% | +2.19% | +0.00% | -92.51% | 3.1534 | PASS |
| 17 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +10.87% | 4.8067 | PASS |
| 18 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -1.16% | 4.6834 | PASS |
| 19 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +2.08% | 4.7932 | PASS |
| 20 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -15.59% | 4.9135 | PASS |
| 21 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +2.35% | 5.0163 | PASS |
| 22 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -5.25% | 4.8566 | PASS |
| 23 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +0.94% | 4.7903 | PASS |
| 23 | deletion + recovery SFT | 1.000 | 0.00% | 2.64% | +2.64% | +0.00% | -101.47% | 3.1370 | PASS |
| 24 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +7.37% | 4.7771 | PASS |
| 24 | deletion + recovery SFT | 1.000 | 0.00% | 2.64% | +2.64% | +0.00% | -100.18% | 3.2291 | PASS |
| 25 | raw deletion | 0.998 | 0.22% | 2.64% | +2.42% | +2.60% | +0.21% | 4.9712 | PASS |
| 26 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -1.41% | 4.7626 | PASS |
| 26 | deletion + recovery SFT | 1.000 | 0.00% | 2.64% | +2.64% | +0.00% | -99.35% | 3.2850 | PASS |
| 27 | raw deletion | 0.991 | 0.89% | 2.64% | +1.75% | +2.60% | -45.57% | 4.2455 | FAIL |

## Accepted progressive frontier

| Stage | Original layers dropped so far | Remaining layers | Score | Score loss | Weight-memory reduction | Reduction advantage | Eval VRAM reduction | Generation latency reduction | Selected form |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | [24] | 27 | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +7.37% | raw deletion |
| 2 | [24, 23] | 26 | 1.000 | 0.00% | 5.28% | +5.28% | +5.21% | +1.62% | raw deletion |
| 3 | [24, 23, 14] | 25 | 1.000 | 0.00% | 7.92% | +7.92% | +7.81% | -2.75% | raw deletion |
| 4 | [24, 23, 14, 19] | 24 | 1.000 | 0.00% | 10.56% | +10.56% | +10.41% | -31.33% | raw deletion |
| 5 | [24, 23, 14, 19, 3] | 23 | 1.000 | 0.00% | 13.20% | +13.20% | +10.34% | -70.17% | deletion + recovery SFT |
| 6 | [24, 23, 14, 19, 3, 13] | 22 | 1.000 | 0.00% | 15.84% | +15.84% | +13.06% | -56.46% | deletion + recovery SFT |
| 7 | [24, 23, 14, 19, 3, 13, 10] | 21 | 1.000 | 0.00% | 18.47% | +18.47% | +15.78% | -55.56% | deletion + recovery SFT |
| 8 | [24, 23, 14, 19, 3, 13, 10, 21] | 20 | 1.000 | 0.00% | 21.11% | +21.11% | +18.50% | -46.67% | deletion + recovery SFT |
| 9 | [24, 23, 14, 19, 3, 13, 10, 21, 22] | 19 | 1.000 | 0.00% | 23.75% | +23.75% | +21.22% | -54.63% | deletion + recovery SFT |
| 10 | [24, 23, 14, 19, 3, 13, 10, 21, 22, 5] | 18 | 1.000 | 0.00% | 26.39% | +26.39% | +23.94% | -45.98% | deletion + recovery SFT |

## Retained deployment/training profiles

| Profile | Remaining layers | Weight-memory reduction | Score loss | Model |
|---|---:|---:|---:|---|
| full_quality | 28 | 0.00% | 0.00% | `artifacts\phase_2_progressive_reduction_thorough_v2\reference\full_model` |
| balanced | 18 | 26.39% | 0.00% | `artifacts\phase_2_progressive_reduction_thorough_v2\frontier\stage_10_drop_24_23_14_19_3_13_10_21_22_5\model` |
| min_memory | 18 | 26.39% | 0.00% | `artifacts\phase_2_progressive_reduction_thorough_v2\frontier\stage_10_drop_24_23_14_19_3_13_10_21_22_5\model` |

## Decision rule

A step can enter the frontier only when:

1. measured loaded weight-tensor reduction is greater than relative heuristic-score loss;
2. measured evaluation VRAM and generation latency are recorded as secondary evidence (they may be noisy and are not substituted for the stable weight-tensor rule);
3. no critical semantic failures appear;
4. semantic, category, completion, and target-NLL ceilings pass; and
5. all evidence and the parent checkpoint remain reproducible.

A positive ratio is **not** enough for release by itself. The 50-case probe is for trend discovery; human review and later phase controls remain mandatory.

## Workflow handoff

- `full_quality` continues Phase 3 from the original all-layer SFT adapter.
- `balanced` and `min_memory` use materialized reduced checkpoints and attach a fresh compatible DPO LoRA adapter in Phase 3.
- Phase 4 soups must use ingredients from the same selected profile and architecture.
- Phase 4B remains a post-alignment revalidation/refinement stage, not a reason to discard this Phase 2 evidence.
