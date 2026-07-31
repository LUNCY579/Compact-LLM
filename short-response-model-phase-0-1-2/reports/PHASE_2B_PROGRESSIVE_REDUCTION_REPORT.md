# PHASE 2B — PROGRESSIVE STRUCTURAL REDUCTION REPORT

## Purpose

Extend the negative Phase 2 selective-tuning result with a different question:
**which complete layers can be removed, optionally recovered, and combined progressively?**
This does not reinterpret the failed one-layer tuning gate as a success.

## Evidence boundary

- Probe cases: 50, selected deterministically from development data.
- Sealed evaluation used: **No**.
- Full-reference heuristic score: 1.000.
- Full-reference target NLL: 4.8916.
- Every current layer is tested as a raw deletion at each progressive stage.
- Recovery fine-tuning is limited to the strongest 3 raw candidates per stage unless configured otherwise.

The automated score previously saturated at 1.0, so this study also records target-token NLL and side-by-side generated-output deviation.

## Stage 1 individual-layer results

| Dropped original layer | Variant | Score | Score loss | Weight-memory reduction | Reduction advantage | Eval VRAM reduction | Generation latency reduction | Target NLL | Rule |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | raw deletion | 0.707 | 29.33% | 2.64% | -26.69% | +2.60% | -235.98% | 14.0962 | FAIL |
| 1 | raw deletion | 0.353 | 64.67% | 2.64% | -62.03% | +2.60% | -110.72% | 11.5845 | FAIL |
| 2 | raw deletion | 0.260 | 74.00% | 2.64% | -71.36% | +2.60% | -46.94% | 10.0382 | FAIL |
| 3 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +16.87% | 5.2018 | PASS |
| 4 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +2.65% | 5.2024 | PASS |
| 5 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | +10.01% | 5.0286 | PASS |
| 6 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -10.52% | 5.1360 | PASS |
| 7 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -21.15% | 4.8305 | PASS |
| 8 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -31.34% | 4.8804 | PASS |
| 9 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -1.25% | 4.7941 | PASS |
| 10 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -6.49% | 4.9657 | PASS |
| 11 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -30.24% | 4.8516 | PASS |
| 12 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -5.45% | 4.5643 | PASS |
| 13 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -14.52% | 4.9923 | PASS |
| 14 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -26.48% | 4.5775 | PASS |
| 15 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -0.52% | 4.6789 | PASS |
| 16 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -43.41% | 4.7749 | PASS |
| 16 | deletion + recovery SFT | 0.993 | 0.67% | 2.64% | +1.97% | +0.00% | -167.83% | 3.1972 | FAIL |
| 17 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -14.45% | 4.9971 | PASS |
| 18 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -27.83% | 4.6876 | PASS |
| 19 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -15.29% | 4.9164 | PASS |
| 20 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -58.27% | 4.9566 | PASS |
| 21 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -27.13% | 5.0797 | PASS |
| 22 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -71.95% | 4.7705 | PASS |
| 23 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -41.24% | 4.8442 | PASS |
| 23 | deletion + recovery SFT | 0.993 | 0.67% | 2.64% | +1.97% | +0.00% | -137.54% | 3.2573 | FAIL |
| 24 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -18.24% | 4.8579 | PASS |
| 25 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -21.38% | 5.0635 | PASS |
| 26 | raw deletion | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -54.03% | 4.6310 | PASS |
| 26 | deletion + recovery SFT | 1.000 | 0.00% | 2.64% | +2.64% | +0.00% | -137.22% | 3.2781 | PASS |
| 27 | raw deletion | 0.993 | 0.67% | 2.64% | +1.97% | +2.60% | -117.70% | 4.1699 | FAIL |

## Accepted progressive frontier

| Stage | Original layers dropped so far | Remaining layers | Score | Score loss | Weight-memory reduction | Reduction advantage | Eval VRAM reduction | Generation latency reduction | Selected form |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | [26] | 27 | 1.000 | 0.00% | 2.64% | +2.64% | +2.60% | -54.03% | raw deletion |
| 2 | [26, 13] | 26 | 1.000 | 0.00% | 5.28% | +5.28% | +5.21% | -71.07% | raw deletion |
| 3 | [26, 13, 14] | 25 | 1.000 | 0.00% | 7.92% | +7.92% | +7.81% | +2.61% | raw deletion |
| 4 | [26, 13, 14, 10] | 24 | 1.000 | 0.00% | 10.56% | +10.56% | +7.62% | -44.19% | deletion + recovery SFT |
| 5 | [26, 13, 14, 10, 6] | 23 | 1.000 | 0.00% | 13.20% | +13.20% | +13.02% | +17.86% | raw deletion |
| 6 | [26, 13, 14, 10, 6, 19] | 22 | 1.000 | 0.00% | 15.84% | +15.84% | +15.62% | +8.87% | raw deletion |

## Retained deployment/training profiles

| Profile | Remaining layers | Weight-memory reduction | Score loss | Model |
|---|---:|---:|---:|---|
| full_quality | 28 | 0.00% | 0.00% | `artifacts\phase_2_progressive_reduction\reference\full_model` |
| balanced | 22 | 15.84% | 0.00% | `artifacts\phase_2_progressive_reduction\frontier\stage_06_drop_26_13_14_10_6_19\model` |
| min_memory | 22 | 15.84% | 0.00% | `artifacts\phase_2_progressive_reduction\frontier\stage_06_drop_26_13_14_10_6_19\model` |

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
