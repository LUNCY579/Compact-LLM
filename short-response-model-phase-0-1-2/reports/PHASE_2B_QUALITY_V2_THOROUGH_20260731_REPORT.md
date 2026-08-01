# PHASE 2B — PROGRESSIVE STRUCTURAL REDUCTION REPORT

## Purpose

Extend the negative Phase 2 selective-tuning result with a different question:
**which complete layers can be removed, optionally recovered, and combined progressively?**
This does not reinterpret the failed one-layer tuning gate as a success.

## Evidence boundary

- Probe cases: 150, selected deterministically from development data.
- Sealed evaluation used: **No**.
- Full-reference heuristic score: 0.933.
- Full-reference target NLL: 4.8624.
- Every current layer is tested as a raw deletion at each progressive stage.
- Recovery fine-tuning is limited to the strongest 6 raw candidates per stage unless configured otherwise.

The legacy three-check score saturated at 1.0, so this study also records target-token NLL and side-by-side generated-output deviation.

## Stage 1 individual-layer results

| Dropped original layer | Variant | Quality | Quality loss | Weight-memory reduction | Reduction advantage | Eval VRAM reduction | Generation latency reduction | Target NLL | Rule |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | raw deletion | 0.144 | 84.58% | 2.64% | -81.94% | +2.60% | -222.24% | 13.7890 | FAIL |
| 1 | raw deletion | 0.005 | 99.44% | 2.64% | -96.80% | +2.60% | -61.34% | 11.5138 | FAIL |
| 2 | raw deletion | 0.105 | 88.72% | 2.64% | -86.08% | +2.60% | -15.13% | 10.2266 | FAIL |
| 3 | raw deletion | 0.913 | 2.15% | 2.64% | +0.48% | +2.60% | +31.62% | 5.2377 | FAIL |
| 4 | raw deletion | 0.913 | 2.05% | 2.64% | +0.59% | +2.60% | +22.48% | 5.0194 | FAIL |
| 5 | raw deletion | 0.906 | 2.88% | 2.64% | -0.24% | +2.60% | +23.25% | 5.0005 | FAIL |
| 6 | raw deletion | 0.921 | 1.27% | 2.64% | +1.37% | +2.60% | +26.20% | 5.0072 | FAIL |
| 7 | raw deletion | 0.924 | 0.91% | 2.64% | +1.73% | +2.60% | +12.93% | 4.7547 | FAIL |
| 7 | deletion + recovery SFT | 0.911 | 2.28% | 2.64% | +0.36% | +0.00% | -51.78% | 3.1986 | FAIL |
| 8 | raw deletion | 0.899 | 3.66% | 2.64% | -1.02% | +2.60% | +16.73% | 4.8559 | FAIL |
| 9 | raw deletion | 0.918 | 1.53% | 2.64% | +1.11% | +2.60% | +33.52% | 4.8117 | FAIL |
| 10 | raw deletion | 0.911 | 2.28% | 2.64% | +0.36% | +2.60% | +29.91% | 4.7447 | FAIL |
| 11 | raw deletion | 0.887 | 4.87% | 2.64% | -2.23% | +2.60% | +12.31% | 4.9065 | FAIL |
| 12 | raw deletion | 0.912 | 2.19% | 2.64% | +0.45% | +2.60% | +26.04% | 4.4333 | FAIL |
| 13 | raw deletion | 0.929 | 0.42% | 2.64% | +2.22% | +2.60% | +26.38% | 4.9410 | FAIL |
| 13 | deletion + recovery SFT | 0.886 | 4.98% | 2.64% | -2.34% | +0.00% | -65.04% | 3.1213 | FAIL |
| 14 | raw deletion | 0.928 | 0.47% | 2.64% | +2.17% | +2.60% | +20.20% | 4.5488 | FAIL |
| 14 | deletion + recovery SFT | 0.909 | 2.56% | 2.64% | +0.08% | +0.00% | -53.76% | 3.0788 | FAIL |
| 15 | raw deletion | 0.903 | 3.20% | 2.64% | -0.56% | +2.60% | +33.10% | 4.5838 | FAIL |
| 16 | raw deletion | 0.925 | 0.87% | 2.64% | +1.77% | +2.60% | +16.43% | 4.7063 | FAIL |
| 16 | deletion + recovery SFT | 0.899 | 3.66% | 2.64% | -1.02% | +0.00% | -61.66% | 3.1024 | FAIL |
| 17 | raw deletion | 0.934 | 0.00% | 2.64% | +2.64% | +2.60% | +31.76% | 4.8067 | FAIL |
| 17 | deletion + recovery SFT | 0.892 | 4.39% | 2.64% | -1.75% | +0.00% | -40.91% | 3.1917 | FAIL |
| 18 | raw deletion | 0.898 | 3.70% | 2.64% | -1.06% | +2.60% | +22.03% | 4.6834 | FAIL |
| 19 | raw deletion | 0.911 | 2.37% | 2.64% | +0.27% | +2.60% | +25.29% | 4.7932 | FAIL |
| 20 | raw deletion | 0.890 | 4.57% | 2.64% | -1.93% | +2.60% | +9.88% | 4.9135 | FAIL |
| 21 | raw deletion | 0.901 | 3.36% | 2.64% | -0.72% | +2.60% | +23.72% | 5.0163 | FAIL |
| 22 | raw deletion | 0.900 | 3.48% | 2.64% | -0.84% | +2.60% | +19.61% | 4.8566 | FAIL |
| 23 | raw deletion | 0.929 | 0.38% | 2.64% | +2.26% | +2.60% | +23.08% | 4.7903 | FAIL |
| 23 | deletion + recovery SFT | 0.901 | 3.42% | 2.64% | -0.78% | +0.00% | -59.19% | 3.1429 | FAIL |
| 24 | raw deletion | 0.924 | 0.95% | 2.64% | +1.69% | +2.60% | +28.38% | 4.7771 | FAIL |
| 25 | raw deletion | 0.903 | 3.19% | 2.64% | -0.55% | +2.60% | +22.89% | 4.9712 | FAIL |
| 26 | raw deletion | 0.915 | 1.89% | 2.64% | +0.75% | +2.60% | +21.83% | 4.7626 | FAIL |
| 27 | raw deletion | 0.586 | 37.12% | 2.64% | -34.48% | +2.60% | -186.53% | 4.2455 | FAIL |

## Accepted progressive frontier

| Stage | Original layers dropped so far | Remaining layers | Quality | Quality loss | Weight-memory reduction | Reduction advantage | Eval VRAM reduction | Generation latency reduction | Selected form |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | [] | - | - | - | - | - | - | - | no accepted reduction |

## Retained deployment/training profiles

| Profile | Remaining layers | Weight-memory reduction | Quality loss | Model |
|---|---:|---:|---:|---|
| full_quality | 28 | 0.00% | 0.00% | `artifacts\phase_2b_quality_v2_thorough_20260731\reference\full_model` |
| balanced | 28 | 0.00% | 0.00% | `artifacts\phase_2b_quality_v2_thorough_20260731\reference\full_model` |
| min_memory | 28 | 0.00% | 0.00% | `artifacts\phase_2b_quality_v2_thorough_20260731\reference\full_model` |

## Decision rule

A step can enter the frontier only when:

1. measured loaded weight-tensor reduction is greater than relative quality-score loss;
2. measured evaluation VRAM and generation latency are recorded as secondary evidence (they may be noisy and are not substituted for the stable weight-tensor rule);
3. no critical semantic failures appear;
4. semantic, category, completion, and target-NLL ceilings pass; and
5. all evidence and the parent checkpoint remain reproducible.

A positive ratio is **not** enough for release by itself. The configured development probe is for trend discovery; human review and later phase controls remain mandatory.

## Workflow handoff

- `full_quality` continues Phase 3 from the original all-layer SFT adapter.
- `balanced` and `min_memory` use materialized reduced checkpoints and attach a fresh compatible DPO LoRA adapter in Phase 3.
- Phase 4 soups must use ingredients from the same selected profile and architecture.
- Phase 4B remains a post-alignment revalidation/refinement stage, not a reason to discard this Phase 2 evidence.
