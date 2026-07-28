# PHASE 1 STOP GATE

## Purpose

Compare the untouched base model with the all-layer dense-SFT adapter using the
same benchmark and decoding configuration. This automated gate is evidence for
human review; it does not approve Phase 2 by itself.

## Inputs

- Base summary: `artifacts\baseline_evaluation\summary.json`
- Adapter summary: `artifacts\dense_sft_evaluation\summary.json`

## Comparison

| Metric | Base | Dense SFT | Difference |
|---|---:|---:|---:|
| Mean heuristic score | 0.804 | 0.995 | +0.191 |
| Median reply tokens | 16.0 | 9.0 | -7.0 |
| P90 reply tokens | 22.0 | 11.0 | -11.0 |
| Complete utterance rate | 1.000 | 1.000 | +0.000 |
| Peak evaluation VRAM (GB) | 1.135 | 1.173 | +0.038 |

## Automated Checks

| Check | Result |
|---|---|
| score_improvement | PASS |
| median_reply_tokens | PASS |
| p90_reply_tokens | PASS |
| complete_utterance_rate | PASS |
| category_regression_limit | PASS |

## Category Regressions Above Allowed Limit

| Category | Score drop |
|---|---:|
| None | 0.000 |

## Gate Status

`PASS_PENDING_HUMAN_REVIEW`

## Required Human Review

Before starting the layer scan, inspect the lowest-scoring adapter cases and
confirm that the model is:

- naturally brief rather than mechanically truncated;
- varied rather than relying on canned replies;
- appropriate across the represented personalities and categories;
- not materially worse on any important category;
- clearly improved enough to justify the next experiment.

Record the final human decision below:

- Human decision: `PASS`
- Reviewer: Cheng Li
- Date: 2026-07-27
- Notes: Good Dense SFT generated. Permission to move on to Phase 2. 
