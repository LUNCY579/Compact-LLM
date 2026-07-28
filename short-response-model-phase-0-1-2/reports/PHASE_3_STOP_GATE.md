# PHASE 3 STOP GATE

## Purpose

Compare the accepted dense SFT reference with the DPO candidate. The automated checks are evidence for human review; they do not approve Phase 4 by themselves.

## Inputs

- SFT summary: `artifacts\dense_sft_evaluation\summary.json`
- DPO summary: `artifacts\phase_3_dpo_evaluation\summary.json`

## Comparison

| Metric | SFT | DPO | Difference |
|---|---:|---:|---:|
| Mean heuristic score | 0.995 | 1.000 | +0.005 |
| Median reply tokens | 9.0 | 9.0 | +0.0 |
| P90 reply tokens | 11.0 | 11.0 | +0.0 |
| Complete utterance rate | 1.000 | 1.000 | +0.000 |
| Critical semantic failures | 0 | 0 | +0 |
| Major semantic issues | 0 | 3 | +3 |

## Automated Checks

| Check | Result |
|---|---|
| score_not_regressed_more_than_0.02 | PASS |
| no_critical_failures | PASS |
| critical_failures_not_increased | PASS |
| major_issues_not_increased | FAIL |
| complete_utterance_rate | PASS |
| p90_reply_tokens | PASS |

## Automated Gate Status

`FAIL`

## Required Human Review

Before Phase 4, inspect blinded SFT-versus-DPO outputs and confirm:

- concise-correct replies improve over verbose or inaccurate replies;
- role-playing voice, persona, mood, and relationship context remain consistent;
- responses remain natural and varied rather than becoming canned;
- clarification behavior is appropriate when information is missing;
- factual, context, and safety behavior do not materially regress;
- low-scoring and semantically flagged cases are acceptable.

DPO must improve preference behavior rather than merely shorten replies.

- Human decision: `PENDING`
- Reviewer:
- Date:
- Notes:
