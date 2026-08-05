# PHASE 3 STOP GATE

## Governing rule

DPO advances when its weighted proportional quality gains are greater than its weighted proportional regressions. No absolute score, pass-rate, issue-count, completion, or token-length floor controls promotion.

| Metric | SFT | DPO | Change |
|---|---:|---:|---:|
| Mean quality | 0.6820 | 0.6868 | +0.0048 |
| Descriptive quality index | 0.6433 | 0.6452 | +0.0019 |
| Worst-decile quality | 0.2685 | 0.2685 | +0.0000 |
| P90 tokens (diagnostic) | 11.0 | 13.0 | +2.0 |

## Relative losses

| Dimension | Proportional loss |
|---|---:|
| brevity_appropriateness | 0.0000% |
| category_tail | 0.0000% |
| comprehensibility | 0.0000% |
| context_consistency | 0.0000% |
| contract_fulfillment | 0.0000% |
| contract_violation_rate | 0.0000% |
| conversation_naturalness | 0.0000% |
| critical_defect_rate | 0.0000% |
| factual_correctness | 0.0000% |
| issue_cost | 0.0000% |
| logic_factual_consistency | 0.0000% |
| mean_quality | 0.0000% |
| mechanical_failure_rate | 0.0000% |
| mechanical_integrity | 0.0000% |
| observable_defect_rate | 0.0000% |
| prompt_function_alignment | 0.0000% |
| response_completeness | 0.0000% |
| safety_boundary | 0.0000% |
| task_adequacy | 0.0000% |
| worst_decile_quality | 0.0000% |

## Relative gains

| Dimension | Proportional gain |
|---|---:|
| brevity_appropriateness | 0.0000% |
| comprehensibility | 0.0000% |
| context_consistency | 0.0000% |
| contract_fulfillment | 0.0000% |
| conversation_naturalness | 0.0000% |
| factual_correctness | 0.0000% |
| logic_factual_consistency | 0.0000% |
| mean_quality | 0.7038% |
| mechanical_integrity | 0.0000% |
| prompt_function_alignment | 7.3126% |
| response_completeness | 0.0000% |
| safety_boundary | 0.0000% |
| task_adequacy | 0.0000% |
| worst_decile_quality | 0.0000% |

- Combined quality opportunity loss: `0.0000%`
- Combined quality gain: `0.6799%`
- Net relative quality margin: `+0.6799%`
- Quality profile: `balanced`

## Automated status

`PASS_PENDING_HUMAN_REVIEW`

## Human decision

Review blinded replies, preference behavior, logic, context, safety, and proportional issue-rate changes.

- Human decision: `PASS`
- Reviewer: Cheng Li
- Date: 2026-08-05
- Notes:
