# Phase 2 Tie and EOS Counterfactuals

Role: **DEV GOLD-PREFIX COUNTERFACTUAL / EXPLORATORY**. The current confidence tie-break exactly reproduces the frozen hard K=7 local anchor.

Tied decisions: 1,097/18,760 token positions (5.85%).

## Tie policy

| Policy | Token acc. | All-token example acc. | Changed tokens | Rescue/harm vs current |
|---|---:|---:|---:|---:|
| confidence_sum | 78.33% | 25.26% | 0 | 0/0 |
| smallest_token_id | 78.42% | 25.01% | 404 | 63/46 |
| track_a_priority | 78.79% | 25.31% | 675 | 177/90 |

## EOS policy

| Policy | Token acc. | All-token example acc. | EOS overrides | Rescue/harm vs ordinary |
|---|---:|---:|---:|---:|
| ordinary | 78.33% | 25.26% | 0 | 0/0 |
| strict_majority | 75.86% | 15.58% | 472 | 5/467 |
| confidence_threshold_0.5 | 75.22% | 12.47% | 724 | 71/653 |
| minimum_tokens_3 | 70.10% | 10.12% | 1720 | 88/1631 |

These counterfactuals isolate local decision effects. They cannot establish sequence-level gains because an EOS override changes every later prefix under free running.
