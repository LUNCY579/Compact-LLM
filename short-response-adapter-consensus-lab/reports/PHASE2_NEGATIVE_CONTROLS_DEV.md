# Phase 2 bounded DEV negative controls

Status: **BOUNDED_DEV_NEGATIVE_CONTROLS_AUDITED**

This is an audit of the pre-existing 8-example-per-condition DEV smoke packet. It is negative-control evidence, not a powered accuracy experiment.

## Condition summaries

| Committee | Method | Correct / 8 |
|---|---|---:|
| general_k1 | hard_plurality | 0 / 8 |
| general_k1 | probability_mean | 0 / 8 |
| general_k1 | logit_mean | 0 / 8 |
| duplicate_general_k3 | hard_plurality | 0 / 8 |
| duplicate_general_k3 | probability_mean | 0 / 8 |
| duplicate_general_k3 | logit_mean | 0 / 8 |
| core_k2 | hard_plurality | 0 / 8 |
| core_k2 | probability_mean | 1 / 8 |
| core_k2 | logit_mean | 1 / 8 |
| weak_injection_k3 | hard_plurality | 0 / 8 |
| weak_injection_k3 | probability_mean | 1 / 8 |
| weak_injection_k3 | logit_mean | 0 / 8 |
| overconfident_k3 | hard_plurality | 1 / 8 |
| overconfident_k3 | probability_mean | 2 / 8 |
| overconfident_k3 | logit_mean | 0 / 8 |
| fixed_random_proxy_k3 | hard_plurality | 4 / 8 |
| fixed_random_proxy_k3 | probability_mean | 3 / 8 |
| fixed_random_proxy_k3 | logit_mean | 4 / 8 |

## Paired sequence controls

| Left | Right | Method | Exact sequence matches |
|---|---|---|---:|
| general_k1 | duplicate_general_k3 | hard_plurality | 8 / 8 |
| general_k1 | duplicate_general_k3 | probability_mean | 8 / 8 |
| general_k1 | duplicate_general_k3 | logit_mean | 8 / 8 |
| core_k2 | weak_injection_k3 | hard_plurality | 4 / 8 |
| core_k2 | weak_injection_k3 | probability_mean | 6 / 8 |
| core_k2 | weak_injection_k3 | logit_mean | 2 / 8 |
| weak_injection_k3 | overconfident_k3 | hard_plurality | 0 / 8 |
| weak_injection_k3 | overconfident_k3 | probability_mean | 1 / 8 |
| weak_injection_k3 | overconfident_k3 | logit_mean | 0 / 8 |
| core_k2 | overconfident_k3 | hard_plurality | 0 / 8 |
| core_k2 | overconfident_k3 | probability_mean | 1 / 8 |
| core_k2 | overconfident_k3 | logit_mean | 0 / 8 |

Duplicate General copies are exactly invariant on all 24 method-example comparisons. Weak-adapter injection changes some trajectories. Increasing the same Social adapter's logit scale from 1 to 5 changes all hard sequences in this packet because confidence-based tie resolution is scale-sensitive at tied vote counts; this does not contradict top-1 proposal invariance away from ties.

The fixed-pool proxy is not a matched random-seed or competence-matched control. With only eight examples, no accuracy difference here supports a causal specialization claim.

Raw SHA-256: `638754c533475a55e21c4f0e97dbf7f28cffdf66c19b4eedb849d5a2e9bfa38e`.
