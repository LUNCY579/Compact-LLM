# v3 E1 — Transport Channel Capacity

Operator: P4 target-basis spectral profile transfer, rank 8, strength 0.1.

## What crosses the bridge

- Projected tensors: **196** of 310 target tensors.
- Donor-derived scalars transmitted: **1568** (196 tensors x 8 normalised singular values).
- That is 3.898e-07 of Qwen3-4B and 1.914e-07 of Qwen3-8B.
- Donor **directional** information transmitted: **none**. Every direction in the
  injected update is one of the target's own top-r singular directions or the
  target weight itself.

## Closed form validated against the operator's own record

Max relative error 4.37e-03 over 196 tensors.

Delta energy: 95.85% in the rank-8 block, 4.15% parallel to the target weight.

## Hold the operator fixed, vary the donor

| Arm | cos(delta_a, delta_b) |
|---|---:|
| real_4b_vs_real_8b | 0.9682 |
| tensor_shuffled_donors_seed17 | 0.5955 |
| tensor_shuffled_donors_seed23 | 0.5467 |
| tensor_shuffled_donors_seed42 | 0.5900 |
| two_random_donors_seed17 | 0.4917 |
| two_random_donors_seed23 | 0.3843 |
| two_random_donors_seed42 | 0.3231 |
| real_4b_vs_null_donor | -0.2462 |
| real_8b_vs_flat_donor | 0.3837 |
| identical_donors_upper_bound | 1.0000 |

Real cross-donor cosine **0.9682** versus strongest donor-varying control **0.5955** (lift +0.3727).

## Interpretation

HT1 holds: the channel is narrow and carries no directions.
HT2 does not hold: the cross-donor agreement is real and tensor-specific,
since destroying tensor correspondence costs most of it.

The two findings together say the v1 headline statistic was a true statement
about a very small quantity. Qwen3-4B and Qwen3-8B genuinely agree about
1568 numbers, and the operator was never able to ask them anything else.
