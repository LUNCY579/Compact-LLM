# v3 E1 — Transport Channel Capacity (corrected)

Operator: P4 target-basis spectral profile transfer, rank 8, strength 0.1.

Supersedes `attempt_0001_unfair_nulls/`. See the protocol-deviation note at the end.

## What crosses the bridge

- Projected tensors: **196** of 310 target tensors.
- Donor-derived scalars transmitted: **1568** (196 tensors x 8 normalised singular values).
- That is 3.898e-07 of Qwen3-4B.
- Donor **directional** information transmitted: **none**. Every direction in the
  injected update is one of the target's own top-r singular directions or the
  target weight itself.

The closed form has a sharper reading than E1's first attempt used:

> `a_i = strength * ||W_target|| * (p_i - q_i)`

where `p` is the donor's normalised top-r spectrum and `q` is the target's own
(verified to 1.50e-14 relative). The two profiles are nearly parallel —
median cos(p, q) = 0.9974 — so the delta is
essentially their *radial gap*. That is the quantity any fair null must preserve.

## How much of the delta is donor-independent?

Replace each donor profile with the pooled mean for its module role. That 'donor'
carries 56 numbers in total and no tensor-specific information at all.

- Projecting each real delta onto it explains **84.9%** of the 4B delta energy
  and **89.6%** of the 8B delta energy.
- The orthogonal residual still agrees across donors at cosine **0.8472**, so a genuine
  tensor-specific component survives; it is simply a minority of the signal.
- Under plain subtraction rather than projection the residual cosine is 0.8507.

## Hold the operator fixed, vary the donor

| Arm | cos(delta_a, delta_b) |
|---|---:|
| real_4b_vs_real_8b | 0.9682 |
| norm_matched_random_seed17 | 0.9146 |
| norm_matched_random_seed23 | 0.9121 |
| norm_matched_random_seed42 | 0.9109 |
| layer_shuffled_within_role_seed17 | 0.7697 |
| layer_shuffled_within_role_seed23 | 0.7731 |
| layer_shuffled_within_role_seed42 | 0.7869 |
| sign_shuffled_seed17 | -0.0653 |
| sign_shuffled_seed23 | -0.0590 |
| sign_shuffled_seed42 | -0.0676 |
| role_mean_donor_vs_real_8b | 0.8715 |
| identical_donors_upper_bound | 1.0000 |

Real cross-donor cosine **0.9682**. Architecture-only null band **0.7697 – 0.9146**.

The strongest null lives in E1c: iid Gaussian matrices at the *true donor tensor
shapes*, through the identical operator, reach **0.9988** across three seed pairs —
above the real donors. See `shape_matched_donor_null.md`.

## Interpretation

**HT1 holds**: the channel is narrow and carries no directions.

**HT2 holds**: 84.9% of the delta energy is donor-independent by exact algebra —
reproducible from a 56-number role-mean donor. The amendment's wording is that *some*
of the agreement is a property of `T`, and that is amply satisfied.

**The cross-donor agreement statistic carries no evidence about donor content.** The v1
report read cosine 0.932 as showing the donor signal is not random. But donors sharing
nothing except their tensor shapes reach 0.9988 through the same operator — *better* than
the real donors. The statistic should be retired as evidence, not merely down-scoped.

That is not the same as saying nothing donor-specific is present. After projecting out the
role-mean construction, a tensor-specific residual carrying 15% of the delta
energy still agrees across the two donors at cosine 0.8472. Real donor structure does
reach the target; it is a minority of an already narrow channel, and the headline statistic
was never measuring it.

This does not weaken the study's negative result — it strengthens it. It removes the
premise that would otherwise justify building a wider operator on the grounds that
'the donors demonstrably agree'.

## Protocol deviation in the first attempt

The amendment freezes norm-matched random, layer-shuffled, and sign-shuffled controls for E1-E3. The first attempt ran a norm-RANDOMISED synthetic donor, an all-key shuffle that also destroyed module role, and no sign-shuffled arm. It also published a 'null donor' arm whose delta is an exact algebraic zero, so its reported cosine of -0.2462 was the angle between two rounding-error vectors. This corrected run implements the frozen controls.
