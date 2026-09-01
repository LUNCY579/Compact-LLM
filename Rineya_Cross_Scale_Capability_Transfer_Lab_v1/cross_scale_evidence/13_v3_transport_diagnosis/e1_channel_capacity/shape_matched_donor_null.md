# v3 E1c — The Shape-Matched Random-Donor Null

E1's synthetic donor randomised the top-r energy fraction over a six-fold range per
tensor, which no real weight matrix does. This is the null that holds shape fixed and
randomises only content.

| Arm | cos(delta_a, delta_b) |
|---|---:|
| real_4b_vs_real_8b | 0.9682 |
| shape_matched_gaussian_donors_seed17 | 0.9988 |
| shape_matched_gaussian_donors_seed23 | 0.9988 |
| shape_matched_gaussian_donors_seed42 | 0.9988 |

Real Qwen3-4B vs Qwen3-8B: **0.9682**.
Shape-matched Gaussian donors: **0.9988** (mean of 3 seed pairs).
Lift of the real donors over the null: **-0.0306**.

## Verdict

The real cross-donor agreement does **not** exceed what two random matrices of the
same shapes produce through this operator. It is in fact slightly lower.

**HT2 holds and E1's Section-2 conclusion is retracted.** The v1 headline statistic
(cosine 0.932 between the projected 4B and 8B deltas, read as evidence that the donor
signal is not random) is a property of the P4 operator and of the donor tensor shapes.
It is not evidence of a shared cross-scale direction, because donors that share nothing
but their shapes reproduce it.

The mechanism is the one E1 already established. The delta's coordinates are
`a_i = strength * ||W_target|| * (p_donor_i - p_target_i)`, where `p` is a normalised
top-r spectrum. The normalised top-r spectrum of a large matrix is governed mostly by
its aspect ratio, so any two donors of these shapes produce nearly the same `p`, hence
nearly the same `a`, hence a high cosine.
