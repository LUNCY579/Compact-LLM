# Withdrawn — E1 attempt 0001 used unfair donor-varying nulls

Preserved because the amendment forbids silently discarding a result.

Its Section 2 concluded that the 4B/8B agreement is "real and tensor-specific"
because the real cosine 0.9682 far exceeded its controls at 0.32-0.60. Adversarial
review found the controls invalid:

1. The synthetic donor multiplied the profile norm by `uniform(2, 12)`, randomising
   the donor's top-r energy fraction over a six-fold range. The delta is exactly
   `a_i = strength * ||W|| * (p_i - q_i)` with `p` and `q` near-parallel, so that
   radial magnitude *is* the signal. The null destroyed the quantity under test.
2. The layer shuffle permuted across all keys, which also scrambled module role.
   Role is what sets the profile norm, so the arm was over-destroyed.
3. No sign-shuffled arm was run, although the amendment freezes one.
4. The `real_4b_vs_null_donor = -0.2462` row is an exact algebraic zero: feeding
   the target's own profile makes `desired == target_s`, so both the rank block and
   the perpendicular coordinate vanish. The published number is the cosine between
   two rounding-error vectors, rescued by the denominator guard.

Items 1-3 are protocol deviations from the frozen measurement rules, not judgement
calls. See `../channel_capacity.md` for the corrected run and
`../shape_matched_donor_null.md` for the strongest null.
