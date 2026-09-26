# Independent review: resolved findings; prototype UNADOPTED

Reviewer `/root/resume_proof/review_resume` performed a read-only review and
in-memory CPU reproductions; no reviewer files were written.

The initial review found three concrete gaps:

1. A modified applied-token total or finite-gradient total could be accepted.
2. A modified optimizer LR or scheduler epoch could be accepted.
3. Invalid RNG state was rejected after checkpoint weights had been copied.

The prototype now reconciles every dose counter against the trajectory, checks
each attempt's token count against its exact ordered examples, validates AdamW
configuration/state and the fixed applied-update LR schedule, checks scaler
progression, and validates RNG using private generators before checkpoint
weights are copied. Regression cases exercise each reported corruption and
assert that the fresh adapter's values remain unchanged on rejection. The
reviewer confirmed these three findings were addressed.

The follow-up review found a float32 numerical edge in scaler validation:
Python-double scale replay could reject a valid checkpoint after scale
underflow or growth saturation. Replay now uses float32 arithmetic and the
finite-growth guard. Tests use the actual CPU GradScaler to prove valid scale
underflow to zero after 170 overflows and growth saturation at 2**127. Both
pass; the original scaler resume and malformed-scaler regressions also pass.

The reviewer agreed that exact training-state recovery requires both callbacks
and loggers to be RNG-neutral, nonmutating observers. External event delivery
must be reconciled/deduplicated from committed records. These interface limits
are explicit in README.md.

Validation sequence: the complete suite passed **55 tests in 38.93 seconds**.
After the narrow scaler arithmetic correction, the targeted scaler regression
selection passed **4 tests in 4.66 seconds**, including the two added numerical
edge cases (38 other resume cases deselected). The full suite was not repeated
after that narrow correction, following the root's bounded-work instruction.
The first growth-saturation fixture run used an oversized Python integer for
`init_scale`, which PyTorch rejected before training; the test was corrected to
pass the intended floating-point value. No production behavior changed for that
fixture correction.

No remaining blocker was identified for the short CPU fixture. This is not
approval for canonical-model/GPU integration. The artifact remains UNADOPTED.
