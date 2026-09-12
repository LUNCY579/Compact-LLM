# HOTFIX-004 — calibrate the G1_QUALITY nondegeneracy floor, and correct HOTFIX-003

Date: September 11, 2026 (America/Chicago). This amendment **tightens** one check and
**corrects a false claim** in HOTFIX-003. It loosens nothing and opens no closed stage.

## Why

`G1_QUALITY` was first executed on real weights at implementation `7d48de21`. It did its job
— B28 PASS 8/8, D40 FAIL, where D40 is the arm that passes `G1_WIDTH` and the full parity
grid with zero violations while generating a degenerate `</think>` loop. But the run exposed
two defects in the gate itself.

| check | D40 value | limit | outcome |
|---|---:|---:|---|
| `ce_increase_within_margin` | −0.1308 | ≤ 0.5 | PASS — a CE-only gate misses D40 entirely |
| `kl_to_source_within_margin` | 1.4909 | ≤ 1.0 | **FAIL** |
| `top1_agreement_above_floor` | **0.5032** | ≥ 0.5000 | PASS by 0.0032 |
| `generation_terminates` | 0.750 | ≥ 0.8 | **FAIL** |
| `generation_not_degenerate` | 4 / 24 | ≥ 2 | PASS — **misses a textbook degenerate model** |

## Change 1 — the nondegeneracy floor now scales

An absolute floor of 2 distinct outputs is meaningless at 24 prompts. D40 emits a repeated
`</think>` loop, produces 4 distinct strings, and passed the check whose entire purpose is to
catch that. The rule becomes

    distinct >= max(min_distinct_outputs, ceil(min_distinct_output_ratio * generation_rows))

with `min_distinct_outputs` 2 and `min_distinct_output_ratio` 0.25. At 24 prompts the floor
is 6, so D40's 4 now fails. The receipt records `distinct_outputs_required` alongside
`distinct_outputs` so the applied floor is visible rather than inferred.

This is a **tightening**, justified on principle — a gate that cannot reject the canonical
degenerate case is not doing its job — and not by the desire to change D40's verdict, which
was already FAIL on two other checks.

## Change 2 — HOTFIX-003's threshold claim was overstated

HOTFIX-003 said: *"No plausible setting of these thresholds changes any arm's verdict."*

**That is false as written.** D40's top-1 agreement is 0.5032 against a 0.5000 floor: a floor
of 0.51 fails it, 0.50 passes it. The claim holds only for the **overall verdict** — D40 fails
regardless, because KL (1.4909 vs 1.0) and termination (0.750 vs 0.80) both fail
independently — not for the per-check outcome. The provenance text in `protocol.json` is
corrected accordingly.

**The top-1 floor was deliberately NOT moved.** Having seen 0.5032, raising the floor would be
tuning a threshold on a measured outcome, which the protocol forbids and which this programme
has criticised elsewhere. The knife-edge is recorded instead, and a future arm landing near it
must not be resolved by moving it.

## What is explicitly not claimed

Passing `G1_QUALITY` still means only that an arm is not obviously broken. It is scored on
156 calibration rows and 24 pilot prompts of one narrow arithmetic family, single
deterministic run, no seeds. It is not evidence of quality, recovery or efficacy, and it does
not substitute for a registered DEV or sealed evaluation.

Note also that B28's pass is largely tautological: for that arm the implementation compares
the source to itself, so CE-increase, KL and top-1 are exactly 0, 0 and 1 by construction.
Only termination and nondegeneracy are informative for the source arm, and all of the gate's
discriminating power is on converted arms.
