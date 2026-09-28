# HOTFIX-003 — G1_QUALITY: the gate the ladder was missing

Date: September 11, 2026 (America/Chicago). This amendment **adds** a gate. It loosens
nothing, changes no existing threshold, and opens no closed stage. G3/G4/G5 remain closed.

## The defect this fixes

Every stage in the ladder is satisfied by a model that emits token salad.

| stage | what it actually checks |
|---|---|
| `G1_WIDTH` | registered dims, tensor accounting, finite logits, save/reload, param count |
| `G1_CED_*` | the above plus strict cache-order parity |
| `G2_CACHED_*` | 25 steps completed, params reached, teacher frozen, save/reload |

All of those are **plumbing or self-consistency** properties. None compares an arm's output
to the frozen source. Measured consequence, on the already-open calibration and pilot
splits:

| arm | mean CE | perplexity | KL→source | top-1 | pilot hits | behaviour |
|---|---:|---:|---:|---:|---:|---|
| B28 frozen source | 5.546 | 256 | 0 | 1.000 | 6/24 | works |
| D40 ordinary width | 5.360 | 213 | 1.468 | 0.500 | 0/24 | repeats `</think>` |
| C28 CED | 12.585 | 292,267 | 9.154 | 0.014 | 0/24 | token salad |
| C40 CED | 12.960 | 425,015 | 9.878 | 0.004 | 0/24 | token salad |

`ln(151936) = 11.9312`, so both CED arms are worse than uniform. **`G1_WIDTH` PASSed D40**,
whose free generation is a degenerate loop, and parity is no protection either: D40 passes
the full `compare_paths` grid with **zero** violations while being degenerate.

Note particularly that D40's CE is **better** than the source. A CE-only gate would have
passed it. Catching it requires divergence-from-teacher and nondegeneracy checks too, which
is why this stage has five quality checks rather than one.

## What is added

`G1_QUALITY_<arm>` for B28, D40, C28 and C40, with parents `G0`, `G1_WIDTH`,
`G1_CED_C28` and `G1_CED_C40` respectively. Forward-only; it trains nothing and cannot
license training. Checks:

    ce_increase_within_margin     arm CE − source CE  <= 0.5 nats
    kl_to_source_within_margin    mean KL(source‖arm) <= 1.0 nats
    top1_agreement_above_floor    argmax agreement    >= 0.5
    generation_terminates         termination rate    >= 0.8
    generation_not_degenerate     distinct outputs    >= 2
    source_unchanged, calibration_unchanged, pilot_unchanged

CE and KL are scored on the **calibration** split, generation on the **pilot** split, both
already open. `ensure_disjoint` is applied. **No sealed or DEV data is read**, and
`forbidden` records `sealed_test_access`, `dev_access` and
`threshold_tuning_on_student_outcomes`.

## Threshold provenance — stated plainly

These thresholds were set **after** the arms above were measured, which is the wrong order
and is disclosed rather than hidden. They are chosen on principle, not fitted: CE and KL
bound divergence from the frozen teacher at roughly half a nat and one nat; the top-1 floor
sits between chance and majority agreement; the termination floor **reuses the existing
`G0.min_termination_rate` of 0.8** rather than inventing a value.

The reason this is tolerable: the measured arms are nowhere near the boundary. C28 has
KL 9.154 and top-1 0.014 against limits of 1.0 and 0.5 — an order of magnitude out on both.
D40 fails on KL (1.468 vs 1.0) and termination (0.75 vs 0.80). **No plausible setting of
these thresholds changes any arm's verdict**, so the conclusions do not rest on the choice.
A future arm landing near a boundary must not be resolved by moving one.

## Ordering, and what this cannot do

C28 and C40 already hold **FAIL** parity receipts, so `validate_receipt` refuses their
quality stage: a failed construction cannot be quality-qualified. The gate is therefore
demonstrated on B28 (expected PASS) and D40 (expected FAIL) — and D40 is the whole point,
since it is the arm every existing gate accepted.

This stage is a floor, not a capability claim. Passing it means an arm is not obviously
broken. It is scored on 156 calibration rows and 24 pilot prompts of one narrow arithmetic
family, single deterministic run, no seeds. It is **not** evidence of quality, recovery or
efficacy, and it does not substitute for a registered DEV or sealed evaluation.
