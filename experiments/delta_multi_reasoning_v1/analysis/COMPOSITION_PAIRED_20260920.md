# Composition, decomposed — the three contrasts nobody had paired

**2026-09-20, zero model calls.** All on `select-views-001`, bank `views_sha256 7dd53f2a…`,
digest `a329eb7bb06f`. Computed from the stored per-view `canonical` flags.

COMPOSE-001 and MAG-001 report composition as aggregate differences (−60, +44, −16). The
programme's standing rule is **never quote a pooled net**. Decomposed:

| contrast | scores | gains | losses | net | exact McNemar |
|---|---:|---:|---:|---:|---:|
| T16 → **TH16 raw sum** | 235 → 175 | 9 | **69** | **−60** | 1.38e-12 |
| **TH16 → TH16s** (rescale only) | 175 → 219 | **63** | 19 | **+44** | 1.15e-06 |
| T16 → **TH16s** (composition at matched norm) | 235 → 219 | 11 | 27 | **−16** | **0.01385** |

## Three things the nets were hiding

**1. The residual cost of matched-norm composition is REAL, not noise.** −16 at p = 0.0139 on
38 discordant pairs. This is a **within-draw** contrast — both arms are deterministic transforms
of the same trained deltas — so it is valid at its own discordance and the ~20-row cross-draw
floor does not apply. **Composition at matched magnitude is cheap, not free.**

**2. No contrast here is purely subtractive.** The raw sum still *gains* 9 rows T16 lacked;
matched-norm composition gains 11. Rescaling loses 19 rows that the raw sum had. **Composition
reorders items, it does not merely remove them** — the same shape as `nesting fails 21/21 pairs`
in `per-item-headroom-unaddressable`, where scaling a delta reorders rather than nests.

**3. The rescale's +44 is 63 gains against 19 losses.** Reporting it as "+44" implies a one-way
repair. It is a two-way trade that nets strongly positive.

## What this does NOT establish, and the identity caveat

**The T16 arm used here is `rank001-20260918/arm-r16`, not COMPOSE-001's own `arm-T16`.**
No artifact carries per-view flags for `compose001/arm-T16`, so it cannot be paired directly.

The stand-in is the same construction — P₁₆(δ_T) — and scores **235, matching exactly**. But the
receipts differ in the seventh decimal of the retained-norm fraction:

    compose001 T16      delta_norm_fraction_retained  0.812249925436939
    rank001/arm-r16     delta_norm_fraction_retained  0.8122500916811055

**~1.7e-7 apart — a rebuild, not a byte-identical arm**, the same trap as `arm-rebuild-s10`
against `arm-s10`. The aggregate agreement at 235 is strong circumstantial evidence they behave
identically, and it is **not** a receipt. Treat the −16 as measured against *a* P₁₆(δ_T) at 235,
not against COMPOSE-001's specific bytes.

## Every composition arm's weights are RECLAIMED

| arm | on disk |
|---|---|
| `mag001/arm-TH16s`, `compose001/arm-{TH16,T16,H16}`, `tt001/build-*` | **16 MB sidecars, 0 safetensors** |

So **completing TH16s's blocked battery** (long context, literal `UNKNOWN`,
`references_retained`, gate v2 — all `exit=2` on the host-RAM qualify gate) requires
**rebuilding the arm first**, not merely running probes. That is a materially larger cost than
"the RAM has freed, just run it."

Rebuild inputs that do survive: `compose001-20260918/H` (3 MB — metadata, not weights),
`rrp015-20260917/control-hf` (16 MB sidecar), and `trained_MIX_fp32.safetensors` (1,201 MB).
**Whether δ_T and δ_H themselves are reconstructible on this host is not established here** and
must be checked before anyone costs the battery.

---

# The −16 is ATTENUATION, not interference — and H's skill looks free

I was about to decompose the 60-row composition cost as *"44 magnitude + 16 interference"*.
**That would have been wrong**, and the error is the one
`matched-norm-attenuates-components` exists to prevent.

## The construction detail that changes the reading

    TH16s = s_t * ( P16(delta_T,t) + P16(delta_H,t) )      per 2D tensor
    s_t   = ||P16(delta_T,t)|| / ||P16(delta_T,t) + P16(delta_H,t)||

The rescale is applied to the **sum**. So the sum carries T16's magnitude — verified,
`ratio_arm_over_T16 = 1.0000000004`, 220 tensors, **0 fallbacks** — but **T's own component
inside TH16s is at `s_t` of its truncated strength, not full strength.**

    s_t over 220 tensors:  min 0.5175   median 0.6750   max 0.7496

**TH16s is not "full T plus attenuated H". It is 0.675 × (T + H).** Comparing it to T16 at full
strength therefore compares two different T magnitudes, and attributes the difference to H.

## The control the comparison needs, interpolated

T's effective scalar α inside TH16s = `s_t × 0.8123` (the rank-16 norm fraction) = **0.5482** at
the median. Against INT-001's native α ladder (145 / 219 / 234 / 235 at α = 0.25 / 0.50 / 0.75 / 1.00):

| | effective α | predicted acquisition |
|---|---:|---:|
| T alone at median `s_t` | 0.5482 | **~221.9** |
| **TH16s observed** | — | **219** |
| | | **residual −2.9 rows** |

> **TH16s sits within 3 rows of where T alone would sit at the same attenuation — inside the
> 7-row floor. The −16 against T16 is T's own shrinkage, not H's presence.**

**So the entire 60-row composition cost is magnitude, and at matched total magnitude H's skill
costs essentially nothing in acquisition.** That is a much stronger result than "composition is
cheap": it is *"composition is free, and what you pay is the shrinkage you would pay anyway"*.

It also explains why `J_decision` is identical (0.7500) at TH16 and TH16s while acquisition moves
44 rows — **H's skill is not competing with T for acquisition at all.**

## Three caveats, because this is an interpolation and not a built control

1. **The comparator is a measured ray, not an arm.** INT-001's α is a scalar on the **full**
   delta; TH16s's T-component is a *scaled rank-16 truncation*. RANK-001's N16 licenses rough
   comparability (norm 0.812 scoring 236 against α=0.75's 234), and this programme has a
   precedent for ray interpolation landing within 1% — but it is still a prediction, not a
   measurement.
2. **`s_t` is NOT uniform** — 0.5175 to 0.7496 across tensors. The prediction at the extremes
   spans **195 to 226**, so −2.9 is a residual against the *median* and the band is wide.
3. **MAG-001's own registered A1 (`acquisition >= 220` → "interference was magnitude, not skill
   conflict") landed at 219 — failed by ONE row.** On its own that is a knife-edge and settles
   nothing; the attenuation control above is the better evidence, and it points the way A1 was
   reaching for.

## The missing control, and it needs no training

> **Build T alone, scaled per tensor by the SAME `s_t` vector.** Pure weight arithmetic over
> deltas already on disk. If it scores ~219, H is free and composition's cost is entirely
> magnitude. If it scores ~235, the attenuation account is wrong and the −16 really is
> interference.

That is the single cheapest decisive experiment available on the composable-skills question, and
**it is a same-skill control of exactly the kind `matched-norm-attenuates-components` says is
required and was never built.**

---

# SYNTHESIS — both attenuation controls are in, and composition explains nothing that a scalar does not

SCALED-T-001 (mine) and SCALED-H-001 (the paired session's) are the same experiment applied to
the two components of `TH16s`. Together they close the account.

## The two controls

    TH16s = s_t * ( P16(delta_T) + P16(delta_H) )      s_t median 0.675
    SCALED_T = s_t * P16(delta_T)      <- T's component, H absent
    SCALED_H = s_t * P16(delta_H)      <- H's component, T absent

| component at `s_t` ≈ 0.675 | endpoint | result |
|---|---|---|
| **δ_T** (generator) | acquisition /384 | **−12**, p = 0.0428 — attenuation **costs** |
| **δ_H** (gate) | boundary sensitivity /112 | 88 → 86, **−2** — no measurable change |
| **δ_H** (gate) | answerable specificity /112 | **93 → 111, +18** — attenuation **helps** |

> **The same 0.675 attenuation costs the generator and helps the gate.** Each skill has its own
> magnitude operating point; a single shared rescale lands near H's and below T's.

## What each of us had to withdraw

| claim | withdrawn by | replaced with |
|---|---|---|
| *"the −16 is skill interference"* (mine) | SCALED-T | T's own attenuation; H costs ≤10 rows, powered null |
| *"δ_T supplies the constraint that makes δ_H usable"* (theirs) | SCALED-H | **δ_H was over-driven; scaling it down alone recovers 18 of the 19 rows** |
| *"composition produced a J neither component has"* (theirs) | SCALED-H | **`SCALED_H` reaches J 0.7589 — the highest on record — containing no δ_T at all** |

**Each of us refuted our own headline by building the control the other's result implied.**
Neither could have got there alone: the confound on T was invisible from H's instrument and vice
versa.

## The consequence for the composable-skills question

`SCALED_H → TH16s` is **4 and 1 discordant pairs** on the two halves. At that discordance the
smallest attainable p values are **0.125 and 1.0** — the instrument cannot certify a difference
in either direction. **This is a BOUND, not a null:** δ_T adds at most ~3 rows per half, and
whether it adds anything is unresolvable. *(Contrast SCALED-T's nd = 16, where a 10-row effect
would have been detected — that one is a powered null and this one is not.)*

> **So at matched magnitude, composition buys nothing measurable that a per-skill scalar does not
> already buy — and on the gate instrument a single scaled delta BEATS the composition.**

The "composable skills gap" on this pair reduces to a **magnitude-calibration** problem, not an
interference problem. Two skills coexist; what the shared rescale costs is that one scalar serves
both, and the two skills want different scalars.

**And `0.675` was never chosen for H — it is whatever `s_t` happened to be.** It is already the
best arm on that instrument and **the curve between 0 and 1 is entirely unmeasured.** There is no
reason to believe it is the optimum.

## What I verified, and what I did not

**Verified from the artifacts** (`scaledh001-20260920/`, instrument `ba1f2982`, weight
`21cd17a8`): J_decision 0.7589, sensitivity 86/112, specificity 111/112, answerable-credited
**41** — the highest of any arm including the frozen source's 40. The J ranking
`SCALED_H 0.7589 > TH16s 0.7500 > H16 0.6161`. The aggregate specificity move 93 → 111 = +18,
consistent with their paired 18/0. And the min-attainable p values at nd = 4 and nd = 1.

**NOT verified: the paired splits.** `per_row` carries only `row_type`, `family`, `strict` and
`credited` — no abstention flag — and the instrument judges abstention through a
`rule_module_sha256` I do not have. **My proxy (prediction begins "UNKNOWN") returned 0
abstentions for arms recorded at 88 and 86** — impossible on its face, which is the canary firing
again. The 18/0 and 3/5 splits are recorded as theirs, single-implementation.

*(Fifth wrong extractor this week, and again caught only by collision with a number already held
rather than by reading the code.)*
