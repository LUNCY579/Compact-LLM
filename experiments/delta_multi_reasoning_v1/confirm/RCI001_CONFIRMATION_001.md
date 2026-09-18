# RCI-001 confirmation read #1 — CONFIRMED: the delta damages abstention

**The programme's first confirmed result.** Registered in `RCI001_CONFIRM_PREREG.md` and
committed (`0fa37e4`) before the seal was opened. The read budget is now drawn down: 1 of 1
used, sealed bank spent. Two systems read, no third.

---

## Primary endpoint — correct literal `UNKNOWN` on the 128 unanswerable rows

| system | correct abstention | rate |
|---|---:|---:|
| **B** frozen Supa 1.7.0 `962abc5b…` | **40/128** | 0.312 |
| **D** promoted k=0.25 core `a28629c9…` | **25/128** | 0.195 |
| difference | **−15** | |

**Exact two-sided McNemar on the unanswerable rows: D-gains 0, D-losses 15, p = 6.10 × 10⁻⁵.**

Registered criterion was *p ≤ 0.05 **and** D < B*. Both hold. **CONFIRMED.**

**The asymmetry is total and was not required by the criterion: zero gains, fifteen losses.**
There is no item in the 128 where the promoted core abstains correctly and the frozen source
does not. The delta did not trade abstentions — it only lost them.

Registered prediction was B in 30–50 and D in 10–28. **B 40, D 25 — both inside the band**,
predicted from a disjoint calibration twin before the seal was opened.

## What this confirms, stated narrowly

> **The k=0.25 delta damages literal-`UNKNOWN` abstention relative to the frozen source.**

Confirmed on a fresh instrument that was never used to select anything, whose floors were
published before any model call, and which neither system had ever seen. That claim was
previously report-only on a `SPENT` suite and is the programme's most robust negative — every
arm damages abstention, invariant to base model, scale, parameterisation and precision.

**What it does not confirm.** Not the magnitude of the Demo suite's 13→12; different
instrument, different absolute rates. Not any long-context claim. Not any statement about
r=16, N16, the α line, the k line, the 4B arms or any composition arm — **none were read, and
reading one would have been selection.** It does not license promotion, release, or
`quality_improvement_established`. One training draw of one delta.

---

## Secondary endpoints — reported, and barred from the verdict by the prereg

| | B | D |
|---|---:|---:|
| overall | 265/512 = 0.518 | 235/512 = 0.459 |
| answerable | 225/384 | 210/384 |
| **false abstention on answerable** | **0** | **0** |
| per-pack all-four-correct | 22/128 | 12/128 |

Zero false abstention on either side, so neither system reaches its abstention score by
emitting the sentinel indiscriminately — which is the control that makes the primary endpoint
mean what it says.

Per-pack all-four-correct is the recency-immune metric (see the erratum below): its shortcut
floor is **0/128**, and B 22 / D 12 discriminates in the same direction as the primary.

---

## ERRATUM 1 — the published aggregate floor was too low, and the defect is mine

**Raised by the peer session, verified here independently, and published on my own record
because the seal is my artifact.**

The published binding floor for aggregate accuracy was **0.2773** (`per_task_type_majority`).
A predictor in the same admissibility class beats it:

**`last_candidate_in_context`** — answer whichever member of the task type's label vocabulary
occurs latest in the context. One uniform rule, no labels at inference:

| | sealed |
|---|---:|
| `last_candidate_in_context` | **190/512 = 0.3711** |
| published floor | 0.2773 |

Per task type, answerable rows — reproduced here exactly:

| task type | recency rule |
|---|---:|
| `temporal_correction` | **78/78 = 1.000** |
| `rule_exception` | **75/75 = 1.000** |
| `object_state_tracking` | 26/78 = 0.333 |
| `alias_resolution` | 11/78 = 0.141 |
| `count_with_exclusions` | 0/75 = 0.000 |

**The cause is a construction defect in my generator.** `render()` emits the correction *after*
the date it corrects and the exception *after* the rule it qualifies, because that is the
natural narrative order. Recency is therefore perfectly confounded with correctness on 153 of
384 answerable rows, and "trust the last thing said" scores 100% there without reading.

**Corrected published floor for aggregate accuracy: 0.3711.** Both systems still clear it on
the sealed bank (0.518 and 0.459), but the margin is much smaller than the build claimed and
any future reading of an aggregate score from this bank must use 0.3711.

### What the defect does and does not reach

| quantity | recency floor | status |
|---|---:|---|
| **unanswerable stratum (the primary endpoint)** | **0/128** | **immune** |
| **per-pack all-four-correct** | **0/128** | **immune** |
| aggregate accuracy | 190/512 | **defective, floor corrected** |

Any extractor answers *something* on an unanswerable row, so it can never emit the sentinel and
scores zero on the abstention stratum — and zero on any pack containing one. **The primary
endpoint of this read is untouched by the defect**, which is why the confirmation stands.

That is luck resting on design rather than luck alone: the per-pack anti-luck structure was
copied from the Demo suite before there was a known reason to need it, and the read was
recommended on abstention for unrelated reasons. Both turned out load-bearing.

### Timing, stated plainly

The peer's audit arrived with the words *"reads_used is still 0, so this is free right now and
impossible to fix afterwards."* **It was not still 0.** I had drawn down the budget and run
both systems minutes earlier; their audit and my read overlapped. So the floor correction
landed after the seal was spent, and could not be applied before it as they intended.

The read survives because the defect does not reach its primary endpoint. Had I registered
*aggregate accuracy* as primary, this read would have been compromised and unrepeatable. **The
margin between a sound result and a wasted instrument was the choice of endpoint, not the
process** — and the process should not have left that to a good instinct.

### The fix for the next build

`build_rci.py` must randomise clause order so that corrections and exceptions are as often
stated before as after the material they modify, and `last_candidate_in_context` must be added
to the published predictor set so the floor is computed against it at build time rather than
audited afterwards. Neither is done here; RCI-001 is spent and the changes belong to RCI-002.

### One quantity deliberately not adopted as a floor

The peer computed a per-task-type best-of-strategies envelope reaching 0.5996 — above both
calibration systems — and declined to publish it as a floor because it is an envelope chosen
with labels, not a deployable predictor. **That restraint is correct and is exactly the
distinction whose absence falsified C1.** It is recorded here as a design flag: on this bank,
neither calibration system beats a per-task-type-tuned heuristic, which is worth knowing when
reading any aggregate score from it and is a further reason the aggregate is the weak endpoint.

---

## Status

`reads_used` 1 of 1. **RCI-001 is spent.** Content hash `3f9434f7…` verified intact; the seal
record now reads `ALREADY READ`.

The generator is not spent. `build_rci.py` can mint RCI-002 from a fresh disjoint entity
partition at zero model and zero instrument cost — with the clause-order fix and the recency
predictor in the published floor set. The scarcity is partition space, and extending it is
writing new entity pools.

`quality_improvement_established` stays false. `release_permitted` stays false. Nothing
promoted. One claim confirmed, and it is a caution.
