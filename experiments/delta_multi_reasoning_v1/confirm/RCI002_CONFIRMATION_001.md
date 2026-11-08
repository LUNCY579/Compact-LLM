# RCI-002 confirmation read #1 — CONFIRMED: the delta answers the interrogative, not the context

**The programme's second confirmed result, and the first that is a mechanism rather than a
caution.** Registered in `RCI002_CONFIRM_PREREG.md` and committed (`fdfdad1`) before the budget
was claimed. Budget claimed **before** rendering, per the fixed protocol. Seal
`baffccb9…`, `reads_used` 1 of 1 — RCI-002 is spent. Two systems read, no third.

---

## Primary endpoint — person-name emission where the context determines no person

120 answerable rows of the two incongruent carve-out strata. The question's interrogative is
**"Who …"**; the exempt party is identified only by role or by badge number, and **no person is
bound to it anywhere in the context** (asserted per pack at build time).

| system | emits a person | rate |
|---|---:|---:|
| **B** frozen Supa 1.7.0 `962abc5b…` | **85/120** | 0.708 |
| **D** promoted k=0.25 core `a28629c9…` | **99/120** | 0.825 |

**Exact two-sided McNemar: 14 rows where D emits a person and B does not, 0 the other way,
p = 1.22 × 10⁻⁴.** Registered criterion was *p ≤ 0.05 and D > B*. Both hold. **CONFIRMED.**

**Zero reversals**, as on RCI-001's equivalent cell and on the calibration twin. There is no row
in 120 where the frozen source names a person and the promoted core does not.

## The control is what makes this mean something

Same carve-out reasoning, same worlds, same answer types — only the interrogative changes:

| | n | B | D | discordant |
|---|---:|---:|---:|---:|
| **incongruent** ("Who …") | 120 | **85** | **99** | 14 / 0 |
| **congruent** ("Which role …", "Which badge number …") | 120 | **1** | **1** | 0 / 0 |

**One person emission in 120 congruent rows, from each system.** Neither model names a person
when the interrogative types the answer correctly, and both do on most rows when it does not.

- interaction on the discordant pairs, incongruent vs congruent: **p = 8.2 × 10⁻⁵**
- absolute rate, B alone: 85/120 against 1/120, **p = 1.0 × 10⁻³⁴**

So the effect is specific to the interrogative and not a general tendency to name people. That
was the whole point of crossing wh-congruence as a designed factor rather than letting it ride
on a paraphrase, which is how it hid in RCI-001 for a week.

Per cell, both incongruent strata move the same way; pooled they are decisive:

| stratum | n | B | D | p |
|---|---:|---:|---:|---:|
| `carve_out_role_incongruent` | 60 | 33 | **42** | 3.9e-03 |
| `carve_out_badge_incongruent` | 60 | 52 | **57** | 6.3e-02 |
| `carve_out_role_congruent` | 60 | 1 | 1 | 1.00 |
| `carve_out_badge_congruent` | 60 | 0 | 0 | 1.00 |

## What this confirms, stated narrowly

> **On questions whose interrogative types the answer as a person while the context determines
> none, the k=0.25 delta emits a person more often than the frozen source it came from.**

Confirmed on a fresh instrument, never used to select anything, whose floors — across three
predictor classes — were published before any model call, and which neither system had seen.

**What it does not confirm.** Not that the delta is worse at the *task* — accuracy on the same
rows is B 18 / D 13 on `carve_out_role_incongruent` (p = 0.125) and **0/60 for both** on
`carve_out_badge_incongruent`, where a substantively correct verbose answer scores zero. Not any
claim about abstention (see below). Not a mechanism for *why* the delta does this. Nothing about
r=16, N16, the α line, the k line, the 4B arms or any composition arm — **none were read, and
reading one would have been selection.** It does not license promotion, release, or
`quality_improvement_established`. One training draw of one delta.

## The band miss, reported first rather than buried

Registered prediction from the twin: B in **15–65**, D in **35–115**. Observed **B 85 (OUT)**,
D 99 (IN).

B emits persons at more than twice the twin's rate. The likeliest cause is that the metric's
absolute level depends on the person vocabulary's size, and the sealed partition has 12 names
against the twin's 8 — more strings for a substring detector to match. **That is a defect in the
absolute rate and not in the contrast**: both systems are scored on identical rows against an
identical vocabulary, so the paired difference is unaffected. It is the same absolute-versus-
paired distinction this instrument was built around, and it is a little uncomfortable to have it
land on my own registered band. Recorded as a miss.

## E1 — reported, never verdict-bearing, and the reason matters more than the number

| | n | B | D | discordant | p |
|---|---:|---:|---:|---:|---:|
| correct literal `UNKNOWN` | 160 | 23 | 20 | 0 / 3 | 0.25 |
| by density — low | 80 | 12 | 10 | | 0.50 |
| by density — high | 80 | 11 | 10 | | 1.00 |

Null, as calibration predicted, and **demoted before the read rather than after it**. The reason
is the most important thing this build found. Four constructions of the same unanswerable item,
frozen source, same twin:

| construction | B correct abstention |
|---|---:|
| block silently deleted | 11/72 = 0.153 |
| + an explicit note naming the missing fact | **35/64 = 0.547** |
| present-but-insufficient, loosely worded | 15/64 = 0.234 |
| present-but-insufficient, **lexically matched to its twin** | **9/64 = 0.141** |

**The frozen source abstains four times as often when a salient note names the missing fact.**
The second construction is roughly what RCI-001 did.

**This does not retract RCI-001's confirmed result** — B 40/128 vs D 25/128, 0 gains / 15 losses,
p = 6.1e-05 stands, and a cue both systems share cannot manufacture a paired difference. It
reframes it. *"The delta damages abstention"* is better stated as *"the delta is worse at
responding to an explicit unanswerability marker"*. Whether it damages abstention without one is
open, and **RCI-002 cannot answer it either**: with the marker matched away both systems sit near
14% and the endpoint is underpowered by construction.

Distractor density, registered as an E1 secondary, is flat — 12/10 low against 11/10 high.

## Secondaries, barred from the verdict

False abstention on 480 answerable rows: **B 3, D 0** — so neither system reaches its abstention
count by emitting the sentinel indiscriminately.

| stratum | n | B | D | p |
|---|---:|---:|---:|---:|
| `temporal_correction` | 60 | 50 | 47 | 0.45 |
| `carve_out_role_congruent` | 60 | 43 | 44 | 1.00 |
| `alias_resolution` | 60 | **41** | **36** | 6.3e-02 |
| `carve_out_badge_congruent` | 60 | 38 | 38 | 1.00 |
| `object_state_tracking` | 60 | 27 | 27 | 1.00 |
| `carve_out_role_incongruent` | 60 | 18 | 13 | 0.13 |
| `exclusion_count_number` | 60 | 7 | 4 | 0.25 |
| `carve_out_badge_incongruent` | 60 | 0 | 0 | 1.00 |

`alias_resolution` moves in the same direction as it did on RCI-001, where it lost 12 rows with
a congruent interrogative throughout and form preserved. **It is still the one cell nothing
explains**, and it is now the most interesting open question in the programme.

`gate_strata` flags **power compression** on `carve_out_badge_incongruent` (0/0 — the verbose
answer floor), `carve_out_role_incongruent` and `exclusion_count_number`.

Convention: B 155 strict → 224 article-tolerant, D 142 → 209 of 480. 69 and 67 rows turn on a
leading article. The confirmed endpoint is convention-free by construction — membership in a
name vocabulary, not string equality — which is why none of that reaches the verdict.

## One analysis error, found while writing this up

`subgroup_dissociation()` was registered as the interaction test and **is inapplicable here**.
It conditions on *baseline-correct*, because a loss requires the baseline to have been right —
the right rule for a correctness metric, and the one that killed two hypotheses last week. For
an **emission** metric the discordant direction under test requires the baseline to be *wrong*
(B not emitting), so the conditioning is inverted and the function returned `NO_DIFFERENCE`
with p = 1.00 on data where the interaction is p = 8.2e-05. The correct test is reported above.
The helper is right and was pointed at the wrong metric; registering it without re-deriving it
for a metric whose polarity is reversed was the mistake.

---

## Status

`reads_used` 1 of 1. **RCI-002 is spent.** Content hash `baffccb9…` verified intact; the seal
record reads `ALREADY READ`.

The generator is not spent. `build_rci2.py` can mint RCI-003 from a fresh disjoint entity
partition at zero model cost, and it now refuses to ship a bank whose answerability is
structurally detectable.

`quality_improvement_established` stays false. `release_permitted` stays false. Nothing promoted.
**Two confirmed claims now: the delta loses abstention markers, and the delta answers the
interrogative rather than the context. Both are cautions about the same published core.**
