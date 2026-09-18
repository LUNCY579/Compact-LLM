# RCI-002 — registered before the seal is opened

**Written and committed BEFORE `reads_used` is claimed, before the sealed bank is rendered, and
before any sealed row is scored. RCI-002 carries a read budget of ONE, like its predecessor.**

RCI-001 is spent. Its generator is not, and this is its successor with the defects fixed and
**two** endpoints instead of one. Build: `confirm/build_rci2.py` → `rci002-20260918/`.

---

## Why there are two endpoints

RCI-001 registered one endpoint and confirmed it. Decomposing that read afterwards turned up a
**second and larger effect nobody had separated**, and then five successive mechanism accounts
for it died — three of them because a single "damage" framing had merged two different things.
The whole point of this instrument is that the second effect gets registered in advance instead
of being discovered in a post-hoc split of five task types.

| | E1 — abstention | E2 — wh-incongruence |
|---|---|---|
| rows | all **160** unanswerable | **120** answerable rows of the two incongruent carve-out strata |
| metric | correct literal `UNKNOWN` | exact match under the declared normaliser |
| status | **replication** of RCI-001's confirmed result on a fresh bank | **new**, and the first pre-registered test of the wh-compliance account |
| convention risk | none — a literal sentinel admits no article or punctuation ambiguity | handled by a declared article-tolerant normaliser, published in `RCI_BUILD.json` |

Both are tested with **exact two-sided McNemar** on discordant pairs. **Holm-Bonferroni across
the family of two.** Confirmation criterion per endpoint: **Holm-adjusted p ≤ 0.05 and D < B.**

Reporting rule fixed now: repair and damage are reported separately and never netted, and any
subgroup claim is scored **over the rows B got right**, not over all rows
(`analysis/paired.py::subgroup_dissociation`). That denominator killed two of the five dead
hypotheses and it is not going to be re-derived after the fact a third time.

## The systems

| | artifact | frozen | chosen how |
|---|---|---|---|
| **B** | `Rineya-Supa-1.7.0-devfreeze`, `962abc5b…` | 2026-08-26 | the frozen source; not chosen at all |
| **D** | `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze`, `a28629c9…` | 2026-09-17 | promoted by SPARSE-001's rule, registered before any arm was built |

Both were frozen long before RCI-002 existed, so reading them is confirmation and not selection.
**No third arm is read.** Reading r=16, N16, the α line, a MAG-001 arm or a 4B arm here would be
selection, would spend the instrument, and is out of scope.

## The claim under test on E2

From the decomposition of RCI-001, recorded in `RCI001_CONFIRMATION_001.md` Appendix A:

> On `rule_exception`, D scored **25/25** where the question read *"Which role …"* and **2/25**
> where the same question read *"Who …"*, emitting a person name on **23/25** of the latter and
> **0/25** of the former. The exempt role has **no occupant anywhere in the context**, so what
> D emits is an arbitrary person, not a conflated one.

**The wh-compliance account:** the delta made the answer more determined by the question's
surface form and less overridable by the context. That is *exploratory* — post-hoc, on a spent
bank, and the fifth account proposed for this data. It is registered here rather than claimed.

### The 2×2 that discriminates it

Identical carve-out reasoning in all four strata — four rules over four places, one exempt role
each, and the question names the place. Frequency, first position and last position are uniform
across candidates **by construction**, which is why there are four carve-outs rather than one.

| stratum | interrogative | answer |
|---|---|---|
| `carve_out_role_congruent` | "Which role …" | the ROLE |
| `carve_out_role_incongruent` | "Who …" | the ROLE |
| `carve_out_badge_congruent` | "Which badge number …" | the BADGE |
| `carve_out_badge_incongruent` | "Who …" | the BADGE |

In the incongruent strata the exempt party is identified **only** by badge number, never by
role or name, so the question is determinate and **no person is determined**. Asserted per pack
at build time, not assumed.

| hypothesis | role/cong | role/incong | badge/cong | badge/incong |
|---|---|---|---|---|
| **wh-compliance** | none | **DAMAGE** | none | **DAMAGE** |
| **role↔occupant binding** | damage | damage | damage | damage |
| **output-type shift** | damage | damage | none | none |

**Registered prediction:** damage concentrated in the two incongruent strata, with the
congruent strata not significantly damaged. Tested as an interaction, conditioned on B-correct.

**What disconfirms it.** Damage in the congruent strata too → the effect is not about the
interrogative. Damage in neither → it did not replicate at all, and the RCI-001 observation was
a property of that bank. **Both are reportable and neither is a failure of the instrument.**

Note the badge/incongruent cell scored *no damage* on RCI-002's first calibration twin, when its
question still read *"…Answer with the badge number."* Naming the answer type re-types it and
destroys the incongruence. That wording is gone; the cell is now a bare "Who". This is recorded
because it is the kind of thing that otherwise gets quietly fixed and never mentioned.

## Registered secondary on E1 — distractor density

Unanswerable rows are built at **low** and **high** same-class distractor density, balanced
across packs. Flat loss across densities implies a confidence shift; loss scaling with density
implies capture. Secondary, and it cannot change the E1 verdict.

## What is different from RCI-001, and why

| fix | the defect it answers |
|---|---|
| clause **and** block order randomised | every correction followed what it corrected; `last_candidate_in_context` scored 78/78 on two strata |
| **extraction** predictors in the published set at build time | all four published floors were distributional; the real floor was 0.3711, not 0.2773, and was found after the seal was spent |
| **structural** predictors, a third class | a decoy sentence listed 3 of 4 roles, always omitting the exempt one; set difference scored **1.000**, and **120/120 on the sealed bank**, while every published floor read 0.20–0.29 |
| **the build REFUSES to ship** above a floor bound | RCI-001 *reported* its floors, and a report cannot stop a defective bank shipping |
| an adversarial red-team pass **before** sealing | that is how the set-complement leak was found — six lenses, each proposal executed rather than argued |
| per-stratum gate (`gate_strata`) | `temporal_correction` was above ceiling *and* at its extraction floor at once; a fifth of the bank was decoration |
| wh-congruence as an explicit **crossed factor** | it was confounded with `variant` and carried the largest effect in the bank, unseen for a week |
| `paraphrase` holds the interrogative **fixed** | same confound, other side |
| references stored **bare**, normaliser declared | 52 of 384 rows turned on a definite article; two sessions reconciled them by hand twice |
| one carve-out **per role** | with one exception the congruent cell was 24/24 — a ceiling in the control cell the whole 2×2 leans on |
| a **lacuna note in every row** | silently deleting a block left prose that reads complete; B abstained on 0 of 8 rows in seven of nine strata, and E1 had no power at all |
| **claim the read budget before rendering** | RCI-001 incremented it after, so the record said `unspent` while a read was in flight — and an audit that arrived in that window was too late to apply |

## Cost and irreversibility

640 rows × 2 systems = 1,280 generations at 24 new tokens, 10 chunks per system, ~2.5 minutes
per system on this host. **After this read the sealed bank is spent and cannot be reused for
any purpose.**

The durable asset is the generator, not the bank. `build_rci2.py` can mint RCI-003 from a fresh
disjoint entity partition at zero model cost. The scarcity is partition space, and extending it
is writing new entity pools.

## What a confirmation here does and does not license

**Does.** E1 would make the abstention damage confirmed on a *second*, independent instrument —
the first replication of anything in this programme. E2 would make the wh-compliance account a
pre-registered finding rather than the fifth post-hoc story about one read.

**Does not.** It does not license promotion, release, or `quality_improvement_established`. It
says nothing about r=16, N16, the α line, the k line, the 4B arms or any composition arm. It
remains **one training draw of one delta**, and REP-001 measured single-draw spreads of 7 and 13
rows on other endpoints. A second draw is the thing no instrument can substitute for.
