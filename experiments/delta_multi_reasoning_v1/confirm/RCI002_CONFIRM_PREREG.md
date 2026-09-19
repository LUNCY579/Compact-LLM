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
| metric | correct literal `UNKNOWN` | **emits a person name**, where the context determines none |
| status | **replication** of RCI-001's confirmed result on a fresh bank | **new**, and the first pre-registered test of the wh-compliance account |
| convention risk | none — a literal sentinel admits no article or punctuation ambiguity | none — membership in a name vocabulary, not string equality with a reference |
| registered band (from the twin) | B **42–133**, D **22–83** | B **20–80**, D **30–105** |
| direction | **D < B** | **D > B** |

### E2 is an emission rate, and accuracy could not measure it

**This operationalisation was chosen on the calibration twin, after the accuracy version failed
there. The sealed bank was not consulted and has never been read.** That is what a twin is for —
it is openly spendable and exists to settle exactly this kind of question — but choosing an
endpoint definition using data is worth stating plainly rather than presenting as foresight.

Accuracy on the incongruent strata came back **0/24 for both systems** on `carve_out_badge_incongruent`.
Not a capability floor: asked *"Who was permitted to enter X?"* where the context identifies the
exempt party only by badge number, the frozen source answers *"The holder of badge 14 was
permitted to enter the Bell Passage without one."* — substantively correct, and exact match
scores it zero against the reference `14`. A floor created by answer form.

What the wh-compliance account predicts is not a wrong answer but a wrong **answer class**:
emitting a person where the interrogative types one and the context determines none. That is
also what the RCI-001 decomposition actually measured — 0/25 person emissions on *"Which role"*,
23/25 on *"Who"* — with accuracy standing in for it. So E2 measures it directly.

Twin result, reported in full including the version that did not work:

| | n | B | D | discordant | p |
|---|---:|---:|---:|---:|---:|
| **E2 emission, incongruent** | 48 | 20 | **27** | 7 D-more / **0** B-more | 0.016 |
| **E2 control, congruent** | 48 | **0** | **0** | 0 / 0 | 1.00 |
| E2 accuracy — *not used* | 48 | 14 | 12 | 0 / 2 | 0.50 |

**The control is what makes E2 mean what it says.** On congruent rows of the *same* carve-out
reasoning, neither system emits a person at all. So E2 is not measuring a general tendency to
name people; it is specific to the interrogative, which is the factor under test.

**A paired emission endpoint is immune to every answer-agnostic predictor by construction.** A
deterministic rule emits the same string for both systems, producing zero discordant pairs, so
it cannot manufacture a paired difference. Floors bound what an *absolute* rate means; they do
not bear on a paired contrast. The same applies to E1, whose mechanical floor is high and whose
verdict is unaffected by that.

### Analysis

Both are tested with **exact two-sided McNemar** on discordant pairs. **Holm-Bonferroni across
the family of two.** Confirmation criterion per endpoint: **Holm-adjusted p ≤ 0.05 and the
registered direction** — **D < B** for E1 (the delta abstains less), **D > B** for E2 (the delta
emits a person more often where none is determined). The directions differ because the metrics
point opposite ways; a single "D < B" rule would have been wrong for E2 and is not used.

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

Predicted **person-name emission** by cell:

| hypothesis | role/cong | role/incong | badge/cong | badge/incong |
|---|---|---|---|---|
| **wh-compliance** | none | **D > B** | none | **D > B** |
| **role↔occupant binding** | D > B | D > B | D > B | D > B |
| **output-type shift** | D > B | D > B | none | none |

**Registered prediction:** the excess is confined to the two incongruent strata, and the
congruent strata show no person emissions from either system. On the twin the congruent cells
read 0 and 0 — if the sealed bank reproduces that, the three hypotheses separate cleanly.

**What disconfirms it.** Person emissions on congruent rows too → the effect is not about the
interrogative, and role-binding or output-type is the better account. No excess on incongruent
rows → it did not replicate, and the RCI-001 observation was a property of that bank. **Both are
reportable and neither is a failure of the instrument.**

### Red-team result, added before the claim — E1's true mechanical floor is **1.000**

Receipt: `confirm/redteam_rci2.py` → `REDTEAM_VERIFIED.json`. The twenty-four proposals my
workflow could not verify are implemented and executed. Two of them solve E1 outright:

| attack | target | n | rate | published floor | |
|---|---|---:|---:|---:|---|
| **`trigger_absence_abstainer`** | **E1** | 64 | **1.000** | 0.000 | abstain when the task's characteristic phrase is absent |
| **`lacuna_reader`** | **E1** | 64 | **1.000** | 0.000 | read the lacuna note, abstain if it names this task's block |
| `block_census_abstainer` | E1 | 64 | **0.000** | 0.000 | count blocks — the length tell, and it fails |
| `sole_alias_anchor` | `alias_resolution` | 32 | **0.750** | 0.292 | unique filler of a template |
| `exam_template_day` | `temporal_correction` | 32 | **0.750** | 0.542 | day inside a fixed template |
| everything else | | | ≤ 0.281 | | does not beat its floor |

**E1's abstention task is mechanically solvable at 1.000 by a regex.** The build claimed E1 was
extraction-immune; that claim was true and too narrow — no extractor can *emit the sentinel*,
but a rule that checks whether the determining phrase is present can. That is a floor the
published set did not contain, found by the same class of attack that broke RCI-001, and this
time **before** the seal rather than after.

`block_census_abstainer` scoring **0.000** is the lacuna design working: answerable rows drop a
block too, so block counts carry nothing. The tell that survives is semantic, not structural.

**What this changes, and what it does not.**

- **It does not touch E1's verdict.** E1 is registered as a **paired** endpoint. A deterministic
  rule emits the same string for both systems, yields zero discordant pairs, and cannot
  manufacture a difference. Nor can it compress power here: B and D sit at 0.55 and 0.33 on the
  twin, nowhere near the rule's 1.000, so discordant pairs do not collapse.
- **It forbids any absolute claim from E1.** "B abstains correctly on N of 160" is not evidence
  that B is doing something hard — it is evidence that B does part of what a regex does
  completely. **No sentence anywhere may read E1 as a competence measurement.** Registered here
  so it cannot be quietly forgotten when the result is written up.
- **`alias_resolution` and `temporal_correction` have true floors of 0.75**, not the 0.29 and
  0.54 published. Neither is an endpoint; both are controls, and their absolute readings are
  correspondingly weak.

**The fix for RCI-003, specified now rather than discovered later:** make the unanswerable
variant *present but insufficient* rather than absent. Keep the determining sentence's template
and destroy only its content — "the standing exception applied to one role, but the entry naming
it is illegible". Phrase present, answer absent, and `trigger_absence_abstainer` drops to chance.
Not done here: it requires a rebuild and a fresh calibration, and E1's paired verdict does not
need it.

### Cross-bank check, added before the claim — receipt `CROSS_BANK_EMISSION.json`

RCI-002's twin did not reproduce RCI-001's **accuracy** effect (B 14 → D 12, p = 0.50, against
15 → 2) while it did reproduce an **emission** effect. I had said the sealed read could not
separate "accuracy was always a proxy" from "four carve-outs made the task harder". The peer
session pointed out it can be separated *now*, with no read at all: RCI-001's predictions are on
disk, and re-scoring recorded output under a different metric spends nothing.

| bank | cell | n | B | D | D-more | B-more | p |
|---|---|---:|---:|---:|---:|---:|---:|
| RCI-001 | `rule_exception`/paraphrase — **"Who"** | 25 | 10 | **23** | 13 | **0** | 2.4e-04 |
| RCI-001 | `rule_exception`/base — "Which role" | 25 | 0 | 0 | 0 | 0 | 1.00 |
| RCI-001 | `rule_exception`/counterfactual — "Which role" | 25 | 0 | 1 | 1 | 0 | 1.00 |
| RCI-002 twin | `carve_out_role_incongruent` — **matched cell** | 24 | 5 | **9** | 4 | **0** | 0.125 |
| RCI-002 twin | both incongruent strata pooled — **E2** | 48 | 20 | **27** | 7 | **0** | 0.016 |
| RCI-002 twin | both congruent strata | 48 | 0 | 0 | 0 | 0 | 1.00 |

**Both explanations are partly right, which neither of my two stories allowed.** The direction
replicates with **zero reversals on both banks** and the congruent controls read zero on both —
so accuracy was standing in for emission. But D's emission rate falls from **0.92 to 0.375**
between banks, so the four-carve-out design really did shrink the effect. Calling this a clean
replication would be as wrong as calling the accuracy result a failure to replicate.

**Consequence for this prereg:** the registered E2 direction stands, and the registered band
(D 30–105 of 120) already reflects the twin's reduced rate rather than RCI-001's. Power at
n = 120 with the twin's 7/0 discordance scales to roughly 17/0.

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
