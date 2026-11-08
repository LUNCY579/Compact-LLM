# What this programme now knows, and the five things worth doing next

**Written 2026-09-18 after the first confirmed result. Everything below is either measured with
a receipt, or labelled as a hypothesis with the test that would settle it. Costs are in wall
clock on this host; "no training" means zero optimizer steps.**

---

## Part 1 — what is established, by confidence

### Confirmed (two instruments, each registered before its read, neither used to select)

**1. The delta answers the interrogative rather than the context.** RCI-002, 120 incongruent
rows: B emits a person on 85, D on **99**, **14 discordant / 0 reversals**, p = 1.22 × 10⁻⁴.
The control decides it — on **congruent** rows of the same reasoning, each system emits a person
**once in 120**. Interaction p = 8.2 × 10⁻⁵. This is the programme's first confirmed *mechanism*
rather than a caution, and it is the fifth account of the RCI-001 damage, registered in advance
rather than told afterwards.

**2. The delta damages literal-`UNKNOWN` abstention.** 40/128 → 25/128 on RCI-001's sealed bank,
p = 6.10 × 10⁻⁵, **zero gains and fifteen losses**. Still confirmed — but **reframed by RCI-002**:
the frozen source abstains 0.547 when a salient note names the missing fact and **0.141** when
the same fact is missing and the sentence is phrased like its answerable twin. So this is better
stated as *"the delta is worse at responding to an explicit unanswerability marker"*. Whether it
damages abstention without one is open; RCI-002's E1 was underpowered by construction once the
marker was matched away (B 23/160, D 20/160, p = 0.25) and was **demoted before the read**.

### Measured, one training draw, report-only

| finding | receipt |
|---|---|
| a 0.056%-relative delta installs a large narrow capability (14 → 232/235 of 384) and costs generality | INT-001 / SPARSE-001 |
| **magnitude is the operative variable** — mask, rank truncation and scalar scaling are *one lever*; a full-rank update at matched norm scores 236 vs rank-16's 235 | DMRF-001 (N16) |
| **allocation matters and the trained profile is near-stationary** — flattening the relative depth profile at fixed dose costs **51 rows** (p = 2.2 × 10⁻¹⁰); steepening gains nothing and turns over | MAG-001 |
| the composition penalty **partitions**: ~73% magnitude, a real **16-row residual** of genuine conflict | MAG-001 `TH16s` |
| **subspace overlap does not predict interference** — near-orthogonal deltas still destroy each other; 0/21 pairs nested | COMPOSE-001 + DMRF-001 |
| **a skill's subspace is set by its data, not its seed** — two draws of one recipe converge at 0.60 against a 0.0885 null | COMPOSE-001 |
| per-item headroom is real (**+31**, 4× the floor) and **unaddressable**: family cap +3, no feature partition beats a size-matched random one | DMRF-001 |
| **scale amplifies the tiny-delta effect** — 270/384 and 180/256 on a 4B base from 142 s of training, beating every 596M arm on both axes at once | QWEN-FT-001 |
| multi-reasoning adds nothing here — MR is the *identity* on the delta candidate, 0 repairs / 0 damage over 128 items, role oracle headroom 2 | DMRF-001 |

### What the confirmation read decomposes into — **exploratory, receipt-backed**

Receipt: `confirm/decompose_rci.py` → `rci001-20260918/RCI_DECOMPOSITION.json`. Full write-up in
`confirm/RCI001_CONFIRMATION_001.md` Appendix A.

**Four mechanism hypotheses were proposed off this read and all four are dead** — two mine
(H-TMD, output-type shift), two the peer's (recency adoption, defeasible override). H-TMD died
because the damage also hits *answerable* rows; output-type died because D emits a person name
on **78/78** `alias_resolution` rows and still loses 12, so the answer *form* is intact.

What survives is **two separate effects**, and only the second is the confirmed one:

| | n | B | D | gains | losses | p | loss rate over B's correct rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| **binding**, answerable | 153 | 97 | 67 | 1 | **31** | 1.5e-08 | **0.320** |
| non-binding, answerable | 231 | 128 | **143** | **18** | 3 | 1.5e-03 | 0.023 |
| binding, unanswerable | 51 | 3 | 2 | 0 | 1 | 1.00 | 0.333 |
| non-binding, unanswerable | 77 | 37 | 23 | 0 | **14** | 1.2e-04 | 0.378 |

**Effect 1 — a task-specific answerable regression**, 32.0% of B's correct rows lost against
2.3% elsewhere, Fisher 3.3e-10. Both sessions first read this as **role/occupant conflation**.
**That is the fifth dead account.** Splitting by VARIANT — which nobody had done — localises the
whole effect to one cell, and article-tolerant scoring removes the convention noise:

| cell | interrogative | n | B | D | losses | rate given B-correct |
|---|---|---:|---:|---:|---:|---:|
| `rule_exception`/base | **which role** | 25 | 25 | **25** | **0** | 0.000 |
| `rule_exception`/counterfactual | **which role** | 25 | 25 | 24 | 1 | 0.040 |
| `rule_exception`/paraphrase | **who** | 25 | 15 | **2** | **13** | **0.867** |

Person-name emissions: **0/25** base, **23/25** paraphrase. And the conflation account needs an
occupant that does not exist — in the generator `exempt_role = roles[1]` while the only
role→person binding is `roles[0] → alias_person`, and on 21 of 25 rows D emits *some other*
person entirely. There is nothing to conflate.

The same variant split runs through the positive finding: `count_with_exclusions` is base 13/13,
counterfactual 14/15, **paraphrase B 3 → D 16** — 14 of the 17 gains — and both emit bare numbers
throughout, so it is not output form either. B is brittle to *"After the withdrawal, how many…"*
and double-subtracts; D is not.

**One statement covers the largest damage and the largest gain**, in opposite directions: the
delta made the answer more determined by the question's surface form and less overridable by the
context. Helps when the form points at the right operation, destroys when it types the wrong
entity. **Registered, not claimed** — it is the fifth account of this data and the four before it
also looked elegant. `alias_resolution`'s 12 losses are uniform across variants with a congruent
interrogative throughout, and nothing explains them.

**Effect 2 — the abstention loss, and it is UNIFORM.** Not concentrated anywhere: 0.476 on
`temporal_correction` against 0.200 on `object_state_tracking` (Fisher 0.159), 0.333 binding
against 0.378 non-binding (Fisher **1.00**). About a third of the source's correct abstentions
go, wherever they are. That matches the programme's long-standing invariance finding and no
proposed mechanism touches it.

**The denominator did the work, and it is the transferable lesson.** Against *all rows* the two
groups look dissociated on the unanswerable half too (Fisher 4.5e-03) — the reading both
sessions initially took. Against **the rows B got right**, the only rows where a loss is
possible, it is **gone (Fisher 1.00)**: the binding group abstained correctly on 3 of 51 rows and
had nothing to lose. H-TMD's concentration premise fails identically — "10 of 15 losses" is a
count, and B abstains most often exactly there. Now enforced by
`analysis/paired.py::subgroup_dissociation`.

---

## Part 2 — the five things worth doing, ranked

### 1. Cross reasoning structure with answer type, on RCI-002. *No training. ~1 hour.*

Effect 1 is the larger finding and it rests on a post-hoc split of five task types. One build
tests it properly and kills the last surviving alternative at the same time.

**BUILT — this is RCI-002's endpoint E2**, and the design changed once the variant split landed.
The badge-number probe as originally specified was **confounded**: "Which badge number…" is a
congruent interrogative, so wh-compliance and conflation both predict no damage there and the
probe discriminates nothing. Crossing **wh-congruence** explicitly is the fix:

| stratum | interrogative | answer | wh-compliance | binding | output-type |
|---|---|---|---|---|---|
| `carve_out_role_congruent` | "Which role…" | ROLE | none | damage | damage |
| `carve_out_role_incongruent` | "Who…" | ROLE | **DAMAGE** | damage | damage |
| `carve_out_badge_congruent` | "Which badge number…" | BADGE | none | damage | none |
| `carve_out_badge_incongruent` | "Who…" | BADGE | **DAMAGE** | damage | none |

Three hypotheses, four cells, no two patterns alike. Identical carve-out reasoning throughout,
and in the incongruent strata the exempt party is identified **only** by badge number, so the
question is determinate and no person is determined — asserted per pack at build time.

**Keep the abstention arm separate**, as E1: unanswerable rows at low and high distractor
density. Uniformity predicts flat.

Prereg: `confirm/RCI002_CONFIRM_PREREG.md`. **The sealed read is not spent.**

### 1b. ~~Settle `alias_resolution`~~ — **PROBE BUILT AND RUN. Salience refuted.**

The one stratum nothing explains, now much better characterised and no better explained —
`ALIAS_RESOLUTION_INVESTIGATION.md`. Pooled across both sealed banks it is **0 repairs / 17
damages, p = 1.53e-05**: purely subtractive, replicated on disjoint entity pools. Context that
stops it being over-read: the delta *repairs* 30 answerable rows across those banks, so zero
repairs is a signature and not the background.

**My salience-over-binding account is refuted** — every row the delta lost it named a salient
mover, and so does the frozen source on 13 of its own 13 errors, with identical error profiles
(Fisher p = 1.000 both banks). The delta changed the *rate* of an error the source already
makes, not its kind. And nothing measurable distinguishes the lost rows.

**ALIAS-001** (`probe/build_alias_probe.py`, a mechanism scout — no seal) crossed salience as a
designed factor. Damage is present in all three conditions and does **not** shrink when the
answer is made the loudest person in the context — it rises, 8 → 11 → 17, at rates 0.105 → 0.136
→ 0.160, with Fisher `low` vs `high` conditioned at **p = 0.383**. Verdict **UNIFORM_WEAKENING**.
The manipulation worked — both systems gain a lot from salience, B 76 → 106 and D 68 → 89 — so
this is a refutation, not a null.

**Pooled across three independent banks: 1 repair / 54 damages, p = 3.1e-15.**

The peer's corpus audit supplies the other leg: RRP-015's 384-row control file contains **zero
role phrases and zero person names**, verified independently here. Alias resolution is a skill
the fine-tuning never exercised. **Capacity is now the leading account** — and, adopting the
peer's caveat verbatim, absence in the data does not establish causation.

**What remains, and it is now the cheapest decisive test in the programme:** a dose ladder on
this probe. The bank is re-runnable at zero build cost and the α and k lines already exist.
Uniform weakening predicts monotone decline in D with magnitude at a constant damage-rate ratio
across all three salience conditions. That would move capacity from leading to measured.

### 2. If an account survives (1): test the repair it implies. *Training. ~2 h + a grant.*

Which repair is worth building depends entirely on which account lives, and they point
opposite ways. Wh-compliance implies a training mixture that varies the interrogative against a
fixed answer type — cheap, and it would *also* predict the abstention loss, since committing to
an answer type from the question is exactly what stops you noticing the context determines
nothing. Uniform abstention loss alone implies something about calibration and nothing about
question form.

**Do not design the repair before (1) returns.** FIVE accounts have died here, and each would
have justified a different and wasted training run. The two that survived longest died on a
denominator; the fifth died on a `groupby` nobody had run.

**Predeclare that this may fail**, and that the prior is poor: every attempt in this programme
to repair abstention has failed or been corrected downward — M4's apparent repair was later
corrected from 13 to 9 literal.

### 3. Diverse deltas, not cleverer application of one. *Training. The direction nothing has closed.*

Two independent lines converge here. The redundancy null says +31 is thin because all seven
arms are scalar variants of **one** δ; and the composition work says near-orthogonal deltas
still interfere, with a 16-row residual beyond magnitude.

Everything this programme has tried is *one delta, applied differently*, and that is now
closed on three grounds — magnitude is the only lever, allocation is already at a stationary
point, and per-item selection is unaddressable. **The untried direction is more than one
genuinely different delta**, and the open question is whether the 16-row conflict residual
grows, stays flat, or vanishes with a third and fourth skill.

This is the expensive one and it needs a training grant.

### 4. Replicate the 4B result at two more seeds. *Training. ~1 h. The cheapest real win.*

The single largest effect in the corpus — 270/384 and 180/256, beating the entire 596M frontier
on both axes from 142 seconds — rests on **one draw**, and REP-001 measured single-draw
retention varying by 13 long-context rows.

If it replicates, the strategic conclusion ("the 596M line is a cost decision, not a capability
one") is established rather than suggested. If it does not, the headline is one sample. Either
answer is worth more per GPU-minute than anything else on this list.

Also outstanding and cheap: a **BF16** arm. 4-bit cost its own control 11 long-context rows, so
BF16 should score above 180 and has never been run.

### 5. ~~RCI-002~~ — **BUILT, READ, SPENT**. *`confirm/build_rci2.py` → `rci002-20260918/`.*

Done, and it cost more than the "two one-line fixes" this entry predicted — the fixes below
are what shipped, and four more were found by measuring rather than reasoning:

- **randomise clause order** so corrections and exceptions are as often stated *before* as
  after the material they modify — this is what made `last_candidate_in_context` score 100% on
  two of five task types;
- **add extraction predictors to the published floor set** so the floor is computed at build
  time rather than audited after the seal is spent;
- **redesign `temporal_correction`'s answerable half, keep its unanswerable half.** Answerable
  it is 78/78 for both systems and 78/78 for the recency rule — decoration. Unanswerable it
  carried 10 of the 15 confirmed losses and works;
- **claim the read budget before rendering, not after.** RCI-001's drawdown and its render
  raced, which is how a floor correction arrived minutes after the seal was already spent;
- **gate per stratum** (`gate_strata`) and **report convention sensitivity** at build time —
  52 of 384 answerable rows turned on a leading article, and two sessions spent real effort
  reconciling numbers that never disagreed about the model.

640 sealed rows, 8 strata, one surviving confirmation endpoint. Read spent 2026-09-18 on
operator authorisation; result in `confirm/RCI002_CONFIRMATION_001.md`. **CONFIRMED.**

What the build actually taught, beyond the planned fixes: the gate blocked its own bank **five
times**, each on a defect introduced while fixing the previous one. The decoy register added to
close a positional leak opened a set-complement leak worth 1.000. Listing a random *subset* of
each class then broke frequency ties. Only listing **every** member of every class, with one
carve-out per role, makes position and frequency uninformative by construction. **None of that
was foreseeable from the RCI-001 post-mortem** — it came from running the gate and the red team
against each draft.

---

## Part 3 — what is now reusable

`method/floors.py`, fixture-tested against the defects that actually happened:

| check | the failure it exists to catch |
|---|---|
| **extraction predictors** | RCI-001 published four floors and **not one read the context**; a recency rule beat all of them |
| distributional predictors | MRS spent a scout at the constant-answer baseline |
| **ceiling** | DMRF-SCOUT-001's long-context stratum was 32/32 — cleared its floor, could not show a regression |
| three-way verdict | CAL n=6 reported a real 17% signal as FAILS when the answer was "this split cannot certify it" |
| `immunity()` | choose a primary endpoint no shortcut class can reach — and keep **extraction** immunity distinct from **distributional**, which endpoint choice does *not* buy |
| `require_pack_structure()` | RCI-001 survived an unanticipated defect on a habit copied from the Demo suite; per-pack all-correct had a 0/128 shortcut floor. A habit is not a safeguard |
| `gate_strata()` | RCI-001's aggregate looked sound while a fifth of its answerable rows were above ceiling *and* at the extraction floor |
| `convention_sensitivity()` | 52 of 384 rows turned on a definite article; prefer a convention-free endpoint, and a literal sentinel is one |
| `subgroup_dissociation()` | **the denominator.** A subgroup where the baseline is near the floor cannot show damage; scoring losses against all rows reports the baseline and calls it the effect |
| **`structural()`** — a third predictor class | RCI-002's first draft cleared every published floor at 0.20–0.29 and fell to one line of set arithmetic: a decoy sentence listed 3 of 4 roles and always omitted the exempt one. **1.000 on both carve-out strata, 120/120 on the sealed bank.** No positional or distributional rule can express "the member of list A missing from list B" |
| **the build REFUSES, it does not report** | RCI-001 published its floors and shipped anyway. `build_rci2.py` exits non-zero above a floor bound, and it has now blocked its own bank five times |
| **an adversarial pass BEFORE sealing** | six lenses, every proposal executed rather than argued. That is how the set-complement leak was found — while it was still fixable |

Plus, already committed and used: `analysis/paired.py` (repair/damage never netted, noise floor
enforced in code), `audit/feature_partition_ceiling.py` (upper-bounds every selector reading a
feature set, immune to power), and the **calibration-twin + seal** pattern in `confirm/`, which
is how you ship an instrument you are not allowed to score.

### The three process lessons, stated so they transfer

1. **Put the number in a script, not in prose.** Three load-bearing figures in this programme —
   `deep-any`'s 35, C1's oracle headroom, and a routing composite — existed only in narration.
   One was unreproducible, one was wrong by a factor of ten, and one was never executed.
2. **State the scope inside the sentence.** "Not in X" and "not anywhere" are different claims;
   I conflated them twice and was wrong both times.
3. **Apply your own standard inward.** I wrote `noise_verdict` to stop within-noise differences
   being reported as directions, ran it on a peer's margins, and not on my own finding.
4. **Check the denominator before proposing a mechanism.** Four mechanism hypotheses died on
   this one read; the last two both started from counts that were really the *baseline's*
   distribution. A rate over the rows the baseline got right would have killed both on the
   first pass, at no cost.

---

## Part 4 — what the evidence says *not* to do

- **More α/k/rank points.** One lever, three parameterisations, frontier flat. A fourth
  reparameterisation buys another battery and the same curve.
- **A selector for this delta.** Per-item headroom is real and unaddressable: family cap +3,
  and no feature partition beats a size-matched random one except `status`, whose ceiling is
  +7 — exactly the noise floor.
- **Per-layer gating.** Closed on a measurement rather than on C1's miscomputed zero: the
  trained allocation is an asymmetric, non-monotone optimum.
- **Multi-role inference on this backend.** MR is the identity; roles diverge textually and
  produce correlated outcomes, so there is nothing for a selector to select between.
- **Spending RCI-002 on an arm comparison.** That is selection, and it burns the instrument.

## The one-paragraph version

A tiny weight update installs a large narrow skill on a small base and reliably costs
generality; every way of *applying* that update turns out to be one lever acting through
magnitude, and the trained allocation is already at a stationary point. What the update costs
is now confirmed rather than observed: it damages abstention, purely subtractively, and that
damage is **uniform** — about a third of the source's correct abstentions wherever they occur,
with no mechanism yet proposed that survives. Decomposing the same read turned up a **second
and larger effect nobody had separated**: a task-specific answerable regression on anything
requiring a role to be resolved to its occupant, 32% against 2.3%. Four mechanism hypotheses
died getting there, and the thing that killed the last two was a denominator. The only
intervention that has ever beaten the frontier on both axes at once is **a bigger base model**,
and that rests on a single draw that nobody has replicated. The direction nothing has closed is
**more than one delta**.

---

# UPDATE 2026-09-18 (late) — what changed after the dose ladder

Appended, not rewritten. Everything above stands as written.

## 1. The mechanism question moved, and the answer is not the one being chased

Six mechanism accounts had died; **capacity** was the survivor by default, on the corpus null
(RRP-015's 384-row training file contains zero role phrases and zero person names). ALIAS-002
then produced a perfect monotone dose response (ρ = −1.000 in every salience condition) — which
was registered at the time as **necessary and not sufficient**, because any magnitude-scaling
mechanism predicts it.

`analysis/coupling_ladder.py` asked the question that separates them, using data already on
disk. **Marginal damage per acquisition row gained spans 160× across the α ladder.** The top
rung buys **+1 row of 384** and costs **20.8% of all the alias damage** at p = 0.0034.

**Strong capacity — "the damage is the price of acquisition" — does not survive that.** What
survives is a **coverage** form: the corpus determines *which* skills are exposed, magnitude
determines *how much* they lose. Those are different claims and the programme had been using
one word for both.

## 2. There is an operating-point result and the deployed artifact is on the wrong side of it

α ≈ 0.5–0.75 now has **two independent arguments**: INT-001 selected α = 0.50 from the
acquisition side on `retained` (n = 47), and this arrives at the same region from the damage
side on a 360-row preservation endpoint. Past α ≈ 0.75 the update is paying preservation for
nothing. **The deployed artifact is α = 1.** This is the most actionable thing on the page.

## 3. Two method lessons that generalise past this programme

**The oracle keeps coming back, and it is subtle the second time.** The peer's routing composite
reported "each arm keeps its own score on its own domain" — a *benchmark-level* oracle, reported
after correctly rejecting the *per-item* oracle. Rejecting one kind of envelope is not rejecting
the class. See `C1_CORRECTION_20260918.md`; this is the second instance.

**Endpoint classes differ in floor immunity, and it is not obvious which is which.** A paired
B-vs-D contrast survives a saturated floor, because a deterministic answer-agnostic rule emits
the same string for both systems and yields zero discordant pairs. A **route-vs-sum** comparison
does **not**, because there the floor predictor *is* the route arm's mechanism. Ask which of
those shapes an endpoint has before trusting a published floor to protect it.

**"Matched magnitude" needs the question *matched on what*.** `analysis/norm_audit.py` found the
composition arm's total norm matched to ten significant figures while **each skill sat at 68.9%
of its solo strength** — and that a same-skill control does not control for it, because aligned
deltas AVERAGE under the same formula where orthogonal ones ATTENUATE.

## 4. Things now known about the weights that no endpoint would have shown

- `cosine(δ_T, δ_H)` on the rank-16 composition updates is **0.0027** against a recorded
  *subspace* overlap of 0.1508. The updates are already orthogonal as vectors; interference was
  still severe. A disjoint-block test is therefore about shared-**coordinate** interference, not
  orthogonality.
- The LWF anchor **rotates** the update rather than shrinking it: cosine to the unanchored arm
  goes **1.000 → 0.524 → 0.349** across λ = 0 → 0.25 → 0.5 while the norm only falls to 82%.
- **λ = 0.25 sits ~58° off λ = 0 and acquires 197 against 198.** Many directions reach the same
  acquisition, so a trained δ is a representative of a large equivalence class — and the class
  members need not be equally compatible with a second skill. That is a search space nobody has
  looked in.

## 5. Reproducibility gap closed

ALIAS-001/002 were generated by an ad-hoc scratchpad script that was never committed — the
failure process lesson 1 already names. `probe/generate_alias.py` is the reconstruction, and it
is **proven** rather than asserted: it reproduces the frozen source's existing 64-row chunk
**byte-for-byte, 64/64**. It also records torch, the CUDA build, the interpreter and the
generator's SHA, none of which the original manifest did.
