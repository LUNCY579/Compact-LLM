# What this programme now knows, and the five things worth doing next

**Written 2026-09-18 after the first confirmed result. Everything below is either measured with
a receipt, or labelled as a hypothesis with the test that would settle it. Costs are in wall
clock on this host; "no training" means zero optimizer steps.**

---

## Part 1 — what is established, by confidence

### Confirmed (one instrument, registered before the read, never used to select)

**The k=0.25 delta damages literal-`UNKNOWN` abstention.** 40/128 → 25/128 on RCI-001's sealed
bank, exact McNemar p = 6.10 × 10⁻⁵, **zero gains and fifteen losses**. Registered criterion
met. This is the programme's only confirmed claim and it is a caution, not a capability.

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

**Effect 1 — a task-specific answerable regression** on tasks needing a role resolved to its
occupant (`rule_exception`, `alias_resolution`): 32.0% of B's correct rows lost against 2.3%
elsewhere, Fisher 3.3e-10. It looks like **role/occupant conflation** — asked for a role D names
the person (24/75 name emissions against B's 10; 14 of its 19 losses), asked to resolve an alias
it names the wrong person. Never previously separated from Effect 2, and it is the larger effect.

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

**The cross.** Hold the reasoning structure fixed and vary what the answer *is*: ask
`rule_exception` for the exempt role's **badge number**, ask `count_with_exclusions` for the
**names** of the items counted. The two accounts invert:

| probe | role/occupant conflation predicts | output-type shift predicts |
|---|---|---|
| `rule_exception` → a number | damage **persists** — you must still resolve which role is exempt | damage **disappears** |
| `count_with_exclusions` → names | **no damage** — no role to resolve | damage **appears** |

Output-type is already refuted on the `alias_resolution` form evidence, so this is confirmation
rather than a coin flip — which is the right time to register a prediction. **Registered: damage
persists on the numeric `rule_exception` probe and does not appear on the named-count probe.**

**And keep the abstention arm separate.** Effect 2 is uniform, so vary distractor density on
unanswerable rows as its own endpoint: zero, low and high same-type tokens. Uniformity predicts
flat; anything else is a live mechanism for the one confirmed claim.

Both need RCI-002 (see 5) or a spendable scout — **not** the spent RCI-001. Register the two
effects as **two endpoints**; a single "damage" framing is what hid Effect 1 for a week.

### 2. If an account survives (1): test the repair it implies. *Training. ~2 h + a grant.*

Which repair is worth building depends entirely on which account lives, and they point
opposite ways. Role/occupant conflation implies training rows that force the role→occupant
binding to be *re-derived* rather than recalled. Uniform abstention loss implies something
about calibration and nothing about binding at all.

**Do not design the repair before (1) returns.** Four hypotheses have already died here, two
of them mine, and each would have justified a different and wasted training run.

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

### 5. RCI-002, with the two defects fixed. *No training. ~30 min.*

RCI-001 is spent. Its generator is not, and the known defects are all cheap:

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

Fresh entity pools are the only real cost. **Build it before it is needed**, so the next
confirmation is not blocked on instrument construction.

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
