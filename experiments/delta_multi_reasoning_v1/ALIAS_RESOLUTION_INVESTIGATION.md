# `alias_resolution` — replicated, purely subtractive, and still mechanism-free

**2026-09-18, after RCI-002's read. Zero model calls** — everything here runs on predictions
already recorded by two spent confirmation reads. Receipt: `analysis/alias_salience.py` →
`rci002-20260918/ALIAS_SALIENCE.json`. **Exploratory throughout.**

This was the one cell nothing explained after five dead mechanism accounts and one confirmed
one. It is now much better characterised and no better explained, which is worth writing down
precisely because the temptation is to stop at a sixth story.

---

## 1. It replicates, and it is purely subtractive

| bank | n | B | D | repairs | damages | p |
|---|---:|---:|---:|---:|---:|---:|
| RCI-001 sealed | 78 | 65 | 53 | **0** | 12 | 4.9e-04 |
| RCI-002 sealed | 60 | 41 | 36 | **0** | 5 | 6.3e-02 |
| **pooled** | **138** | | | **0** | **17** | **1.53e-05** |

**Zero reversals in seventeen discordant pairs, across two independently generated banks with
disjoint entity pools and a rebuilt generator.** Neither bank alone is decisive — RCI-002's cell
is p = 0.063 — and pooled they are.

### The context that stops this being over-read

Across both sealed banks and every answerable stratum, the delta **repairs 30 rows and damages
56**. It is *not* uniformly worse, so a stratum with zero repairs is a signature rather than the
background. Strictly nested strata: `alias_resolution` on **both** banks, `rule_exception` on
RCI-001 (now understood as [wh-compliance](confirm/RCI002_CONFIRMATION_001.md)), and
`exclusion_count_number` on RCI-002.

## 2. The hypothesis I tested, and why it is refuted

Both generators name the aliased person exactly once — *"The \<role\> that week was \<name\>,
who is not named again in this account"* — while movers are named up to three times. So the
correct answer is systematically the **least salient** person present.

> **Salience-over-binding:** asked "Who", the delta reaches for the most available person rather
> than the one the text binds to the role.

This would have unified the alias cell with the confirmed result — one mechanism, two surfaces:
where no person is bound it invents one (confirmed, 99/120), where one is bound but others are
louder it takes a louder one.

**It is refuted as stated.** Salience-over-binding *is* the failure mode — on RCI-001, **all 12**
rows the delta lost it named a mover, and on RCI-002 four of five. But:

| | errors | named a mover | error profile |
|---|---:|---:|---|
| **B** frozen source | 13 | **13/13** | 3 none-of-these + 10 most-frequent-and-last |
| **D** on the same shared rows | 13 | **13/13** | 3 none-of-these + 10 most-frequent-and-last |
| D on the 12 rows it uniquely lost | 12 | 12/12 | 4 none-of-these + 8 most-frequent |

Fisher on the mover share of errors, D vs B: **p = 1.000 on both banks.** The profiles are
identical. **The delta did not change the kind of error; it increased the rate of an error the
frozen source already makes.** Mean mention-count of the emitted person barely moves either —
1.603 for D against 1.500 for B, with the correct answer at 1.308.

So the answer to "does the delta reach for salience *more*" is no. The answer to "is salience
what both models reach for when they fail" is yes, and that is a fact about the task, not about
the delta.

## 3. Nothing measurable distinguishes the rows it loses

| | people present in context | times the alias is named |
|---|---:|---:|
| rows D lost (n = 12) | 2.75 | 1.25 |
| rows D kept (n = 53) | 2.92 | 1.40 |

Both differences are small, in the direction you would predict, and nowhere near separable at
this n. This echoes the programme's earlier per-item result — real headroom, no input-visible
feature that reaches it — and it is the third time a "which rows?" question here has come back
featureless.

## 4. What is now established, and what is not

**Established.** A purely subtractive, replicated regression on role→person binding, 0/17,
p = 1.5e-05, whose error character is indistinguishable from the frozen source's own errors and
whose affected rows carry no distinguishing feature.

**Not established.** Any mechanism. Specifically **not** salience-over-binding as a *differential*
account, and not [wh-compliance](confirm/RCI002_CONFIRMATION_001.md) — the interrogative is
congruent throughout this stratum and the answer form is preserved (a person name on 78 of 78
RCI-001 rows).

**This is a sixth account's worth of temptation and I am not taking it.** The honest position is
that the delta weakens a binding step uniformly, and "uniformly weakens" is a description, not a
mechanism. It is compatible with the delta simply having less capacity left for a skill its
training never exercised, which is not testable from these artifacts.

## 5. What would settle it, and what it costs

1. **A dose ladder on this one stratum.** If the effect is a uniform weakening, alias accuracy
   should fall **monotonically** with delta magnitude, and the error profile should stay
   mover-dominated at every dose. If instead it is a threshold, that is a different claim. The
   α line and the k line already exist; this needs no training, only a fresh alias-only bank and
   about six generation passes. **Cheapest decisive next step.**
2. **Vary salience directly.** Name the aliased person three times and the movers once, inverting
   the salience ordering. Uniform weakening predicts the *gap* between B and D persists; a
   salience account predicts it collapses. Also no training, one build.
3. **Check whether the training data contains any role→person binding at all.** If RRP-015's
   recipe is purely state-transition JSON, "no capacity left for an unexercised skill" becomes
   the leading account, and it is checkable against the corpus rather than the model.

(2) and (1) together separate the three live readings with no training and no new instrument
class — and (2) is the one that would have caught me if I had published the salience account.

## 6. One process note

I wrote the salience hypothesis down, predicted the direction, and it failed on the control —
the frozen source does exactly the same thing. That control existed because the last five
accounts died without one. **The measurement that killed this hypothesis took four lines**: count
what *both* systems reach for, not just the candidate. Every previous dead account in this
programme would have been killed the same way and none of them was.

---

# AMENDMENT 1 — the probe was built, and salience is refuted in the opposite direction

**Same day. ALIAS-001, a MECHANISM SCOUT — no seal, openly spendable.** Build
`probe/build_alias_probe.py`, scoring `probe/score_alias_probe.py`, receipt
`alias001-20260918/ALIAS_PROBE_RESULT.json`. Third entity partition, disjoint from both spent
banks. 40 packs x 3 variants x 3 conditions = 360 rows, both systems, 720 generations.

Salience was made a **designed factor** — the same task, the same congruent interrogative, the
same binding sentence, varying only how often each person is named in neutral activity lines:

| condition | alias named | each distractor | n | B | D | repair | damage | damage given B-correct | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `low` | 1 | 3 | 120 | 76 | 68 | **0** | 8 | 0.105 | 7.8e-03 |
| `equal` | 2 | 2 | 120 | 88 | 77 | 1 | 12 | 0.136 | 3.4e-03 |
| `high` | 3 | 1 | 120 | 106 | 89 | **0** | **17** | 0.160 | 1.5e-05 |

**Registered predictions were: salience → the gap shrinks monotonically; uniform weakening → the
gap is roughly constant. Neither happened in the direction salience needed.** The damage rate
*rose* across the conditions — 0.105, 0.136, 0.160 — and the raw gap rose with it, 8 → 11 → 17.
Fisher on `low` vs `high`, conditioned on the rows B got right: **p = 0.383**, so the rise is not
significant and the result is consistent with constant damage. Verdict by the registered rule:
**UNIFORM_WEAKENING**.

**The manipulation worked, which is what makes this a refutation rather than a null.** Both
systems improve substantially as the answer becomes more salient — B 76 → 106, D 68 → 89. So
salience matters a great deal for the task. It simply does not differentially rescue the delta:
making the correct answer the *loudest* person in the room left the delta losing **17** rows,
more than in any other condition.

`high`'s absolute rates are not readable — the answer is the most frequent person there by
construction and `most_frequent_candidate_in_context` scores 1.000. The gate flagged it
(`BELOW_FLOOR` for B at 106/120) exactly as the build manifest said it would. The paired contrast
is readable in every condition, because a deterministic rule emits the same string for both
systems and cannot create a discordant pair.

## Pooled across three independent banks

| bank | repairs | damages |
|---|---:|---:|
| RCI-001 sealed | 0 | 12 |
| RCI-002 sealed | 0 | 5 |
| ALIAS-001 probe | 1 | 37 |
| **total** | **1** | **54** |

**p = 3.11e-15.** Three entity partitions, three generators, two sealed instruments and one open
probe. The regression is as solid as anything in this programme.

## What now stands, and what killed what

**Salience is dead as an account.** Not merely unsupported — tested with the factor varied, and
the effect went the wrong way. That is two of my own accounts refuted on this cell, the first by
a baseline control I put in, the second by a probe I built to catch myself.

**The capacity account is now the leading one, and it has independent support.** The peer session
audited RRP-015's training corpus: the 384-row control file contains **zero role phrases and zero
person names** — the only alphabetic content in 1.33 MB is instruction vocabulary over JSON state
transitions. Their detector fires on 512 of 512 RCI-001 rows, so the null is informative rather
than a detector failure. I verified the core of it independently: `microfit-train.jsonl` for the
control returns 0 candidate person names and 0 role phrases.

So alias resolution is a skill the fine-tuning **never exercised**, and the delta uniformly
weakens it regardless of how easy the retrieval is made.

**Their caveat is the right one and I am adopting it verbatim:** absence in the data does not
establish causation. The delta could weaken binding for reasons unrelated to what it was trained
on. **Leading account, not established account.**

## What would establish it

A dose ladder on this probe — the α and k lines already exist, and this bank is re-runnable at
zero build cost. Uniform weakening predicts monotone decline in D with delta magnitude, at a
constant damage-rate ratio across all three salience conditions. That is now the cheapest
remaining test, and unlike everything above it would move the account from leading to measured.

The peer's LWF-001 preservation anchor tests the converse and my prediction is registered there
pre-run: an anchor toward the frozen base should help most on skills absent from the training
mixture. Alias binding is now a measured, thrice-replicated instance of exactly one.

---

# AMENDMENT 2 — the dose ladder: a perfect monotone response, and what it does not buy

**ALIAS-002.** Predictions registered in `probe/ALIAS002_DOSE_PREREG.md` and committed
(`6ef39c6`) **before the first arm was materialised**. Scoring script written while the arms were
still generating. Receipt `alias001-20260918/ALIAS_DOSE_RESULT.json`.

Five doses of `theta(alpha) = theta_supa + alpha * (theta_trained - theta_supa)`, FP32, built by
INT-001's own hash-bound `build_arm.py` on its registered grid. Each arm materialised, scored on
the **same 360 rendered prompts**, and deleted. Same tokenizer bytes across arms, so the render
is reused rather than recomputed.

| alpha | pooled acc | `low` | `equal` | `high` |
|---:|---:|---|---|---|
| 0.00 | **0.750** | 0.633 | 0.733 | 0.883 |
| 0.25 | 0.725 | 0.600 (4 dmg) | 0.717 (2) | 0.858 (3) |
| 0.50 | 0.672 | 0.575 (7) | 0.667 (8) | 0.775 (13) |
| 0.75 | 0.633 | 0.550 (10) | 0.633 (13) | 0.717 (20) |
| 1.00 | **0.603** | 0.542 (11) | 0.583 (19) | 0.683 (25) |

**Spearman(alpha, pooled accuracy) = −1.0000, exact permutation p = 0.0167** — which is the
*floor* attainable at n = 5, where the null admits only 120 orderings. Monotone non-increasing
across every rung. **And rho = −1.000 separately in all three salience conditions**, each at the
same exact p. Damage rates conditioned on the alpha=0 correct set rise monotonically too:
0.05 → 0.09 → 0.13 → 0.14 in `low`, 0.02 → 0.09 → 0.15 → 0.22 in `equal`, 0.03 → 0.12 → 0.19 →
0.24 in `high`.

**No dose × salience interaction**, as capacity predicted. Drops from alpha=0 to alpha=1 are
0.092 / 0.150 / 0.200 — spread 0.108, inside the registered "roughly parallel" bound of 0.15 —
and the interaction test, damage at the top dose in `low` against `high` conditioned on the
alpha=0 correct set, gives **Fisher p = 0.137**. Three near-parallel declines.

Every registered disconfirmer failed to appear: not non-monotone, not flat, no significant
interaction.

## A cross-parameterisation check that fell out for free

The promoted **k = 0.25 core scores 234/360 = 0.650 on this bank, landing between alpha = 0.50
and alpha = 0.75.** Masking the smallest three quarters of the update's elements preserves most
of its norm, where scaling every element to a quarter does not — so the two "0.25"s are very
different doses, and the masked one behaves like alpha ≈ 0.6 here. That is an independent
confirmation, on a task neither was designed around, of the programme's finding that mask, rank
and scalar are **one lever acting through magnitude**.

## What this establishes, held to what I registered

**"The delta uniformly weakens an unexercised binding skill in proportion to its magnitude" is
now MEASURED rather than described.** A perfect monotone dose response, replicated across three
salience conditions, with no interaction.

**It does not establish capacity as the cause, and I registered that in advance.** Magnitude is
already known to be the single operative lever in this programme; *any* mechanism scaling with
magnitude predicts exactly this curve. Monotone decline is **necessary for the capacity account
and not sufficient for it**. The corpus null — zero role phrases and zero person names in the
384-row training file — remains the only positive evidence for capacity, and absence in the data
does not establish causation.

What would settle it is a **contrast**: a second skill that the training data *does* exercise,
measured on the same ladder. Capacity predicts the exercised skill declines more slowly, or not
at all, at the same doses. That is the discriminating test and it is the natural next build,
since the ladder arms are now a known quantity and the acquisition instrument already measures
an exercised skill.

## The other thing worth taking from this

This is the fourth mechanism question asked of this one cell. The first three were answered by
looking at data already in hand, and two of them produced accounts that died. This one was
answered by **varying the thing** and registering the prediction first — and it is the only one
of the four that produced a number I would put in front of someone.


---

# AMENDMENT 3 — the damage tracks MAGNITUDE, not acquisition

**Same day, zero model calls. A LOOK, and labelled one throughout.** Receipt:
`analysis/coupling_ladder.py` → `alias001-20260918/COUPLING_LADDER.json`.

Amendment 2 established a monotone dose response and registered, at the time, that **monotone
decline is necessary for the capacity account and not sufficient for it** — because any
mechanism scaling with magnitude predicts the same curve. This is the question that *does*
separate them, and it turned out to be answerable from data already on disk.

INT-001's acquisition ladder and ALIAS-002's damage ladder sit on the **same five alphas** and
the **same build recipe**. Placing them side by side:

| α | acquisition /384 | % of total gain | alias acc | % of total damage | retained /47 | arith /150 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 14 | 0.0% | 0.750 | 0.0% | 47 | 47 |
| 0.25 | 145 | 59.3% | 0.725 | 17.0% | 43 | 59 |
| 0.50 | 219 | 92.8% | 0.672 | 52.8% | 42 | 67 |
| 0.75 | 234 | **99.6%** | 0.633 | 79.2% | 39 | 68 |
| 1.00 | 235 | 100.0% | 0.603 | 100.0% | 40 | 71 |

## The statistic, and it is not a correlation

Rank correlation cannot answer this: acquisition rises monotonically and alias falls
monotonically, so ρ = −1.000 between them *by construction*. The question is about **shape**,
so the statistic is **marginal damage per acquisition row gained** on each rung:

| rung | acq gain | alias drop | **pp per acq row** | repair | damage | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| 0 → 0.25 | +131 | 2.5pp | 0.019 | 0 | 9 | 0.0039 |
| 0.25 → 0.50 | +74 | 5.3pp | 0.071 | 0 | 19 | <0.0001 |
| 0.50 → 0.75 | +15 | 3.9pp | 0.259 | 1 | 15 | 0.0005 |
| **0.75 → 1.00** | **+1** | **3.1pp** | **3.056** | 1 | **12** | **0.0034** |

**Marginal cost per acquisition row spans 160× across the ladder.**

Strong capacity — *the damage is the price paid for the new skill* — predicts this ratio is
roughly constant: each row of new skill costs about the same amount of old skill. It is not.
**Damage per unit α is roughly flat (2.5 / 5.3 / 3.9 / 3.1); acquisition per unit α collapses
(131 / 74 / 15 / 1).**

The top rung is the cleanest cell. It buys **one row of 384** — 0.5% of the total acquisition
gain — and costs **20.8% of all the alias damage**, at **1 repair / 12 damages, p = 0.0034**.
Past α ≈ 0.75 the update is paying preservation for nothing.

Per condition on that rung: `low` 66→65 (p = 1.000), `equal` 76→70 (p = 0.031), `high` 86→82
(p = 0.219). Pooling is the registered primary; only `equal` is individually significant and the
per-condition n is small.

## Stated as a positive, which is how the peer session put it and they were right

This is **an independent argument for the α ≈ 0.5–0.75 operating point**, reached from the
damage side where INT-001 reached it from the acquisition side. The two arguments are
independent: INT-001 selected α = 0.50 on `retained`/`arithmetic`, and this arrives at the same
region from a preservation endpoint with 360 rows rather than 47. **The deployed artifact is
α = 1.**

## What this is not

1. **It is a LOOK.** The hypothesis was generated by placing two existing tables side by side.
   Nothing here is confirmatory. The registered TEST is `probe/LWF_ALIAS_PREREG.md` Addendum 1,
   which uses arms from a different lane where acquisition varies at roughly fixed recipe.
2. **The two curves come from different software stacks** — acquisition transformers 4.51.3,
   alias 5.14.1. Each curve is internally consistent, so the SHAPE comparison holds; **no
   cross-curve level comparison does**, and none is made.
3. **Comparing an acquisition endpoint's rise against a preservation endpoint's fall is not a
   like-for-like comparison of levels.** The claim is about two trajectories over one shared
   x-axis, measured on the same hash-bound arms. `arithmetic_permissive` *rises* across the
   ladder (47 → 71), so it is a partially-exercised endpoint and not a damage curve at all; it
   is carried above for completeness. `references_retained` is n = 47 and its last-rung +1 is
   inside its own noise.
4. **It does not kill the capacity account.** What it embarrasses is the *coupling* form —
   "damage is the price of acquisition." A **coverage** form, in which the training corpus
   determines *which* skills are exposed while magnitude determines *how much* they lose, is
   untouched and remains the leading synthesis.

---

# AMENDMENT 4 — direction matters at fixed magnitude, and that qualifies Amendment 3

**LWF-ALIAS, a joint experiment with the peer session.** Their arms, our instrument; neither
side could run it alone, and Gate v2 has no binding endpoint. Prereg `probe/LWF_ALIAS_PREREG.md`
plus two addenda, all committed **before any prediction file was read**. Scoring
`probe/score_lwf_alias.py`, receipt `lwf001-20260918/LWF_ALIAS_RESULT.json`. MECHANISM SCOUT.

The harness was reconstructed for this run (`probe/generate_alias.py`) and **validated by
reproducing the frozen source's existing 64-row chunk byte-for-byte, 64/64**, so all five arms
are scored under demonstrably matched conditions on the identical 360 rendered prompts.

## The arms, and the fact that reframed the design before scoring

| arm | λ | ‖δ‖ | cos to λ=0 | alias /360 | repairs | damages | p vs Supa | acquisition /384 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Supa | — | 0 | — | **0.750** | 0 | 0 | — | 14 |
| own-α0.5 | — | 0.4161 | **1.000000** | 0.658 | 1 | 34 | <1e-4 | **208** |
| λ = 0 | 0.00 | 0.8323 | 1.000 | **0.553** | 1 | 72 | <1e-4 | 198 |
| λ = 0.25 | 0.25 | 0.7427 | 0.5243 | 0.631 | 1 | 44 | <1e-4 | **197** |
| λ = 0.5 | 0.50 | 0.6800 | 0.3486 | **0.681** | 7 | 32 | 1e-4 | 125 |

**The anchor rotates the update rather than shrinking it** — cosine 1.000 → 0.524 → 0.349 while
the norm falls only to 89% and 82%. The scalar control is exactly collinear (cosine 1.000000,
ratio 0.500000), which is also the reader's self-check, and it passed exactly. The peer session
independently reproduced both cosines to six decimals.

## The result

Residuals against the **within-lane** magnitude ray (Supa → own-α0.5 → λ=0, exactly collinear):

| arm | observed | predicted from ‖δ‖ | residual | vs λ=0 | p |
|---|---:|---:|---:|---:|---:|
| λ = 0.25 | 0.631 | 0.576 | **+5.5 pp** | 34 gains / 6 losses | <1e-4 |
| λ = 0.5 | 0.681 | 0.591 | **+8.9 pp** | 55 gains / 9 losses | <1e-4 |

**Registered verdict: ANCHOR_PROTECTS.** Both residuals positive, increasing in λ, far outside
the registered ±2.5 pp band, and the larger significant against its norm-nearest ladder arm.

**The discriminating cell behaved as only one account predicted.** Addendum 1 registered that
ANCHOR PROTECTS and STRONG CAPACITY both expect a positive residual at λ = 0.5, so λ = 0.25 is
what separates them: it acquires **197 against λ = 0's 198**, so capacity predicts **no**
residual there. It returned **+5.5 pp at p < 1e-4**. **Strong capacity is disconfirmed on its
own discriminating cell** — a registered test on a different lane, independent of Amendment 3's
LOOK, and agreeing with it.

### λ = 0.25 stated plainly

Against λ = 0 it costs **one acquisition row**, carries **89% of the norm**, and gains
**+7.8 pp of alias binding**, of which **+5.5 pp is not explained by magnitude**. Same skill
acquired, different direction, materially less collateral damage. **This is the first result in
this programme that moves the frontier by choosing DIRECTION rather than DOSE.**

## How this qualifies Amendment 3

Amendment 3 concluded that damage tracks magnitude rather than acquisition. That was measured
**along a single ray** — the α ladder scales one fixed δ. This amendment measures **across
directions** and finds magnitude is **not** sufficient there: at ‖δ‖ = 0.68 the anchored arm
reaches 0.681 where the collinear ray predicts 0.591.

**Both hold, and together they are sharper than either:**

> Along a fixed direction, damage scales with magnitude and is indifferent to how much was
> acquired. Across directions at fixed magnitude, damage varies substantially. **So damage is a
> function of direction and magnitude, and not of acquisition.**

The practical reading: you cannot buy back an unexercised skill by acquiring less, but you may
be able to buy it back by moving somewhere else at the same distance.

## The two things that cut against over-reading it

**1. The anchor is still dominated, and this is the actionable line.** own-α0.5 vs λ=0.5 on
alias: **0.658 vs 0.681, 16 gains / 24 losses, p = 0.268 — not significant.** A plain scalar at
half the norm reaches statistically the same alias accuracy while keeping **208 acquisition rows
against 125**. The registered disconfirmer ("own-α ≥ λ=0.5 on alias") did not strictly fire,
since own-α is 2.3 pp behind — but a tie at p = 0.27 is not the anchor's last axis holding, it
is the anchor failing to buy what a scalar gives away free. **LWF-001's falsification stands,
and this does not rehabilitate it.**

**2. The direction confound, registered in Addendum 2 before scoring and held to here.** The
anchored arms are **off** the calibration ray. A positive residual is consistent both with *the
anchor protecting the skill* and with *that direction simply damaging alias less*, and **these
arms cannot separate those two.** No claim is made that the anchor mechanism is responsible.

The conclusion survives either reading, and it is the one worth keeping: **at fixed magnitude,
direction determines how much of an unexercised skill is lost.** Whether an anchor loss is a
good way to *find* such a direction is a separate and much weaker question than whether such
directions exist. They exist.

## What would settle the remaining question

A **direction control**: a delta rotated ~58° from λ = 0 by something other than an anchor, at
matched norm, acquiring ~197. If it also shows a positive alias residual, the effect is
direction and the anchor is incidental. If only anchored directions show it, the anchor
mechanism is real. One training run, in the peer's lane.

**One avenue the peer has already closed, with numbers.** The obvious cheap proxy — that
equally-acquiring T directions might differ in *alignment to a second skill* — does not hold:
all four T directions sit at |cos| ≤ 0.0015 to δ_H, including two that are 58° apart from each
other. Alignment is evidently not the axis that governs interference, since interference was
severe at that alignment. The direct test (two composition builds) remains open; the cheap proxy
does not survive.

**Scope.** One training draw per arm, one seed, three λ points. Absolute rates in the `high`
salience condition remain unreadable by construction; the paired contrasts are readable in every
condition. `quality_improvement_established` false, `release_permitted` false.

---

# AMENDMENT 5 — DIRCTL-001: the residual survives, and got larger

**Prereg `probe/DIRCTL001_PREREG.md`, committed before either arm was built.** Receipt
`lwf001-20260918/DIRCTL_RESULT.json`. MECHANISM SCOUT.

Amendment 4's residual was measured against a **piecewise-linear interpolation** across a wide
unmeasured gap. DIRCTL-001 replaced the interpolated value at each anchored arm's norm with a
**real arm at exactly that norm on exactly that direction** — built by the peer's own
`scale_own_delta.py`, verified at **cos = 1.000000** to λ=0 and norms **0.742705 / 0.680032**,
matching the anchored arms to all printed digits.

| pair | anchored | collinear twin | **residual** | anchored gains / losses | p (Holm, family of 2) |
|---|---:|---:|---:|---:|---:|
| λ = 0.25 vs sc-m025 | 0.631 | **0.558** | **+7.2 pp** | 32 / 6 | **2.43e-05** |
| λ = 0.5 vs sc-m05 | 0.681 | **0.586** | **+9.4 pp** | 43 / 9 | **4.08e-06** |

**Registered verdict: DIRECTION_SURVIVES.** No interpolation appears anywhere in those numbers.

**The interpolation had been optimistic, so Amendment 4 understated the effect.** It assumed
0.576 at ‖δ‖ = 0.7427 where the measured value is 0.558 (−1.8 pp), and 0.591 at 0.6800 where the
measured value is 0.586 (−0.5 pp). The residuals move **+5.5 → +7.2 pp** and **+9.0 → +9.4 pp**.

## Two free results from the same arms

**Curvature: APPROXIMATELY_LINEAR.** Deviations from the endpoint chord are +0.7 / −0.3 / −1.6 pp
across the five collinear points. This is the first direct read of the magnitude curve's shape in
this lane and it puts a small error bar on any residual computed here.

**ALIAS-002's dose response REPLICATES on an independently trained delta's ray.** Five collinear
points at ‖δ‖ = 0.000 / 0.416 / 0.680 / 0.743 / 0.832 give accuracies 0.750 / 0.658 / 0.586 /
0.558 / 0.553 — **ρ = −1.0000, exact permutation p = 0.0167**, the floor at n = 5. The monotone
magnitude→damage relation is not a property of the RRP-015 delta; it holds on the LWF lane too.

## What this does NOT establish, and the peer session has made that sharper

**Both off-ray arms here were produced by the same anchor loss.** So the supported claim is
narrower than "direction beats dose": it is that *this* anchor buys alias preservation its
magnitude reduction does not explain.

The peer session has since built the control Addendum 2 named — `arm-lambda0-ord23`, rotated
from λ=0 by **row order alone**, no anchor, same rows, same objective — and measured what that
rotation costs on **acquisition**:

| arm | rotation | ‖δ‖ ratio | acquisition |
|---|---:|---:|---:|
| λ = 0 | 0° | 1.000 | 198 |
| **ord23** (row order) | **53.6°** | 1.02 | **170** |
| λ = 0.25 (anchor) | **58.4°** | 0.89 | **197** |

**A larger rotation costs one acquisition row; a smaller one costs twenty-eight.** So rotation
cost is **mechanism-dependent, not angle-dependent** — measured, not argued. If that holds for
cost, it may hold for benefit, and my headline assumes it does not.

**DIRCTL-002 is registered** (`probe/LWF_ALIAS_PREREG.md` Addendum 3) and runs `ord23` against
its own norm-matched collinear twin. If `ord23` shows no residual, Amendment 4's claim narrows
in the record from "direction beats dose" to "an anchor loss buys preservation beyond its
magnitude" — a substantially smaller claim, and the one the evidence would then support.

---

# AMENDMENT 6 — three mechanisms, one effect: direction is a real axis of the frontier

**DIRCTL-002 and ROTATION-002, both registered before their arms existed.** Receipts
`lwf001-20260918/DIRCTL2_RESULT.json` and `ROTATION2_RESULT.json`. MECHANISM SCOUTS.

Amendment 5 left one gap: both off-ray arms came from the **same anchor loss**, so the supported
claim was "this anchor spares alias", not "direction spares alias". Two arms close it.

## DIRCTL-002 — a non-anchor rotation spares alias too

`arm-lambda0-ord23` is the peer session's control, and its `TRAIN.json` was verified here rather
than taken on report: **`lambda = 0.0`** (no anchor), `seed = 17`, `update_groups = 28`,
`source_weight_sha256 = 962abc5b…` — identical to λ=0 in every field except **`order_seed = 23`**.
The only difference is the order the rows were grouped in.

Measured here: ‖δ‖ = **0.849139**, cos to λ=0 = **0.593855** (53.6° off the ray at 1.02× norm).
Its collinear twin was built at `a = 1.020245875` — an **extrapolation** 2% past λ=0, flagged as
such — and landed at ‖δ‖ = 0.849139, err **2.10e-08**.

| | accuracy | residual | gains / losses | p |
|---|---:|---:|---:|---:|
| `ord23` | **0.5861** | **+4.4 pp** | 21 / 5 | **2.49e-03** |
| collinear twin (same norm) | 0.5417 | — | — | — |

**Registered verdict: GEOMETRY.** A rotation produced by nothing but row order spares alias
binding beyond what its magnitude explains.

## ROTATION-002 — and so does a magnitude mask, in the other lane

`k = 0.25` and a scalar at `α = 0.836382` have **identical norm** (target 0.694055, achieved
0.694055, err 7.06e-09) and sit **33.2° apart** — ROTATION-001 showed a mask is a projection, so
its cosine to the ray is exactly its norm ratio.

| | accuracy | difference | gains / losses | p |
|---|---:|---:|---:|---:|
| `k = 0.25` mask | **0.6500** | **+2.8 pp** | 11 / 1 | **0.0063** |
| `α = 0.836382` scalar | 0.6222 | — | — | — |

**The α-ladder interpolation was accurate**: it assumed 0.6228 and the measured value is 0.6222,
off by **0.1 pp**. So ROTATION-001's +2.7 pp figure was reliable, and no other residual in this
programme computed against that ladder needs re-deriving. That was the disconfirmer flagged as
most consequential, and it did not fire.

## Together

| arm | lane | mechanism | angle | norm ratio | **residual** | p |
|---|---|---|---:|---:|---:|---:|
| `k = 0.25` | RRP-015 | magnitude mask, post-hoc | 33.2° | 0.836 | **+2.8 pp** | 0.0063 |
| `ord23` | LWF | row order, training-time | 53.6° | 1.020 | **+4.4 pp** | 2.5e-03 |
| λ = 0.25 | LWF | anchor loss, training-time | 58.4° | 0.892 | **+7.2 pp** | 2.4e-05 |
| λ = 0.5 | LWF | anchor loss, training-time | 69.6° | 0.817 | **+9.4 pp** | 4.1e-06 |

**Three independent mechanisms, two lanes, four arms, every residual positive and significant.**
One is a post-hoc edit of a finished delta, one is a training-time reordering, one is a
training-time loss term. Nothing they share explains this except that each moves the update off
the collinear ray at fixed distance.

> **At fixed magnitude, direction determines how much of an unexercised skill is lost.**

**No angle→residual ordering is claimed**, and the numbers above are deliberately not read as a
trend. ROTATION-001 showed a mask's angle carries no information its norm does not, and the peer
session measured that rotation *cost* is mechanism-dependent — `ord23`'s 53.6° costs **28
acquisition rows** where the anchor's 58.4° costs **one**. Angle is not the operative variable;
that it correlates here across mechanisms is not evidence that it is.

## What this does to the programme's standing conclusion

`delta-compression-is-magnitude` holds that mask, rank truncation and scalar scaling are **one
lever acting through magnitude**. ROTATION-002 says that is **true for acquisition and false for
preservation**, and the reason it was not caught is power: the preservation endpoint it was
established on is n = 47, where 2.8 pp is one row. This bank is n = 360, where it is ten.

**The practical form, stated as a trade rather than as dominance.** At matched norm the mask
gives **+2.8 pp of alias binding** for about **2 acquisition rows** (232 against an interpolated
~234 for the scalar — and that acquisition figure is interpolated from INT-001's ladder, not
measured here). Favourable, not free, and worth measuring directly before anyone acts on it.

## Scope

One bank, one delta per lane, one draw per arm. `ord23` is **not acquisition-matched** (170
against λ=0.25's 197) — tolerable only because damage was already established indifferent to
acquisition, and stated rather than waved past. The `ord23` twin is an extrapolation, not an
interpolant. `quality_improvement_established` false, `release_permitted` false.

---

# AMENDMENT 7 — the claim narrows to ONE skill, and my justification for ord23 is replaced

**Amendment 6 called direction "a real axis of the frontier." That is too broad and is narrowed
here.** The peer session ran the second preservation instrument I said was the biggest structural
weakness in the joint evidence, and it came back without sparing. Their numbers are taken at face
value — their eval receipts do not carry the summary fields, so I could not reconstruct them, and
that is stated rather than implied. Everything they have reported this session that I *could*
check has been exact.

| arm | angle | alias residual (this bank) | references retained /47 | arithmetic /150 |
|---|---:|---:|---:|---:|
| λ = 0 | 0° | — | 36 | 63 |
| λ = 0.25 | 58.4° | **+7.2 pp** | **36** | **64** |
| ord23 | 53.6° | **+4.4 pp** | **34** | **59** |

**Neither rotated arm spares anything on either endpoint.** The narrowing is accepted.

## But one of their two endpoints cannot speak to this, and I verified why independently

The peer argued arithmetic is decisive on power: n = 150, so a +7.2 pp effect would be 10.8 rows,
and only +1 was observed. **That argument does not hold, because arithmetic is not a preservation
endpoint in this programme.** From INT-001's own frontier:

| α | 0.00 (Supa) | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|
| arithmetic /150 | **47** | 59 | 67 | 68 | **71** |

**Arithmetic RISES with dose.** It is a partially *acquired* skill, not a damaged one — and the
LWF lane agrees, with λ=0 at 63 against the source's 47, **+16 above the source**. Nothing is
being forgotten there, so there is nothing for direction to spare, and a null on it is close to
uninformative regardless of its n. Power is not the binding consideration; **endpoint class is.**

## The endpoint that *is* the right class is genuinely uninformative, not negative

`references_retained` is a real preservation endpoint — Supa scores 47, λ=0 scores 36, so it is
**damaged by 11 rows**. λ=0.25 shows 36: no sparing. But scale the alias effect proportionally
before reading that as a negative:

  * on alias, λ=0 loses 71 of Supa's 270 correct rows, and λ=0.25 recovers **28** of them — a
    **39%** recovery of the damage;
  * the same fractional recovery on references would be **0.39 × 11 ≈ 4 rows**;
  * an exact McNemar on 4 discordant pairs bottoms out at **p = 0.125**. It could not have
    reached significance even if it were entirely real.

**So the properly-classed second endpoint is underpowered by a factor of roughly three, and its
null is weak evidence rather than a refutation.** Observing 0 where ~4 were expected is mildly
suggestive of no transfer and nothing stronger.

## Where that leaves the claim

> **Direction sparing is demonstrated on exactly one skill — `alias_resolution` — across three
> independent mechanisms. It has not been shown on any second skill, and the one endpoint of the
> right class is too small to have shown it.**

That is weaker than Amendment 6 and it is what the evidence supports. **The blocking item is now
explicit and is nobody's side-quest: a properly powered preservation instrument on a second
skill the training corpus does not exercise.** Until that exists, "direction beats dose" is a
one-skill result.

## My justification for `ord23` is replaced with a better one

I registered that `ord23`'s acquisition mismatch (170 vs 197) was tolerable "because damage is
indifferent to acquisition." The peer is right that this **imports an on-ray finding (the α
ladder, collinear) into an off-ray comparison** — precisely the class of move DIRCTL exists to
remove. I flagged it as load-bearing and it was the weak point.

**The data defends the verdict better than my argument did, and the replacement is theirs:**

| arm | acquisition cost | alias spared |
|---|---:|---:|
| `ord23` | **−28 rows** | **+4.4 pp** |
| λ = 0.25 | **−1 row** | **+7.2 pp** |

If losing acquisition *caused* sparing, the arm that lost 28 rows would spare **more** than the
one that lost 1. It spares **less**. The confound is **anti-correlated** with the effect, so
"arms that acquire less also spare more" predicts the opposite ordering to the one observed.
**The GEOMETRY verdict stands on that, not on acquisition-indifference.**

## Two corrections to the record from the same exchange

**The mask's price is measured, not interpolated.** The peer built and scored the missing arm:
the scalar at α = 0.836382 reaches **237 acquisition** against the mask's **232**, paired
**p = 0.38**. So the mask buys **+2.8 pp of alias for 5 acquisition rows**, and the acquisition
difference is **not significant**. Amendment 6 estimated ~2 rows from an interpolation; the
measured figure is 5. (The scalar's 237 sits *above* the full delta's 235 — the α curve's own
non-monotonicity again.)

**On BF16, worth more than the infrastructure note I gave it.** Every delta in this programme is
`FP32_trained − BF16_source`, so part of what is called "the delta" is the source's own
quantisation residual. That does not invalidate anything — BF16 is what ships, so measuring
against it is correct — but it means **‖δ‖ carries a rounding floor that is identical across all
arms and therefore cancels in every paired comparison made here.** Nobody should later read a
small ‖δ‖ as a small update.

---

# AMENDMENT 8 — the reallocation framing is withdrawn, and my own stratum is verified clean

**REPORT-ONLY** (the 256-row suite is SPENT). Zero model calls. Script
`analysis/lc_endpoint_class.py`, receipt `LC_ENDPOINT_CLASS.json`.

Amendment 7's Addendum 2 registered **reallocation** — that rotation moves damage rather than
removing it — on the peer session's finding that λ=0.25 loses 17 long-context rows. **The peer
has since retracted that headline themselves, and their argument reproduces exactly here.**

## The stratification, recomputed independently

Each stratum classified by `Δ = λ=0 − Supa`, using **only the source and the unrotated delta** —
no rotated-arm data, so it stratifies rather than selects on outcome. Answerable rows only.

| stratum | n | Supa | λ=0 | Δ | class | λ=0.25 losses | ord23 losses |
|---|---:|---:|---:|---:|---|---:|---:|
| causal_disambiguation | 24 | 23 | 24 | +1 | PRESERVED | 4 | 0 |
| latest_verified_record | 24 | 24 | 23 | −1 | PRESERVED | 1 | 0 |
| multi_condition_identity | 24 | 18 | 18 | 0 | PRESERVED | 1 | 0 |
| object_state_tracking | 24 | 24 | 24 | 0 | PRESERVED | 0 | 0 |
| temporal_correction | 24 | 24 | 24 | 0 | PRESERVED | 1 | 2 |
| **count_with_exclusions** | 24 | 16 | 22 | **+6** | **IMPROVED** | **7** | **7** |
| **rule_exception** | 24 | 15 | 18 | **+3** | **IMPROVED** | 3 | 1 |
| alias_resolution | 24 | 4 | 1 | −3 | **BELOW_GATE** | 1 | 1 |

| arm | IMPROVED | PRESERVED | **rate ratio** |
|---|---:|---:|---:|
| λ = 0.25 | 10/40 = 0.250 | 7/113 = 0.062 | **4.0×** |
| ord23 | 8/40 = 0.200 | 2/113 = 0.018 | **11.3×** |

**Rotation damage sits 4–11× more heavily on strata the delta IMPROVED.** By the endpoint-class
rule both sessions now apply, damage on an improved stratum is **the delta's own gain being
given back** — an acquisition cost, not forgetting. The shared region found in
`LC_LOSS_REGION_FINDING.md` is `count_with_exclusions`, which is **entirely** in that class.

**So "rotation relocates forgetting" is withdrawn.** It was counting acquisition give-back as
preservation loss.

## My own stratum, checked rather than assumed

The obvious next worry is whether the alias result is the same artifact. It is not:

> **ALIAS-001: Supa 0.750 → λ=0 0.553, Δ = −0.197. DAMAGED.**

That is a genuine preservation endpoint — the delta costs it — so sparing it is sparing real
forgetting, not returning an unearned gain. **Verified, not assumed**, because this is exactly
the check the peer's retraction implies and it would have been convenient to skip.

## What the corrected arithmetic actually says — and it is NOT "gains exceed losses"

Restricting the long-context loss to PRESERVED strata changes the conservation sum, and my first
reading of that was wrong. **Row counts across instruments are not summable** — alias is n = 360,
long context n = 256, the process bank n = 384 — so the comparison must be in rates:

| arm | alias (preservation) | LC PRESERVED (preservation) | ratio |
|---|---:|---:|---:|
| λ = 0.25 | **+7.8 pp** | **−6.2 pp** | 0.79 |
| ord23 | **+3.3 pp** | **−1.8 pp** | 0.54 |

Both arms are net positive on preservation, **but the margin is far smaller than row counts
implied, and the two arms do not agree on a ratio.** So the conserved-ratio reading registered
over all 256 rows was using the wrong denominator — and on the right denominator the ratio is
neither 1.0 nor stable across arms. **Conservation is not rescued by the correction; it is
differently wrong.**

## The account that survives, stated no more strongly than it should be

> **Rotation appears to trade ACQUISITION for PRESERVATION** — it gives back the delta's own
> gains (the IMPROVED strata, and for `ord23` 28 process rows) and buys preservation on a
> genuinely damaged skill.

And the two mechanisms differ sharply in how good that trade is: λ=0.25 gives up **1** process
acquisition row, `ord23` gives up **28**, for alias gains of +7.8 and +3.3 pp. **The anchor is a
much better trade than a row reshuffle** — which is what one would expect from an intervention
designed to preserve versus an arbitrary reordering, and is the first thing in this programme
that distinguishes two rotations by anything other than their angle.

**Two instruments, one draw each, a SPENT suite, and a rate comparison across banks that differ
in difficulty.** Nothing here is confirmatory.

---

# AMENDMENT 9 — two denominators, both legitimate, and the one the conservation ratio mixed

Re-scored from the committed raw generations after the peer session reported that two of three
alias figures "reproduce under neither denominator". **They reproduce exactly. They are a
different contrast.**

| arm | acc /360 | **vs λ=0** | **vs its collinear twin** | twin | twin acc | norm ratio |
|---|---:|---:|---:|---|---:|---:|
| Supa | 0.7500 | — | — | — | — | — |
| λ=0 | 0.5528 | — | — | — | — | 1.000 |
| λ=0.25 | 0.6306 | +7.8 | **+7.2** | sc_m025 | 0.5583 | 0.892 |
| λ=0.5 | 0.6806 | +12.8 | **+9.4** | sc_m05 | 0.5861 | 0.817 |
| ord23 | 0.5861 | +3.3 | **+4.4** | d2twin | 0.5417 | 1.020 |

  * **vs λ=0** — the arm's *total* effect: direction **and** its norm change together.
  * **vs collinear twin** — the *direction* effect with **magnitude held fixed**. This is the
    quantity DIRCTL-001 and DIRCTL-002 were built to isolate, and it is uniformly what this
    programme's `+7.2 / +9.4 / +4.4` figures are.

The gap between the columns is the norm effect, and it carries the right sign throughout:
**+0.6, +3.3, −1.1**, negative exactly where the norm ratio crosses 1.0 (`ord23` is the only arm
*larger* than λ=0). That is the collinear ladder restated, so it is an internal consistency
check rather than new evidence — but it establishes that both columns are computed correctly.

## What was actually wrong, and it was not the figures

**The conservation ratio mixed contrasts.** It took the alias term as a **twin residual**
(+9.4 pp) and the long-context term as a **λ=0 difference** (−17/256). Dividing one by the other
compares two different things. The peer's conclusion — that the band is void on its arithmetic —
**stands**; the attribution does not, and the difference matters because "the +9.4 was
unreproducible" would retroactively discredit DIRCTL's entire construction, which reproduces to
the decimal.

## The consequence: the queued control is the missing half of the ratio

A well-posed conservation test needs both terms on one denominator:

| denominator | alias term | long-context term | ratio |
|---|---:|---:|---|
| **vs λ=0** | +12.8 pp | −6.6 pp | **1.93** — computable today, and not 1.0 |
| **vs collinear twin** | +9.4 pp | **does not exist** | — |

**No norm-matched twin has ever been scored on long context.** That is exactly the `sc_m025` run
already queued behind the lease. It is therefore not a side control: it is the only route to a
conservation ratio with a consistent denominator, and it must be read **per endpoint class**,
because the λ=0 version above is contaminated by the IMPROVED-stratum give-back established in
Amendment 8.

## Corrections taken from the same exchange

**λ=0.5's λ=0-referenced alias gain is +12.8 pp, not the +9.4 quoted in the joint document** —
+9.4 is its twin residual, and the conservation input needed the λ=0 figure. The anchor's
dose-response on preservation is therefore steeper than either session had, and **λ=0.5's
process acquisition cost becomes the highest-value unmeasured number here**: if it lands near
λ=0.25's one row while buying +12.8 pp, that is the best acquisition-versus-preservation trade
this programme has produced.

**`--timeout 2700` was withdrawn by the peer and was never implemented here.** The probe's 900 s
cap is part of its sha-pinned contract, so raising it would break the binding equality every arm
was scored under. Only the *lesson* was kept — a timeout sized against uncontended throughput
converts "slow" into "no result" — and the remedy is a quiet box, which is what the lease is for.

**One coincidence flagged before anyone reads it as a result:** `sc_m05` and `ord23` both score
**0.5861**. Different arms, different constructions, identical to four decimals at n = 360.

---

# AMENDMENT 10 — the replication arm landed, and it takes back most of Amendment 8

**λ=0.5 scored 146/256 on long context.** Computed independently here from the peer session's
evidence file (`analysis/lc_endpoint_class.py`, receipt `LC_ENDPOINT_CLASS_REPL.json`).
REPORT-ONLY — the suite is SPENT.

## Two registered tests, both resolved, both negative

**Conservation: FALSIFIED.** 146 falls in the registered **145–161** band — *loss no worse than
λ=0.25's, so damage saturates; conservation falsified, reallocation survives but is not
one-for-one*. It also misses the corrected ~129 band derived from the reproducible +12.8 pp
input. **Conservation dies on the registered arithmetic and on the corrected arithmetic both**,
which is as clean as a falsification gets. The peer predicted their own point prediction would
fail; it did.

**P3 — the endpoint-class concentration: FAILS REPLICATION.**

| arm | IMPROVED | PRESERVED | **rate ratio** | Fisher p | stage |
|---|---:|---:|---:|---:|---|
| λ = 0.25 | 10/40 = 0.250 | 7/113 = 0.062 | **4.04** | 0.0026 | discovery |
| ord23 | 8/40 = 0.200 | 2/113 = 0.018 | **11.30** | 0.0004 | discovery |
| **λ = 0.5** | **6/40 = 0.150** | **12/113 = 0.106** | **1.41** | **0.5679** | **REPLICATION** |

Registered threshold **≥ 2.0**. Observed **1.41 at p = 0.57**. Only the direction survives.
`count_with_exclusions` is damaged as registered but by 3 rows against the discovery arms' 7 and
7, so the shared-region signal weakens with it.

## What that does to Amendment 8, which is this document's own

Amendment 8 withdrew "reallocation" **on the strength of the 4.04× and 11.30×**. Those do not
replicate, and **λ=0.5 damages PRESERVED strata harder than either discovery arm** — 10.6%
against 6.2% and 1.8%.

  * The withdrawal of **reallocation** stands. It was never supported, and the replication arm
    does not revive it.
  * The **replacement** — "rotation trades acquisition for preservation" — is **weakened**. The
    acquisition-give-back reading is **arm-dependent, not a property of rotation**. Some rotated
    arms give back acquisition; at least one does genuine preservation damage.

**So neither account generalises, and Amendment 8 was over-stated in the direction of its own
conclusion one step after I congratulated the peer for withdrawing theirs.** Recorded plainly
rather than softened: two discovery arms agreed, the replication arm did not, and the honest
position is that the concentration is not established.

## One pattern flagged and deliberately NOT banked

Alias gain against PRESERVED long-context loss, both on the λ=0 denominator — the only
denominator that survives Amendment 9's correction:

| arm | alias | LC-preserved | loss/gain |
|---|---:|---:|---:|
| λ=0.25 (anchor) | +7.8 | −6.2 | **0.79** |
| λ=0.5 (anchor) | +12.8 | −10.6 | **0.83** |
| ord23 (reshuffle) | +3.3 | −1.8 | 0.54 |

**The two anchored arms agree at ≈0.8; the reshuffle does not.** This is three points across two
mechanisms — and it is exactly the "two points fitting a ratio" shape the peer registered
*against themselves* earlier today, which is why it is recorded as an observation with the
caveat attached and **not** proposed as a conserved quantity. Any test of it must be registered
before a fourth arm exists, not after.

## What still stands

The **direction result itself is untouched by this**: it rests on paired contrasts against
norm-matched collinear twins on the alias bank (Amendment 9's table, reproducing to the decimal),
not on the long-context stratification. What this amendment removes is the *explanation* of
what rotation costs elsewhere — which is now open again.

**Status: discovery complete, replication run, replication FAILED. Nothing promoted.**

---

# AMENDMENT 11 — the regression is about BINDING, not about persons

**PRES-005, a clean bank: no cue, zero doubled articles, every candidate mention count asserted,
all three floor classes cleared at 0.167–0.175.** Screen = frozen Supa against λ=0 on 360 rows.
Build `probe/build_pres4.py` → `pres005-20260919/`. MECHANISM SCOUT.

## The result

| stratum | Supa | λ=0 | **baseline-correct** | **damaged** | **rate** | **repairs** | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `person_binding` | 0.850 | 0.592 | 102 | **31** | **0.304** | **0** | 9.3e-10 |
| `object_binding` | 0.842 | 0.700 | 101 | **17** | **0.168** | **0** | 1.5e-05 |
| `place_binding` | 0.925 | 0.842 | 111 | 10 | 0.090 | **0** | 1.9e-03 |

Damage is scored over **the rows the source got right**, per the denominator rule.

> **The delta damages role→filler binding whatever the filler is — persons, objects and places
> — and does so PURELY SUBTRACTIVELY in every stratum: 0 repairs, 58 damages, three for three.**
>
> **`alias_resolution` was one instance of a general binding regression, not a fact about
> persons.**

## The person-vs-object contrast, at a matched baseline

`person_binding` and `object_binding` start at **102 and 101** correct of 120 — a one-row
difference — so the contrast needs no adjustment:

**31/102 = 0.304 against 17/101 = 0.168. Ratio 1.81×, Fisher exact p = 0.0312.**

**Persons are damaged about twice as heavily as objects**, at equal source competence, in one
bank, one construction, one run.

## It replicates across two independent builds

| stratum | PRES-004 (carried the `The the` cue) | PRES-005 (clean) |
|---|---|---|
| `person_binding` | 0 repairs / 29 damages, −24.1 pp | **0 / 31, −25.8 pp** |
| `object_binding` | 1 repair / 18 damages, −14.1 pp | **0 / 17, −14.2 pp** |

**Near-identical.** Which also settles the cue question empirically: a shared surface cue moved
the *absolute* level and left the *paired* contrast untouched, exactly as the immunity argument
predicts. `object_binding`'s single repair in PRES-004 becomes zero here, so the subtractive
signature is now unbroken across all three strata.

## What this does and does not license

**Does.** `alias_resolution`'s five-bank, purely-subtractive regression is **one face of a
general binding deficit**. Any account of it — capacity, coverage, or anything else — now has to
explain damage to object- and place-valued bindings too, at rates ordered person > object >
place.

**Does not.** It does not explain *why* persons are worse. The 1.81× is measured and
unexplained, and `place_binding` sits above ceiling (Supa 0.925) so its rate is floored by
headroom rather than comparable.

**And it says nothing about ALIAS-001's absolute level**, because PRES-005 is 20 points easier
than ALIAS-001 for reasons three tested hypotheses failed to explain. Every number above is
within-bank, which is why the difficulty gap does not touch it.

One draw, one delta, one lane. `quality_improvement_established` false.

---

# AMENDMENT 12 — the knee is at α = 0.75, and my own receipt said so

**The native ladder's long-context column existed all along**, in
`int001-20260917/evidence/`. Verified here: `a025` **163**, `a050` **164**, `a075` **163**,
`a100` **157**, against Supa's **161** — all at implementation digest `a329eb7bb06f` on `cuda`,
identical to every arm either session scored today, so all five are directly comparable.

## The completed native frontier

| α | acq /384 | alias /360 | refs /47 | arith /150 | long ctx /256 |
|---|---:|---:|---:|---:|---:|
| 0.00 | 14 | **270** | **47** | 47 | 161 |
| 0.25 | 145 | 261 | 43 | 59 | 163 |
| 0.50 | 219 | 242 | 42 | 67 | **164** |
| **0.75** | **234** | **228** | 39 | 68 | **163** |
| 1.00 | **235** | 217 | 40 | **71** | 157 |

The alias column reproduces ALIAS-002's dose ladder exactly (270/261/242/228/217).

## Read downward, the exchange rate collapses after one step

| step | acq given up | alias recovered | rate |
|---|---:|---:|---:|
| **1.00 → 0.75** | **1** | **11** | **10.98 : 1** |
| 0.75 → 0.50 | 15 | 14 | 0.93 : 1 |
| 0.50 → 0.25 | 74 | 19 | 0.26 : 1 |
| 0.25 → 0.00 | 131 | 9 | 0.07 : 1 |

> **α = 0.75 gives up ONE acquisition row and recovers ELEVEN alias rows and SIX long-context
> rows. The full delta is dominated on both preservation axes at essentially no acquisition
> cost.**

Alias term paired from the raw generations: **12 gains / 1 loss, exact McNemar p = 0.0034.**

## The part I have to own: this is in my own committed receipt

`COUPLING_LADDER.json`, written hours earlier, records the 0.75 → 1.00 rung as
**+1 acquisition row for 1 repair / 12 damages at p = 0.0034**, and a **160× spread** in
marginal cost across the ladder. Read in the operating-point direction that *is* the knee.

**What I wrote at the time was "an independent argument for the α 0.5–0.75 operating point."**
I hedged across a range when the number identified a point, and then anchored on 0.50 because
INT-001 had selected it — on a rule that never saw long context. **The 160× was the signal and
I described it correctly without converting it into the recommendation it implied.**

The peer session searched one directory and declared a number missing, three separate times
today. I computed the decisive number and under-read it. Different failure, same cost, and mine
was the one already committed to this repository.

## What it licenses, and what it does not

**Does.** On the RRP-015 lane, α = 0.75 dominates the deployed artifact (α = 1) on alias and
long context for about one acquisition row. That is the most deployment-relevant number this
programme has produced.

**Does not.** One lane. The arm-versus-rule distinction stands: this is *"the best measured
operating point among the arms built here"*, not *"keep three quarters."* The LWF lane's ladder
has a single α = 0.5 point — which this now shows is **past the knee** — so it has never been
measured where it matters. **LWF α = 0.75 is the cheap discriminating follow-up**, and unlike
everything else today it does not already exist.
