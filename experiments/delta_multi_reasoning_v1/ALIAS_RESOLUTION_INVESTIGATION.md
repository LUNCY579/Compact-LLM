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
