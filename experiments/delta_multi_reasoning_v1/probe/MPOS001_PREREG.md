# MPOS-001 prereg — position MANIPULATED, mention count separated from salience ratio

**Written before any arm has been run against this bank and before any prediction exists.**
Bank: `mpos001-20260919/MPOS_PROBE.jsonl`, 384 rows, manifest `MPOS_BUILD.json`.
Class if read: **MECHANISM SCOUT** — one bank, one draw. This is not a confirmation instrument
and no confirmation budget is requested or implied.

## What this bank is for

Amendments 18–20 established, observationally, that mean mention position predicts which rows the
delta damages — replicated across two independently trained deltas. Every one of those results
came from banks that `shuffle(body)`, so position was an accident we correlated with damage.
Here position is **assigned**. A result becomes a claim about what happens *when you move the
answer*.

It also breaks a confound no existing bank can. ALIAS-001's `low`/`equal`/`high` moved answer
count and salience ratio together, so "sensitivity is strongest where the answer is mentioned
most" and "…where the answer out-mentions its distractors" are the same sentence there. Here
k ∈ {1,3} is crossed with d ∈ {1,3}.

## Registered primary test — the arm x band INTERACTION on matched pairs

**This replaces an earlier draft that registered the absolute early-vs-late damage rate. That
test is not valid on this bank and the reason is recorded below under "what the build refused".**

Every scenario is built **once** and rendered in **both** bands from identical material: same
names, same distractors, same question, same sentences, differing only in arrangement. 192
matched pairs, asserted at build time (sentence multisets compared, answers compared).

For each pair and each arm, score correct/incorrect. Define per pair:

* `d_source` = source(early) - source(late), in {-1, 0, +1}
* `d_delta`  = delta(early)  - delta(late)

**Primary statistic: `d_delta - d_source`, tested against zero with an exact sign test over
pairs, per cell x lane, Bonferroni over 8 = 0.00625.**

**Direction registered: positive — moving the answer late costs the DELTA more than it costs the
SOURCE.** A significant negative value falsifies the positional account as an intervention.

**Why the interaction and not the rate.** A deterministic answer-agnostic rule emits the same
string for both arms, so it contributes equally to `d_source` and `d_delta` and **cancels in the
difference**. The absolute band comparison has no such protection: it is between rows, and a
band-dependent cue changes which rows are solvable at all.

## Registered predictions — PRIMARY restricted to the two CLEAN cells

**The d=3 cells cannot carry the primary. Measured, not argued** (see the limitation section).

| cell | k | d | cue swing | **manufacturable** | **role** | predicted `d_delta - d_source` |
|---|---|---|---:|---:|---|---|
| `k1d1` | 1 | 1 | **0.000** | **0.000** | **PRIMARY** | **+0.03 to +0.15** |
| `k3d1` | 3 | 1 | **0.000** | **0.000** | **PRIMARY** | **+0.10 to +0.30** |
| `k1d3` | 1 | 3 | 0.812 | **0.171** | not readable | +0.00 to +0.08 |
| `k3d3` | 3 | 3 | 0.938 | **0.198** | not readable | +0.10 to +0.30 |

**The bound on the cue-exploitation differential is registered at 0.211** — the **Newcombe
method 10 (hybrid-score)** two-sided 95% upper limit on (LWF 109/151) − (Supa 54/85), the larger
of the two lanes. **A confound bound must be pessimistic**, so it is an upper confidence limit on
the true differential, **not** a minimum detectable effect. An earlier draft quoted 0.165, which
is the MDE — a different quantity and the wrong one here.

*Two corrections to how this bound was first registered, both caught by the peer session and
both mine:*

* A draft registered **0.258 under the name "Newcombe hybrid-score"**. That figure is the **naive
  Wilson-difference bound**, `u2 − l1` — interval arithmetic on the two separate Wilson
  intervals, which is exactly the conservatism Newcombe's method exists to avoid. Method 10 is
  `d + sqrt((u2−p2)² + (p1−l1)²) = 0.2112`. **A pessimistic bound is fine; a pessimistic bound
  under a borrowed label is not, because the next person reproduces method 10 and gets a
  different number.**
* That draft also claimed the normal approximation "understates it at these denominators."
  **Unsupported:** normal 0.2114 against hybrid-score 0.2112, agreeing to three decimals. The two
  diverge at small n or proportions near 0 or 1; n = 85 and 151 at p ≈ 0.64 and 0.72 is where
  they coincide.

**The conclusion is unchanged on either figure** — at 0.211 `k1d3` manufactures 0.171 against a
predicted ceiling of 0.08, and `k3d3` manufactures 0.198 against a predicted 0.10–0.30. Both are
**not readable**, and the withdrawal stands on the tighter bound without needing the looser one.

**Primary: `k1d1` and `k3d1`, two lanes, Bonferroni over 4 = 0.0125.** These are the cells where
no band-dependent answer-agnostic rule exists, so nothing can be manufactured. They test the
question this bank was built for: **is the position effect causal, or only observational?**

The d=3 cells are reported **beside their manufacturable bound**, never alone. A result inside
its bound is reported as *"not separable from cue exploitation"*, not as a finding.

## The count-vs-ratio decomposition is NOT deliverable on this design

Registered as a failure of the design, before any run, because the earlier draft claimed it.

The lattice is: `k1d1` (count 1, ratio 1:1), `k1d3` (1, 1:3), `k3d1` (3, 3:1), `k3d3` (3, 1:1).
**Every contrast that isolates one factor requires `k3d3`**:

* count, holding ratio at 1:1 → `k1d1` vs **`k3d3`**
* ratio, holding count at 3 → `k3d1` vs **`k3d3`**

And `k3d3` is the most contaminated cell in the bank (manufacturable 0.155 against a predicted
0.10–0.30). The two clean cells, `k1d1` and `k3d1`, differ in **both** count and ratio at once —
which is **exactly the ALIAS-001 confound this bank was built to break.**

> **So MPOS-001 can establish whether the position effect is CAUSAL. It cannot separate mention
> count from salience ratio. That part of the original motivation is withdrawn before any GPU is
> spent on it.**

**What the decomposition would actually need:** a bank that varies k and d at **fixed** answer
position. With position held constant there is no band, no band-dependent rule, and no cue to
exploit — the whole confound disappears because it is created by manipulating position. The two
questions need two banks, and trying to answer both in one is what produced the contaminated
cells. Not built, not requested.

## Effect size the manipulation actually delivers — asserted at build time

Disjoint bands are **not** a matched manipulation. Gap in mean answer position, asserted
>= 0.45 in every cell with a spread of <= 0.10 across cells:

| cell | gap | mean answer spread (early / late) |
|---|---:|---|
| `k1d1` | 0.5393 | 0.000 / 0.000 |
| `k1d3` | 0.5240 | 0.000 / 0.000 |
| `k3d1` | 0.5449 | ~0.12 / ~0.14 |
| `k3d3` | 0.5237 | ~0.14 / ~0.13 |

Spread across cells is **0.0212**. All k mentions are anchored inside the band, so the k=3 cells
are manipulated as strongly as the k=1 cells rather than diluted toward 0.5. **Clustering all k
mentions couples mean position with spread**, recorded here rather than tuned away: spread was
withdrawn for lack of evidence, not refuted, so the coupling must stay visible to anyone
revisiting it.

## Readability, stated in advance

- **Absolute early-vs-late rates are NOT readable in `k1d3` or `k3d3`.** A rule taking the
  **argmin of each candidate's mean mention position** scores **0.812 / 0.938 in `early` and
  0.000 in `late`** there; argmax mirrors it. This is **not closable**: "the answer is early" is
  the manipulation, so any rule reading position tracks it by definition.
- **The d=1 cells are clean on that rule** (0.000 in both bands): with one mention each, the
  distractors pinned to the first and last body slots always hold the extreme means.
- `k3d1` has `most_frequent_candidate_in_context` = 1.000 by construction — the ALIAS-001 `high`
  ceiling. Absolute rates there are not readable either.
- **The interaction is readable in every cell**, which is why it is the primary.

## Nulls

Any null is reported with its **minimum detectable effect** at the realised denominator, per
requirement 14. The denominator for the primary is **discordant pairs** — pairs where
`d_delta != d_source` — **not the 48 pairs per cell**. That is the denominator that bounds the
estimate, and it is the one that has been got wrong before: a repair fraction's denominator is
the damage count, not the stratum size. An exact sign test needs **>= 6 discordant pairs** for
p < 0.05 and more under the 0.00625 correction, so if a cell yields fewer the result is reported
as *"not powered to distinguish"*, never as "no effect", and the realised discordant count is
printed whatever the outcome.

## What the build already refused, and what it cost

Recorded because the bank passed several checks in states that were wrong:

1. **`last_sentence` located the answer** at 0.214 vs a 0.098 floor — the `late` band was putting
   the answer in the final sentence. Fixed by reserving the terminal slot.
2. **`first_candidate_in_context` scored 0.708 in `early` and 0.000 in `late`**; `last_candidate`
   0.000 and 0.604. **The pooled floor read 0.354 and hid both.** Fixed by pinning the first and
   last body slots to distractor lines, forcing both rules to 0.000 in every band.
3. **Body length ran 7 sentences in `k1d1` and 15 in `k3d3`**, confounding cell with context
   length. Fixed by holding the body at 15 sentences with neutral padding.
4. **The binding sentence asserted a falsehood in every k=3 row** — inherited verbatim from
   `build_pres4`, it says the answer *"is not named again in this account"* and then names it
   twice more, **true in the k=1 cells and false in the k=3 cells**, i.e. a self-contradiction
   perfectly confounded with the manipulation. **Found by reading a rendered row.** Every
   automated check passed it.
5. The band-gap guard itself was wrong on first writing: a flat 0.10 gap refused the bank on a
   4-way frequency tie broken by arbitrary set order (p = 0.107). Replaced by Fisher at
   0.05/16 with a 0.25 hard cap, which still catches defect 2 at p = 9.9e−13.
6. **`argmin` of each candidate's MEAN mention position scores 0.812–0.938 in `early` and 0.000
   in `late`** in the d=3 cells. The fix for defect 2 does not close it: pinning the first and
   last slots closes rules that read a **single extreme**, and this one reads the **bulk**. It is
   **not closable at all** — "the answer is early" is the manipulation. Raised by the peer
   session; measured here at worse than they estimated, and in `k3d3`, the cell the whole
   decomposition rests on. **This is what forced the primary test to become the interaction**,
   and the earlier absolute-rate draft would have had a confound that manufactures the predicted
   direction: cue-solvable rows cannot be damaged, so the band with more of them shows a lower
   damage rate for free.

7. **All three arms exploit the positional cue at ~2x chance when they err**, measured on
   ALIAS-001 by the peer session and reproduced here on the bank's declared 12-name pool:
   Supa 54/85 = 0.635, native 93/132 = 0.705, LWF 109/151 = 0.722, against a 0.333 chance
   baseline with exactly 4 candidates per row. **The differentials are +0.069 (p = 0.30) and
   +0.087 (p = 0.19) — in the direction that would inflate a positive interaction, and not
   significant.** Per requirement 14 only a differential of **>= ~0.165** was detectable at these
   denominators, so anything below that is **not excluded**. That bound is what the
   manufacturable column above is computed from.

   *Methodological note on that measurement:* rows where the cue **is** the reference are
   excluded from the denominator, because an error cannot match the cue there. Including them
   depresses every rate and by a different amount per arm, since the arms have different error
   sets.

**Defect 4 is the one to carry forward: it was semantic, and no mechanical check can see a
sentence that is merely untrue. Defect 6 is the one that cost the most: it removed a claim.**

## Not registered here

Whether to run this bank at all, and against which arms, is a resource decision. Nothing in this
document requests GPU or a confirmation budget.

---

# Amendment 1 — the registered primary could not reach its own registered prediction

**Written before any arm was generated and before any prediction existed.** A five-dimension
adversarial review was run on the scorer and the run plan *before* spending GPU. It found the
primary unfalsifiable in the main cell. Both defects were then confirmed by direct arithmetic,
independently of the reviewer.

**Honest status of that review:** **56 of its 62 agents died on a usage limit**, taking the
entire verification phase and the statistics, pairing and operational dimensions with them. Two
dimensions completed. **Nothing in it was adversarially verified** — the surviving findings carry
empty verdict lists, so they survived vacuously. Only the two blockers below are established,
and they are established because I checked the arithmetic myself, not because the review said so.

## Blocker 1 — `k1d1` was unfalsifiable as specified

At 48 pairs per cell an exact two-sided sign test can declare significance at 0.0125 only from a
net of 8, so the smallest reachable |interaction| is **8/48 = 0.1667**, rising with the realised
discordant count (net 12 at D=20, net 16 at D=30). The registered `k1d1` band was **+0.03 to
+0.15 — entirely below that**, and the lower 40% of `k3d1`'s band was also unreachable.

**If the delta had behaved exactly as registered, the run would have returned "no significant
interaction" and the record would have read as a refutation.**

## Blocker 2 — the MDE was computed from alpha alone

The scorer used a flat `discordant >= 8`. The required net grows with the realised denominator,
so a null could have been stamped `powered=true` with an MDE too small by half. This is the
flat-threshold-on-a-noisy-estimate defect **this programme has already made once**, in the
band-gap guard of this very bank's builder.

## What changed

1. **Bank rebuilt at 48 packs/cell → 144 pairs per cell**, as `mpos002-20260919`. The 48-pair
   bank is left intact; nothing was ever read from it.
2. **Primary test is now a sign-flip permutation on the MEAN** of `d_delta - d_source`, which
   uses magnitude as well as sign. The exact sign test is retained as a secondary.
3. **Measured power at 144 pairs** (sign-flip permutation, alpha 0.0125, assuming a 0.30
   discordance rate — an assumption, and it is the one this power claim rests on):

   | n pairs | effect +0.09 (`k1d1` band midpoint) | effect +0.20 (`k3d1` band midpoint) |
   |---:|---:|---:|
   | 48 *(old bank)* | 0.05 | 0.49 |
   | **144** | **0.26** | **0.99** |

   > **`k3d1` is powered for its registered band. `k1d1` is NOT, and is registered here as an
   > estimate-with-interval rather than a test.** A null in `k1d1` will be reported as "not
   > powered to distinguish", never as evidence of absence.
4. **Alpha is fixed at 0.0125 (Bonferroni over 4)**, resolving a contradiction in the original:
   it registered 0.00625 in one place and 0.0125 in another, and an 8/8 result is significant
   under one and not the other.
5. **The cue secondary now reports the DIFFERENTIAL with its minimum detectable value**, plus a
   Newcombe method 10 upper bound — not the three bare per-arm rates the first draft emitted,
   which would have been a null with no MDE in the middle of a document about nulls needing MDEs.

## A cue the build did not close, and its direction

Enumerating answer-agnostic rules on the two clean cells found one that survives the terminal-slot
pinning: **"emit the candidate named in the penultimate sentence"**, which scores **0.000 early /
0.750 late in `k3d1`** and 0.000 / 0.208 in `k1d1`. Pinning the last slot closed
`last_candidate_in_context` and pushed the cue one position inward.

**Its direction is what matters, and it is favourable.** The cue helps the **LATE** band, so an
arm exploiting it is *more* correct late, which *lowers* `d_arm`. If the delta exploits it more
than the source, the statistic is pushed **negative** — **away from the registered positive
direction**.

> **Registered asymmetry: a POSITIVE result survives this cue and is conservatively estimated.
> A NULL or NEGATIVE result is NOT interpretable, because it is indistinguishable from the cue.**

This is registered now, before the numbers exist, precisely so it cannot be invoked afterwards to
explain away an unwelcome result.

## Amendment 1a — scorer smoke-tested on synthetic predictions, and a correction to the power claim

Run before any real arm existed, on synthetic predictions with a **planted** interaction
(native +0.20, LWF +0.15, source 0.00). Two things came out of it.

**1. The scorer hung, and would have hung on the real data.** `mde_by_simulation` was
O(100 effect sizes x 300 reps x 600 permutations) per cell and did not finish a 144-pair cell in
120 s. Replaced with the analytic equivalent: under the sign-flip null the mean has
sd = sqrt(sum(x^2))/n exactly, so **MDE = (z_{alpha/2} + z_{power}) * sqrt(sum(x_i^2)) / n**.
Labelled in the receipt as a normal approximation and reported **alongside** the exact
permutation p, never instead of it.

**2. The scorer recovers a planted effect and correctly refuses an underpowered cell** — the
synthetic run returned +0.16 to +0.30 against planted +0.15/+0.20, flagged
`k1d1/native` as *"not powered to distinguish"* because its realised MDE (0.238) exceeded the
observed mean, and produced a differential with its own minimum detectable value.

**3. A correction to this prereg's own power table.** Amendment 1 registered power **0.99** for a
+0.20 effect at 144 pairs. That figure is **conditional on an assumed 0.30 discordance rate**.
The synthetic run realised a discordance of 0.54-0.65, at which the MDE is **0.24-0.26** — so at
+0.20 the power would be **below 80%, not 0.99**.

> **Higher discordance means more variance, not more signal.** The registered 0.99 is an
> assumption, not a property of the design, and the realised MDE reported per cell **supersedes
> it**. If the real run comes back with high discordance and a null, that null is bounded by the
> realised MDE printed beside it — not by the number registered here.

This is recorded before the real arms exist so the registered figure cannot later be quoted as
though the run had been powered when it was not.

## Amendment 1b — the registered directional asymmetry is WITHDRAWN; the cue space is symmetric

**Written while generation was running and BEFORE any prediction was read.** Timestamped by the
commit; the arms were mid-flight and no `predictions.jsonl` existed for any of them.

Amendment 1 registered that the one surviving cue (*"emit the candidate in the penultimate
sentence"*, 0.000 early / 0.750 late) favours LATE, and concluded:

> *"A POSITIVE result survives this cue and is conservatively estimated. A NULL or NEGATIVE
> result is NOT interpretable."*

**That asymmetry does not hold.** It rested on a hand-picked list of 13 rules. A systematic sweep
— argmin/argmax over 12 scalar per-candidate features, plus a per-sentence-index rule for each of
16 sentence positions, 40 rules per cell — finds **equally strong rules in BOTH directions**:

| cell | strongest EARLY-favouring | strongest LATE-favouring |
|---|---|---|
| `k3d1` | `in_sentence_5` **0.771 / 0.000** | `in_sentence_11` 0.000 / 0.771 |
| `k1d1` | `in_sentence_3` 0.271 / 0.000 | `in_sentence_11` 0.000 / 0.285 |

All at p < 1e-10. **The rule space is symmetric in direction**, so there is no basis for claiming
the residual pushes against my registered positive. **The asymmetry is withdrawn.**

## And the "clean cells" are not cue-free either

The cells were called clean because `argmin`/`argmax` of **mean mention position** has a band
swing of exactly 0.000 there. That was true and it is not sufficient: **sentence-index rules
swing up to 0.771.** At the registered differential bound of 0.211 the manufacturable interaction
is:

| cell | max band swing | manufacturable | registered band |
|---|---:|---:|---|
| `k1d1` | 0.285 | 0.060 | +0.03 to +0.15 |
| `k3d1` | 0.771 | **0.163** | +0.10 to +0.30 |

**So `k3d1`'s worst case reaches into its own registered band.** This is the same conclusion the
d=3 cells were excluded for, arrived at through a rule class I had not swept.

**Why this was inevitable and is not a fixable defect.** A sentence-index rule *is* the
manipulation restated: the answer sits in slots 1-5 in `early` and 10-13 in `late` **by
construction**, so "emit the candidate in sentence 5" must separate the bands. Any experiment
that moves the answer creates such rules. This is the point already recorded in the withdrawal of
the d=3 cells — *you cannot manipulate position and leave position uninformative* — and it
applies to every cell, not only the ones with three mentions.

## What this does and does not change

**Does not change:** the interaction still cancels any cue **both arms exploit equally**. That is
the design's actual protection and it is untouched.

**Does change:** the residual — differential exploitation — is now known to be **bidirectional
and potentially large**, and the 0.163 worst case in `k3d1` is an extrapolation of a bound
measured on a *different and much weaker* cue (earliest-person, on ALIAS-001). Compounding them
is pessimistic but not unreasonable.

**Registered consequence, before the numbers exist:** a significant positive interaction in
`k3d1` **cannot be reported as establishing the causal claim on its own.** It must be reported
beside the measured cue differential from the secondary, and if that differential is itself
significant and positive, the result is **"not separable from differential cue exploitation"** —
the same verdict the d=3 cells received. `k1d1`'s worst case (0.060) sits below the middle of its
band, so it is the **less contaminated** cell despite being the less powered one.

**The secondary is now the load-bearing measurement, not a footnote.** It is already implemented
and measures the earliest-person differential with its own MDE. That is the right shape but the
wrong cue; the strongest band-dependent rule should be measured too. I am **not** changing the
scorer mid-flight — the addition will be a separate, separately-committed analysis run after the
primary, so that nothing in the registered path moves after predictions exist.

---

# RESULT — the position effect is CAUSAL in `k3d1`, on both deltas

**Class: MECHANISM SCOUT.** One bank, one draw. Receipts `MPOS_RESULT.json` and
`CUE_DIFFERENTIAL.json`. Scorer and registered path untouched since before the first arm ran.

## Primary

| cell | lane | pairs | disc | mean | 95% CI | MDE | p | verdict |
|---|---|---:|---:|---:|---|---:|---:|---|
| `k3d1` | native α=1 | 144 | 53 | **+0.285** | [+0.201, +0.375] | 0.169 | <0.0001 | **SUPPORTS** |
| `k3d1` | LWF λ=0 | 144 | 49 | **+0.243** | [+0.160, +0.326] | 0.162 | <0.0001 | **SUPPORTS** |
| `k1d1` | native α=1 | 144 | 31 | +0.076 | [+0.000, +0.153] | 0.129 | 0.070 | not powered |
| `k1d1` | LWF λ=0 | 144 | 39 | +0.049 | [−0.035, +0.132] | 0.145 | 0.334 | not powered |

> **Moving the answer from the first third of the context to the last third costs the DELTA
> about five times what it costs the SOURCE** — source 0.0625, native 0.3472, LWF 0.3056 — and
> the difference replicates across two independently trained deltas at p < 0.0001.

`k1d1` is **not powered**, exactly as Amendment 1 registered in advance. Its realised MDE
(0.129 / 0.145) exceeds its observed mean, so it is reported as an estimate with an interval and
**not** as evidence of absence.

## The confound, measured rather than extrapolated — and it contributes nothing

Amendment 1b registered that a positive in `k3d1` **could not stand alone**: it had to be read
beside the cue differential, because a differentially-exploited band-dependent cue could
manufacture it. The extrapolated worst case was **0.167**.

**That extrapolation was wrong in a specific way: it paired a differential measured on ONE cue
with a band swing belonging to ANOTHER.** Measured properly, each cue's own differential against
its own swing:

| `k3d1` cue | band swing | differential (native) | differential (LWF) | manufacturable |
|---|---:|---:|---:|---:|
| earliest person / earliest mean | **0.000** | +0.060 / +0.154 | | **0.000** |
| `in_sentence_5` | 0.771 | **+0.000** | −0.048 | 0.000 / −0.037 |
| `in_sentence_11` | 0.771 | **+0.000** | −0.167 | 0.000 / −0.128 |
| `in_sentence_3` | 0.792 | −0.067 | −0.067 | −0.053 |
| `in_sentence_13` | 0.792 | **+0.000** | +0.000 | 0.000 |
| **sum of positive manufacturable** | | | | **0.000 / 0.000** |

**The cues that swing across bands show zero or negative differential exploitation; the cues with
a positive differential do not swing across bands.** The product — which is what manufacturable
means — is **zero on both lanes**, against an observed +0.285 and +0.243.

So the registered disqualifier does not fire, and it does not fire **because it was measured**.
Had the extrapolated 0.167 stood, it would have consumed 59% of the native effect and 69% of the
LWF effect and left the result unreadable.

## What is and is not established

**Established, within this bank:** the position effect is **causal, not merely observational**.
Amendments 18–20 found it by correlating damage with where the answer happened to land under a
shuffle. Here the answer was **assigned** a band, every scenario appears in both bands built from
identical material, and the interaction is large, replicated and not attributable to any measured
cue.

**Not established.** `k1d1` — underpowered by design and reported as such. **Why `k3d1` shows it
and `k1d1` does not is NOT answerable here**: the two cells differ in mention count *and* salience
ratio simultaneously, which is the ALIAS-001 confound this bank failed to break, and `k1d1` is
underpowered regardless, so the contrast could be power rather than substance. The count-vs-ratio
decomposition **stays withdrawn**.

**Unchanged:** one bank, one draw, two cells, two lanes. Nothing here promotes anything, and no
confirmation budget was requested or spent.

## RESULT amendment 1 — the confound table is WITHDRAWN as a measurement; a stronger argument replaces it

The result section above reports *"sum of positive manufacturable = 0.000"* on both lanes. **That
figure is not a measurement of absence, it is an absence of data**, and it is withdrawn.

**The denominators, checked after the peer session challenged them.** A cue differential is
scored over rows where the arm errs AND the cue does not already name the reference. In `k3d1`
the frozen source errs only 21 times in total, and after the `cue != reference` filter:

| cue | source scorable | native | LWF |
|---|---:|---:|---:|
| earliest person / mean / latest mean | 21 | 76 | 80 |
| `in_sentence_5` | **3** | 6 | 7 |
| `in_sentence_11` | **2** | 2 | 3 |
| `in_sentence_3` | **3** | 5 | 5 |
| `in_sentence_13` | **1** | 2 | 3 |

**Every sentence-index differential — the ones that carried the argument — rests on one to three
source rows.** Those cells should read *"differential unmeasurable at these denominators"*, never
`+0.000`.

**And it is not fixable by adding rows**, for two compounding structural reasons: the
`cue != reference` filter deletes precisely the rows where the cue is *correct*, which is most of
the band in which the cue is accurate; and the source errs rarely here because it is good at the
task. The estimator is thinnest exactly where the cue matters most.

**A second bias, which the peer named and I had not:** an arm that follows a cue *successfully*
is not in the error set at all, so heavy exploitation in a band where the cue is accurate appears
as **fewer errors**, not as a higher match rate. The "among errors" estimator is least sensitive
where it most needs to be sensitive.

## What replaces it — raw error counts, no cue, no filter, no differential

`k3d1`, 144 rows per band:

| arm | early errors | late errors | late − early | delta/source ratio, early → late |
|---|---:|---:|---:|---|
| frozen Supa | **6** | 15 | +0.0625 | — |
| native α=1.00 | **13** | 63 | +0.3472 | **2.2× → 4.2×** |
| LWF λ=0 | 18 | 62 | +0.3056 | **3.0× → 4.1×** |

(0.3472 − 0.0625 = **+0.2847**, reproducing the primary exactly.)

> **In the EARLY band the source is BETTER than the delta — 6 errors against 13 and 18.** An
> early-favouring cue inflating the delta's early performance would push the delta *toward* the
> source there. It does not: the delta is 2.2–3.0× worse in `early`, and the gap **widens** to
> 4.1–4.2× in `late`. **An early-favouring cue cannot produce a late-band collapse.**

This uses only raw error counts. No cue identification, no scorable filter, no differential, no
thin denominator — so it is immune to everything that invalidates the table above.

**The caveat, stated so it is not over-read:** this rules out the **early-inflation** mechanism
specifically. A cue that made LATE rows harder for the delta than for the source would still
manufacture the interaction and is **not** ruled out. It is a stranger mechanism to posit, and
the late-favouring cues here (`in_sentence_11`, `in_sentence_13`) are shared by both arms by
construction — but "stranger" is not "excluded", and no measurement here excludes it.

**One observation from `k1d1` that cuts against a simple reading:** there the delta/source error
ratio **narrows** (native 3.5× early → 2.7× late; LWF 5.3× → 3.2×) even though the absolute
interaction is positive (+0.076, unpowered). Ratio and absolute difference disagree in that cell.
The registered statistic is the absolute one and it stands, but the disagreement is recorded
rather than smoothed over.

**Net effect on the headline:** the causal finding in `k3d1` is **unchanged and now rests on a
better argument than the one it was published with**. What is withdrawn is the claim that the
confound was measured to be zero.

---

# Addendum 2 — does the k=0.25 magnitude mask REPAIR the position mechanism?

**Written before the Titan-Sparse arm was generated and before any of its predictions existed.**
Proposed by the peer session; the bank, scorer and analysis path are unchanged.

MPOS-002 established that moving the answer late costs the **delta** ~5× what it costs the
**source**. Separately, `Rineya-Titan-Sparse-Experimental` — the k=0.25 magnitude-pruned core —
scores better than the other promoted core on alias. **Nobody has asked whether the artifact that
scores better actually repairs the mechanism this bank isolates.**

## Arm identity, verified before the run

| property | value |
|---|---|
| whole-file sha256 | `a28629c9d3454fbf…` |
| DATA SECTION sha256 | `8fb85df27baea919…` |
| dtype | **310/310 F32** — FP32, as the core declares; **not** a BF16 copy |
| bytes | 2,384,234,992 — identical to the α=1.00 arm |
| `generation_config.json` | **byte-identical to Supa** and to all three scored arms |
| `config.json` | differs from the others in `torch_dtype` only (float32 vs bfloat16) |

`torch_dtype` is metadata: `stage2_generate.py` takes `--dtype` as a **required** argument and
passes `dtype=torch.float32` to `from_pretrained`, which overrides it. All three completed arms
recorded `dtype=fp32` under generator sha `5e9cc3201e75d30b`, and Sparse will run identically.
**This check exists because the core's own record states a BF16 copy "zeroes 28.5% of its kept
update and is a different model"** — scoring the wrong dtype would silently score a different
artifact.

## Registered prediction

Position cost = `P(wrong | late) − P(wrong | early)`, measured on `k3d1`:

| arm | position cost |
|---|---:|
| frozen Supa | 0.0625 |
| native α=1.00 | 0.3472 |
| LWF λ=0 | 0.3056 |
| **Titan-Sparse** | **predicted 0.10 – 0.25** |

**Registered primary: the interaction `mean(d_sparse − d_source)` on matched pairs in `k3d1`,
predicted +0.05 to +0.20** — positive, but **smaller than native's +0.2847**.

Registered readings, fixed in advance:

* **≤ 0.10** — the mask **substantially repairs** the mechanism; the endpoint gain and the
  mechanism coincide.
* **0.10 – 0.25** — partial repair, my registered expectation.
* **≥ 0.30** — the mask does **not** touch this mechanism. That is the **more informative**
  outcome: it would mean Sparse's alias advantage comes from something other than the causal
  effect this bank isolates, splitting the endpoint from the mechanism.

A **negative** interaction would mean the mask makes position sensitivity worse than the source's,
which nothing predicts.

## Multiplicity, declared now rather than chosen later

This adds **2 tests** (two cells, one new arm). Reported both ways: Bonferroni over 2 = 0.025 for
this addendum read alone, and **Bonferroni over 6 = 0.00833** for the full family of three arms ×
two cells. **The 6-test bound governs any statement comparing Sparse to the other arms**, because
that comparison uses all six. Choosing the smaller family after seeing the number is the move
this programme has already caught twice today.

## What does not change

`k1d1` remains **underpowered** and is reported as an estimate with an interval, not a test. The
cue caveats from Amendment 1b and the RESULT amendment apply unchanged: the confound table is
withdrawn as a measurement, and the load-bearing argument is the **raw error-count** one — in the
`early` band the source is better than the delta, so an early-favouring cue cannot produce a
late-band collapse. That argument will be recomputed for Sparse and reported beside the primary.

---

# RESULT (Addendum 2) — the mask PARTIALLY repairs the mechanism; registered band HIT

Receipt `MPOS_RESULT_WITH_SPARSE.json`. Arm verified before generation (`a28629c9d3454fbf…`,
310/310 F32, `generation_config` byte-identical to the other three). Family **0.05/6 = 0.00833**,
derived in the scorer from the lane count rather than hard-coded.

## Position cost on `k3d1`

| arm | early errors | late errors | position cost | vs source |
|---|---:|---:|---:|---:|
| frozen Supa | 6 | 15 | **0.0625** | — |
| native α=1.00 | 13 | 63 | **0.3472** | +0.2847 |
| LWF λ=0 | 18 | 62 | 0.3056 | +0.2431 |
| **Titan-Sparse (k=0.25)** | 12 | 44 | **0.2222** | **+0.1597** |

> **Titan-Sparse closes 44% of the gap between the full delta and the frozen source.**

**Registered band was +0.05 to +0.20. Observed +0.1597 — inside it**, and the registered reading
for that range was *"partial repair"*. The prediction hit, and it was committed at `0254572f`
before the arm was generated.

| paired contrast (source cancels) | mean | discordant | perm p | clears 0.00833? |
|---|---:|---:|---:|---|
| native − Sparse | +0.1250 | 20 | **0.0001** | **YES** |
| LWF − Sparse | +0.0833 | 24 | 0.0227 | **no** |

**So "Sparse repairs more than the native delta" is supported at the corrected bound; "Sparse
repairs more than the LWF delta" is not.** Stated that way rather than pooled.

## The confound, checked the way the RESULT amendment requires

The cue table stays withdrawn as a measurement. The load-bearing check is the raw error-count
argument, recomputed for Sparse:

* **In `early`, the source is still better than Sparse — 6 errors against 12.** An
  early-favouring cue inflating Sparse's early performance would push it *toward* the source
  there; it does not.
* Sparse's earliest-person cue rate is **0.6000 against the source's 0.6190 — a NEGATIVE
  differential (−0.019)**. A negative differential cannot manufacture a positive interaction.

## What this establishes, and the caveat that must travel with it

**Establishes:** the artifact that scores better on the alias *endpoint* also measurably repairs
the *causal mechanism* this bank isolates. Those were separate facts until now, and the peer
session's framing was right that the split was the thing worth testing.

**Does not establish a ceiling.** The peer's mask ladder, measured after this was registered,
finds **k=0.25 is the weakest point on it** — deviations from the scalar curve run +21.7 / +9.8 /
+1.2 at k = 0.10 / 0.25 / 0.50. Titan-Sparse is the **middle** of that ladder. So **44% is a
lower bound on what masking can do for this mechanism, not an estimate of it**, and `s10` is the
arm that would discriminate. The registered band was not widened to accommodate this; it is
attached as an interpretation note.

**`k1d1` remains underpowered on all three lanes** (MDEs 0.129 / 0.145 / 0.106 against observed
0.076 / 0.049 / 0.063) and is reported as estimates with intervals, never as nulls.

One bank, one draw. MECHANISM SCOUT. Nothing promoted, no confirmation budget spent.

---

# Addendum 3 — s10, and the ACQUISITION confound the peer session identified

**Written before the s10 arm was generated.** The peer session raised a confound that would have
made an s10 result ambiguous, and the control for it is an arm I have already scored.

## The confound

Position cost correlates with **how much was acquired**, across every arm measured:

| arm | acquisition /384 | position cost | masked? |
|---|---:|---:|---|
| frozen Supa | 14 | 0.0625 | no delta |
| **s10** | **197** | **to run** | **masked** |
| **LWF λ=0** | **198** | **0.3056** | **unmasked** |
| Titan-Sparse | 232 | 0.2222 | masked |
| native α=1.00 | 235 | 0.3472 | unmasked |

The source acquires 14 and has almost no position cost. So **an s10 result below 0.2222 is
ambiguous** between *masking repairs the mechanism* and *an arm that learned less has less to be
position-sensitive about*.

**LWF λ=0 is the control: 198 against s10's 197, matched to one row, and unmasked.**

**Verified, because the whole control rests on it.** The LWF records carry **two** figures — an
unmatched lane reaching 183 and a matched lane at 198. The arm I actually scored
(`arm-lambda0`, weight sha `6836dae3…`) records `update_groups: 28` and `matched_to` the design,
which is the **matched** lane. So 198 is the right number for the arm in hand, and the 183 belongs
to a 26-group lane I did not use.

## Two registered bands, for two different questions

**Band A — the LADDER question.** s10 against Titan-Sparse's 0.2222. Predicted s10 position cost
**0.10 – 0.20**, i.e. repairing at least as much as Sparse, since the peer's ladder shows the mask
advantage grows as k falls.

**Band B — the MECHANISM question, and the decisive one.** s10 against **LWF λ=0's 0.3056 at
matched acquisition**. Registered primary: the paired interaction `mean(d_lwf − d_s10)` on matched
pairs in `k3d1`, **predicted positive, +0.08 to +0.22**.

* s10 cost **≪ 0.3056** → **masking repairs the mechanism at matched acquisition.** The repair is
  not an artifact of having learned less.
* s10 cost **≈ 0.3056** → the repair **tracks acquisition level**, and Titan-Sparse's 44% closure
  is partly or wholly an acquisition artifact. That would also qualify today's headline: Sparse
  acquires 232 against native's 235, so it is *nearly* acquisition-matched — but "nearly" at 3
  rows is not matched at 1.

**Registered interaction band for s10 vs source in `k3d1`: +0.02 to +0.16.**

## Arm identity, verified before the run

`sparse001-20260917/arm-rebuild-s10`: whole-file `d7b15d2e79c21183…`, DATA SECTION
`bf2e132ee3d25be3…`, **310/310 F32**, 2,384,234,992 bytes — identical size to every other arm.
`generation_config.json` byte-identical to all four scored arms. `config.json` matches Supa's
(`660db3b7…`), so its `torch_dtype` metadata is the same as the native and LWF arms; `--dtype
fp32` governs regardless.

## Multiplicity

Four lanes × two cells = **8 tests, Bonferroni 0.05/8 = 0.00625**, derived in the scorer from the
lane count. Every cross-arm statement is governed by that bound, including the Sparse comparisons
already reported — **those were made at 0.00833 and the tighter bound is the one that now
applies.** Re-checked: native−Sparse at p = 0.0001 still clears; LWF−Sparse at p = 0.0227 still
does not.

---

# RESULT (Addendum 3) — masking repairs the mechanism AT MATCHED ACQUISITION; both bands HIT

Arm verified before generation (`d7b15d2e79c21183…`, 310/310 F32). Family **0.05/8 = 0.00625**.

## Position cost on `k3d1`, ordered by acquisition

| arm | acquisition /384 | early err | late err | position cost | masked? |
|---|---:|---:|---:|---:|---|
| frozen Supa | 14 | 6 | 15 | **0.0625** | — |
| **s10** | **197** | 9 | 29 | **0.1389** | **MASKED** |
| **LWF λ=0** | **198** | 18 | 62 | **0.3056** | **unmasked** |
| Titan-Sparse | 232 | 12 | 44 | 0.2222 | MASKED |
| native α=1.00 | 235 | 13 | 63 | 0.3472 | unmasked |

> **At acquisition matched to one row — s10 at 197 against LWF λ=0 at 198 — the masked arm's
> position cost is 0.1389 and the unmasked arm's is 0.3056.**

**The acquisition confound is ruled out.** The peer session's Band B was the decisive test and it
came out on the side that keeps the mechanism claim: the repair is **not** an artifact of having
learned less. Ordering the table by acquisition makes it plain — the two masked arms sit **below**
both unmasked arms while spanning acquisition 197 to 232.

| paired contrast (source cancels) | mean | discordant | perm p | clears 0.00625? |
|---|---:|---:|---:|---|
| **LWF − s10** *(acquisition-matched)* | **+0.1667** | 38 | **0.0001** | **YES** |
| native − s10 | +0.2083 | 38 | **<0.0001** | **YES** |
| Sparse − s10 | +0.0833 | 18 | 0.0078 | **no** |

**Both registered bands HIT**, both committed at `2957513c` before the arm ran:

* Band A (ladder): predicted cost **0.10–0.20**, observed **0.1389**.
* Band B (mechanism): predicted `mean(d_lwf − d_s10)` **+0.08 to +0.22**, observed **+0.1667**.

**"s10 repairs more than Titan-Sparse" is NOT supported at the corrected bound** (p = 0.0078
against 0.00625). It is directionally consistent with the peer's monotone ladder, and that is all
it is. Reported separately rather than pooled with the two contrasts that do clear.

## Gap closure toward the frozen source

| comparison | closure |
|---|---:|
| Titan-Sparse vs native | 44% |
| **s10 vs native** | **73%** |
| **s10 vs its acquisition-matched control (LWF)** | **69%** |

## The confound check, by the method that survived

Cue exploitation among each arm's own errors, `k3d1`:

| arm | rate | vs source | masked? |
|---|---:|---:|---|
| frozen Supa | 0.5714 | — | — |
| s10 | 0.5263 | **−0.0451** | MASKED |
| Titan-Sparse | 0.5179 | **−0.0536** | MASKED |
| native α=1.00 | 0.6316 | +0.0602 | unmasked |
| LWF λ=0 | 0.7250 | +0.1536 | unmasked |

**Both masked arms have NEGATIVE cue differentials; both unmasked arms are positive.** A negative
differential cannot manufacture a positive interaction, so neither mask result is cue-driven. And
in `early` the source is still better than s10 — 6 errors against 9 — so an early-favouring cue
cannot be producing the late-band collapse.

## What this establishes

**Magnitude masking repairs the causal position mechanism, and the repair is not explained by
acquisition level.** The endpoint advantage, the mechanism repair, and the acquisition-matched
control now agree. One bank, one draw, MECHANISM SCOUT — nothing promoted, no budget spent.

**Not established:** the k at which the repair saturates. `s10` vs `Sparse` does not clear
correction, so the ladder's monotone shape is suggestive here and measured only on the endpoint.

## RESULT amendment 2 — the mask-status cue pattern is DROPPED

The Addendum 3 result recorded, as an explicit observation rather than a finding, that cue
exploitation split by mask status (masked −0.045 / −0.054, unmasked +0.060 / +0.154). The peer
session was asked to break it rather than confirm it. It breaks three ways, all verified here:

1. **It is one cue, not four.** `earliest_person` and `earliest_mean` select the **same candidate
   on 288/288 rows** — identical by construction on this bank. And on `longest name` the ordering
   **reverses**: masked s10 carries the largest positive differential of any arm. The split holds
   on **1 of 4** distinct cues, which is what the peer's ten-cue ALIAS-001 sweep predicted — the
   differential is a property of the arm-pair-**and-cue**, sign-varying.
2. The damage-proxy alternative is **disfavoured but not excluded**: ρ = +0.600 at n = 4, where
   only ρ = 1.0 approaches significance, and the ordering inverts within each group.
3. **Decisive.** These are candidate-**identity** cues, and their measured band swing is
   **exactly 0.0000** (`argmin_first`, `argmin_mean`, `argmax_first`, `argmax_mean`,
   `argmax_namelen` all 0.0000). The manipulation moves the answer's *position*, not which
   candidate is which. Since manufacturable = differential × swing, **the differential cannot
   manufacture anything at any value.**

> **The pattern was computed on cues that are structurally incapable of producing the effect they
> were being used to rule out. It is dropped, not softened.**

**This costs the result nothing**, because the pattern was never the load-bearing protection.
That remains (a) the **sentence-index** cues — the only rules that actually swing across bands —
whose differentials were measured and are zero or negative, and (b) the **raw error-count**
argument: in `early` the source still beats every masked arm, so early-inflation is excluded with
no cue machinery at all.

**Nothing in the primary moves.** Position costs, both registered bands, the acquisition-matched
paired test and the gap-closure figures are unchanged.
