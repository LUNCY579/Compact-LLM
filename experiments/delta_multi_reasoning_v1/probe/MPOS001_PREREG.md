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
