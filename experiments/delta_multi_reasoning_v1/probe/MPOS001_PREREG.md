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

| cell | k | d | cue swing | manufacturable | **role** | predicted `d_delta - d_source` |
|---|---|---|---:|---:|---|---|
| `k1d1` | 1 | 1 | **0.000** | **0.000** | **PRIMARY** | **+0.03 to +0.15** |
| `k3d1` | 3 | 1 | **0.000** | **0.000** | **PRIMARY** | **+0.10 to +0.30** |
| `k1d3` | 1 | 3 | 0.812 | 0.134 | secondary | +0.00 to +0.08 |
| `k3d3` | 3 | 3 | 0.938 | **0.155** | secondary | +0.10 to +0.30 |

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
