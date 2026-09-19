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

## Registered primary test

For each cell × lane: **damage rate in `late` minus damage rate in `early`**, scored over the
rows **the frozen source answers correctly** (the damage denominator rule), Fisher exact,
**Bonferroni over 8 tests (4 cells × 2 lanes) = 0.00625**.

**Direction is registered: `late` > `early`.** A significant reversal falsifies the positional
account as an intervention.

## Registered predictions, with the bands I will be judged against

| cell | k | d | ALIAS-001 analogue | **predicted late−early** |
|---|---|---|---|---|
| `k1d3` | 1 | 3 | `low` | **+0.00 to +0.08** — `low` was the observational null |
| `k3d1` | 3 | 1 | `high` | **+0.15 to +0.35** — `high` carried ρ=+0.43 |
| `k1d1` | 1 | 1 | none | **+0.05 to +0.20** |
| `k3d3` | 3 | 3 | none | **+0.15 to +0.35** |

**The decomposition this bank exists to make.** If the effect tracks **mention count**, `k3d1`
and `k3d3` both show it and `k1d1`/`k1d3` do not. If it tracks **salience ratio**, `k3d1` shows
it and `k3d3` does not. **`k3d3` is the discriminating cell** and my registered prediction above
backs *count* over *ratio*.

## Readability, stated in advance

- **Absolute rates are readable only in `k1d1` and `k3d3`** (frequency-neutral: answer and each
  distractor mentioned equally often).
- **`k3d1` has `most_frequent_candidate_in_context` = 1.000 by construction** — the ALIAS-001
  `high` ceiling. Absolute rates there are not readable; the paired arm contrast is.
- **Paired arm contrasts are readable in every cell**, because a deterministic answer-agnostic
  rule emits the same string for both arms and cannot create a discordant pair.
- **The early-vs-late contrast is a BETWEEN-ROW comparison, so it is NOT protected by that
  immunity.** Its validity rests on the band-gap check below, not on pairing.

## Nulls

Any null is reported with its **minimum detectable effect** at the realised denominator, per
requirement 14. Given ~48 source-correct rows per cell-band, the detectable difference is
roughly **0.20–0.25** at the corrected bound; anything smaller will be reported as *"not powered
to distinguish"*, never as "no effect".

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

**Defect 4 is the one to carry forward: it was semantic, and no mechanical check can see a
sentence that is merely untrue.**

## Not registered here

Whether to run this bank at all, and against which arms, is a resource decision. Nothing in this
document requests GPU or a confirmation budget.
