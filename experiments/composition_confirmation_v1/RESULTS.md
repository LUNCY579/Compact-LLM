# CONFIRM-001 — results

Run 2026-09-24, 16:57–18:11 (attempt 2; attempt 1 failed at model load before any row was generated,
see `ATTEMPT_LOG.md`). Bank `T_BANK_V3X.json` (864 rows, sha256 `173aa372…6557c`), read once.
Registered in `PREREG.md` (c3a6f8a0) and `AMENDMENT_1.md` (97c390d4, 83f9716f), both committed
before any arm was scored. Registered output: `CONFIRM_RESULT.json`. Independent audit:
`audit_independent.py` → `AUDIT_INDEPENDENT.json`.

**One skill (the transition contract), one base, one donor pair. No quality, retention or release
claim.**

## Headline — the registration requires a failure to be stated first

**C3 FAILED. The claim that adding dOpallo at full strength costs at most 10% of Titan-Sparse's
skill is not confirmed.** ADD-100 keeps **413 of SPARSE's 447** primary-correct rows (92.4%). The
registered one-sided 95% lower bound of (ADD-100 − SPARSE) is **−49 rows** (bootstrap), below the
**−44.7-row** margin. One-sided p = 0.11, and Holm does not reject. A non-bootstrap check agrees:
the normal-approximation lower bound is −48.4 and p = 0.11. The two-sided 95% interval puts the
loss at **3.8%–11.4%** of SPARSE's score.

The development result does not replicate as stated. FRESH-DUAL-SKILL-001 reported interference
of −5 rows, "not distinguishable from zero" (p = 0.33), and 95.7% retention on clean rows. On this
fresh, content-disjoint bank the interference is **real: 56 rows lost, 22 gained, net −34,
p = 1.5e-4**. The development point estimate (4.3% loss) is inside the new interval, but the
interval extends past the 10% margin.

This is **not confirmed, but also not refuted**. The point estimate (7.6%) is below 10%, and the
data cannot place the loss above 10% either. It is also **not "an underpowered test"**. The pre-read
power (0.90) was conditional on the development estimate. The observed loss is nearly twice that
estimate, so the failure mainly reflects a larger effect than development suggested. The
extension to 864 rows was designed to give the test room to pass, and it did not pass.

## Registered verdicts

| limb | registered rule (n = 864) | observed | verdict |
|---|---|---|---|
| gates G1/G2/G3 | BASE ≤ 172; SPARSE ≥ 346; separation ≥ 260 | 4; 447; 443 | **PASS**: bank accepted |
| **C1** attenuation | SPARSE − AVG ≥ 10% of SPARSE, Holm | **116 rows (26%)**, lost 139 / gained 23, p = 2.0e-21 | **PASS** |
| **C2** addition beats averaging | ADD-100 > AVG, Holm | **+82** (331 → 413), lost 33 / gained 115, p = 7.8e-12 | **PASS** |
| **C3** interference small | lower 95% bound of (ADD − SPARSE) > −44.7, both readings | −34; lower bound −49; one-sided p = 0.11 | **FAIL** |
| C4 convention (descriptive) | ADD-100 ≥ 50% trailing periods; SPARSE, AVG ≤ 2% | 831 / 0 / 0 of 864 | pattern met |
| C5 location (own α) | SPARSE_ATTN − RAND_MASK ≥ +30 and p < 0.05 | **+44**, lost 4 / gained 48, p = 1.3e-10 | "location effect replicates", **but read it through C6** |
| C6 mask vs expectation (own α) | \|RAND_MASK − SCALAR_P\| ≤ 18 | **0** (279 vs 279; 1 lost / 1 gained) | **mask acts like its expectation** |

Primary-rule scores (strip whitespace, then at most one trailing `.`): BASE 4, SPARSE 447,
AVG 331, ADD-100 413, SPARSE_ATTN 323, SCALAR_MATCHED 373, RAND_MASK 279, SCALAR_P 279.
Strict scores are identical except ADD-100 = 13.

## What is confirmed

**The practical rule is confirmed: never average task vectors; give each donor its own full
coefficient.** Averaging costs 26% of the donated skill. Full-strength addition recovers 82 of
those 116 rows, and attenuation is 3.4× the interference cost. This is now a registered result on
a fresh, content-disjoint bank read once. It is no longer only an explanation.

**What is not confirmed: that the second donor is free.** Adding a donor with **no identified
skill** and twice dSparse's norm (‖dOpallo‖ 1.357 vs ‖dSparse‖ 0.694) costs roughly **4–11%** of
the first donor's skill. All of that cost is content errors, as the audit below shows.

## C5 must be read through C6, as the amendment required before the read

AMENDMENT_1 stated: *"C5 cannot be read without that anchor."*

- **C6 establishes the anchor unusually cleanly.** RAND_MASK (a Bernoulli mask keeping 40.2% of
  dSparse's elements, norm 0.4401) and SCALAR_P (0.402·dSparse, norm 0.279) produce **identical
  text on 856 of 864 rows**. Only 2 rows differ in correctness. The unstructured part of the mask,
  sqrt(p(1−p))·‖dSparse‖ ≈ 0.34 of norm spread over 60% of the elements, did **nothing
  measurable**.
- **So C5's control is effectively a 0.40× uniform scalar,** not "random elements at matched
  strength".
- **At the same norm, uniform shrinkage beats attention-only.** SCALAR_MATCHED (0.634·dSparse,
  norm 0.4401) scores 373, and SPARSE_ATTN scores 323: **−50 rows**, lost 61 / gained 11,
  p = 1.5e-9.

The ordering at norm 0.4401 is uniform scalar (373) > attention-only (323) > random mask (279),
and the random mask equals the 0.40 scalar (279). **There is no evidence that attention is a
privileged place to put this delta.** The registered C5 verdict is correct as a statement about
its own contrast ("attention beats random elements of equal norm"). But what it measures is
effective magnitude, not location. Descriptively, attention-only scores like a uniform scalar of
about 0.51, by linear interpolation between the 0.402 and 0.634 scalars.

**New mechanistic result (C6, registered): on this skill, a random Bernoulli mask of a task vector
acts like the task vector scaled by its keep rate.** That is the premise of drop-and-rescale
merging (DARE). It *predicts* that the mask followed by a 1/p rescale would recover about SPARSE's
score with 60% of the delta's non-zeros removed. That prediction is **untested**.

## Audit (independent of the registered code)

- **A. Recompute.** Every registered number was reproduced from the raw predictions by a separate
  implementation: per-arm counts, all discordant pairs, p-values, Holm, periods, the secondary.
  **0 mismatches.** Each arm's strict count equals the scorer's own `correct` field.
- **B. Robustness.** C3 fails under both the bootstrap and the normal approximation. The secondary
  (the original 288 rows under the original thresholds) fails C3 more strongly: −20 of 149, 86.6%
  retained, registered lower bound −30 against a −14.9 margin.
- **C. Read the text.** I checked whether ADD-100's losses are a scoring artifact, and they are not.
  - Of the 56 rows ADD-100 loses relative to SPARSE, **0 are formatting residue**. No
    whitespace-normalised, period-stripped output equals the reference.
  - **52 are wrong values**: 34 JSON answers with one field wrong, 1 with two fields wrong, 17
    wrong integers (14 off by more than 2).
  - **1 is structurally malformed**: every field is repeated. It parses to the right values only
    because the JSON parser keeps the last duplicate.
  - **3 return the unchanged input state.**
  - No ADD-100 output has a double period, and none contains whitespace.
  - The 22 rows ADD-100 gains are single-value corrections of SPARSE's errors. dOpallo perturbs
    the arithmetic in both directions, at a net cost.
  - **39 of the 56 lost rows are also lost by AVG.** In 19 of the content errors, ADD-100 and AVG
    produce the identical wrong answer. The interference falls mostly on rows that are fragile to
    any perturbation.
- **D. Subgroups** (descriptive, post hoc; they do not change any verdict).
  - Loss by family: ledger_replay −15/131, prerequisite_plan −17/155, register_machine −2/161.
    Family heterogeneity permutation p = 0.074.
  - By bank part: original V3 rows −20/149 (13.4%), extension rows −14/298 (4.7%), permutation
    p = 0.038.
  - Both parts come from one generator with different seeds, so they are exchangeable by
    construction, and SPARSE scores exactly 51.7% on each. The part difference is a nominal result
    among two post-hoc tests and does not survive a multiplicity correction. **It must not be used
    to favour either part.** It does mean the interference estimate is less stable than its
    interval suggests.
- **E. Provenance.**
  - All four raw files are on the committed bank hash, with an identical runtime: python 3.11.15,
    torch 2.6.0+cu124, transformers 5.14.1, FP32, greedy, batch 1, `max_new_tokens` 48, the
    development environment.
  - Every arm has exactly the bank's 864 ids.
  - Preflight re-hashed all six existing arms against their build records: 6/6 match.
  - RAND_MASK and SCALAR_P re-hashed now: both match their BUILD.json. They share p = 0.40201196,
    and SCALAR_P's record points at RAND_MASK's hash.
  - In the log, the gates were analysed (`ALL_PASS`) before any arm was built, and the lease was
    released after the last arm.
  - The analysis, driver, builders and bank on disk are byte-identical to the branch tip.

## Consequences for earlier claims in this repository

- `fresh_dual_skill_instrument_v1/RESULTS.md`: "interference is not distinguishable from zero" and
  "attenuation is 6.6× interference" **do not replicate**. On the fresh bank, interference is
  −34 (p = 1.5e-4) and the ratio is 3.4×. The ordering (attenuation ≫ interference) replicates.
- The period-convention account replicates: 831/864 trailing periods, 0 for SPARSE and AVG, and
  strict 13 vs primary 413. But **the convention is not the whole story**. Under the tolerant rule a
  real content cost remains.
- `module_disjoint_composition_v1`: SPARSE_ATTN < SCALAR_MATCHED at matched norm replicates (−50;
  development −19 on V2).

## Limits

One skill, one base, one donor pair, one decode budget. Opallo has no identified skill, so nothing
here measures a second skill arriving. The loss measured is the **cost of adding a skill-less delta
of this norm**. Retention of Supa's own skills was not measured. The bank is now spent.
