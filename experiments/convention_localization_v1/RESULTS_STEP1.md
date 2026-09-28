# CONVENTION-LOC-001, step 1: results (exploratory, exposed V2 bank)

Run 2026-09-25 inside the operator-authorised chain (`composition_confirmation_v2/run_chain.py`), under one
GPU lease, with scorer `score_arms.py` in the development environment. Registered in `RESULTS_STEP0B.md`
(96ebaaa5) and `STEP1_PREREG.md` (3c5e125d) before any arm existed. Output: `STEP1_RESULT.json` (also
recomputed independently, earlier, into a scratch file; identical). Each arm's weights were deleted after
scoring, and its sha256 is in `arms/*/BUILD.json`.

## Registered verdicts, in registration order

**β arms: linear-logit period predictions.** Pass needs the count in band, per-row agreement ≥ 173/192,
and tolerant score ≥ 92.

| arm | β | predicted periods | observed | band | per-row agreement | tolerant | verdict |
|---|---|---|---|---|---|---|---|
| ADD-O50 | 0.50 | 3 | **0** | [0, 8] | 189 | 100 | **PASS** |
| ADD-O60 | 0.60 | 12 | **2** | [7, 17] | 180 | 100 | **FAIL** (count below band) |
| ADD-O90 | 0.90 | 127 | **148** | [95, 159] | **169** | 98 | **FAIL** (agreement < 173) |

The full period curve on V2 is 0 / 2 / 63 / 148 / 178 at β = 0.50 / 0.60 / 0.75 / 0.90 / 1.00. The linear model
got the **direction and the rough location** of the onset right, and it predicted β = 0.75 (61 vs 63) and
the β ≤ 0.5 safe zone (0 periods, where ≤ 2% was predicted). But the onset is **steeper than linear**:
too few periods at 0.6 and too many at 0.9. The true period margin is **convex in β**, so linear
interpolation from the endpoints is a first approximation, not a law. Titan's content skill is untouched
across the whole sweep: tolerant 98–100 against SPARSE's 102.

**H1 / H1c: the audit session's P1–P3, and my additive prediction.**

| test | result | verdict |
|---|---|---|
| P1 H1 (dO on MLP only): periods ≤ 5 | **0** | **PASS** |
| P1 H1c (dO on attention + layer-norm only): ≥ 96 sufficient, ≤ 19 not | **0** | **NOT SUFFICIENT** |
| P2 H1 tolerant vs SPARSE: lower bound > −10.2 | 102 vs 102, bound −3 | **PASS** |
| P3 strict(H1) − strict(ADD-100) ≥ 31, p < 0.05 | 102 vs 6: **+96**, p = 7e-26 | **PASS** |
| mine: H1c ≥ 150 periods; H1 + H1c within 178 ± 25 | 0; sum 0 | **FALSIFIED** |

## What this means

**The imported period is carried by neither module alone.** MLP-only Opallo gives 0 periods, and so does
attention-plus-layer-norm-only Opallo. Together they give 178. The convention needs **both parts at
once**: an AND-like interaction across modules, not a sum.

This falsifies my registered module-additive extension of the linear-logit account. It also
**corrects** MODULE-DISJOINT-001's P4 heading, "the output convention is carried by ATTENTION, not MLP". P4
measured MLP-only (0 periods) but never the attention-only half, so "carried by attention" was an
inference, and it is wrong. Both halves are necessary.

**A practical result from the audit session's H1.** Titan-Sparse plus Opallo's MLP half at full strength
keeps **all** of Titan's skill: 102/102 tolerant and 102 strict, against ADD-100's 6 strict. It imports no
period, while carrying 60.8% of dOpallo's energy. Whether H1 keeps Opallo's own held-out reply shift (P_O)
is **not measured here**. It needs the O-side endpoint (the audit session's ADDENDUM_1 comparison,
H1 vs ADD-O50).

## Limits

One exposed bank (192 rows, 11 content-contaminated), one donor pair and one convention. These are
exploratory. Nothing here is confirmed.
