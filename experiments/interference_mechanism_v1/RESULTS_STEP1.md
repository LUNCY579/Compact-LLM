# INTERFERENCE-MECH-001, step 1: results (exploratory, exposed T_BANK_V3X)

Run 2026-09-25 inside the operator-authorised chain, under one GPU lease, with scorer `score_arms.py`
(development environment, `max_new_tokens` 48). Registered in `STEP1_PREREG.md` (2bc80704) before any
arm existed. Output: `STEP1_RESULT.json`. Reference arms are CONFIRM-001's recorded SPARSE (447) and
ADD-100 (413) on the same rows. Weights were deleted after scoring, and their sha256s are in
`arms/*/BUILD.json`.

## Registered verdicts

Bands: SAME is |diff| ≤ 18; HIGHER/LOWER is ≥ 30 with p < 0.05.

| id | arm | construction | score | vs | lost / gained | p | verdict | prediction |
|---|---|---|---|---|---|---|---|---|
| P1 | DARE-O50 | dS + M₀.₅(dO)/0.5 | 412 | ADD-100 413 | 3 / 2 | 1.0 | **SAME** | SAME: **held** |
| P2 | DARE-O10 | dS + M₀.₁(dO)/0.1 (‖Δ‖ = 4.35) | 415 | ADD-100 413 | 4 / 6 | 0.75 | **SAME** | SAME: **held** (the risky one) |
| P3 | DISJOINT-DARE | M(dS)/0.5 + (1−M)(dO)/0.5 | 409 | ADD-100 413 | 5 / 1 | 0.22 | **SAME** | SAME: **held** |
| P4 | SPARSE-125 | 1.25·dS | 448 | SPARSE 447 | 27 / 28 | 1.0 | SAME | no prediction |
| P5 | ADD-A125 vs SPARSE-125 | 1.25·dS + dO | interference −34 at α = 1 → **−21** at α = 1.25 | | | | I = **+13**, CI [−4, +30] | I ≥ +10 **and** CI excluding 0: **NOT MET** |
| P6 | ADD-A125 | | 427 | SPARSE 447 | 58 / 38 | 0.052 | **INCONCLUSIVE** | SAME or HIGHER: **not met** |

## What this means

**1. Drop-and-rescale (DARE) and disjoint-support merging are functionally plain addition on this pair.**
All three land within 4 rows of ADD-100, with only 5–10 discordant rows out of 864:
- **P2 is the risky case, and it held.** It drops 90% of Opallo's elements and rescales the rest ×10,
  adding unstructured noise of norm ≈ 4.1 (three times ‖dO‖).
- **P3 shows coordinate overlap is not what hurts.** Giving the two donors disjoint coordinates buys
  nothing.

These results extend CONFIRM-001's C6 ("a random mask acts like its expectation") from 0.34 of noise norm
to about 4. They contradict the premise that DARE/TIES-style overlap removal reduces interference, at
least for this pair. The interference is **functional**, a push on low-margin rows (steps 0 and 0c), and
it is invariant to how the delta is spread across coordinates.

**2. Boosting the skill donor partly offsets the cost, but not significantly.**
- Raising Titan to 1.25 changes nothing on its own (P4: 448 vs 447). There is a lot of row churn,
  27 lost and 28 gained, so the dose curve is flat by α = 1.
- With Opallo present, it cuts the interference from −34 to −21 (P5, I = +13). The CI [−4, +30] includes
  0, so the registered remedy is **not established**.
- Descriptively (not registered): ADD-A125 427 vs ADD-100 413 is +14, lost 21 / gained 35, p = 0.081.
- The boosted composite still sits 20 rows below SPARSE (P6, p = 0.052).

## Limits

One exposed bank, one donor pair. All results are exploratory; confirming any of them needs a fresh bank.
