# CONVENTION-LOC-001, step 1: registered predictions (exploratory, exposed bank)

Written 2026-09-24. **None of these arms had been built** when this was committed. Bank:
T_BANK_V2 (192 rows, sha256 `fc094136…32e01`). It is **exposed**, and 11 of its rows are
content-contaminated, so every result here is exploratory. Scorer: `score_arms.py`, greedy FP32
batch 1, `max_new_tokens` 48, run in the development environment (torch 2.6.0+cu124,
transformers 5.14.1). Builder: `build_beta_arms.py`. Its positive control requires the recipe at
β = 0.75 to reproduce ADD-O75 within 1e-6, and its partition check requires dOpallo to split
exactly into MLP and attention/layer-norm parts.

## Arms

| arm | construction | source of the prediction |
|---|---|---|
| ADD-O50 | SPARSE + 0.5·dO | my linear-logit model, `RESULTS_STEP0B.md` |
| ADD-O60 | SPARSE + 0.6·dO | same |
| ADD-O90 | SPARSE + 0.9·dO | same |
| H1 | SPARSE + dO\|MLP | the audit session, `FOLLOWUP_PROPOSAL.md` §2 |
| H1c | SPARSE + dO\|ATTN+LN | the same, as the localisation control |

Here dO = ADD-100 − SPARSE, the FP32 reconstruction to within about 1 ulp.

## Predictions

**β arms** (repeated verbatim from `RESULTS_STEP0B.md`, commit 96ebaaa5): trailing-period rows
3 / 12 / 127, bands [0, 8] / [7, 17] / [95, 159]. Per-row agreement between predicted and
generated periods ≥ 173/192. Period-tolerant score ≥ 92.

**H1 / H1c** (the audit session's P1–P3). They were written for n = 288. They are restated here
for n = 192 in the same proportions:
- **P1.** H1 emits a trailing period on ≤ 3% of rows (≤ 5). **Falsified** if > 10% (> 19).
  H1c: ≥ 50% (≥ 96) means attention and layer-norm are **sufficient** for the habit. ≤ 10% (≤ 19)
  means they are not. Anything between is inconclusive.
- **P2.** tolerant(H1) − tolerant(SPARSE) has a lower one-sided 95% paired-bootstrap bound above
  −10% of SPARSE's tolerant score (10,000 resamples, seed 20260924).
- **P3.** strict(H1) − strict(ADD-100) ≥ 30% of SPARSE's strict score, with exact McNemar
  p < 0.05.

**My additional prediction, from the linear-logit account.** If the period push is linear and
additive across modules, the two partial pushes should sum to the whole. Behaviourally, one arm
should carry nearly all of it and the other nearly none. **Registered:**
- H1c's period count ≥ 150. This is the whole push minus a small MLP share. On V2, ADD-100 has
  178.
- periods(H1) + periods(H1c) is within 178 ± 25.

If both H1 and H1c come out low, the push needs both parts together: a non-additive, AND-like
mechanism that would falsify the account. If both come out high, it is redundantly carried.

**Paired comparison noted from the audit session** (their `ADDENDUM_1.md`): at matched period
rate, does the module route (H1, 60.8% of dO's energy at full strength) keep more of Opallo's
held-out shift than the magnitude route (ADD-O50, 25% of the energy)? That needs the O-side,
per-token P_O endpoint. It is **not** scored here. The arms built here are the ones it would use.

## What this step can and cannot say

It can localise the imported convention to a module class, and it can test linearity in β on
fresh arms, both on one exposed bank. It cannot say what Opallo's own effect retains: there is no
O-side endpoint here. Any mechanism result would need a fresh bank to be confirmed.
