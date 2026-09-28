# CONVENTION-LOC-001, step 0b: the imported period is a linear-logit conflict (exploratory)

**Zero forwards.** The input is FA-001's exact teacher-forced record (`TF_EXACT.json`, exposed V2
bank, 192 rows) plus the recorded generations. Script: `linear_conflict.py`. Output:
`LINEAR_CONFLICT.json`. This result is post-hoc on an exposed bank. The predictions at the end are
stated **before** the arms that test them exist.

## The quantity

At the convention site the model chooses between the period variant and the gold continuation:
`}.` vs `}` on the 128 JSON rows, and `.` vs EOS on the 64 integer rows. Both candidates sit at
one position, so m = log p(period) − log p(gold) is an **exact logit difference** (the normaliser
cancels), and m > 0 means greedy decoding emits the period.

## Results

**T2: linearity in β, out of sample.** Predict ADD-O75 (Supa + dS + 0.75·dO) from SPARSE (β = 0)
and ADD-100 (β = 1) **alone**: m̂ = 0.25·m(SPARSE) + 0.75·m(ADD-100).

| | predicted | measured |
|---|---|---|
| period rows, teacher-forced | 61 | 68 |
| period rows, generation | 61 | 63 |
| per-row sign agreement with TF | 185/192 | — |
| per-row agreement with generation | **184/192** | (TF of the arm itself: 187/192) |
| per-row correlation of m | **0.980** | — |
| curvature (measured / linear shift) | **1.02** | — |

Built from its endpoints, the linear model predicts **which rows** acquire the period at β = 0.75
almost as well as teacher-forcing the arm itself. Its weakest site is the JSON merge (0 predicted,
6 measured, of 128).

**T1: additivity across donors.** dOpallo pushes the margin by **+11.3 logits on BASE and +12.4 on
SPARSE** (per-row r = 0.86). The interaction is +1.1 logits, CI [+0.68, +1.52], which is **9.8%**
of the effect and slightly super-additive. On this site dOpallo's push is nearly independent of
whether Sparse's skill is present.

**T3: the curve this implies.** Predicted period rows by β (α = 1):

| β | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rows | 0 | 2 | 3 | 6 | 12 | 23 | 46 | 61 | 70 | 97 | 127 | 167 | 182 |

The onset is sharp because the host's margins cluster. Integer rows flip first (61/64 by 0.75);
the JSON merge flips late (above 0.85).

## What this means

Composition here behaves as **added logits**. The "collapse" of SPARSE's strict score under
ADD-100 is not interference with Sparse's skill: content stays additive within FA-001's registered
margin. It is a **second additive behavior that overrides the host's contract where its push
exceeds the host's margin.** That reframes the answer to "can we acquire a new skill with barely
any loss?". Additive composition keeps both behaviors. When the two behaviors write to **the same
output decision**, the stronger one wins, and **you can predict which rows will flip from two
endpoint forward passes, without building intermediate arms.**

A practical rule follows. Before choosing coefficients, teacher-force the host and the full-strength
composite on the host's contract sites, then read off the largest β that keeps the override under
a tolerance (here β ≈ 0.5 for ≤ 2%). This transfers only if the linearity in T2 holds for other
donors, which is untested.

## Registered predictions for step 1 (arms NOT built when this was committed)

Arms: `ADD-O50`, `ADD-O60`, `ADD-O90` = SPARSE + β·(ADD-100 − SPARSE), FP32, scored with
`score_arms.py` on T_BANK_V2 at `max_new_tokens` 48. Pass criteria, per arm:

| arm | predicted period rows | count band | per-row agreement (predicted vs generated) |
|---|---|---|---|
| ADD-O50 | 3 | [0, 8] | ≥ 173/192 |
| ADD-O60 | 12 | [7, 17] | ≥ 173/192 |
| ADD-O90 | 127 | [95, 159] | ≥ 173/192 |

The count band is pred ± max(5, 25% of pred). A second prediction follows from content
additivity: each arm's period-tolerant score is ≥ 92/192, which is 90% of SPARSE's 102.
**Falsified** if any count falls outside its band or any agreement falls below 173/192. A failure
at β = 0.9 while β = 0.5–0.6 passes would locate a nonlinearity in the high-β region, where the
JSON merge sits.

## Limits

One bank (exposed), one donor pair, one conflict site type. Teacher-forced margins, validated
against generation only at β ∈ {0.75, 1}. "Additive logits" is shown at the convention site. At
content positions FA-001 records a real 6–10% interference (47% in `prerequisite_plan`).
