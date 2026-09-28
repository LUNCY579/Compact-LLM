# CONFIRM-001 — Amendment 1 (pre-read)

Written 2026-09-24, **before any arm was scored** on T_BANK_V3 or on the bank this amendment creates.
No CONFIRM-001 prediction file exists. The read budget is still ONE and is unspent.

PREREG.md forbids regenerating the bank or adding an arm **after any score has been read**. No score
has been read, so both changes below are inside the registration window. They are committed before
the extension rows are generated and before the driver runs.

## Why: the registered design could not reach its own bar

A pre-read power check (`power_preread.py`, output in `POWER_PREREAD.txt`) resampled the exposed
development bank (V2, 181 clean rows) to the fresh bank's size and applied the registered decision rule
exactly, using the functions in `analyse_confirm.py`.

| n | P(gates) | P(C1) | P(C2) | P(C3) | P(C1 ∧ C2 ∧ C3) |
|---|---|---|---|---|---|
| 288 (as registered) | 1.000 | 1.000 | 1.000 | **0.519** | **0.519** |
| 576 | 1.000 | 1.000 | 1.000 | 0.764 | 0.764 |
| 720 | 1.000 | 1.000 | 1.000 | 0.840 | 0.840 |
| **864 (amended)** | 1.000 | 1.000 | 1.000 | **0.899** | **0.899** |

At n = 288, C3 (the interference margin: ADD-100 keeps at least 90% of SPARSE's skill) passes
with probability about 0.52 **even if the fresh bank behaves exactly like development**. That makes
a C3 failure uninformative. It could not tell "interference exceeds 10%" apart from "the instrument
could not see 5.7 points of headroom at this n". The development point sits at 95.7% retained,
while the margin sits at 90%.

The mistake was mine, made at registration: I never asked whether the manipulation could clear
the bar (the memory rule "Q7 CAN THE MANIPULATION REACH THE BAR"). The PREREG sentence "288 rows
resolve effects of roughly 5% of SPARSE or more" was never computed. Correcting a sample size before
any data exists is ordinary design, not a rescue. The power figures are conditional on the
development point estimate and **do not propagate its uncertainty**, so they are optimistic if
the true retention is nearer 90%.

## Change 1 — the bank grows to 864 rows

- `T_BANK_V3X.json` = T_BANK_V3's 288 rows **verbatim and in order** (sha256 `20597c22…c7507a`,
  seed 20260924), followed by **192 new rows per family** (576 in all). The new rows use the same
  recorded-marginal sampler with seed **20260925** (`make_bank_ext.py`).
- New rows are rejected on the same three keys as before (full payload, operative key,
  (before-state, answer) pair). The exclusion set is the same one plus **every T_BANK_V3 row**.
  After generation the collision count is recomputed and must be 0 on all three keys. A
  **positive control** requires the operative-key check to flag all 288 V3 rows.
- n = 864 = 3 × 288, so the family balance (288 per family) and every proportional threshold carry over.

## Change 2 — thresholds are restated for n = 864

| rule | registered (n = 288) | amended (n = 864) |
|---|---|---|
| G1 base floor, BASE ≤ 20% | ≤ 57 | ≤ **172** |
| G2 donor ceiling, SPARSE ≥ 40% | ≥ 115 | ≥ **346** |
| G3 separation, ≥ 30pp | ≥ 86 | ≥ **260** |
| C1, C2, C3 | proportional (10% of SPARSE; direction; 10% margin) | unchanged |
| C4 | proportional (≥ 50%, ≤ 2%) | unchanged |
| C5 effect band | ≥ 10 rows and p < 0.05 | ≥ **30** rows and p < 0.05 |
| C5 null band | \|diff\| ≤ 6 | \|diff\| ≤ **18** |

The original gates were floored: 115 is 39.9% and 86 is 29.9pp. The amended row thresholds satisfy the
stated percentages exactly (floor for the upper bound G1, ceiling for the lower bounds G2 and G3).
The C5 bands scale by n/288 = 3, so "effect" and "null" keep the same meaning as a fraction of rows.

**What governs:** the full 864-row analysis. The original 288 V3 rows are **also** analysed under
the original thresholds and reported as a descriptive secondary. That reading is reported whatever
it says, and it does not govern. Nothing about the original design is hidden. It sits beside the
governing result with the power figure above.

C3's registered bound (5th percentile, one-sided 95%) governs as committed. The 2.5th percentile is
added as a descriptive number and does not govern.

## Change 3 — one new arm, SCALAR_P, and a separate question C6

**Motivation.** RAND_MASK keeps each non-zero element of dSparse with probability p ≈ 0.402. Its
expectation is exactly E[M ⊙ dSparse] = p·dSparse. So RAND_MASK is **not** "dSparse at 63% strength"
the way SCALAR_MATCHED is (both have norm 0.4401). It is p·dSparse, which has norm 0.28, plus
unstructured noise of norm sqrt(p(1−p))·0.694 ≈ 0.34. Random perturbations usually move a network far
less than structured ones of equal norm. A first-order account therefore predicts
**RAND_MASK ≈ Supa + 0.402·dSparse**. If it holds, C5 compares the attention component with an
effectively 0.40× scalar, not with "random elements at matched strength". C5 cannot be read without
that anchor.

| arm | construction | status |
|---|---|---|
| `SCALAR_P` | Supa + p·dSparse, where p is read from RAND_MASK/BUILD.json after RAND_MASK's weight hash is re-checked | **to build**, after RAND_MASK, under the same lease |

**C6 (separate question, its own α = 0.05, not in the Holm family).** RAND_MASK − SCALAR_P:
- **mask acts like its expectation:** |diff| ≤ 18;
- **mask exceeds / falls short of its expectation:** diff ≥ +30 / ≤ −30 and p < 0.05;
- anything else is **inconclusive**.

No directional prediction is registered for C5 beyond its original outcomes. Development puts
SPARSE_ATTN (62/181) below SCALAR_MATCHED (81/181, α = 0.634), and where an α ≈ 0.40 scalar lands on
this skill has never been measured.

## Change 4 — pre-flight checks added to the driver

Before any scoring, `run_confirm.py` re-hashes all six existing arms against their BUILD.json
`weight_sha256` and writes `PREFLIGHT.json`, including the torch TF32/precision flags that the
scoring subprocesses inherit. Any mismatch stops the run before a single row is read.
`analyse_confirm.py` now also refuses any run file whose decode budget is not `max_new_tokens = 48`
(memory rule: a bank does not identify the decode budget).

## Unchanged

Arms BASE, SPARSE, AVG, ADD-100, SPARSE_ATTN, SCALAR_MATCHED and RAND_MASK; the primary and
secondary scoring rules; greedy FP32 batch-1 decoding at 48 tokens with thinking disabled and the
byte-exact template; the Holm family {C1, C2, C3}; the gates-first rule, meaning a gate failure
rejects the bank with no attempt 2; read budget ONE.

## Cost

About 3× the scoring time: 8 arms × 864 rows at roughly 0.6–1.3 s/row, plus two streaming builds,
for about 1.5–2 h under one GPU lease.
