# Is there an Opallo skill to preserve? — O-SKILL-DEFINABILITY-001

Date: 2026-09-23. **Zero model loads, zero forwards.** CPU-only analysis over recorded predictions.
Four analyses; each analysis's top three claims were re-derived independently by two verifiers
(a *reproduce* lens and a *premise* lens), 24 verdicts in all. Scripts and structured results are
in `o_definability/`. Every number below was produced by those scripts in this attempt.

## Why this was asked

A composition plan for `W0 + α·dSparse + β·dOpallo` asks for an O-bank and a preservation ratio
`P_O = (M_merge − M_base) / (M_Opallo − M_base)`. `ARITHMETIC_LANE.md` specifies how to build a
fresh arithmetic bank for "Opallo's donated skill". Before building it, check whether Opallo has a
skill of its own to preserve, on that instrument or any other. A ratio needs a denominator that
exists, and that belongs to Opallo rather than to any delta.

## Answer: not on arithmetic, and no Opallo-specific skill is established anywhere

### 1. Opallo never trained on arithmetic

- Checkpoint `915af77f` (the `OPALLO` arm's source) took **41 optimizer steps** on 41 rows,
  5,233 tokens. The rows are the first 41 of `Random(17).shuffle(splits_large/train.jsonl)`
  (sha `cd57e650`); the harness replay reproduces them, and the seed-18 negative control fails.
- The rows span 24 conversational reply classes, 1–5 rows each. The 10,000-row pool has **0 rows**
  of `percentage_reasoning`, `ratio_proportion` or `time_duration`.
- The loss was **full-sequence CE**. Only **7.49%** of supervised positions were the reply.
- Opallo's only recorded in-domain gain is dev CE 5.578 → 2.202 nats. **At least 94.0%** of that
  gain is on non-reply positions (system prompt, user turns, template). This is a bound computed
  from the recorded numbers, because reply CE cannot fall below 0.
- (The "28 updates" figure often quoted belongs to Titan, the RRP-015 control, not to Opallo.)

### 2. The panel gain is shared by every SFT delta on Supa

On the 150-row panel (probe `5f6df1c1`, scorer `retention.py` `c52b6a48`, correct sets reproduced
**by identity**), the rows Opallo fixes are, above chance, the rows every other delta fixes:

| donor (gained set vs Supa, permissive) | ∩ Opallo-FP32's 33 | expected | hypergeom. p |
|---|---:|---:|---:|
| Titan dense `int001_a100` | 19 / 31 | 9.9 | 5.1e-5 |
| stored Titan `rrp015_control` | 20 / 30 | 9.6 | 3.0e-6 |
| **Titan-Sparse `int001_s25d`** (a28629c9) | **21 / 31** | 9.9 | **7.8e-7** |
| **H16 — abstention/clarification rows only** | **21 / 29** | — | **1.1e-7** |

- H16 was trained only on abstention/clarification rows, and it gains **all 12** rows that every
  Opallo seed and every Titan draw gains.
- Titan-Sparse scores 72/150 here and Opallo FP32 scores 74. The panel cannot tell the two donors
  apart.
- Titan-Sparse has full 150-row predictions on disk, in
  `sparse001-20260917/eval/s25d-retention/retention_predictions.json`. An earlier note said it had
  counts only; that premise was wrong.

**A donor-specific part exists, but it is small.** Mean gained-set Jaccard is 0.466 across the 12
Titan × Opallo pairs, against 0.649 between Opallo seeds and 0.661 between Titan draws. The exact
label-permutation p is 1/35, the floor this design allows.

The Opallo-only residue is **5 rows** permissive and 3 strict: rows gained by at least 3 of 4
Opallo seeds and 0 of 3 Titan draws. Seed-to-seed discordance on this panel is 15–24 rows. It is
also **confounded with probe device**: every Opallo draw ran on CPU, and every Titan, D1 and H16
draw ran on CUDA.

The hypergeometric null overstates the overlap, because Supa's wrong rows are not equally fixable
(51 of 103 are never fixed by any of the 7 draws). The within-recipe versus cross-recipe
comparison is the one that matters, and it gives the same conclusion.

### 3. About two-thirds of Opallo's typed-scorer gain is answer format

Opallo gains 32 rows and loses 6, a net of +26 (BF16, permissive).

- **Row by row:** 9 of the 32 gains are **FORMAT**. On those rows Supa already stated the right
  value, but in a form the grammar rejects: `'4:50 PM + 45 minutes = 5:35 PM'` → `'5:35 PM.'`,
  `'It ends at 10:15 AM.'` → `'10:15 AM.'`. 22 are **VALUE** gains, and 1 is non-termination. All
  6 losses are VALUE.
- **Format-blind:** the question is whether the correct value appears anywhere in the output. The
  extractor is built only from the scorer's own parsers. Positive controls: 150/150 golds, and
  1,500/1,500 golds wrapped in synthetic format damage. Negative controls: 0–4/141 fires on
  deranged targets.
  - On that measure Supa scores 65/150 and Opallo 74, a net of **+9**.
  - Discordant rows are 22 vs 13, **two-sided exact sign test p = 0.18**. Under the typed scorer
    they are 32 vs 6, p = 2.4e-5.
- **Reconciliation:** scorer net 26 = value net 9 + 17 format rows. Supa has 18 right-value rows
  that the grammar rejects; Opallo has 1.
- **Hidden regressions:** Opallo turns 7 of Supa's right-value-in-prose rows into terse **wrong**
  values. Most come from a systematic **+60-minute clock error**: the whole output parses to
  exactly target + 60 min on 8 of 31 clock rows for Opallo, against 1 for Supa.

The format/value boundary for compound durations such as `'1 hour and 25 minutes'` (target
25 min) is a judgement call. The primary classification follows the scorer's own semantics.
Counting those rows as format instead raises the row-level format share to 57.7%.

### 4. What Opallo does carry: a terse, period-terminated reply

- On the panel, Opallo ends **150/150** answers with `.`; Supa ends 43/150. Mean length is 6.7
  characters against 11.1.
- Its 41 training replies have a median of 5 words and end `.` 34 / `?` 6 / `!` 1.
- This is the one Opallo behaviour measured to survive composition. On `T_BANK_V2` the
  trailing-period count runs BASE 0, SPARSE 0, AVG 0, OPALLO 79, ADD-O75 63, **ADD-100 178**.
  That is what drives ADD-100's strict 6/192 against its period-tolerant 97/192.

## Consequences

1. **P_O on arithmetic is not definable as Opallo preservation.** A soup containing Titan-Sparse
   shows Opallo-like panel gains with no Opallo contribution. The Opallo-specific denominator is
   about 5 rows, inside the seed floor.
2. **An S×O (transition × arithmetic) joint bank is ill-posed for this pair.** There is no
   Opallo-specific arithmetic component to compose. The only prior joint bank, COMPOSE-REQ-001
   (transition × abstention), scored 0/64 on every arm while its oracle scored 64/64.
3. **A new broad arithmetic bank is not recommended.** `ARITHMETIC_LANE.md`'s plan would build an
   instrument whose "Opallo high" gate Titan-Sparse also passes. The v3 generator's percentage and
   counterfactual payload spaces are fully used by existing files, so a disjoint bank would need a
   declared distribution shift. BINDING-DIAGNOSTIC-001's fresh bank is floored for Supa (4/96
   strict anchors; 0/64 numeric-only ratio), is admitted for no Opallo or Titan read, and has no
   grant.
4. **The only definable O-endpoint left** is reply-content teacher-forced NLL on the **6,741**
   never-consumed rows of Opallo's 24 classes.
   - It is disjoint from Titan's data on all channels, and it needs one forward pass per row.
   - Its denominator has never been measured. Recorded data bracket it at about 0.03–0.43
     nats/token.
   - It is gate-able only behind four controls:
     1. **Denominator gate:** the Opallo−Supa paired gain clears seed and BF16/FP32 cast noise.
     2. **Specificity:** Titan-Sparse alone gives P_O ≈ 0.
     3. **Content, not prior:** matched gold minus within-class mismatched gold is above 0, because
        answers are heavily canned (`'yes.'` alone is 239 of 6,741).
     4. **Dose:** P_O is monotone along the existing β ladder.
   - If the denominator gate fails, Opallo has **no measurable held-out skill of its own**.

## Limits

- This covers one panel (n = 150), 4 Opallo seeds and 3 independent Titan draws. The permutation
  p-values sit at their floors.
- Transforms of one delta (alpha, sparse, rank) are not counted as independent evidence.
- The probe-device confound (Opallo on CPU, the others on CUDA) cannot be sized from existing
  artifacts. It can only inflate the donor-specific part, not the shared one.
- None of this says Opallo is useless. It says the composition plan's O-side has no
  Opallo-specific quantity on any instrument that exists. A new instrument must establish one
  before P_O means anything.

## Reproduction

`o_definability/format_vs_value/` runs step1–step6 in order; `donor_overlap/` runs `overlap.py`,
`specificity.py`, `third_donor.py` and `h16_pairs.py`; `opallo_in_domain/` runs q0–q7; and
`bindiag_composereq/` holds its own scripts. Run everything with `python -B`. `retention.py` is
imported read-only by file path under a private module name, and its sha `c52b6a48` is asserted.
`WORKFLOW_RESULTS.json` holds every claim, control and verifier verdict. The one partial
refutation (22 of 24 verdicts not refuted) targets an interpretive clause in BINDING-DIAGNOSTIC-001's
**own** record, *"deltas shift the annotation convention, not the value"*, and no conclusion above
relies on that clause.
