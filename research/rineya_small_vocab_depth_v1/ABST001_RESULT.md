# ABST-001 — result: LC lowers the gold UNKNOWN likelihood on V4/20, in the teacher's direction; the sign follows depth

**Generated** by `write_abst_result.py` from the committed artifacts only: records `abst001-20260925/` `result/ABST_PRIMARY.json`, `result/ABST_H3.json`, the attempt record and the BOUNDARY marker (sha256s in `result/ABST_DESCRIPTIVE.json`). No number in this document was typed by hand.
**Run:** code 8486d817 (R1, harness 31/31, finalized chain), launched via `launch_abst.ps1`, lease abst001. BOUNDARY 20260926T065525Z. **GO/HOLD stays HOLD; no quality claim; no release.**

## 1. Registered outcome

| step | test | result | threshold | decision |
|---|---|---|---|---|
| 1 | H1, cell A (S20/V4), UNKNOWN rows | LC-worse **189** / better 67 (ties 0), p = 6.1e-15 | ≥ 145 | **LC_LOWERS_UNKNOWN** |
| 2 | UNKNOWN-row anchor vs T4 | closer **197** / farther 59, p = 7.6e-19 | ≥ 147 | TOWARD |
| 2b | known-row anchor vs T4 | closer **255** / farther 1, p = 2.2e-75 | ≥ 147 | rejects toward |
| — | over-suppression (plain < T < LC) | 32 / 256 rows | majority = ≥ 128 | no majority |
| H2 | | | | **TEACHER_CONSISTENT_DIRECTION** |
| 3 | Holm at 0.0125 | D+ 256/256 · B 189/256 less likely · C 256/256 more likely · dV 228/256 · dD 255/256 | 153 … 147 | rejected: D+, B-, C+, dV, dD |
| H3 | | | | **SIGN_FOLLOWS_DEPTH** |

**Size** (reported, no gate): the median per-row ratio NLL_lc / NLL_plain on UNKNOWN rows is at least ×1.397 (97.5% lower bound).

## 2. What it licenses (the registered R1 table, applied)

> **Pooled over the four RTG1-dev family pools, equally weighted, conditional on the fitted T:** LC makes the gold reference `UNKNOWN` **less** likely on V4/20 on most rows (189/256); **in prerequisite_schedule the majority went the other way** (64/64 LC-better). It moves V4/20 in the teacher's direction on both `UNKNOWN` rows (197/256) and known rows (255/256), without an over-suppression majority (32/256). This is consistent with moving toward a teacher that does not abstain on these pools. **Generic damage is excluded**: 0 known rows have NLL_plain < NLL_T (the registered count rule needs fewer than 147). **It does not restore abstention, and it is not shown to be teacher-specific**: a teacher-blind shift of mass from `UNKNOWN` to the digits passes the same tests. **HLCC's abstention caveat stays in force, reinforced**: the direction-only licence does not license abstention, and LC lowers the likelihood of the gold `UNKNOWN` answer.
>
> **H3 (SIGN_FOLLOWS_DEPTH):** at depth 20 the sign is LC-worse under V4 **and** V2 (and V0); at depth 18 it is LC-better under every vocabulary. The depth set carries A's sign, and changing depth alone reverses it. No recommendation changes, and no set is chosen.
>
> *Abstention where the teacher abstains (e.g. CC0) is not measured by any registered instrument.*

## 3. Refusal controls (all passed before the boundary)

- R4 fit canaries: S20 0.0, S18 0.0 (≤ 1e-6).
- R5 tier 1 (exact): A_plain 0.0, D_plain 0.0, T4 0.0, D_lc 0.0, F_lc 0.0 (≤ 1e-6).
- R5b tier 2 (B = 1 path): max 0.000123 (≤ 1e-3); d(D) sign disagreements 0 (reported).
- R1 structure_id: 512 distinct, overlap 0, controls {'used': True, 'train': True}.
- R6 projection 13.9 min (cap 60). W: down_proj restored. No earlier crossing.

## 4. What did not complete

- Greedy generation finished A-plain and A-LC, then raised `AcceleratorError: CUDA error: an illegal memory access was encountered` during the **teacher (T4)** generation. The attempt record classifies it 'deterministic or unknown' with primary_written True. Per §9 there is no rerun.
- The W check after A-LC generation copies weights to the CPU (a device sync), so an earlier asynchronous fault would have been raised there; the fault arose in T4 generation, after every measurement cell, ABST_PRIMARY and ABST_H3.
- **NOT_MEASURED:** T4 and T0 greedy outputs; forecast F9 is NOT_SCOREABLE. `ABST_RESULT.json` was not written; the registered descriptive block below is computed from the attempt record's stored values (§7 computes descriptives from stored per-row values; same function, no model).

## 5. Descriptive (registered readings; never a verdict)

| cell | UNKNOWN rows LC worse / better | UNKNOWN LC closer to T | known LC worse / better |
|---|---|---|---|
| A | 189 / 67 | 197 | 0 / 256 |
| B | 189 / 67 | 198 | 0 / 256 |
| C | 0 / 256 | 44 | 0 / 256 |
| D | 0 / 256 | 48 | 0 / 256 |
| E | 187 / 69 | 198 | 0 / 256 |
| F | 0 / 256 | 56 | 0 / 256 |

- **By family (cell A, UNKNOWN rows, LC worse / better):** counterfactual_state 64/0; evidence_chain 62/2; prerequisite_schedule 0/64; register_machine 63/1.
- **Geometry:** plain V4/20 is below the teacher on 220/256 (85.9%) of UNKNOWN rows and on 0/256 (0.0%) of known rows. Median gap closure on UNKNOWN rows 0.702; rows crossing the teacher in either direction 63 (over-suppressing, one direction: 32). Teacher median UNKNOWN NLL 8.09 nats.
- **M1:** TEACHER_TRACKING (nearly implied by the geometry; not independent evidence). **M2:** BIAS_SHIFT (AUROC plain 0.547, LC 0.558, teacher 0.558).
- **Row-level tracking (within-family partial Spearman; correlational, licenses nothing):** UNKNOWN 0.476 (TRACKING_SIGNAL; pooled 0.716; per family counterfactual_state 0.19, evidence_chain 0.39, prerequisite_schedule 0.61, register_machine 0.62); known 0.787 (TRACKING_SIGNAL). Rows move in proportion to where the teacher sits, within families; a row feature could still drive both, so this is not 'tracks the teacher'.
- **UNKNOWN token alone:** LC worse 189 / better 67; agrees with the two-token endpoint on 256/256 rows. **Gold digit alone (known rows):** LC worse 29 / better 227; digit-only anchor closer 217 / farther 39.
- **Gap cue:** 192 UNKNOWN / 114 known rows carry it; cue-only AUROC 0.652.
- **Greedy (4 tokens; T4/T0 NOT_MEASURED):** A-plain abstains on 22/256 UNKNOWN rows, false-abstains on 10/256 known rows, answers 11/256 known rows correctly; A-LC: 19/256, 8/256, **87/256**.
- **Vocabulary at depth 20 (UNKNOWN rows):** median output-support term -0.0194, input term 0.0000; tokenisation identity: V4_input_equals_V0 146/33 of 179; V4_input_differs 43/34 of 77.

**Wording rules applied (§7):**
- Rule 1: |AUROC_T4 − 0.5| = 0.058 > 0.05, so rule 1 does not force the substitution; **but** every AUROC here (0.547, 0.558, 0.558) is below what the literal gap cue alone reaches (0.652), so none is evidence of discrimination beyond reading the cue. This document therefore says 'UNKNOWN likelihood', not 'abstention', for every model property.
- Rule 2: the UNKNOWN token alone agrees with the endpoint on 256/256 rows, so 'UNKNOWN likelihood' (not 'reference-sequence likelihood') is used.
- Rule 2b: the digit-only anchor (217 closer / 39 farther) agrees in direction with step 2b, so the known-row statement may say 'the gold digit'.
- Rule 3: no 'restores abstention', 'abstention is safe', 'tracks the teacher' or 'teacher-specific' anywhere.

## 6. Forecasts (scored; not gates) — 8/9 scoreable hit

| # | forecast | observed | scored |
|---|---|---|---|
| F1 | H1 = LC_LOWERS_UNKNOWN | LC_LOWERS_UNKNOWN | hit |
| F2 | cell-A worse fraction >= 0.70 | 0.738 | hit |
| F3 | H2 = TEACHER_CONSISTENT_DIRECTION | TEACHER_CONSISTENT_DIRECTION | hit |
| F4 | prerequisite_schedule worse <= 0.50, every other family >= 0.70 | counterfactual_state 64/64, evidence_chain 62/64, prerequisite_schedule 0/64, register_machine 63/64 | hit |
| F5 | D+ rejected | D 256/256 more likely | hit |
| F6 | attribution = SIGN_FOLLOWS_DEPTH | SIGN_FOLLOWS_DEPTH | hit |
| F7 | M1 = TEACHER_TRACKING | TEACHER_TRACKING | hit |
| F8 | plain V4/20 NLL below the teacher's on >= 90% of UNKNOWN rows | 220/256 (85.9%) | **miss** |
| F9 | teacher abstains greedily on <= 5% of UNKNOWN rows (T4 and T0) | NOT_SCOREABLE: T4/T0 greedy NOT_MEASURED | NOT_SCOREABLE |
| F10 | in D, LC moves away from the teacher on the majority of UNKNOWN rows | 208/256 farther | hit |

## 7. Disclosures

- The run is R1 (71acefb8) plus a harness-forced two-line fix (1d2f5552), finalized at 8486d817; the cold reviewer re-checked R1 and the fix diff first-hand before launch.
- The descriptive block is computed after the run, from the attempt record, because the result file was never written (§4). ABST_PRIMARY's per-row values equal the attempt record's on all 512 rows (checked by this generator).
- Spent-data framing (§1 of the prereg) predicted most of this: F1–F7 and F10 follow from the spent geometry plus HLCC's lead, and TEACHER_CONSISTENT_DIRECTION was close to implied once H1 replicated. The new information is the replication on fresh rows, the family reversal's size, the depth attribution, and the descriptive greedy numbers.
