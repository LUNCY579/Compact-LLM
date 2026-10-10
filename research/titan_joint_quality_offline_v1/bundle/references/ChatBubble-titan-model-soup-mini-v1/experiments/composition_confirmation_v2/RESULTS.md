# CONFIRM-002 (soup lane): results

Run 2026-09-25 (Stage A 14:27–15:03Z, Stage B 15:03–15:53Z) inside the operator-authorised chain, one read of
each fresh bank. Registered in `PREREG.md` (779cd431) and `AMENDMENT_1.md` (e2d6b4d4), with banks committed at
51345bb5, all before any arm was scored.
- Registered output: `CONFIRM002_RESULT.json`.
- Independent audit: `audit_confirm002.py` → `AUDIT_CONFIRM002.json` (0 problems).
- Stage A: `STAGE_A_RESULT.json`.

**Arms:** Titan-Sparse (T, transition skill) + 0.25·P16(δ_H) (H, boundary/abstention skill), plus
full-strength δ_H as a contrast. One base, one T donor, one H donor. No release or quality claim. The sealed
HSKILL-CONFIRM-002 instrument was **not** read.

## Headline: three confirmatory limbs pass, one fails, and the separate C5 fails

**Confirmed, on fresh unexposed banks:** adding the boundary delta at its own coefficient, 0.25, to Titan-Sparse
**gives the model a new, measurable skill**: +67 boundary abstentions out of 336, p = 1.4e-13. It **costs nothing
measurable of Titan's transition skill**: +4 rows, with a one-sided lower bound of −2 against a −21.5-row
margin. It **does not make the model refuse answerable questions**: 109/109 attempts, unchanged. This is the first
confirmed answer in this lane to the operator's original question: *can a second skill be acquired with barely
any loss of the first?* **Yes, for this pair, at this coefficient.**

**Not confirmed:** that the composite keeps at least 75% of the boundary skill's solo effect (C3). The point
estimate is **86%** (67 of 78), but the one-sided 95% lower bound is **0.718**, below the registered 0.75.
PREREG §6 wrote that a C3 failure means "the H skill does not survive composition". That overstates what this
failure shows. The skill clearly **survives** (C2: +67, p = 1e-13). What is not shown is that ≥ 75% of its solo
size survives. The confirmed statement is ≥ 72% with 95% confidence.

**Separate question C5, also failed:** full-strength δ_H does **not** cost ≥ 10% of Titan's transition skill
under the registered period-tolerant rule. It loses only 7 rows (423 vs 430, p = 0.49) against a required 43.
Strictly scored it collapses to 260, because 342 of 864 answers gain a trailing period. As with Opallo, the
full-strength donor mostly imports **answer formatting**. The development expectation that full strength wrecks
Titan (COMPOSE-001: 235 → 139) came from different deltas (P16 δ_T), a different bank and a different
rule, and it does not transfer.

## Registered verdicts

| limb | rule | observed | verdict |
|---|---|---|---|
| G1–G3 | BASE ≤ 172, SPARSE ≥ 346, separation ≥ 260 (T_BANK_V4) | 5, 430, 425 | pass |
| G_H | H25 − BASE boundary ≥ +17, p < 0.05 | **+78** (82 gained / 4 lost), p = 5.8e-20 | pass |
| **C1** T kept | lower 95% bound of (TH25 − SPARSE) > −5% of SPARSE (−21.5) | +4 (9 gained / 5 lost), bound **−2.0**, Holm-significant | **PASS** |
| **C2** H arrives | TH25 − SPARSE boundary ≥ +17, Holm | **+67** (78 / 11), p = 1.4e-13 | **PASS** |
| **C3** H kept | lower 95% bound of P_H > 0.75, Holm | P_H **0.859**, bound **0.718**, p = 0.104 | **FAIL** |
| **C4** no over-refusal | lower bound of (TH25 − SPARSE) attempts > −5 | 0 (0 discordant rows), bound 0 | **PASS** |
| C5 (own α) | SPARSE − THFULL ≥ 10% of SPARSE, p < 0.05 | loss **7** (41 / 34), p = 0.49 | **FAIL** |

**Scores:**

| arm | T primary | T strict | boundary abstain | answerable attempt | answerable correct |
|---|---|---|---|---|---|
| BASE | 5 | 5 | 185 | 109 | 41 |
| SPARSE | 430 | 430 | 183 | 109 | 37 |
| H25 | — | — | 263 | 109 | 41 |
| **TH25** | **434** | 434 | **250** | 109 | 35 |
| THFULL | 423 | **260** | 249 | 109 | 32 |

## Audit

**A. Re-scoring from text.** Every boundary and answerable decision was re-derived from the raw generation
text with `boundary_rule`, and every transition answer with the primary rule. The stored flags and the
registered counts are reproduced exactly (0 problems).

**B. Robustness of the verdicts.**

| limb | re-check | result |
|---|---|---|
| C1 | independent bootstrap / normal approximation | lower bound −2.0 / −2.15; passes either way |
| C3 | different seed / delta method | lower bound 0.716 / 0.716; fails either way |
| C5 | recomputation | loss of 7 reproduced |

**C. Provenance.**
- One runtime across all Stage B files: python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, FP32,
  greedy, `max_new_tokens` 48.
- Both bank hashes match.
- All five arm hashes were verified at preflight, and the three new arms' BUILD hashes equal those
  preflight values.
- Stage A's harness canary reproduced the native probe byte for byte: 224/224 identical texts for both
  BASE and SPARSE.

**D. Reading the text.**
- TH25's 78 boundary gains are real behaviour changes, most often clarification questions. Examples:
  *"Set the speed to 20."* → *"What speed do you mean?"*; *"Natural."* → *"What natural way would that sound?"*
  Gains by class: clarification_request 20, uncertainty 19, boundary_setting 13, safety_boundary 12,
  insufficient_information 9, clarification 5.
- Of its 11 losses, the 3 knowledge_boundary rows are a **predicate artefact, not lost skill**. SPARSE says *"I
  cannot see your calendar."* (credited). TH25 says *"I need your calendar details."*, an equally appropriate
  boundary reply that the published regex does not credit.
- This bears on C3's narrow miss but does not change it: the registered rule governs.
- The 5 transition rows TH25 loses are single-value arithmetic slips.

**E. Question cores.** The registered core key produced 336 distinct cores out of 336 rows, so the descriptive
cluster bootstrap equals the row bootstrap. A coarser key (the first six words) finds only 13 rows on repeated
cores (329 unique). On this bank, core-level dependence is small, so the row-level intervals are not being
flattered by it.

## Stage A (development, exposed banks), for the record

- **Canary:** PASS (counts and texts identical).
- **Development estimates:**

  | measure | SPARSE | TH25 | THFULL |
  |---|---|---|---|
  | V3X transition | 447 | 444 | 437 |
  | VIEWS boundary | 70 | 86 | 78 |

- **Projected power at Stage B sizes:** C1 0.996, C2 1.0, C3 0.82, C4 1.0, C5 0.0. C5's zero power was
  visible before the read and is disclosed here. The registered design was not changed in response.
- **Fresh bank vs selection bank:** the boundary effect was larger on the fresh bank (+78 solo vs a
  transported ≈ +45).

## What this changes

- **The operator's original question has a confirmed positive answer for one pair.** Give each donor its own
  coefficient: the skill donor at full strength, the second skill's donor small (0.25 here). The composite
  then gains the second skill at no measurable cost to the first, and without over-refusal. With CONFIRM-001:
  - averaging costs 26% (confirmed there);
  - a skill-less large donor at full strength costs ~8% (not confirmed free);
  - a real second skill at its own small coefficient costs ~0% (confirmed here).
- **How much of the second skill's solo effect survives composition** is not pinned down: 86% point estimate,
  ≥ 72% confirmed. Part of the shortfall is the abstention predicate missing "I need your…" phrasings.
- **Full strength is not needed and does not help.** THFULL gains no more boundary skill than TH25 (249 vs 250)
  and imports a trailing-period format (342 rows). It does not, however, damage Titan's content under the
  tolerant rule (C5 failed).

## Limits

- One base (Supa), one T donor (Titan-Sparse), one H donor (COMPOSE-001's δ_H, as P16).
- The boundary pool is templated, so the result generalises to held-out rows of the same seven classes, not
  to new question types.
- Retention of Supa's broader skills (the n = 47 panel, long context) was not measured.
- Both fresh banks are now spent.
