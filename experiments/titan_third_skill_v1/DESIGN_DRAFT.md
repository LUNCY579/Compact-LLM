# TRISKILL-001 — design draft (NOT registered; nothing below binds until PREREG.md is committed)

**Question.** Can a **third, independently trained generator skill** be added to Titan-Transition
at its **own** coefficient, so that it arrives measurably without measurable loss of the transition
skill (T) or the boundary skill (H), and without over-refusal?

This is the experiment `COMPOSITION_SYNTHESIS_20260921.md` names as the highest-value missing one:
a second **generator**, since δ_H is a gate skill. It is also the programme's first **three-skill**
composition ("never attempted").

## What the evidence licenses, and what it does not

| premise | status in the records | consequence for this design |
|---|---|---|
| give each delta its own coefficient; never average, never rescale the sum | confirmed (CONFIRM-001 C1/C2: averaging costs 26%; INDEP-001/002/004: separable coefficients) | new arm = `TT + β·δ_C`, with β chosen for δ_C alone |
| a second real skill at a small own coefficient is free | confirmed for ONE pair (CONFIRM-002 C1, C2, C4) | the thing to replicate with a third, different skill |
| orthogonality protects composition | **not supported**:
- δ_T vs δ_H has cosine 0.0027, yet same-skill draws overlap 0.60 and compose *better*;
- dSparse vs dOpallo has cosine 0.0025, yet the same block's write-out matrices (`o_proj`,
  `down_proj`) collide at 3–7× chance;
- ARMPERP made the deltas exactly orthogonal, and it bought nothing | geometry is recorded **descriptively only**, never used to select or explain |
| the trained magnitude is the right magnitude | **refuted** (δ_H's band [0.10, 0.50]; trained at 1.00 it was harmful; benefit saturates early) | β ladder {0.25, 0.50, 1.00}; plateau rule, never argmax |
| strict exact match reads content | **refuted** twice (Opallo's trailing period; THFULL 342 periods) | format and content scored in separate columns (SIF strict vs semantic) |
| Opallo has a skill to donate | **refuted** (O-SKILL-DEFINABILITY; OSIDE-001 NULL) | Opallo plays no role |

## Phases

0. **Source screen** (registered, `PHASE0_PREREG.md`). Candidate C or P, by headroom.
1. **Train δ_C** from exact Supa in the native lane, with the δ_T / δ_H recipe unchanged:
   - blocks 8–27; objective `assistant_response_plus_eos`; MQ CpuAdamW at LR 1e-5; WSD 20/60/20;
     FP32;
   - order seed 17; processed budget **54,172 tokens = 28 updates** (2,048 tokens per update),
     so the dose equals δ_T's and δ_H's;
   - rows: SIF-001 TRAIN `atomic_C`, JSON variants only (full, counterfactual, distractor). This is
     about 165 rows at a median of 290 tokens. The 512 `alternate_format` rows are excluded, so the
     delta does not also learn tag/plain conventions.

   Needs a lane registration commit, a grant, ≥ 23.03 GiB free disk and ≥ 7.4 GiB free RAM.
2. **Development.** Build and score one arm at a time, then delete its weights. Receipts keep the
   hashes.
   - Arms: `C(β) = Supa + β·δ_C` and `TTC(β) = TT + β·δ_C`, with β ∈ {0.25, 0.50, 1.00}.
   - Control: `TT + R`, where R is a random direction of norm ‖β*·δ_C‖. It asks whether any delta
     of that norm costs T or H.
   - Banks, all exposed and development-grade:
     - SIF CAL `atomic_C`, for C;
     - T_BANK_V3X (864), for T;
     - HSKILL-VIEWS-002 (224), for H;
     - SIF DEV `atomic_C`, for the Phase 0 comparison.
   - Rule (to be registered): β* = the **smallest** β whose C gain over TT is at least 80% of the
     largest C gain among the TTC arms, subject to losing ≤ 2% of TT's T score and ≤ 5 boundary
     rows.
3. **Confirmation**, on fresh banks built before any arm is scored, each read once:
   - `C_FRESH`: a new SIF-001 generator draw of atomic C (new seed, whole-world exclusion against
     every SIF split and against δ_C's training rows), about 336 JSON rows;
   - `T_BANK_V5`: 864 rows, from `make_t_bank_v4.py`'s generator with a new seed, excluding every
     V2/V3/V3X/V4 payload;
   - `H_FRESH`: 48 per boundary class (7 × 48 = 336) from the unused pool. The remaining supply is
     149–682 rows per class, measured 2026-09-26. There are **no** answerable arithmetic rows left
     in that pool, so over-refusal is counted on the T and C rows, which are all answerable.
   - Arms: BASE, TT, TTC(β*), C(β*).
   - Limbs (Holm over C1–C4):
     - C1: T kept, lower 95% bound of (TTC − TT) > −5% of TT;
     - C2: H kept, lower bound of (TTC − TT) boundary abstentions > −5% of TT's;
     - C3: C arrives, (TTC − TT) on C_FRESH ≥ the registered row threshold, p < 0.05;
     - C4: no over-refusal, i.e. refusals on T+C rows do not rise, lower bound > −5 rows.
   - Descriptive: P_C = (TTC − TT)/(C − BASE); semantic vs strict; geometry.
4. **Naming.** Only if C1–C4 pass. Any name is the operator's to assign; a composite is not
   promoted by this experiment.

## Operator decisions this draft needs

1. The skill (C unless Phase 0 says otherwise).
2. Disk: ~17 GiB must be freed before training. The candidates are rebuildable soup-lane arm weights
   (see the session report). No deletion without explicit approval.
3. Host RAM ≥ 7.4 GiB at launch; Unity currently holds about 2.5 GB.
4. A training grant: one native run, 28 updates, ≤ 600k processed tokens across all attempts,
   1 retry, bounds as in H-GRANT.json.
