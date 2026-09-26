# TRISKILL-001 — a third, generator skill onto Titan-Transition at its own coefficient

**Registered 2026-09-26.** This is written before δ_C is trained, before any composite arm exists,
and before any Stage A or Stage B bank is read. Phase 0 (`PHASE0_RESULT.md`, 49ebd957) selected
the skill. It supersedes `DESIGN_DRAFT.md`.

**Operator authorization, 2026-09-26, in session:** "Go with C, delete those arms, grant approved."
The grant covers the single δ_C training run in §2. Nothing here authorizes release, naming or
promotion. `quality_claim` is false and `release_permitted` is false.

## 1. Question and hypotheses

Titan-Transition (TT) is `Supa + dSparse + 0.25·P16(δ_H)`, sha `a4d73353…`. It carries two
confirmed skills: the transition contract (T) and boundary abstention (H). Can a **third,
independently trained generator skill**, C, be added at its **own** coefficient β, so that:

- **(i)** C arrives measurably;
- **(ii)** T is kept within 5%;
- **(iii)** H is kept within 5%;
- **(iv)** the composite does not start refusing answerable rows?

This is the programme's first three-skill composition. It is also the first test of a second
*generator* skill, which the 2026-09-21 synthesis names as the highest-value missing experiment.

- **Prior, stated before any data:** composition at an own coefficient worked for T+H (CONFIRM-002
  C1, C2, C4). A second donor with no skill at full strength cost 3.8–11.4% (CONFIRM-001). δ_C is a
  real skill, so the prediction is **pass** at a small β.
- **Named risk, from Phase 0:** TT's output convention (bare answers) conflicts with C's
  `{"answer": N}` contract. A format conflict can make a composite *look* like it failed. That is
  why content and format are separate endpoints (§4).

Orthogonality is **not** a hypothesis here. The records refute it as the operative variable
(synthesis §5, ARMPERP-001, DELTA-SUBSPACE-GEOMETRY-001). Geometry is reported descriptively (§7)
and never used to select or to explain a verdict.

## 2. The donor δ_C — one native training run

- **Source:** exact Supa 1.7 (`962abc5b…dd24b`), primary checkout path (gate 9).
- **Data:** SIF-001 TRAIN `data/train/atomic_C.jsonl`, JSON-contract variants only (full,
  counterfactual, distractor). That is 1,536 candidate rows.
  - The 512 `alternate_format` rows are excluded, so δ_C does not also learn tag/plain conventions.
  - SIF's world-hash split guarantees that no C training world appears in DEV, CAL or CHECK. The
    fresh bank (§5) excludes every SIF world by construction.
- **Selection:** the rows are sorted by id and shuffled with `random.Random(17)`. The frozen prefix
  is taken while cumulative processed tokens stay **≤ 54,172**, δ_T's processed budget. There is
  no search over lengths (the lane's rule).
- **Recipe.** Every qualified field is copied from δ_T's plan (`dose-D1-PLAN.json`):
  - blocks 8–27; objective `assistant_response_plus_eos`; MQ CpuAdamW, peak LR 1e-5;
  - WSD 20/60/20 to 0.1×; betas (0.9, 0.95); ε 1e-8; wd 0; clip 1.0;
  - FP32; one exposure; order seed 17; 2,048 tokens per update.

  **The dose therefore equals δ_T's and δ_H's** (≈ 28 updates).
- **Lane:** `ChatBubble-rineya-reasoning-posttraining-v1`, controller `train`. It is registered as
  `TRISKILL-001-RULES-SKILL` by a local commit that follows CLOCKFIX-001's pattern exactly.
- **Grant bounds:** 1 training launch plus 1 retry; ≤ 600,000 processed tokens across all attempts;
  ≤ 8 GiB new disk. No RL, teacher, cloud or release.
- **Artifact:** `trained_C_fp32.safetensors` (the blocks 8–27 tensors), with its sha256 recorded.
  `δ_C = trained_C − Supa(FP32)`.

## 3. Arms

Each arm is built in FP32 with a float64 per-tensor sum, then built, verified, scored and deleted
one at a time. BUILD.json keeps each sha256.

| arm | construction | role |
|---|---|---|
| BASE | Supa | floor |
| TT | Titan-Transition (published, `a4d73353…`) | the two-skill parent |
| C100 | Supa + 1.00·δ_C | the donor as trained |
| C25, C50 | Supa + β·δ_C | solo effect at each β |
| TTC25, TTC50, TTC100 | TT + β·δ_C | **the composites** |
| TTR | TT + R, where R is a Gaussian direction on δ_C's tensors, rescaled **per tensor** to ‖β*·δ_C‖ (seed 20260926) | norm-matched null: does *any* update of this size cost T or H? |

TTR is built only after β* is fixed (§6), and at β* only.

## 4. Endpoints: content and format are always separate columns

- **C (the new skill).** The primary endpoint is SIF-001 **semantic** correctness from
  `sif.scoring.score_answer`: the right value under any clean wrapper. Strict correctness (the
  `{"answer": N}` contract **and** the value) is a secondary format endpoint.
- **T.** The primary rule is exact match after stripping whitespace and at most one trailing `.`
  (CONFIRM-002's primary rule). Strict exact match is reported separately.
- **H.** The CONFIRM-002 `boundary_rule` (rule module `416cfb6a…`), applied to the generation, is
  the abstention decision on boundary rows.
- **Refusal on answerable rows (C4).** The same `boundary_rule` is applied to every generation on
  the T and C banks. On those rows an abstention is always wrong.
- **Decoding:** greedy, FP32, TF32 off, batch 1, `max_new_tokens` 48, Qwen3 template with thinking
  off, hermes runtime (3.11.15 / 2.6.0+cu124 / 5.14.1).

## 5. Banks

**Stage A (development, exposed):**
- SIF DEV+CAL `atomic_C`, 192 JSON rows. DEV was read in Phase 0; CAL has not been read.
- T_BANK_V3X (864).
- HSKILL_VIEWS_V2 (224; `ba1f2982…`).

**Stage B (confirmation).** Every Stage B bank is built by committed code **and committed before any
Stage A arm is scored**. Each is read **once**.
- **C_FRESH, 336 rows.** A new SIF-001 generator draw of atomic C (112 parent worlds × full,
  counterfactual and distractor), with a new seed.
  - It is rejected on any `world_sha256` or prompt hash in any SIF-001 split. **Positive control:**
    the same check flags all 128 SIF DEV C worlds.
  - Its floor (majority answer, echo-seed) is published from the keys at build time.
- **T_BANK_V5, 864 rows.** `make_t_bank_v4.py`'s oracles and sampler, seed 20260927.
  - It is rejected on any full payload, operative key or (before, answer) pair of the historical
    exclusions or of T_BANK, V2, V3X or V4. **Positive control:** the check flags all 864 V4 rows.
- **H_FRESH, 336 rows.** 48 per boundary class, taken from the unused pool with salt
  `TRISKILL-001-H-FRESH`.
  - It excludes VIEWS, the sealed CONFIRM (re-derived; the file is never opened), SOUP_V3 and
    δ_H's 648 training rows, each by the same checks and controls as `make_hskill_soup_v3.py`.
  - The measured supply is 149–682 unused rows per class. There are **no answerable rows** left in
    that pool, so C4 uses the T and C banks.
- **`HSKILL-CONFIRM-002` (sealed) is not read.**

## 6. Stage A rules (development; select β*, and can stop the experiment)

**Harness canary.** TT's Stage A scores must equal CONFIRM-002 Stage A's TH25, which is the same
bytes: V3X **444**, VIEWS boundary **86**. BASE must also reproduce Phase 0's DEV-C predictions
byte-for-byte. Any mismatch stops Stage A.

**A1, donor gate.** C100 − BASE on C semantic must be ≥ +30 of 192, with exact McNemar p < 0.01.
If it fails, **STOP**: training did not install C, and nothing is composed.

**A2, eligibility of each TTC(β):**
- T loss (TT − TTC(β) on V3X primary) ≤ 8 rows (≈ 2% of 444);
- VIEWS boundary loss ≤ 5 rows;
- g(β) = TTC(β) − TT on C semantic ≥ +20 of 192.

**A3, the pick.** β* is the **smallest** eligible β whose g(β) is ≥ 0.8 × the largest eligible
g. This is the plateau rule, never the argmax. If no β is eligible, **STOP**: no Stage B read, and
the result is "C does not compose onto TT at a free coefficient on the development banks".

**Power at Stage B sizes** is projected from the Stage A estimates and reported. It is **not** used
to change any Stage B rule.

## 7. Stage B limbs (one read; Holm over C1–C4 at family α = 0.05)

Arms: BASE, TT, TTC(β*), C(β*), TTR.

| limb | rule |
|---|---|
| **C1** T kept | one-sided 95% lower bound of (TTC − TT) on T_BANK_V5 primary > −5% of TT's T_BANK_V5 score (paired bootstrap, 10,000 resamples, seed 20260926; normal approximation as a robustness check) |
| **C2** H kept | one-sided 95% lower bound of (TTC − TT) boundary abstentions on H_FRESH > −5% of TT's H_FRESH count |
| **C3** C arrives | (TTC − TT) on C_FRESH semantic ≥ +34 rows (≈ 10% of 336) **and** exact McNemar one-sided p < 0.05 |
| **C4** no over-refusal | one-sided 95% upper bound of refusals(TTC) − refusals(TT), over the 1,200 T_BANK_V5 + C_FRESH rows, < +5 |

**Headline.** C1–C4 all pass means "a third skill arrives with no measurable loss of the first two".
Any fail is reported first, by limb.

**Descriptive, never a verdict:**
- P_C = (TTC − TT) / (C(β*) − BASE) on C_FRESH semantic (share of C's solo effect kept);
- the TTR contrasts;
- strict format columns and trailing-period counts;
- per-family T and per-class H;
- read-the-text samples of every lost row class (at least 5 per class);
- geometry: global and per-tensor cosine of δ_C with dSparse and P16(δ_H); norms and the
  ‖Σ‖² cross term; same-block write-out subspace collision (SUBSPACE method, with its Haar null).
  **Geometry does not enter any rule.**

## 8. Amendments and stops

- An amendment is admissible only for an execution failure unrelated to model outputs (for example
  a driver reset or a missing dependency). It must be committed before the affected read, and must
  keep every bank, arm, coefficient, margin and rule.
- No coefficient, arm or bank is added after Stage A is read. No Stage B row is looked at before
  the Stage B arms are all scored.
- If C1–C4 pass, naming is the operator's decision. This experiment promotes nothing.

## 9. Known limits, stated before the data

- One base, one draw of δ_C, one draw each of the existing donors.
- C_FRESH shares SIF-001's generator, templates and wrappers, so it tests held-out worlds, not new
  task types.
- Broad Supa retention (the n = 47 panel and long context) is not measured.
- TT is FP32. BF16 serving is not qualified for any arm.
