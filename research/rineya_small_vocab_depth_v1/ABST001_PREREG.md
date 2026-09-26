# ABST-001 — does linear compensation lower the UNKNOWN likelihood on the recommended student, and does that move it toward the teacher?

**Status:** REGISTERED 2026-09-25 at **2f0a21f5**. **REVISED R1 on 2026-09-26, before any forward pass of the real teacher on the panel** (§13). R1 answers:
- the cold review (Rineya Delta session, records `abst001-20260925/ABST_PREREG_COLD_REVIEW.md`, 9a3399f);
- a peer review of the uncommitted R1 runner (soup lane).

- a four-lens review of the R1 revision itself, before its commit (§13, "R1 review round"). Every finding was checked first-hand against the code.

The panel was built CPU-only before registration and is pinned by hash. The original registered text is at 2f0a21f5. Every change R1 makes is listed in §13 with the finding it answers.
**Branch:** `research/rineya-small-vocab-depth-distill-v1` · **Records:** `rineya-dataset-retention-records/abst001-20260925/`.
**Parents:** HLCC-001 (`HLCC001_PREREG.md`, `HLCC001_RESULT.md`); SMALLVD-001 (the S6 frontier and LC_ARMS).

**Design provenance.** Three independent designs were produced (confirm-first, attribution-first, behaviour-first), each scored by three adversarial critics (statistics, hygiene, interpretation). No model ran. This document synthesises them:
- **Backbone** from confirm-first (scored 8/8/6).
- **Teacher anchor and paired contrasts** from attribution-first (6/6/7).
- **Behaviour-first** rejected: its primary sat on a census, and its harm tail was unreachable. Its greedy readout and its exclusion of designer-seen rows are kept.

## 1. Why

HLCC-001 confirmed linear compensation (LC) on the recommended V4_32768/20 student, in direction only. It left one post-hoc lead, which was not registered and not tested:
- **V4/20:** on 16 RTG1 abstention rows whose reference is `UNKNOWN`, LC made the reference **less** likely in 13 (median ratio NLL_plain/NLL_lc ×0.671).
- **The lead is family-split** (R1, cold review M4; recomputed first-hand from `HLCC_RESULT.json` and the RTG1 family labels):
  - prerequisite_schedule: 3/3 LC-better;
  - counterfactual_state, evidence_chain and register_machine: **13/13 LC-worse**.
- **18-block set (V2):** it made the reference **more** likely in all 16 (×1.665).
- **Why it can't be attributed:** the sets differ in both depth set and vocabulary, and HLCC stored one vocabulary per set.
- **HLCC's known-answer rows (R1, cold review M1; recomputed first-hand).** On the same V4/20 set, the 16 RTG1 known-answer rows were LC-**better 16/16** (median NLL_plain/NLL_lc ×3.266). On the 18-block set they were also 16/16 (×8.343). **LC is not generic damage on these rows.**

Spent design data change how the lead should be read. They come from SMALLVD-001's already-spent HOLDOUT/SELECT rows; nothing new was run.

- **The teacher does not abstain on RTG1.** Its greedy answer on the 32 RTG1 abstention rows with an `UNKNOWN` reference is a digit every time (27 of them "0"). It gets 9 of 32 known-answer rows right, near the 20% chance level for five options. RTG1's `UNKNOWN` is the dataset's gold label, not the teacher's behaviour.
- **Median NLL of `UNKNOWN`+`<|im_end|>`** on the 17 HOLDOUT `UNKNOWN` rows:

  | configuration | V0 | V2 | V4 |
  |---|---|---|---|
  | teacher, 28 blocks | 8.16 | 8.09 | 8.09 |
  | plain, 24 blocks | 2.16 | 2.16 | 2.15 |
  | plain, 20 blocks | 4.34 | 4.31 | 4.26 |
  | plain, 19 blocks | 6.95 | 6.74 | 6.55 |
  | plain, 18 blocks | 9.20 | 8.75 | 8.40 |
  | 18 blocks + LC | 4.76 | 4.76 | — |

  - **Vocabulary:** within about 0.1 nats at 20, 24 and 28 blocks. At 19 and 18 blocks, V4 sits 0.40 and 0.80 nats below V0: the vocabulary gap grows as depth is removed.
  - **Pruning:** at 20 blocks the plain student is far **more** `UNKNOWN`-prone than the teacher, on all 17 rows (T/plain from 1.14 to 2.85).
  - **Known rows:** the plain student sits **above** the teacher (T/plain 0.110–0.573 on 12/12).
- **On the 18-block set, LC moves the student away from the teacher.** The plain student sits near or above the teacher. LC lowers its NLL to 4.76 and is closer to the teacher on only **1 of 17** rows. HLCC's "18-block benefit" on `UNKNOWN` rows was, on this reading, movement **away** from the teacher.

So "LC lowers the `UNKNOWN` likelihood on V4/20" may be the student moving back toward the teacher. It may equally be a teacher-blind shift of probability mass from `UNKNOWN` to the digits: given the geometry above and HLCC's 16/16, **both** readings predict the same direction on both strata (§2, §6 D1). ABST-001 does three things:
1. It replicates the lead on fresh rows.
2. It asks whether LC moves V4/20 in the teacher's direction on **both** `UNKNOWN` and known-answer rows, without over-suppressing. Generic damage fails the known-row condition; a teacher-blind answer-type shift does not.
3. It attributes the sign across the depth × vocabulary corners.

## 2. Hypotheses

- **H1 (primary, cell A = S20/V4, `UNKNOWN` rows).** LC makes the teacher-forced NLL of `UNKNOWN`+`<|im_end|>` **higher** in more rows than lower. This is a claim about the **gold** reference: H1 means LC makes the correct answer less likely.
- **H2 (anchor, given H1).**
  - **Step 2:** on `UNKNOWN` rows, LC moves the per-row NLL toward the teacher's (T4, same vocabulary): |log NLL_lc − log NLL_T| < |log NLL_plain − log NLL_T|, in more rows than away.
  - **Step 2b:** the same holds on **known-answer** rows. There the geometry is reversed: the plain student is less confident in the gold digit than the teacher. So generic damage (NLLs up) moves away from the teacher there, as long as plain sits above the teacher on most known rows (§9 count rule).
  - **TEACHER_CONSISTENT_DIRECTION** (R1 name; formerly FIDELITY) requires both, **and** fewer than half of the `UNKNOWN` rows with defined values may **over-suppress** (NLL_plain < NLL_T < NLL_lc, meaning LC's `UNKNOWN` NLL passes above the teacher's).
  - **What the label can and cannot show (R1, cold review M1).** The label is **direction-consistent, not teacher-specific.**
    - A teacher-blind shift of mass from `UNKNOWN` to the digits raises NLL(`UNKNOWN`) and lowers NLL(gold digit). It therefore passes steps 1, 2 and 2b.
    - With the noisy null cell built for this (§6, D1), the label fires in **99.9%** of draws.
    - Given the spent geometry and HLCC's known-row 16/16, the label is close to implied if H1 replicates. It is forecast F3, not a test of fidelity.
    - What step 2b **does** reject is generic damage (most NLLs inflated, regardless of the teacher; D2: 0.0000), **but only because** the plain student sits above the teacher on the known rows. Under generic damage a known row can move toward the teacher only where plain is **below** it. With 60% of known rows reversed, generic damage earns the label in 47.2% of draws (D2r).
    - The guarantee on real data is therefore a **count rule** (§9). The generic-damage exclusion holds only if fewer than **147** known rows have NLL_plain < NLL_T; the count is stored as `plain_nll_below_teacher`. Below 147, pure generic damage cannot produce the 147 closer rows that step 2b needs.
    - A row-level tracking statistic is registered **descriptively** (§7). **No outcome licenses abstention.**
- **H3 (attribution).** The signs in B = S20/V2, C = S18/V4 and D = S18/V2, and two paired contrasts, say whether the depth set or the vocabulary carries the sign. D's sign is stated neutrally ("UNKNOWN more/less likely"), never as benefit or harm. H3 is **UNTESTED** unless H1 = LC_LOWERS_UNKNOWN and step 2 rejects in some direction. B, C and D are measured anyway and are reported descriptively in every branch (cold review #11).

**Population and scope (R1, cold review M4).**
- **What a claim covers.** Every claim is a **pooled** majority over the four RTG1-dev family pools, equally weighted (64 rows each), conditional on the fitted T. The per-family direction is reported beside every licence. If a family's majority goes the other way, the licence sentence names that family. There is no per-family test.
- **What it does not cover:** new generator families, RTG1's reserved holdout, or tasks where the teacher abstains. Under a family random effect, the row-level test falsely confirms for new families at a rate of 0.21 (§6 S1). That is why the claims stay scoped to the pools.
- **The fit set shares the families.** Both LC maps (S20 and S18) were fitted on SMALLVD's SELECT rows. SELECT holds **35 RTG1-dev rows** from the same four families:
  - prerequisite_schedule 14, evidence_chain 9, register_machine 8, counterfactual_state 4;
  - 15 with an `UNKNOWN` reference and 20 known;
  - the fit uses the teacher's S1 continuations, not the gold references.

  The panel excludes those ids and their structure_ids (§5 R1), and the claim is conditional on T. Even so, the fit set is not family-disjoint from the panel.

## 3. Panel (fixed; built; read once)

`abst001-20260925/panel/ABST_PANEL.jsonl`, sha256 `133725905b4c641aa316f6ddc7f4447e40a252769793d2b39ebed5dbdf09a17f`, 512 rows. The manifest is `ABST_MANIFEST.json` (sha256 `367b34a90f98c090743dc3b9ec0d2aa1ba17717bf52af02c2a9b5d8a9c60ceac`). The builder is `build_abst_panel.py` (sha256 `76d8b8119a68ddf506e446e1828f101711ee6dca89a8c3ff3db93a3facd384e1`), run under transformers 4.51.3 with `enable_thinking=False`.

- **Pool:** RTG1 dev only. RTG1 holdout is RESERVED and train is training data. CC0 is dropped (§11 D4).
- **Eligibility:**
  - the id is not in LAYER_PANELS, CONFIRM_PANEL or TOKENIZER_CORPUS (96 excluded);
  - not one of the 3 designer-seen rows;
  - the normalised last user turn does not equal any used row's or any RTG1-train row's (0 excluded). **R1:** this check's positive control passed by construction and could not fail (cold review #23). It is superseded by the run-time structure_id check with a real positive control (§5 R1).
  - not flagged by the six non-RCI instruments (0 flagged; exact-path 4/4, sentence-path 4/4, negative control clean). RCI-001/002 and HSKILL-CONFIRM-002 are never opened.
- **Sampling:** per family, `bc.ordered(ids, 'abst001:<stratum>:<family>')`, first 64. This is a stratified simple random sample without replacement from fixed pools.
  - `UNKNOWN` stratum: 256 rows, from pools of 112, 109, 109 and 102.
  - Known stratum (reference 0–3): 256 rows, from pools of 468, 462, 461 and 473.
- **Sealed reserve:** 176 `UNKNOWN` rows (48/45/45/38), listed by id. ABST-001 never measures them.
- **structure_ids (R1).** Computed at R1, CPU-only, on panel metadata with the registered `structure_overlap`:
  - 512 distinct and 0 missing;
  - **0** shared with any used row or with RTG1 train;
  - **both positive controls caught.** A used row's structure_id and an RTG1-train row's structure_id are each **appended to the panel list** and must raise the overlap count by exactly one. All 19,157 train rows carry a structure_id.
  - R1 review: a first R1 version drew its probe from the banned set and tested membership, which passed by construction.

  This agrees with the cold review's independent count.
- **The gap cue (R1, cold review #22; case-insensitive after the R1 review).** The phrase "not recorded" appears in the last user turn, never in the system prompt. It occurs in lower case and also as "NOT RECORDED". Rows with the cue in either case, UNKNOWN / known, per 64:

  | family | UNKNOWN | known |
  |---|---|---|
  | counterfactual_state | 64 | 39 |
  | prerequisite_schedule | 64 | 41 |
  | register_machine | 64 | 34 |
  | evidence_chain | 0 | 0 |

  **Cue-only AUROC (UNKNOWN against known) = 0.652.**
  - A lower-case-only match gives 45/27, 64/41, 42/15, 0/0 and AUROC 0.633. That is the cold review's count; its round 1 gave 0.652.
  - The runner stores the case-insensitive flag.

## 4. Cells, compensation, measurement

**Depth configurations:**

| config | kept blocks |
|---|---|
| S20 (recommended) | 0,1,2,3,4,7,9,10,11,15,16,17,18,19,20,21,22,25,26,27 |
| S18 | 0,1,2,3,4,9,10,11,15,16,17,18,20,21,22,25,26,27 |
| teacher | all 28 blocks |

- Each student is measured plain and with LC.
- LC is exactly SMALLVD-001's: fitted on the 255 SELECT rows with the S1 continuations, ridge toward I (λ = 1e-3 × the median eigenvalue), position 0 excluded, fp32. There is no free parameter.

**Measurement** (`abst_measure.py`):
- **One row per forward pass (B = 1).** The pool-conditional H0 needs each row's value to depend on that row alone. This also makes the measurement deterministic for fixed code and inputs, which is what makes a rerun harmless (§9).
- **Every pass is read under every support.** One pass per (config, input vocabulary ∈ {V0, V4, V2}) is read under all three output supports.
- **Stored per row, for every config × input × support:**
  - total and per-token nats;
  - first-position log-probs of UNKNOWN, 0–3 and `<|im_end|>`;
  - the log retained mass;
  - the top-5 ids (matched supports only).
- **Also stored per row:** tokenisation identity (as at 2f0a21f5) and, new in R1, the gap-cue flag.
- **Rows are streamed.** Each row lands in the in-progress record as it is measured.
- **Matched cells:** A = S20/V4, B = S20/V2, C = S18/V4, D = S18/V2, E = S20/V0, F = S18/V0, and the teacher's T4, T2 and T0.
- **Greedy generation** (4 new tokens, B = 1) for A-plain, A-LC, T4 and T0 on all rows. Descriptive only.

## 5. Refusal controls (all before the first real-teacher forward pass on the panel)

- **R1, data:**
  - panel/manifest hash;
  - strata (8 × 64);
  - references matching their stratum;
  - id disjointness recomputed, including the designer-seen rows;
  - the reserve disjoint from the panel;
  - **R1 revision: structure_id disjointness** from every used row and RTG1 train, all 512 distinct and none missing (cold review #23). It has two positive controls, a used row's and a train row's structure_id each appended to the panel list and each raising the overlap count by one, and every train row must carry a structure_id. RTG1 dev/train come in by an explicit `--rtg1-sft` argument (new in R1) and are hash-gated.
- **R1 revision, the run itself:**
  - a real run is refused unless the panel is the **registered** one (`<records>/abst001-20260925`, fixed in the runner) and `--out` is its `result/`;
  - the device must be cuda and the teacher weights must hash to the registered sha256 (`962abc5b…`, the value in HLCC_RESULT and LC_ARMS);
  - every gated module must have been loaded from the package directory, and `hostbudget` from rab001. rab001_exploration precedes the package on `sys.path`, so a same-named file there would shadow a gated one.
- **R2, vocabulary** (checked on text): UNKNOWN and 0–3 are each a single token with the same old id in V0, V4 and V2; `<|im_end|>` is retained; every reference + `<|im_end|>` lies inside V4 and V2.
- **R3, identity fold ×2** (S20, S18): folding I must reproduce the plain student bitwise on SELECT.
- **R4, fit canary ×2** (tolerance ≤ 1e-6; a wrong kept set refuses):
  - S20 against HLCC-001's recorded fits: 0.31561964900369427, 0.46619289682902004, 0.4304032137161287, 0.1880707556197988.
  - S18 against LC_ARMS: 0.3427810384834908, 0.4493614433555329, 0.38891843565293577, 0.21465167924673445.
- **R5, measurement canary, tier 1** (exact; the recorded code path `vs.answer_bpb`). It runs on SMALLVD-001's spent HOLDOUT reference rows (186) with the recorded batching. Every per-row nats must reproduce within 1e-6, and the recorded kept sets must equal the registered ones:
  - A-plain against S6 `V4_32768/20L`;
  - D-plain against `V2_65536/18L`;
  - T4 against `V4_32768/28L`;
  - D-LC against LC_ARMS lc V2;
  - F-LC against LC_ARMS lc V0.
- **R5b, tier 2** (the new B = 1 path, same rows and cells): |nats − recorded| ≤ 1e-3. Sign agreement of d(D) is **reported, not gated**: once the nats clause holds it cannot fail.
- **No canary covers S20-LC** (cold review #38): no S20-LC nats were recorded on spent rows. Mitigations:
  - R4 checks the S20 fits against HLCC's;
  - the S18-LC canaries exercise the same `folded()` path;
  - **W** (R1): every `down_proj` weight must be bitwise equal to its state at load, before the boundary and after every LC cell.
- **R6, measured throughput:** the projection must be ≤ 60 min. Generation is priced as prefill plus 4 decode steps at 64 token-equivalents each, an upper bound. The per-token rate is calibrated on long-context HOLDOUT rows, which biases the projection upward (conservative).
- **Budget, code and RAM re-checks (R1, §9), just before the boundary:**
  - the read-budget rules again;
  - the files of every loaded module unchanged since import (the marker records the import-time hashes);
  - free host RAM ≥ 2.5 GiB (cold review M2).
- **Boundary marker.** Just before the panel is read, a create-only, fsynced `attempts/BOUNDARY_<utc>_<pid>.json` is written. It holds:
  - every control's numbers;
  - the code hashes;
  - pid, epoch and a per-launch `launch_id` (R1);
  - the evidence on earlier crossings (R1, §9).

**Shown to fail and to pass** on the tiny random dry model (CPU, 3 rows per stratum), at registration:
- **Pass:** fit canaries 0.0; tier 1 0.0 on all five cells; tier 2 at 8.6e-6 to 2.0e-5.
- **Refuse:** a fit residual +1e-5 (R4), a recorded S6 nats +1e-5 (R5) and a wrong kept set (R4). No output directory was created in any of the three.
- R1 changes runner code, so the dry end-to-end run and the three negative controls are **re-run on the R1 code**. Both are now scenarios of the committed harness (E2E, NC; §10), not ad-hoc runs.

**Pre-registered route after a benign canary failure** (the panel stays unread):
1. No run until an amendment is registered.
2. Only a tolerance may change. It must rest on a diagnostic session that reads SELECT/HOLDOUT rows only: the run-to-run spread s, and the real-model method-variant shifts (ridge ×2, ×0.5, position 0 included).
3. The new tolerance is 10·s, below half the smallest variant shift.
4. Otherwise the outcome is **CANARY_UNRESOLVED**, with the panel unspent.

## 6. The registered test (`abst_verdict.py`)

**Endpoint:** s = sign(NLL_plain − NLL_lc) per row and cell.
- The sign is taken on the **difference**, so nothing can raise after the read.
- Bitwise ties, NaN and missing values count as ties, dropped and counted.
- **H0 (pool-conditional, given T):** each family pool is split evenly. The row-level exact sign test is exact for i.i.d. signs, and conservative under stratification and sampling without replacement.

| step | test | α | threshold (n = 256, no ties) | labels |
|---|---|---|---|---|
| 1 | H1, cell A | one-sided 0.025 **in each direction** | ≥ **145** worse (or better) | **LC_LOWERS_UNKNOWN** / LC_RAISES_UNKNOWN / NOT_CONFIRMED. The two directional labels together have size **0.039**. The chain continues only on LC_LOWERS_UNKNOWN, carrying 0.025. |
| 2 | `UNKNOWN`-row anchor (A vs T4) | two-sided 0.025 (0.0125 per tail) | ≥ **147** closer or farther | TOWARD / AWAY_FROM_TEACHER / ANCHOR_UNRESOLVED (stops) |
| — | after step 2 rejects, the carried 0.025 is split equally | | | |
| 2b | known-row anchor (A vs T4), only after TOWARD | one-sided 0.0125 toward the teacher | ≥ **147** | combined with over-suppression, see below |
| 3 | Holm at 0.0125 over D±, B±, C±, dV = sign(d_B − d_A) > 0, dD = sign(d_C − d_A) > 0 | Holm | 153, 152, 152, 151, 151, 150, 149, 147 | see below |

**Tree-level error (R1, cold review #17).** At the exact null, the whole tree's family-wise error equals step 1's two-directional **0.039**. Counting directional (type III) errors near the null, the worst case is about **0.042**. **Bound stated: about 0.04.** "Carrying 0.025" is exact only for the non-directional H2 null.

**H2 labels** (R1 names, cold review M1 and M6):

| label | condition |
|---|---|
| **TEACHER_CONSISTENT_DIRECTION** | TOWARD, step 2b rejects, and fewer than half the `UNKNOWN` rows with defined values over-suppress |
| TOWARD_TEACHER_OVER_SUPPRESSION | TOWARD, and at least half of those rows over-suppress |
| TOWARD_TEACHER_NOT_CORROBORATED | TOWARD, no over-suppression majority, and step 2b does not reject |
| AWAY_FROM_TEACHER | the away tail of step 2 rejects |
| ANCHOR_UNRESOLVED | step 2 rejects in neither direction (the chain stops) |

**Over-suppression (R1, cold review #18/#12).**
- A row over-suppresses when NLL_plain < NLL_T < NLL_lc: LC's `UNKNOWN` NLL passes **above** the teacher's. Crossings in the other direction are not counted.
- **Denominator:** the `UNKNOWN` rows with all three values finite (256 unless a value is missing). This replaces 2f0a21f5's non-tied anchor count.
- **Majority:** 2k ≥ n_defined.

**Step 3 labels** (R1: every label states its referent; cold review M6). These apply only if D+ is rejected (LC makes `UNKNOWN` more likely at S18/V2). Here −1 is A's sign (LC makes `UNKNOWN` less likely) and +1 is D's:

| B sign | C sign | label | reading for A's sign | reading for the D reversal |
|---|---|---|---|---|
| worse | better | SIGN_FOLLOWS_DEPTH | depth-20 carries A's sign | changing depth alone reverses |
| better | worse | SIGN_FOLLOWS_VOCABULARY | V4 carries A's sign | changing vocabulary alone reverses |
| worse | worse | A_SIGN_EITHER_FEATURE_SUFFICIENT__REVERSAL_NEEDS_BOTH | either A feature suffices | both changes are needed to reverse |
| better | better | A_SIGN_NEEDS_BOTH_FEATURES__EITHER_CHANGE_REVERSES | A needs both features | either change alone reverses |
| one signed, one not | | PARTIAL | | |
| neither signed | | UNRESOLVED | | |

If D− is rejected instead, the label is NO_REVERSAL; otherwise D_UNRESOLVED. Steps that are not reached are **UNTESTED**.
- **dV and dD** are reported with their Holm decisions. They test a shift in effect size (d_X,i = log(NLL_plain/NLL_lc)) between cells, not which factor carries the sign, and carry no licence (cold review #7).

**Size** (reported, no gate): a lower confidence bound on the **median per-row ratio NLL_lc/NLL_plain** (order statistic, 97.5%).

**Power and MDE** (exact binomial, n = 256; conservative, because sampling without replacement adds a finite-population correction):

| step | P = 0.58 | 0.60 | 0.62 | 0.65 | 0.70 | MDE (80% power) |
|---|---|---|---|---|---|---|
| step 1 | 0.694 | 0.877 | 0.966 | 0.998 | 1.000 | 0.591 |
| steps 2 / 2b | 0.600 | 0.818 | 0.941 | 0.995 | 1.000 | 0.599 |
| Holm, first step | | | | | | 0.622 |

The lead's own 13/16 has a 95% interval of [0.544, 0.960]. **NOT_CONFIRMED cannot exclude P(worse) below about 0.59.**

**Null calibration of the registered statistic** (`design/ABST_NULL_SIM.json`, **regenerated on the final R1 code**, seed 20260926, all seven checks pass). Draw counts:
- 20,000 for N1–N4, D1, D2 and D2r;
- 5,000 for N5, N6, P1–P5, P4b and S1.

Each draw takes 64 rows per family without replacement from the real pool sizes and runs through `verdict()` itself.

R1 makes the known-row cells **noisy** (cold review M1). In 2f0a21f5 they were deterministic: LC 0.7× or 1.3× plain, teacher 0.3× plain on every row. Now:
- the teacher is at 0.11–0.57× plain (the spent range), or at 1.75–9.1× on a reversed share of rows;
- LC moves the digit NLL down on a random 85% of rows (up on 85% in the away variant), by a random factor;
- on UNKNOWN rows the teacher is at 1.14–2.85× plain (the spent range).

**What noise does not buy (R1 review).** D1's rate, D2's rate and P4's power are set by the chosen alternatives, not by the design:
- the per-row closer rates are about 0.66 on UNKNOWN rows and 0.85 on known rows;
- D2 holds by construction of the known-row geometry, as §2 says;
- P4's check (≥ 0.95) passes for any code that implements steps 1, 2 and 2b.

Two reported cells show the dependence: **D2r** (generic damage with 60% of known rows reversed) and **P4b** (a known-row effect of only 0.60 toward).

| cell | result |
|---|---|
| N1a / N1b, half worse, odd pools rounded down / up | directional false rates ≤ 0.00115 |
| N2 / N3, heterogeneous nulls | ≤ 0.00015 |
| N4, `UNKNOWN`-anchor null | false step-2 rejection 0.02005 |
| **D1, teacher-blind answer-type shift** (`UNKNOWN` NLL up on 70% of rows, gold-digit NLL down on 85%, noisy, regardless of the teacher, spent geometry) | TEACHER_CONSISTENT_DIRECTION in **99.9%**: **reported, not checked.** The label is not teacher-specific. |
| **D2, teacher-blind generic damage** (most NLLs up: 70% of UNKNOWN rows, 85% of known rows; plain above the teacher on every known row) | step 2 says TOWARD in 99.9%, but TEACHER_CONSISTENT_DIRECTION **0.0000** (checked ≤ 0.025 + 3SE; holds by construction of the geometry) |
| **D2r**, the same with 60% of known rows reversed | TEACHER_CONSISTENT_DIRECTION **47.2%**: reported. This is why §9 carries the count rule. |
| N5 / N6, Holm false rejections over true nulls | 0.0000 / 0.0010 (bound 0.0125) |
| power | lead shape 1.000; homogeneous 0.60 0.952 (with the finite-population correction); TEACHER_CONSISTENT_DIRECTION truth 1.000 (extreme alternative); **P4b, known-row effect 0.60: 0.814**; SIGN_FOLLOWS_DEPTH truth 1.000 |
| S1, superpopulation family effect | false LC_LOWERS for new families 0.21 |

**Unit tests** (`tests/test_abst_verdict.py`, `tests/test_abst_describe.py`, **17 pass** on the R1 code):
- every label is reachable, and degenerate values never raise;
- the null tests run on the registered statistic;
- teacher-blind generic damage does not earn TEACHER_CONSISTENT_DIRECTION;
- **a teacher-blind answer-type shift DOES earn it.** The test pins the limitation so that no licence can forget it.
- **over-suppression is pinned** (R1 review: swapping in the 2f0a21f5 rule kept the suite green):
  - one direction only: a reverse crossing does not count;
  - the defined-rows denominator: ties count, None/NaN/inf do not;
  - the boundary 2k ≥ n;
- the tracking statistic behaves as specified:
  - near zero under a uniform shift, and under a z-dependent shift with an independent teacher (the raw correlation is then strongly negative);
  - strongly positive under row-level tracking;
  - **near zero within families** when a family-level shift aligned with family teacher offsets makes the pooled partial look like tracking;
- the new descriptive outputs are asserted: cue counts and AUROC, digit-only reading, digit anchor, tracking fields.
- a first anchor-null construction was not a null (P(closer) = 0.41) and failed correctly;
- a Clopper-Pearson bug (upper bound 4e-19) was caught by a known-value test and fixed.

## 7. Descriptive (registered; never a verdict or a selection) — `abst_describe.py`, from stored per-row values only

**Per LC cell (A–F), `UNKNOWN` and known rows:**
- better/worse counts, with a 95% Clopper-Pearson interval;
- median d;
- `plain_nll_below_teacher` (count);
- LC closer to / farther from the teacher;
- rows where LC crossed the teacher;
- median gap closure (1 means reaching the teacher; more than 1 means overshooting).

**First-position readouts:**
- the margin m = log p(UNKNOWN) − logsumexp log p(0..3);
- AUROC(m; `UNKNOWN` against known) for plain, LC and the teacher;
- the format decomposition.

**Cell A:**
- by family (**the per-family direction printed beside every licence**, §2), and by tokenisation identity;
- **the `UNKNOWN` token alone** (per_token[0]): sign counts, and agreement with the two-token endpoint;
- **R1: the gold digit alone on known rows** (per_token[0]; step 2b's endpoint includes `<|im_end|>`, cold review #40). Two readings: the plain-vs-LC sign count, and **the step-2b anchor on the digit alone** (closer/farther to the teacher's digit NLL).

**Row-level tracking (R1, cold review M1; descriptive only, no α, licenses nothing).**
- **Rows:** cell A, with plain and LC at S20/V4 and the teacher T4, every NLL read under the V4 support as the total nats of reference + `<|im_end|>`. `UNKNOWN` rows and known rows are computed **separately**, over the rows where all three NLLs are finite and > 0.
- **Statistic:** with δ = log NLL_lc − log NLL_plain, x = log NLL_T and z = log NLL_plain, report the **within-family Spearman partial correlation ρ(δ, x | z)**. Ranks are taken within each family and centred on the family's mean rank. The partial Pearson correlation of those centred ranks, pooled over families, is (r_δx − r_δz·r_xz) / √((1 − r_δz²)(1 − r_xz²)).
  - **Why within family (R1 review).** A teacher-blind shift that is uniform within each family but differs between families gives a **pooled** partial of about 0.4 to 0.6 with no row-level tracking, if the family teacher offsets line up with it. F4 forecasts exactly such a family split.
  - It is zero in expectation when δ is independent of x given z within families.
  - It is positive when rows move in proportion to where the teacher sits.
  - Under independence at n = 256 its SD is about 0.063.
  - Also reported: the pooled partial, the four per-family partials, and the raw Spearman ρ(δ, x − z) (confounded through z).
- **Readings:**
  - TRACKING_SIGNAL if ρ ≥ 0.2 (about 3.2 SD);
  - NO_TRACKING_SIGNAL if |ρ| < 0.1;
  - WEAK_OR_NEGATIVE otherwise.
  - The raw Spearman ρ(δ, x − z) is reported for comparison only; it is confounded through z.
- **Limit.** A TRACKING_SIGNAL is still correlational: a row feature such as the gap cue could drive both the teacher's NLL and LC's change. The result may say "row-level changes correlate / do not correlate with the teacher's position (partial ρ = …)". It may not say "LC tracks the teacher" or anything teacher-specific.

**Other descriptives:**
- the vocabulary factor at depth 20, split into output-support and input terms;
- baseline against endpoint (A against D);
- greedy abstention, false abstention and known accuracy;
- **R1: the gap cue.** Rows with the cue per stratum, and the cue-only AUROC (0.652, §3), printed beside M2 and wording rule 1.

**Pre-registered readings (labels, not tests):**
- **M1 (among LC-worse rows in A):** TEACHER_TRACKING if ≥ 0.75 moved closer; LC_ARTEFACT if ≤ 0.25; MIXED otherwise. Like step 2, it is nearly implied by the geometry and is **not independent evidence**.
- **M2:** DISCRIMINATION_LOSS if AUROC_lc < AUROC_plain − 0.03; BIAS_SHIFT if the median Δm < 0 on both strata and |ΔAUROC| ≤ 0.03; OTHER otherwise.
- **Wording rule 1:** if |AUROC_T4 − 0.5| ≤ 0.05, every "abstention" word in the result becomes "`UNKNOWN`-token propensity".
  - **R1:** a teacher or student AUROC above 0.55 can come from reading the literal cue (cue-only AUROC 0.652). The result says so whenever such an AUROC is not above 0.652.
- **Wording rule 2:** if the `UNKNOWN`-token-alone majority sign disagrees with the two-token endpoint's majority, the result says "reference-sequence likelihood", not "`UNKNOWN` likelihood".
- **Wording rule 2b (R1):** if the digit-only anchor on known rows does not agree in majority direction with step 2b (closer vs farther), the known-row part of any licence says "digit-plus-`<|im_end|>` sequence", not "the gold digit".
- **Wording rule 3 (R1):** no outcome licenses "restores abstention", "abstention is safe", "tracks the teacher" or "teacher-specific".

## 8. Registered forecasts (scored in the result; not gates; a forecast about a step not reached is NOT_SCOREABLE)

- **F1.** H1 = LC_LOWERS_UNKNOWN.
- **F2.** The cell-A worse fraction is ≥ 0.70.
- **F3.** H2 = TEACHER_CONSISTENT_DIRECTION (R1 name). Close to implied by F1 given HLCC's known-row 16/16 (§2).
- **F4.** In cell A, the prerequisite_schedule worse fraction is ≤ 0.50, and each other family's is ≥ 0.70.
- **F5.** D+ is rejected (LC makes `UNKNOWN` more likely at S18/V2).
- **F6.** The attribution label is SIGN_FOLLOWS_DEPTH (R1 name).
- **F7.** M1 = TEACHER_TRACKING. This is nearly implied by F8 and not independent.
- **F8.** Plain V4/20's NLL is below the teacher's on ≥ 90% of `UNKNOWN` rows.
- **F9.** The teacher abstains greedily on ≤ 5% of `UNKNOWN` rows (T4 and T0).
- **F10.** In cell D, LC moves away from the teacher on the majority of `UNKNOWN` rows (spent: 1 of 17 closer).

## 9. What each outcome licenses; read budget; execution

**Every licence carries two statements (R1):**
- **Scope.** *Pooled over the four RTG1-dev family pools, equally weighted, conditional on the fitted T. The per-family majority directions are given beside it, and a family whose majority goes the other way is named in the sentence.*
- **Residual.** *Abstention where the teacher abstains (e.g. CC0) is not measured by any registered instrument.*

| outcome | licence |
|---|---|
| LC_LOWERS_UNKNOWN + **TEACHER_CONSISTENT_DIRECTION** | LC makes the gold reference `UNKNOWN` **less** likely on V4/20 on most rows. It moves V4/20 in the teacher's direction on both `UNKNOWN` and known rows, without an over-suppression majority. This is consistent with moving toward a teacher that does not abstain on these pools. **Generic damage is excluded only if fewer than 147 known rows have NLL_plain < NLL_T**; otherwise the licence says "generic damage is not excluded". **It does not restore abstention, and it is not shown to be teacher-specific**: a teacher-blind shift of mass from `UNKNOWN` to the digits passes the same tests (§6 D1). **HLCC's abstention caveat stays in force, reinforced:** the direction-only licence does not license abstention, and LC lowers the likelihood of the gold `UNKNOWN` answer. (R1, cold review M1: 2f0a21f5 *withdrew* the caveat here; that licence is revoked.) |
| LC_LOWERS_UNKNOWN + TOWARD_TEACHER_OVER_SUPPRESSION | LC pushes V4/20's `UNKNOWN` NLL past the teacher's on at least half the rows with defined values. **Caveat retained**, reworded as "LC over-suppresses `UNKNOWN` on V4/20". |
| LC_LOWERS_UNKNOWN + TOWARD_TEACHER_NOT_CORROBORATED | Direction toward the teacher on `UNKNOWN` rows only; generic damage is not excluded. **Caveat retained.** |
| LC_LOWERS_UNKNOWN + AWAY_FROM_TEACHER | LC pushes V4/20 away from the teacher on `UNKNOWN`. LC is not recommended for V4/20 where `UNKNOWN` is a valid answer, pending a mechanism study. |
| LC_LOWERS_UNKNOWN + ANCHOR_UNRESOLVED | Replicated (LC makes the gold `UNKNOWN` less likely), not interpreted. **Caveat retained.** H3 is untested; B, C and D are reported descriptively. |
| LC_RAISES_UNKNOWN | **Tested reading (gold):** LC makes the gold reference `UNKNOWN` **more** likely on most rows, so the lead is refuted in direction. **The teacher-relative clause is DESCRIPTIVE only** (R1, cold review M5): the stored anchor counts for cell A (closer/farther) are quoted, with no α, because the chain stops at step 1. The caveat changes to "the V4/20 lead did not replicate; on the RTG1-dev pools LC raised the `UNKNOWN` likelihood". H2 and H3 are untested. |
| NOT_CONFIRMED | Not replicated at MDE 0.59. The caveat stays in weakened form ("unresolved"). It is not withdrawn. |
| H3 labels | Say which factor carries the sign, in the label's stated referent (§6). No recommendation changes, and no set is chosen. |

**In every case:** GO/HOLD stays HOLD; no quality claim; no release.

**Read budget: ONE primary.** R1 hardens the budget (cold review M2 and M3, soup review). The rules live in `run_abst.py` itself, not only in the launcher.
- **In one sentence:** at most **3** boundary crossings in total; none after **two recorded (infrastructure) failures**; none after any primary exists. 2f0a21f5 allowed one crossing plus one rerun after an infrastructure failure. The extra crossing exists only for hard kills (reboot, power loss), which leave no attempt record and persist no panel values.

- **Where things are written.**
  - A real run reads **only** the registered panel and writes **only** to `<records>/abst001-20260925/result`. Its attempts directory is `<records>/abst001-20260925/attempts`. The location is fixed in the runner, so a copy of the panel elsewhere is refused (R1 review).
  - A `--dry-run` is refused anywhere under the records tree.
  - The launcher also pins `--out` (`abst_run.sh`).
- **A run lock.** `attempts/RUN.lock` is taken O_EXCL, holding the pid and epoch, for the life of the run.
  - A lock whose holder is not running is renamed aside, and the rename is recorded. "Not running" covers a pid running now but created after the lock's epoch: a pid reused after a reboot.
  - A lock held by a running holder refuses.
- **The refusal rules.** They are evaluated **at start-up, after the run lock, after the GPU lease and just before BOUNDARY.** After the lease, free RAM is re-checked at 3.6 GiB, because the wait can last 2 h (cold review #35). A launch refuses if:
  - an `ABST_PRIMARY.json` exists **anywhere** under the root (any depth);
  - **3** BOUNDARY markers exist (the cap on crossings);
  - any attempt record is **not** classified as infrastructure;
  - any attempt record says the primary was written (R1 review: this still holds if the result is moved out of the tree);
  - **2** attempt records exist: no crossing follows two recorded failures;
  - the process of an earlier crossing is still running (its pid running and created no later than the marker's epoch; if the process cannot be opened, psutil reads its creation time; unknown fails closed);
  - the loaded code differs from any earlier marker's (cold review #29/31).
  - An unreadable (truncated) record raises, which also refuses.
- **Hard kills (soup review).** A reboot or power loss leaves a marker and **no** attempt record. The attempt record is matched to its marker by `launch_id`.
  - Such a crossing persisted no panel values, and before `ABST_PRIMARY.json` nothing about results is printed. It counts against the cap of 3 but not against the two-recorded-failures rule.
  - A native crash (e.g. an access violation in a CUDA call) also leaves no record. If it is deterministic, it wastes at most the remaining crossings, and it cannot select, for the reason below. The restart evidence would then show no reboot.
  - The next launch records every earlier crossing in its refusal controls. If any were hard kills, it also records the System-log restart and power events (User32 1074, Kernel-Power 41/109, EventLog 6008) since the earliest one. These are read from the XML, whose SystemTime is true UTC; wevtutil's text format prints local time with a "Z" suffix, measured at R1. **The evidence is descriptive, never a gate.** A force-kill of a hung job is also data-blind, and a log-dependent gate would refuse it.
  - Three operator restarts happened within 9.5 h on 2026-09-25/26 (19:39:51Z, 00:53:53Z, 05:02:48Z). **The operator is asked not to restart the host during the ABST GPU window.**
  - After a hard kill, the stale `abst001` GPU lease is released by the registrant with `gpu_lease.release(token)`. The registrant first asserts the label `abst001`, the dead holder pid and the token, under the operator-approved own-lease rule. Otherwise the operator releases it.
- **Why a rerun cannot select (soup review).** An infrastructure attempt record persists the finished cells and the rows of the cell in progress. What keeps a rerun harmless is two things together:
  - the code-hash rule: a rerun runs the **same code**. The hashes are those of every module actually loaded from the package and rab001 directories, taken at import. The files must be unchanged at the boundary, and the gated modules must load from the package;
  - deterministic measurement: B = 1, fixed rows, tier-1 exactness.

  A rerun can only reproduce the same values; it cannot choose among them.
- **Classification (cold review #32; R1 review).** Infrastructure means only:
  - LowMemoryAbort;
  - GPU or **host** out-of-memory: OutOfMemoryError, "out of memory", Python MemoryError (including numpy's allocation error), and torch's CPU allocator "not enough memory". Host RAM is this host's binding constraint;
  - the Windows TDR signature "CUDA error: unknown error".

  A KeyboardInterrupt, or any other CUDA error (e.g. a deterministic device-side assert), is **not** infrastructure.
- **Order after the boundary (cold review M2 and #33).**
  1. The PRIMARY cells come first: A-plain, A-LC and T4, on all 512 rows.
  2. Between them, a RAM dip **waits** (poll 30 s, bound 30 min) instead of aborting. A resident, idle model loses nothing by pausing.
  3. There is **no wait between the last cell of a stage and that stage's write**: the primary, `ABST_H3.json` and `ABST_RESULT.json` (R1 review).
  4. `ABST_PRIMARY.json` is written create-only and fsynced, in a temporary `result.partial_<launch_id>/` that is then renamed to `result/`. A hard kill between the mkdir and the write therefore leaves no empty `result/` to block the rerun. It holds steps 1, 2 and 2b, and per-row A-plain, A-LC and T4 for **all 512 rows**, so step 2b is re-verifiable from it.
  5. The remaining cells follow, with bounded waits.
  6. `ABST_H3.json` is written before generation.
  7. Generation runs, with the W check after the LC generation.
  8. `ABST_RESULT.json` is written last.
- **The lock and the lease.** The lock is released even if the lease release raises.
- **Keeping the budget state safe.** The markers, attempt records and result are untracked files in the shared records working tree. The registrant commits them to records promptly after the run, and no session may run `git clean` or `git stash -u` there (R1 review #13).
- **After `ABST_PRIMARY.json` exists** there is no rerun. Unmeasured parts are NOT_MEASURED.
- Every post-boundary failure writes `attempts/ATTEMPT_*.json`. It holds the refusal-control evidence, every finished cell and the rows of the cell in progress.
- **Tested on the dry model** by the committed harness `design/abst_harness.py` (§10). The pass log is a precondition of the GPU run.

**Execution — a hash chain:**
- `launch/launch_abst.ps1` verifies `launch/abst_run.sh`'s sha256 before the WMI create.
- `abst_run.sh` verifies `launch/abst_hashes.txt`'s own sha256, then runs `sha256sum --strict -c` over every gated file:
  - the 12 local modules (`abst_verdict`, `abst_measure`, `abst_describe`, `run_abst`, `hlcc_verdict`, `linear_compensation`, `virtual_student`, `layer_ablation`, `layer_select`, `run_layer_study`, `param_accounting`, `gpu_lease`) and `hostbudget`;
  - the panel and manifest; LAYER_PANELS; TOKENIZER_CORPUS; S1; S6; LC_ARMS; HLCC_RESULT; CONFIRM_PANEL;
  - the V4/V2 token maps and tokenizers;
  - the teacher's tokenizer, config and weights;
  - **R1:** RTG1 `dev.jsonl` and `train.jsonl`;
  - **R1:** the pinned venv's interpreter and `pyvenv.cfg`, and the version files of torch, transformers, tokenizers and safetensors (cold review #34).
- The chain is written by the committed `launch/finalize_abst_gate.py`, once. **It refuses** unless `design/ABST_HARNESS_PASS.json` is an R1 all_pass log whose recorded module hashes cover every gated module and `hostbudget` and equal the files on disk. The pass log itself joins the hash list.
- `launch_abst.ps1` also verifies `launch/smallvd_runner.py`, the WMI runner, by hash. The uv CPython 3.12 that runs it is not hashed (disclosed).
- The result also records the loaded-module hashes, the module origins and the torch, transformers and tokenizers versions.
- The only in-launch retry is on the GPU-reset signature, before the primary exists.

**Resources:**
- lease `abst001`;
- at least 3.6 GiB free host RAM at launch and again after the lease;
- ≥ 2.5 GiB just before the boundary;
- after the boundary, a bounded wait below 1.5 GiB;
- CUDA fraction 0.78;
- about 2.0M panel forward tokens, about 1.0M generation token-equivalents and about 4M for the controls (SELECT identity/fits and the canaries);
- estimated 25–40 min, with R6 refusing above 60;
- disk writes under 20 MB.

Windows commit headroom is not gated (disclosed, cold review #35).

**Queue:** after the peers' queued jobs, in the order agreed with them: OSIDE resume, then CLOCKFIX when its disk and RAM allow, then ABST. ABST also waits for the cold reviewer's re-check of R1 and for the harness pass log.

## 10. Code

`abst_verdict.py` · `abst_measure.py` · `abst_describe.py` · `run_abst.py` · `build_abst_panel.py` · `design/abst_null_sim.py` + `design/ABST_NULL_SIM.json` · **`design/abst_harness.py`** + `design/ABST_HARNESS_PASS.json` (R1) · `tests/test_abst_verdict.py` · `tests/test_abst_describe.py` · `launch/abst_run.sh` · `launch/launch_abst.ps1` · **`launch/finalize_abst_gate.py`** (R1) · `launch/abst_hashes.txt` (written by the finalizer).

**R1 state of the launch chain.**
- At the R1 commit, `abst_run.sh` and `launch_abst.ps1` carry the placeholders `__HASHES_SHA__` and `__RUN_SH_SHA__`, and `abst_hashes.txt` is absent. **An unfinalized chain cannot launch:** the ps1 compares against a placeholder.
- The finalizer runs after the harness passes on the committed R1 code. The chain is then shown to pass, and to fail on a tampered file. Both steps are committed with their sha256s.

**The dry harness** (`design/abst_harness.py`, cold review M7). It is committed, not gated, and drives `run_abst.main()` on the tiny random CPU model (the 20-block stand-in is [0, 1, 3, 4], the 18-block stand-in [0, 1, 3]; 3 rows per stratum).
- Each scenario has its own parent directory and is crash-isolated: a crash is recorded as that scenario's FAIL.
- A before/after snapshot shows that nothing the harness could create appears under the panel's records directory.
- Every fixture file is hashed into the pass log.

| group | scenarios |
|---|---|
| output keying | refused, with nothing created: a dry run into `<records>/…/result`, into a new dir under the panel dir, elsewhere in the records tree, or into the panel dir; a dry run with a **copied** panel into the real result dir; a real run with any other `--out`; a real run with a copied panel |
| NC, negative controls | a recorded LC fit residual +1e-5 (R4), a recorded S6 nats +1e-5 (R5), a wrong kept set (R4): refused before any marker |
| E2E | a clean run: the primary equals the final steps 1–2, ABST_H3 equals the final verdict, every row has 15 cells × 3 supports and 4 greedy cells, and the descriptive, structure-control and module-origin fields are present |
| attempt records | a failure on the first primary cell (call 6) gives a non-infrastructure record and no primary; a failure mid-cell keeps exactly the one finished row; a failure after the primary (call 9) leaves `ABST_PRIMARY.json` with UNKNOWN and known rows |
| S1–S5 | a deterministic failure blocks a rerun; one rerun after an infrastructure failure completes, then a sibling `--out` is refused; after a clean run the same or a sibling `--out` is refused; **two recorded failures refuse the third launch** (the cold review's missing case); an infrastructure failure after the primary is final, even with the result moved deeper in the tree or out of it |
| S6–S8, hard kills (approximated by files: no record, dead marker pid, a stale RUN.lock) | two hard kills, then the third completes and lists both, with the restart query returning code 0; three hard kills, then the fourth is refused on the cap; one recorded failure plus one hard kill, then the third completes with prior_crossings [infrastructure, NONE] |
| L1–L3, lock | a live holder refuses; a dead holder is set aside and recorded; **a reused pid in the lock** is set aside |
| P1–P2, C1, K1, O1 | a live earlier process refuses, and the run completes once it ends; **a reused pid does not block**; changed code refuses, and restored code completes; a file changed between import and the boundary refuses; a shadowed gated module refuses |
| M3a–M3c | a primary (at depth 1 or 3), or a live concurrent crossing, appearing during the pre-boundary canary is refused **just before the boundary**, with no marker written |
| B1, T1 | a pre-boundary RAM refusal (in the real SystemExit form) writes no marker and no record, and a relaunch then completes; a truncated attempt record makes the next launch refuse |
| W1–W3 | a post-boundary dip **waits** and completes; a dip past the bound gives an infrastructure record; a dip right after the last primary cell still leaves `ABST_PRIMARY.json` written first |
| I | classification: host and GPU OOM, TDR and LowMemoryAbort are infrastructure; KeyboardInterrupt, a device-side assert and SystemExit are not |

**Status at the R1 commit.**
- **Done on the R1 code:** the null sim (all checks pass) and the unit tests (17 pass).
- **Not yet run:** the harness and the dry end-to-end run with its three negative controls. They need about 3.5 GB of commit for the CUDA torch import. They are queued, with the peers' agreement, after the soup lane's OSIDE run ends.
- **The GPU run does not launch until** the harness passes and the pass log is committed. Any code change they force is a further disclosed pre-run revision, with the sim, the hashes and the pass log regenerated. A pass log from before R1 does not count.

## 11. Disclosures

- **D1, panel built before registration.** Built CPU-only; no forward pass of the real teacher.
- **D2, dry runs touched panel rows.** The dry runs and harnesses ran a **random** tiny model on 24 panel rows (3 per stratum). Their outputs carry no information about the real teacher.
- **D3, canaries on spent SMALLVD HOLDOUT rows.** R5 re-reads SMALLVD-001's spent HOLDOUT reference rows to reproduce recorded numbers (precedent: LC_ARMS `reproducibility_vs_S5`). HLCC's panel is **not** touched (its §9 forbids any rerun). The operator may overrule this before the run.
- **D4, CC0 dropped.** Hygiene review found:
  - 17 of the 86 eligible fresh CC0 `unknown` rows already have model predictions in another lane;
  - packets overlap the seen panels;
  - only 45 rows pass the length filter.

  CC0 is the only place this teacher abstains (cue-driven, one template). Abstention where the teacher abstains is therefore **not measured**, and every licence says so.
- **D5, the teacher-anchor frame comes from spent design data** (§1, spent rows only). This includes the finding that LC at S18 moves **away** from the teacher (1 of 17 closer). That finding was surfaced by the pre-registration review, not by the designers. The anchor tests were added because of this frame, which is admissible before any fresh read. **R1:** HLCC's known-row 16/16 (§1) was spent data too. It was not cited in 2f0a21f5, and it makes step 2b close to implied if H1 replicates.
- **D6, designer-seen rows.** Three fresh RTG1 known-answer rows appeared in design-agent tool output. They are excluded by id. No design or review agent ran a model or wrote a file: one design command loaded tokenizers only.
- **D7, R1 read panel text and metadata, no model.** R1 counted the gap cue in panel prompts and ran the structure_id check on panel ids, both in pure Python. It recomputed HLCC's spent per-row values by family.
- **D8, the fit set shares the families** (§2): 35 of SELECT's 255 rows are RTG1-dev rows from the same four families.
- **D9, peer reviews.** R1 answers a cold review by the Rineya Delta session (records 9a3399f) and a peer review of the runner by the soup lane. The cold reviewer re-checks R1 against the code before the run.

## 12. Pre-registration review (adopted before the 2f0a21f5 commit)

**Review:** four lens reviewers (statistics, runner, measurement, interpretation) and up to four refutation attempts per lens (eight were needed), all read-only. **Adopted:**

- **The anchor is nearly implied by H1** (statistics and interpretation). Step 2b and the no-overshoot condition were added, and the label is conjunctive. The teacher-blind-damage simulation cell and unit test were added. *(R1: the label only rejects generic damage, not a teacher-blind answer-type shift. See §13 M1.)*
- **Wording for LC at S18.** LC at S18 moves away from the teacher on spent data. The neutral D± wording is used, with F10 and D5.
- **Licence table.** It was inconsistent. Every row now carries the CC0 residual; LC_RAISES is the away direction; NOT_CONFIRMED keeps a weakened caveat. *(R1: LC_RAISES now gives the gold reading, with the teacher clause descriptive.)*
- **Read budget across launches.** A BOUNDARY marker was added, with startup refusal rules. *(R1: hardened, §9.)*
- **Hash gate.** It now uses `--strict` and verifies the hash list itself; the ps1 verifies the sh.
- **Attempt records.** They now carry the refusal evidence and the rows of the cell in progress.
- **R1** re-checks the designer-seen rows and the manifest's content evidence. **R5b's** sign clause is reported, not gated. **R6** has a generation upper bound. Resource numbers are corrected.
- **Accuracy fixes:**
  - §1's vocabulary claim;
  - the size-bound wording;
  - step 1's two-direction size of 0.039;
  - N1's rounding (both tilts are now simulated);
  - the known-row field name;
  - the teacher-AUROC threshold;
  - a reading on the `UNKNOWN` token alone;
  - NOT_SCOREABLE for unreached steps.

**Refuted by the skeptics and not adopted:** that the registered H3 inputs could be lost to a descriptive-stage failure. The in-memory records are bound before any cell runs, and cells are atomic.

## 13. Revision R1 (2026-09-26, before any forward pass of the real teacher on the panel)

**Scope of the revision.** Every change is text, runner code or simulation, and **none uses panel outcomes**: no model has read the panel. The verdict code's decision rules are unchanged in their thresholds and α. The changes are label names, over-suppression replacing overshoot, and the fields listed below.

**Cold review, major findings** (`ABST_PREREG_COLD_REVIEW.md`):

| # | finding | R1 change | where |
|---|---|---|---|
| M1 | FIDELITY cannot separate teacher-tracking from a teacher-blind UNKNOWN→digit shift, and its licence withdrew HLCC's caveat | **Label renamed** TEACHER_CONSISTENT_DIRECTION. **Licence re-attributes and never withdraws**; the caveat is reinforced. HLCC's 16/16 is disclosed. The sim's known rows are noisy. **D1 is now the answer-type shift (99.9%, reported)**, D2 is generic damage (0.0000, checked, by construction of the geometry) and D2r shows the geometry dependence (47.2%), hence the count rule. A unit test pins the limitation. Row-level tracking is registered descriptively. Wording rule 3. | §1, §2, §6, §7, §9; `abst_verdict.py`, `abst_describe.tracking`, `design/abst_null_sim.py`, tests |
| M2 | a RAM guard between the last primary cell and the write; after the boundary a dip aborts | no wait between the last primary cell and the write; post-boundary dips **wait** (bounded 30 min); pre-boundary RAM check at 2.5 GiB | §9; `run_abst.wait_for_ram`, `PRIMARY_ORDER` |
| M3 | budget checks only before a 2-hour lease wait | the rules run at start-up, after the lock, after the lease (with a RAM re-check) and just before BOUNDARY; O_EXCL run lock; live-pid and code-hash refusals over **every** earlier marker; attempts keyed to the panel's records dir | §9; `run_abst.budget_check`, `RunLock` |
| M4 | licences read per pool for a pooled majority; the lead is family-split | pooled scope in every licence; per-family directions beside it, and a reversing family named; the SELECT family overlap disclosed | §1, §2, §9, D8 |
| M5 | the LC_RAISES licence asserts an untested teacher-relative direction | gold reading tested; teacher clause descriptive, quoting the anchor counts | §9 |
| M6 | EITHER_SUFFICIENT / BOTH_REQUIRED invert under the reversal reading | labels renamed to state their referent, with both readings tabulated | §6; `abst_verdict.ATTRIBUTION` |
| M7 | the budget harness is uncommitted, and two scenarios never test their rule | `design/abst_harness.py` committed, with a pass log required before launch; **two recorded failures then a third** (S4); the different-parent case is closed in the runner (a real run's `--out` is fixed) and tested (G3) | §10 |

**Cold review, minor findings:**
- **#18/#12, overshoot denominator and direction.** Replaced by over-suppression, counted in one direction (plain < T < LC) over the rows with defined values.
- **#17, tree-level error.** Stated as about 0.04 (0.039 at the null, about 0.042 with directional errors).
- **#23, the content-duplicate control was vacuous.** Replaced by a run-time structure_id check with a real positive control. R1's full-panel count: 512 distinct, 0 overlap, control caught.
- **#22, the gap cue.** Cue counts and cue-only AUROC are recorded (§3), counted case-insensitively after the R1 review (0.652; the lower-case count is 0.633), and the cue AUROC is computed in the result. Wording rule 1 carries a cue clause.
- **#38, no S20-LC canary.** Added the W check: `down_proj` bitwise equal to load, before the boundary and after every LC cell.
- **#33, persistence.** The primary is fsynced, holds the known rows, and H3 is written before generation.
- **#32, infrastructure classification.** Narrowed to OOM, TDR and LowMemoryAbort.
- **#29/31, the rerun is not bound to the code.** Code hashes are compared against every earlier marker, and the real `--out` is fixed in the runner.
- **#11, H3 lost in some branches.** Stated as UNTESTED there, with B, C and D descriptive.
- **#42, stale docstring.** Fixed.
- **Unverified round-1 items adopted:**
  - **#7:** dV/dD carry no licence.
  - **#34:** RTG1, venv and library files are hash-gated; gated modules must load from the package (no shadowing from rab001_exploration); the WMI runner is verified by hash. The uv CPython that runs the runner is still not hashed.
  - **#35:** RAM is re-checked after the lease; commit headroom is disclosed.
  - **#36:** hard kills are handled; a stale run lock is set aside; a truncated record fails closed; the stale lease is released under the own-lease rule.
  - **#40:** a known-row digit-only reading is added.
  - **#43:** library versions are recorded.
- **Not adopted:** #8, #9, #10, #13, #20, #25, #26, #37, #39 and #41. They were unverified and are wording or notes; the registered route or text stands. The effect of #39 on R6 is conservative.

**Soup-lane peer review of the R1 runner:**
- **Two operator reboots could spend the read with no verdict.** Now at most 3 crossings, and none after two recorded failures; hard-kill evidence (launch_id matching plus System-log restart events) is recorded descriptively; the operator is asked not to restart during the window.
- **Pid reuse after a reboot.** Liveness now compares the process creation time with the epoch recorded in every marker and lock.
- **The reviewer's rationale is stated** next to the cap: code-hash binding plus deterministic measurement means a rerun reproduces and cannot select.

**R1 review round (before the R1 commit).** Four read-only lens reviewers (runner, harness, prereg against code, statistics) returned 43 findings on the uncommitted R1. Each was checked first-hand; the review record is kept by the registrant.

Changes:
- **Runner:**
  - host MemoryError and torch's "not enough memory" count as infrastructure;
  - no RAM wait before the H3 or result write;
  - a primary at any depth refuses, and so does an attempt record's primary flag;
  - the registered panel location is fixed in the runner, and dry runs are refused anywhere in the records tree;
  - the structure_id controls are appended to the panel list (both used and train), and every train row must carry a structure_id;
  - code binding uses the loaded modules' hashes taken at import, with a module-origin check and an unchanged-at-boundary check;
  - `pid_alive` fails closed when a process cannot be opened;
  - the primary is written via a temporary directory;
  - the lock is released even if the lease release raises;
  - a real run requires cuda and the registered teacher sha;
  - the W check runs after LC generation.
- **Descriptive:**
  - tracking is within family (the pooled partial is confounded by family shifts);
  - a digit-only step-2b anchor and wording rule 2b;
  - the gap cue is case-insensitive (0.652).
- **Simulation:**
  - UNKNOWN geometry 1.14–2.85;
  - D2's wording corrected, and its dependence on the geometry stated;
  - D2r and P4b added;
  - draw counts stated per cell.
- **Licence:** the count rule for the generic-damage exclusion.
- **Tests:** over-suppression pinned (direction, denominator, boundary); tracking nulls; the new outputs asserted (17 tests).
- **Harness:**
  - the negative controls and the end-to-end run are now committed scenarios;
  - copied-panel, reused-pid-lock, file-changed, module-shadowed, truncated-record and deeper or moved-out primary cases;
  - prior_crossings matching checked;
  - restart-query return code checked;
  - a vacuous check removed;
  - crash isolation;
  - every fixture hashed.
- **Finalizer:** refuses without a matching R1 all_pass pass log, which joins the hash list; the ps1 verifies the WMI runner.
- **Text:**
  - the recorded-failure rule is stated as coded ("no crossing after two recorded failures");
  - the draw counts;
  - a change-log line that credited R1 with removing a `build_calibration` import 2f0a21f5 never had is removed;
  - tokenisation identity is no longer tagged as new.

**Not adopted:** marking the three purely confirmatory harness sub-checks as vacuous (they stay as consistency checks). Snapshotting the whole records tree is also not adopted: peers write there concurrently, so the runner's records-tree refusal and the panel-dir snapshot are used instead.

**Checks re-run on the R1 code before commit:**
- `design/ABST_NULL_SIM.json` regenerated on the final code, all seven checks passing: `abst_verdict.py` sha256 `abeb12ea…`, `abst_null_sim.py` sha256 `2e440cce…`.
- 17 unit tests pass.
- The structure_id controls: both caught on the full panel (512 distinct, 0 overlap). With the train side's structure_id removed, the train control fails, as it should.
- `pid_alive` checked on self, on self with an epoch 1 h old (False), on a live child, on a child against an earlier epoch (False) and on a dead pid (False).
- The restart query returned tonight's three restarts, in UTC.
- The harness and dry battery: see the next paragraph.

**Harness-forced fix (2026-09-26, after the cold re-check of 71acefb8).** The first harness run on 71acefb8 failed every scenario that launches the runner.
- **Cause:** `code_hashes()` walks `sys.modules`, and `torch.ops` and `torch.classes` carry RELATIVE `__file__` values (`_ops.py`, `_classes.py`). Resolved against a working directory of the package, they named package files that do not exist, so the call raised FileNotFoundError at start-up, before any control.
- **Effect on a real run:** it would have crashed before reading anything, which fails safe.
- **Fix:** only absolute paths to existing `.py` files are hashed. In the pinned venv, `code_hashes()` then returns exactly the 12 gated modules and `hostbudget`. Under Python 3.11 the script-mode `__main__.__file__` is absolute, so the real launch records `run_abst.py` under the same key.
- **Record:** the failed run's log is kept in the registrant's scratchpad, and no pass log was written from it. The pass log in `design/ABST_HARNESS_PASS.json` is from the re-run on the fixed code.
