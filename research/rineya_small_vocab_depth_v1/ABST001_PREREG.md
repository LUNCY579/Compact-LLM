# ABST-001 — does linear compensation lower the UNKNOWN likelihood on the recommended student, and does that move it toward the teacher?

**Status:** REGISTERED 2026-09-25, before any forward pass of the real teacher on the panel. The panel was built CPU-only before this document and is pinned by hash. Before commit, the prereg and code went through an adversarial pre-registration review; every finding the skeptics upheld is adopted (§12).
**Branch:** `research/rineya-small-vocab-depth-distill-v1` · **Records:** `rineya-dataset-retention-records/abst001-20260925/`.
**Parents:** HLCC-001 (`HLCC001_PREREG.md`, `HLCC001_RESULT.md`); SMALLVD-001 (the S6 frontier and LC_ARMS).

**Design provenance.** Three independent designs were produced (confirm-first, attribution-first, behaviour-first), each scored by three adversarial critics (statistics, hygiene, interpretation). No model ran. This document synthesises them:
- **Backbone** from confirm-first (scored 8/8/6).
- **Teacher anchor and paired contrasts** from attribution-first (6/6/7).
- **Behaviour-first** rejected: its primary sat on a census, and its harm tail was unreachable. Its greedy readout and its exclusion of designer-seen rows are kept.

## 1. Why

HLCC-001 confirmed linear compensation (LC) on the recommended V4_32768/20 student, in direction only. It left one post-hoc lead, which was not registered and not tested:
- **V4/20:** on 16 RTG1 abstention rows whose reference is `UNKNOWN`, LC made the reference **less** likely in 13 (median ratio ×0.671).
- **18-block set (V2):** it made it **more** likely in all 16 (×1.665).
- **Why it can't be attributed:** the sets differ in both depth set and vocabulary, and HLCC stored one vocabulary per set.

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
- **On the 18-block set, LC moves the student away from the teacher.** The plain student sits near or above the teacher; LC lowers its NLL to 4.76 and is closer to the teacher on only **1 of 17** rows. HLCC's "18-block benefit" on `UNKNOWN` rows was, on this reading, movement **away** from the teacher.

So "LC lowers the `UNKNOWN` likelihood on V4/20" may be the student moving back toward the teacher. ABST-001 does three things:
1. It replicates the lead on fresh rows.
2. It asks whether LC moves V4/20 toward the teacher on **both** `UNKNOWN` rows and known-answer rows, without overshooting. Generic damage fails the known-row condition.
3. It attributes the sign across the depth × vocabulary corners.

## 2. Hypotheses

- **H1 (primary, cell A = S20/V4, `UNKNOWN` rows).** LC makes the teacher-forced NLL of `UNKNOWN`+`<|im_end|>` **higher** in more rows than lower.
- **H2 (anchor, given H1).**
  - **Step 2:** on `UNKNOWN` rows, LC moves the per-row NLL toward the teacher's (T4, same vocabulary), i.e. |log NLL_lc − log NLL_T| < |log NLL_plain − log NLL_T|, in more rows than away.
  - **Step 2b:** the same holds on **known-answer** rows. There the geometry is reversed: the plain student is less confident in the gold digit than the teacher. So generic damage moves away from the teacher while teacher-directed change moves toward it.
  - **FIDELITY requires both, and** fewer than half the `UNKNOWN` rows may cross the teacher (no overshoot majority).
  - **Caution (review):** given the spent geometry, step 2 alone is nearly implied by H1 unless LC overshoots, and a teacher-blind inflation of every NLL passes step 2 in 99.9% of simulated draws (§6). Step 2 alone is therefore a statement of **direction**, not of fidelity.
- **H3 (attribution).** The signs in B = S20/V2, C = S18/V4 and D = S18/V2, and two paired contrasts, say whether the depth set or the vocabulary carries the sign. D's sign is stated neutrally ("UNKNOWN more/less likely"), never as benefit or harm.

**Population.** The claims cover the RTG1-dev sampling pools (the four families), with the fitted T. They do not cover new generator families, RTG1's reserved holdout, or tasks where the teacher abstains. Under a family random effect, the row-level test falsely confirms for new families at a rate of 0.20 (§6), which is why the claims stay scoped to the pools.

## 3. Panel (fixed; built; read once)

`abst001-20260925/panel/ABST_PANEL.jsonl`, sha256 `133725905b4c641aa316f6ddc7f4447e40a252769793d2b39ebed5dbdf09a17f`, 512 rows. The manifest is `ABST_MANIFEST.json` (sha256 `367b34a90f98c090743dc3b9ec0d2aa1ba17717bf52af02c2a9b5d8a9c60ceac`). The builder is `build_abst_panel.py` (sha256 `76d8b8119a68ddf506e446e1828f101711ee6dca89a8c3ff3db93a3facd384e1`), run under transformers 4.51.3 with `enable_thinking=False`.

- **Pool:** RTG1 dev only. RTG1 holdout is RESERVED and train is training data. CC0 is dropped (§11 D4).
- **Eligibility:**
  - the id is not in LAYER_PANELS, CONFIRM_PANEL or TOKENIZER_CORPUS (96 excluded);
  - not one of the 3 designer-seen rows;
  - the normalised last user turn does not equal any used row's or any RTG1-train row's (0 excluded; the positive control passed);
  - not flagged by the six non-RCI instruments (0 flagged; exact-path 4/4, sentence-path 4/4, negative control clean). RCI-001/002 and HSKILL-CONFIRM-002 are never opened.
- **Sampling:** per family, `bc.ordered(ids, 'abst001:<stratum>:<family>')`, first 64. This is a stratified simple random sample without replacement from fixed pools.
  - `UNKNOWN` stratum: 256 rows, from pools of 112, 109, 109 and 102.
  - Known stratum (reference 0–3): 256 rows, from pools of 468, 462, 461 and 473.
- **Sealed reserve:** 176 `UNKNOWN` rows (48/45/45/38), listed by id. ABST-001 never measures them.

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
- **One row per forward pass (B = 1).** The pool-conditional H0 needs each row's value to depend on that row alone.
- **Every pass is read under every support.** One pass per (config, input vocabulary ∈ {V0, V4, V2}) is read under all three output supports.
- **Stored per row, for every config × input × support:**
  - total and per-token nats;
  - first-position log-probs of UNKNOWN, 0–3 and `<|im_end|>`;
  - the log retained mass;
  - the top-5 ids (matched supports only).
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
  - the manifest's content-duplicate evidence clean, with its positive control passed.
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
- **R5b, tier 2** (the new B = 1 path, same rows and cells): |nats − recorded| ≤ 1e-3. Sign agreement of d(D) is **reported, not gated**: once the nats clause holds it cannot fail (review).
- **R6, measured throughput:** the projection must be ≤ 60 min. Generation is priced as prefill plus 4 decode steps at 64 token-equivalents each, which is an upper bound.
- **Boundary marker.** Just before the panel is read, a create-only, fsynced `attempts/BOUNDARY_<utc>_<pid>.json` is written, holding every control's numbers and the code hashes.

**Shown to fail and to pass** on the tiny random dry model (CPU, 3 rows per stratum):
- **Pass:** fit canaries 0.0; tier 1 0.0 on all five cells; tier 2 at 8.6e-6 to 2.0e-5.
- **Refuse:** a fit residual +1e-5 (R4), a recorded S6 nats +1e-5 (R5) and a wrong kept set (R4). No output directory was created in any of the three.

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
| 2b | known-row anchor (A vs T4), only after TOWARD | one-sided 0.0125 toward the teacher | ≥ **147** | combined with overshoot, see below |
| 3 | Holm at 0.0125 over D±, B±, C±, dV = sign(d_B − d_A) > 0, dD = sign(d_C − d_A) > 0 | Holm | 153, 152, 152, 151, 151, 150, 149, 147 | see below |

**H2 labels:**

| label | condition |
|---|---|
| **FIDELITY** | TOWARD, step 2b rejects, and fewer than half the `UNKNOWN` rows cross the teacher |
| TOWARD_TEACHER_OVERSHOOT | TOWARD, and at least half the rows cross the teacher |
| TOWARD_TEACHER_NOT_CORROBORATED | TOWARD, no overshoot majority, and step 2b does not reject |
| AWAY_FROM_TEACHER | the away tail of step 2 rejects |
| ANCHOR_UNRESOLVED | step 2 rejects in neither direction (the chain stops) |

**Step 3 labels.** These apply only if D+ is rejected (LC makes `UNKNOWN` more likely at S18/V2):

| B sign | C sign | label |
|---|---|---|
| worse | better | DEPTH |
| better | worse | VOCABULARY |
| worse | worse | EITHER_SUFFICIENT |
| better | better | BOTH_REQUIRED |
| one signed, one not | | PARTIAL |
| neither signed | | UNRESOLVED |

If D− is rejected instead, the label is NO_REVERSAL; otherwise D_UNRESOLVED. Steps that are not reached are **UNTESTED**.

**Size** (reported, no gate): a lower confidence bound on the **median per-row ratio NLL_lc/NLL_plain** (order statistic, 97.5%).

**Power and MDE** (exact binomial, n = 256; conservative, because sampling without replacement adds a finite-population correction):

| step | P = 0.58 | 0.60 | 0.62 | 0.65 | 0.70 | MDE (80% power) |
|---|---|---|---|---|---|---|
| step 1 | 0.694 | 0.877 | 0.966 | 0.998 | 1.000 | 0.591 |
| steps 2 / 2b | 0.600 | 0.818 | 0.941 | 0.995 | 1.000 | 0.599 |
| Holm, first step | | | | | | 0.622 |

The lead's own 13/16 has a 95% interval of [0.544, 0.960]. **NOT_CONFIRMED cannot exclude P(worse) below about 0.59.**

**Null calibration of the registered statistic** (`design/ABST_NULL_SIM.json`, seed 20260926, 20,000 draws per null cell). Each draw takes 64 rows per family without replacement from the real pool sizes and runs through `verdict()` itself:

| cell | result |
|---|---|
| N1a / N1b, half worse, odd pools rounded down / up | directional false rates ≤ 0.0012 |
| N2 / N3, heterogeneous nulls | ≤ 0.00015 |
| N4, `UNKNOWN`-anchor null | false step-2 rejection 0.0202 |
| **D1, teacher-blind generic damage** (the review's scenario: every NLL inflated regardless of the teacher, with the spent geometry) | step 2 says TOWARD in **99.9%**, but FIDELITY **0.0000** |
| N5 / N6, Holm false rejections over true nulls | 0.0000 / 0.0016 (bound 0.0125) |
| power | lead shape 1.0, homogeneous 0.60 0.950 (with the finite-population correction), FIDELITY truth 1.0, DEPTH truth 1.0 |
| S1, superpopulation family effect | false LC_LOWERS for new families 0.20 |

**Unit tests** (`tests/test_abst_verdict.py`, `tests/test_abst_describe.py`, 12 pass):
- every label is reachable, and degenerate values never raise;
- the null tests run on the registered statistic, including the teacher-blind-damage test;
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
- by family, and by tokenisation identity;
- **the `UNKNOWN` token alone** (per_token[0]): sign counts, and agreement with the two-token endpoint.

**Other descriptives:**
- the vocabulary factor at depth 20, split into output-support and input terms;
- baseline against endpoint (A against D);
- greedy abstention, false abstention and known accuracy.

**Pre-registered readings (labels, not tests):**
- **M1 (among LC-worse rows in A):** TEACHER_TRACKING if ≥ 0.75 moved closer; LC_ARTEFACT if ≤ 0.25; MIXED otherwise. Like step 2, it is nearly implied by the geometry and is **not independent evidence**.
- **M2:** DISCRIMINATION_LOSS if AUROC_lc < AUROC_plain − 0.03; BIAS_SHIFT if the median Δm < 0 on both strata and |ΔAUROC| ≤ 0.03; OTHER otherwise.
- **Wording rule 1:** if |AUROC_T4 − 0.5| ≤ 0.05, every "abstention" word in the result becomes "`UNKNOWN`-token propensity".
- **Wording rule 2:** if the `UNKNOWN`-token-alone majority sign disagrees with the two-token endpoint's majority, the result says "reference-sequence likelihood", not "`UNKNOWN` likelihood".

## 8. Registered forecasts (scored in the result; not gates; a forecast about a step not reached is NOT_SCOREABLE)

- **F1.** H1 = LC_LOWERS_UNKNOWN.
- **F2.** The cell-A worse fraction is ≥ 0.70.
- **F3.** H2 = FIDELITY.
- **F4.** In cell A, the prerequisite_schedule worse fraction is ≤ 0.50, and each other family's is ≥ 0.70.
- **F5.** D+ is rejected (LC makes `UNKNOWN` more likely at S18/V2).
- **F6.** The attribution label is DEPTH.
- **F7.** M1 = TEACHER_TRACKING. This is nearly implied by F8 and not independent.
- **F8.** Plain V4/20's NLL is below the teacher's on ≥ 90% of `UNKNOWN` rows.
- **F9.** The teacher abstains greedily on ≤ 5% of `UNKNOWN` rows (T4 and T0).
- **F10.** In cell D, LC moves away from the teacher on the majority of `UNKNOWN` rows (spent: 1 of 17 closer).

## 9. What each outcome licenses; read budget; execution

**Every row below carries the residual statement:** *abstention where the teacher abstains (e.g. CC0) is not measured by any registered instrument.*

| outcome | licence |
|---|---|
| LC_LOWERS_UNKNOWN + **FIDELITY** | On the RTG1-dev pools, where the teacher does not abstain, LC moves V4/20 toward the teacher on both `UNKNOWN` and known rows, without an overshoot majority. HLCC's abstention caveat is **withdrawn for those pools**. HLCC's direction-only licence is unchanged. |
| LC_LOWERS_UNKNOWN + TOWARD_TEACHER_OVERSHOOT | LC pushes V4/20 past the teacher on most rows. **Caveat retained**, reworded as "LC over-suppresses `UNKNOWN` on V4/20". |
| LC_LOWERS_UNKNOWN + TOWARD_TEACHER_NOT_CORROBORATED | Direction toward the teacher on `UNKNOWN` rows only; generic damage is not excluded. **Caveat retained.** |
| LC_LOWERS_UNKNOWN + AWAY_FROM_TEACHER | LC pushes V4/20 away from the teacher on `UNKNOWN`. LC is not recommended for V4/20 where `UNKNOWN` is a valid answer, pending a mechanism study. |
| LC_LOWERS_UNKNOWN + ANCHOR_UNRESOLVED | Replicated, not interpreted. **Caveat retained.** |
| LC_RAISES_UNKNOWN | The lead is refuted in direction. Given the geometry, LC moves V4/20 **further from the teacher toward `UNKNOWN`** (the stored anchor counts are reported). The caveat changes to "LC may move V4/20 away from the teacher toward UNKNOWN on RTG1". |
| NOT_CONFIRMED | Not replicated at MDE 0.59. The caveat stays in weakened form ("unresolved"). It is not withdrawn. |
| H3 labels | Say which factor carries the sign. No recommendation changes, and no set is chosen. |

**In every case:** GO/HOLD stays HOLD; no quality claim; no release.

**Read budget: ONE.**
- The PRIMARY cells (A-plain, A-LC, T4; all 512 rows, so steps 1, 2 and 2b are all computable) are measured first. `ABST_PRIMARY.json` (steps 1–2) is written create-only before any other cell.
- **Startup refusals.** A launch refuses if:
  - any `*/ABST_PRIMARY.json` exists under the records parent;
  - two BOUNDARY markers exist;
  - one marker exists and any attempt record is not classified as infrastructure.
- **After a failure following the boundary:** one same-command rerun is permitted only after an infrastructure failure (or a hard kill that left no record), and only before `ABST_PRIMARY.json` exists. After the primary exists there is no rerun, and unmeasured parts are NOT_MEASURED.
- Every post-boundary failure writes `attempts/ATTEMPT_*.json`, holding the refusal-control evidence, every finished cell and the rows of the cell in progress.
- **Tested on the dry model** (scratch harness):
  - a failure before the primary gave a record and no primary;
  - a failure after it left `ABST_PRIMARY.json` standing.
  - The cross-launch rules are exercised by `harness_abst_budget.py`, which must pass before the GPU run (§10).

**Execution — a hash chain:**
- `launch/launch_abst.ps1` verifies `launch/abst_run.sh`'s sha256 before the WMI create.
- `abst_run.sh` verifies `launch/abst_hashes.txt`'s own sha256, then `sha256sum --strict -c` over every gated file:
  - the 13 local modules (`abst_verdict`, `abst_measure`, `abst_describe`, `run_abst`, `hlcc_verdict`, `linear_compensation`, `virtual_student`, `layer_ablation`, `layer_select`, `run_layer_study`, `param_accounting`, `gpu_lease`, `hostbudget`);
  - the panel and manifest; LAYER_PANELS; TOKENIZER_CORPUS; S1; S6; LC_ARMS; HLCC_RESULT; CONFIRM_PANEL;
  - the V4/V2 token maps and tokenizers;
  - the teacher's tokenizer, config and weights.
- The only retry is on the GPU-reset signature, before the primary exists.

**Resources:**
- lease `abst001`;
- at least 3.6 GiB free host RAM, stand-down below 1.5, CUDA fraction 0.78;
- about 2.0M panel forward tokens, about 1.0M generation token-equivalents and about 4M for the controls (SELECT identity/fits and the canaries);
- estimated 25–40 min, with R6 refusing above 60;
- disk writes under 20 MB.

**Queue:** after the peers' queued jobs, in the order agreed with them.

## 10. Code

`abst_verdict.py` · `abst_measure.py` · `abst_describe.py` · `run_abst.py` · `build_abst_panel.py` · `design/abst_null_sim.py` + `design/ABST_NULL_SIM.json` · `tests/test_abst_verdict.py` · `tests/test_abst_describe.py` · `launch/abst_run.sh` · `launch/launch_abst.ps1` · `launch/abst_hashes.txt`. The sha256s of the three launch files are recorded in the registration commit message.

**Dry runs on the registered code** (tiny random 6-layer teacher, CPU; the 20-block stand-in is [0, 1, 3, 4], the 18-block stand-in [0, 1, 3]):
- **Done:**
  - end to end, with the primary equal to the final steps 1–2 and the records complete;
  - all three negative controls refuse before the boundary, with no result or attempts directory;
  - a failure injected before the primary gives one attempt record and no primary.
- **PENDING, two checks.** Both test code behaviour only and see no data:
  - a failure injected **after** the primary (it passed on the pre-review code);
  - the cross-launch budget harness (`harness_abst_budget.py`: a deterministic failure blocks a rerun; one rerun after an infrastructure failure; a third launch or a different `--out` refuses).
- **Why they are pending.** At registration the host's free RAM swung between 0.06 and 4.3 GiB, held by desktop apps. The CUDA build of torch commits about 3 GB at import, so every attempt refused at its own 0.8 GiB gate. The gate was not lowered.
- **What happens next.** Both checks run before the GPU run. The run does not launch until both pass. Any code change they force is a disclosed pre-run revision, with the hash chain regenerated.

## 11. Disclosures

- **D1, panel built before registration.** Built CPU-only; no forward pass of the real teacher.
- **D2, dry runs touched panel rows.** The dry runs and harnesses ran a **random** tiny model on 24 panel rows (3 per stratum). Their outputs carry no information about the real teacher.
- **D3, canaries on spent SMALLVD HOLDOUT rows.** R5 re-reads SMALLVD-001's spent HOLDOUT reference rows to reproduce recorded numbers (precedent: LC_ARMS `reproducibility_vs_S5`). HLCC's panel is **not** touched (its §9 forbids any rerun). The operator may overrule this before the run.
- **D4, CC0 dropped.** Hygiene review found:
  - 17 of the 86 eligible fresh CC0 `unknown` rows already have model predictions in another lane;
  - packets overlap the seen panels;
  - only 45 rows pass the length filter.

  CC0 is the only place this teacher abstains (cue-driven, one template). Abstention where the teacher abstains is therefore **not measured**, and every licence says so.
- **D5, the teacher-anchor frame comes from spent design data** (§1, spent rows only). This includes the finding that LC at S18 moves **away** from the teacher (1 of 17 closer). That finding was surfaced by the pre-registration review, not by the designers. The anchor tests were added because of this frame, which is admissible before any fresh read.
- **D6, designer-seen rows.** Three fresh RTG1 known-answer rows appeared in design-agent tool output. They are excluded by id. No design or review agent ran a model or wrote a file: one design command loaded tokenizers only.

## 12. Pre-registration review (adopted before commit)

**Review:** four lens reviewers (statistics, runner, measurement, interpretation) and up to four refutation attempts per lens (eight were needed), all read-only. **Adopted:**

- **The anchor is nearly implied by H1** (statistics and interpretation). Step 2b and the no-overshoot condition were added, and FIDELITY is conjunctive. The teacher-blind-damage simulation cell and unit test were added.
- **Wording for LC at S18.** LC at S18 moves away from the teacher on spent data. The neutral D± wording is used, with F10 and D5.
- **Licence table.** It was inconsistent. Every row now carries the CC0 residual; LC_RAISES is the away direction; NOT_CONFIRMED keeps a weakened caveat.
- **Read budget across launches.** A BOUNDARY marker was added, with startup refusal rules.
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
