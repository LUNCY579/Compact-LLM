# HLCC-001 — confirming linear compensation on a fresh panel

**Status:** REGISTERED 2026-09-25 (267cec9b). REVISED **R1** (d321280c) and **R2** (this commit), both from Rineya Delta's cold review and both before any run. The real teacher has not run a single forward pass on the confirmation panel. The panel is already built and pinned by hash below; it was built CPU-only (disclosures D1–D7, revision log §12).
**Branch:** `research/rineya-small-vocab-depth-distill-v1` · **Records:** `rineya-dataset-retention-records/hlcc001-20260925/` · **Cold review:** `hlcc001-20260925/HLCC_PREREG_COLD_REVIEW.md` (records cdbf844).
**Parent:** SMALLVD-001 (Amendments 2 and 5; `smallvd001-20260924/lc_arms/LC_ARMS.json`).

## 1. Why this experiment exists

SMALLVD-001's exploratory arm H-LC fits a ridge least-squares map T for each run of dropped blocks. T is folded into the down_proj of the kept block just before the run (W ← TᵀW), so the student needs no extra parameters.

- **Registered verdict: UNCALIBRATED.** The observed per-row spread `s_ratio_hat = 1.25` fell outside every calibrated band (≤ 0.8). The shape was also OUT_OF_FAMILY: skewed, heavy-tailed, and correlated with plain KL.
- **Descriptive result: large.** On the SMALLVD-001 HOLDOUT, r = 0.59. Answer bpb fell from 1.765 to 1.272. LC beat the plain student on KL in 253/257 rows (median ratio 2.41×) and on answer nats in 171/186 rows.

Neither description can confirm itself: those rows have been seen, and the test that was registered could not be calibrated for that shape. HLCC-001 therefore changes three things:

- **The data:** a fresh panel, disjoint from everything SMALLVD-001 read.
- **The test:** an exact test whose validity needs no shape assumption.
- **The student:** the depth set the programme actually recommends (SMALLVD-001 §5: V4_32768/20), not only the 18-block set on which H-LC was exploratory.

## 2. Hypothesis and population

**H-LCC.** Take SMALLVD-001's compensation (same method, same fit set) and fold it into the recommended depth-20 student. Then, among fresh rows drawn from the panel's sampling pools, more rows improve than worsen on:

- (i) per-row teacher→student KL;
- (ii) per-row negative log-likelihood of the reference answer under the student's V4 vocabulary.

Each comparison is against the same student without folds.

**Population (R2).** The claim covers **the sampling pools the panel was drawn from**, conditional on the fitted T. It does not cover new books, generator families, generator tasks or code packages. The panel shares most of its generators with the fit set: long-context tasks 11/11, RTG1 families 4/4, v3 classes 28/31. Only 1 of its 19 books is in the fit set, so books are the only source-fresh axis. No unit rule could make this panel license generalisation to new generators. The all-axis sensitivity analysis (§6) reports how the direction holds up when shared generators are collapsed.

## 3. Panel (fixed; already built; read once)

`hlcc001-20260925/panel/CONFIRM_PANEL.jsonl`, sha256 `8eec83da9fccfc7e2a253ebf2ce4fab51be319a4c1def8ce1e196343602c0220`. The manifest is `CONFIRM_MANIFEST.json`. The builder is `build_confirm_panel.py`, sha256 `c8d2c6f2…274ce`; it ran under transformers 4.51.3 with `enable_thinking=False`.

- **Composition:** 320 rows, 32 in each of the ten SMALLVD-001 categories (four v3 chat groups, abstention_unknown, long_context, four text categories). 192 chat rows carry a reference answer, 18 of which are `UNKNOWN`.
- **Sampling:** each category is `bc.ordered(ids, salt)`, a content-independent hash order that takes the first 32 eligible rows. That is a simple random sample, without replacement, from a fixed pool. RTG1 is stratified 16/16 on UNKNOWN, and v3 is one row per fresh family.
- **Disjointness, recomputed from the written rows:**
  - 0 ids shared with LAYER_PANELS (SELECT or HOLDOUT).
  - 0 chat ids shared with the tokenizer corpus.
  - 0 text rows drawn from any panel's source rows.
  - 0 v3 template families shared with any row SMALLVD-001 used.
  - v3 rows are one per family.
- **Content check against the eight evaluation instruments:** exact-path control 4/4, sentence-path control 4/4, negative control clean. 0 rows were flagged.
- **Text rows:** drawn from the tokenizer corpus's HELD-OUT split. Before HLCC-001, only the tokenizer's held-out token-ratio measurement ever touched them — never layer selection or the compensation fit.

## 4. Students and compensation (no free parameter)

| role | kept blocks (of 28) | removed | fold points p (run) | vocabulary for bpb |
|---|---|---|---|---|
| **PRIMARY** — the recommended V4_32768/20 | 0,1,2,3,4,7,9,10,11,15,16,17,18,19,20,21,22,25,26,27 | 5,6,8,12,13,14,23,24 | 4 (5–6), 7 (8), 11 (12–14), 22 (23–24) | V4_32768 |
| SECONDARY — the registered 18-block A2/A3 set | 0,1,2,3,4,9,10,11,15,16,17,18,20,21,22,25,26,27 | 5,6,7,8,12,13,14,19,23,24 | 4 (5–8), 11 (12–14), 18 (19), 22 (23–24) | V2_65536 |

**Compensation:** exactly SMALLVD-001 Amendment 2, `linear_compensation.fit_compensation` (sha256 `a8abc72e…a637d`):

- **Fit set:** LAYER_PANELS SELECT rows (all 255) with the S1 teacher continuations. Nothing from the confirmation panel is used.
- **Method:** ridge toward I, λ = 1e-3 × the median eigenvalue of XᵀX, position 0 excluded, fp32 model, folds applied cumulatively.
- **Settings:** no refit, no λ choice, no set choice.

**Refusal controls (the run exits and writes nothing).** Controls (a)–(c) all run **before the real teacher performs any forward pass on the confirmation panel** (R1, R2), so a refusal never spends the panel's read. The console line `refusal controls passed ... the confirmation panel is read now` marks the boundary.

- **(a) Identity fold.** For each set, folding I must reproduce the plain student's hidden states bitwise, on the SELECT rows.
- **(b) Fit canary.** The SECONDARY set's fresh fits must reproduce SMALLVD-001's recorded full-precision `fit_residual_rel_fitted` at every fold point within **1e-6**. The recorded values are 0.342781, 0.449361, 0.388918 and 0.214652 at p = 4, 11, 18, 22.
  - The canary is required (`--smallvd-lc` is mandatory).
  - It refuses a mismatched kept set rather than skipping, and an empty fit list gives ∞, which refuses.
  - **It can fail and it can pass** (`design/hlcc_canary_variants.py`, tiny dry model; cold review F3):

    | fit | largest residual difference | canary |
    |---|---|---|
    | registered method | 0.0 | passes |
    | ridge doubled (`ridge_rel` 2e-3) | 1.94e-5 | trips |
    | ridge halved (5e-4) | 9.24e-6 | trips |
    | position 0 included | 3.53e-3 | trips |
    | one recorded residual +1e-5 | — | trips |
    | wrong kept set | — | refuses |

  - **Pre-registered route after a canary failure on the GPU** (the panel stays unread):
    1. No HLCC run until an amendment is registered.
    2. The only admissible change is the tolerance. It must rest on a GPU diagnostic session that reads **SELECT rows only**: two fresh fits of the SECONDARY set (run-to-run spread s, together with the recorded values), plus the three method variants above fitted on the real model.
    3. The new tolerance is 10 × max(s, the difference from the recorded values). It must stay below **half** the smallest real-model variant shift.
    4. If no tolerance meets both conditions, HLCC-001 is recorded as **CANARY_UNRESOLVED** with the panel unspent.
- **(c) Data checks.**
  - Rows whose split is not `confirm`, or any id overlap with LAYER_PANELS, refuse the run.
  - The all-axis unit map must match the panel's hash and cover every row.
  - Every reference answer, re-tokenised by V4 and by V2, must stay inside that vocabulary together with `<|im_end|>`. This is `answer_bpb`'s assertion, checked on the text before any forward pass (R2, F4).

**After the boundary** (R2, F4):

- Any failure writes a create-only `H/attempts/ATTEMPT_<utc>_<pid>.json` before exiting. The record holds the exception, the traceback, a classification hint and every per-row value measured so far.
- A non-positive KL still refuses (it is never clipped); it now does so with a record.
- An infrastructure failure (GPU reset, OOM, RAM stand-down) permits the same-command rerun (§9).
- A deterministic failure ends the experiment as **NO VERDICT, READ SPENT**, and is reported as such.
- Tested by an injected failure (scratch harness):
  - a deterministic failure writes 1 record, holding the 30 PRIMARY plain rows measured so far;
  - an injected stand-down writes 1 record, labelled infrastructure;
  - a pre-boundary canary refusal writes none.

## 5. Measures

For each row i, with d > 0 meaning compensation is better:

- **d_kl,i = log(KL_plain,i / KL_lc,i)** over all 320 rows.
  - Per-position teacher→student KL, averaged over the row's scored positions.
  - Chat rows: the teacher's own greedy 64-token continuation, regenerated here under SMALLVD-001 S1's settings, plus `<|im_end|>`.
  - Text rows: every position (≤ 512 tokens).
  - Measured through the **teacher head and vocabulary**. This is the **V0/20** student (depth only), exactly as in SMALLVD-001.
- **d_bpb,i = log(nats_plain,i / nats_lc,i)** over the 192 chat rows with a reference answer.
  - The reference answer's total NLL under teacher forcing, with prompt and reference re-tokenised in the set's vocabulary: V4 (32,768 retained ids) for PRIMARY, which is the **V4/20** student itself.
  - T was fitted on V0-tokenised SELECT rows, so this limb also tests whether the compensation **transfers across tokenisation**. A bpb failure does not by itself mean "LC does not work at depth".

## 6. The registered test (R2)

**H0 (pool-conditional).** Condition on the fixed pools and on the fitted T, which never sees the panel. Each row's d is then a fixed function of the row, and the sampled rows' signs are **independent by construction**, whatever book, task or package a row came from.

H0: in the pools, a row is as likely to worsen as to improve (P(d > 0) ≤ ½, pooled over the fixed category sizes).

**Test.** For each measure, a one-sided exact **sign test over rows**:

- Rows with an exact 0 are dropped and counted.
- α = **0.0125 each**: two co-primary tests, Bonferroni to 0.025 one-sided, valid under any dependence between the limbs.
- The test is exact for i.i.d. signs. It is conservative under stratification with category-specific rates and under sampling without replacement.
- It needs no symmetry, no shape family and no resampling.
- Measured under a heavily skewed null whose median is 0 but whose mean is about +2.1, the false-CONFIRM rate is ≤ α. Mine: 0.0083 (KL) and 0.0080 (bpb) over 4,000 simulations at the panel's sizes. The review's: 0.0100 and 0.0105.

**Verdict** (`hlcc_verdict.verdict`), on the PRIMARY set:

| verdict | condition |
|---|---|
| **HARMFUL** | either test rejects toward compensation-WORSE at α (checked first) |
| **CONFIRMED** | both tests reject toward compensation-better at α |
| **PARTIAL** | exactly one test does |
| **NOT_CONFIRMED** | otherwise |

With no ties, CONFIRMED needs at least **181/320** positive KL rows (56.6%) and **113/192** positive bpb rows (58.9%).

**Size annotation** (CONFIRMED only). `size_large` is true when both distribution-free order-statistic lower confidence bounds on the row median of d (1 − α) reach log 1.25 — a median per-row reduction of at least 20% on both measures. Each bound is also reported as a minimum median reduction factor.

**Power** (exact, `power_sign_test`):

| P(d > 0) | 0.58 | 0.60 | 0.62 | 0.65 | 0.70 |
|---|---|---|---|---|---|
| KL, 320 rows | 0.719 | 0.905 | 0.980 | 0.999 | 1.000 |
| bpb, 192 rows | 0.435 | 0.656 | 0.835 | 0.967 | 1.000 |

SMALLVD-001's descriptive positive fractions were 0.98 (KL) and 0.92 (bpb). A NOT_CONFIRMED is uninformative for effects with P(d > 0) below about 0.595 (KL) or 0.615 (bpb), where power falls below 0.8.

**Registered sensitivity analysis: all-axis units, majority sign, direction only.**

- **Units:** the connected components of rows sharing **any** generator or source axis. The axes are book, author, code package, markdown document, long-context task, source packet and anchor card, RTG1 family, and v3 class.
- **The map:** `hlcc001-20260925/HLCC_ALLAXIS_UNIT_MAP.json`, sha256 `b5137cde…a64`, built by `design/hlcc_allaxis_units.py` from provenance recovered by the cold review.
  - Its canary reproduced the review's counts exactly: 127 / 42, and **121 / 36** with anchor cards. The anchor-card maps are the registered ones.
  - All 32 long-context rows form a single unit.
- **Unit statistic:** sign(#rows improving − #rows worsening), with equal counts dropped as ties. It never averages, so it holds α where a unit *median* does not (§12 R2).
- **Test:** the same sign tests and verdict labels, with no size annotation.
- **Power:** KL 121 units, 0.985 at P = 0.70; bpb 36 units, only 0.466 at 0.70 and 0.911 at 0.80.
- **Reading:** it is reported beside the verdict and never replaces it. If it disagrees, the claim is stated as "holds for these pools; not shown to survive collapsing shared generators".

**SECONDARY set.** The same test, reported as a replication. It does not enter the verdict. PRIMARY is measured first (R2).

**Descriptive only** (never a verdict and never a selection):

- The 267cec9b unit rule's verdict (`unit_of` + unit median), kept for continuity; it is withdrawn as a test (§12).
- The category-mean KL objective and its ratio.
- Answer bpb under V0, V4 and V2.
- The generation flip rate against the teacher. These are **V0/20** generations.
- Per-category KL-better counts.
- The shape of log(LC/plain) with SMALLVD-001's A5.8 statistics, including corr(log KL_plain, log(LC/plain)).
- Per-row d.

## 7. Registered forecasts (scored in the result; not gates)

- **F1.** PRIMARY positive fraction over KL rows ≥ 0.90.
- **F2.** PRIMARY positive fraction over bpb rows ≥ 0.80.
- **F3.** `size_large` is true.
- **F4.** corr(log KL_plain, log(LC/plain)) < 0: the rows the plain student gets most wrong gain the most.
- **F5.** SECONDARY is also CONFIRMED.
- **F6.** The LC flip rate (V0/20 generations) is below the plain flip rate on PRIMARY.
- **F7.** The all-axis sensitivity analysis agrees in direction on KL.

## 8. What each verdict licenses (scoped, R2 / F5)

Every claim reads: "among rows from these pools, with this T, more rows improve than worsen", plus the size bound.

- **CONFIRMED with `size_large`.**
  - The LC-folded **V4/20** becomes the recommended **initialisation** for any future recovery or distillation training.
  - For the actual student this rests on the **bpb limb**. The KL limb is evidence about its depth component (V0/20 through the teacher head).
  - The use still requires G5, a qualified training lane, the operator's teacher decision and a grant permitting architectural change.
- **CONFIRMED without `size_large`.** Only the **direction** is licensed; the initialisation recommendation does not change. CONFIRMED is reachable with 43% of rows worse.
- **PARTIAL.**
  - **KL only:** the depth-20 student through the teacher head is closer to the teacher's distribution. No claim is made for the V4 student or for answers.
  - **bpb only:** the V4/20 student's reference-answer likelihood improves (including the transfer across tokenisation). No claim is made about distributional fidelity.
- **NOT_CONFIRMED.** SMALLVD-001's descriptive effect did not replicate at the power above. LC is not recommended.
- **HARMFUL.** LC is barred for this student.
- **NO VERDICT, READ SPENT** (a deterministic failure after the boundary). Reported with its attempt record. There is no rerun.

**In every case:**

- GO/HOLD stays **HOLD**: none of its missing limbs is touched.
- There is no quality claim and no release.
- The result is not used to choose among sets, λ values or fold points: there is nothing to choose.

## 9. Read budget, execution, resources

- **Read budget:** the panel is read **once** by the real run.
  - After an **infrastructure** failure (GPU reset, OOM, RAM stand-down), the same command may be rerun on the same code. The attempt record and any partial console lines are disclosed.
  - No other rerun is permitted.
- **Registered command** (R = `smallvd001-20260924`, H = `hlcc001-20260925`, T = the Rineya-Supa-1.7.0-devfreeze teacher), in `.venv-rab001-cu126`, detached through the WMI runner by a hash-gated script:

```
python -B run_hlcc.py --teacher T --panels R/calibration/LAYER_PANELS.jsonl --study R/layer_study \
  --confirm H/panel/CONFIRM_PANEL.jsonl --vocab-primary R/vocab/V4_32768 --vocab-secondary R/vocab/V2_65536 \
  --primary-kept 0,1,2,3,4,7,9,10,11,15,16,17,18,19,20,21,22,25,26,27 \
  --secondary-kept 0,1,2,3,4,9,10,11,15,16,17,18,20,21,22,25,26,27 \
  --smallvd-lc R/lc_arms/LC_ARMS.json --allaxis-map H/HLCC_ALLAXIS_UNIT_MAP.json --out H/result --device cuda
```

- **Resources:**
  - GPU lease label `hlcc001`.
  - Host RAM: at least 3.6 GiB free to start (`hostbudget.require`), stand-down below 1.5 GiB, CUDA memory fraction 0.78 (no WDDM spill).
  - **Order:** after the soup lane's "ended", CLOCKFIX-001 runs first (agreed with Rineya Delta, because R2 changed code). HLCC runs after CLOCKFIX's "ended".
  - Estimated 30–45 min: SMALLVD-001's LC arms took 643 s and 410 s per set on 257 HOLDOUT rows.
- **Output:** `H/result/HLCC_RESULT.json`, created only at the end (create-only). It records the panel, map and code sha256s.

## 10. Code

| file | sha256 |
|---|---|
| `hlcc_verdict.py` (R2) | `baf465b803c25d26d95d70c90dc138e68b774d617bd4e18932e4bb7f90952531` |
| `run_hlcc.py` (R2) | `7cbfa2391cfa96f1dbfae37e6ddfe5bc2b803aff76c2e97560ee233dd116981d` |
| `build_confirm_panel.py` | `c8d2c6f2ceac5a3bd523020bf0710dc00d12b779d4f7aa1730747f26675274ce` |
| `design/hlcc_allaxis_units.py` | `41fdf365d55d19b14dbfe33b7558b41af7f02e82c637ce0e8a9e485d78f877cc` |
| `design/hlcc_canary_variants.py` | `b5a41be2fbda0b78afef6ad82cf3f89cc83dd8fad816361a9e5d27452b223940` |
| `tests/test_hlcc_verdict.py` (11 pass) | `8153e96dc81178ac13842c26d928d0190c8c26469840763251379191bbcbde4d` |
| data: `HLCC_ALLAXIS_UNIT_MAP.json` | `b5137cde168a667619fa31edea45b7cd5c15b1791b8268601eafa2baef407a64` |

**Dry run** (tiny random 6-layer teacher, CPU, 3 rows per category, R2 code):

- It ran end to end, with PRIMARY measured first.
- The canary reproduced exactly (difference 0.0).
- Every per-row d and every fit was bitwise identical to the R1 dry run.
- The row-level verdict dict equals R1's `row_level_sensitivity`, except for the added `with_size` key.

## 11. Disclosures

- **D1 — panel built before registration.** The panel was built before this document, on the CPU, with no forward pass of the real teacher. Its hash is pinned above.
- **D2 — dry runs touched panel rows.** The dry runs (267cec9b, R1, R2) and the R2 harness ran a **random** tiny model on 30 panel rows (3 per category). Their outputs carry no information about the real teacher.
- **D3 — empty probe directory.** A mistyped probe of the builder created an empty directory `H/panel_probe_do_not_use/`: the builder creates its output directory before loading the tokenizer. Git does not track empty directories. It is left in place, because nothing may be removed from the records tree without operator approval.
- **D4 — builder flaw kept.** The builder still has that early-mkdir flaw. It was not fixed, so that the manifest's `builder_sha256` matches the committed file.
- **D5 — the 267cec9b unit rule (revised text).**
  - **When:** the rule was added after the panel was built. At that point only row ids and their metadata had been inspected, and no model output of any kind.
  - **Adopted:** book (19 behind the 32 prose rows) and long-context task (11).
  - **Considered and rejected:** v3 class (visible in each row's `source` field, up to 12 rows per class), on the grounds that the rows are one per template family.
  - **Never examined:** RTG1 family (4 families behind 32 rows), author, source packet, anchor card and code package. They are not in the panel rows.
  - **Rejection was wrong:** that was an error, not a decision, and the cold review's F1 measured its cost. The rule's "row-level is anticonservative" figure (0.029) holds only under a superpopulation reading; under the design the panel actually has, the row-level test holds α.
- **D6 — the cold reviewer's inspection** (from the review, §7):
  - The reviewer read metadata and source provenance for all 320 rows: ids, categories, sources, RTG1 families, cc0 `source_provenance`, and code/markdown paths recovered by hashing.
  - One exploratory print showed the first ~120 characters of **two** v3 prompts and their references ("Yes." and "Yes, they conflict.").
  - The reviewer saw no model output of any kind. The all-axis map derives from that inspection.
- **D7 — R2 was adopted after the reviewer's inspection, before any model output.** Its evidence was the review's CPU simulations and my independent recomputation, which used a different seed and my own code:
  - unit-median false CONFIRM 0.0152 (KL) and 0.0150 (bpb), against row-level 0.0083 and 0.0080;
  - the mechanism, exactly: the mean of two i.i.d. skewed-null draws is positive 63.5% of the time;
  - RTG1 families 11/9/6/6.

  R2 moves the registered test to the **more powerful** statistic. That is admissible only because no output exists and the choice follows from how the panel was sampled. It is disclosed so that it cannot be mistaken for a post-hoc switch.

## 12. Revisions before the run

- **R1 (2026-09-25, d321280c; cold review, before any real run).** In the 267cec9b code, the fit canary and the identity control ran only *after* the real teacher had generated on the confirmation panel and collected its hidden states. A canary refusal would therefore have spent the panel's single read and recorded nothing.
  - Both controls, and both sets' fits, now run on SELECT rows before the panel is touched.
  - The identity control checks the fold mechanism on SELECT rows instead of panel rows.
  - Nothing measured changed: rerun on the dry model, every per-row d, per-unit d and fit was bitwise identical to the 267cec9b code's dry run.
- **R2 (2026-09-25, this commit; cold review F1–F6, before any real run).**
  - **Registered test → row-level, with a pool-conditional H0 (F2, option A).** The 267cec9b unit rule held α under neither coherent reading.
    - Under the superpopulation reading it collapsed only 2 of the panel's 8 shared-generator axes. With a shared effect on the other axes, the review measured false-CONFIRM rates of 0.06–0.17 (F1).
    - Under the design-based reading, the median of an even-sized unit is the **mean** of its middle rows. That mean is positive more than half the time under a skewed null. The 267cec9b skewed-null test never called `collapse()`, so it never tested the registered statistic.
    - `tests/test_hlcc_verdict.py` now tests that statistic directly: 150 two-row units under the skewed null give a unit-median false-confirm rate > 0.5, while the row-level test and majority-sign units hold α.
  - **All-axis majority-sign units → the registered sensitivity analysis.** The map is pinned by hash, and its canary reproduced the review's counts (§6).
  - **Refusals after the boundary → create-only attempt records (F4).** The `answer_bpb` vocabulary assertion is now checked on text before the boundary.
  - **PRIMARY is measured first** (review, minor).
  - **Fit canary (F3).** Shown to discriminate real method variants on the dry model. The route after a benign GPU failure is pre-registered (§4).
  - **§8 licensing scoped (F5):** the KL limb is V0/20; the bpb limb is V4/20, including the transfer across tokenisation; the initialisation recommendation requires `size_large`; flip rates are V0/20.
  - **Disclosures D5 revised; D6 and D7 added (F6).**
