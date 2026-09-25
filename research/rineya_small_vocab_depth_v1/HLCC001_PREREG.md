# HLCC-001 — confirming linear compensation on a fresh panel

**Status:** REGISTERED 2026-09-25. Registered before the real teacher has run a single forward pass on the confirmation panel. The panel is already built, pinned by hash below, CPU-only (disclosures D1–D5).
**Branch:** `research/rineya-small-vocab-depth-distill-v1` · **Records:** `rineya-dataset-retention-records/hlcc001-20260925/`
**Parent:** SMALLVD-001 (Amendments 2 and 5; `smallvd001-20260924/lc_arms/LC_ARMS.json`).

## 1. Why this experiment exists

SMALLVD-001's exploratory arm H-LC fits a ridge least-squares map T for each run of dropped blocks. T is folded into the down_proj of the kept block just before the run (W ← TᵀW), so the student needs no extra parameters.

- **Registered verdict: UNCALIBRATED.** The observed per-row spread `s_ratio_hat = 1.25` fell outside every calibrated band (≤ 0.8). The shape was also OUT_OF_FAMILY: skewed, heavy-tailed, and correlated with plain KL.
- **Descriptive result: large.** On the SMALLVD-001 HOLDOUT, r = 0.59. Answer bpb fell from 1.765 to 1.272. LC beat the plain student on KL in 253/257 rows (median ratio 2.41×) and on answer nats in 171/186 rows.

Neither description can confirm itself: those rows have been seen, and the test that was registered could not be calibrated for that shape. HLCC-001 therefore makes three changes:

- **The data:** a fresh panel, disjoint from everything SMALLVD-001 read.
- **The test:** an exact test whose validity needs no shape assumption.
- **The student:** the depth set the programme actually recommends (SMALLVD-001 §5: V4_32768/20), not only the 18-block set on which H-LC was exploratory.

## 2. Hypothesis

**H-LCC.** On rows none of SMALLVD-001's stages ever read, folding SMALLVD-001's compensation (same method, same fit set) into the recommended depth-20 student reduces its per-row teacher→student KL. It also reduces the per-row negative log-likelihood of the reference answer under the student's V4 vocabulary. "Reduces" means for the median independent unit, compared with the same student without folds.

## 3. Panel (fixed; already built; read once)

`hlcc001-20260925/panel/CONFIRM_PANEL.jsonl`, sha256 `8eec83da9fccfc7e2a253ebf2ce4fab51be319a4c1def8ce1e196343602c0220`. The manifest is `CONFIRM_MANIFEST.json`. The builder is `build_confirm_panel.py`, sha256 `c8d2c6f2…274ce`; it ran under transformers 4.51.3 with `enable_thinking=False`.

- **Composition:** 320 rows, 32 in each of the ten SMALLVD-001 categories (four v3 chat groups, abstention_unknown, long_context, four text categories). 192 chat rows carry a reference answer, 18 of which are `UNKNOWN`.
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

- **Fit set:** LAYER_PANELS SELECT rows with the S1 teacher continuations. Nothing from the confirmation panel is used.
- **Method:** ridge toward I, λ = 1e-3 × the median eigenvalue of XᵀX, position 0 excluded, fp32 model, folds applied cumulatively.
- **Settings:** no refit, no λ choice, no set choice.

**Refusal controls (the run exits and writes nothing).** Controls (a) and (b) run on SELECT rows, for both sets, **before the real teacher performs any forward pass on the confirmation panel** (Revision R1). A refusal therefore never spends the panel's read. The console line `refusal controls passed ... the confirmation panel is read now` marks the boundary.

- **(a) Identity fold.** For each set, folding I must reproduce the plain student's hidden states bitwise, on the SELECT rows.
- **(b) Fit canary.** The SECONDARY set's fresh fits must reproduce SMALLVD-001's recorded `fit_residual_rel_fitted` at every fold point within 1e-6. The recorded values are 0.342781, 0.449361, 0.388918 and 0.214652 at p = 4, 11, 18, 22. The canary set runs first, so a failure costs one fit.
  - It is required (`--smallvd-lc` is mandatory).
  - It refuses a mismatched kept set rather than skipping.
  - Dry run: it reproduced the tiny model's recorded fits with a difference of 0.0. With one recorded residual perturbed by 1e-5 it refused, and with a wrong kept set it refused; both refusals left no output.
- **(c) Split and KL checks.** Rows whose split is not `confirm`, or any id overlap with LAYER_PANELS, refuse the run. A non-positive KL refuses too (never clipped).

## 5. Measures

For each row i, with d > 0 meaning compensation is better:

- **d_kl,i = log(KL_plain,i / KL_lc,i)** over all 320 rows.
  - Per-position teacher→student KL, averaged over the row's scored positions.
  - Chat rows: the teacher's own greedy 64-token continuation, regenerated here under SMALLVD-001 S1's settings, plus `<|im_end|>`.
  - Text rows: every position (≤ 512 tokens).
  - Measured in the teacher vocabulary, so it isolates the depth effect exactly as SMALLVD-001 did.
- **d_bpb,i = log(nats_plain,i / nats_lc,i)** over the 192 chat rows with a reference answer.
  - The reference answer's total NLL under teacher forcing, in the set's vocabulary (V4 for PRIMARY).
  - This is the combined student's quantity.

## 6. The registered test

**Units.** The sign test is exact only for independent signs, and the panel's rows are not all independent: the 32 prose rows come from 19 books, and the 32 long-context rows from 11 generator task types. The test therefore runs on units, defined by `hlcc_verdict.unit_of`:

- **Text rows:** the source document (prose and markdown drop the chunk index; py/js ids are one file each).
- **Long-context rows:** the generator task (`rineya_v3_sft__<task>__…`).
- **All other rows:** the row itself. v3 rows are one per template family; the 32 RTG1 rows have 32 distinct `structure_id`s.

Each unit contributes the median of its rows' d. This gives **284 KL units** and **171 bpb units**.

**Why units, measured** (`tests/test_hlcc_verdict.py`). The null draws a shared shift per cluster (median 0) at the panel's real cluster sizes:

| within-cluster noise | row-level sign test (false CONFIRM) | unit-level sign test (false CONFIRM) |
|---|---|---|
| 0.05 | 0.0290 | 0.0123 |
| 0.5 | 0.0193 | 0.0083 |

At the tight setting the row-level rate is 2.3× α; the unit-level test holds α (0.0125). The row-level test is still computed, as a **sensitivity analysis only**.

**Test.** For each measure, a one-sided exact **sign test** on the unit medians:

- H0: P(unit median > 0) = P(unit median < 0).
- Units with an exact 0 carry no sign and are dropped (the count is recorded).
- α = **0.0125 each** (two co-primary tests, Bonferroni to 0.025 one-sided).
- The test needs no symmetry, no shape family and no resampling.
- Measured under a heavily skewed null whose median is 0 but whose MEAN is about +2.1: the false-CONFIRM rate stays ≤ α in both skew directions. So the estimand really is the median, and a mean-based reading would have rejected.

**Verdict** (`hlcc_verdict.verdict`), on the PRIMARY set:

| verdict | condition |
|---|---|
| **HARMFUL** | either test rejects toward compensation-WORSE at α (checked first) |
| **CONFIRMED** | both tests reject toward compensation-better at α |
| **PARTIAL** | exactly one test does |
| **NOT_CONFIRMED** | otherwise |

With no ties, CONFIRMED needs at least 162/284 positive KL units and 101/171 positive bpb units.

**Size annotation** (CONFIRMED only). `size_large` is true when both distribution-free lower confidence bounds on the median unit d (order statistics, 1 − α) reach log 1.25 — that is, a median reduction of at least 20% on both measures. Each bound is also reported as a minimum median reduction factor.

**Power** (exact, `power_sign_test`), at the unit counts:

| P(d > 0) | 0.58 | 0.60 | 0.62 | 0.65 | 0.70 |
|---|---|---|---|---|---|
| KL, 284 units | 0.652 | 0.859 | 0.962 | 0.998 | 1.000 |
| bpb, 171 units | 0.421 | 0.630 | 0.808 | 0.955 | 0.999 |

SMALLVD-001's descriptive positive fractions were 0.98 (KL) and 0.92 (bpb). A NOT_CONFIRMED verdict is therefore informative about an effect of that size, but not about an effect with P(d > 0) below about 0.6 on KL or 0.62 on bpb, where power falls below 0.8.

**SECONDARY set.** It gets the same test and is reported as a replication. It does not enter the verdict.

**Descriptive only** (never a verdict and never a selection):

- The category-mean KL objective and its ratio.
- Answer bpb under V0, V4 and V2, and the generation flip rate against the teacher.
- Per-category KL-better counts.
- The shape of log(LC/plain) with SMALLVD-001's A5.8 statistics, including corr(log KL_plain, log(LC/plain)).
- Per-row and per-unit d.

## 7. Registered forecasts (scored in the result; not gates)

- **F1.** PRIMARY positive fraction over KL units ≥ 0.90.
- **F2.** PRIMARY positive fraction over bpb units ≥ 0.80.
- **F3.** `size_large` is true.
- **F4.** corr(log KL_plain, log(LC/plain)) < 0: the rows the plain student gets most wrong gain the most.
- **F5.** SECONDARY is also CONFIRMED.
- **F6.** The LC flip rate is below the plain flip rate on PRIMARY.

## 8. What each verdict licenses

- **CONFIRMED.** The benefit of compensation is confirmed on fresh rows, as a median-unit effect, for the recommended student. The LC-folded V4/20 becomes the recommended **initialisation** for any future recovery or distillation training. That use still requires G5, a qualified training lane, the operator's teacher decision and a grant permitting architectural change. `size_large` qualifies the claim's strength.
- **PARTIAL.** Only the confirmed measure is licensed:
  - KL only: closer to the teacher's distribution, with no claim about answers.
  - bpb only: better reference likelihood, with no claim about distributional fidelity.
- **NOT_CONFIRMED.** SMALLVD-001's descriptive effect did not replicate at the power above. LC is not recommended.
- **HARMFUL.** LC is barred for this student.

**In every case:**

- GO/HOLD stays **HOLD**: none of its missing limbs is touched.
- There is no quality claim and no release.
- The result is not used to choose among sets, λ values or fold points: there is nothing to choose.

## 9. Read budget, execution, resources

- **Read budget:** the panel is read **once** by the real run. After an infrastructure failure (GPU reset, OOM, RAM stand-down) the same command may be rerun on the same code, and any partial console lines already printed are disclosed. No other rerun is permitted.
- **Registered command** (R = `smallvd001-20260924`, H = `hlcc001-20260925`, T = the Rineya-Supa-1.7.0-devfreeze teacher), in `.venv-rab001-cu126`, detached through the WMI runner:

```
python -B run_hlcc.py --teacher T --panels R/calibration/LAYER_PANELS.jsonl --study R/layer_study \
  --confirm H/panel/CONFIRM_PANEL.jsonl --vocab-primary R/vocab/V4_32768 --vocab-secondary R/vocab/V2_65536 \
  --primary-kept 0,1,2,3,4,7,9,10,11,15,16,17,18,19,20,21,22,25,26,27 \
  --secondary-kept 0,1,2,3,4,9,10,11,15,16,17,18,20,21,22,25,26,27 \
  --smallvd-lc R/lc_arms/LC_ARMS.json --out H/result --device cuda
```

- **Resources:**
  - GPU lease label `hlcc001`.
  - Host RAM: at least 3.6 GiB free to start (`hostbudget.require`), stand-down below 1.5 GiB, CUDA memory fraction 0.78 (no WDDM spill).
  - Launched only after the soup lane sends "ended".
  - Estimated 30–45 min: SMALLVD-001's LC arms took 643 s and 410 s per set on 257 HOLDOUT rows.
- **Output:** `H/result/HLCC_RESULT.json`, created only at the end (create-only). It records the panel and code sha256s.

## 10. Code

| file | sha256 |
|---|---|
| `hlcc_verdict.py` | `0179acad620667043bc246c7b95c4cc40dc6d32caddf73f2ab70b0d723c94254` |
| `run_hlcc.py` (R1) | `74b55d433b4cbbd62f73fe77f66b62b40f6035d4b461b2cf2d4c24718a897a99` (was `4d8702c6…` at 267cec9b) |
| `build_confirm_panel.py` | `c8d2c6f2ceac5a3bd523020bf0710dc00d12b779d4f7aa1730747f26675274ce` |
| `tests/test_hlcc_verdict.py` (9 pass) | `1bd0029f0de99221a5f512f1a540036a9d916aa4fbc75395b9868130754e12ae` |

Dry run: tiny random 6-layer teacher, CPU, 3 rows per category. It ran end to end, the canary reproduced exactly, and both negative controls refused and wrote nothing.

## 11. Disclosures

- **D1 — panel built before registration.** The panel was built before this document, on the CPU, with no forward pass of the real teacher. Its hash is pinned above.
- **D2 — dry run touched panel rows.** The dry run ran a **random** tiny model on 30 panel rows (3 per category). Its outputs carry no information about the real teacher.
- **D3 — empty probe directory.** A mistyped probe of the builder created an empty directory `H/panel_probe_do_not_use/`: the builder creates its output directory before loading the tokenizer. Git does not track empty directories. It is left in place, because nothing may be removed from the records tree without operator approval.
- **D4 — builder flaw kept.** The builder still has that early-mkdir flaw. It was not fixed, so that the manifest's `builder_sha256` matches the committed file.
- **D5 — unit rule added after seeing ids, before any model output.** The unit rule (§6) was added after the panel was built. At that point only row ids and metadata had been inspected: 19 books behind the 32 prose rows, 11 tasks behind the 32 long-context rows. No model output of any kind had been seen. It replaced a row-level test that the measurements above show is anticonservative on this panel.

## 12. Revisions before the run

- **R1 (2026-09-25, from Rineya Delta's cold review, before any real run).** In the 267cec9b code, the fit canary and the identity control ran only *after* the real teacher had generated on the confirmation panel and collected its hidden states. A canary refusal would therefore have spent the panel's single read and recorded nothing.
  - Both controls, and both sets' fits, now run on SELECT rows before the panel is touched.
  - The identity control checks the fold mechanism on SELECT rows instead of panel rows.
  - Nothing measured changes. Rerun on the dry model, every per-row d, per-unit d and fit was bitwise identical to the 267cec9b code's dry run.
  - The tampered canary now refuses before the "panel is read" line and writes nothing.
  - The tests, the measures and the verdict are unchanged.
