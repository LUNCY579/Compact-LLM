# OSIDE-001: does the module route (H1) keep more of Opallo's held-out reply shift than the magnitude route (ADD-O50)?

Registered 2026-09-25, before any arm below is scored on this instrument. This implements two registered but
unbuilt pieces of the Titan–Opallo audit lane (`titan_opallo_exact_endpoint_v3/proposal_offline/`):
- **ADDENDUM_1 §3:** the comparison H1 vs ADD-O50;
- **FOLLOWUP_PROPOSAL §2 P4:** the per-token O-side re-run, with the denominator gate passed first.

The S side is already measured (CONVENTION-LOC-001 step 1, T_BANK_V2, exposed bank). Both arms remove Opallo's
imported trailing period (0 of 192 rows each), and both keep Titan's skill:

| arm | trailing periods | tolerant | strict |
|---|---|---|---|
| ADD-O50 | 0 | 100 | 100 |
| H1 | 0 | 102 | 102 |
| SPARSE (Titan alone) | 0 | 102 | 102 |

The two arms are within 2 rows of each other on Titan's bank, so the S side cannot separate them. This test asks which route keeps more of what Opallo itself adds:
- **H1** keeps dOpallo's MLP half at full strength: 60.8% of dO's energy.
- **ADD-O50** keeps 0.5·dO everywhere: 25% of the energy.

## 0. What kind of evidence this is

- **The instrument is exposed.** It is the 6,741 held-out rows of `o_reply_nll.make_rows()`. Nine arms were
  scored on it on 2026-09-23 (`O_REPLY_NLL.json`), at row level only: per-row sums and means.
- **What has never been measured:**
  - per-token values, for any arm;
  - the content/terminal split;
  - H1, ADD-O50, H1c, OPALLO-BF16 and the three seed replicates, on this instrument.
- **So this is a registered development comparison, not a confirmation.**
- **Read budget: one run.** The only allowed rerun is after a harness failure (canary FAIL, crash). It resumes
  after the last complete score file, and it is logged in `ATTEMPT_LOG.md`.
- **P_O^content measures retention of a teacher-forced reply shift. It is not an identified skill.**
  O-SKILL-DEFINABILITY's "content, not prior" control is still unresolved (its Erratum 1). Nothing here
  licenses "Opallo's skill is preserved", "lossless" or any release claim.

## 1. Arms

| arm | construction | weight sha256 | role |
|---|---|---|---|
| BASE | FP32 view of Supa's stored BF16 | `75b0a92c…` | reference |
| OPALLO | Supa + dO (compose.py endpoint) | `5ff1c9a5…` | denominator |
| OPALLO-BF16 | OPALLO's parameters rounded to BF16 in memory, FP32 compute | (from OPALLO) | cast noise, DG2 |
| SEED23, SEED29, SEED41 | Supa + the trained tensors of Opallo's seed replicates | new | seed noise, DG3 |
| SPARSE | Titan-Sparse | `6ae85f5b…` | specificity, canary |
| ADD-100 | Supa + dS + dO | `95a75767…` | plain-addition context, canary |
| **ADD-O50** | SPARSE + 0.5·(ADD-100 − SPARSE) | must equal `bc69a4ec…` | **contrast** |
| **H1** | SPARSE + dO on MLP tensors only | must equal `3ff2aa59…` | **contrast** |
| H1c | SPARSE + dO on attention + layer-norm tensors only | must equal `ea46496c…` | descriptive (P4 lists it) |

**Details and pre-registration checks for each arm**

- **OPALLO-BF16.** This equals the published Opallo checkpoint bit for bit (`GEOMETRY.json`: `opallo_roundtrip`
  596,049,920 of 596,049,920 elements). The scorer spot-checks three tensors against the published file:
  0 mismatches in the smoke test (§11).
- **SEED23, SEED29, SEED41.** These use the same recipe as Opallo (B28last20, lr 1e-5, b5200, wsd 0.2/0.2,
  β2 0.95) with seeds 23, 29 and 41. Each seed also draws its own 39–40 training rows.
  - They are built with `compose.py`'s OPALLO-endpoint recipe (`build_o_arms.py seed N`).
  - **Positive control, run before registration:** the same code applied to seed 17 reproduces OPALLO's
    file exactly (sha `5ff1c9a5…`, 2,384,235,032 bytes).
  - The trained files' sha256 must match their ARM_DOSE records, which are listed in `build_o_arms.py`.
- **ADD-O50, H1, H1c.** These are rebuilt with CONVENTION-LOC-001's own `chunks()` and `coef_fn()`
  (`build_o_arms.py rebuild`).
  - **Checked before registration:** all three rebuild streams hash to the recorded, committed values.
  - The driver checks again at build time. A mismatch deletes the weights and stops the run: an arm is
    defined by its hash.

**Resources:** one rebuilt arm is on disk at a time (2.4 GB). Its weights are deleted after it is scored, and
`BUILD.json` plus `WEIGHTS_DELETED.json` are kept.

## 2. Readout

**Scoring.** Each reply token's teacher-forced NLL is taken on the same rows, prompt, answer and tokenisation
as `o_reply_nll.py`, which is imported rather than copied:
- FP32, TF32 off, `use_cache` False;
- `ROWS_MANIFEST.json` (sha `19834d34…`, tokenizer only, committed with this file) fixes the rows and token
  counts.

**Terminal-token rule.** A row's final reply token is **terminal** iff its decoded text, stripped, is
non-empty and consists only of `.`, `?` and `!`.
- This holds for **6,724 of 6,741 rows**: `.` 6,299, `?` 329, `!` 90, `!!` 6.
- The other 17 end in an emoji byte fragment, and all their tokens count as content.
- Every row has at least one content token. The shortest reply has 2 tokens. There are 41,419 content tokens
  and 48,143 tokens in all.

**Quantities.**
- c_r(X) = the mean NLL of row r's content tokens under arm X.
- f_r(X) = the NLL of row r's terminal token.
- M^c(X) = the mean of c_r(X) over all 6,741 rows. This is row-weighted, like the existing P_O.
- **P_O^content(X) = (M^c(BASE) − M^c(X)) / (M^c(BASE) − M^c(OPALLO)).**
- P_O^final is the same ratio, using f over the 6,724 terminal rows.
- The secondary token-weighted version pools content NLL over content tokens.

**Class-cluster bootstrap.** This covers the 24 reply classes. Each draw resamples classes with replacement
and keeps every row of each resampled class. All arms are resampled jointly, so ratios and differences
carry their dependence.
- 10,000 draws, seed 20260925;
- percentile intervals.

## 3. Canary (harness check; must pass)

**Arms:** BASE, OPALLO, SPARSE and ADD-100.

**Conditions:**
- The per-row mean over all tokens, computed exactly as `o_reply_nll` computed it, reproduces
  `O_REPLY_NLL.json` (sha `f5cbaea0…`) with max |diff| ≤ 1e-5.
- Every token count matches.
- The mean of the stored per-token list agrees with that row mean to 1e-5.

**On failure:** BASE and OPALLO are scored first. If either fails, the run stops before any new arm is built
or scored (`HARNESS_FAILURE`).

## 4. Denominator gate (every limb must pass before any P_O^content verdict)

- **DG1, sampling.** The class-cluster 95% interval of the content gain Δc = M^c(BASE) − M^c(OPALLO) lies above 0.
- **DG2, cast.** |M^c(OPALLO-BF16) − M^c(OPALLO)| ≤ 0.05·Δc.
- **DG3a, the gain reproduces across seeds.** Each replicate's content gain has a class-cluster 95% interval
  above 0.
  - This is computed on R' = the 6,663 rows that none of seeds 23, 29 or 41 consumed.
  - Seed 17 consumed none of the pool, by construction.
- **DG3b, the gain is large against seed noise.** On R', Δc(seed 17) ≥ 3 × the sample SD of the four seeds'
  content gains (17, 23, 29, 41).

**If a limb fails:**
- No P_O^content verdict is issued (`NOT_ISSUED_DENOMINATOR_GATE_FAILED`).
- The raw content contrast M^c(ADD-O50) − M^c(H1), in nats per token with its class-cluster interval, is
  still reported. It needs no denominator.

**Why seed noise does not enter the contrast's validity.** H1 and ADD-O50 are two transforms of the same
dO(seed 17), so seed noise does not enter the contrast. DG3 checks that the denominator is a property of
the recipe rather than of one draw. The seed-mean denominator is reported as a sensitivity (§7).

## 5. The registered comparison (ADDENDUM_1 §3)

**D = P_O^content(H1) − P_O^content(ADD-O50)**, with a class-cluster interval. The decision interval is
chosen as in §6.

| decision interval | verdict |
|---|---|
| lies entirely above 0 | **MODULE_ROUTE_BETTER** |
| contains 0 | **NULL_MODULE_ROUTE_BUYS_NOTHING**: the magnitude/plateau rule again |
| lies entirely below 0 | **MODULE_ROUTE_WORSE** |

**How a null is reported.** It is stated together with its detectable effect:
- MDE = (z(1 − α/2) + z(0.80)) × SE_boot(D);
- α is the level of the decision interval;
- the nominal-5% MDE is reported too.

## 6. Calibration of the interval (null cell; measured, not assumed)

**The problem, found before registration.** Percentile intervals over few clusters can undercover. This was
measured before registration (`CAL_DRYRUN_EXPOSED_ALLTOKEN.json`), on exposed all-token row means and proxy
arm pairs. It is not the registered quantity, which has never been computed. The 95% class-cluster interval
excluded a true zero **8.2%, 9.5% and 11.7%** of the time:

| proxy pair | calibrated α |
|---|---|
| ADD-100 − ADD-O75 | 0.02 |
| ADD-O75 − AVG | 0.01 |
| SPARSE − AVG | 0.005 |

**The null-cell simulation, run at analysis time on the observed data:**
1. Shift every H1 row by one constant so that the pseudo-population of the 24 observed classes has D = 0
   exactly, while keeping the observed class heterogeneity of the contrast.
2. Draw 2,000 samples of 24 classes from it.
3. Give each sample its own percentile interval (1,000 inner draws).
4. Count how often 0 is excluded. The seed is 20260926.

**Decision rule.**
- If the 95% interval's null-cell rate is ≤ 0.075, the registered 95% percentile interval decides.
- Otherwise the decision interval is the percentile interval at level 1 − α*. Here α* is the largest α in
  {0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.002} whose null-cell rate is ≤ 0.05.
- If no level qualifies, the verdict is `INDETERMINATE_UNCALIBRATABLE_INTERVAL`.
- The 95% interval is always reported beside the decision interval.

Given the dry run, **a widened decision interval is the expected case.**

## 7. Secondary and descriptive (no verdicts)

- **FOLLOWUP_PROPOSAL P4 readings, on the point estimate of P_O^content(H1):**
  - Reading A (the content shift is carried by MLP): ≥ 0.85.
  - Reading B (mostly attention-carried style): ≤ 0.60.
  - Otherwise inconclusive.
- **Per arm, each with a class-cluster interval:** P_O^content and P_O^final for every arm, and
  P_O^content by token.
- **D by token.**
- **Additivity:** P_O^content(H1) + P_O^content(H1c), compared with 1.
- **Specificity:** P_O^content(SPARSE). O-SKILL-DEFINABILITY's control 2 used "about 0.25".
- **Seed sensitivity:** D with the four-seed mean denominator, on R'.
- **Raw content means** M^c(X), and the full-reply means.

## 8. My predictions (registered; not verdict rules)

- **Canary:** exact (0 difference), as the smoke test was on 30 rows.
- **DG2:** passes with a large margin, since only the trained tensors' rounding differs.
- **DG1 and DG3a:** pass. Any SFT delta on these classes lowers held-out NLL (SPARSE's all-token P_O is 0.16).
  DG3b is the limb that can fail.
- **The comparison: not MODULE_ROUTE_BETTER.** Point estimate of D in [−0.15, +0.05].
  - **Reasoning:** on all tokens the β ladder saturates early (ADD-O75 keeps 0.957 against ADD-100's
    0.972), so ADD-O50 should sit near the plateau.
  - Separately, the imported period needs **both** of Opallo's modules. That warns that the MLP half alone
    may not carry the content shift either.
  - The plateau rule predicts a null; the both-halves finding predicts WORSE. **I expect NULL or WORSE, with
    low confidence in which.**

## 9. What each outcome would license

- **MODULE_ROUTE_BETTER.** At a matched convention rate (0 periods for both), keeping dO's MLP half at full
  strength retains more of Opallo's held-out content shift than halving all of dO. This is one donor pair on
  an exposed pool, at development grade. Titan's side is already clean for H1 (102/102).
- **NULL.** At the stated MDE, the module restriction buys nothing over a scalar. This is the
  magnitude/plateau rule on a fourth construction.
- **MODULE_ROUTE_WORSE.** The scalar route keeps more, which means the attention and layer-norm half carries
  content too.
- **None of the three licenses** "Opallo's skill is preserved", "lossless" or "functionally independent".

## 10. Execution

`run_o_side.py` runs in the hermes venv (python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1), launched
through WMI. It runs after CLOCKFIX, under the shared GPU lease.

**Preflight:**
- hashes of every source, trained file, manifest and the recorded run;
- the seed-builder positive control;
- host RAM ≥ 3.5 GiB and disk ≥ 6 GiB.

**Order:**
1. BASE and OPALLO, then the canary on both.
2. OPALLO-BF16, SPARSE and ADD-100, then the canary on all four.
3. For each of ADD-O50, H1 and H1c: rebuild, check the hash, score, delete.
4. For each of SEED23, SEED29 and SEED41: build, score, delete.
5. `analyse_o_side.py` → `O_SIDE_RESULT.json`.

**Cost:**
- 11 scorings at about 5.6 GPU-min each (the recorded run took 307.5–339.6 s per arm): about 65 GPU-min.
- 6 CPU builds: seconds each on a warm file cache, about 2 min cold.
- Per-process VRAM fraction 0.78, and a RAM wait before each model load.

## 11. Done before registration (no registered quantity read)

- `ROWS_MANIFEST.json`, from the tokenizer only.
- The seed-builder positive control (PASS).
- Hash checks of the three rebuild streams (all match).
- The calibration dry run on exposed all-token data (§6).
- `test_o_side_offline.py`: 12/12 pass. Every outcome and every gate limb is shown to be able to fail and
  to pass, and the canary catches a 1e-4 drift.
- **A GPU smoke test of the scorer, under the lease:**
  - BASE, first 30 rows: max |diff| against `O_REPLY_NLL.json` is 0.0, and the token counts are equal.
  - OPALLO-BF16, first 3 rows: 0 mismatches on all three spot-checked tensors.
  - Both outputs went to the session scratch directory, not to `scores/`. The only values seen were BASE's
    (already recorded) and 3 rows of the cast arm.

## 12. Files

| file | purpose |
|---|---|
| `PREREG.md` | this registration |
| `ROWS_MANIFEST.json`, `make_manifest.py` | the row and token manifest |
| `seekst.py` | seek/read safetensors reader (no memory map) |
| `build_o_arms.py` | rebuild / seed / seed-control / delete |
| `score_o_tokens.py` | per-token scorer |
| `analyse_o_side.py` | canary, gate, comparison, calibration, descriptive table |
| `run_o_side.py` | the driver |
| `test_o_side_offline.py` | offline tests |
| `CAL_DRYRUN_EXPOSED_ALLTOKEN.json` | the §6 dry run |
| `ATTEMPT_LOG.md` | run attempts |
| (outputs, after the run) | `PREFLIGHT.json`, `scores/*.json`, `arms/*/BUILD.json`, `arms/*/WEIGHTS_DELETED.json`, `O_SIDE_RESULT.json` |
