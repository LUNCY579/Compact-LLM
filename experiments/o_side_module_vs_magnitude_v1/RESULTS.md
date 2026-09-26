# OSIDE-001: results

Run 2026-09-26, attempt 3, 05:07–05:57Z, rc 0, via WMI under the GPU lease `oside-001`. It was registered in
`PREREG.md` (962abee4) and pre-run `AMENDMENT_1.md` (da566f03) before any arm was scored.

**Outputs:**
- registered result: `O_SIDE_RESULT.json`
- per-token scores: `scores/*.json` (11 arms × 6,741 rows)
- independent audit: `audit_o_side.py` → `AUDIT_O_SIDE.json` (0 problems)
- attempt history: `ATTEMPT_LOG.md`

**Evidence grade.** This is a registered development comparison on an exposed pool (PREREG §0). It is not a
confirmation. P_O^content measures retention of a teacher-forced reply shift; it is not an identified skill.

## Headline

**The registered verdict is NULL_MODULE_ROUTE_BUYS_NOTHING.**

**The setup.** Both routes drop Opallo's imported trailing period on Titan's bank (0 of 192 rows each), and both
keep Titan's skill (102 and 100 tolerant). The routes are:
- **H1**, the module route: Titan-Sparse plus Opallo's MLP half at full strength;
- **ADD-O50**, the magnitude route: Titan-Sparse plus half of all of Opallo.

**What each keeps of Opallo's held-out content shift:**

| arm | P_O^content | class-cluster 95% interval |
|---|---|---|
| H1 | **0.774** | [0.633, 0.886] |
| ADD-O50 | **0.756** | [0.603, 0.870] |

**The contrast:**
- **D = P_O^content(H1) − P_O^content(ADD-O50) = +0.018.**
- Decision interval **[−0.004, +0.049]**. This is the calibrated 98% interval: the registered 95% interval's
  null-cell rate was 8.45%, above the 7.5% limit, so α* = 0.02.
- The effect detectable at 80% power at that level is **0.036**; nominally it is 0.032.
- The uncalibrated 95% interval, [−0.0009, +0.043], also contains 0, so the verdict does not depend on the
  calibration.

**What this means.** At a matched convention rate, restricting Opallo to its MLP half buys nothing measurable over
simply halving it. That is the magnitude/plateau rule again, on a fourth construction. The simpler, surgery-free
scalar is enough.

## Registered checks, in order

| check | result |
|---|---|
| Canary (PREREG §3; AMENDMENT_1) | **PASS**, exact. BASE, OPALLO, SPARSE and ADD-100 reproduce `O_REPLY_NLL.json` row by row with max abs diff **0.0**. Token counts equal; 0 non-finite values. It passed at stage 1 in attempts 2 and 3, and at stage 2. |
| Provenance (AMENDMENT_1) | **PASS** on all 11 files, on one device. It checks the weight hash (registered, or BUILD.json for the seeds), the cast flag, no row limit, 6,741 rows, runtime 3.11.15 / 2.6.0+cu124 / 5.14.1, and the manifest's input hashes. |
| DG1: content gain Opallo − Supa | **0.555** nats/token, class-cluster CI [0.363, 0.759]: **PASS** |
| DG2: BF16 cast noise | 0.0016 nats/token, **0.28%** of the gain against a 5% limit: **PASS** |
| DG3a: gain in every seed replicate (R′ = 6,663 rows) | seed 23: 0.420 [0.274, 0.585]; seed 29: 0.481 [0.302, 0.680]; seed 41: 0.461 [0.257, 0.669]: **PASS** |
| DG3b: gain vs seed SD | gains on R′ 0.555 / 0.420 / 0.481 / 0.461, SD 0.056, **ratio 9.8** against ≥ 3: **PASS** |
| Null-cell calibration (§6) | rates by α: 0.05 → 0.0845, 0.04 → 0.0715, **0.03 → 0.0545**, **0.02 → 0.0425**, 0.01 → 0.0295. So α* = 0.02. |
| Registered comparison | D = +0.018; decision interval [−0.004, +0.049]: **NULL** |

## Descriptive (registered; no verdicts)

| arm | P_O^content (row) | P_O^content (token) | P_O^final (terminal token) |
|---|---|---|---|
| OPALLO-BF16 (cast) | 0.997 [0.987, 1.012] | 0.997 | 0.999 |
| SPARSE (Titan alone) | 0.076 [−0.087, 0.195] | 0.032 | 0.311 |
| ADD-100 (plain addition) | 0.910 [0.764, 1.004] | 0.912 | 1.058 [1.030, 1.118] |
| ADD-O50 | 0.756 [0.603, 0.870] | 0.737 | 1.011 |
| H1 (MLP half) | 0.774 [0.633, 0.886] | 0.738 | 1.022 |
| H1c (attention + LN half) | 0.713 [0.546, 0.833] | 0.709 | 0.997 |
| SEED23 / 29 / 41 | 0.763 / 0.870 / 0.834 | 0.750 / 0.866 / 0.998 | 0.996 / 0.943 / 0.938 |

- **P4 reading** (FOLLOWUP_PROPOSAL): P_O^content(H1) = 0.774, between 0.60 and 0.85, so **inconclusive**. Its
  class-cluster interval spans both thresholds.
- **Opallo's content shift is carried redundantly, not partitioned.**
  - Each half alone keeps about three quarters of it: H1 0.774, H1c 0.713.
  - Additivity P(H1) + P(H1c) = **1.49**, CI [1.18, 1.71], far above 1.
  - This is the content-side counterpart of CONVENTION-LOC-001's finding that the imported period needs both halves
    together (AND-like). Here each half is nearly sufficient on its own (OR-like).
- **Plain addition keeps 91% of the content shift, not 97%.** The earlier all-token P_O of ADD-100 (0.972) was
  inflated by the terminal token, where ADD-100 over-shoots (P_final 1.058). With the terminal token excluded,
  full-strength addition with Titan loses about 9% of Opallo's content shift. The interval reaches 1.004, so this is
  not significant.
- **Terminal-token NLL is not the generation convention.** H1, ADD-O50 and H1c all keep Opallo's full teacher-forced
  gain on the terminal `.`/`?`/`!` of Opallo-class replies (P_final ≈ 1.0), yet H1 and ADD-O50 emit **no** trailing
  period when generating on Titan's bank. The two measure different things, so P_final must not be used as a
  convention readout.
- **The shift is mostly recipe-level, not seed-specific.** Opallo's other seeds reproduce 76–87% of seed 17's
  content shift. Titan-Sparse alone reproduces about 8% (row-weighted) or 3% (token-weighted): the specificity
  control holds.
- **Token-weighted D is null too:** +0.001, CI [−0.018, +0.022]. Seed-mean denominator D: +0.021.

### A post-hoc reading (audit; not registered; label only)

**The small row-weighted H1 advantage lives in very short replies.** H1 fits better on 3,679 of 6,741 rows (54.6%)
and in 14 of 24 classes, but it depends on reply length:

| content tokens | rows | H1 better | mean nats (ADD-O50 − H1) |
|---|---|---|---|
| 1 | 746 | 73.6% | +0.059 |
| 2–3 | 1,436 | 59.5% | +0.017 |
| 4–7 | 1,909 | 55.0% | +0.005 |
| ≥ 8 | 2,650 | 46.3% | −0.003 |

**Reading:** the MLP-only route fits Opallo's terse one-word answers slightly better, and on longer content the two
routes are indistinguishable. This is why the token-weighted D is about 0. It is a hypothesis for a future
registration, not a result.

**Per-class P_O ratios are ill-conditioned** wherever Opallo's own class-level gain is near zero or negative.
Examples: causal_reasoning has D_class −3.1 on a nats difference of +0.003; boundary_setting has D_class +0.49 on
−0.019. Per-class effects should be read in nats (`AUDIT_O_SIDE.json` → `per_class`). The pooled ratio is well
conditioned: the DG1 interval is far from 0.

## Predictions (PREREG §8), scored

| prediction | outcome |
|---|---|
| Canary exact | **correct** (0.0) |
| DG2 passes by a large margin | **correct** (0.28%) |
| DG1 and DG3a pass | **correct** |
| DG3b is the limb that can fail | it passed at ratio 9.8 |
| Not MODULE_ROUTE_BETTER; D in [−0.15, +0.05]; NULL or WORSE | **correct**: NULL, D = +0.018 |

The plateau reading (NULL) beat the both-halves reading (WORSE). The content shift turned out redundant across
modules, unlike the period.

## Independent audit (`audit_o_side.py`, CPU only, own code paths)

- **A. Provenance and file identity.** All 11 score files carry the expected weight hash, cast flag, runtime,
  device and input hashes. They match the file hashes recorded in the result, and the three rebuilt arms certify
  their registered hash: 0 problems.
- **B. Point estimates.** Recomputed from the raw per-token lists with an independent content split, they reproduce
  D and all 11 content means exactly (difference < 1e-12).
- **C. An independent class bootstrap** (explicit class resampling, PCG64 seed 7, 4,000 draws):

  | quantity | audit | registered |
  |---|---|---|
  | SE | 0.0112 | 0.0113 |
  | 95% interval | [−0.0007, +0.0431] | [−0.0009, +0.0430] |
  | 98% interval | [−0.0043, +0.0482] | [−0.0043, +0.0493] |

- **D. An independent null cell** (seed 11, 600 × 600): 10.2% at α = 0.05 and 5.7% at α = 0.02, against the
  registered 8.45% and 4.25%. The difference is within about 1.5 binomial SE at 600 outer draws. Both confirm that
  the uncalibrated interval is anticonservative. Either way, NULL holds at every level, because the uncalibrated
  interval already contains 0.

## Run history (harness only; nothing was rescored)

| attempt | what happened |
|---|---|
| 1 | Preflight refused on disk (3.0 GiB): the pagefile had grown to 19.25 GiB after a host reboot. |
| 2 | Killed inside ADD-100 by an operator-initiated restart at 00:53:53Z. It had passed the stage-1 canary exactly, and 4 complete files were kept. |
| 3 | Resumed and completed. |

Before the run, two adversarial reviews found 16 minor defects, all fixed in AMENDMENT_1. The run exercised several
of those fixes:
- resume by completeness, not by file existence;
- provenance pins;
- a post-score check before each weight deletion.

## What this changes

- **For composing Titan with Opallo without the imported period, use the scalar.**
  - ADD-O50 (β = 0.5) and the MLP-only route keep the same share of Opallo's held-out content shift, about 3/4.
  - They keep the same Titan skill on the development bank.
  - Module surgery adds nothing detectable (80% power at 0.036). An advantage up to +0.049 is not excluded.
- **"Where does Opallo's shift live?"** Not in one module: each half carries most of it. Its trailing-period
  convention needs both halves at once.
- **The ADDENDUM_1 question is answered** at development grade: *module route buys nothing over a scalar*.

## Limits

- The pool is exposed (development grade): one donor pair, one base.
- The readout is teacher-forced NLL on Opallo's 24 reply classes, not generation.
- Opallo's content is unidentified: the content-vs-prior control is unresolved (O-SKILL-DEFINABILITY Erratum 1).
- The estimand is row-weighted, and short canned replies weigh as much as long ones (see the post-hoc table).
- Class-cluster generalisation rests on 24 classes. The interval is calibrated by a null cell, which is itself a
  Monte Carlo estimate.
- **Rebuild note.** The source arms in `titan_opallo_skill_vector_v1/arms` are on disk. Rebuilding them with
  `compose.py` would first need a vault restore of `int001-20260917/trained_fp32.safetensors` (sha 07156fb5…), which
  was deleted on 2026-09-26 by the operator-approved retention audit. The restore command is in that directory's
  `WEIGHTS_DELETED.json`.
