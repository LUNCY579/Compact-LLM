# HDOSE-001 — where is the gate skill's own magnitude optimum?

> **STATUS: REGISTERED, NOT BUILT.** No arm exists and nothing is scored as of this line.

**Class: MECHANISM SCOUT.** Zero training, zero optimizer steps. Pure weight arithmetic over
deltas already on disk. Promotes nothing.

## Why this, and why it is now a one-parameter search

SCALED-T-001 and SCALED-H-001 between them showed that **composition's entire effect on this
pair is attenuation**:

| arm | construction | J_decision | sens /112 | spec /112 |
|---|---|---:|---:|---:|
| `H16` | `P₁₆(δ_H)` — full truncated strength | 0.6161 | 88 | **93** |
| `SCALED_H` | `s_t · P₁₆(δ_H)`, `s_t` median **0.675** | **0.7589** | 86 | **111** |
| `TH16s` | `s_t · (P₁₆(δ_T) + P₁₆(δ_H))` | 0.7500 | 84 | 112 |

**`SCALED_H` is the highest J on record and contains no δ_T at all.** So "δ_T supplies the
constraint that makes δ_H usable" is withdrawn — δ_H was simply **over-driven**, and reducing its
magnitude is all composition did.

> **`0.675` was never chosen for H. It is whatever `s_t` happened to be** — a ratio computed from
> δ_T's norms for an entirely different purpose. It is already the best arm on this instrument
> and **the whole curve between 0 and 1 is unmeasured.** There is no reason to believe it is the
> optimum.

**This is a one-parameter search, and it needs no second delta and no composition.**

## Construction

    HDOSE(alpha) = theta_supa + alpha * P16(delta_H,t)        uniform scalar, FP32

**Uniform α, deliberately, not per-tensor.** `SCALED_H` uses a *per-tensor* vector whose median
is 0.675. A uniform sweep is the simpler object and it doubles as a control: if uniform α = 0.675
reproduces `SCALED_H`'s J, the per-tensor structure carries nothing here.

**Grid, built and scored one at a time (see the disk constraint below): α ∈ {0.50, 0.25}**,
read against the existing α = 1.0 (`H16`) and the per-tensor 0.675 (`SCALED_H`).

## BUILD GATE — the joint canary, adopted from SCALED-H-001

A δ_H-only build has nothing of its own to check against. So recompute **both** of MAG-001's
recorded per-tensor norms and require both to match:

    ||P16(delta_T,t)||                      == recorded norm_T
    ||P16(delta_T,t) + P16(delta_H,t)||     == recorded norm_combined

at **relative 2e-6** (the tolerance calibrated in SCALED-T-001 Amendment 1, against which the
paired session measured a 25× margin). That is a joint check on the source, **both** trained
checkpoints, the tensor mapping and the SVD path in one pass. **Refuse to write on any failure.**

The statistic is `sqrt(sum of top-16 squared singular values)`, **not** the norm of the
reconstructed `U·S·Vʰ` — the latter is lossy by ~6e-5 and failed 140/140 when SCALED-T-001 first
used it.

## Primary endpoint

`J_decision` on **HSKILL-VIEWS-002** (`instrument_sha256 ba1f2982…`, 224 rows, 112 boundary +
112 answerable), with **sensitivity and specificity reported separately and never only as the
index** — `T16` and `H16` both score J = 0.6161 to sixteen digits while differing at p = 0.0043
on the boundary half, so the index hides its own decomposition.

Paired exact McNemar per half against `H16` and against `SCALED_H`.

## Registered predictions

| # | prediction | reading |
|---|---|---|
| **P1** | some α < 0.675 gives **J > 0.7589** | the optimum is **below** `SCALED_H`; 0.675 was luck, not a maximum |
| **P2** | both α = 0.50 and α = 0.25 give **J < 0.7589** | 0.675 is at or near the optimum, and the curve falls away on the low side |
| **P3** *(control)* | uniform α = 0.675 would reproduce `SCALED_H`'s J within the resolvable band | the per-tensor structure carries nothing; **not built in this grid**, registered for the next one |

**Declared before the numbers:** `J_decision = sensitivity + specificity − 1` on two halves of
112. **A difference in J that rests on fewer than 6 discordant pairs on its moving half cannot
reach p ≤ 0.05 and must be reported as UNCERTIFIABLE, not as a null.** SCALED-H-001's
`SCALED_H → TH16s` comparison already landed there at nd = 4 and 1, and that lesson is imported
rather than rediscovered.

## What would make this uninterpretable

- **Build-gate failure** → void, report the divergence, do not score.
- **Specificity saturating at 112.** `SCALED_H` is already at 111/112 and `TH16s` at 112/112.
  **If a lower α pins specificity at 112, J can only move through sensitivity**, and the
  specificity limb becomes a ceiling rather than a measurement. Registered now because it is the
  likely shape of the result.
- Scoring under any instrument other than `ba1f2982…`.

## The disk constraint, and a decision with a cost

Free disk is **13.56 GiB, already below the programme's 14 GiB floor.** Four arms at 2.27 GiB
each is not available.

**So arms are built, scored, hashed and then RECLAIMED, one at a time** — peak footprint one arm.
The construction and `out_weight_sha256` are recorded so each is rebuildable in ~80 s.

**This is the same reclaim policy that killed Route A on BRIDGE-001** and forced a retrain to
re-score the dose ladder. It is being adopted deliberately here, with its cost stated: **anything
wanting to re-score these arms later pays a rebuild.** Given the deltas survive and the build is
80 seconds, that is the right trade at 13.56 GiB free — but it is a trade, not a free action.

---

# WITHDRAWN — duplicate of H-ALPHA-001, which is finer and already finished

**Not run.** The paired session registered and completed the same experiment as
`halpha001-20260921/H-ALPHA-001` on a finer grid (α = 0.05 / 0.10 / 0.15 / 0.25 / 0.50 / 0.675 /
0.85) while this was being written. **No arm of HDOSE-001 was ever written to disk.**

Their result, which supersedes this design entirely:

| α | sens /112 | spec /112 | `J_decision` |
|---:|---:|---:|---:|
| 0.00 *(source)* | 71 | 111 | 0.6250 |
| **0.25** | **86** | **112** | **0.7679** |
| 0.50 | 85 | 112 | 0.7589 |
| 0.675 *(= `SCALED_H`)* | 86 | 111 | 0.7589 |
| 0.85 | 85 | 104 | 0.6875 |
| 1.00 *(`H16`, as trained)* | 88 | **93** | 0.6161 |

> **Quarter magnitude is the optimum. The trained delta's full strength buys 2 boundary rows and
> costs 19 answerable ones. The gate skill is roughly 4× over-driven as trained.**

**My registered P2 would have been wrong** — I gave even odds to 0.675 being near-optimal, and
0.25 beats it. My grid `{0.50, 0.25}` would have found the direction but not the shape: their
0.05/0.10/0.15 rungs are what establish there is no further gain below 0.25.

**Also, both predictions registered here were about J, and the real finding is not in J.**
Sensitivity is nearly flat across the whole range (86 / 85 / 86 / 85 / 88) — **the entire shape
of the J curve is specificity** (111 → 112 → 112 → 111 → 104 → 93). A sweep registered on the
composite index would have measured the right curve for the wrong reason, which is the same
defect as `T16` and `H16` sharing `J = 0.6161` while differing at p = 0.0043.

## The canary earned its keep, and this is the part worth keeping

The build **refused**, correctly, and for a reason that is not a tolerance artifact:

    CANARY: worst norm_T 1.294e-06   worst norm_combined 1.497e-01   (tol 2e-06)
    REFUSED: 140 tensors exceed tolerance. Nothing written.

`norm_T` passed at 1.29e-06 while `norm_combined` failed at **15% relative** on every 2D tensor.
The cause is mine and is a genuine construction error:

> **`P₁₆(δ_T) + P₁₆(δ_H)` is generally rank-32, not rank-16.** I computed its norm as
> `sqrt(sum of its top-16 squared singular values)` — copying the statistic that is correct for
> `P₁₆(δ_T)`, where it equals the Frobenius norm *because that matrix is rank-16 by
> construction*. Applied to the sum it silently discards the other 16 directions, ~15% of the
> norm. `norm_combined` is a plain Frobenius norm.

**This is the sixth extractor-class bug this week and the FIRST caught by a gate rather than by
collision with a number someone already held.** The other five were caught because an output
happened to contradict something known. This one would have produced a plausible arm and a
plausible J, and the only thing standing between that and the record was a check that had no
reason to exist except that the paired session's build had used it.

**A statistic that is correct for one object is not correct for a sum of two such objects**, and
the mistake is invisible because both expressions are "the norm of a rank-16-ish thing".
