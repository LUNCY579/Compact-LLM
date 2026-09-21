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
