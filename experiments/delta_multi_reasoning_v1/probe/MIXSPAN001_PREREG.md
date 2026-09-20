# MIXSPAN-001 — is joint training weight-space addition?

> **STATUS: REGISTERED, NOT RUN.** Nothing has been computed as of this line.

**Class: MECHANISM SCOUT. Zero model calls, zero training, zero GPU.** Pure geometry over three
deltas already on disk. Promotes nothing.

## The question, and why it is the sharpest one left

Three ways of putting two skills in one 596M model have been measured on the same bank:

| route | budget | acquisition |
|---|---|---:|
| **MIX** — joint training on the union corpus | ONE budget, 1,032 rows | **225** |
| composition, raw sum | TWO budgets | 175 |
| composition, norm-matched (`TH16s`) | TWO budgets | 219 |

**Joint training on one budget beats composition on two.** SCALED-T-001 has since shown
composition's cost is attenuation rather than interference — so the remaining question is not
*why composition is expensive* but **whether joint training is doing something weight-space
addition cannot reach at all.**

> **If `δ_MIX` lies in the span of `{δ_T, δ_H}`, joint training is approximately weight-space
> addition and MIX's advantage is an allocation/magnitude effect — recoverable by better
> composition. If it does not, joint training finds a direction no sum of the two solo deltas
> can produce, and THAT is the composable-skills gap.**

Nothing on any refuted list touches this. `δ_MIX` appears in the record only as an acquisition
and long-context **level** (225 / 134) and as a rehearsal arm; its **geometry has never been
looked at**.

## Construction

Per trained tensor `t`, ordinary least squares with two unknowns over all elements:

    delta_MIX,t  ~=  a_t * delta_T,t  +  b_t * delta_H,t

Computed twice — once on the **full** deltas, once on their **rank-16 projections**, because the
composition arms are built from `P16`. Reported:

- per-tensor `a_t`, `b_t`, residual fraction `||resid|| / ||delta_MIX,t||`
- global cosine between `δ_MIX` and its best fit in the span
- the same statistics for a **same-skill control**: `δ_T(seed 23)` fitted against
  `span{δ_T(seed 17), δ_H}`, which is the value the MIX figure must be read against

**Sources, all hash-bound in `TRAINED_TENSOR_BACKUP.json` and verified present:**

    Supa source                  Rineya-Supa-1.7.0-devfreeze/model.safetensors        1137 MB
    delta_T  = trained - source  int001-20260917/trained_fp32.safetensors             1201 MB
    delta_H  = trained - source  compose001-20260918/trained_H_fp32.safetensors       1201 MB
    delta_MIX= trained - source  rrp015-20260917/trained_MIX_fp32.safetensors         1201 MB

## BUILD GATE — the canary, run first and reported before any MIX number

**Fit `δ_T` against `span{δ_T, δ_H}`.** The exact answer is known: `a = 1`, `b = 0`, residual
fraction `0`. If the solver does not return that to ~1e-6 the extractor is wrong and **no MIX
number is reported.**

This is the canary rule applied to a solver rather than a parser — the fourth extractor bug in
two days was caught only by collision with a number already held, and this supplies one
deliberately.

## Registered predictions

| # | prediction | reading |
|---|---|---|
| **P1** | global cosine **≥ 0.75** and residual fraction **≤ 0.50** | joint training **IS** approximately weight-space addition; MIX's +6 over `TH16s` is allocation, and better composition should recover it |
| **P2** | global cosine **< 0.25** and residual fraction **> 0.75** | joint training is **NOT** weight-space addition; it reaches a direction no sum of solo deltas produces, and that is the gap |
| **P3** *(control)* | the same-skill fit lands **above** the MIX fit on cosine | confirms the statistic discriminates; if `δ_T(s23)` fits `span{δ_T(s17), δ_H}` no better than `δ_MIX` does, the statistic is measuring the span's expressiveness rather than anything about joint training |

**Anything between P1 and P2 is reported as a NUMBER, not a verdict**, and explicitly as
"partially in the span". Registered now so an intermediate value cannot be argued into either
camp afterwards.

## Confounds, declared before the numbers exist

1. **The budgets differ.** `δ_MIX` was trained on 1,032 rows at double the token budget;
   `δ_T` on 384 and `δ_H` on 648. So MIX is not "T+H at one budget" — it is a different amount of
   training. A high cosine would still be informative; a low one is confounded with dose and
   **must not be read as "joint training is special"** without that caveat attached.
2. **A per-tensor 2-parameter fit is weak.** With ~3M elements and 2 unknowns, a high residual is
   meaningful; a low residual is nearly guaranteed for any two vectors that span enough. **The
   same-skill control in P3 is what makes the number readable** and is the reason it is
   registered rather than optional.
3. One draw of each delta. One lane. Geometry only — **no behavioural claim follows from this
   document.**

## Scope

Geometric. It says nothing about what any arm scores, and **no arm is built or run.** If P2
fires, the follow-up is a behavioural experiment that this document does not register.
