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

---

# RESULT — "partially in the span", and the control supplies the scale the registration lacked

> The STATUS banner above is superseded by this section rather than edited.

**Build gate PASSED.** Fitting `δ_T` onto `span{δ_T, δ_H}` returned `a = 1.000000000`,
`b = 6.381e-19`, residual fraction `4.482e-16`. The solver is right.

## The registered contrast

    delta_MIX ~= a_t * delta_T(s17) + b_t * delta_H     220 tensors, FULL deltas, per-tensor OLS

    global cosine(delta_MIX, best fit in span)  0.7429
    residual fraction                           median 0.6691   [0.5386, 0.7872]
    a_t (weight on delta_T)                     median +0.457   [+0.026, +0.840]
    b_t (weight on delta_H)                     median +0.817   [+0.479, +1.086]

| # | registered | observed | |
|---|---|---|---|
| **P1** | cos ≥ 0.75 **and** residual ≤ 0.50 | 0.7429 / 0.669 | **MISSED** — by 0.007 on cosine, decisively on residual |
| **P2** | cos < 0.25 **and** residual > 0.75 | 0.7429 | **decisively FAILED** |
| **P3** | same-skill fit lands **above** the MIX fit | **0.525 vs 0.743 — BELOW** | **FAILED as written** |

**So the registered verdict is the middle one: report a NUMBER, not a verdict.** That rule was
registered precisely so an intermediate value could not be argued into either camp, and 0.743 is
intermediate.

## P3 failed in the OPPOSITE direction to the worry, and that is the finding

P3 existed to catch a statistic that cannot discriminate — the fear was that a 2-parameter fit
over ~3M elements would fit *anything* well, so a same-skill draw would also score high.

The opposite happened. Fitting another draw of the **same** skill scores **lower**:

| target | basis | cos | `b` median | |
|---|---|---:|---:|---|
| `δ_T(s23)` | `span{δ_T(s17), δ_H}` | 0.5246 | **−0.0002** | δ_H contributes **nothing** |
| `δ_T(s23)` | `span{δ_T(s17), δ_T(s29)}` | **0.5906** | +0.3140 | two genuinely useful basis vectors |
| **`δ_MIX`** | `span{δ_T(s17), δ_H}` | **0.7429** | **+0.8172** | both load-bearing |

**The degrees-of-freedom confound is controlled by the middle row.** The first control uses only
one effective basis vector (`b ≈ 0`, because δ_H is useless for fitting a T-redraw), so comparing
it to MIX's two-DOF fit would not have been like-for-like. Giving the same-skill control **two
useful** basis vectors raises it to 0.591 — still far below MIX's 0.743.

> **δ_MIX is better explained by the span of the two solo deltas than an independent redraw of
> δ_T is explained by the span of two other δ_T draws.**

And `b` is the discriminating signal: **0.000 when fitting a T-redraw, 0.817 when fitting MIX.**
δ_H is not a generic basis vector that helps fit anything — it is load-bearing specifically for
the jointly-trained delta. That is the check that makes the span result mean something.

## What the registration got wrong, and the general form

**The registration compared 0.743 against 1.0 and called that "partially in the span".** The
control says the right denominator is the **replication ceiling of 0.591** — the most any
two-DOF fit achieves between independent runs of one skill.

> **Against 1.0, MIX looks substantially outside the span. Against what replication itself
> achieves, it is comfortably inside.** A number needs its own baseline, not the theoretical
> maximum.

Third instance of this shape: `matched-construction-is-not-matched-difficulty` (a scale borrowed
from another bank is not a scale), the arithmetic panel measuring transfer against a source that
scores 47 rather than 0, and now this.

## What this does and does not support

**Supports:** joint training is, to a good approximation, **weight-space addition of the two
skill directions**, at roughly `0.46 · δ_T + 0.82 · δ_H`. MIX's 225 against composition's 219 is
then an *allocation* difference — MIX weights H nearly twice as heavily as T — rather than joint
training reaching somewhere addition cannot.

**Does not support** — and these were declared before the numbers existed:

1. **"MIX is addition."** The residual is still **0.669**: about two thirds of each tensor's MIX
   update lies outside the span. *More addition-like than replication* is not *is addition*.
2. **The budgets differ.** δ_MIX saw 1,032 rows at double the token budget against δ_T's 384 and
   δ_H's 648. The `b > a` weighting is consistent with H's larger share (648 vs 384) and is
   **confounded with dose**. It cannot be read as "joint training prefers the gate skill".
3. **Geometry only.** No arm was built or run and **no behavioural claim follows.** That MIX sits
   inside the span does not establish that any particular sum reproduces its 225.

One draw of each delta, one lane.
