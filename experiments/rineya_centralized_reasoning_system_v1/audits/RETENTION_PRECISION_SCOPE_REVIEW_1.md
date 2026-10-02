# Retention precision scope: feasibility arithmetic only

2026-10-02, reviewer `/root/scientific_audit`. Read
`contracts/LEARNING_ATLAS_MICROSTUDY_V1.json`, `contracts/INSTRUMENTS_V2.json`,
the atlas/v2 retention and parent-interval functions and learning hypothesisH01.
No source/data change, bank creation, source/gold evaluation, model/framework/
tokenizer, host database, worker or training occurred. Shared model/workspace
context remains; this is independently recomputed design arithmetic, not evidence
that any model retained a skill or a power/precision registration.

The actual preliminary recipe has checkpoints0/4/8, at most32 optimizer parents,
source-correct membership frozen at0, original-source and approved-incumbent
comparisons, unchanged2pp aggregate and5pp represented-family margins, and an
explicit ban on noninferiority claims without adequate power.

## Best-case exact zero-damage bounds

Assume **n independently sampled eligible source-correct parent units**, each with
one predeclared binary damage outcome at one fixed look. Observe0 damaged parents.
Then P(0 damages | p)=(1-p)^n, and the exact one-sided95% upper bound is
U=1-0.05^(1/n). This is binomial-tail inversion, not an asymptotic rule of three.
The exact binomial confidence procedure's coverage is described in the primary
[R stats documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html).

| Independent eligible parents n | 0-damage one-sided95% upper bound |
| ---: | ---: |
| 8 | 31.234398% |
| 16 | 17.074972% |
| 32 | 8.936820% |
| 64 | 4.572970% |

To make U<=delta even in this zero-damage best case, the minimum is
ceil(log(alpha)/log(1-delta)):

- At delta=.02, alpha=.05: **149** parents; n148 gives2.003795%, n149 gives1.990482%.
- At delta=.05, alpha=.05: **59** parents; n58 gives5.033934%, n59 gives4.950761%.

These are **feasibility floors**, not full power, actual enrollment targets or
retention evidence. Damages above zero, meaningful alternative rates, correlated
units, missing observations and additional claims require a separate design.
They apply to a binary damage-rate estimand, not automatically the registered
paired accuracy-difference noninferiority endpoint.

## Units, endpoints and checkpoint looks

Variants of one parent do not increase independent n. One may predeclare an
any-variant-damaged parent indicator, but that is a different estimand from a
variant-weighted accuracy loss. Equal/fixed parent weighting and within-parent
aggregation must be declared. Do not apply the binomial formula directly to
continuous parent-mean deltas or count repeated variants as independent trials.

Damage and net gain are distinct: accuracy delta=(repairs-losses)/scheduled items
can be zero while previously correct items are lost. Report source-correct losses/
denominators, source repairs, worst-family effects and incumbent losses separately.
Conditional source-correct damage is not raw all-item accuracy; any conversion or
sufficient preservation rule needs explicit weighting/enrollment assumptions.

Checkpoint0 enrolls baseline membership; updates4/8 are two post-update looks.
Illustration only: allocating alpha=.05/2=.025 to **one** predeclared statement
at each of those two fixed looks raises the zero-damage floors to183 for2% and72
for5%. This adopts **no** multiplicity policy and does not cover additional family,
skill, recipe or source/incumbent comparisons. Repeated checking until a pass or
adaptive checkpoint/recipe selection is not an unadjusted fixed-look test; freeze
the look schedule and multiplicity/sequential error rule before outcomes.

## What is missing for a real registration

The32 optimizer parents are known TRAIN exposure, not an independently enrolled
retention panel. No qualified source/checkpoint0 semantic outcomes currently
establish which original-skill parents are source-correct; the eligible n is
unmeasured. Source and approved-incumbent evidence must be identified separately.
The v2 atlas correctly leaves absent incumbent quantities/null denominators
unavailable and `noninferiority_established=false`; the original v1 fallback to
source must not be used to claim an unavailable incumbent comparison. Parent
bootstrap intervals are descriptive and likewise grant no noninferiority.

A justified precision/power registration needs a fixed target population and
independent parent sampling/enrollment, exact source/incumbent/checkpoint IDs,
qualified scoring and failure rules, baseline accuracy/source-correct rate,
binary-damage or paired-difference estimand/weighting, unchanged margins, explicit
alpha/multiplicity/looks, a meaningful alternative and target power, and the
dependence/missingness/replication assumptions used in its calculation. Evaluate
power for that specified test, including family allocation and training draw/order
variation. No panel access, margin relaxation, new bank, promotion or claim of
preservation is authorized or implied by these calculations.
