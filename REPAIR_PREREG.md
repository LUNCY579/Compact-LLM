# Repair-mixture pre-registration

Written **before** any repair arm was trained. Thresholds fixed here, not after.

## The target, and why it is this one

The dataset programme set out to fix "general reasoning" and that option is now closed: two
independent instruments agree the model does not do multi-step reasoning, and the puzzle
probe put the base model exactly on its guessing floor (`PUZZLE_CEILING.md`).

What replaced it is better, because it is localized. Opallo's 26-row long-context regression
decomposes into two named behaviours (`LONG_CONTEXT_RETENTION.md`):

| loss | rows | real under both scorers? |
|---|---:|---|
| terse answer formatting | 17 | no — presentation only |
| **abstention / `UNKNOWN`** | **11** | **yes** |

Abstention is the one worth training for. Supa emits `UNKNOWN` on **13 of 64** unanswerable
long-context rows; Opallo on **2**. That gap is identical under strict and
containment-tolerant scoring, so it is a behaviour the model lost, not a string the scorer
missed.

## The corpus, built from rows already cleared

**248 of the pool's 10,000 rows already end in an explicit knowledge-limit reply** —
`"I don't know based on the available information."` — spread across
`insufficient_information`, `evidence_grounded_dialogue`, `evidence_inference`,
`boundary_setting` and `tactful_disagreement`. No new corpus is imported, so there is **no
new contamination surface**: these are the same rows `splits_large` was already cleared for.

`splits_curated_repair_a33_b5233_s17` — 42 rows, **14 of them abstention (33.3%)**, 5,302
supervised tokens against the deliverable's 5,233. Interleaved rather than front-loaded,
because 248 abstention rows would otherwise consume the entire budget and leave the
arithmetic panel nothing to learn from.

Note the thing that makes this non-obvious: the pool's median final reply is **32
characters**. The training data is already terse. Opallo's verbosity is drift *away* from
its targets, not imitation of them — which is why the repair targets abstention, a
behaviour that can be taught by example, rather than terseness, which apparently cannot be
taught by more of what was already there.

## Design

Matched-budget, matched-everything-else. Both arms are `B28last20`, lr 1e-5, `wsd:0.2:0.2`,
β₂ 0.95, at training seeds **17, 23, 29**.

* **Treatment:** `splits_curated_repair_a33_b5233_s17`
* **Control:** `splits_curated_random_b5233_s17` — which at selection-seed 17 reproduces the
  deliverable's 41 rows *exactly, in the same order*, so the control is the shipped model.

Both splits hold exactly their budget, so `arm_dose` consumes **all** rows at every seed and
the training seed changes only the **order**. The four deliverable seeds confounded
"different rows" with "different order"; this does not.

## Pre-registered tests

**Primary — abstention.** `UNKNOWN` emissions on the 64 unanswerable long-context rows,
three treatment arms against three matched control arms.

> **Restored** if the treatment mean is **>= 7 of 64**, midway between Opallo's 2 and Supa's
> 13, **and** the pooled paired McNemar across the three matched pairs, on those 64 rows,
> gives p < 0.05 in the treatment's favour.

Both conditions, not either. A mean of 7 with a null test is three noisy runs; a significant
test at a mean of 3 is a real but useless effect.

**Retention constraint — fixed in `DATASET_PROGRAMME_PREREG.md` before any of this.** The
repair is only an improvement if it holds what Opallo already has:

| | threshold |
|---|---|
| mean panel over three seeds | **>= 62** |
| mean reference answers retained | **>= 37** |

62 and 37 are the worst of the deliverable's own four seeds. An arm that buys abstention by
spending the panel has made a **trade**, and will be reported as a trade with both numbers,
not as a win.

**Reported, not tested:** long-context all256, showcase/stability, per-variant and
per-task-type strata, output length, forge. These are strata and secondary instruments;
testing each would be the multiple-comparisons error this programme has already made.

## What would falsify the whole approach

If the treatment restores abstention **and** holds the panel, the repair mixture is the
recommended corpus. If it restores abstention **and** costs the panel, the two objectives
are in tension at this budget and the honest output is a documented trade-off curve, not a
model. If it does **not** restore abstention, then abstention is not teachable from 14 rows
of in-pool examples and the next step is a different corpus, not a different share.

**The share is not a hyperparameter to sweep.** One value, 0.33, is tested. Sweeping shares
and reporting the best would be selection on the test instrument, which is the failure mode
this programme has already had to withdraw findings for.
