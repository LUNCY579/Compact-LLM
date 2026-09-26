# Rineya-Dataset review for the next composition study

Reviewed 2026-09-26 while TRISKILL-001 Stage A was scoring. This note does not
alter TRISKILL's registered arms, banks, coefficients or decisions.

`Rineya-Dataset` is MAXRL-COMP-002: 41,500 generated primary rows in 10,450
parent clusters. Its four operations are reference resolution (R), state update
(U), exact subset selection (S), and precedence ordering (P). TRAIN contains
12,000 atomic rows and separate RU/US/SP bridge and answerability banks. RS/RP/UP,
all four triples and RUSP are absent from TRAIN and CALIBRATION but present in
the exposed DEV/TEST/CHALLENGE banks. The answer envelope is shared across
skills, which should make format conflict easier to distinguish from the
content conflict observed in TRISKILL-001. The repository's full audit reports
zero exact cross-split collisions, 56,100 active-stage counterfactual controls,
and 462 passing tests. It explicitly reports **zero native model loads and zero
native training runs**. Difficulty, tokenizer lengths, actual success rates and
preservation enrollment remain unqualified.

I independently parsed the four atomic ID manifests: each has exactly 3,000
unique IDs and all six pairwise intersections are empty. This checks the donor
bank inventory, not the full generator or native behavior.

## A bounded next experiment

1. Qualify the exact Rineya tokenizer, chat template, context length, response
   budget, verifier callback and native MaxRL trainer first. Measure the frozen
   source on atomic R/S, RS and legacy guards. If source competence has no useful
   headroom, pick another pair before training.
2. Start R and S donors from the **same immutable anchor** and train only on
   `manifests/atomic_R.json` and `manifests/atomic_S.json`, with the same rollout,
   optimizer, token and tuning budget. Treat bridge rows, warmstart and replay as
   separate interventions. Require each donor to acquire its own skill on
   independent development parents before constructing a sum.
3. Develop separate coefficients `alpha_R` and `alpha_S` on DEV. Freeze them
   before TEST. Compare anchor, both single-skill arms at those same coefficients,
   and `anchor + alpha_R*delta_R + alpha_S*delta_S`. Score atomic R and S **and**
   the dependent RS task; retaining the two atomics alone does not establish a
   successful joint answer. Report per-delta norms, cross terms and same-block
   subspace overlap as descriptive, not as a pass rule.
4. Enroll actual source-correct legacy guard membership, then actual R-donor
   correct membership before any R→S continuation. Report newly lost rows and
   gains separately, with parent-cluster uncertainty. Compare direct composition
   against sequential R→S without replay, with a fixed R replay budget, and a
   separately declared bridge intervention. Never train on guard, DEV, TEST or
   CHALLENGE rows during these comparisons.

The v2 evaluation banks are exposed, so even a positive run is a local research
result rather than sealed confirmation or a guarantee of no forgetting. The
dataset can test composition and retention after native qualification; the data
itself cannot enforce orthogonal weights or lossless skill addition.
