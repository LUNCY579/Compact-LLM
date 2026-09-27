# COMP-RS-001 — native R/S acquisition, composition and preservation

**Working protocol, not yet a training grant or frozen statistical preregistration.**
Q0 source/data binding and Q1 Supa asset/tokenizer checks are complete on the
local Windows host. Native loader, completion/logprob/backward parity, actual
source competence, preservation enrollment, cost ceilings and statistical
margins remain open. No R or S donor has been trained.

This document turns the user-provided R/S design into executable gates without
changing the audited MAXRL-COMP-002 corpus or its binary verifier. The dense,
same-anchor comparison is the primary study. Sequential order, replay and
mechanism interventions retain separate identities.

## Stage 0 — exact native qualification, then freeze the study

Candidate anchor: `Rineya-Supa-1.7.0-devfreeze`, weight SHA-256
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
Candidate runtime: `.venv-mq001` (Python 3.11.15, PyTorch 2.10.0+cu126,
Transformers 4.57.1). These are **candidates**, not a training binding until
native tests pass. The Q1 tokenizer census found a maximum 2,307 prompt tokens
on the exposed planning banks; a 128-token response candidate fits the model
config's 40,960 positions. The loaded tokenizer reports `bos_token_id=None`
although the config lists 151643; the runner must follow the actual rendered
chat prefix and bind EOS/PAD behavior explicitly.

The native gate must prove loaded tensor identity, exact trainable inventory,
completion/EOS masks, selected-token logprobs against independent teacher
forcing, cache/full-prefix and padded/unpadded parity, and a separate
backward-only unchanged-state probe. Then collect actual frozen-source R/S
CALIBRATION groups and complete `guard_dev` enrollment with fixed settings.
Measure all-failure/mixed/all-success rates before selecting a group size or
effect margins. Record generated-token, prefill, verifier, memory, wall and
disk costs. No TEST/CHALLENGE or guard_test/challenge read in this stage.

Select exactly one R/S pair, anchor, parameter inventory, precision, two
objective recipes, ceilings, seed policy, coefficient plateau rule, primary
endpoints, noninferiority margins, parent-cluster uncertainty and multiplicity
**before** the first donor run. The original MAXRL-COMP-002 comparison of GRPO
versus MaxRL remains a primary objective factor with matched tuning allowance;
neither is silently substituted by SFT or a cached oracle rollout.

## Stage 1 — independent donors

Train R and S separately from the identical immutable anchor, using only
`manifests/atomic_R.json` and `atomic_S.json` (3,000 TRAIN parents each). Keep
trainable tensors, tokenizer, sampling group size, precision, optimizer family,
token/compute budget and tuning allowance matched within each objective
comparison. Both policies sample their **own** online rollout groups after
diverging. Preserve raw token IDs, policy snapshot IDs, all generated attempts,
binary verifier outcomes, stop reasons and cost. No bridge, replay, warmstart,
pruning or projection enters this baseline.

Each donor must pass an independently specified atomic-skill acquisition gate
on development parents before its delta is composed. A donor that did not
acquire its own skill cannot be evidence for or against skill composition.

## Stage 2 — separate coefficient calibration on DEV only

Measure R-alone and S-alone dose curves, provisionally at coefficients
`{0.25, 0.50, 0.75, 1.00}`. The final grid and a smallest-on-plateau selection
rule, including its tolerance, must be frozen before any DEV dose response is
opened. Do not use an arm's numerically highest single point when several are
within the declared plateau. Freeze `alpha_R` and `alpha_S` separately; never
rescale their sum.

## Stage 3 — primary dense composition

Build the four arms from one exact anchor:

```
B  = theta0
R  = theta0 + alpha_R * delta_R
S  = theta0 + alpha_S * delta_S
RS = theta0 + alpha_R * delta_R + alpha_S * delta_S
```

The R and S solo controls use the **same coefficients** as RS. Check tensor
keys, finite values, tied aliases, per-tensor norms, total norm and cross term,
construction/cast, weight hashes and unchanged anchor bytes. Score atomic R,
atomic S, dependent RS and source-correct legacy guards. Freeze every arm
before one held-out TEST read; CHALLENGE is a later bounded stress read.

The unchanged binary verifier is the strict/reward endpoint. The separately
registered bounded content parser diagnoses clean wrapper failures. Report
strict/content by arm, family and signature; neither a format rescue nor net
accuracy may cancel lost source-correct rows.

For every dependent RS row, publish the BASE/R/S/RS correctness vector and
count **composition-only repairs** (`B=R=S=0, RS=1`) and **joint damage**
(`max(R,S)=1, RS=0`). Parent-cluster intervals use whole generated worlds as
the unit. Retaining atomic R and S without RS success is not a joint-skill
result.
The exposed DEV bank contains 250 aligned parent worlds with one R, S and RS
core view each; its source-bound ID/parent digests are in `evidence/Q1_DEV_SCOPE.json`.

## Stage 4 — sequential order controls

From the same anchor and matched budgets, run both no-replay orders `R→S` and
`S→R`. Enroll acquired-R correct membership before R→S and acquired-S before
S→R. Compare lost previously correct rows, not merely net score. The order
panel distinguishes recency/asymmetry from generic sequential interference.
Fixed replay for both orders and explicit RU/US/SP bridge adaptation are
separately registered follow-ups, launched only if the baseline warrants them.
Replay uses atomic TRAIN rows and the separate legacy replay bank, never
evaluation guards or held-out combinations.

## Stage 5 — descriptive mechanisms

Measure unclipped target-token log-odds interaction
`I = f_RS - f_R - f_S + f_B` separately on R, S and RS prompts. Predeclare
token/position selection and denominator for any normalized `kappa`; clipping
is robustness only. Record global/per-layer/per-module delta cosine and norm,
effective rank, top-k energy, same-block versus cross-block/Haar subspace
collision, especially `o_proj` and `down_proj`. On fixed development inputs,
collect activation covariance and the weighted cross term
`tr(delta_R^T delta_S Sigma)` under a bound memory/precision protocol.
These explain outcomes; no cosine or collision threshold decides success.

## Stage 6 — distinct follow-up

Only after a dense R/S baseline exists, test F-WANDA magnitude, activation
and Fisher-aware pruning at identical keep counts on the donor that causes
measured retention/composition damage. Keep the other donor and coefficients
fixed. RoLA, NGU, GFlowRL, tangent arithmetic and cross-scale transport are
other experiments, not extra arms in COMP-RS-001.

## Still-open fields before a training grant

Actual source competence and mixed-group rates; selected full trainable
inventory; GRPO/MaxRL native parity; training token/rollout/optimizer ceilings;
coefficient plateau tolerance; atomic/RS/legacy margins and power; source and
acquired-skill membership; response-level strict/content reliability; independent
evaluation and historical-exclusion limits. None is inferred from CPU tests,
synthetic fixtures or adjacent Titan results.
