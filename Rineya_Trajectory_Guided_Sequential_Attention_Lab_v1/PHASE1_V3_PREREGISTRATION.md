# TGSA Phase 1 preregistration — study v3 (powered confirmatory)

**Registered:** 2026-08-29, before any confirmatory outcome exists.
**Study ID:** `tgsa-phase1-identifiability-v3-2026-08-29`
**Branch:** `research/tgsa-phase1-identifiability-v3-powered-confirmatory`
**Parent commit:** `e0063f0e784da69a33a7669956ff0ac13daa2cd6` (tag `tgsa-phase1-v2-launch-ready-frozen`)
**Pinned Phase-0 ancestor:** `aefb70ebed2e7f57b9cbfe25c24dfdc0aee2fd4e`
**Configuration:** `configs/phase1_identifiability_v3.json`
**Supersedes:** `tgsa-phase1-identifiability-v2-2026-08-29` (launch-ready, never executed, preserved immutably)

## Scope of the amendment

v2 closed every engineering and evidentiary defect from the launch audit and was
mechanically launch-ready. It was blocked on one thing: statistical power.

**v3 changes only the statistical design.** Specifically: the primary decision
rule, the interval estimator, and the replication level. Everything else is
inherited from v2 unchanged and must remain unchanged:

- the three required arms `cascade`, `history_control`, `learned_delta`;
- identical model architecture except for registered mechanism restrictions;
- canonical per-seed supernet initialization with byte-identical shared tensors;
- the residual-around-identity trajectory encoder with zero-initialized residual;
- the same fixed training budget of exactly 2,500 updates per cell;
- zero dropout;
- the same filtered task generator, including generation-time rejection of
  trivial primary examples;
- the same `direct_lookup` non-inferiority guardrail;
- the same replay-derangement alignment falsifier with registered derangement
  indices `[0, 1, 2]`;
- train -> full-matrix barrier -> confirmatory-evaluate orchestration;
- the complete hash-bound evidence-chain verifier;
- no post-hoc exclusions;
- no confirmatory outcome inspection before the matrix is frozen.

Two supporting reports are part of this registration and were produced before any
outcome exists:

- `PHASE1_V3_POWER_REPORT.md`
- `PHASE1_V3_CALIBRATION_REPORT.md`

## Research question

Does learned finite-difference trajectory guidance add capability beyond

1. sequential depth itself, and
2. an information-matched learned two-state history pathway?

## Hypothesis hierarchy

- **H1a:** `learned_delta` > `history_control` on the macro-average of held-out
  pointer-chase and state-update strata.
- **H1b:** `learned_delta` > `cascade` on the same estimand.
- **H1c:** aligned `learned_delta` > `replay_derangement(learned_delta)`.
- **Guardrail:** `learned_delta` is non-inferior to `history_control` on
  `direct_lookup`.

### Interpretation, fixed in advance

| Outcome | Registered interpretation |
|---|---|
| H1a, H1b, H1c all pass | Controlled evidence that trajectory-specific, example-aligned information adds capability beyond the sequential cascade and an information-matched history control. |
| H1b passes, H1a fails | Evidence for an extra learned recurrent pathway, **not** for trajectory-specific capability. |
| H1a and H1b pass, H1c fails | The finite-difference parameterization may help, but example-aligned trajectory content has **not** been shown causal. |
| Primary interval includes zero | **Inconclusive, not success.** |
| Effect is negative | Evidence **against** the mechanism in this setting. |

The analyzer emits this interpretation directly into `analysis.json` under
`hypothesis_hierarchy.interpretation`, so it cannot be renegotiated after the fact.

## What changed, and why

### 1. Primary gate: lower confidence bound above zero

The v2 rule required **both** `observed >= +0.020` **and** `lower bound > 0`.
The first clause caps detection probability at exactly 50% when the true effect
equals the threshold, because an unbiased symmetric estimator exceeds its own
mean half the time. Measured across 10,000 trials, that component sits at
0.487–0.510 at every seed count and every variance scenario. **No amount of
replication can lift the v2 conjunction above ~50% at a true +0.020.**

The v3 primary gate is therefore the standard test:

> **lower 95% confidence bound above zero.**

`+0.020` is retained as the **registered minimum effect of interest**: the effect
the replication level is *powered to detect*, reported descriptively against the
observed effect in every artifact. It is not raised, not lowered, and not
weakened — it moves from being a gate clause to being the design target, which is
the conventional formulation.

Corresponding powering targets: `+0.020` for H1a and H1b, `+0.015` for H1c.

### 2. Interval estimator: `crossed_cluster_t`

Seed and example are crossed: every training seed is evaluated on the same
materialized test split. The registered estimator is the two-way cluster-robust
variance of Cameron, Gelbach & Miller (2011) with a `t_{m-1}` reference:

```
V = c_s * V_seed + c_e * V_example - V_cell
```

on the weighted residuals of the registered macro-average. Expanding term by term
recovers `sigma_a^2/m + sigma_b^2/(K n) + sigma_eps^2/(m K n)`, so it is
essentially unbiased for the true sampling variance.

Measured calibration: null rejection 0.0229–0.0260 (nominal 0.025) and coverage
0.9492–0.9535 (nominal 0.95) across four seed counts and four variance scenarios.

The v2 incumbent `crossed_pigeonhole` was **not** retained: it is unstable, with
coverage ranging from 0.9103 (undercovering) to 0.9843 (grossly overcovering)
depending on a nuisance parameter unknown before launch. `crossed_pigeonhole_t`
is anticonservative at small seed counts. `seed_cluster` undercovers when the
example component dominates.

`seed_cluster` and both pigeonhole variants are computed and reported as
**preregistered sensitivity analyses** on every contrast. Only
`crossed_cluster_t` gates. `seed_cluster` was explicitly not adopted as primary,
because doing so would narrow the claim to this materialized test split.

### 3. Replication: 20 seeds

Registered seeds (20): 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111,
1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020.

The first five are the v2 seeds, making v3 a strict superset; the rest follow a
fixed arithmetic sequence so the list is reproducible and evidently not selected.

Measured detection probability at a true +0.020 under the registered gate
(20,000 trials): 0.8962 at `sigma_a = 0.02`. Sixteen seeds is the smallest count
whose lower confidence bound clears 0.80 (0.8117, CI [0.8063, 0.8171]); 20 was
registered to leave margin, because `sigma_a` is an assumption rather than a
measurement.

**Stated residual risk.** At `sigma_a = 0.04`, 20 seeds gives only 0.492 power at
a true +0.020. Reaching 0.60 there needs 28 seeds; 0.80 needs more than 32. If
the realised seed spread is at the elevated end, a null result at +0.020 must be
reported as **underpowered**, not as evidence against the mechanism. The analysis
reports the realised seed-level spread so this becomes checkable after the fact.

### 4. The precision floor

`Var(theta_hat) = sigma_a^2/m + sigma_b^2/(K n) + sigma_eps^2/(m K n)`. The
middle term does not shrink with replication, because every seed sees the same
test split. Adding seeds attacks only the first and third terms, which is why the
power curves flatten. Lowering that floor requires a larger test split, which is
a dataset-version change and is out of scope for v3.

## Fixed design

- 20 paired seeds; 3 required arms; 60 registered cells.
- Identical materialized examples, minibatch order, update count, optimizer,
  width/depth, and canonical initial shared tensors within each seed.
- `history_control` and `learned_delta` state dictionaries and parameter counts
  are exactly identical at initialization; every arm is a byte-exact restriction
  of its seed's canonical supernet.
- Dropout zero. Exactly 2,500 updates. No dev-based checkpoint selection.
- The final checkpoint is hashed at end of training and re-verified before the
  evaluation phase reads it.
- Test predictions retained at instance level.

## Data

Generated deterministically from seed 731928, materialized before training.
Cross-split latent signatures disallowed. 5,000 train / 900 dev / 1,800 test.
Primary families reject trivial items at generation time (measured: pointer-chase
trivial rate 0.302 -> 0.000, copy-start baseline 0.150 -> 0.000).

Registered test ranges partially overlap training ranges, so every claim says
**held-out generated-task performance on instance-held-out examples**, never
uniform out-of-range extrapolation.

## Primary estimand

For each paired seed and each registered non-guardrail stratum: paired per-example
correctness difference; average within stratum; macro-average strata within seed;
average across the 20 paired seed effects. Interval by `crossed_cluster_t` at 95%.

Answer-NLL improvement is a supportive endpoint, not a replacement for accuracy.

## Fixed gates

| Gate | Rule |
|---|---|
| H1a | lower 95% bound of (`learned_delta` − `history_control`) > 0 |
| H1b | lower 95% bound of (`learned_delta` − `cascade`) > 0 |
| H1c | lower 95% bound of (aligned − replay-deranged) > 0 |
| Guardrail | lower bound of the `direct_lookup` contrast >= −0.01 |
| Falsifier coverage | minimum per-seed length-matched derangement coverage >= 0.95 |
| Evidence integrity | every check in `scripts/verify_phase1_evidence.py` passes |

The integrity gate is **derived** from the verifier, never a constant.

## Decision rule

`decision` is forced to `smoke_only` unless the configuration declares
`confirmatory: true` **and** the analyzed matrix is the complete registered
arm/seed matrix **and** the study ID matches **and** the config hash matches.

## Secondary difficulty-response analyses (non-gating)

Preregistered, and explicitly unable to rescue a failed primary endpoint.

Effect size is estimated within bins of registered generator-recorded latent
variables: pointer-chase effective hop count; state-update effective step count;
registered context bucket; evidence position; evidence span; distractor count;
node count. Discrete variables bin on their own value. Continuous variables are
cut at quantiles **of the materialized dataset**, fixed before training. No bin
is chosen after seeing outcomes.

Each variable also reports a centred unit-norm linear trend contrast across its
ordered bins, estimated with the same registered estimator.

The mechanistically interesting signature — little or no gain on `direct_lookup`
and increasing gain as effective iterative depth rises — is secondary evidence
only. A monotone difficulty response with a null primary contrast is a hypothesis
for the next study, not evidence for this one.

## Efficiency (descriptive, never gating)

Recorded: training wall clock, seconds per update, prefill latency, decode-proxy
latency, throughput, peak memory, parameter count. All under `gating: false`.

The research objective is to determine whether sequential trajectory-guided
computation can reliably increase accuracy **even when the price is slower
inference**. A slower arm that is reliably more accurate is an acceptable and
potentially important result. No efficiency metric may veto a capability-positive
result, and sequential computation must not be optimized away before it is known
whether it helps.

## Exclusions and failures

No post-hoc exclusions. Difficulty rejection happens at generation time. Failed
attempts are immutable, content-addressed, never overwritten. `--force-data` is
refused for a confirmatory study; `--overwrite` cannot be combined with it;
confirmatory execution from a dirty checkout is refused. A partial matrix may be
analyzed only with `--allow-partial` and remains `smoke_only`.

## Claim ceiling

Even if every gate passes, the strongest supported statement is:

> In a controlled tiny decoder trained on the registered synthetic iterative
> tasks, an aligned finite-difference representation improved held-out
> generated-task capability beyond an information-matched history representation
> and beyond the sequential cascade alone.

Not established: natural-language generalization, long-context performance,
Qwen/GQA compatibility, wall-clock or memory efficiency, scaling behaviour,
novelty, broad architectural superiority, or an arXiv-primary claim.

## Change control

Any change to task generation, counts, arms, seeds, architecture, optimizer,
training steps, interventions, estimand, estimator, gate, or powering targets
after confirmatory outcomes are visible requires a new study ID, a new output
directory, and an amended preregistration. Existing artifacts remain immutable.
