# TGSA Phase 1 v3 — outcome-blind power report

**Produced:** 2026-08-29, before any confirmatory outcome exists.
**Estimator:** `crossed_cluster_t` (two-way cluster-robust, `t_{m-1}` reference).
**Registered primary gate:** lower 95% confidence bound above zero.
**Simulator:** `tgsa/phase1_simulation.py`, driven by `scripts/phase1_power_simulation.py`.
**Trials:** 10,000 per cell for the full grid; 20,000 per cell for the seed-count refinement.

Nothing in this report reads a checkpoint, a prediction file, or any TGSA
outcome. The simulator draws synthetic paired differences from an explicit
generative model and runs *the registered estimator* over them, so the procedure
that was calibrated is the procedure that is registered.

## Generative model

The registered crossed structure is preserved exactly:

```
d[s, k, e] in {-1, 0, +1}
E d[s, k, e] = theta + a_s + b_[k, e]
a_s      ~ N(0, sigma_a^2)     per training seed
b_[k, e] ~ N(0, sigma_b^2)     per example, SHARED across every seed
P(+1) = (rate + mu) / 2,  P(-1) = (rate - mu) / 2
```

`rate = 0.30` is the disagreement rate (fraction of examples on which the two
arms differ). The parameterisation is calibrated so `E[d] = mu` exactly, which
is verified by a regression test — a simulator whose `true_effect` is not on the
accuracy scale silently reports power for the wrong effect size, and an earlier
draft had exactly that defect.

Layout: 17 primary strata x 70 examples = 1,190 primary test examples, matching
the registered generator's primary-family structure.

**Registered strata are FIXED, not random.** The estimand macro-averages over the
exact strata the generator produces, so a stratum-level random effect is not a
source of sampling variability. See the calibration report for why this matters.

## Headline result: the v2 gate had a structural 50% ceiling

The v2 rule required **both** `observed >= +0.020` **and** `lower bound > 0`.
Decomposing it at a true effect of exactly +0.020 (10,000 trials):

| component | m=5 | m=8 | m=10 | m=12 | m=15 | m=20 |
|---|---:|---:|---:|---:|---:|---:|
| `lower CI bound > 0` alone | 0.278 | 0.468 | 0.589 | 0.677 | 0.791 | 0.902 |
| `observed >= +0.020` alone | 0.500 | 0.490 | 0.500 | 0.500 | 0.497 | 0.503 |
| **v2 conjunction** | 0.251 | 0.392 | 0.460 | 0.488 | 0.496 | 0.503 |

The middle row is flat at 0.50 in **both** seed count and seed variance. That is
mathematically necessary: an unbiased, symmetric estimator exceeds its own mean
half the time, so when the true effect *equals* the threshold, requiring the
point estimate to clear it is a coin flip. **No seed count can push the v2
conjunction above ~50% at a true +0.020.** This is a property of the decision
rule, not of replication.

The v3 gate therefore tests against zero, and +0.020 becomes the effect the
replication level is *powered for* — the conventional formulation. The threshold
was not raised, and the claim was not weakened: the gate is strictly the
statistical test, and the observed effect is still reported against +0.020
descriptively.

## Seed-count refinement (20,000 trials per cell)

Detection probability at a true effect of +0.020 under the registered gate:

| seeds | sigma_a = 0.02, power [95% CI] | sigma_a = 0.04, power [95% CI] |
|---:|---|---|
| 13 | 0.7230 [0.7168, 0.7292] | 0.3286 [0.3222, 0.3352] |
| 14 | 0.7604 [0.7545, 0.7663] | 0.3572 [0.3505, 0.3638] |
| 15 | 0.7923 [0.7866, 0.7979] | 0.3802 [0.3735, 0.3870] |
| **16** | **0.8117 [0.8063, 0.8171]** | 0.4085 [0.4017, 0.4153] |
| 17 | 0.8476 [0.8426, 0.8525] | 0.4299 [0.4231, 0.4368] |
| 18 | 0.8673 [0.8625, 0.8719] | 0.4542 [0.4473, 0.4611] |
| 19 | 0.8799 [0.8753, 0.8843] | 0.4789 [0.4720, 0.4858] |
| **20** | **0.8962 [0.8919, 0.9004]** | 0.4920 [0.4851, 0.4989] |
| 24 | 0.9435 [0.9402, 0.9466] | 0.5824 [0.5756, 0.5892] |
| 28 | 0.9667 [0.9641, 0.9691] | 0.6460 [0.6393, 0.6525] |
| 32 | 0.9850 [0.9833, 0.9866] | 0.7087 [0.7024, 0.7150] |

At a true effect of +0.030:

| seeds | sigma_a = 0.02 | sigma_a = 0.04 |
|---:|---:|---:|
| 16 | 0.9892 | 0.7322 |
| 20 | 0.9981 | 0.8305 |
| 28 | 1.0000 | 0.9393 |

## Type-I control

Empirical rejection rate under the null (true effect 0.000), registered gate:

| seeds | 13 | 15 | 16 | 18 | 20 | 24 | 28 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sigma_a = 0.02 | 0.0250 | 0.0233 | 0.0222 | 0.0237 | 0.0215 | 0.0221 | 0.0270 | 0.0257 |
| sigma_a = 0.04 | 0.0259 | 0.0239 | 0.0244 | 0.0262 | 0.0260 | 0.0242 | 0.0260 | 0.0252 |

Nominal one-sided rate is 0.025. Observed range is 0.0215–0.0270 across every
seed count and both variance scenarios. **The registered gate is correctly
calibrated, not anticonservative.**

Interval coverage of the true effect is 0.949–0.954 against a nominal 0.95.

## Registered replication level

**20 seeds.**

Justification:

* 16 seeds is the smallest count whose *lower* confidence bound on power clears
  the 0.80 target (0.8117, CI [0.8063, 0.8171]). It qualifies, but with roughly
  one percentage point of margin.
* `sigma_a` is an **assumption, not a measurement**. It is unknown until the
  matrix runs, and the seed-count answer is highly sensitive to it. Sizing at the
  bare minimum for an assumed variance leaves no headroom if the assumption is
  even slightly optimistic. 20 seeds gives 0.896 at the central scenario — real
  margin — for a 25% compute increase over 16.
* At 20 seeds, power at +0.030 is 0.998 (central) and 0.831 (elevated), so the
  design is comfortable for effects moderately above the powering target.
* Type-I is 0.0215 at 20 seeds, indistinguishable from nominal.

Registered seeds (20): 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111,
1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020. The first five are the v2
seeds, so the v3 design is a strict superset; the remainder follow a fixed
arithmetic sequence so the list is reproducible and obviously not selected.

## Residual risk, stated plainly

**At elevated seed variance (`sigma_a = 0.04`), 20 seeds gives only 0.492 power
at a true +0.020.** Reaching the 0.60 robustness target at that variance requires
**28 seeds**; reaching 0.80 requires more than 32.

This is not hidden and is not resolved by the registered design. If seed-to-seed
variance turns out to be at the elevated end, a null result at +0.020 will be
genuinely ambiguous rather than informative, and should be reported as
underpowered rather than as evidence against the mechanism.

Two facts bound this risk:

* the design uses byte-identical canonical initialization, an identical fixed
  step budget, identical data order, and zero dropout within each seed, which is
  the configuration most likely to produce *low* seed variance;
* the analysis reports the realised seed-level spread, so the assumption becomes
  checkable after the fact even though it cannot be checked before.

## The precision floor

`Var(theta_hat) = sigma_a^2/m + sigma_b^2/(K*n) + sigma_eps^2/(m*K*n)`.

The middle term does **not** shrink with replication: every seed is evaluated on
the same materialized test split, so example-level variation is common to all of
them. Adding seeds attacks only the first and third terms. This is why the power
curves flatten, and why the honest ceiling on this design is set by the test-set
size rather than by the number of training runs.

Enlarging the test split — not adding seeds — is the lever that lowers that
floor. That is a dataset-version change and is out of scope for v3.

## Reproduction

```bash
python scripts/phase1_power_simulation.py \
  --config configs/phase1_identifiability_v3.json \
  --output power_audit.json \
  --primary-gate lower_bound_above_zero \
  --seed-counts "13,14,15,16,17,18,19,20,24,28,32" \
  --effects "0.000,0.020,0.030" \
  --seed-sds "0.02,0.04" \
  --trials 20000
```
