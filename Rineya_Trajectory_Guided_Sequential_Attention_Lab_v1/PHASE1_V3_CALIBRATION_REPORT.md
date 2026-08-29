# TGSA Phase 1 v3 — null-calibration and coverage report

**Produced:** 2026-08-29, before any confirmatory outcome exists.
**Driver:** `scripts/phase1_inference_calibration.py`
**Trials:** 20,000 per cell (analytic estimators); 3,000 x 999 resamples (bootstrap estimators).
**Layout:** 17 strata x 70 examples, matching the registered primary-family structure.

The v2 remediation reported a ~7% null rejection rate for the registered gate and
flagged it as possibly Monte Carlo noise, possibly small-sample bootstrap
miscalibration, possibly a real inferential defect. This report resolves that
question and selects the v3 estimator on evidence rather than assertion.

## Question 1 — where did the v2 ~7% come from?

**Answer: a simulator misspecification, amplified by Monte Carlo noise. Not an
estimator defect.**

The v2 simulator drew a **stratum-level random effect** in addition to seed and
example effects. That is a modelling error. The registered strata are a *fixed*
set produced by the registered generator, and the estimand macro-averages over
those exact strata, so stratum-level variation is not a source of sampling
variability for this estimand. Injecting one adds variance that no correctly
specified estimator can track, which manufactures apparent anticonservatism.

Probe, `crossed_cluster_t` at m=5, sigma_a=0.02, 20,000 trials:

| injected stratum SD | null rejection | coverage | SE calibration ratio |
|---:|---:|---:|---:|
| 0.00 (correct) | 0.0240 | 0.9521 | 0.938 |
| 0.02 | 0.0290 | 0.9406 | 0.867 |
| 0.05 (v2 simulator) | **0.0598** | **0.8785** | 0.653 |

The v2 figure also rested on 150 trials, whose Monte Carlo standard error at a
7% rate is +/-2.1 percentage points — far too noisy to distinguish 3% from 11%.

Both causes are now removed: the simulator treats strata as fixed by default,
and every number in this report uses at least 3,000 trials.

## Question 2 — is the v2 incumbent estimator acceptable?

**Answer: no. It is unstable, not merely anticonservative.**

Null rejection rate of `lower CI bound > 0` (nominal **0.025**):

| estimator | m | sigma_a=0.01 | 0.02 | 0.04 | 0.06 |
|---|---:|---:|---:|---:|---:|
| **`crossed_cluster_t`** | 5 | 0.0241 | 0.0245 | 0.0240 | 0.0233 |
| | 10 | 0.0244 | 0.0229 | 0.0235 | 0.0232 |
| | 16 | 0.0245 | 0.0244 | 0.0260 | 0.0253 |
| | 20 | 0.0237 | 0.0243 | 0.0240 | 0.0231 |
| `crossed_pigeonhole` (v2) | 5 | — | 0.0157 | **0.0453** | — |
| | 10 | — | 0.0080 | 0.0283 | — |
| | 20 | — | 0.0067 | 0.0143 | — |
| `crossed_pigeonhole_t` | 5 | — | **0.0467** | 0.0330 | — |
| | 10 | — | 0.0300 | 0.0350 | — |
| | 20 | — | 0.0277 | 0.0287 | — |
| `seed_cluster` | 5 | 0.0272 | 0.0237 | 0.0251 | 0.0245 |
| | 20 | **0.0314** | 0.0278 | 0.0225 | 0.0270 |

Interval coverage of the true effect (nominal **0.95**):

| estimator | m | sigma_a=0.01 | 0.02 | 0.04 | 0.06 |
|---|---:|---:|---:|---:|---:|
| **`crossed_cluster_t`** | 5 | 0.9517 | 0.9518 | 0.9492 | 0.9529 |
| | 20 | 0.9524 | 0.9528 | 0.9509 | 0.9512 |
| `crossed_pigeonhole` (v2) | 5 | — | 0.9687 | **0.9103** | — |
| | 20 | — | **0.9843** | 0.9677 | — |
| `crossed_pigeonhole_t` | 5 | — | **0.9157** | 0.9260 | — |
| | 20 | — | 0.9463 | 0.9430 | — |
| `seed_cluster` | 20 | **0.9346** | 0.9442 | 0.9499 | 0.9470 |

Reading:

* **`crossed_pigeonhole` (the v2 incumbent) is unstable in both directions.** At
  m=5 with elevated seed variance it undercovers badly (0.9103, null 0.0453);
  at m=20 with low seed variance it grossly overcovers (0.9843, null 0.0067).
  Its error rate depends on a nuisance parameter that is unknown before launch,
  so its behaviour on the real study could not be predicted. That is a stronger
  objection than "slightly anticonservative".
* **`crossed_pigeonhole_t` does not fix it.** Studentizing helps at large m but is
  clearly anticonservative at small m (coverage 0.9157, null 0.0467 at m=5).
* **`seed_cluster` undercovers exactly where theory says it must** — at low seed
  variance (0.9346 at m=20, sigma_a=0.01), where the example component dominates
  the variance it ignores by construction.
* **`crossed_cluster_t` is essentially exact everywhere**: null 0.0229–0.0260
  against nominal 0.025, coverage 0.9492–0.9535 against nominal 0.95, across
  every seed count and every variance scenario tested.

## Registered decision

**Primary estimator: `crossed_cluster_t`.**

Two-way cluster-robust variance in the sense of Cameron, Gelbach & Miller (2011)
with a `t_{m-1}` reference distribution. It satisfies the replacement criteria:

1. **Preserves example pairing.** It operates on the paired per-example panel and
   refuses input whose per-stratum example sets differ across seeds.
2. **Correctly treats seed replication.** The seed component is a cluster total
   over training seeds and carries `m-1` degrees of freedom, which the `t_{m-1}`
   reference accounts for.
3. **Good null calibration in simulation.** Demonstrated above at 20,000 trials
   across four seed counts and four variance scenarios.
4. **No confirmatory outcome used.** Every number here is simulated.
5. **Documented mathematical justification.** Expanding
   `V = c_s V_seed + c_e V_example - V_cell` term by term recovers
   `sigma_a^2/m + sigma_b^2/(Kn) + sigma_eps^2/(mKn)` up to an `O(1/(mKn))`
   correction, so the estimator is essentially unbiased for the true sampling
   variance. This is verified numerically by
   `test_two_way_variance_matches_the_true_sampling_variance`.
6. **Comprehensive regression tests.** `tests/test_phase1_inference.py`.
7. **Registered before execution.** `PHASE1_V3_PREREGISTRATION.md`.

It is also the *claim-preserving* choice: it uses both seed and example variation,
so the claim generalises over training seeds **and** over examples drawn like the
registered generator's. `seed_cluster` was not adopted as primary — it would have
narrowed the claim to this materialized test split, and adopting it merely
because it is simpler or easier to satisfy was explicitly rejected.

## Sensitivity analyses

All four estimators are computed and reported on every registered contrast.
Only `crossed_cluster_t` enters a gate. If the conclusion flips between
estimators, that fact is visible in the analysis artifact rather than buried.

## Reproduction

```bash
python scripts/phase1_inference_calibration.py \
  --output inference_calibration.json \
  --trials 20000 --bootstrap-trials 3000 --bootstrap-samples 999 \
  --seed-counts "5,10,16,20" --seed-sds "0.01,0.02,0.04,0.06"
```

Bootstrap estimators run on a reduced sub-grid because each cell costs three
orders of magnitude more than an analytic cell. The skipped cells are listed
explicitly in the report artifact under `skipped_conditions` — never silently
dropped.
