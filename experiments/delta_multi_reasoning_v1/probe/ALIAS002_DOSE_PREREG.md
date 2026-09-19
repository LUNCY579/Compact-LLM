# ALIAS-002 — dose ladder on the alias probe, registered before any arm is built

**Written and committed BEFORE the first arm is materialised.** ALIAS-001 is a mechanism scout
with no read budget, so nothing here is spent by running it — but the predictions are worth
fixing in advance anyway, because this is the fourth mechanism question on this cell and the
previous three were all answered by looking.

## What this tests

`alias_resolution` is a purely subtractive regression replicated on three banks — **1 repair /
54 damages, p = 3.1e-15** — and salience is refuted as its driver (ALIAS-001: damage present in
all three salience conditions, rising rather than shrinking as the answer is made loudest,
Fisher `low` vs `high` conditioned p = 0.383). The leading account is **capacity**: the peer
session's corpus audit found RRP-015's 384-row training file contains zero role phrases and
zero person names, verified independently here, so this is a skill the fine-tuning never
exercised.

Capacity is a *leading* account, not an established one. Absence in the training data does not
establish causation. This is the test that would move it.

## The ladder

`theta(alpha) = theta_supa + alpha * (theta_trained - theta_supa)`, FP32 throughout, endpoints
copied rather than computed, built by INT-001's own hash-bound `build_arm.py` on its registered
grid **alpha ∈ {0, 0.25, 0.50, 0.75, 1.00}**. Alpha = 0 is the frozen source and its predictions
already exist from ALIAS-001. Four arms are materialised, scored on the **same 360 rendered
prompts**, and deleted.

The rendered prompts are reused unchanged: every arm carries Supa's tokenizer files byte for
byte, so re-rendering would produce identical bytes and a needless second hash to reconcile.

## Registered predictions

| account | alias accuracy across the ladder | dose × salience |
|---|---|---|
| **uniform weakening / capacity** | declines **monotonically** in alpha | **no interaction** — the damage-rate ratio between `low`, `equal` and `high` stays flat |
| **threshold** | flat, then a sharp drop past some alpha | — |
| **salience, already refuted** | — | damage would *shrink* at `high`; ALIAS-001 says it does not |

**Primary quantity: Spearman rank correlation between alpha and alias accuracy, over the five
doses, pooled across conditions.** Registered as **negative and monotone**.

**Secondary, and the one that discriminates capacity from a generic dose effect:** the
per-condition slopes. Capacity predicts three roughly parallel declines. A dose × salience
interaction would mean magnitude and salience trade against each other, which no current account
predicts and which I would report as unexplained rather than absorb.

## What would disconfirm

- **Non-monotone** accuracy across the ladder — the effect is not a dose effect at all, and
  "uniform weakening" is wrong in a way ALIAS-001 could not see.
- **Flat** accuracy with alpha while alpha = 1 still damages — a threshold, not a weakening.
- A **significant dose × salience interaction** — salience returns as a modifier even though it
  is refuted as a driver.

All three are reportable and none is a failure of the probe.

## What a monotone decline would and would not license

**Would.** It makes "the delta uniformly weakens an unexercised binding skill, in proportion to
its magnitude" a *measured* claim rather than a leading account, and it gives the peer session's
preservation anchor a sharp target: the anchor should **flatten this ladder**, not merely raise
its intercept.

**Would not.** It would not establish that the absence of role→person binding in the training
corpus is the *cause*. A dose response is consistent with capacity and with any other mechanism
that scales with magnitude, and this programme has already established that magnitude is the
single operative lever across mask, rank and scalar parameterisations. **Monotone decline is
necessary for the capacity account and is not sufficient for it.**

Nothing here licenses promotion, release, or `quality_improvement_established`. One training
draw of one delta.

---

# OUTCOME — recorded here so the bands and the result are not in different files

**Registered prediction: Spearman(α, alias accuracy) negative and monotone. FIRED.**

| α | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|
| pooled alias | 0.750 | 0.725 | 0.672 | 0.633 | 0.603 |

**ρ = −1.0000, exact permutation p = 0.0167** (the n=5 floor), monotone on every rung and
ρ = −1.000 separately in all three salience conditions. **No dose × salience interaction**
(drops 0.092 / 0.150 / 0.200, Fisher p = 0.137). Receipt: `alias001-20260918/ALIAS_DOSE_RESULT.json`.

**The registered caveat held and is repeated because it is the load-bearing one:** *monotone
decline is NECESSARY for the capacity account and NOT SUFFICIENT for it.* What separated them
was the SHAPE comparison in `analysis/coupling_ladder.py` — marginal cost per acquisition row
spanning **160×** — not the correlation. Downstream: `ALIAS_RESOLUTION_INVESTIGATION.md`
Amendments 2, 3 and 12.
