# Phase-1 v4 — competence result: the regime family is REJECTED

**Status: `V4_REGIME_FAMILY_REJECTED`.** Both registered competence measurements
returned `NO_GO`, and the registered escalation ladder returns
`NO FURTHER MEASUREMENT PERMITTED`. No confirmatory matrix was dispatched.

This is a result about the **experimental setup**. It says nothing whatever about
the TGSA mechanism, in either direction.

---

## 1. What was measured

Two regimes on the certified `easy_nontrivial` band, three arms × three seeds
each, corrected harness (holdout generated before train), identical seeds,
identical total compute — 20,000 steps × batch 64 = **1,280,000 sample
presentations in both**. The regimes differ only in how many distinct examples
that compute is spread over.

| regime | distinct examples | epochs | verdict | failed checks |
|---|---:|---:|---|---|
| `easy_50k_20k` | 49,998 | 25.6 | **NO_GO** | reliability, memorization, **saturation** |
| `easy_200k_20k` | 199,998 | 6.4 | **NO_GO** | reliability |

---

## 2. OBSERVED — the finding that decides the phase

`OBSERVED` **Only one of the three task families ever moves.** Splitting every
run by whether it completed the training transition, and averaging per family:

| family | 50k: stalled → transitioned | Δ | 200k: stalled → transitioned | Δ |
|---|---|---:|---|---:|
| `direct_lookup` (guardrail, 1 hop) | 0.1710 → 0.1521 | **−0.019** | 0.1522 → 0.1630 | **+0.011** |
| `pointer_chase` (primary, 2 hops) | 0.1660 → 0.1708 | **+0.005** | 0.1673 → 0.1680 | **+0.001** |
| `state_update` (primary, 2–3 steps) | 0.0370 → 0.9838 | **+0.947** | 0.0878 → 0.9773 | **+0.890** |

`SUPPORTED` **The "phase transition" this programme has been tracking since the
pilot is the model learning `state_update` arithmetic, and nothing else.** The
bimodality in overall accuracy (≈0.12 versus ≈0.44) is entirely this one family
crossing from floor to ceiling. The overall figure is arithmetic:
(0.15 + 0.17 + 0.98)/3 ≈ 0.43.

`OBSERVED` **Both pointer families are inert.** They sit at 0.15–0.18 against a
0.0377 trivial baseline — roughly 4.4× baseline, so not at chance — and they do
not respond to *any* lever tried: not a 4× increase in distinct data, not the
update budget, not whether the model undergoes its transition.

`OBSERVED` **Hop count does not modulate the pointer families.** `direct_lookup`
is one hop and `pointer_chase` is two, and they are indistinguishable:
0.152 vs 0.171 at 50k, 0.163 vs 0.168 at 200k. A model for which composition
depth was the binding difficulty would separate them. This one does not.

`SUPPORTED` **Neither primary family can support the mechanism contrast on this
band.** `state_update` saturates at 0.98 — no room for a decrement, and it is
accumulation arithmetic rather than the iterative pointer-following the
mechanism targets. `pointer_chase` has no dynamic range and, by the hop-count
evidence, is not measuring composition depth for this model at all. The
difficulty it presents is the retrieval format itself.

This is why the band is rejected rather than re-tuned. It is not that the two
primary families are merely mismatched in difficulty; it is that the one whose
structure the hypothesis is about does not discriminate that structure.

---

## 3. OBSERVED — the epoch/diversity trade-off

`OBSERVED` At identical compute, spreading it over 4× the distinct data moves
the transition rate from 89% (8/9) to 33% (3/9), while collapsing the
memorization gap from ~0.16 to ~0.03.

| regime | epochs | transitioned | memorization gap | saturated? |
|---|---:|---:|---:|---|
| 200k × 20k | 6.4 | 33% | 0.007–0.056 | no |
| 50k × 20k | 25.6 | 89% | 0.157–0.167 | **yes** |

`SUPPORTED` The two failure modes trade off along the epoch axis and the band
offers no setting that clears both. Fewer epochs removes memorization and
saturation but leaves most seeds untransitioned; more epochs transitions
reliably but memorizes *and* saturates `state_update`.

`OBSERVED` Of the six stalled runs at 200k, two were still descending at the
final step and four had plateaued. So the budget is genuinely part of the story —
but §2 shows what more budget would buy is a saturated `state_update`, not a
competent `pointer_chase`.

---

## 4. The ladder's answer

```
competence_easy50k_20k.json:  NO_GO  failed: [reliability, memorization, saturation]
competence_easy200k_20k.json: NO_GO  failed: [reliability]

NO FURTHER MEASUREMENT PERMITTED: failed checks with no registered response:
['no_primary_family_saturated']. The ladder only answers a truncated transition
or an epoch-driven memorization gap. REJECT the regime family and revisit the band
```

Identical in both report orderings, which is the property the order-independence
fix was made for.

`no_primary_family_saturated` has no registered response **by design**: a
saturated family is a band problem, and no data or budget change addresses it.
Two of the four permitted regimes were measured; the remaining two
(50k × 30k, 200k × 30k) are not reached, because the ladder rejects on an
unanswerable failure rather than continuing to spend.

---

## 5. UNSUPPORTED — what this does not show

* **Anything about the TGSA mechanism.** No hypothesis in the H1a/H1b/H1c ladder
  was tested. No confirmatory matrix ran.
* That trajectory guidance does or does not help. A regime in which the relevant
  family is inert cannot answer that, in either direction.
* That the model is incapable of multi-hop pointer composition. It was never
  shown to do *one*-hop retrieval either, so composition was never reached.
* That capacity or optimization are not the cause. The pilot's only capacity
  regime (`easy_50k_wide`, 192-dim / 4-layer) was dropped without running, and
  no optimizer setting has been varied. Untested, not excluded.

### The between-arm numbers, recorded and used for nothing

`learned_delta` was the one arm with a stalled seed at 50k (CV 0.542 against
0.032 and 0.013 for the controls), and it had the highest `state_update` figure
at 200k (0.430 vs 0.354, 0.369). Both are between-arm differences on development
data with three seeds. The gate is arm-blind precisely so these cannot become
results, and they have not been used for selection, for the verdict, or for any
claim. They are recorded because omitting them would be the more misleading
choice.

---

## 6. What the gate earned

The competence gate cost two development runs — about eight GPU-hours — and
prevented a confirmatory matrix of roughly 9–26 GPU-hours from running on a band
where the primary contrast family is inert. v3 spent its full matrix to discover
an equivalent problem after the fact.

Three checks fired that a simpler gate would not have had:

* `no_primary_family_saturated` — the check that decides this phase, and the one
  easiest to dismiss as a formality when it was written.
* `competence_is_reliable_across_seeds` — caught a bimodal regime whose *mean*
  cleared every accuracy threshold.
* `generalization_gap_acceptable` — caught the memorization the extra epochs buy.

---

## 7. NEXT PERMITTED EXPERIMENT

The band is revisited under a **new preregistration**, not an extra rung here.
The evidence above constrains its design:

1. **The pointer families need to be learnable at one hop before hop count can
   mean anything.** `direct_lookup` at 0.15 is the blocking observation. Until a
   model reaches competence on one-hop retrieval, `pointer_chase` cannot measure
   composition and no mechanism contrast on it is interpretable.
2. **Depth is the untested lever.** The task is 2-hop composition and the model
   has exactly 2 layers. The pilot deprioritized capacity on the grounds that
   more capacity on 5,000 examples memorizes rather than generalizes — correct
   at 5,000 examples, and no longer binding at 200,000, where the memorization
   gap is 0.03. `easy_50k_wide` (192-dim / 4-layer) was in the pilot grid and
   never ran.
3. **The families should be difficulty-matched.** Any band where one primary
   family saturates while the other is inert cannot support the estimand, which
   macro-averages over strata.
4. **Compute may simply be short.** Reaching 25 epochs on 200,000 examples needs
   ~78,000 steps, roughly 4× what was spent. That is a legitimate hypothesis and
   it is outside the registered ceiling of this preregistration.

Any of these is a change to the regime, so each belongs in a new registration
with its own competence gate, not in this one.
