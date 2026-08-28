# Pre-registration — can a 0.6B family uniformly beat a 4B model?

**Registered:** 2026-08-27, before any inference was run on the TEST split.
**Status:** OPEN. No TEST number exists yet.

This document fixes the claim, the analysis, and the failure conditions in
advance. It exists because the result it targets would be easy to manufacture by
accident, and impossible to distinguish afterwards from the real thing.

---

## 1. The hypothesis under test

The informal claim in circulation is that models face an **invisible limit**: a
fixed capacity cannot be improved on one skill except by trading against another,
so a smaller model can never comprehensively match a larger one.

We already observe this trade inside our own results. The S1 abstention gate
gained six unanswerable rows on showcase64 and lost `RLC-060-C`, a
counterfactual. Pushing abstention harder trades counterfactual accuracy for
unanswerable accuracy along precisely the predicted axis.

**What we actually test.** The tradeoff hypothesis is a claim about *a fixed
parameter vector optimised under a single objective*. It is not a claim about
systems. We therefore test the strongest honest version:

> **H₁ (the claim we try to establish).** A Rineya system built from 0.6B-class
> components can beat Qwen3-4B on **all four** variant families simultaneously,
> on data never used for any selection, at **equal or lower inference compute**.
>
> **H₀ (the null we try to reject).** No such configuration exists; any Rineya
> gain on one variant family is offset by a loss on another.

Failing to establish H₁ does **not** prove the invisible limit exists. It leaves
it standing. This asymmetry is deliberate and is not to be reported otherwise.

---

## 2. Why a new evaluation substrate was required

All 256 rows of `Rineya_Long_Context_Demo_v1` are spent:

- **showcase64** was selection data for the published family;
- **stability192** was inspected when the S4 abstention threshold was chosen;
- every row is stamped `intended_use: "demo/evaluation only; do not train on this exact suite"`.

Any uniform win obtained by tuning against those rows would be selection, not
capability, and would look identical to the real result. The suite ships no
generator, so one was written: `scripts/generate_suite.py`.

### Splits

| Split | Scenarios | Rows | Seed | Purpose |
|---|---:|---:|---:|---|
| **DEV** | 64 | 256 | 20260827 | all tuning, probe fitting, threshold selection |
| **TEST** | 128 | 512 | 20260828 | **SEALED.** Single use, Stage 8 only |
| **TRAIN** | 256 | 1024 | 20260829 | Stage 5/6 training data |

Zero context-hash overlap between any pair of splits (verified: 192 / 384 / 768
unique contexts, all intersections empty). `scripts/validate_suite.py` reports
0 failures and 0 warnings on all three.

### Fidelity to the original design

Each scenario carries one *decisive binding* expressed three ways, mirroring the
original suite:

- **base** — the binding is stated;
- **counterfactual** — the binding is **swapped**; every entity remains present,
  so surface statistics barely move and only the relation flips;
- **unanswerable** — the binding is **destroyed**; every entity remains present,
  so the model must detect a missing *link*, not a missing name.

`paraphrase` reuses the base context with a reworded question. The validator
enforces all of these as hard failures, including that both candidate answers
appear in both answerable contexts and survive into the unanswerable context.

---

## 3. Sealing protocol for TEST

1. TEST is generated once and its file hashes recorded below.
2. **No TEST row is read, scored, or inspected** — by any person or script —
   until Stage 8 is entered.
3. Only aggregate structural statistics (counts, word-length distributions) have
   been produced from TEST, by `validate_suite.py`. No content, no predictions.
4. TEST is scored **once**. If a bug forces a rerun, that fact is recorded here
   and the result is reported as a second look, not a first.
5. Any configuration choice made after seeing a TEST number invalidates the
   claim for that configuration.

**TEST file hashes (sha256, first 16):**
`test_full.jsonl` `951ac28059521d3b` · `test_prompts.jsonl` `b17c3dd4bc156172` ·
`test_answer_key.jsonl` `c178b0c8cb72b48f`

---

## 4. The compute-parity rule

Parameter count is the wrong axis: if Rineya spends nine forward passes to beat
one Qwen3-4B pass, nothing about capacity has been shown.

> **Budget.** 4B ÷ 0.6B ≈ **6.7×**. A Rineya configuration may spend at most
> **6 forward passes** of a 0.6B-class model per item and remain compute-eligible.

System48's nine passes are **over budget** and cannot be used for the primary
claim. It may be reported as a reference point only.

Probe heads, logit-level rules, and calibration layers cost no additional forward
pass and are unrestricted. A configuration exceeding the budget is reported
separately and never as evidence for H₁.

---

## 5. Primary endpoint and decision rule

**Primary endpoint.** Per-variant strict-exact correct counts on TEST (128 rows
per variant family), Rineya configuration versus Qwen3-4B, same protocol.

**H₁ is established only if all four hold:**

| # | Condition |
|---|---|
| 1 | Rineya ≥ Qwen3-4B on **each** of base, paraphrase, counterfactual, unanswerable |
| 2 | Rineya > Qwen3-4B on **overall** TEST-512, paired exact McNemar **p < 0.05** |
| 3 | The configuration is **compute-eligible** (≤ 6 passes) |
| 4 | The configuration was frozen **before** TEST was scored |

Condition 1 is deliberately strict — ties are permitted, losses are not. That is
what "uniformly beat" means and it is the whole point of the exercise.

**Secondary endpoints**, reported regardless: overall strict-exact; per-variant
McNemar with Holm correction across the four families; abstention precision and
recall; measured passes per item.

**Pre-declared analysis choices**, fixed now to prevent later drift:

- correctness = `strict_exact` as computed by the suite's own
  `evaluate_predictions.py`; no alternative scorer is admissible;
- comparisons are **paired** on row id, exact McNemar, two-sided;
- α = 0.05; the four per-variant tests are Holm-corrected; the overall test is not;
- no row is excluded for any reason; coverage must be 512/512 or the run is void;
- protocol is frozen: BF16, greedy, seed 17, thinking disabled, 24 new tokens,
  4096 context, truncation forbidden — identical for both models.

---

## 6. Declared failure modes

These are the outcomes we expect to have to report, and we commit now to
reporting them plainly:

- **Most likely failure: counterfactual.** Supa-A1 trails Qwen3-4B by 7 on the
  old suite's counterfactual variant, and abstention work makes it worse, not
  better. If the program lands as "wins 3 of 4", it is reported as *H₁ not
  established*, not as a partial victory.
- **Compute-budget failure.** If only a >6-pass configuration wins uniformly,
  H₁ is not established. It is reported as an unbudgeted reference result.
- **Regression to base.** Supa-A1's +9 on base is the family's real asset. Any
  configuration that trades it away to buy abstention has demonstrated the
  invisible limit rather than refuted it, and will be reported that way.
- **Generator artifact.** If Qwen3-4B and Rineya both score far above or below
  their old-suite levels on DEV, the generator is suspected of altering
  difficulty. TEST is then void as a fairness instrument and must be rebuilt.

---

## 7. Stage plan and gates

| Stage | Work | Gate |
|---|---|---|
| **0** | Fresh substrate + this document | ✅ complete |
| 1 | Ship S1 readout as a model class | must not regress base on DEV |
| 2 | Multi-signal calibrated gate | AUC must exceed the best single signal |
| 3 | Hidden-state probe | **decisive**: beats AUC 0.79 ⇒ readout-limited; plateaus ⇒ representation-limited |
| 4 | Attention-mass features | must add signal independent of Stage 2 |
| 5 | Targeted training on TRAIN absence pairs | only if Stage 3 says representation-limited |
| 6 | Counterfactual workstream | the deficit no abstention work can fix |
| 7 | Compute-matched routing (≤6 passes) | oracle union on DEV must exceed best single arm |
| 8 | **Single sealed TEST evaluation** | one shot, configuration already frozen |

Every stage is measured on **DEV**. TEST is touched exactly once, at Stage 8.

---

## 8. What a null result means

If H₁ is not established, the honest conclusion is:

> On this benchmark, at this scale, under a 6-pass compute budget, we could not
> construct a 0.6B-class configuration that uniformly matches Qwen3-4B.

That is a bounded empirical statement about one benchmark and one budget. It is
**not** evidence that the invisible limit is a law, and it will not be written up
as such. Equally, if H₁ *is* established, the claim is bounded the same way: one
suite, one comparison model, one protocol.
