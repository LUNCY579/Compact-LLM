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

**TEST file hashes (sha256, first 16) — final substrate, current:**
`test_full.jsonl` `8bb8420cb293` · `test_prompts.jsonl` `d9b8b8456384` ·
`test_answer_key.jsonl` `dda248c62b6e`

### Amendment 1 — substrate v1 withdrawn, 2026-08-28

Substrate v1 (TEST hashes `951ac280…`, `b17c3dd4…`, `c178b0c8…`) was **withdrawn
before TEST was ever scored**. The §6 generator-artifact gate fired on DEV.

*What was wrong.* Six of the eight unanswerable builders left a defensible answer
in the context, so the variant did not test abstention at all. The clearest case:
`alias_resolution` asked "Who personally carried the {object}?" while the context
still stated "{call sign} carried the {object}", so `Harrow` was a legitimate
answer even with the call-sign-to-person binding destroyed. Both models answered
the call sign on every such row. Measured UNKNOWN rates on v1 DEV unanswerable
rows, out of 8 per builder per model:

| builder | Qwen3-4B | Supa-A1 |
|---|---:|---:|
| `alias_resolution` | 0 | 0 |
| `count_with_exclusions` | 0 | 0 |
| `latest_verified_record` | 0 | 0 |
| `object_state_tracking` | 1 | 0 |
| `temporal_correction` | 0 | 1 |
| `rule_exception` | 0 | 6 |
| `causal_disambiguation` | 6 | 1 |
| `multi_condition_identity` | 8 | 7 |

*Why it mattered for fairness, not merely difficulty.* The defect pushed
unanswerable to the floor for both models, and unanswerable is precisely where
Qwen3-4B held its largest advantage on the original suite. Old-suite → v1 DEV,
Qwen3-4B's unanswerable fell 30 → 14 while Supa-A1's fell only 21 → 15. Overall,
Qwen3-4B lost 41 points and Supa-A1 only 26. A suite that selectively neutralises
the larger model's best axis would manufacture the result this programme is
trying to test. v1 DEV showed Supa-A1 ahead 142 to 137; **that number is
withdrawn and is not evidence for anything.**

*The fix.* The one builder that worked, `multi_condition_identity` (15/16), was
the only one carrying an explicit "no participant can be shown to satisfy …"
statement matching exactly what the question asks. That pattern is now applied to
all eight builders, and `alias_resolution`'s question was changed to demand a
*named participant* rather than "who". All three splits revalidate at 0 failures,
0 warnings.

*Standing.* TEST v1 was never read, scored, or inspected — only aggregate
structural statistics were produced from it. The seal is therefore intact and v2
TEST remains a genuine first look. DEV was rebuilt on the same seed and rescored
from scratch; every v1 DEV number is void.

---

### Amendment 2 — abstention on this benchmark family is partly lexically cued

Recorded 2026-08-28, before TEST was scored. This one bounds what H₁ can mean and
is the most important caveat in this document.

Substrate v2's hidden-state probe returned **AUC 1.000** for detecting
unanswerable rows. That looked like a decisive Stage 3 result. It is not, and
chasing it down produced a finding about the benchmark itself.

A trivial bag-of-words classifier over raw context text separates answerable from
unanswerable at:

| suite | BoW AUC |
|---|---:|
| **original `Rineya_Long_Context_Demo_v1`** | **1.000** |
| this lab, substrate v2 | 0.947 |
| this lab, final substrate (DEV) | **0.911** |

So the original suite is *perfectly* separable from surface text alone, and our
substrate is measurably **less** separable than the reference. Lexical cueing is
inherent to the design: destroying a binding necessarily changes the text, and any
systematic change is detectable. The absolute 0.85 target we first wrote was
neither achievable nor meaningful, and the validator now requires only that we are
no more separable than the reference suite.

**What this means for the claim.** On this family, "detect that the answer is not
determined" is substantially solvable by recognising unanswerable-style phrasing
rather than by reasoning about whether the queried binding survives. The
interesting gap is that the base models barely exploit it — ungated abstention
sits at 9–15 of 64 while the signal is near-perfectly decodable from the hidden
state. That is strong evidence the deficit is **readout-limited**, and it is not
an artifact of our generator, since the reference suite carries the property more
strongly.

But it also means:

> A probe-driven win on the unanswerable variant is a **benchmark-specific
> engineering result**. It is not, on its own, evidence that a 0.6B model has
> acquired an abstention capability that a 4B model lacks.

Any Stage 8 result will therefore report the unanswerable component with this
caveat attached, and the headline claim will be stated as bounded to this suite.
A uniform win driven mostly by the unanswerable column is a weaker result than the
same margin spread across base, paraphrase and counterfactual, and will be
described that way rather than aggregated into a single triumphant number.

### Amendment 3 — statistical and protocol hardening, 2026-08-28

Recorded before TEST was scored. Four weaknesses were found by self-review of the
design, three of them fixed by changing the experiment rather than by adding
caveats.

**3a. Condition 1 had no error control (fixed).** §5 condition 1 requires
`delta >= 0` on all four variants but attaches no significance test. Under the
null each variant is close to a coin flip, so the conjunction alone fires far
more often than 5% and must not be read as a significance claim. A properly
controlled uniformity test is now computed alongside it:

> `T = min over variants of delta_v`, with an exact permutation null obtained by
> flipping discordant pairs within each variant. H₁ requires `T >= 0`; the
> permutation p supplies the error control condition 1 lacks.

`compare_models.py` now reports **two verdicts**: the one as pre-registered, and
an error-controlled one that additionally requires the uniformity p < 0.05. If
the two disagree, the error-controlled verdict governs and the discrepancy is
reported.

**3b. TEST was underpowered (fixed by enlarging it).** At 512 rows the minimum
detectable effect was roughly 9 percentage points per variant, so a null verdict
on a small per-variant gap would have been uninformative. TEST was regenerated at
**256 scenarios / 1024 rows**, halving the per-variant standard error. New hashes:
`test_full.jsonl` `b8f42adf501ed695` · `test_prompts.jsonl` `f9f143989f6dd767` ·
`test_answer_key.jsonl` `6b26dae5ba8a26be`. Validates at 0 failures with zero
context-hash overlap against DEV or TRAIN. TEST had still never been scored when
this change was made, so the seal is intact.

**3c. Device asymmetry (fixed by construction).** Qwen3-4B originally needed
`--max-gpu-memory` splitting to fit, while the Rineya core ran pure CPU. BF16
reduction order differs between devices, which adds noise to greedy decoding.
**Both models now run pure CPU on TEST** (`--device-map cpu`), so the confound is
eliminated rather than measured. The DEV Qwen3-4B run used GPU splitting; DEV does
not inform the frozen Rineya configuration, which is Supa 1.7 + a probe fit on
Supa's own CPU signals, so that asymmetry cannot bias the claim.

**3d. No interval estimates (fixed).** Every delta now carries a 95% percentile
CI bootstrapped over **scenarios**, not rows. The four variants of a scenario
share a context, so row-level resampling would treat correlated observations as
independent and give intervals that are too narrow.

### Disclosure — what was seen before the substrate was rebuilt

Substrate v1's DEV comparison was inspected (Supa-A1 ahead 142–137) *before* the
decision to rebuild. Every rebuild was justified by a model-agnostic defect —
unanswerable variants leaking defensible answers, then lexical shortcuts, then a
counterfactual that dropped its own base answer — and each was caught by an
automated validator check rather than by inspecting who won. But the ordering
means blindness cannot be claimed for the decision to iterate. Every substrate
change is committed with its defect evidence so the reasoning can be audited
independently.

### Stages not run

**Stage 4 (attention-mass features) was skipped**, not executed: the hidden-state
probe already saturates the unanswerable column, leaving nothing for it to add.
**Stage 5 was skipped** on its pre-registered condition — Stage 3 showed
readout-limitation, not representation-limitation. **Stage 6 was skipped** because
the counterfactual deficit that motivated it did not reproduce on the fresh
substrate.

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
