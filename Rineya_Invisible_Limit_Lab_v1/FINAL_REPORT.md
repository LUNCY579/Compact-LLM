# Can a 0.6B model uniformly beat a 4B model? — Stages 0–8 report

**Status:** Stages 0–8 complete. Stage 6 (contrastive training) in progress; its
results are **not** in this report.
**Date:** 2026-08-28
**Pre-registration:** [`PREREGISTRATION.md`](PREREGISTRATION.md), filed before any
TEST inference existed, with four amendments.

---

## 1. The headline, stated precisely

> **H₁ was not established.** A 0.6B-class configuration did **not** uniformly beat
> Qwen3-4B. It reached a **statistical tie overall** — 721 vs 720 of 1024 —
> while winning one variant family decisively and losing two.

The temptation is to report this as "0.6B matches 4B." That is defensible for the
*aggregate* and misleading about the *composition*, so both are given below and the
composition is what matters.

### Sealed TEST, 1024 rows, scored once

| Split | Rineya-frozen | Qwen3-4B | delta | 95% CI | p | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| **OVERALL** | **721** | **720** | **+1** | [−51, +53] | 1.0000 | — |
| base | 151 | 157 | −6 | [−25, +14] | 0.6101 | 0.6101 |
| paraphrase | 171 | 193 | **−22** | [−39, −5] | 0.0183 | 0.0549 |
| counterfactual | 144 | 165 | **−21** | [−41, −2] | 0.0439 | 0.0878 |
| unanswerable | **255** | 205 | **+50** | [+38, +63] | **<0.0001** | **<0.0001** |

Uniformity statistic `T = min_v(delta_v) = −22`, permutation **p = 0.9728**.
Verdict, both as pre-registered and error-controlled: **H₁ NOT ESTABLISHED**.

### What this is evidence for

- **Compute-matched parity is real.** A 0.6B core at **one forward pass** finished
  one row ahead of a 4B model at one forward pass, on data neither had seen, with
  identical decoding protocol on the same device. That is a genuine and useful
  engineering result.
- **The skill trade is real, and measured.** The tie is +50 on abstention against
  −22 and −21 on reasoning. Both losing intervals **exclude zero**. This is the
  "invisible limit" hypothesis being *displayed*, not refuted.

### What this is not evidence for

- Not "0.6B beats 4B." The overall delta is +1 with CI [−51, +53]. It is a tie.
- Not a capacity claim. Rineya's core carries task-specific long-context SFT
  (`step200`); Qwen3-4B was zero-shot throughout. See §7.
- Not a refutation of the invisible limit. Per pre-registration §8, failing to
  establish H₁ **leaves the hypothesis standing**; it does not prove it is a law.

---

## 2. Why the null is trustworthy rather than lazy

A null result is only informative if the design could have detected a real effect.

**Minimum detectable effect**, from observed discordance scaled to TEST size:

| Split | n | MDE (rows) | MDE (pp) |
|---|---:|---:|---:|
| OVERALL | 1024 | 40 | 3.9% |
| base | 256 | 22 | 8.6% |
| paraphrase | 256 | 22 | 8.6% |
| counterfactual | 256 | 20 | 7.8% |
| unanswerable | 256 | 22 | 8.6% |

The −22 and −21 losses sit *at* the detection threshold; the +50 win is far past
it. So "not established" means **a uniform win does not exist at this scale**, not
that we could not see one.

**False-positive rate of the decision rule**, by simulation under an explicit null
(equal systems, each discordant pair a fair coin, 200 000 draws):

| Rule | FPR |
|---|---:|
| §5 condition 1 alone (`delta >= 0` on all four) | **8.51%** |
| §5 condition 2 alone (overall, p<0.05) | 2.34% |
| **Both — the actual verdict rule** | **1.67%** |

Condition 1 alone would have been badly inflated. This was found by self-review
*before* the verdict and fixed by adding the uniformity permutation test
(Amendment 3a). The rule as used is conservative.

---

## 3. Stage-by-stage findings

### Stage 0 — the evidence base had to be rebuilt first

All 256 rows of `Rineya_Long_Context_Demo_v1` were spent: showcase64 was selection
data for the published family, stability192 was inspected when the S4 abstention
threshold was chosen, and every row is stamped
`intended_use: "demo/evaluation only; do not train on this exact suite"`. No
generator ships with it, so one was written.

Four splits, no shared context hash between any pair:

| Split | Scenarios | Rows | Seed | Use |
|---|---:|---:|---:|---|
| DEV | 64 | 256 | 20260827 | all tuning, probe fitting, thresholds |
| TEST | 256 | 1024 | 20260828 | sealed; scored once at Stage 8 |
| TEST2 | 256 | 1024 | 20260830 | sealed; reserved for Stage 6 |
| TRAIN | 256 | 1024 | 20260829 | Stage 6 training data |

**The substrate took four iterations, and every defect was caught by an automated
check before any TEST row was scored.** This is the least glamorous and most
important part of the work:

1. **v1** — six of eight `unanswerable` builders left a defensible answer in the
   context. `alias_resolution` asked "Who carried the object?" while the context
   still said "Harrow carried the object"; both models answered `Harrow` on every
   such row. The variant was not testing abstention at all.
2. **v2** — unresolvability vocabulary appeared only in unanswerable contexts, so
   `"no surviving record"` sat in 88% of them and 0% of answerable ones. A
   bag-of-words classifier separated the classes at AUC 0.947 and a hidden-state
   probe hit **AUC 1.000** — measuring template detection, not abstention.
3. **v3** — the distractor fix was incomplete: casing mismatch meant the shared
   phrase family still skewed.
4. **v4** — `object_state_tracking`'s counterfactual silently dropped its own base
   answer from the context. It passed DEV only because filler mentions places
   incidentally, and surfaced on the larger split.

**Why v1's withdrawal mattered.** v1 DEV showed Supa-A1 *ahead* 142–137. That
number is void. The broken unanswerable builders were suppressing Qwen3-4B's single
largest advantage: old suite → v1 DEV, Qwen's unanswerable fell 30→14 while
Supa-A1's fell only 21→15. A suite that selectively flattens the larger model's
best axis manufactures exactly the result this programme set out to test.

**Calibration finding.** The *original* suite is bag-of-words separable at
**AUC 1.000**; our final substrate is **0.911** — measurably *less* separable than
the reference. Lexical cueing is inherent to this benchmark family, not something
we introduced. The consequence for interpretation is Amendment 2 and §6 below.

### Stages 1–3 — the abstention deficit is *readout*-limited, decisively

All gates evaluated on DEV with **GroupKFold on `scenario_id`** (the four variants
of a scenario share a context; an ungrouped split would leak near-identical rows
across the fold boundary) and thresholds swept inside training folds only.

| Configuration | AUC | correct | gained | lost | abstention precision |
|---|---:|---:|---:|---:|---:|
| ungated baseline | — | 127/256 | — | — | — |
| S1 logit scalar (shipped form) | 0.837 | 142 | 30 | 15 | 0.564 |
| S4 logit scalar | 0.839 | 142 | 31 | 16 | 0.562 |
| S2 multi-signal, logit features | 0.843 | 143 | 32 | 16 | 0.554 |
| **P-linear, hidden state** | **0.999** | **175** | 50 | **2** | **0.865** |
| P-MLP, hidden state | 0.974 | 168 | 47 | 6 | 0.762 |
| P + S combined | 0.999 | 175 | 50 | 2 | 0.877 |

**0.837 → 0.999 on identical rows.** The published gate reads a scalar off the
logit layer — a 151 936-way projection of a 1024-dim state — and throws away
almost all of the signal the hidden state carries. Practical effect: +48 correct,
and the probe is far more *precise*, costing 2 answerable rows where the logit
gates cost 15–16.

Per the pre-registered criterion this settled **Stage 5 as a no-go**: the
representation is not the limit, so training for abstention was not warranted.

**A hypothesis of mine that did not survive.** I argued the logit-layer signals
were decorrelated and would combine well. They do not — multi-signal 0.843 versus
single-signal 0.839. The logit features are largely redundant.

**Where the invisible limit shows up first.** Every logit gate *pays* for
abstention: counterfactual 32→28, paraphrase 43→35. The probe is the only
configuration that mostly escapes the trade — 50 gained against 2 lost — because
its precision is high enough that abstention stops cannibalising answerable rows.

### Stage 4 — skipped

Attention-mass features were never run. The probe saturates the unanswerable
column, leaving nothing for them to add. Recorded as skipped, not executed.

### Stage 7 — routing buys nothing

Answer-blind routing among Rineya arms only. Qwen3-4B is excluded as an arm: at
~6.7× the cost it breaks the 6-pass budget, and putting the comparison model inside
the candidate would be circular.

| Policy | Passes | Correct |
|---|---:|---:|
| **supa_a1 + probe gate** | **1** | **170/256** |
| + unknown-gate(nimbus16) | 2 | 170 |
| + unknown-gate(pulse16) | 2 | 170 |
| + consensus(baya17, halo_unc15) | 3 | 170 |
| + consensus(nimbus16, pulse16) | 3 | 165 |
| supa_a1 alone | 1 | 141 |
| *ORACLE union over five arms* | *5* | *223* |

Every eligible multi-pass policy **ties or loses** to the single probe-gated core.
The arms share the Nimbus base and `step200` adapter, or the Halo donor lineage, so
they agree almost everywhere and consensus adds no information. The oracle union of
223 says 48–53 rows of genuine complementarity exist, but **no answer-blind policy
reaches it**.

**This undercuts the published cascade strategy.** System48's nine arms buy little
that one probe-gated core does not — and System48 is over the compute budget
anyway, so it was ineligible for the primary claim.

Stage 7 also disqualified the newly-published checkpoint: probe on the **ungated**
Supa 1.7 scores 175/256, the same probe on **Supa-A1** (which carries
`abstain_logit_bias 5.15625` in config) scores 170. The fixed bias makes false
abstentions the probe would not. The frozen configuration therefore uses the
ungated core.

### Stage 8 — the sealed verdict

Frozen and committed (`f318a04`) before TEST was scored:

> **`Rineya-Supa-1.7.0-devfreeze` + linear hidden-state abstention probe**
> 1 forward pass · 3072-dim features (layers −1/−8/−15) · threshold 0.142 ·
> compute-eligible against the 6-pass budget

Probe fit on all 256 DEV rows, threshold by cross-validation within DEV, applied
once to TEST. Result in §1.

**The probe generalised.** An answer-blind check before scoring: abstention rate
27.8% on 1024 unseen rows (285/1024) against a true unanswerable base rate of 25%.
Slightly trigger-happy by ~2.8 points, nowhere near collapse. Had it memorised DEV
the rate would have been wildly off-base. It converted to +50 rows on TEST.

**DEV predicted the direction and understated the magnitude** — counterfactual −8
on DEV became −21 on TEST — exactly as expected from having selected the
configuration on DEV. This is why the sealed split was worth three hours of CPU.

---

## 4. Method notes that materially affect the numbers

- **Correctness** is delegated to the suite's own `evaluate_predictions.py`, never
  reimplemented, so this lab and the published family cannot silently diverge on
  what counts as correct.
- **Protocol frozen and identical** for both models: BF16, greedy, seed 17,
  thinking disabled, 24 new tokens, 4096 context, truncation forbidden.
- **Device parity.** Both models ran **pure CPU** on TEST. Qwen3-4B originally
  needed GPU splitting to fit in memory, which would have put the two models on
  different numerical substrates; that was eliminated by construction rather than
  measured, at a cost of ~78 minutes (Amendment 3c). A free GPU was offered and
  declined for the sealed comparison for this reason.
- **Confidence intervals** bootstrap over **scenarios**, not rows, because the four
  variants of a scenario share a context and row-level resampling would treat
  correlated observations as independent.
- **Threshold selection detail that quietly decides results.** Per-fold probe
  thresholds were `[0.142, 0.148, 0.073, 0.134, 0.775]`. Four cluster tightly, one
  fold is an outlier. The frozen config takes the **median (0.142)**; the mean
  would have been dragged to 0.234. Fixed before TEST, not after.

---

## 5. Honest limitations

1. **Single benchmark.** Every conclusion is bounded to this suite and protocol.
2. **Single comparison model.** Only Qwen3-4B. A claim about "larger capacity"
   ideally rests on more than one comparison point.
3. **Qwen3-4B ran with `--disable-thinking`.** Required for protocol parity, but
   Qwen3 is a hybrid reasoning model and this is not Qwen3-4B at its best. The
   comparison is "under Rineya's frozen protocol," which is the right comparison for
   this use case and not a general capability ranking.
4. **Greedy decoding, single seed.** Deterministic, so there is no run-to-run
   variance to report — but equally no estimate of model-level variance. The only
   uncertainty quantified is benchmark sampling.
5. **The probe fits 3072 features on 256 rows.** Regularised at C=0.05 and validated
   on 1024 unseen rows, but the ratio is uncomfortable.
6. **Amendment 2 caveat, restated because it is the most important one.** Abstention
   on this benchmark family is *partly lexically cued* — the reference suite is
   BoW-separable at AUC 1.000. The +50 unanswerable win is therefore in part
   benchmark-specific engineering, not demonstrated capability. A uniform win driven
   mostly by that column would have been a weaker result than the same margin spread
   across base, paraphrase and counterfactual.

---

## 6. Disclosure of process

**Substrate v1's DEV comparison was inspected (Supa-A1 ahead 142–137) before the
decision to rebuild.** Every rebuild was justified by a model-agnostic defect and
each was caught by an automated validator check rather than by inspecting who won.
But the ordering means blindness cannot be claimed for the decision to iterate.
Every substrate change is committed with its defect evidence so the reasoning can be
audited independently.

**A crash occurred mid-pipeline** (segfault on Qwen3-4B model load, out of memory).
It was audited: no partial output existed, `set -e` stopped the chain, and every
surviving file was verified against the current substrate by manifest hash and by
re-tokenising prompts to compare recorded `input_tokens`. Zero contamination, zero
recomputation lost.

**Stages 4, 5 and 6 were skipped, not run** at the time of the Stage 8 verdict —
Stage 4 because the probe saturates its target column, Stage 5 on its
pre-registered condition, Stage 6 because it was reordered to follow the verdict.

---

## 7. What would actually attack the wall

The entire remaining deficit is **paraphrase −22 and counterfactual −21**, and no
abstention engineering touches either. Both are reasoning failures.

**Stage 6**, pre-registered in Amendment 4 and now in progress, trains a contrastive
LoRA on the untouched 1024-row TRAIN split — answerable variants only, 768 rows,
loss masked to answer tokens. The contrastive pressure is structural: each scenario
contributes a `base` and a `counterfactual` with near-identical contexts and
*different* answers, so a model that pattern-matches on surface familiarity cannot
satisfy both examples at once.

**The fairness consequence is recorded in advance.** Rineya would receive 768
in-distribution examples; Qwen3-4B receives none. A Stage 6 win therefore licenses:

> a *task-specialised* 0.6B system beats a *general* 4B model on this suite at equal
> compute

and **not** "a 0.6B model matches a 4B model's capacity." If Stage 6a closes the
gap, **Stage 6b gives Qwen3-4B the identical training** before any capacity claim is
made. Stage 6 is evaluated on **TEST2**, a separate sealed split, because reusing
TEST would make it a second look at data already used for a verdict.

---

## 8. Reproduction

```bash
# substrate (deterministic; same seed reproduces byte-identical output)
python scripts/generate_suite.py --scenarios 64  --seed 20260827 --split dev   --prefix RILDEV   --outdir data
python scripts/generate_suite.py --scenarios 256 --seed 20260828 --split test  --prefix RILTEST  --outdir data
python scripts/validate_suite.py --full data/dev_full.jsonl --key data/dev_answer_key.jsonl

# signals, gates, routing
python scripts/capture_signals.py --model <core> --prompts data/dev_prompts.jsonl --output <out>
python scripts/fit_gates.py   --signals <out> --key data/dev_answer_key.jsonl
python scripts/route_stage7.py --key data/dev_answer_key.jsonl --arms <dir> --signals <out>

# frozen config and verdict
python scripts/apply_frozen_config.py --dev-signals <dev> --dev-key data/dev_answer_key.jsonl \
    --target-signals <test> --output <preds>
python scripts/compare_models.py --key data/test_answer_key.jsonl --a <preds> --b <qwen> --passes-a 1
python scripts/power_analysis.py --dev-report <dev_h2h.json> --test-scale 4.0
```

Evidence lives under `artifacts/rineya_invisible_limit_v1/` (gitignored, as with
every artifact tree in this repository). Scripts, answer keys and this report are
committed.

---

## 9. One-paragraph summary

A 0.6B-class Rineya configuration — one dense core plus a linear probe on its own
hidden state, at a single forward pass — reached a statistical dead heat with
Qwen3-4B on 1024 sealed rows of a purpose-built long-context suite: 721 to 720.
The pre-registered hypothesis that it could beat the larger model on *every* skill
family was **not established**: it won abstention by 50 rows and lost paraphrase by
22 and counterfactual by 21, both with intervals excluding zero. The design was
powerful enough to detect those losses, and the decision rule's false-positive rate
was 1.67%. Along the way the published family's own abstention gate was shown to be
reading its signal at AUC 0.837 when 0.999 was available in the hidden state, and
the nine-arm System48 cascade was shown to buy nothing that one probe-gated core
does not. The invisible-limit hypothesis survives this test — visibly, as a
measured trade between abstention and reasoning — and the remaining deficit is now
narrow, specific, and the target of Stage 6.
