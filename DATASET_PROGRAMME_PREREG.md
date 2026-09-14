# Dataset programme: pre-registration

Written **before** any of the measurements below were taken. The goal is a training corpus
that fixes what Opallo is measurably bad at without costing what it is measurably good at.
Every threshold here is fixed now, because this programme's record is that claims made
after seeing a score do not survive.

---

## 1. What the survey found

Twelve corpora live in this repository. The survey below is the whole inventory; the three
that matter are marked.

| corpus | rows | shape | verdict |
|---|---:|---|---|
| **`Rineya-Puzzle-Reasoning-51k`** | 40,800 / 4,080 / 4,080 / 2,040 | ~90-token prompt, one short exact answer | **candidate** |
| `Rineya_General_Reasoning_..._v3.0` | 50k + 30k + 30k + 14k SFT | the pool `splits_large` was drawn from | **in use** |
| **`Rineya_Short_Response_Behavior_Corpus_v1.0`** | 10,000 + 2,000 | terse replies, median 33 chars | **candidate** |
| `rineya_reasoning_forge_v1` | ~1,400-token prompts | already tested at 11.5x dose | tested, null |
| `Rineya_Long_Context_Demo_v1` | 256 | evaluation suite, not training data | instrument |
| `Rineya_LCR_SFT_v1` | 12-pack sample only | full data not present in this worktree | unavailable |
| `Rineya_General_..._v2.0`, `Invisible_Limit_Lab`, `Sequence_Experiment`, `short-response-model-phase-0-1-2`, `Rineya_Abstention_Gate_Lab_v1`, `Rineya_*_Soup_Mini_*` | — | superseded by v3.0, or lab/eval rather than SFT | not used |

### The finding that reframes the question

`splits_large` holds **10,000 rows**. The deliverable consumed **41 of them** -- 0.41% of
the pool -- and the four seeds drew **completely disjoint** row sets (pairwise overlap 0)
scoring **74, 62, 73, 68**. The panel's own three classes (`percentage_reasoning`,
`ratio_proportion`, `time_duration`) appear in **none** of the training data.

So the entire 47 -> 73 result rests on ~40 rows, transferring from 36 unrelated classes,
and *which* 40 rows you happen to draw is worth up to 12 panel rows. At this dose the
dominant lever is **selection**, not volume -- and volume has already been measured to
hurt: 15,600 tokens scores 64, a 25,000-token mixture 48, and 11.5x reasoning data moved
reasoning accuracy by zero.

**This is why the plan below curates rather than accumulates.**

`splits_large` rows kept only `source_row_id`, but that joins back to v3.0's full curation
record for **10,000 of 10,000 rows** (verified). `difficulty`, `capability_group`,
`length_mode`, `domain`, `speech_act`, `hidden_reasoning_target` and `quality_rationale`
are all recoverable without rebuilding the pool -- so curated subsets inherit
`splits_large`'s contamination properties instead of needing new guards.

I looked at whether the four seeds' composition predicts their score. It does not, at n=4
across six variables; the one ordering that looks suggestive (fewer `TINY` rows -> higher
score) is a post-hoc correlate from a sample of four and **is not used to design anything
below**.

---

## 2. Weak and strong, as measured

**Strong, must not lose**

| | status |
|---|---|
| Arithmetic panel, 73/150 | measured on the shipped core |
| 41 of 47 reference answers | measured |
| In-distribution reasoning (forge `test_iid`) | preserved, p=0.804 |
| **Long-context, Supa's flagship ability** | **never measured on Opallo** |

**Weak, want to fix**

| | status |
|---|---|
| General reasoning | forge 16%, unmoved by schedule, depth, dose, architecture, soup or reasoning data |
| Off-distribution output discipline | forge `test_transfer` 44 -> 36, termination 184 -> 178 |

---

## 3. Experiments, in order, each blocking the next

### L1 -- long-context retention

Covered by its own `LONG_CONTEXT_RETENTION_PREREG.md`. Runs first because if Opallo has
broken Supa's headline capability, preservation data becomes the corpus's first constraint
and every mixture below changes.

### P1 -- is the reasoning ceiling capacity, or task distance?

The last session concluded the 596M model has a **capacity** ceiling, on the grounds that
nothing moved forge. That conclusion has a confound nobody controlled: forge prompts run
~1,400 tokens against training rows of ~128, so "cannot reason" and "cannot reason *over a
long record*" were never separated.

`Rineya-Puzzle-Reasoning-51k` separates them. Prompts are ~90 tokens, every answer is a
short string, and every row is solver-verified to have exactly one solution
(`solution_count: 1`). Split disjointness was verified here on `id`, `prompt_sha256`,
`template_id`, `structural_signature` and raw prompt text -- all zero -- and zero panel
stems appear in its TRAIN.

**Measure:** Supa and Opallo, greedy, on **450 rows of `test`, 30 per family across all 15
families**, sampled with seed 17 and frozen to a file before either model runs.

**Chance is not zero** on this instrument and will be reported per family: `binary_class`
and `boolean` families are two-way, `three_way_logic` is three-way. A per-family chance
floor is computed from `answer_type` **before** scoring and published beside the result.

**Pre-registered decision rule.** Let `s` be Supa's overall exact-match rate.

| outcome | reading | action |
|---|---|---|
| `s < 0.10` | too far to reach from a 5k-token budget | **stop.** Report it; do not build a puzzle split |
| `0.10 <= s <= 0.80` | headroom exists | proceed to P2 |
| `s > 0.80` | nothing to gain | **stop.** Report it |

Opallo-vs-Supa on the same 450 rows is a paired deterministic comparison; McNemar exact,
two-sided, alpha 0.05. Reported, not used as a gate.

### P2 -- conditional on P1: the curated corpus

Only if P1 lands in the headroom band. Design fixed now so it cannot be tuned to P1's
result:

* TRAIN is built by **selection at a fixed 5,233-supervised-token budget** -- the budget
  that produced the deliverable -- not by increasing the budget. Volume is already measured
  to hurt.
* Arms are run at **the same seed**, so the only difference is which rows were selected.
* Every arm is replicated at **three independent seeds before any claim**, because on this
  panel a single seed is close to uninformative: same-config pairs run 15-24 discordant and
  one reached p=0.0352 on seed alone.

**Retention constraint, fixed now.** A curated arm is only worth reporting as an
improvement if it holds the existing strengths: panel **>= 62** (the worst of the
deliverable's four seeds) and reference answers retained **>= 37** (likewise). An arm that
buys reasoning by spending the panel is a different trade, and gets described as one rather
than as a win.

---

## 4. What will not be done

* **No more raw volume.** Measured: 15,600 tokens -> 64, 25,000-token mixture -> 48, 11.5x
  reasoning data -> zero change. More data is the intervention this programme has most
  thoroughly falsified.
* **No selection fitted to the panel.** Rows are chosen by corpus-intrinsic metadata
  (`difficulty`, `capability_group`, family, verified solution count), never by whether
  they improve the panel. Choosing training rows by test score is how a 47-row leak gets
  built, and this programme has already built one.
* **No single-seed claims.** Three seeds minimum, pooled.
* **No new split directory overwrites an old one.** Create-only; a training corpus is bound
  into every completed run's `data_sha256`.
