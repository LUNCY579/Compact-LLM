# RTG-1 — Rineya scout-aligned reasoning corpus

**Version `RTG-1.0.0`. Classification: TRAINING CORPUS / OPEN DEVELOPMENT.**
Branch `research/rineya-reasoning-corpus-v1`, parent `48b0c58`.
Seed `20260917`. Deterministic: regeneration reproduces all 12 data files byte-for-byte.

24,000 rows over four task families, built to train a 596M Rineya checkpoint on the
capabilities the frozen MRS scout bank measures, while being provably disjoint from it.

| | rows | train | dev | holdout |
|---|---|---|---|---|
| `register_machine` | 6,000 | | | |
| `counterfactual_state` | 6,000 | | | |
| `prerequisite_schedule` | 6,000 | | | |
| `evidence_chain` | 6,000 | | | |
| **total** | **24,000** | 19,157 | 2,395 | 2,448 |

Every row carries a **unique canonical structural fingerprint** — 24,000 rows, 24,000
structures. Name-only variants of one structure share an answer and teach nothing extra, so
the variant cap is 1.

## Why this corpus and not the delivered ones

Two corpora already existed. Both were re-audited rather than taken on report.

**`Rineya_LCR_Full_v1_6000`** is P0 STOP, and the blocking failure is arithmetically
unrepairable by relabelling: across all 6,000 rows only 1,902 have native context in
[300,600] while the contract fixes the short band at 2,000, so **at least 98 scenarios must
be regenerated**. Its re-audit enlarged the violation count from 41 to 98 rather than
clearing it.

**`rineya_long_context_cc0_v1_main`** fails on two independent grounds. Its abstention label
is **perfectly predictable from the question alone** — balanced accuracy **1.000**, with
TP 3,668 / FP 0 / FN 0 / TN 40,348 on train and the same on validation, from the single rule
"predict UNKNOWN iff the final user message contains a certified archival shelf date". And
separately, the user's length concern is correct: even the four *shortest* rows per shard in
its smallest stratum render at 2,917–2,943 prompt tokens, so `prompt_le_1920` and
`total_le_2048` are both false with no usable short subset. Truncation is forbidden. It is
unusable here regardless of length.

Neither corpus targets the scout bank's capabilities either. Transfer has to be measured on
the unchanged scout, so a corpus aimed at those four families is the direct instrument.

## Two defects found in the evaluation bank, and what the corpus does about them

The scout bank is frozen and evaluation-only; it is **not** modified. But training on data
that shares its defects would teach the defect, so both were measured and designed against.

### 1. Abstention is a perfect biconditional in three of four families

| family | scout: P(UNKNOWN \| gap) | scout: cue balanced accuracy | RTG-1 |
|---|---|---|---|
| `register_machine` | **1.000** | **1.000** | 0.289 / **0.500** |
| `counterfactual_state` | **1.000** | **1.000** | 0.289 / **0.500** |
| `evidence_chain` | **1.000** | **1.000** | 0.286 / **0.500** |
| `prerequisite_schedule` | 0.545 | 0.896 | 0.295 / **0.500** |

In the bank, `initial is None` ⟺ UNKNOWN for the two register families (48/12, **zero**
off-diagonal), and "the route's endpoint has no record" ⟺ UNKNOWN for `evidence_chain`
(24/6, zero off-diagonal). A model can obtain every abstention from one cue without doing
the task. The bank's own audit measured a *pooled* `missing_marker` feature at 0.302 against
a 0.229 floor, which reads as mild — pooling across families hid it.

An information gap is **logically necessary** for an undetermined answer: a complete record
always determines the result. So "no gap, therefore answerable" is sound reasoning, not a
shortcut, and is not something to engineer away — `P(UNKNOWN | no gap) = 0.000` here by
construction and that is correct. The defect is the converse. RTG-1 attacks it by making
gaps **common among answerable rows**, so that seeing one says little:

* `SET k` makes a value determined even when the initial value is unrecorded;
* an unrecorded event makes it undetermined even when the initial value is known;
* a branching link makes a route ambiguous with every record present;
* records absent from a node **off** the route, or an unrecorded dependency row for a job
  that is not an ancestor of the queried one, are real gaps that **do not matter**.

Gaps appear in ~70% of rows; 30% still carry a complete record. What varies is whether the
missing fact is *relevant*, which is the judgement the task should require. The cue's
balanced accuracy lands at exactly **0.500** — chance — in all four families.

### 2. The bank has nearly exhausted one family's instance space

`prerequisite_schedule` admits only **480** distinct 4-job instances in the entire
construction (256/128/64/32 by `missing`; exhaustively enumerated and independently
re-verified). Only **8** of those have answer "3", which needs a depth-3 chain with the
query at its end — and **the bank uses 6 of the 8**. Exactly two remain.

So balanced 4-job training data at scale is *impossible* while staying disjoint from the
bank. The `match` tier for that family is therefore capped at 10 rows, and 5- and 6-job
schedules carry the volume. They teach the identical algorithm — earliest start is the
longest prerequisite chain, whatever the job count — and the bank's 4-job instances are
strictly simpler than what is trained on.

## Tiers, and why they differ per family

`match` reproduces the scout generator's own parameter range exactly, so a `match` row using
only ADD/XOR renders a **byte-identical** preamble to a bank item (asserted in tests).
`easy` sits below it as curriculum — the checkpoints are at the floor, so shorter instances
matter — and `hard` above it.

| family | easy | match (= eval range) | above |
|---|---|---|---|
| `register_machine` | 1–2 ops (150) | 3–6 ops (3,000) | 7–9 ops (2,850) |
| `counterfactual_state` | 2 ops (150) | 3–6 ops (3,000) | 7–9 ops (2,850) |
| `evidence_chain` | 1–2 links (600) | 3 links (3,000) | 4–5 links (2,400) |
| `prerequisite_schedule` | — | 4 jobs (10) | 5 jobs (1,200), 6 jobs (4,790) |

Every quota sits at or under a **measured** capacity (`capacity.py`, 60,000 draws per cell),
so no cell is ever filled by relaxing balance or repeating a structure. The generator raises
rather than degrading: it names the unfilled bucket and says to reduce the quota.
`prerequisite_schedule/easy` is measured at capacity **0** and deliberately unused — three
jobs cannot reach earliest start 3, so answer "3" is unreachable there.

## Correctness

Every answer is computed by **two independently implemented solvers** that must agree, and
the pair is cross-validated against the bank's own two solvers: on all 120 evaluation worlds,
RTG-1's forward and backward solvers, the scout's forward and exhaustive solvers, and the
frozen gold labels **all agree — four implementations, zero disagreements**.

Fingerprints are canonical. Evidence-chain node integers are arbitrary labels, so a raw-index
fingerprint would let an isomorphic structure receive a different id and slip across a
train/holdout boundary; a recursive subtree signature is used instead, and relabelling
invariance is asserted.

## Disjointness and isolation

* **Structure**: no emitted fingerprint appears in the bank. 7 collisions were generated and
  rejected during the run — the filter does real work.
* **Text**: no name-blinded problem text matches a bank item.
* **Names**: `q` + 7 characters, against the bank's `n` + 6 — disjoint on both prefix and
  length, asserted by regex over the whole corpus.
* **Splits**: 0 structures appear in more than one split.

### Disclosed limitation: semantic isolation is not achievable for two families

Surfaced by independent review. Every sequence of ADD and XOR operations on Z4 composes to
**one of just eight maps** (the order-8 dihedral group), so a register world's answer is
fixed by (initial value, net map). With SET and NOT RECORDED the corpus reaches 65 such
classes — and **all 65 holdout classes also appear in train**. Sequence-level fingerprint
isolation therefore does **not** give semantic isolation for `register_machine` and
`counterfactual_state`.

This is a limit on what the split tests, not a leak: the net map is never written down, so a
model must still compose the listed operations to recover it. The holdout measures that
composition on unseen sequences — as testing addition on unseen operand pairs is still a
real test. Recorded in `evidence/SEMANTIC_EQUIVALENCE.json` with zero class/answer
inconsistencies.

## Balance and shortcut battery

Answers are **exactly uniform**: 4,800 each of 0, 1, 2, 3, UNKNOWN. Dev constant-answer
floor **0.2004**. Every attack is fit on train and scored on **dev**; the holdout is never
read by the audit.

| attack | dev accuracy | floor |
|---|---|---|
| `family_and_gap_marker` | 0.2559 | 0.2004 |
| `gap_marker_only` | 0.2542 | 0.2004 |
| `digit_count` | 0.2295 | 0.2004 |
| `modal_digit` | 0.2279 | 0.2004 |
| `last_digit` | 0.2233 | 0.2004 |
| `question_only` | ~0.2003 | 0.2004 |
| constant answer | 0.2004 | — |

Answerability attacks all sit at **0.500** balanced accuracy (chance); the worst is 0.5010.

Two residuals are disclosed rather than suppressed:

1. **The gap marker carries real information.** It lifts exact-answer accuracy to 0.256
   against a 0.200 floor, because a gap genuinely raises P(UNKNOWN) from 0 to ~0.29. This is
   sound semantics, not a surface artifact, and cannot be removed without making missing
   information meaningless.
2. **`evidence_chain` retains a read-a-printed-value signal**: 0.352 exact on answerable dev
   rows against a 0.251 floor. The mechanism is measured, not guessed — the analytic rate for
   "return the last printed value" is `1/|facts| + (1 - 1/|facts|)/4`, which evaluates to
   0.359 against an observed 0.379 before mitigation. Fact lists were lengthened (2–3 dated
   records per node, mean 7.9 → 10.7 entries), bringing it to 0.352. Driving it to the floor
   would need ~25 records per row. It is irreducible in kind: for a retrieval task the answer
   *is* one of the printed values.

One shortcut was **found and fixed** during construction: a trailing `SET k` made the last
digit in the text *be* the answer, measured at **0.497** exact on answerable
`register_machine` rows against a 0.251 floor. SET and NOT RECORDED are now barred from the
final position, bringing it to 0.270.

## Serving contract

The supervised target is the **bare answer token plus EOS** — because MRS-FP-1 measured that
this family emits exactly one field and then stops. Training and serving therefore share one
contract: the MRS-2 system prompt and `R0` role instruction, **verbatim**, with the manifest
binding the MRS-2 prompt hash.

**Only `R0` is trained**, and that choice is deliberate and neutral. Cell C is A0, which is
R0, so C is in-distribution; A2's three rephrasings and A3's three distinct roles are all
out-of-distribution. Training every role would collapse them and manufacture the "training
made multi-reasoning unnecessary" outcome; training only the direct rephrasings would bias A2
upward against A3. Leaving all six unseen keeps the A2-vs-A3 contrast symmetric. The format
constraint lives in the system prompt, which is constant across roles, so format compliance
should carry over — and MRS-2's smoke gate will catch it if it does not.

## Budget

| quantity | value |
|---|---|
| rendered total tokens | max **551**, p99 457, p50 ~245, mean ~250 |
| token-cap violations | **0** (caps: 2,048 total / 1,920 prompt) |
| truncated or dropped rows | **0** |
| processed tokens, one epoch | 6,224,067 |
| supervised tokens, one epoch | 48,000 |

The supervised fraction is tiny by design: the prompt is context, not a target. Measured with
the exact Supa tokenizer assets, hashes bound in the manifest.

## What this corpus is not

It is open development material, not a confirmation instrument. It has not been trained on,
and no claim about its effect on any checkpoint exists yet. The scout bank remains
evaluation-only and unmodified; this corpus must never be evaluated on, and the bank must
never be trained on. A P0 acceptance pass against the frozen dataset-acceptance requirements
is a separate step, and training authority is separate again.
