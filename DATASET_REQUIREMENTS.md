# How to teach this model a new skill without losing an old one

Derived entirely from measurements in this repository, each rule carrying the
evidence that produced it and the failure it prevents. Nothing here is a general
claim about language models; it is what eleven training runs on a 596M dense core
actually showed.

---

## 1. The new-skill corpus

### 1.1 The target must be atomic, and solver-verified

**Rule.** One rule applied once, with a machine-checkable answer. Not a multi-step
task with a final answer.

**Evidence.** Multi-step targets produced nulls indistinguishable from chance on
two independent instruments — the Reasoning Forge (model 0.160 against a 0.2500
floor) and the 450-row puzzle probe (Supa 102/450 against a 0.229 floor,
`PUZZLE_CEILING.md`). The atomic target produced **289/384 in three consecutive
seeds** (`PROCESS_TRANSITION_ARM.md`). Same model, same optimizer, same window.

**Why.** The programme spent months concluding "this model has a capacity ceiling
on reasoning". The conclusion was an artifact of always testing whole tasks. The
atomic unit of those tasks was learnable all along.

**Check before building.** Can a solver verify every target independently? Forge v2
supplies 36,752 process labels and all 36,752 were re-derived by
`oracle.replay_transition` and agreed — twice, once by the release and once by
`process_arm.py`, which does not trust the recorded audit.

### 1.2 Supervised tokens must be a meaningful fraction of processed tokens

**Rule.** Target ≥ ~5% of the sequence. Below ~1%, you are paying for forward
passes that teach almost nothing.

**Evidence.** The transition corpus runs **9.7%** (21,100 supervised of 216,896
processed) and moved the instrument from 14/384 to 289/384. The registered v2
response-contract arm runs **0.6%** (~9 supervised per 1,603 processed) — sixteen
times less signal for the same GPU (`STATUS.json:95` calls it "0.6% of the
compute").

**Check before building.** Compute `supervised/processed` on the compiled corpus,
not on the raw text. Chat-template overhead and long prompts both push it down.

### 1.3 Prompt length must fit the *qualified* training window, not the eval window

**Rule.** Verify fit with the real tokenizer against the window you are authorized
to train at.

**Evidence.** Forge v2's `sft_reason` view retains **0 of 2,000** `ledger_replay`
rows and **0 of 2,000** `register_machine` rows at the qualified 2048 window
(`forge-v2-prereg-001/V2_FLOORS_AND_FIT_PREREG.md`). An arm trained there cannot
test its own hypothesis. Widening to 4096 is **not** a run but a change to the
frozen design contract (`RRP006_WINDOW_4096_BLOCKER.md`): `design_for` pins
`max_sequence` and `group_updates` refuses any row longer than `update_tokens`.

**Trap.** `data/*/train.jsonl.gz` is **ordered by family**. A prefix sample is a
family sample. A `limit=4000` profile of `sft_reason` reports 0% fit at 2048; the
true figure is ~23%. Measured, recorded, and the reason the fit tables were rebuilt.

### 1.4 Dose: size to the retention budget, not to the acquisition curve

**Rule.** Acquisition keeps improving with data. Retention does not. Pick the dose
from the retention constraint.

**Evidence** (`PROCESS_DOSE_FRONTIER.md`, four rungs, one seed each):

| rows | new skill | panel | retained | gate |
|---:|---:|---:|---:|:--:|
| 384 | 239 | 77 | **40** | PASS |
| 768 | 276 | 68 | 35 | FAIL |
| 1,536 | 289 | 71 | 37 | pass, exactly |
| 3,072 | **304** | 59 | **30** | FAIL |

New skill rises monotonically and never saturates. At 3,072 rows the model goes
*backwards* on `ratio_proportion` against the untrained source (16 → 13) and drops
a termination for the first time anywhere in the programme.

---

## 2. The rehearsal corpus — the part that decides what survives

### 2.1 Rehearsal protects one capability at a time — choose which

> **CORRECTED by RRP-012.** This section first read *"rehearsal must span every
> capability you intend to keep."* That rule was derived from RRP-011 and then
> tested in `REHEARSAL_BLEND.md`. **It is wrong at a fixed budget.** A 50/50 blend
> of abstention and reasoning rehearsal protected *neither*: abstention fell from
> M4's 13/64 to **0**, and long context to 136, below both components. The rule
> below is what the evidence actually supports.

**Rule.** Rehearsal protects the capability it *resembles*, and only that one —
**and only if it holds a critical share of the rehearsal budget.** Presence is not
enough. At a fixed budget you are choosing *which one* capability to protect, not
distributing protection across several. Decide which loss you least can afford and
spend the rehearsal on that.

**Evidence** (`REHEARSAL_COMPOSITION.md`, identical transitions, identical token
budget, only the rehearsal filter varies):

| rehearsal from | long ctx /256 | `UNKNOWN` /64 | panel | retained |
|---|---:|---:|---:|---:|
| *(source Supa)* | 161 | 13 | 47 | 47 |
| none | 150 | 3 | 71 | 37 |
| all classes | 139 | 1 | 74 | 40 |
| `communication` | **118** | 5 | 70 | 40 |
| `reasoning` | **144** | 0 | 69 | **34** |
| **abstention classes** | 137 | **13** | **77** | 39 |
| half abstention + half reasoning | 136 | **0** | 68 | 36 |

Abstention rehearsal returns abstention to **exactly the source's 13/64**. Reasoning
rehearsal gives the best long context and *fails* panel retention. Communication
rehearsal costs 26 long-context rows against reasoning rehearsal.

**The threshold, which is the part that corrects the original rule.** M4 gave
abstention 1,295 rehearsal rows and recovered abstention completely. The blend gave
it **648 rows — half as much — and recovered nothing**. Halving the component did
not halve the effect; it erased it. Protection is a threshold in budget share, not
a gradient, so a fixed budget cannot be spread.

**Why this matters more than "add some old data".** A mixture over everything (the
`all` row) protects nothing in particular: middling on all three axes. Composition
*is* the intervention — and so is concentration.

**Untested:** whether a *larger* rehearsal budget (say 67% of processed tokens
rather than 50%) could protect two capabilities at once. Every arm here held the
budget fixed so composition was the only variable.

### 2.2 Rehearsal must exclude the classes your retention instrument scores

**Rule.** Drop, from the rehearsal pool, every class the retention benchmark
measures. Otherwise "retention" is practice, not transfer.

**Evidence.** The v3 preservation slice has **zero id overlap** with the 150-row
panel but **750 rows in the panel's own three `reply_class` values**. They are
excluded unconditionally and *before* any other filter, and `process_mix.py`
**refuses a source in which the exclusion drops nothing** — so the guard cannot
silently become a no-op.

### 2.3 Mixture ratio: ~50/50 by processed tokens, fixed in advance

**Rule.** Match rehearsal tokens to new-skill tokens. Register the ratio before
running; do not sweep it and report the best.

**Evidence.** At 50/50 (108,372 vs 108,378 tokens) rehearsal turned a gate failure
into a pass and, at **equal retention**, recovered **29 transition answers** that
simply cutting the dose threw away (`PROCESS_REHEARSAL.md`). Replicated 3/3 seeds
with disjoint retention distributions — M1 38–40 against D3 35–37.

**Guard.** `compile_rehearsal` refuses a filter that cannot reach the token budget
rather than under-dosing, because an arm short on rehearsal varies *dose* as well
as composition. It immediately caught the `integrated` group at 105,097 of 108,372.

---

## 3. The evaluation battery — build it before the corpus

### 3.1 Every capability you intend to keep needs an instrument *before* you train

**Rule.** The gate determines what survives. Anything not in the gate is
unprotected.

**Evidence.** The registered gate is `panel ≥ 62 AND references retained ≥ 37 of 47`.
It has no long-context term and no abstention term. Under it, three arms "passed"
while long context fell 161 → 139 and abstention 13 → 1 (`BROAD_RETENTION.md`). The
damage was invisible to the gate for four experiments.

### 3.2 Publish the strongest shortcut floor before any model runs

**Rule.** Compute every answer-agnostic predictor, take the **strongest**, publish
it, then measure.

**Evidence.** On the transition instrument `copy_before` scores **31/128** on
`ledger_replay` — a predictor that ignores the question entirely. Both shipped cores
scored *below* it. Had the floor not been published first, 14/128 would have read as
"some ability". Separately, on `prerequisite_plan` the *stronger* predictor is
`constant_majority` (9/128), not `copy_before` (0/128); naming the weaker one is the
error `RRP002_DEV_V2_FLOOR_CORRECTION.json` had to withdraw, and it was corrected
here by amendment **before** the first model call.

### 3.3 Leak checks must be structural, not substring

**Rule.** Assert the prompt's *structure*. Substring tests go vacuous on small
answer spaces.

**Evidence.** A substring leak detector fired on a clean `prerequisite_plan` row
whose reference is the integer `4`, because `4` appears inside its own event
`{"add":4,…}`. It would have "passed" every other family while being meaningless on
that one. Replaced with: label-bearing id absent, `proposed_after` absent, user
content parses to exactly `{before, event}` — 384/384.

### 3.4 Separate presentation from content in every score

**Rule.** Report strict-exact and a contract-tolerant variant. Never pool them.

**Evidence.** On the panel, seed 17's headline +24 is +12 under the strict policy;
`percentage_reasoning` gains 21 → 30 permissive and *regresses* 21 → 20 strict. The
robust component is `time_duration` (10 → 23 under both). Reporting +24 alone would
have overstated the arithmetic gain by half.

---

## 4. Replication and reporting

- **Three seeds minimum, varying the selection draw**, with the salt fixed so seed 1
  rebuilds byte-identically (verified: sha256 `6fa1bb0d…`). One seed is not a claim.
- **A failing seed is a failure.** D3 passed the gate on 2 of 3 seeds; the honest
  statement is that it gains reliably and forgets unreliably, not that it passes.
- **Report the seed spread onto every comparison.** D3's spread is panel 58–71,
  retained 35–37. D2's "failure" at 35 sits inside it and is therefore seed noise,
  not a dose effect. Three points on the dose curve, not four.
- **Predictions before measurements.** Of the registered predictions across
  RRP-008/009/010/011, most held and three were falsified — the 4096 memory
  estimate, "rehearsal loses less long context", and "panel retention stays ≥ 37 in
  all arms". Each is recorded as falsified in the document that predicted it.

---

## 5. What is still missing, and what it would cost

| gap | why it matters | what is needed |
|---|---|---|
| **No long-context training data** | Every arm loses 17–43 long-context rows and nothing here repairs it. If §2.1 holds, long-context rehearsal is the fix. | `Rineya_LCR_SFT_v1` ships a 12-pack sample only. A real corpus, or generation at ~1,000-word contexts. |
| **Gate has no long-context or abstention term** | Four experiments passed a gate blind to the damage they were doing. | A registered gate extension, written before the next arm. |
| **36 of 39 v3 reply classes uninstrumented** | "Retention" means 3 classes. | A second monitor panel; CPU-only to build, needs its own pre-registration. |
| **Multi-turn rehearsal excluded** | 2,897 of 13,250 rows dropped for using a second serialization mode. | Qualify `compile_dialogue` on this path. |
| **Human data review NOT PERFORMED** | On every DEV bank and the entire Forge v2 release. | No result here can license a release claim regardless of its numbers. |

---

## 6. The short version

1. Atomic, solver-verified targets — not whole tasks.
2. ≥5% supervised-token density; verify against the qualified window with the real tokenizer.
3. Size the dose by the retention budget; acquisition never saturates but retention degrades.
4. Rehearse at ~50/50 by tokens on **one** capability — the one whose loss you least can afford. Rehearsal protects only what it resembles, and only above a critical share of the budget; splitting it across two protected neither.
5. Exclude the retention benchmark's own classes from rehearsal.
6. Build the eval battery **first**; the gate decides what survives, and what is not measured is not protected.
7. Publish the strongest shortcut floor before measuring; three seeds before claiming.
