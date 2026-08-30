# TGSA Phase 1 preregistration — study v4 (competence-first)

**Status: DRAFT — not yet registered.** Registration completes when the frozen
competence report returns `GO` and `scripts/build_phase1_v4_config.py` emits
`configs/phase1_identifiability_v4.json`. Numeric slots marked *(from artifact)*
are populated by that script from the frozen inputs and are not to be typed by
hand. Until then this document fixes the **rules**, not the numbers.

---

## Scope of the amendment

v4 supersedes `tgsa-phase1-identifiability-v3-2026-08-29`.

v3 executed exactly as registered: 60/60 cells, a clean train → barrier →
single-evaluate sequence, 2001 passing evidence-chain checks, and a `decision`
of `inconclusive`. Nothing about its execution is retracted, and the frozen v3
result and its tag remain immutable.

What v3 revealed is that its **regime** was uninformative. All three arms
memorized the 5,000-example training set and scored at the majority baseline on
held-out data; an in-band held-out test (same difficulty, novel instances, zero
latent-signature overlap) gave 0.236 seen versus 0.045 held-out against a 0.0407
majority baseline. Three non-learners were compared, and the comparison
necessarily returned nothing.

> **This is not evidence against TGSA.** A contrast between arms that cannot
> perform the task is uninformative in both directions. v3 constrains the
> *experimental setup*, not the mechanism. Any statement of the form "v3 shows
> trajectory guidance does not help" is unsupported by the v3 record and is
> explicitly out of scope for every document in this program.

**v4 changes only the regime in which the contrast is run**, plus a launch
precondition that makes the v3 failure mode impossible to repeat. The mechanism,
the arms, the information-matched control, the replay-derangement falsifier, the
interval estimator, the primary gate, the hypothesis thresholds, the barrier
protocol, and the evidence chain are carried over from v3 **unchanged**.

---

## Purpose

The purpose of v4 is **not** to make TGSA win.

The purpose is to construct an experimental system in which TGSA is capable of
winning or losing for a scientifically meaningful reason. A negative result in a
competent regime is a result. A null in an incompetent regime is not.

---

## The competence gate (new; the only structural addition)

`tgsa/phase1_competence.py` emits a machine-readable `GO` / `NO_GO` from
development-only evidence. `run_study_matrix` calls `require_go` before it
creates an output root, so a confirmatory matrix **cannot be dispatched** on a
missing, malformed, superseded-schema, or `NO_GO` report.

Eight checks, every threshold frozen from the outcome-blind pilot
(`PHASE1_V4_PILOT_REPORT.md` §7) before any v4 confirmatory run:

| check | why it exists |
|---|---|
| `all_arms_beat_trivial` | the **weakest** arm must clear the strongest registered trivial baseline by ≥ 0.030 |
| `all_arms_reach_absolute_floor` | ≥ 0.070 absolute held-out accuracy |
| `answer_nll_beats_uniform` | v3 was above the majority baseline while *confidently wrong*: NLL 4.03 against ln(32) = 3.466 |
| `every_primary_family_competent` | v3's `state_update` sat below chance while `pointer_chase` was near baseline; a pooled score hides a dead family |
| `predictions_are_input_dependent` | rejects a constant predictor scoring at the majority baseline |
| `no_primary_family_saturated` | a ceiling is as uninformative as a floor — no room for a decrement |
| `generalization_gap_acceptable` | seen − holdout ≤ 0.10; memorization must never substitute for competence |
| `competence_is_reliable_across_seeds` | the worst individual **run** must clear the floor, and per-arm coefficient of variation must be ≤ 0.35 — the signature of a mixture of converged and unconverged runs, as distinct from ordinary dispersion, which is a power problem and is handled below |

The gate is **arm-blind**: every check is evaluated on the weakest arm or on all
arms, never on a between-arm difference.

### The selection rule, stated so it cannot be softened later

> A configuration is **never** selected because `learned_delta` beats another arm
> on development data. Selection is on absolute competence only, scored by the
> minimum across all three arms.

The v4 pilot produced a live test of this rule. In `easy_50k_long`,
`learned_delta` averaged 0.3737 against controls at 0.2633 and 0.2637 — a
+0.11 apparent margin. It was not used, and §2 of the pilot report shows it was
an artifact of which seeds finished a training phase transition, not a mechanism
effect. The rule held under temptation, which is the only time a rule matters.

---

## Research question (unchanged from v3)

Does learned finite-difference trajectory guidance add capability beyond

1. sequential depth itself, and
2. an information-matched learned two-state history pathway?

## Hypothesis hierarchy (unchanged from v3)

- **H1a:** `learned_delta` > `history_control` on the macro-average of held-out
  pointer-chase and state-update strata.
- **H1b:** `learned_delta` > `cascade` on the same estimand.
- **H1c:** aligned `learned_delta` > `replay_derangement(learned_delta)`.
- **Guardrail:** `learned_delta` is non-inferior to `history_control` on
  `direct_lookup`.

### Interpretation, fixed in advance (unchanged from v3)

| Outcome | Registered interpretation |
|---|---|
| H1a, H1b, H1c all pass | Controlled evidence that trajectory-specific, example-aligned information adds capability beyond the sequential cascade and an information-matched history control. |
| H1b passes, H1a fails | Evidence for an extra learned recurrent pathway, **not** for trajectory-specific capability. |
| H1a and H1b pass, H1c fails | The finite-difference parameterization may help, but example-aligned trajectory content has **not** been shown causal. |
| Primary interval includes zero | **Inconclusive, not success.** In a gated-competent regime this is now informative about the mechanism, which it was not in v3. |
| Effect is negative | Evidence **against** the mechanism in this setting. |

The v4 addition to this table is the fourth row's second sentence, and it is the
entire point of the study: a null in a competent regime means something.

---

## Fixed design

| field | value | source |
|---|---|---|
| arms | `cascade`, `history_control`, `learned_delta` | unchanged from v3 |
| difficulty band | `easy_nontrivial` — pointer 2 hops / 5–8 nodes, state 2–3 steps, trivial items rejected | competence report *(from artifact)* |
| train examples per family | *(from artifact)* | competence report |
| update budget | *(from artifact)* | competence report |
| model | 96 dim / 2 layers / 6 heads / rank 16, ~255k params | unchanged from v3; the pilot certified this exact model |
| batch, LR | 64, 1e-3 with warmup + cosine | competence report *(from artifact)* |
| seeds | *(from artifact)* | power report `recommended_seed_count` |
| estimator | `crossed_cluster_t` | unchanged from v3 |
| primary gate | `lower_bound_above_zero` | unchanged from v3 |
| thresholds | history 0.02, cascade 0.02, alignment 0.015, guardrail −0.01 | unchanged from v3 |

### Head-0 semantics (preserved unchanged)

Head 0 has no previous carrier, so its velocity and its control are identically
zero and `effective_conditioned_heads = H − 1`. This is a property of the
mechanism, not a configuration choice, and v4 does not alter it.

---

## Data

v4 tests **in-band**: dev and test are drawn from the same difficulty band as
train, with different dataset seeds and zero latent-signature overlap, so the
only thing that varies is instance novelty.

This is a deliberate change from v3, which used `split_overrides` to make dev
and test *harder* than train. That design stacked difficulty extrapolation on
top of the mechanism contrast; with no arm competent at either, the result could
not separate the two failures. Difficulty extrapolation remains a registered
**non-gating secondary** analysis (below), so the question is deferred, not
dropped.

`OBSERVED` The narrow band's cost to the secondary analyses is recorded under
§Stratum structure above, measured rather than asserted.

This is a stated limitation of choosing a band that has been certified
competent, and it is **not** to be repaired by widening the band after the fact
— a wider band is a different regime and would need its own competence
certification. If difficulty response becomes the question of interest, it
belongs in a follow-up study, not in a post-hoc amendment to this one. The
primary endpoint does not depend on these two variables.

---

### Split generation order — a defect found at confirmatory scale

`OBSERVED` Splits are generated in sequence and each excludes the latent
signatures already taken, so **whichever split is generated first is the only
unbiased draw**. The pipeline generated `train` first.

On this band the non-trivial two-step `state_update` space is exactly
32 starts × 16 first steps × 15 non-cancelling second steps = **7,680 items**. A
200,000-example train split takes 66,666 `state_update` examples and consumed
all 7,680 of them, so dev and test contained **only three-step items** — an
unintended difficulty shift in a study whose whole premise is that dev and test
are in-band. It is not visible at pilot scale, which is why it was not found
until the dataset was materialized at confirmatory size.

Fixed by making the order explicit. v4 configurations generate **test, then dev,
then train**, so the evaluation splits are drawn from the full band and train
takes the remainder — which is the correct direction to lose data, since train
only needs to be large, not representative in the same strict sense. The order
and the identity of the unbiased split are recorded in the dataset manifest.

The default remains `["train", "dev", "test"]` so any existing configuration,
including v3's, regenerates byte-identically.

Measured at 200k, 2,000 test examples per family:

| | train-first (defect) | test-first (fixed) |
|---|---|---|
| test `state_update:s2` | **0** | 983 |
| test `state_update:s3` | 2000 | 1017 |
| primary strata | 2 | **3** |

### Stratum structure

`OBSERVED` The stratum key is `family:hops-or-steps:context_bucket`. With the
order fixed, the certified band yields **3 primary strata** against v3's 17:

| | v4 `easy_nontrivial` | v3 |
|---|---|---|
| `pointer_chase` | `h2:short` — **1** stratum | h4–h8 × {medium, long} — **10** |
| `state_update` | `s2:short`, `s3:short` — **2** strata | s6–s12 — **7** |
| guardrail `direct_lookup` | `h1:short` — 1 | `h1:{medium, long}` — 2 |

Every sequence on this band is ≤ 32 tokens, so `context_bucket` is constant, and
the band fixes pointer chains at 2 hops. Both collapses are structural.

Three consequences, registered rather than discovered later:

1. **The estimand's family weighting changes.** The macro-average weights strata
   equally, so v4 weights `pointer_chase` 1/3 and `state_update` 2/3, where v3
   weighted them 10/17 and 7/17. The *rule* is unchanged — RULE 1 — but its
   realization is not, and a mechanism effect concentrated in `pointer_chase` is
   down-weighted about 2:1 relative to a family-balanced view. This is why
   `per_family_accuracy_delta` is reported alongside the primary estimand and
   must be read with it.
2. **Each stratum is far better sampled.** The primary strata hold ~2,000 /
   ~1,000 / ~1,000 examples against v3's ~40–90, so the within-stratum variance
   component is much smaller while the between-stratum component is estimated
   from 3 units instead of 17. Which dominates is an empirical question for the
   power audit, not something to assert here.
3. **The power audit must be run at `--stratum-count 3`**, not v3's 17, with
   `--examples-per-stratum` ~1333. Running it at v3's structure would report the
   power of a study nobody is going to run.

`OBSERVED` The narrow band also leaves registered **secondary** difficulty
variables degenerate at confirmatory scale: `context_bucket` and, within the
primary families, `effective_hops`. These are reported with a single bin and no
trend contrast is estimable. The band is **not** to be widened after the fact to
recover them — a wider band is a different regime needing its own competence
certification. The primary endpoint does not depend on them.

## Primary estimand (unchanged from v3)

For each paired seed and each registered non-guardrail stratum: paired
per-example correctness difference; average within stratum; macro-average strata
within seed; average across the paired seed effects. Interval by
`crossed_cluster_t` at 95%.

Answer-NLL improvement is a supportive endpoint, not a replacement for accuracy.

## Fixed gates (unchanged from v3, plus the competence precondition)

| Gate | Rule |
|---|---|
| **Competence** | frozen report returns `GO` **before** the matrix is dispatched |
| H1a | lower 95% bound of (`learned_delta` − `history_control`) > 0 |
| H1b | lower 95% bound of (`learned_delta` − `cascade`) > 0 |
| H1c | lower 95% bound of (aligned − replay-deranged) > 0 |
| Guardrail | lower bound of the `direct_lookup` contrast ≥ −0.01 |
| Falsifier coverage | minimum per-seed length-matched derangement coverage ≥ 0.95 |
| Evidence integrity | every check in `scripts/verify_phase1_evidence.py` passes |

## Decision rule (unchanged from v3)

`decision` is forced to `smoke_only` unless the configuration declares
`confirmatory: true` **and** the analyzed matrix is the complete registered
arm/seed matrix **and** the study ID matches **and** the config hash matches.

## Execution protocol (unchanged from v3)

PREPARE → **COMPETENCE GATE** → TRAIN → BARRIER → SINGLE EVALUATE → VERIFY →
ANALYZE. No confirmatory outcome is inspected before the barrier. Failed
attempts are recorded immutably and are not overwritten by retries.

### Rehearsal before dispatch

v4 changes the regime by an order of magnitude in both data volume and update
count, on a band the registered pipeline had never generated, with no
`split_overrides` and a new gate in front of the runner. The identical sequence
is therefore rehearsed at toy scale first — 3 arms × 3 seeds, 2,400 examples,
150 steps — including a **negative control** in which the gate is flipped to
`NO_GO` and the runner must refuse before creating an output root.

The rehearsal is not evidentiary and declares its own study id. Its purpose is
that a defect costs minutes rather than ~18 GPU-hours and an immutable failure
record inside the confirmatory chain. It has already earned that: it found a
provenance probe that would have aborted the first cell of the matrix
(`8fc5607`).

## Secondary difficulty-response analyses (non-gating, unchanged from v3)

Preregistered, and explicitly unable to rescue a failed primary endpoint.
Bins are cut at quantiles **of the materialized dataset**, fixed before
training. No bin is chosen after seeing outcomes.

## Efficiency (descriptive, never gating, unchanged from v3)

Accuracy and capability come first. Throughput and memory are recorded with
`gating: false` and cannot influence any decision.

---

## Escalation

Registered **before any 20,000-update result existed** — the base regime had one
partial arm/seed measurement on disk and nothing else.

A gate that can be re-run until it passes is not a gate. But a gate that permits
no response to a diagnosable failure is superstition rather than rigour: the
pilot's first `NO_GO` had an obvious mechanistic cause and refusing to act on it
would have thrown away the diagnosis.

The line between them is that the moves are fixed in advance, each is tied to
the specific check that failed, each may be taken **at most once**, and the
number of distinct **regimes** measured is capped at **four**.

The cap counts regimes rather than raw measurements, deliberately. Re-measuring
the *same* regime after a documented harness defect is a correction, not an
escalation — it does not change what is being tested, so it cannot be a way of
searching for a regime that passes. Re-measuring the same regime *without* such a
fix is exactly the dice-rolling the cap exists to stop, and it is refused
separately as a rung already attempted. Every measurement, corrected or
superseded, stays in the record.

Each response moves **one axis of the regime that failed**, not to a fixed rung:

| failed check | registered response | why that response |
|---|---|---|
| `competence_is_reliable_across_seeds` | raise updates to **30k**, data unchanged | a mixture of converged and unconverged runs means the budget truncates the training phase transition |
| `generalization_gap_acceptable` | raise data to **200k**, updates unchanged | at a fixed batch size the gap is a function of epochs — 20k updates over 50k examples is ~25.6 epochs, over 200k it is ~6.4. More *updates* would make it worse, so the response is data, not budget |
| both | raise both | the two responses are independent and compose |
| the axis is already at its registered maximum | **REJECT** | the ladder has no move left on the axis the diagnosis points at |
| anything else | **REJECT** | a dead family, a saturated ceiling, or NLL worse than uniform is neither a budget nor a volume problem, and no rung addresses it |

Registered maxima: **200,000** examples and **30,000** updates.

`OBSERVED` The responses are relative because an absolute-rung formulation was
wrong. It told a 200k regime failing on mixture to escalate to "50k × 30k" — a
*reduction* in data as the response to a problem data had already solved. Caught
before the 200k measurement produced a report.

On exhaustion — four measurements taken, a rung repeated, or a failure with no
registered response — the regime family is **rejected** and the difficulty band
is revisited under a new preregistration. That is a result about the
experimental setup, not a failure of nerve, and it is the same kind of result
v3 produced at far greater cost.

`OBSERVED` **The ladder's answer must not depend on report order, and an earlier
revision's did.** Regimes are routinely measured in parallel — two competence
runs on one GPU is the documented throughput ceiling — so "the most recent
report" is whichever finished first. Reading only the last report returned a
permitted move for one ordering and `REJECT` for the other on identical
evidence. Every failing report's registered response is now considered,
responses already measured are dropped, and the smallest remaining regime wins
on a fixed tiebreak that cannot depend on which report scored better.

This was found while two parallel measurements were at eight of nine cells, with
partial results already indicating that the fix changes the answer from `REJECT`
to one further permitted run. Recording that plainly because it is the kind of
change that must not be made quietly: it removes an arbitrary dependence rather
than adding any move, every move it can return was already registered in the
table above, and the four-regime cap still binds. The fix would equally have
produced `REJECT` had both regimes failed the same way.

`scripts/phase1_v4_escalation.py` reads the reports produced so far and answers
what is permitted next, refusing past the cap, so the rule is enforceable rather
than aspirational. `tests/test_v4_escalation.py` proves by exhaustion that no
sequence of permitted moves can run forever.

`OBSERVED` The two competence measurements launched before the split-order fix
generated their **training** split first, so their holdouts inherited the same
depletion: `state_update` holdout items are three-step only. That makes those
measurements **conservative** — the holdout is strictly harder than the
confirmatory test set will be, so competence measured on it under-states
competence on the registered distribution, and a `GO` from them is trustworthy in
the direction that matters. `scripts/phase1_v4_competence_run.py` now generates
the holdout first, so the certified distribution matches the confirmatory one.

`OBSERVED` The 200k × 20k measurement was launched on a single partial
observation — one arm, one seed, showing a 0.168 memorization gap — before this
ladder was written down. The scheduling decision preceded the rule; the rule
makes it a registered rung rather than an improvisation, and it is recorded here
in that order rather than presented as though the ladder came first.

---

## Power

The v3 seed count of 20 was powered for σ_a = 0.02 central / 0.04 robustness at
a +0.020 target effect. v4 **re-runs the outcome-blind power simulation with the
realised σ_a from the competence run** before the seed count is registered. The
count is carried over only if the realised dispersion supports it, and is raised
otherwise.

This is not optional. The pilot showed a regime whose realised σ_a was ≈ 0.20,
where no realistic seed count would have detected the target effect, and the
existing seed count would have been carried forward silently.

---

## Claim ceiling

> Controlled tiny-decoder mechanism identification only. Qwen/GQA,
> natural-language, long-context, and efficiency claims remain out of scope.

The Rineya retrofit track remains **explicitly non-evidentiary** until the
mechanism has been fairly tested here.

---

## Change control

Any change to this document after registration is an amendment with its own
study ID, and supersession must state what changed and why. The `analysis` block
of the registered config is copied from v3 programmatically, so a silent drift
in the statistical design is not possible.
