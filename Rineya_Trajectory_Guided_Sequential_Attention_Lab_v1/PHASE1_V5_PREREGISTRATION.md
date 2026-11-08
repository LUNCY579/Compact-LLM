# TGSA Phase 1 preregistration — study v5 (null-anchored competence)

**Status: DRAFT — not registered.** Registration completes when the prerequisite
probe (§4) passes and the frozen band is emitted. No confirmatory matrix may be
dispatched before then, and none is authorized by this document.

This supersedes `tgsa-phase1-identifiability-v4-2026-08-30`, whose regime family
is **rejected** (`PHASE1_V4_COMPETENCE_RESULT.md`). It does **not** reopen the v4
escalation ladder, and no rung of it may be revived.

---

## 1. The correction that motivates v5

`PHASE1_V4_COMPETENCE_RESULT.md` reports that both pointer families were "inert",
sitting at 0.15–0.18 against a 0.0377 trivial baseline — "roughly 4.4× baseline,
**so not at chance**".

`OBSERVED` **That last clause is wrong, and the correction is the whole basis of
this registration.**

`pointer_chase` and `direct_lookup` answer with one of the values *shown in the
context*, so a model that has learned only "the answer is one of these" already
scores well above a uniform baseline. Everything below is measured on the exact
registered v4 holdout: seed 987654321, the `easy_nontrivial` band, 1000 examples
per family.

`OBSERVED` **A first draft of this section got the null wrong, too permissively,
and the correction matters more than the original claim.** `pointer_chase`
carries `reject_trivial`, which removes `value(start)` from the candidate set.
Measured: the answer equals the start value in **0 of 1000** held-out
`pointer_chase` items, and in **156 of 1000** `direct_lookup` items, which has no
`reject_trivial`. The candidate set is therefore `n-1` for one family and `n` for
the other, and using `1/n` for both understates `pointer_chase`'s null by ~0.030
-- exactly the size of the registered margin.

`OBSERVED` **The candidate set is not the only trivial strategy, and the repo
could already measure the others.** `tgsa/phase1_difficulty.py::shortcut_baselines`
computes `copy_start_accuracy` and `copy_last_value_accuracy`, and **nothing
outside that module reads it**, verified by grep across every `.py` in the lab.
The registered null must be the maximum over every strategy that involves no task
competence:

| family | candidate chance | `copy_start` | `copy_last_value` | **registered null** | v4 measured |
|---|---:|---:|---:|---:|---|
| `direct_lookup` | 0.1587 | **0.1560** | 0.1390 | **0.1587** | 0.1470-0.1710 |
| `pointer_chase` | **0.1859** | 0.0000 | 0.1730 | **0.1859** | 0.1590-0.1770 |
| `state_update` | 0.0312 | 0.0000 | 0.0000 | 0.0340 | 0.354-0.987 |

**`pointer_chase` sat BELOW its own null in every one of v4's runs.** Not at
chance -- below the best strategy that ignores the task entirely. And
`direct_lookup`'s entire measured range is explicable by *copy the start value*,
a strategy scoring 0.1560 that the repository could compute before launch.

They had not learned the task at all.

`OBSERVED` **The v4 gate certified them anyway.** In the 200k report,
`every_primary_family_competent` is `true` and `all_arms_beat_trivial` is `true`,
because the gate compared each family against a global 0.070 absolute floor and a
global 0.0377 majority baseline. `pointer_chase` at 0.1590–0.1733 clears both
comfortably while sitting *below* its own 0.1859 null.

`SUPPORTED` **A global baseline systematically overstates competence on any
family whose answer space is an in-context candidate set.** That is not a
threshold set too low. It is the wrong comparison, and it would have silently
passed a dead family in any future study using the same gate.

This is registered as the first design constraint: **every primary family is
scored against its own null.**

---

## 2. What v5 changes, and what it does not

| | v4 | v5 |
|---|---|---|
| per-family comparison | global floor 0.070 | **that family's own null + margin** |
| one-hop learnability | assumed | **prerequisite probe, gates the study** |
| hop-count modulation | assumed, and unmeasurable | **registered diagnostic, null-matched within bucket** |
| depth / capacity | fixed at 2 layers; wider rung never run | **registered ladder, depth first** |
| data volume vs updates | coupled through steps | **separate registered axes** |
| saturation | terminal reject | **terminal reject, unchanged** |

Unchanged from v3/v4 and **not** to be redesigned: the arms (`cascade`,
`history_control`, `learned_delta`), the information-matched control, the
replay-derangement falsifier, `crossed_cluster_t`, `lower_bound_above_zero`, the
hypothesis thresholds, the barrier discipline, and the evidence chain. v3's
statistical design was never what failed.

---

## 3. The registered band

`state_update` is **removed from the primary family set.** It saturated at 0.98 in
v4, it is accumulation arithmetic rather than the iterative pointer-following the
mechanism targets, and a ceiling is as uninformative as a floor. It is retained
as a **non-gating canary**, and cannot contribute to any gate.

`OBSERVED` **A first draft proposed splitting the pointer task into two
separate families, one per hop count. That would have silently broken the
analysis.**
`tgsa/phase1_response.py` binds six difficulty variables -- `effective_hops`,
`context_bucket`, `evidence_position`, `evidence_span`, `distractor_count`,
`node_count` -- to the *literal* family name `"pointer_chase"`, and no-ops on
anything else. `tgsa/phase1_difficulty.py` hard-codes
`{"pointer_chase", "direct_lookup"}`. Renaming disables the entire
difficulty-response analysis without raising.

`OBSERVED` **And it would have confounded hop with the instruction token.**
`phase1_data.py:250` forces `hops = 1` for `direct_lookup`, and line 258 emits
`TASK_LOOKUP` for it against `TASK_POINTER` for `pointer_chase`. A one-hop family
and a two-hop family therefore differ by a *different task token*, not only by
hop count -- which is why v4's "one-hop and two-hop are indistinguishable" could
never have been evidence about composition in the first place.

**Hop varies WITHIN one family, under one task token, with no generator change:**

```
vocabulary:      domain_size 32, max_hops 8, max_abs_step 8
pointer_chase:   min_nodes 6, max_nodes 12, min_hops 1, max_hops 3,
                 reject_trivial          PRIMARY; hop is a within-family covariate
direct_lookup:   min_nodes 6, max_nodes 12, hops forced to 1,
                 reject_trivial          GUARDRAIL  (reject_trivial is NEW in v5)
state_update:    min_steps 6, max_steps 10, reject_trivial   CANARY, non-gating
```

`primary_families = ["pointer_chase"]`, `guardrail_family = "direct_lookup"`.

`OBSERVED` Adding `reject_trivial` to `direct_lookup`, which v4 omitted, drops
its `copy_start` shortcut from **0.1560 to 0.0000** (measured). That single
change removes the strategy accounting for v4's entire `direct_lookup` range.

`OBSERVED` **Hop is null-matched within a context bucket.** Measured on 3,000
held-out items, the registered null varies by **0.00012** across hops 1-3 in the
`medium` bucket and **0.00212** in `short`. A hop contrast *inside a bucket*
therefore compares conditions sharing a task token and a null to within 0.2
accuracy points, differing only in chain length. **Every hop contrast is
evaluated within a bucket, never pooled** -- pooling reintroduces the node-count
shift that rejection sampling induces across hop.

`split_generation_order` is `["test", "dev", "train"]`. The evaluation splits are
the unbiased draw; v4 shipped a band whose test split was whatever training had
not exhausted.

---

## 4. The prerequisite probe — the cheap gate v4 lacked

v4 spent roughly eight GPU-hours to discover its band could not carry a contrast.
The probe is designed to reach the same class of answer in well under one.

**Measure:** `pointer_chase` restricted to hops=1, `cascade` only, 3 seeds, at the smallest rung of
the capacity ladder, on 50,000 unique examples and 1.28 M sample presentations.

**Pass rule — all three must hold:**

| check | threshold | why |
|---|---|---|
| beats its own null | weakest seed ≥ null **+ 0.150** | v4's pointer families sat within ±0.02 of their null. A model that has genuinely learned retrieval should be far above it, not marginally so. |
| not saturated | mean ≤ **0.95** | a ceiling at one hop leaves no room for two hops to be harder |
| reliable | per-seed CV ≤ **0.35** | the bound already registered in v4, unchanged |

**On failure the study does not proceed to multi-hop at that rung.** One-hop
retrieval that is not learnable makes hop count meaningless — v4's one-hop and
two-hop accuracies were indistinguishable precisely because neither was learned.
The registered response is to advance the capacity ladder (§5), not to adjust
the band.

The probe is `development_only`. It cannot certify anything.

---

## 5. The capacity ladder

Registered in advance, in this order. v4 never ran its wider rung, so capacity is
**untested rather than excluded**, and the 2-hop task on a 2-layer model has
exactly zero depth slack.

| rung | embed_dim | layers | heads | rationale |
|---:|---:|---:|---:|---|
| C1 | 96 | 2 | 6 | v4's model. The baseline, so v5 is comparable to what was measured. |
| C2 | 96 | **4** | 6 | depth first: 2-hop composition needs at least two sequential attention operations and C1 has no slack. Width held fixed so the change is attributable. |
| C3 | **192** | 4 | 6 | width, only after depth has been tested alone. |

At most **three** rungs. If C3's probe fails, the task family is rejected and the
question returns to task design, not to a fourth rung.

`OBSERVED` Raising depth also raises capacity to memorize, and the two must not
be confused. `generalization_gap_acceptable` (seen − holdout ≤ 0.10) is retained
unchanged and applies at every rung, so a rung that "passes" by memorizing fails
on that check instead.

---

## 6. Exposure axes

`unique_train_examples` and `sample_presentations` are **separately registered
axes**. `effective_epochs = sample_presentations / unique_train_examples` is a
recorded descriptor, not a control and not a law.

| axis | registered values |
|---|---|
| `unique_train_examples` | 50,000 / 200,000 |
| `sample_presentations` | 1,280,000 / 2,560,000 |

v4 held presentations fixed at 1,280,000 and varied only diversity, which made
epochs and diversity perfectly confounded — the 25.6-vs-6.4 comparison cannot
separate them. Crossing the two axes breaks that confound.

Every bundle records: `unique_train_examples`, `sample_presentations`,
`optimizer_steps`, `microbatch_size`, `gradient_accumulation`, `effective_epochs`.

---

## 7. The competence gate

All eight v4 checks are retained unchanged. **Two are added**, registered here.

> **Amendment, 2026-09-04.** `every_seed_beats_its_family_null` was added on
> 2026-09-04, after the v5.1 lock but **before any nondevelopment run of this
> plan**. No nondevelopment outcome existed when it was written and none was
> inspected. It is **strictly stronger** than what it supplements — it adds a
> condition and removes none, so it can only ever turn a `GO` into a `NO_GO`,
> never the reverse. It changes no threshold value, no plan field, and no
> registered estimand, so `configs/phase1_v5_plan.json` and its locked blob
> `0538faf4d2ad142dbce9c408910f9dcbcf1fd064` are untouched. Recorded here rather
> than silently, because a check enforced in code and registered nowhere is an
> unregistered gate — which `test_the_prereg_gate_table_lists_every_check`
> enforces, and which is how the omission was caught.

### `every_primary_family_beats_its_own_null`

Each primary family's **weakest arm** must exceed **that family's null** by
`minimum_family_margin_over_null` = **0.030**.

*Threshold placement, in the manner v4's thresholds were placed — strictly
between a demonstrated failure and a demonstrated pass:*

* demonstrated failure: v4's `pointer_chase` cleared its null by **+0.003 to
  +0.018** and was certified competent by the old gate;
* the same margin as `minimum_margin_over_trivial`, which is already registered
  and calibrated against `easy_50k` (+0.054) versus `v3_replica` (+0.005).

**Fail-closed:** if no per-family null is supplied, the check fails. A gate
cannot make a comparison it was never given the terms for, and defaulting to a
global baseline is exactly the error being corrected.

The null is **computed from the materialized holdout**, not asserted, and is the
**maximum over every trivial strategy**: in-context candidate chance
(`1/(n-1)` under `reject_trivial`, `1/n` otherwise), `copy_start`,
`copy_last_value`, and the train and eval majority answers. `shortcut_baselines`
already computed the copy strategies before v4 launched. The defect was never
that they were unavailable; it was that no gate read them.

### `every_seed_beats_its_family_null`

Every **individual run** must exceed the **primary family's null** by
`minimum_family_margin_over_null` = **0.030**, on the same family the run was
scored on (`per_seed_holdout_accuracy` is filled from the first primary family).

This is the correction of §1 applied at the level the reliability check actually
operates on. `competence_is_reliable_across_seeds` compares the worst individual
run to `minimum_absolute_accuracy` = **0.070** — a *global* floor. That value was
placed in v4 against a global majority baseline of **0.0377**, where it was a
sane ~1.9× bar. v5 replaced that baseline with each family's own null, measured
at **0.1318** for `pointer_chase` on the registered E1 holdout, but the per-run
floor was never re-derived. It therefore sits at **0.53× chance**: a run at 0.10
has learned nothing and still clears it.

That is the identical defect §1 corrects at the arm level — *not a threshold set
too low, but the wrong comparison* — reproduced one level down. The gate's own
contract is that **"some runs failing to become competent at all is a competence
problem, and no seed count fixes it."** A run at chance has not become
competent, so it is measured against the null, not against a global floor.

*Threshold placement, in the manner of the other registered thresholds:*

* **demonstrated failure:** per-seed `pointer_chase` accuracies of
  `[0.0966, 0.1948, 0.1948]` give mean 0.16207 and CV 0.3498, so they clear
  `minimum_family_margin_over_null` on the arm mean and sit inside the CV bound
  of 0.35 — and the old gate returns `GO`, all checks true and `reasons` empty,
  with one run **0.0352 below** the family null. Reproduced against the measured
  E1 null on 2026-09-04. The passing region is not a corner: arm means from
  0.1618 to 0.2213 admit a worst run as low as 0.0964.
* **the same 0.030 margin** as `every_primary_family_beats_its_own_null`, so the
  per-run bar and the per-arm bar are the same bar. A run below the arm-level
  margin cannot be described as competent while the arm containing it is.

The CV bound does not close this. It separates a *balanced* two-point mixture
(CV ≈ 0.6) from ordinary dispersion, but a 1-of-3 mixture near the decision
threshold sits at CV ≈ 0.35 by construction and passes.

**Fail-closed:** if no per-family null is supplied, or no primary family is
supplied, the check fails — for the same reason
`every_primary_family_beats_its_own_null` fails closed.

**Why the probe does not already cover this.** The prerequisite probe *does*
make the correct comparison, requiring the weakest seed to clear the null by
0.150 (§4). But `probe.arm` is `cascade` alone, at `hops=1`, on a separate
dataset seed. `history_control` and `learned_delta` are never probed, and they
are the arms the mechanism contrast is built from. The qualification gate is the
only place their individual runs are tested at all.

### `hop_count_modulates_accuracy`

Registered as a **reported diagnostic, not a gate**, at this stage.

Within a context bucket, hops=1 accuracy must exceed hops=3 accuracy. If it does not, hop
count is not the operative difficulty and the composition interpretation of any
subsequent contrast is unsupported — v4's two pointer families differed by
0.005–0.019, which is why its "2-hop composition" framing was unfounded.

It is a diagnostic rather than a gate because the direction is a prediction of
the task design, and gating on it would make a surprising-but-real result
unreportable.

---

## 8. Terminal rejections

Unchanged in spirit from v4, and none of them is an invitation to add a rung.

* `no_primary_family_saturated` — **terminal**. A ceiling cannot show a
  decrement.
* `every_primary_family_beats_its_own_null` failing at every registered capacity
  rung — **terminal**. The task family does not measure what it was chosen to
  measure.
* Any check with no registered response — **terminal**, as in v4.

On any terminal rejection the band is abandoned and the question returns to task
design under a new registration. The v4 ladder recorded that exhaustion "does not
authorize an unregistered fifth rung", and that holds here.

---

## 9. Claim ceiling

> v5 is an apparatus registration. It establishes whether a band exists in which
> the three registered arms can be compared informatively. It does not test the
> TGSA mechanism, and no result from the prerequisite probe or the competence
> measurement may be reported as evidence for or against it.

The programme has **no** mechanism evidence. v3 was inconclusive because no arm
was competent; v4's regime family was rejected before a confirmatory matrix was
dispatched. Nothing in v5 changes that until a competence `GO` exists and a
confirmatory matrix is separately registered and run.

MTP work (Phase 2A/2B) remains non-evidentiary and is not authorized by this
document. Phase 2 may not treat a v5 competence `GO` as its own within-study
qualification: scope identity is enforced by `tgsa/scope.py`, and a report earned
under this registration's task, model, objective and metric certifies only a
study matching them.

---

## 10. Honest risks

* **The probe may fail at every rung.** Then the pointer family is the wrong
  probe for this mechanism and the cost of learning that is roughly three short
  development runs rather than v4's eight GPU-hours. That is the intended
  trade.
* **`node_count` 6-12 gives a pooled `pointer_chase` null of ~0.131** (measured),
  lower than v4's 0.186 because there are more candidates. The bar is lower in
  absolute terms but honest, where v4's was not. The risk runs opposite to the
  first draft's: more candidates means harder retrieval, and the probe may fail
  at every rung for that reason.
* **Depth may fix one-hop and not two-hop.** That would be a real and reportable
  finding about the task, and would still leave the mechanism untested.
* **`state_update`'s removal from the primary families costs continuity** with
  v3 and v4, whose estimands included it. This is stated rather than hidden: the
  v5 primary estimand is not comparable to v3's or v4's, and no cross-study
  pooling is permitted.
