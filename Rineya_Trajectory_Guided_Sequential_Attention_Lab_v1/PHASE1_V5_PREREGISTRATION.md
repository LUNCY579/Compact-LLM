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
context*. A model that has learned only "the answer is one of these" scores
`1/node_count`. Computed on the exact registered v4 holdout (seed 987654321, the
`easy_nontrivial` band, 1000 examples per family):

| family | node counts present | in-context candidate null |
|---|---|---:|
| `direct_lookup` | 5:250, 6:243, 7:268, 8:239 | **0.1587** |
| `pointer_chase` | 5:224, 6:218, 7:275, 8:283 | **0.1558** |

v4 measured `direct_lookup` at 0.147–0.171 and `pointer_chase` at 0.159–0.177.

**Both families were at chance for the correct null.** They were not weakly
learning; they had learned the answer *space* and nothing else.

`OBSERVED` **The v4 gate certified them anyway.** In the 200k report,
`every_primary_family_competent` is `true` and `all_arms_beat_trivial` is `true`,
because the gate compared each family against a global 0.070 absolute floor and a
global 0.0377 majority baseline. `pointer_chase` at 0.1590–0.1733 clears both
comfortably while sitting on top of a 0.1558 null.

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
| hop-count modulation | assumed | **registered check** |
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

`state_update` is **removed from the primary families.** It saturated at 0.98 in
v4, it is accumulation arithmetic rather than the iterative pointer-following the
mechanism targets, and a ceiling is as uninformative as a floor. It is retained
as a **secondary descriptive family only**, and cannot contribute to any gate.

Primary families are `pointer_1hop` and `pointer_2hop` — the same generator, with
hop count as the only difference between them, so that hop count is an
*experimental variable* rather than a property confounded with family identity.

```
vocabulary:      domain_size 32, max_hops 8, max_abs_step 8
pointer_1hop:    min_nodes 4, max_nodes 6, hops 1        (null ~0.204)
pointer_2hop:    min_nodes 4, max_nodes 6, hops 2, reject_trivial
                                                          (null ~0.204)
state_update:    min_steps 2, max_steps 3, reject_trivial   SECONDARY, non-gating
guardrail:       pointer_1hop doubles as the non-inferiority guardrail
```

`node_count` is narrowed from v4's 5–8 to 4–6 deliberately. It is simultaneously
the interference level and the null (`1/node_count`), so it is the one lever that
moves difficulty and the comparison point together, keeping "beat your own null"
well defined at every setting.

`split_generation_order` is `["test", "dev", "train"]`. The evaluation splits are
the unbiased draw; v4 shipped a band whose test split was whatever training had
not exhausted.

---

## 4. The prerequisite probe — the cheap gate v4 lacked

v4 spent roughly eight GPU-hours to discover its band could not carry a contrast.
The probe is designed to reach the same class of answer in well under one.

**Measure:** `pointer_1hop` only, `cascade` only, 3 seeds, at the smallest rung of
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

All eight v4 checks are retained unchanged. **One is added**, registered here:

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

The null is **computed from the materialized holdout**, not asserted: for each
example, `1/node_count` where the answer is drawn from an in-context candidate
set, and `1/domain_size` where it is not.

### `hop_count_modulates_accuracy`

Registered as a **reported diagnostic, not a gate**, at this stage.

`pointer_1hop` accuracy must exceed `pointer_2hop` accuracy. If it does not, hop
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
* **Narrowing `node_count` to 4–6 raises the null to ~0.204**, so the absolute
  accuracy required is higher than v4's. This is deliberate — a higher null is
  a *more* honest bar — but it may prove unreachable at these capacities.
* **Depth may fix one-hop and not two-hop.** That would be a real and reportable
  finding about the task, and would still leave the mechanism untested.
* **`state_update`'s removal from the primary families costs continuity** with
  v3 and v4, whose estimands included it. This is stated rather than hidden: the
  v5 primary estimand is not comparable to v3's or v4's, and no cross-study
  pooling is permitted.
