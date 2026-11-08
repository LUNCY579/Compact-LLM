# TGSA Phase 1 v3 — frozen confirmatory result

**Study:** `tgsa-phase1-identifiability-v3-2026-08-29`
**Execution commit:** `72395b69beb96dc71dc80bc8c73ef75205ea51a1`
**Design freeze:** `04d3c02` (tag `tgsa-phase1-v3-launch-2026-08-29`)
**Matrix:** 3 arms × 20 seeds = 60 cells, 2,500 updates each, all completed
**Evidence chain:** 2,001 checks, all passed, zero errors

Reported under the PART 11 standard. `OBSERVED` / `SUPPORTED` / `HYPOTHESIS` are
marked. Nothing unfavourable is omitted.

---

## 1. OBSERVED FACTS

### 1.1 Execution

`OBSERVED` 60/60 cells trained to exactly 2,500 updates. Single source head
`72395b6…`. Barrier passed before any test prediction existed. One confirmatory
evaluation sweep produced 180 prediction files. Evidence chain: **2,001 checks,
all passed**.

`OBSERVED` Three cells failed transiently during training and were retried to
success: two CUDA out-of-memory, one Windows `PermissionError` on a concurrent
`preflight.json` replace. All three failure records are preserved immutably. Both
are infrastructure races in the sharding driver, not study defects; the retried
cells resumed from their checkpoints and the barrier verified all 60.

### 1.2 Primary contrasts (`crossed_cluster_t`, 95%)

| contrast | observed | 95% CI | SE | P(>0) |
|---|---:|---|---:|---:|
| **H1a** delta − history | **+0.0014** | [−0.0021, +0.0048] | 0.0016 | 0.790 |
| **H1b** delta − cascade | **+0.0004** | [−0.0025, +0.0034] | 0.0014 | 0.619 |
| **H1c** aligned − replay-deranged | **+0.0019** | [−0.0003, +0.0040] | 0.0010 | 0.959 |
| guardrail direct-lookup | −0.0007 | [−0.0052, +0.0038] | 0.0022 | 0.377 |

`OBSERVED` All four sensitivity estimators agree on H1a: `crossed_pigeonhole`
+0.0014 [−0.0042, +0.0070], `crossed_pigeonhole_t` +0.0014 [−0.0020, +0.0046],
`seed_cluster` +0.0014 [−0.0020, +0.0047]. **The conclusion is not
estimator-dependent.**

### 1.3 Realised seed dispersion

| contrast | mean | seed SD | SE of mean | negative seeds |
|---|---:|---:|---:|---:|
| H1a | +0.00135 | 0.00708 | 0.00158 | **9 / 20** |
| H1b | +0.00043 | 0.00598 | 0.00134 | **8 / 20** |
| H1c | +0.00189 | 0.00468 | 0.00105 | **7 / 20** |
| guardrail | −0.00069 | 0.01045 | 0.00234 | 9 / 20 |

`OBSERVED` Realised `sigma_a` for H1a is **0.0071**, well below the 0.020 the
power audit assumed. The study was therefore **more precise than designed**: the
H1a interval half-width is ±0.0035, and the interval **excludes the registered
+0.020 minimum effect of interest entirely**.

### 1.4 The manipulation check — all arms are at chance

`OBSERVED` This is the decisive observation.

| baseline / arm | direct_lookup | pointer_chase | state_update |
|---|---:|---:|---:|
| uniform over 32 answers | 0.0312 | 0.0312 | 0.0312 |
| test-set majority answer | 0.0500 | 0.0433 | 0.0433 |
| `cascade` | 0.0447 | 0.0464 | 0.0264 |
| `history_control` | 0.0436 | 0.0447 | 0.0273 |
| `learned_delta` | 0.0429 | 0.0443 | 0.0307 |

`OBSERVED` **No arm beats the constant "predict the most frequent answer token"
baseline on any family.** On `state_update` every arm is *below* uniform chance.

`OBSERVED` Mean answer-NLL is 4.05–4.07. Uniform over the 32 possible answers is
ln(32) = 3.466; uniform over the whole 106-token vocabulary is ln(106) = 4.663.
**Every arm is worse than a uniform guess restricted to the answer space.**

`OBSERVED` Training loss did fall, 4.711 → 2.88–2.99 over 2,500 updates, so the
models learned *something* — consistent with learning the answer-token type prior
rather than the pointer/state mapping.

`OBSERVED` Development accuracy agrees with test: 0.043–0.044 for every arm.

### 1.5 Intervention coverage

`OBSERVED` Length-matched replay-derangement coverage is **1.0000 on every
seed** — the falsifier applied to every test example.

### 1.6 Secondary difficulty response (non-gating)

| variable | linear trend | 95% CI |
|---|---:|---|
| `effective_hops` | **−0.0109** | **[−0.0206, −0.0012]** |
| `node_count` | −0.0050 | [−0.0148, +0.0048] |
| `distractor_count` | −0.0031 | [−0.0123, +0.0061] |
| `evidence_position` | −0.0027 | [−0.0130, +0.0076] |
| `evidence_span` | −0.0027 | [−0.0152, +0.0098] |
| `effective_steps` | +0.0024 | [−0.0118, +0.0167] |
| `context_bucket` | +0.0037 | [−0.0023, +0.0096] |

By pointer-chase depth: 4 hops +0.0056, 5 hops +0.0054, 6 hops −0.0034,
7 hops −0.0029, 8 hops −0.0074.

---

## 2. REGISTERED DECISION

`OBSERVED` The frozen analyzer returned:

> **`decision: inconclusive`**
> H1a `False`, H1b `False`, H1c `False`
> "inconclusive, not success: the primary confidence interval includes zero"

Gates: H1a FAIL, H1b FAIL, H1c FAIL; guardrail non-inferiority PASS, falsifier
coverage PASS, full registered matrix PASS, evidence integrity PASS.

Not reinterpreted here.

---

## 3. SUPPORTED INTERPRETATION

`SUPPORTED` **The experiment did not test the mechanism, because the manipulation
failed.** All three arms perform at or below a constant-answer baseline, and all
have answer-NLL worse than a uniform guess over the answer space. A contrast
between three models that did not learn the task carries no information about the
trajectory mechanism.

`SUPPORTED` This is a **floor effect**, not a measurement of a small effect. The
registered decision `inconclusive` is correct, and the reason is construct
invalidity rather than insufficient precision.

`SUPPORTED` Precision was *not* the limitation. Realised seed SD was 0.0071
against an assumed 0.020, and the H1a interval [−0.0021, +0.0048] confidently
excludes +0.020. Had the arms been competent, this design would have detected an
effect far smaller than the one it was powered for.

`SUPPORTED` **The registered design has a gap: it contains no competence gate.**
v2/v3 were rigorous about blinding, evidence chains, estimator calibration, and
power, but never asked whether any arm can perform the task above a trivial
baseline. That check should have been a launch gate.

---

## 4. UNSUPPORTED CLAIMS

`OBSERVED` None of the following is supported in either direction:

* that finite-difference trajectory guidance helps (H1a);
* that it beats sequential depth (H1b);
* that aligned trajectory content is causal (H1c);
* that the mechanism **fails** — it was not given a fair test;
* any difficulty-response signature;
* anything about transfer to a pretrained model.

`OBSERVED` Specifically, this result must **not** be reported as evidence against
TGSA. A null between non-learners is uninformative, not negative.

---

## 5. FALSIFIERS AND ANOMALIES

`OBSERVED` **Negative seeds:** 9/20 for H1a, 8/20 for H1b, 7/20 for H1c. Roughly
a coin flip, as expected when the underlying effect is indistinguishable from
zero.

`OBSERVED` **`state_update` is below uniform chance** for every arm
(0.0264–0.0307 vs 0.0312). Worth noting rather than smoothing over.

`OBSERVED` **The difficulty response runs opposite to the hypothesis.** The
`effective_hops` trend is −0.0109 with CI [−0.0206, −0.0012], nominally
excluding zero — the advantage *shrinks* with iterative depth, the reverse of the
theoretically compelling TGSA signature.

`HYPOTHESIS` That trend should not be taken seriously: it is one nominally
significant result among seven exploratory difficulty variables with no
multiplicity correction, computed over arms that are all at chance. Expected
false positives at α=0.05 across 7 variables is 0.35.

`OBSERVED` **H1c was the closest to firing** (P(>0) = 0.959, CI upper −0.0003
from zero). With competent arms this contrast might behave differently; with
chance-level arms it is not interpretable.

`OBSERVED` **Three transient infrastructure failures** (2 CUDA OOM, 1 Windows
file-replace race), all preserved, all retried to success.

`OBSERVED` No non-finite losses, no corrupt checkpoints, no collapsed gates, no
gradient anomalies.

---

## 6. LIMITATIONS

* The registered training budget — 2,500 updates on a 96-dim, 2-layer model —
  is insufficient for 4–8 hop pointer chasing over 16–24 nodes. The v3 decision
  to reject trivial primary items (correct for mechanism isolation) removed the
  easiest ~30% and pushed accuracy further toward the floor.
* Execution moved from CPU to GPU mid-programme. The CPU partial run was
  **discarded, not merged**, and all 60 cells ran in one runtime
  (`torch 2.11.0+cu128`), which the verifier confirms. No confirmatory outcome
  existed before the switch, so it could not bias anything.
* Only the registered synthetic task families were used. Nothing here speaks to
  natural language.

---

## 7. CLAIM CEILING

> Under the registered Phase-1 v3 design, no arm learned the registered tasks
> above a constant-answer baseline, so the study is **uninformative** about
> whether finite-difference trajectory guidance adds capability. The measured
> contrasts are indistinguishable from zero and confidently exclude the +0.020
> effect the study was powered for, but this reflects a floor effect rather than
> a property of the mechanism.

Nothing stronger, in either direction.

---

## 8. NEXT PERMITTED EXPERIMENT

The blocker is **task learnability**, not the mechanism and not the statistics.
The inferential machinery, evidence chain, blinding, and falsifier all worked
exactly as designed.

A v4 study requires a new registration and must add, before any confirmatory run:

1. **A competence gate as a launch precondition.** At least one arm must exceed
   the train-majority baseline by a preregistered margin on the development split
   before the confirmatory matrix is dispatched. Cheap to check, and it would
   have prevented this entire spend.
2. **A capacity/budget pilot**, development-only, sweeping model width, depth,
   and update count until arms are demonstrably competent.
3. **Optionally an easier registered difficulty band**, or a curriculum, so the
   contrast is measured where the arms can actually perform.

The v3 statistical design (estimator, gate, 20 seeds, crossed bootstrap) should
be **retained unchanged** — it was not the failure mode, and the realised
dispersion shows it would have had ample precision.

**The real-model bake remains unauthorized.** The mechanism has still not been
given a fair test.
