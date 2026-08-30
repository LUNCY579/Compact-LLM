# Phase-1 v4 development pilot — competence report (Stages 1–4)

Development-only. Nothing here reads a confirmatory bank, and **no between-arm
difference was used to select anything**. Configurations are scored by the
minimum absolute held-out accuracy across all three arms.

Grid recorded before execution. Primary axis is training-set size, because the
v3 postmortem showed pure memorization; capacity was deliberately *not* the first
lever, since more capacity on 5,000 examples memorizes better rather than
generalizing.

---

## 1. OBSERVED

Every regime trains `cascade`, `history_control`, and `learned_delta` under
identical settings on 2 development seeds, and evaluates on a held-out set drawn
from the **same band** with a different dataset seed (zero latent-signature
overlap), so difficulty is fixed and only instance novelty varies.

| regime | train ex. | steps | band | trivial | min-arm holdout | margin | mem gap | max NLL |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `v3_replica` | 5k | 2.5k | hard | 0.0413 | 0.0463 | +0.0050 | **0.1054** | **3.662** |
| `data_50k` | 50k | 2.5k | hard | 0.0413 | 0.0327 | −0.0087 | 0.0084 | 3.463 |
| `data_200k` | 200k | 2.5k | hard | 0.0407 | 0.0323 | −0.0083 | 0.0056 | 3.461 |
| `data_200k_long` | 200k | 10k | hard | 0.0407 | 0.0477 | +0.0070 | 0.0013 | 3.416 |
| `easy_5k` | 5k | 2.5k | easy | 0.0400 | 0.0940 | +0.0540 | **0.1747** | 3.323 |
| **`easy_50k`** | **50k** | **2.5k** | **easy** | **0.0400** | **0.0937** | **+0.0537** | **0.0124** | **3.182** |

Uniform-answer NLL reference: ln(32) = **3.466**.

Per-arm held-out accuracy for `easy_50k`: `cascade` 0.0937,
`history_control` 0.0937, `learned_delta` 0.1063.

`OBSERVED` **`v3_replica` reproduces the v3 pathology** — margin +0.005 with a
memorization gap of 0.105 and NLL *worse* than uniform. The harness reproduces
the failure it is meant to diagnose, which is what makes the other rows
trustworthy.

`OBSERVED` **More data alone removes memorization but does not create
competence.** `data_50k` and `data_200k` collapse the memorization gap from
0.105 to 0.008/0.006 — and land *below* the trivial baseline, at exactly uniform
NLL. The model stops memorizing and learns nothing instead.

`OBSERVED` **More data plus more updates barely clears trivial on the hard
band.** `data_200k_long` reaches +0.0070 over trivial with NLL 3.416. Real but
negligible: 4.8% absolute on a 1-of-32 task.

`OBSERVED` **The easy band is genuinely learnable.** `easy_5k` reaches 9.4%
against a 4.0% baseline (+5.4 points, 2.3×) — but with a 0.175 memorization gap.

`OBSERVED` **`easy_50k` is the first regime with real generalization.** Same
accuracy as `easy_5k` (9.4%) but the memorization gap collapses to 0.012 and NLL
improves to 3.182, comfortably below uniform. The competence is now earned on
unseen instances rather than recalled.

### 1.1 Incomplete regimes — stated plainly

`OBSERVED` Three of the nine planned regimes did **not** produce a result:

* `easy_50k_long` (50k × 10k steps) — CUDA OOM, twice;
* `easy_50k_wide` (192-dim, 4-layer) — not reached;
* `medium_50k_long` — still running at the time of writing.

These were the regimes designed to answer whether absolute competence rises
above ~9% with more updates or more capacity. **That question is therefore
open**, and it is the reason the final state below is not "ready".

---

## 2. SUPPORTED INTERPRETATION

`SUPPORTED` **Competence is achievable, and the lever is task difficulty
combined with data volume — not capacity.** The 258k-parameter model that failed
completely on the hard band reaches 2.3× the trivial baseline on the easy band
with the same parameter count.

`SUPPORTED` **The v3 hard band is beyond this model.** Even 200k examples and 4×
the updates yield only +0.007 over trivial. Choosing the v3 difficulty band was
the design error; the model was never the problem.

`SUPPORTED` **Data volume and difficulty must move together.** On the hard band,
more data traded memorization for chance-level performance. On the easy band, it
traded memorization for genuine generalization at the same accuracy. Data alone
is not sufficient — it converts a memorizer into either a learner or a guesser
depending on whether the task is within reach.

`HYPOTHESIS` The `easy_nontrivial` band at 50k examples with more updates, or a
wider model, would push absolute competence above 9.4%. Untested — the regimes
that would have answered it did not complete.

---

## 3. UNSUPPORTED CLAIMS

* That 9.4% is *sufficient* absolute competence for a robust mechanism contrast.
  It clears every trivial baseline, but the dynamic range above the floor is
  ~5 points, which is modest.
* That `learned_delta`'s slightly higher `easy_50k` number (0.1063 vs 0.0937)
  means anything. It was not used for selection and must not be read as a
  mechanism signal — 2 seeds, development data, no interval.
* Anything about the TGSA mechanism.

---

## 4. ANOMALIES

`OBSERVED` `data_50k` and `data_200k` land **below** the trivial baseline
(−0.009, −0.008) with NLL at exactly uniform. The models converged to
approximately uniform prediction over the answer space — a "learned to guess"
state, which is a coherent optimum when the task is out of reach and
memorization is unavailable.

`OBSERVED` Two CUDA OOM failures in a single-process pilot loop as allocator
fragmentation accumulated across regimes. Infrastructure, not science; the fix
is a fresh process per regime.

---

## 5. LIMITATIONS

* 2 development seeds per arm per regime. Adequate for a floor/competence
  screen, not for precision.
* 6 of 9 planned regimes completed.
* Only three difficulty bands were defined, and the gap between `easy_nontrivial`
  (2 hops, 5–8 nodes) and `hard` (2–4 hops, 10–20 nodes) is large. The `medium`
  band result is pending.

---

## 6. RECOMMENDED CONFIGURATION (provisional)

Smallest configuration meeting arm-blind competence, on current evidence:

| field | value |
|---|---|
| band | `easy_nontrivial` — pointer 2 hops / 5–8 nodes, state 2–3 steps, trivial items rejected |
| train examples | 50,000 |
| updates | ≥2,500 (10k untested) |
| embed_dim / layers / heads | 96 / 2 / 6 (unchanged from v3) |
| batch / LR | 64 / 1e-3 with warmup + cosine |
| parameters | ~255k |

**Provisional**, because the regimes that would confirm whether a longer or
wider run does materially better did not complete.

---

## 7. COMPETENCE GATE THRESHOLDS (derived, to be frozen)

Derived from this pilot, outcome-blind, before any v4 confirmatory run:

| check | threshold | rationale |
|---|---|---|
| margin over strongest trivial baseline | ≥ **+0.030** | `easy_50k` achieves +0.054; `v3_replica` scored +0.005. Sits well above the failure and below the demonstrated pass. |
| absolute held-out accuracy, weakest arm | ≥ **0.070** | `easy_50k` weakest arm is 0.0937; v3 was 0.045. |
| answer NLL | < **0.97 × ln(32)** = 3.362 | `easy_50k` reaches 3.182; v3 was 4.03 and `data_200k` 3.461. Excludes the "learned to guess" state. |
| memorization gap (seen − holdout) | ≤ **0.10** | v3 was 0.105, `easy_5k` 0.175, `easy_50k` 0.012. |
| distinct predictions | ≥ 4 | rejects a constant predictor. |
| per-family accuracy | above the same floor, and ≤ 0.95 | v3's `state_update` was below chance while `pointer_chase` was near baseline. |

Every threshold is placed strictly between a demonstrated failure and a
demonstrated pass, so the gate is calibrated against observed behaviour rather
than chosen to be permissive.

---

## 8. CLAIM CEILING

> A development pilot demonstrates that all three registered arms can reach
> ~2.3× the strongest trivial baseline with a collapsed memorization gap and
> answer-NLL below uniform, on an `easy_nontrivial` difficulty band with 50,000
> training examples. The v3 regime's failure is attributable to band difficulty
> combined with training-set size, not to model capacity or optimization.

Nothing here bears on the TGSA mechanism.

---

## 9. NEXT PERMITTED EXPERIMENT

1. Complete `easy_50k_long`, `easy_50k_wide`, and `medium_50k_long`, one process
   per regime, to establish whether absolute competence exceeds ~9.4%.
2. Prefer the smallest regime whose weakest arm clears the frozen thresholds
   with margin, ideally at a higher absolute accuracy for better dynamic range.
3. Freeze the competence gate, run it, and dispatch v4 **only** on `GO`.
