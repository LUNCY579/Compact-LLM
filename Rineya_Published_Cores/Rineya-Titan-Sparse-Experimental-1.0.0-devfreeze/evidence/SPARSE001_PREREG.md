# SPARSE-001 — does the update's small-magnitude tail carry the forgetting?

**Written before any arm is built or measured. Exploratory development screen, no training
run, one seed. No promotion, no release, no confirmation claim follows.**

## The hypothesis, and where it came from

INT-001 established two things that together make a sharp, testable claim:

1. The BF16 export of the RRP-015 control erases **60.61%** of the update's elements and 24
   of its 220 trained tensors, and that lossy export scores **better** on the preserved
   capabilities than its own full-precision parent: long context **162 vs 157**, literal
   `UNKNOWN` **10 vs 9**, with acquisition unchanged (234 vs 235).
2. Uniformly halving the whole update (α = 0.50) also improves preservation — long context
   **164**, retained **42/47**, `UNKNOWN` 12 — for 15 acquisition points.

**Hypothesis.** The update's **small-magnitude tail carries the long-context and abstention
damage, while the new capability is carried by its large-magnitude head.** If true, removing
the tail should be strictly better than shrinking everything: it should preserve more
acquisition than α = 0.50 does, at comparable or better retention.

α-scaling and magnitude-pruning are different operations. α shrinks every element, including
the head that carries the skill. Pruning removes only the tail. INT-001 could not separate
them; this screen does.

## Design

Let `δ = θ_ctrl_fp32 − θ_supa`, over the 220 trained tensors only. For a keep-fraction `k`,
let `M_k` be the **per-tensor** mask selecting the largest `k` share of `|δ|` entries in each
tensor, ties broken by flat index ascending so the mask is deterministic.

```
θ(k) = θ_supa + M_k ⊙ δ
```

Per-tensor rather than global, because a global threshold would let the uniform per-block
norms decide allocation implicitly; per-tensor keeps the same share everywhere and makes `k`
mean one thing. FP32 throughout, never routed through BF16, for the reason INT-001 measured.

`k = 1.0` is the unrounded α = 1 model, already measured — **not re-run** (acquisition 235,
arith 71/64, retained 40, long context 157, `UNKNOWN` 9).

### Arms, fixed now

| arm | k (kept share of |δ| per tensor) |
|---|---|
| **S10** | 0.10 |
| **S25** | 0.25 |
| **S50** | 0.50 |

**The grid is fixed at these three.** The BF16 export keeps ~39.4% of elements and scores
162, so S25 and S50 bracket it; S10 tests whether the capability survives aggressive
pruning. Any refinement after seeing results requires its own registration written first.

Note `k` and α are not comparable as numbers: α = 0.5 halves 100% of elements, k = 0.5 keeps
half the elements at full strength. Comparing them is the point of the screen.

## Instruments, permissions, and the selection rule

Identical to INT-001, and the bars are the same.

| instrument | role | permission |
|---|---|---|
| SELECT-VIEWS-001, 384 views | acquisition | **SELECTION-ADMISSIBLE**; floors in its own `FLOORS.json`, published pre-hoc: `ledger_replay` 0.2344, `prerequisite_plan` 0.0781, `register_machine` 0.0703 |
| 150-row arithmetic panel + 47 source answers | retention | the gate's own instrument |
| Long-Context Demonstration Suite, 256 rows | long context | **SPENT**, barred TRAIN/SELECTION/CONFIRMATION — **REPORT-ONLY** |
| 64 unanswerable rows inside that suite | literal abstention | same bank, same bar — **REPORT-ONLY**, and not independent of the long-context score |

> **Selection rule.** Among arms whose acquisition clears **all three** family floors on
> SELECT-VIEWS-001 by the published rule — exact two-sided binomial 95% **lower** bound above
> the stronger of `copy_before` and `constant_majority` — prefer the arm maximising
> `references_retained` subject to `arithmetic_permissive >= 62`. Ties broken by higher
> acquisition. Report strict and permissive arithmetic separately; never pool them.

**Long context and abstention may not select an arm.** They are reported for every arm so the
damage the gate cannot see stays visible, which is the RRP-010 lesson. Selecting on a `SPENT`
suite would be the post-hoc gate its quarantine bar exists to prevent.

## Registered predictions

1. **S50 acquisition ≥ 219** — i.e. at least α = 0.50's, since pruning preserves the head
   that α shrinks. This is the hypothesis's main consequence.
2. **Acquisition is monotone non-decreasing in k** across S10 → S25 → S50.
3. **Long context is non-increasing in k**, and every arm lands in 155–166.
4. **S25 lands within ±2 rows of the BF16 export's 162 on long context**, since it keeps a
   similar share of elements.
5. **At least one arm clears all three floors while retaining ≥ 43 of 47** — the target
   INT-001 failed to reach (its best floor-clearing arm retained 42).
6. **S10 fails at least one family floor.** `ledger_replay` is the binding family and failed
   at α = 0.25.
7. **Literal abstention is non-increasing in k**, within 9–13.

If prediction 1 fails, the hypothesis is wrong: the capability is carried by the whole
update, not its head, and magnitude pruning buys nothing that α-scaling did not.

## Execution

Same harness, same matched conditions as INT-001 and RRP-015, and the same stop rules:
implementation digest `a329eb7bb0`, transformers 4.51.3, `max_context` 4096, TF32 off,
`float32_matmul_precision=highest`, greedy, 0 rows skipped for overflow. Any arm whose
metadata differs is reported non-comparable rather than compared. Sidecars byte-identical to
source and a `MATERIALIZATION.json` receipt per arm, because the retention probe requires
both. One model job at a time against the 15.22 GiB host RAM limit. Nothing written inside a
git worktree. A failed arm is reported, never silently re-rolled.

Each arm also records the realised sparsity actually achieved — kept elements, and the share
of `||δ||` retained — because a nominal `k` and a realised one are not the same number and
INT-001 already showed nominal-versus-realised gaps matter.

## Erratum 1 — the first pass's tie-break was not deterministic, and its arms are not reproducible

**Recorded after the first pass's three arms were measured, before any deterministic number
was read. The defect is in the implementation, not in the registration.**

The design above specifies "ties broken by flat index ascending, so the mask is
deterministic". The implementation used `torch.topk(mag, k, largest=True, sorted=False)`,
which does **not** define which of several equal magnitudes it keeps. This delta contains
many exact ties — unsurprising, since it descends from BF16-stored weights — so the mask is
under-determined.

Detected by rebuilding the k = 0.25 arm and comparing hashes:

| build | weight sha256 | realised kept elements | retained `||δ||` fraction |
|---|---|---:|---:|
| measured first pass | `cde1502a…` | 0.25 | 0.8363817229141474 |
| rebuild, same code | `2f57fe72…` | 0.25 | 0.8363817229141474 |

Identical retained norm to sixteen digits, different bytes — the signature of swapping which
tied positions are kept. The fix replaces `topk` with
`torch.argsort(mag, descending=True, stable=True)`, whose stable order breaks ties by
ascending original index, which is what this document specified. Two independent builds then
agree exactly (`a28629c9…`, twice).

**Consequences, stated rather than smoothed over.**

1. The first pass's measured arms are valid members of the family "keep the top k share by
   magnitude, ties arbitrary", and their scores are real measurements. But the driver deleted
   each arm's weights after scoring to reclaim disk, and those exact bytes **cannot be
   reproduced**. Numbers that cannot be bound to a hash are weaker evidence.
2. Therefore all three arms are **re-measured from deterministic builds** and reported under
   labels `s10d`, `s25d`, `s50d`. The first pass's numbers are retained and reported
   alongside rather than discarded.
3. This yields an unplanned but useful control: if the two passes agree closely, the tie-break
   genuinely does not matter and the arm is robust to it. If they diverge, that is itself a
   finding about how fragile a magnitude-pruned arm is, and it would undercut the screen's
   headline. Either way the comparison is reported.
4. Deleting arm weights before their provenance was verified was the operational mistake that
   turned a fixable bug into a lost artifact. The leading candidate's weights are retained in
   the re-run.

## What this cannot establish

One seed, one training run, one update direction. It cannot establish a reliable rate, cannot
confirm anything on a spent suite, and does not substitute for independent training draws or
the fresh confirmation instrument that remain the real blockers.
`quality_improvement_established` stays false; `release_permitted` stays false.
