# SPARSE-001 — the forgetting lives in the update's tail, and removing it beats shrinking it

**Three arms, measured twice (once with a non-deterministic tie-break, once deterministically
after Erratum 1), no training run, one seed. Registered in `SPARSE001_PREREG.md` before any
arm was built. Five of seven predictions held. Report-only on long context and abstention;
nothing promoted.**

## The hypothesis was confirmed

θ(k) = θ_supa + M_k ⊙ δ, keeping the largest `k` share of |δ| per tensor, FP32 throughout.

The screen existed to separate two operations that INT-001 could not. α-scaling shrinks
**every** element, including the head that carries the new skill. Magnitude pruning removes
only the tail. At the same nominal 0.50 they diverge sharply:

| operation at nominal 0.50 | acquisition /384 | arith | retained /47 | long ctx | `UNKNOWN` |
|---|---:|---:|---:|---:|---:|
| α = 0.50 — shrink all elements by half | 219 | 67 | **42** | **164** | 12 |
| k = 0.50 — keep top half at full strength | **235** | 68 | 39 | 159 | 9 |

**Pruning retains +16 acquisition.** Registered prediction 1 — "k = 0.50 acquisition ≥ 219" —
held decisively at 235. The capability is carried by the update's large-magnitude head, and
the long-context damage by its tail.

The delta is strongly heavy-tailed, which is why this works at all:

| k | realised kept elements | retained fraction of ‖δ‖ |
|---|---:|---:|
| 0.10 | 0.0999996 | **0.6383** |
| 0.25 | 0.2500000 | **0.8364** |
| 0.50 | 0.5000000 | **0.9595** |

A quarter of the elements carry 84% of the update's norm.

## The frontier, and the leading candidate

| model | acq /384 | floors | arith | strict | retained /47 | long ctx | paired p | `UNKNOWN` | Gate v2 |
|---|---:|:--:|---:|---:|---:|---:|---:|---:|:--:|
| Supa 1.7 | 14 | 0/3 | 47 | 41 | 47 | 161 | — | 13 | — |
| Titan (published BF16) | 234 | 3/3 | 69 | 62 | 39 | 162 | 1.00 | 10 | PASS |
| α = 0.50 *(INT-001)* | 219 | 3/3 | 67 | 59 | **42** | **164** | 0.61 | 12 | PASS |
| **k = 0.25** | **232** | 3/3 | **72** | 63 | 41 | **164** | 0.63 | 12 | **PASS** |
| α = 1.00 FP32 | 235 | 3/3 | 71 | **64** | 40 | 157 | 0.57 | 9 | PASS |
| k = 0.50 | 235 | 3/3 | 68 | 61 | 39 | 159 | 0.85 | 9 | PASS |
| k = 0.10 | 197 | **2/3** | 61 | 54 | 42 | 163 | 0.79 | 12 | FAIL |

**k = 0.25 essentially dominates the published core.** Acquisition equal within noise
(232 vs 234), and **+3 arithmetic, +2 source answers retained, +2 long-context rows, +2
abstentions**. Against INT-001's α = 0.50 winner it trades one retained answer for +13
acquisition and +5 arithmetic while tying on long context and abstention.

By the registered rule — among floor-clearing arms, maximise `references_retained` subject to
`arithmetic_permissive >= 62` — **k = 0.25 wins this screen** (retained 41 against k = 0.50's
39; k = 0.10 is excluded for failing a floor).

Candidate weights are retained and hash-verified:
`a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284`, reproducible from source
`962abc5b…` plus trained tensors `07156fb5…` at k = 0.25.

## The tie-break has zero effect — an unplanned robustness control

Erratum 1 forced a full re-measurement from deterministic builds. The two passes are
**identical on every endpoint of every arm**: acquisition, floors, both arithmetic policies,
retained, long context, abstention, and the per-row lost/gained counts.

That converts a defect into a useful control. Which of several tied-magnitude elements the
mask keeps does not affect any measured outcome, so the arm is robust to the mask's
under-determined degrees of freedom. The numbers now also bind to reproducible hashes:
k = 0.10 `d7b15d2e…`, k = 0.25 `a28629c9…`, k = 0.50 `3f6b590b…`.

## Predictions versus outcome

| # | prediction | outcome |
|---|---|---|
| 1 | k = 0.50 acquisition ≥ 219 (the hypothesis's main consequence) | **held**, decisively: 235 |
| 2 | acquisition monotone non-decreasing in k | **held**: 197, 232, 235 |
| 3 | long context non-increasing in k, all within 155–166 | **falsified** on monotonicity: 163, **164**, 159. Range held. |
| 4 | k = 0.25 within ±2 rows of the BF16 export's 162 | **held**, exactly +2: 164 |
| 5 | some arm clears all floors while retaining ≥ 43 | **falsified**: best floor-clearing arm retains 41 |
| 6 | k = 0.10 fails at least one family floor | **held**: 2/3, `ledger_replay` lower bound short |
| 7 | abstention non-increasing in k, within 9–13 | **held**: 12, 12, 9 |

Prediction 5 has now failed in both screens. The retained ≥ 43 *and* all-floors-cleared
corner does not appear to exist on either the α or the k line — the frontier is real and
slightly tighter than hoped.

## What holds across both screens

1. **Long context peaks away from both endpoints**, at 163–164, above the frozen source's
   161, on both the α line and the k line. Every arm is statistically indistinguishable from
   Supa (p = 0.57–0.85), so these are ordered point estimates and not demonstrated
   preservation.
2. **`ledger_replay` is the binding family** in every arm of both screens: lowest rate
   everywhere, and the only family that ever fails. It carries the strongest shortcut floor
   and it is the family D0 failed.
3. **The legacy per-class retention monitor FAILS every arm of both screens**, including
   k = 0.10 and α = 0.25, which lose only 5 and 4 of 47 source answers respectively. A
   monitor that rejects models that barely moved is not discriminating among these
   candidates, and the disagreement with Gate v2 needs a registered resolution before
   anything is selected on either.

## What this does and does not license

**Does.** Magnitude pruning at k = 0.25 is the best-behaved model this programme has
produced, it costs no training run, and it is reproducible from two hashed inputs. It is the
arm to carry forward.

**Does not.** One training draw, one update direction, one seed. Long context and abstention
were **report-only** and did **not** select any arm — the winner was chosen on
SELECT-VIEWS-001 and the arithmetic/retention panel alone, which is why 164/256 is reported
and not claimed. Gate v2 PASS coexists with a legacy monitor FAIL.
`quality_improvement_established` stays false, `release_permitted` stays false, and **no arm
was promoted**: both screens registered that they promote nothing.

## What should happen next

1. **Promote k = 0.25 under its own registration**, or decline to. The selection rule it won
   was registered in advance and it is satisfied, so a promotion registration would not be
   fitting a rule to a result — but it is an operator identity decision, as Titan's was, and
   it is not taken here.
2. **Independent training draws are now the only thing that can strengthen any of this.**
   Every number in INT-001 and SPARSE-001 descends from a single seed, and neither
   interpolation nor pruning can manufacture replication. Two more draws of the RRP-015
   control recipe, each then pruned at k = 0.25, is the honest path to a reliable claim.
3. **Combine the two levers** — prune at k = 0.25 then scale the survivors — in a registered
   2-D screen. Both operations independently helped; they may compose. This is the cheapest
   remaining unexplored direction and still needs no training.
4. **Resolve the two retention gates**, which now have four screens' worth of evidence that
   the legacy monitor is near-unpassable.
5. **The confirmation instrument still does not exist.** 164/256 cannot be confirmed on a
   spent suite, only observed. That remains the binding blocker on every claim here.
