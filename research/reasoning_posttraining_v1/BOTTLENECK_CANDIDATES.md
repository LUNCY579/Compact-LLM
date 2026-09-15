# Bottleneck candidates and their falsifying tests

Purpose: name specific, testable bottlenecks before building the next data mixture, so the
ladder tests a hypothesis instead of shotgunning a corpus. Required by the programme's
improvement loop: *inspect current errors -> name one bottleneck and a falsifying test*.

**Evidence class: EXPLORATORY.** Everything here comes from a 32-row TRAIN-only probe with
n=4 per family, plus one n=1 world. None of it is a measured effect, none of it supports a
quality or generalization claim, and none of these mechanisms has been shown to be the
*dominant* error source. They are hypotheses with tests attached.

Source: `qualify-001` / `train-001` receipts under
`evidence/inspection-20260915/rrp001-runs/`.

## B-1. Last-mentioned / recency shortcut in state accumulation

**The strongest single piece of evidence in this run.** One world out of 32 was never fitted
after 8 exposures and 105 updates: `RRF1-321871923ee868c43bb2`, family `signed_ledger`.

The label was verified independently before blaming the model. The prompt gives
`Opening: account StockPCKFK held 241 units` and eight POSTED deltas
`[-13, -21, +11, -9, +6, -22, -18, -30]` summing to −96, alongside six VOID entries that the
stated rules exclude. So the balance is `241 - 96 = 145`, which is exactly the stored
reference. **The gold label is correct and the instrument is sound.**

The model answered `211`. Testing candidate heuristics, `211` is reachable in exactly one way:

| hypothesis | value |
| --- | --- |
| correct: opening + all 8 POSTED deltas | 145 |
| **opening + LAST delta only (241 − 30)** | **211** |
| opening + FIRST delta only | 228 |
| opening + all deltas including VOID | 94 |
| opening only | 241 |
| opening + last *k* POSTED, k = 2..7 | 193, 171, 177, 168, 179, 158 |

No other tested rule yields 211. The model applied the opening balance and **only the final
delta**, ignoring seven intervening ones — a recency shortcut, not arithmetic noise. This is a
sustained-state-tracking failure, which is one of the programme's four named targets, and it
survived direct gradient pressure on this exact row.

It is also, precisely, one of the shortcut baselines the measurement plan requires. That
baseline is therefore not a hypothetical control: it is an observed failure mode.

**Falsifying test.** Build a `last_delta_only` shortcut scorer and run it, model-free, across
every `signed_ledger` world in DEV. Then compare the model's per-world predictions against
both the correct answer and the shortcut's answer.

- If the model's errors cluster on the shortcut's output well above chance, B-1 is real and
  the ladder should target state accumulation directly (longer required state histories,
  operator depth varied independently of distractor length).
- If model errors are spread across arithmetic near-misses with no concentration on the
  shortcut value, B-1 is refuted as a general mechanism and this was a single coincidence at
  n=1. **This is the honest default expectation, since one world cannot establish a pattern.**

Vary the number of POSTED deltas while holding record count fixed: a recency shortcut should
degrade with more deltas to accumulate, while a genuine execution failure should not care.

## B-2. Abstention miscalibration in both directions

On the 32-world TRAIN probe there are exactly **6** worlds whose reference is `UNKNOWN`.

| probe | literal `UNKNOWN` predictions |
| --- | --- |
| source (untrained) | **12** |
| midpoint | 5 |
| candidate final | **6** |

The source over-abstained roughly twofold: it answered `UNKNOWN` on computable worlds
(references `145`, `RoomLCWTB`, `Atrium`). It also under-abstained in the other direction,
answering `amber` on a `source_boundaries` world whose correct answer is `UNKNOWN`. So the
abstention decision is miscalibrated, not merely conservative.

After fitting, the count matched the true rate exactly. **On the rows it trained on**, so this
is not an abstention generalization result and must not be reported as one.

This independently echoes a prior programme finding that the earlier regression was *abstention
and terseness* rather than retrieval.

**Falsifying test.** Requires the ANSWER / UNKNOWN / IMPOSSIBLE strata that do not yet exist —
current selection quotas cover only `EXACT_PRIMITIVE`. Score the three strata separately and
never pool them. Report false-answer rate on unanswerable worlds and false-UNKNOWN rate on
answerable worlds as two distinct numbers, plus a `constant_UNKNOWN` baseline. If a candidate
improves overall accuracy while its false-answer rate on unanswerable worlds rises, the gain is
abstention collapse and must be rejected under the preservation criterion, not banked.

## B-3. Response-contract violation, distinct from reasoning

`dialogue_memory` scored **0/4** at source, and the failure is presentational: the model emitted
`Use Workshop for my ve...` where the contract wants `venue=Workshop`. It had found the right
answer and failed to express it in the required form. Eight exposures fixed it completely
(4/4 at midpoint and final), which is what one expects of a formatting convention rather than a
reasoning capability.

**Falsifying test.** Score strict format compliance and semantic correctness as *separately
observable* quantities, then re-score every wrong answer under a contract-tolerant parser. Any
family whose accuracy jumps under tolerant parsing is a presentation problem, and spending
reasoning-supervision budget on it is wasted. Conversely, do not let a tolerant parser inflate
the headline: primary accuracy stays strict.

## B-4. Execution and binding errors

Two families failed at source in ways neither B-2 nor B-3 explains:

- `program_trace` 0/4 — arithmetic/execution errors (`12` vs `1`, `3` vs `0`).
- `relational_join` 0/4 — wrong entity bound (`OrenHill` vs `MiraGlass`), i.e. retrieval or
  binding rather than computation.

**Falsifying test.** Use the oracle-evidence and extraction-then-solve diagnostic arms to split
retrieval from execution. If a model given oracle evidence still fails `program_trace`, the
bottleneck is execution; if it succeeds, the bottleneck is retrieval or binding. Note that an
extraction-then-solve arm with an external solver is a *tool-assisted system*, not intrinsic
model reasoning, and must be reported as such.

## What these four imply for the mixture

These are **four different mechanisms**, and three of them (B-2, B-3, B-1) are not reasoning
capacity at all — they are calibration, presentation and a shortcut. Chasing them with one
undifferentiated "more reasoning data" mixture would confound them: B-3 is nearly free to fix,
B-2 risks trading one error direction for the other, and B-1 needs targeted state-history
pressure rather than more exposures of the same worlds.

Before the A/B/C ladder runs, predeclare which of B-1..B-4 each arm is expected to move, and
what result would falsify that expectation. An arm that improves the aggregate while silently
trading B-2's two error directions against each other is not an improvement.

## Explicitly not claimed

- No mechanism here is shown to be dominant, or even to generalize past the probed rows.
- The 8/32 -> 31/32 rise is fitting on the trained rows. It is not evidence about any of B-1..B-4.
- B-1 rests on a single world. The stated default expectation is that it does not replicate.
- No label was altered. The one label examined was verified correct *before* the prediction was
  judged, and no benchmark label has been repaired after seeing model output.
