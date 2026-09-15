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

## B-1. Cross-world memorization interference induced *by* the microfit

**The most interesting result in this run, and not what it first looked like.** One world out
of 32 was never fitted after 8 exposures and 105 updates: `RRF1-321871923ee868c43bb2`, family
`signed_ledger`, gold `145`, predicted `211`.

The label was verified independently before blaming the model. The prompt gives
`Opening: account StockPCKFK held 241 units` and eight POSTED deltas
`[-13, -21, +11, -9, +6, -22, -18, -30]` summing to −96, alongside six VOID entries that the
stated rules exclude. So the balance is `241 - 96 = 145`, exactly the stored reference.
**The gold label is correct and the instrument is sound.**

### A first reading that was wrong

`241 - 30 = 211`, so the error looked like a clean "opening + last delta only" recency
shortcut, and no other arithmetic rule tested reproduced 211 (first delta 228, all deltas
including VOID 94, opening alone 241, last *k* deltas for k=2..7 giving 193/171/177/168/179/158).
That reading was recorded here and **it is refuted below.** It is kept visible rather than
deleted, because the way it failed is the lesson.

### The discriminating evidence

A different world in the same 32-row set, `RRF1-7158b22893bb3c80418b`, is also `signed_ledger`
and its gold answer **is 211** (`opening 233 + posted sum −22`). The two prompts are
structurally near-identical, differing mainly in the account name and the record list. So `211`
is simultaneously reachable by a recency shortcut *and* the memorized answer of a sibling world.
Those two hypotheses are separable, because the untrained model could not have memorized
anything:

| probe | failing world (gold 145) | sibling world (gold 211) |
| --- | --- | --- |
| source baseline, 0 updates | `UNKNOWN` | `UNKNOWN` |
| midpoint | **`211`** | `211` (correct) |
| final | **`211`** | `211` (correct) |

All four `signed_ledger` worlds, by probe:

| probe | (gold, predicted) |
| --- | --- |
| baseline | (145, UNKNOWN) (UNKNOWN, UNKNOWN) (211, UNKNOWN) (411, UNKNOWN) |
| midpoint | (145, **211**) (UNKNOWN, UNKNOWN) (211, 211) (411, 411) |
| final | (145, **211**) (UNKNOWN, UNKNOWN) (211, 211) (411, 411) |

**The source did not compute `241 − 30`; it abstained.** `211` appears only after training. The
recency-shortcut arithmetic was a coincidence: `241 − 30` and `233 − 22` both happen to equal
211. The mechanism actually evidenced is **cross-world answer collision within a family,
induced by fitting** — the model drove training CE to 0.0000, memorized the set, and one world
inherited its structural sibling's answer.

### Why this matters more than the original reading

The headline `31/32` conceals an interference failure rather than a residual reasoning gap. At
just 32 worlds and 8 exposures, one within-family collision already appears. That is a
statement about memorization capacity and inter-world discriminability at this scale, and it
bears directly on the plan to scale exposures and unique worlds: **more exposures of
structurally similar worlds may buy training accuracy while increasing collision risk.**

**Falsifying test.** Score every DEV world against (a) its own gold and (b) the gold of every
*other* world in the same family, and report a collision rate: the fraction of errors whose
prediction exactly equals a sibling world's gold answer.

- If the collision rate is at or near the chance level of matching some sibling answer, B-1 is
  refuted and this was one coincidence at n=1. **That remains the honest default expectation.**
- If errors concentrate on sibling golds well above that chance level, B-1 is real, and the
  ladder must monitor interference — because the fix (more unique worlds, fewer exposures,
  greater inter-world separation) is the *opposite* of what a reasoning-capacity reading would
  prescribe.

Report the chance level explicitly: with 4 worlds per family and few distinct numeric answers,
matching a sibling by accident is not rare, and this single instance does not clear that bar.

Also keep the `last_delta_only` shortcut scorer. It is cheap, it is a required baseline anyway,
and running both discriminates the two mechanisms on every future world instead of one.

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
capacity at all — they are calibration, presentation and memorization interference. Chasing them with one
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
- B-1's first reading (a recency shortcut) was recorded and then refuted by the source-baseline
  probe. It is left visible above, with the evidence that killed it.
- No label was altered. The one label examined was verified correct *before* the prediction was
  judged, and no benchmark label has been repaired after seeing model output.
