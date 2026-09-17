# MRS-2 — preregistration

**Frozen before any MRS-2 model call. OPEN DEVELOPMENT / MECHANISM SCOUT.**
**CONFIRMATORY CLAIM: false. RELEASE CLAIM: false.**

## Questions

1. Does structured multi-reasoning (A3) beat ordinary direct inference (A0)?
2. Does it beat repeated direct attempts at the same call count (A2)? — **primary**
3. Do the different roles produce correct answers on *different* cases?
4. Can the deployable selector exploit that complementarity?
5. Where does the treatment damage cases the baseline got right?
6. What does a recovered case cost?

## Candidates

| Role | Identity | Weights |
|---|---|---|
| Primary comparator | `d1-seed17-bf16-export` | `lc-D1-hf`, 1 192 135 064 bytes |
| Source reference | `supa-1.7-frozen` | `Supa`, 1 192 135 096 bytes |

Both are registered in the MRS-1 `MODEL_MANIFEST.json`, both are FP32/eager/greedy under
`plain_dense_greedy_no_bias`, and both already passed fresh native qualification on this
runner's loader on 2026-09-17. Inference only: zero optimizer steps, zero weight writes,
state digest verified unchanged.

These two do **not** have identical provenance and are not interchangeable. Supa is the
frozen unadapted source; D1 is transition-trained. They are reported separately.

## Registered outcomes

**Primary:** exact answer correctness on the 96 CHECK cases, per arm, paired by case.

**Primary contrast:** `A3 - A2` on `d1-seed17-bf16-export`.
**Practical contrast:** `A3 - A0`.

Also registered: per-family and per-difficulty accuracy; answerable and unanswerable
accuracy; false-abstention rate; unsupported-answer rate; formatting failure; nontermination;
oracle candidate ceiling (analysis only); selector efficiency
(`correct / cases where any candidate was correct`); repairs; damage; net gain; role
disagreement; unique candidate answers; pairwise role complementarity; state validity for the
three stateful families; tokens, calls, wall time, peak GPU, peak host RSS; cost per gross
repair and per net recovered case.

**Registered alongside every one of those: the constant-answer baseline** for the same
stratum, and per-role degeneracy. This is a reporting requirement, not a threshold.

## Interpretations, fixed in advance

| Observation | Registered reading |
|---|---|
| A3 ≈ A2 and role errors strongly overlap | No evidence that differentiated computations add useful diversity in this regime. |
| Oracle ceiling rises but the real selector does not | Candidate diversity exists; selection/verification is the bottleneck. |
| A3 > A2 at a similar call count | Evidence that structured strategy diversity helps this model/task regime. |
| A3 repairs many A0 errors but also breaks many A0 successes | Added computation is unstable; investigate gating/halting, not always-on use. |
| **Any arm at or below the constant baseline** | **The model is not performing the task. No contrast among arms is interpretable as reasoning, in either direction, and no multi-reasoning conclusion may be drawn from this run.** |
| Roles are constant (degeneracy flag set) | Complementarity statistics are vacuous; report them as undefined rather than as evidence of correlated reasoning. |

That fifth row is the one MRS-FP-1 makes likely. It is registered **now**, before the run, so
that a floor result cannot later be narrated as a finding about reasoning diversity.

No significance claim is attached beyond descriptive exploratory McNemar and descriptive
clustered intervals. The scout is small by design.

## What is forbidden

No prompt, parser, selector, budget or decoding change after any MRS-2 model result is
observed. No prompt repair on smoke failure. No selector tuning on CAL, CHECK, or future C/D.
No regeneration of the bank. No reopening of MRS-1. No reading of sealed, ANY_READ,
quarantined or confirmation material. No training. No weakening of a resource guard. No
promotion of an open-development result to confirmation or release.

Malformed outputs are recorded, never repaired. MRS-2 registers **no** deterministic parser
repair of any kind, so a nonzero repair count is a defect.

## Future cells C and D

The same frozen protocol, bank bytes, role prompts, selectors, parser, decoding policy,
budgets and scorer apply to a future trained checkpoint. It enters only through a hash-bound
registration carrying passing P0 dataset-acceptance, P1 training-contract and P2
training-execution receipts, and it must differ from D1 in weight bytes and path while
matching config, tokenizer, policy and runtime. Any incompatibility creates a new experiment
rather than a silent amendment.

Cells C and D produce, against the preserved A and B:

| Quantity | Paired expression |
|---|---|
| Training under ordinary inference | C − A |
| Multi-reasoning before training | B − A |
| Multi-reasoning after training | D − C |
| Interaction | (D − C) − (B − A) |

The analysis computing these, all 16 correctness patterns and every event mask is implemented
and fixture-tested **now**, before any trained candidate exists.
