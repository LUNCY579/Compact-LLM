# MRS-2 — current-candidate results (cells A and B)

**OPEN DEVELOPMENT / MECHANISM SCOUT.** 96 scored CHECK cases, 120 distinct worlds in the
bank, one model call per generation, greedy, inference only.

| Claim | Status |
|---|---|
| SOFTWARE PASS | **TRUE** — 40 MRS-2 tests, 80 probe tests |
| NATIVE EXECUTION PASS | **TRUE** — both candidates, full 120-item generation |
| OPEN DEV RESULT | **TRUE** — cells A and B produced |
| MECHANISM SCOUT RESULT | **TRUE, and negative** — see the floor rule below |
| CONFIRMATORY CLAIM | **FALSE** |
| RELEASE CLAIM | **FALSE** |

Freeze `297aa8865d8330130c0dd9f4f30d0df0054e0bed553e7c81070631eeb82dc49e`, frozen before any
MRS-2 model call. Bank reused from MRS-1 by hash, not regenerated.

## Execution

| Run | Calls | Wall | Peak GPU | Peak host RSS | State | Smoke |
|---|---|---|---|---|---|---|
| `d1-seed17-bf16-export` | 1322 | 444.5 s | 2.90 GiB | 2.42 GiB | unchanged | PASS 8/8 × 9 roles |
| `supa-1.7-frozen` | 1322 | 450.6 s | 2.90 GiB | 2.36 GiB | unchanged | PASS 8/8 × 9 roles |

Zero optimizer steps, zero weight writes, zero retries, no guard weakened. Formatting failure
rate **0.000** on every arm for both models — the contract MRS-FP-1 selected holds at scale,
not just on the 8 probe items.

## The floor rule fires

The best constant answer on these 96 cases scores **0.2292** (answer prior:
0→17, 1→19, 2→20, 3→22, UNKNOWN→18).

| Arm | D1 | vs baseline | Supa | vs baseline |
|---|---|---|---|---|
| A0 direct | 0.2292 | **+0.0000** | 0.1979 | **−0.0312** |
| A1 budget-matched | 0.2292 | +0.0000 | 0.1979 | −0.0312 |
| A2 repeated direct | 0.2500 | +0.0208 | 0.1979 | −0.0312 |
| **A3 multi-reasoning** | **0.2500** | **+0.0208** | **0.1979** | **−0.0312** |
| S1 diagnostic | 0.2604 | +0.0313 | 0.1979 | −0.0312 |

D1's direct arm lands on **exactly** the constant baseline, 22/96 either way. Every Supa arm
is below it. The preregistered interpretation therefore applies:

> Any arm at or below the constant baseline → the model is not performing the task. No
> contrast among arms is interpretable as reasoning, in either direction.

This was registered before the run precisely so a floor result could not later be narrated
as a finding about reasoning diversity. Everything below is reported under that constraint.

Neither model abstains usefully: unanswerable accuracy is **0.000** for D1 and 0.056 for
Supa, so the `UNKNOWN` contract is essentially never exercised correctly.

## Answers to the registered questions

**1. Does multi-reasoning beat ordinary direct inference?** A3 − A0 = **+0.0208** on D1
(3 repairs, 1 damage, net +2 cases, McNemar p = 0.63, CI [−0.021, +0.062]) and **0.0000** on
Supa (1 repair, 1 damage). Two cases on 96, interval spanning zero, from a baseline that is
itself at the floor. No.

**2. Does it beat repeated direct attempts?** This is the primary contrast.
A3 − A2 = **0.0000 on both models.** D1: 2 repairs against 2 damage. Supa: 0 and 0. At
matched call count, structured role diversity buys exactly nothing over asking the same
question four ways.

**3. Is there real complementary correctness between roles?** Partly — and this is the one
mechanism finding that survives the floor. The role prompts *do* induce materially different
computations:

| Model | mean pairwise disagreement, diverse roles | repeated direct | ratio |
|---|---|---|---|
| D1 | 0.168 | 0.043 | 3.9× |
| Supa | 0.089 | 0.021 | 4.3× |

That fourfold gap is not noise in the prompt wording; the roles genuinely diverge. It also
lifts the analysis-only oracle ceiling on D1 from 0.250 (repeated direct) to **0.302**
(diverse roles) — about 5 points of headroom that repetition alone does not create.
But per-pair, "both wrong" dominates every cell (68–72 of 96 on D1, 75–76 on Supa), so the
diversity is mostly diversity among wrong answers.

**4. Can the real selector exploit it?** Selection is *not* the bottleneck here. D1's
selector efficiency — correct given that some candidate was correct — is **0.83** for S2 and
0.86 for S1; Supa's is 0.90. The selectors capture most of the headroom that exists. There
simply is very little.

**5. Where are the regressions?** D1 A3 damages 1 case that A0 got right while repairing 3;
against A2 it repairs 2 and damages 2. Supa A3 repairs 1 and damages 1 against A0. Damage is
real but small, and symmetric with repair.

**6. Cost per recovered case?** D1, A3 over A0: **192 extra generation calls, 384 extra
forward passes, ~53.6k extra tokens and 19.3 s per net recovered case.** Supa's net recovery
is zero, so its cost ratio is undefined and is reported as `null` rather than as a finite
number.

## Per family (n = 24 each, baseline 0.250)

| Family | D1 A0 → A3 | Supa A0 → A3 |
|---|---|---|
| `evidence_chain` | 0.125 → 0.125 | 0.125 → 0.125 |
| `counterfactual_state` | 0.167 → 0.208 | 0.250 → 0.250 |
| `prerequisite_schedule` | 0.292 → 0.333 | 0.250 → 0.250 |
| `register_machine` | 0.333 → 0.333 | 0.167 → 0.167 |

Both models are worst on `evidence_chain` at **half** the constant baseline — multi-hop
link-following with superseding records is the hardest family for both. D1 sits above
baseline on the two families whose structure most resembles its transition training
(`register_machine`, `prerequisite_schedule`), which is suggestive but rests on 2 cases per
family and is not a claim.

## The A1 control worked exactly as registered

A1 − A0 = **0 repairs, 0 damage, identical accuracy** on both models. Under a single-field
contract the model terminates after the answer regardless of allowance, so raising the cap
from 64 to 320 tokens changes literally nothing. Token-budget matching is vacuous here, as
registered in advance. No arm in this experiment is compute-matched and none is reported as
such.

## Comparability limitation

Supa and D1 share a serving policy, dtype, tokenizer and runtime, but they are **not**
interchangeable: Supa is the frozen unadapted source, D1 is transition-trained from a
different lineage. They are reported separately and never pooled.

## What this means for the next step

Cells A and B exist and are frozen. The bottleneck they identify is **baseline task
competence**, not reasoning diversity (which is real) and not selection (which is efficient).
A checkpoint that cannot beat a constant answer leaves the multi-reasoning question
unanswerable, because there is no correct-answer mass for extra computation to recover.

The operational consequence for Phase II is in
[MRS2_POSTTRAIN_CONTRACT.md](MRS2_POSTTRAIN_CONTRACT.md): cells C and D run the identical
protocol, but a **baseline-competence precondition** is checked and reported first, so that a
trained checkpoint still at the floor is recognised as such instead of generating an
uninterpretable 2 × 2.
