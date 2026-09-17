# MRS-2 — post-training contract for cells C and D

**Frozen alongside the MRS-2 instrument, before any trained candidate exists.**
Supersedes MRS-1's posttrain contract only because MRS-1 is unrunnable; every constraint
MRS-1 imposed is carried forward unless the single-field contract made it impossible.

Cells A and B are **already produced** and are immutable:

|  | ordinary inference | multi-reasoning |
|---|---|---|
| `d1-seed17-bf16-export` | **A = 0.2292** | **B = 0.2500** |
| future trained checkpoint | C — not produced | D — not produced |

`supa-1.7-frozen` is retained as a separately labelled source reference (A = B = 0.1979),
not as the paired comparator.

## Binding requirements when the trained candidate arrives

1. **Same bank.** The MRS-1 frozen bank bytes, reused by hash. Never regenerated, never
   extended, never resampled — not even if the new checkpoint trained on similar material.
2. **Same everything else.** Identical role prompts, selectors, parser, decoding policy,
   chat template, token and output budgets, call budget, item order, scorer and output
   contract. No prompt may be rewritten because the new model "doesn't like it."
3. **Both cells.** Ordinary (C) and multi-reasoning (D) are both run. The study is never
   reduced to old-model-versus-new-model.
4. **Hash-bound entry.** The candidate enters only through a registration carrying passing,
   hash-bound P0 dataset-acceptance, P1 training-contract and P2 training-execution receipts,
   and must differ from D1 in weight bytes and path while matching config, tokenizer, policy
   and runtime. `runner.candidate()` enforces this; it is not a documentation request.
5. **No selector tuning** on CAL, CHECK, or on C/D outcomes.
6. **Incompatibility creates a new experiment**, never a silent amendment. Loading a BF16
   export into FP32 does not recreate lost precision and does not make two artifacts one.
7. **Gold enters only after generation.** The oracle ceiling remains analysis-only and is
   never reported as a treatment score.

## The precondition this run adds

MRS-2 measured both existing candidates at or below the constant-answer baseline. That makes
one addition necessary, and it is registered here **before** any trained checkpoint exists so
it cannot be chosen to suit a result:

> **Baseline-competence precondition.** Before interpreting C or D, report cell C against the
> constant-answer baseline for the same 96 CHECK cases. If C does not exceed that baseline,
> the 2 × 2 is reported as **UNINTERPRETABLE AS REASONING** and the deltas are published as
> descriptive numbers only, with no reading of training effect, multi-reasoning effect or
> interaction.

This is a **reporting gate, not a pass/fail gate on the model, and not a stopping rule.** C
and D are still run and still published in full. Nothing is hidden, discarded or retried.
The gate governs only what may be *concluded*.

Rationale: `interaction = (D − C) − (B − A)` is arithmetic that will produce a number from
any four accuracies, including four floor accuracies. A number computed from noise around a
constant baseline is not evidence about reasoning, and this contract refuses to let it be
presented as such.

## Quantities to be reported

| Quantity | Paired expression |
|---|---|
| Training under ordinary inference | C − A |
| Multi-reasoning before training | B − A |
| Multi-reasoning after training | D − C |
| Interaction | (D − C) − (B − A) |

Plus all 16 boolean correctness patterns; training-only repair; multi-only repair; both
available; post-training multi extra; multi lost after training; harm from training; harm
from multi pre- and post-training; and the cases where training makes multi-reasoning newly
useful, unnecessary or harmful. Per family and per difficulty. Event masks may overlap and
are not forced into an exclusive partition.

All of this is **already implemented and fixture-tested** in
`experiments/multi_reasoning_scout_v1/analysis.py::factorial`, reused unchanged by
`experiments/multi_reasoning_scout_v2/analyze.py::future_2x2`, and verified by hand against a
four-item fixture. It was written before any trained candidate existed, so it cannot be
designed around C and D.

## Complementarity after training

Recompute, for the trained candidate, everything MRS-2 reports for D1: role-wise correctness,
every role pair's both-correct / first-only / second-only / both-wrong counts, disagreement,
unique candidates, oracle ceiling, selector accuracy and efficiency, correct-candidate
rejection, repair and damage, formatting and nontermination, abstention behaviour, and actual
token / call / latency / memory cost including cost per recovered case.

Then answer descriptively: did training raise all roles together, change which cases they get
right, make their errors more or less correlated, improve selection, remove the need for
extra routes, or make added computation more useful?

**Do not assume the perspectives that diverged on D1 still diverge after training.** MRS-2
found diverse-role disagreement roughly 4× repeated-direct disagreement on both current
checkpoints; whether that survives new training is exactly the open question, and it must be
measured, not inherited.

## Decision rules, fixed now

| Observed | Reading |
|---|---|
| C ≫ A, D ≈ C, B > A | Training absorbed what the scaffold supplied; prioritise trained ordinary inference. |
| B − A ≈ D − C, both positive | Approximately additive; assess combined-system cost. |
| C > A and (D − C) > (B − A) | Training made the model better able to exploit extra reasoning computation — the synergy case, and the one that would justify investigating native adaptive computation. |
| B > A but D ≤ C | Scaffold transfer is unstable; retain the regression, do not assume scaffolds carry over. |
| Oracle ceiling rises, selector does not | Candidate generation has headroom; verification is the bottleneck. |
| Roles stay correlated and A3 ≈ A2 | Weak evidence for multi-perspective reasoning; do not internalise unnecessary architecture. |
| **C at or below the constant baseline** | **Uninterpretable as reasoning. Report and stop concluding.** |

"Clearly", "approximately" and "similar" are descriptive summaries reported with raw counts
and the registered intervals. They are not new thresholds.

## Internalisation

No architecture-level change is justified by a prompt protocol alone, and none is justified
by a positive B or D by itself. Identify *which* registered computation supplies complementary
correct answers or reliable correction first, then consider the smallest mechanism that could
internalise that one operation — a compact verifier, an alternative low-rank query read, one
bounded refinement pass, a small learned selector, or a conditional compute gate — relating
it to the existing query-refinement, A06, recurrence, looped-flow, matched-MLP, EAM, CED and
XSG results and preserving their distinct qualifications and negative findings.

On current evidence none of these is justified: the diversity is real but does not convert,
and the selector already captures most of the little headroom that exists.

Open-development A/B/C/D results are never independent confirmation. A protected confirmation
stage needs its own prospectively frozen instrument and its own authority.
