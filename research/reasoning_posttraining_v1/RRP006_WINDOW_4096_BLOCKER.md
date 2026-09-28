# The 4096 training window is refused by the contract, not by memory

**Finding: a 4096 training-window qualification cannot be launched through the
sanctioned controller today.** Two independent structural refusals fire before
any GPU work begins. Neither is a memory limit, and neither is fixed by freeing
RAM or closing sessions.

This corrects a cost estimate that described the qualification as "~10–15
minutes of GPU on the existing configuration". The GPU time may well be 10–15
minutes; getting to it is a change to the frozen design contract.

## The two refusals, reproduced

```
max_sequence=2048: ACCEPTED
max_sequence=4096: REFUSED -> frozen plan disagrees with the qualified native inventory
4096-token row at update_tokens=2048: REFUSED -> sequence cannot fit update group
```

### 1. `design_for` pins `max_sequence` to the Design default

`experiments/reasoning_posttraining_v1/controller.py:121-138` rebuilds the design
from exactly three keyword arguments:

```python
d = Design(processed_budget=plan["processed_tokens"],
           selection_seed=plan["selection_seed"], order_seed=plan["order_seed"])
if d != Design(processed_budget=d.processed_budget):
    raise ValueError("microfit may not re-seed the frozen design")
...
if (d.max_sequence != plan["max_sequence"] or ...):
    raise ValueError("frozen plan disagrees with the qualified native inventory")
```

`d.max_sequence` is therefore **always 2048** (`config.py:17`). Any plan
declaring 4096 is refused. The comment on line 125-126 is explicit that
`processed_budget` is the only field permitted to differ, and the equality check
on line 127 enforces it — so the window cannot be widened by passing a different
`Design` either.

### 2. `group_updates` refuses a row longer than `update_tokens`

`experiments/model_quality_gpu_v1/data.py:133-144` raises
`"sequence cannot fit update group"` when `len(row["ids"]) > token_limit`, and
`Design.update_tokens` is **2048** (`config.py:18`). A 4096-token presentation
fails at plan-build time, before the controller is even reached.

## What this means

Widening the training window is not a run. It is an amendment to the frozen
design that `code_sha256` and `mq_code_sha256` bind into **every completed
comparison in this programme**. It needs its own registration, and the operator
authorization obtained for "a bounded 10–15 minute measurement" does not cover
it, because that description was wrong.

The separate refusal noted elsewhere — `engine.py:44` rejecting
`stop_after_updates` on the native inventory — is real but is *not* the binding
one, and it is avoidable without a code change by registering a small
presentation set that terminates naturally.

## What is still true, and unblocked

`arm.py` now threads an explicit `train_window` (default 2048, behaviour at the
default byte-identical; 21 tests in `tests/test_arm.py`, 289 in the suite). That
is a **necessary but not sufficient** prerequisite: the builder can now express a
4096 set, and the controller will still refuse to run it.

## The consequence for the trace-vs-final experiment

At 2048 the `sft_reason` view retains **0 of 2,000 `ledger_replay`**, **0 of
2,000 `register_machine`** and **7 of 2,000 `evidence_chain`** rows. Until the
window question is settled, a trace arm must pre-register that **no per-family
claim may be made about those three families** — which is most of what the arm
exists to test.

Three honest options, in increasing cost:

1. **Run the trace arm at 2048 anyway**, scoped to the five families that do fit,
   and say plainly that the state-tracking and abstention families were excluded
   by the window, not by the result.
2. **Amend the design contract** under its own registration, accepting that the
   amendment is a new binding and that prior comparisons are read against the old
   one.
3. **Spend the lane on `process/train_transitions` instead** — 80-token rows,
   100% fit at 2048, a floor of exactly 0.500, and the same state-tracking skill
   the long families carry. It needs no window change at all.

Option 3 is the only one that buys the target skill without touching a frozen
contract, and it is the cheapest of the three.
