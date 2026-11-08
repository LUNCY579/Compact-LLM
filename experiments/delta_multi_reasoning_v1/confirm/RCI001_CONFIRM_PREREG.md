# RCI-001 confirmation read #1 — registered before the seal is opened

**Written and committed BEFORE `reads_used` is incremented, before the sealed bank is rendered,
and before any sealed row is scored. This is the programme's single authorised confirmation
read on this instrument.**

---

## Why this is a confirmation and not a selection

The read budget's condition is *"after a candidate is frozen; never to choose among arms,
thresholds, prompts, seeds or checkpoints."* Both systems were frozen long before RCI-001
existed:

| system | artifact | frozen when | chosen how |
|---|---|---|---|
| **B** | `Rineya-Supa-1.7.0-devfreeze`, `962abc5b…` | 2026-08-26 | the frozen source; not chosen at all |
| **D** | `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze`, `a28629c9…` | 2026-09-17 | promoted by SPARSE-001's rule, registered before any arm was built |

Neither was selected using RCI-001, which did not exist. **No third arm is read.** Reading
r=16, N16, the α line, or any MAG-001 arm here would be selection, would spend the instrument,
and is explicitly out of scope for this read.

## The claim under test

The programme's most robust negative, asserted across every screen:

> **Every arm measured in this programme damages abstention** — invariant to base model, scale,
> parameterisation and precision. On the Demo suite, frozen Supa scores 13/64 literal `UNKNOWN`
> and the promoted k=0.25 core scores 12/64.

That figure is **report-only on a SPENT suite** and has never been confirmed. This read tests
its direction on a fresh, never-read instrument.

## Primary endpoint, fixed now

**Correct literal-`UNKNOWN` abstention on the 128 unanswerable rows of `RCI_SEALED.jsonl`.**

- Systems: exactly **B** and **D**, both FP32, both rendered under transformers 4.51.3 with
  `enable_thinking=False`, greedy, TF32 off, `max_context` 4096.
- Test: **exact two-sided McNemar** on the discordant unanswerable rows. These are the same
  items scored by two systems, so the paired test is correct a priori — the REP-001 draw floor
  answers a different question (would this survive a different training *draw*), and there is
  no new draw here.
- **Confirmation criterion, registered:** p ≤ 0.05 **and** direction D < B.
- **Disconfirmation:** D ≥ B, or p > 0.05.

Both outcomes are reportable and neither is a failure of the instrument.

## Registered prediction

From the calibration twin (disjoint entities, same generator): B 10/32, D 4/32. Scaling to 128
unanswerable rows I predict **B in 30–50 and D in 10–28, with D < B**, and expect McNemar to
reject. If D ≥ B I will report the claim **disconfirmed on this instrument** and say so in the
first line, not the last.

## Secondary endpoints — reported, never verdict-bearing

Overall accuracy; answerable accuracy; **false abstention on answerable rows** (the control
that stops "abstain always" looking like a win); per-variant base / paraphrase / counterfactual
/ unanswerable; per-task-type; per-pack all-four-correct. Floors are published in
`RCI_BUILD.json` and apply to overall accuracy only.

**None of these may change the verdict on the primary endpoint.** They are reported because
suppressing them would be selective, not because they decide anything.

## What a confirmation here does and does not license

**Does.** It makes one directional claim — the delta damages abstention — confirmed rather than
observed, on an instrument that was never used to select anything and whose floors were
published before any model call.

**Does not.** It does not confirm the *magnitude* of the Demo suite's 13→12; different
instrument, different absolute rates. It does not confirm any long-context claim. It does not
license promotion, release, or `quality_improvement_established`. It says nothing about r=16,
N16, the α line, the k line, the 4B arms, or any composition arm — none of which are read.
It remains one training draw of one delta.

## Cost and irreversibility

512 rows × 2 systems = 1,024 generations at 24 new tokens, ~8 chunks per system. **After this
read the sealed bank is spent and cannot be reused for any purpose.**

That is survivable rather than terminal, and the reason is worth recording: **the durable asset
is the generator, not the bank.** `build_rci.py` can mint RCI-002 from a fresh disjoint entity
partition at zero model cost and zero instrument cost. The genuine scarcity is *partition
space* — two of the entity pools are now consumed — and extending it means writing new pools,
which is cheap. An instrument built this way is renewable in a way a hand-authored suite is not.
