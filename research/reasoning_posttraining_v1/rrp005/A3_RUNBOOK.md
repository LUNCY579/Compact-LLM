# A3 — acquire only genuinely new measurements

## Reuse is checked by identity, never by whether a prediction is correct

The successful historical `baseline-pres-002` contains 737 actual native predictions.
The missing-RAG-input defect invalidates the affected question/gold interpretation; it
does not invalidate predictions on the unrelated unchanged prompts.

Fresh audit found **545 unchanged requests** (384 direct reasoning + 161 single-turn v3):
same prompt and target, exact native input/label/prefix/EOS boundaries, same source asset
hashes, same complete parameter digest, same recorded runtime, same 2048 window and
64-token output allowance. The old worker exited successfully and verified source state
unchanged. **All 301 fresh overlapping predictions match token-for-token**, including
termination and recorded input lengths. See `REUSE_AUDIT.json`.

These are **reused completed measurements**, not new model executions. Cache allocation
policy differs between old and new runs; it does not intentionally change arithmetic,
and the observed overlap agrees exactly. No inference-cost/latency equivalence is claimed.
No case was selected by correctness and no changed prompt inherits its old output.

Thus only **308 inputs** require acquisition: 181 repaired RAG cases, 31 restored multi-turn
v3 cases and 96 evidence-removal negatives. A final source-baseline analysis must attribute
the 545 old and 308 new measurements separately, check disjoint exhaustive coverage of
all 853 unique inputs, then report main-bank and paired-probe results separately. Do not
fabricate a native run claiming it executed the union in one worker.

## Correct a misleading interpretation of the old score

The old “29/29 answers regardless; zero evidence sensitivity” claim used a recognizer for
two exact abstention sentences. Inspection found **9/29 pairs** with an unambiguous natural
abstention on the missing-evidence half and a correct reference-matching answer on the
answerable half. Example: “I cannot determine who booked the room without additional
information.” The prompt requested a compact conclusion, not the scorer's exact sentence.

The zero **canonical-phrase** count remains a valid description of that recognizer; zero
**semantic** awareness is refuted. `SEMANTIC_ABSTENTION_CORRECTION.json` preserves all nine
examples as an exposed-DEV agent review, not independent human review. Registered numeric
reference scores are not retroactively changed. Describe them as reference/format agreement;
do not call every noncanonical refusal an unsupported answer. Reasoning improvements must
also show verifier-backed answer gains; canonical wording alone is insufficient.

## Resource backpressure

The last resource dip coincided with a `privatescan_lib` Rust compiler in the separate
PrivateScan project, observed using about 3.2 GiB private RAM. The model worker's RSS
was approximately 2.45 GiB and stayed almost flat while host availability dropped by
about 2 GiB. It is unsafe to infer an OS leak or model leak from available RAM alone.
No process was terminated.

A3 wraps the **unchanged** guard in a bounded scheduler: if RAM/VRAM is temporarily low,
it performs no subsequent model operation until that same guard passes. It may wait at
most 90 seconds per episode and 240 seconds total. Other errors propagate immediately;
an exhausted wait aborts with durable completed cases intact. It never retries a model
operation, restarts a worker automatically, lowers a threshold, changes dtype/device,
or drops a difficult input. Waiting is included in the existing watchdog and wall budget.
Tests prove no dependent operation executes during refusal and actual oversleep is recorded.

## Frozen scope and command

`PLAN_A3.json` / `GRANT_A3.json`, experiment `RRP-005-PRESERVATION-DELTA-A3`.
All remaining call/token/time/disk allowances transfer **exactly** from the settled A2
journal; none of these is expanded. Only two launch slots and one retry are added for the
new acquisition/scheduler. Training, adaptive selection, RL, paid services and release
remain zero. A3 is separately qualified on its actual source/runtime/checkout.

```powershell
& ./research/reasoning_posttraining_v1/rrp005/run_delta_a3.ps1 -RunId delta-001
```

The frozen acquisition bank is
`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/baseline-reuse-audit-001/CHANGED_OR_NEW_INPUTS.jsonl`.
Output journal is sibling `preservation-delta-a3-journal`. Keep code/checkout fixed while
running or while same-binding continuation remains necessary.

## Additional TRAIN-only preparation

`PROCESS_TRANSITION_AUDIT.json` independently replays all **36,752** supplied process
labels. A prospective sample selects one positive step per parent, then 128 parents per
family. All **384** new state-transition views fit 2048 tokens; maximum sizes are 156
(ledger), 162 (registers) and 119 (prerequisite masks). Input contains only current state,
event and a public transition rule; proposed next state and validity labels are excluded.

These are **candidates, not admitted training data**. Derived-view overlap, mixture weights,
parent grouping and transfer evaluation are still required. Local-step competence does not
prove full-history reasoning. In particular, processed-token proportions do not equal
supervised-loss proportions; the future replay mixture must report both.
