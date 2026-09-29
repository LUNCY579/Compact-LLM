# Next experiment

**Goal blocked after three resumed turns. Next native step: E0 only after a meaningful resource-state change.**
The latest OS-only observation at 05:54 UTC was 7.795 GiB, with no Python workers.
It is below the 8.156 GiB pre-import state that produced the latest v4 failure;
this is an inference against retrying, not a new post-import measurement.
The fresh three-turn blocked audit passed; see receipts/RESUMPTION_003_BLOCKED_AUDIT.json.
No eligible independent FAST task remains on the current critical path.

The second resumed turn measured 7.287 GiB available before imports at 05:52 UTC,
already below the 7.4 GiB post-import floor, with no Python workers. No attempt
was launched. See receipts/RESUMPTION_002_BLOCKED_AUDIT.json. The earlier v3/v4
attempts below remain the latest native evidence.
Latest attempt v4 stopped at preload: 7.378871917724609 GiB host RAM available
after imports, below 7.4 GiB. Its source was
`62ad48e2ade0ec4b3023e2aef5736bc34c6022d1`; the result is
`results/e0_titan_v4/E0_RESULT.json`. Accounting version 2 records zero attempts,
returns, failures and pending calls across all eight operation classes.

Earlier v3 stopped at 6.936687469482422 GiB on source
`5e5fe36ebc6c2725c61c4c4cf78d3f85344cba80`; its original receipts remain immutable.
OS-only RAM rose from 7.649 GiB before v3 to 8.251 GiB, then measured 8.156 GiB
before v4. That meaningful change justified the new bounded attempt. Both failures
occurred in the same resumed goal turn: blocked-turn count 1, failed-preload-attempt
count 2. The earlier seven-turn audit is historical and is not carried forward.
E0 requires at least 7.4 GiB available after runtime imports and 6 GiB
throughout the worker, plus 4 GiB GPU and 4 GiB disk admission. Keep every floor
unchanged. The preload block is not evidence that Titan cannot fit or learn.
Native model loads, optimizer updates and generated tokens remain zero; efficacy
is UNMEASURED.

The finite independent preparation has been completed and checked: exact source
identity, method audit, acquisition/replay data, language and long-context candidate
inputs, historical E1 input/schedule plans, actual DEV prompt/scorer binding, and
AMENDMENT_008 attempted-work accounting. The accounting source-review/CPU-test task
is complete: 52 stdlib tests passed in 0.681 s (process 1.1190241 s), and 12 existing
tiny CPU tests passed separately in 1.459 s (process 6.6929375 s). Evidence is in
`receipts/E0_ACCOUNTING_V2_CPU_VALIDATION.json`. The revised accounting code remains
natively UNQUALIFIED; these checks do not replace native evidence.
Do not regenerate banks, retry unchanged admission, invent grading
or add generic wrappers to substitute for the missing native evidence.

## Resume sequence

After host headroom changes, qualify the accounting-v2 bounded E0 source with the
pinned Python 3.11.15 environment, exact Titan weight/config/tokenizer hashes and
external process-tree containment. The original math, inputs, numerical tolerances
and resource guards remain unchanged. Commit the source changes first; use
a new immutable run directory. Recheck live resources and stop on any failed gate.
Qualify exact load, 32 zero-adapter cases, likelihood/masks/causality, 256/512/1024
profiles, finite confined gradients, reversible update and deterministic resume.
No current E0 pass, native source responses or trained candidate exists. Another
attempt needs a meaningful resource-state change beyond the v4 admission state.

Only after measured E0 should numerical native collection budgets and the actual
runtime identity be frozen, the collector/executor be completed against that
runtime, and unchanged-source responses be captured. Language still needs qualified
reference labels, reviewer assignment and calibration. The 2048/4096 forward-only
extension remains separate. E1 seed 101 follows all of those gates; seeds 211/307
require its registered passing DEV screen. CHECK/FINAL stay closed.

## Prepared DEV binding

`manifests/DEV_INPUT_SCORER_BINDING.json` indexes
`artifacts/mimo_central/dev_binding_v1`, produced at
`ead560f3ed0fcc879e2b26f77003b9053d37553d` in 107.513 s.
It contains 832 rows/544 parent groups and 890874 prompt tokens:
22914 acquisition, 80773 deterministic retention, 780288 long context, 6899 language.
The policy file has exactly six fields; only verified IDs/mask or equivalent
messages go to a model. Row IDs are routing metadata and never prompt content.
Complete evaluator data are separate. Short-bank slices exclude 6883 reference-token
positions; twins and all eight long-context variants remain grouped.

Pure deterministic scorers enforce the prospectively frozen malformed policy and
check pinned helper code before imports and each call. Language scoring refuses
unqualified exemplar matching. The static binding has no fabricated runtime,
decoding, seed or budget registration. It is not an enrollment manifest or native
grant. Forty new tests passed in 87.582 s; a separate audit checked 832 row mappings
and 34 file identities without finding a discrepancy. No model accuracy was measured.

The historical E1 TRAIN plan remains 5120 sequences/680532 nonpadding/63546 supervised tokens,
256 steps with 16 acquisition+4 replay sequences, seed 101 only, 13 warmup steps and
final LR 0. Its source bindings predate the accounting-v2 gate-checker change;
preserve existing plans/receipts and rebind a new plan to qualified current source
and runtime before native execution. Current best stays the unchanged exact Titan
checkpoint. Additional heavy training is not justified until native qualification
and preservation gates actually pass.
