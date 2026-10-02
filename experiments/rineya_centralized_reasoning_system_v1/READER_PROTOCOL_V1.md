# Exact-prefix deterministic reader qualification v1

This separately registered endpoint is `CRS_EXACT_PREFIX_GREEDY_SERVING_576_V1`.
It uses one frozen Titan BF16 checkpoint, stock eager Qwen3, batch one, unpadded
full prefixes, no cache, no adapters and no gradients. The complete input may
contain at most 512 tokens and the response at most 64 tokens. Input length
includes instructions, source, state, procedure, schema and chat-template tokens.
Unsupported inputs fail without truncation. The original bulk capture remains
FAILED at the unchanged maximum 0.03125 and mean 0.01 bounds. A new serving
qualification cannot qualify bulk likelihood, training or RL.

The design review is `audits/READER_QUALIFICATION_DESIGN_REVIEW_1.md`. This is
executable source continuation awaiting fresh reader review, not native evidence.
The initial low-RAM/Photos/Unity refusal is historical host evidence; this builder
makes no current host-fit claim. The later native whole-path failure remains an
independent closed prerequisite, described below. Module import initializes no
framework, tokenizer or model. All tests in this reader milestone are lightweight
stdlib CPU fixtures; none is retained as a native success receipt.

The fixed maximum schedule is:

| Control | Maximum forwards | Input positions |
| --- | ---: | ---: |
| Direct/no-op parity at 128, 512 and 576 | 6 | 2432 |
| Historical natural owned/stock/replay | 9 | 1161 |
| 512-token natural owned/stock/replay | 192 | 104352 |
| Frozen mechanical prefixes 512 through 576 | 65 | 35360 |
| Historical 131 and long bulk diagnostics | 2 | 707 |
| Maximum 576 baseline and three suffix controls | 4 | 2304 |
| Total | 278 | 146316 |

The maximum is 590 logical construction/load/310-copy/forward calls and 134
owned-plus-stock natural emissions. Actual counts depend only on natural EOS;
all fixed mechanical controls still execute. Replay and mechanical inputs are
separate exposure, never generated model tokens. The final 576-token forward
is nonemitting and cannot produce a 65th response token.

Natural serving stops at the first newly emitted EOS (151645 or 151643), including
EOS at the cap. EOS already present in the prompt does not terminate a new
response. Raw rendering retains special tokens and whitespace. Visible rendering
removes only a terminal EOS by token ID. Historical output is frozen to
[16, 22, 151645]. A long natural cap-stop may remain unobserved. The fixed 64-token
mechanical continuation is shape/resource evidence and cannot stand in for an
observed unconstrained 64-token response.

The owned loop and pinned stock `generate` execute sequentially. Effective logits
processors must be empty; stopping criteria are exactly length and EOS. The stock
FP32 raw-logit copy must preserve the BF16 bytes. Selection uses raw lowest-ID
argmax; processed scores are unused. The retained old capture receipt
`e669a6945520b7c91bb5d00382ceab2ac4ab5cdc8f09d9273ebd4915ae8017a2`
anchors historical online logprobs [-0.5026993751525879, -0.41050243377685547,
-0.0005460678366944194], selected raw logits [20.625, 20.25, 25.25] and the three
selected IDs separately from stock/replay self-consistency. The old likelihood
bounds are unchanged and the old failed bulk endpoint remains separately anchored.

External completion verification reconstructs the exact conditional per-call
plan. Every saved forward binds its stage, prefix IDs/hash, call ID, position
charge, retention shape, raw logit/hidden hashes, finite scalar evidence, argmax,
complete actual causal-mask representation and actual rotary positions. Exactly
one durable control manifest binds all six block manifests, natural rollouts,
bulk diagnostics and causal comparisons to the final result. Missing, duplicate,
foreign, altered or bool-for-integer evidence fails. Maximum-shape mutations at
positions 573, 574 and 575 protect positions 572 through mutation-start minus one
and require raw-byte equality of protected logits and normalized hidden state. Each comparison also binds the baseline and mutation forward records, unprotected positions and slice hashes; both unprotected logits and hidden bytes, and their full retained hashes, must change. Constant-output controls reject without adding any forward.

Tokenizer/template tests remain deferred native prerequisites. They verify the
fixed raw-text roundtrip, rendered UTF-8/token equivalence and historical prompt
suffix. The contract pins all tokenizer sidecars, the chat template, generation
source files and all 310 serialized checkpoint tensor names. Header inspection
pins names without reading tensor payload or importing a framework. Public
requests admit only token IDs, all-one attention mask and output cap; no oracle,
gold, scorer metadata or evaluator feedback enters the prompt interface.

`reader_execution.py` adapts source-pinned native fencing, Windows Job/query
handle transfer, tree monitoring, durable idempotency and failure accounting into
this new namespace. The original native/operations code and D2 dose are untouched.
Whole-path qualification of these inherited primitives remains quarantined.
Limits stay 600 seconds, 2 GiB process-tree RSS, 1.25 GiB host reserve, 3.3125 GiB
pre-worker RAM, 4/0.5 GiB GPU free, 4 GiB disk free and a 100 ms watchdog. One model
worker runs at a time. Qualification and failures count against 7200 seconds per
epoch and 14400 seconds per rolling 24 hours. Unknown creation/outcomes preserve
the reservation and require reconciliation; there is no automatic retry.

A fully complete contained supervisor first publishes
`COMPLETE_READER_EVIDENCE_PENDING_INDEPENDENT_REVIEW`. The launch source approval
is never actual native evidence approval. A separate identified independent
post-run reviewer must bind exact worker, EVENTS, completion, supervisor,
containment, owned-tree transfer, GO/fence, runtime, loader/model, tokenizer and
source-launch review references. Publication and serving replay complete worker
and artifact validation before permitting `PASS_EXACT_PREFIX_SERVING_576_SCOPED`.
Source-only approvals, blank/builder reviewers, caller status dictionaries,
minimal hash-valid unexecuted bundles and prior D2 receipts cannot open serving.
The independent review remains a human/agent evidence disposition over shared
privileges; it is not cryptographic proof of execution or OS secrecy.

`ReaderSession.serve` exposes the qualified token endpoint. CLI `--serve` remains
closed until a separate public request/DEV collection registration. The 16-case
DEV extraction diagnostic needs its own frozen output-token fit and registration;
complete event JSON may exceed 64 tokens. Central CHECK/FINAL and training stay
closed. This implementation establishes no reasoning gain, confidence calibration,
autonomous discovery, weight acquisition or retention result.

After resolving the inherited prerequisite, fresh reader source review, a clean
commit and actual host admission, the gated commands are:

`python -B -m experiments.rineya_centralized_reasoning_system_v1.reader --preflight --contract-sha256 SHA --review SOURCE_REVIEW`

`python -B -m experiments.rineya_centralized_reasoning_system_v1.reader --qualify --contract-sha256 SHA --review SOURCE_REVIEW --output CRS/results/NEW_RUN --job-key FROZEN_KEY --epoch EPOCH`

`python -B -m experiments.rineya_centralized_reasoning_system_v1.reader --publish-reviewed --contract-sha256 SHA --output CRS/results/NEW_RUN --evidence-review POST_RUN_REVIEW`

Source review status is `APPROVED_FOR_READER_QUALIFICATION`. Post-run status is
`APPROVED_READER_NATIVE_EVIDENCE`, review kind `POST_RUN_NATIVE_EVIDENCE`, with
`actual_native_evidence_reviewed=true` and full run-specific references. This
builder cannot approve either stage.

## Inherited supervision quarantine

The root native validation at source 1fd failed after 4.219 worker seconds with
an unregistered owned runtime executable during framework import. Owned-tree exit
was verified; no model loaded in that attempt. The native operational milestone
exhausted three repair rounds, and its whole-path supervision claim is QUARANTINED.
Reader preflight, worker, publication and serving all deny this prerequisite.
A reader namespace cannot broaden the executable registry, relax a limit or bypass
the fourth-repair prohibition. Existing failures and charged costs are preserved.

The resolving checkpoint is a separately registered, frozen and independently
reviewed no-model owned-Job inventory experiment. It must capture exact PID,
creation time, executable path/hash, Job membership and source operation phase
before causal attribution; retain failures; bound CPU time/observations/bytes;
and prove complete owned-tree exit. It grants no model launch or qualification.
Do not infer that nvidia-smi or any other executable caused the historical failure.
The reader preflight reports quarantine without new host checks. The resolver
needs its own reviewed contract and an explicit operations disposition before any
future reader qualification.

## Preserved first freeze

`history/reader_initial_v1/` retains all five initial reader files byte-exact
against the first scientific review pins. `RECOVERY_MANIFEST.json` records that
the builder missed the initial timely snapshot and recovered these exact bytes
from recorded tool calls after repair began. Unmatched reconstruction drafts are
retained only in ignored outputs. The initial failure and review remain replayable.


## Final run and publication controls

A single immutable run identity binds the job key, epoch, registered request hash,
source/contract, source review reference, canonical host database, owner/fence,
600-second reservation/time and exact command hash. GO, the launcher/runtime
identities, child events/results, containment/runtime/model/tokenizer receipts,
supervisor, terminal job snapshot and immutable measured-exit receipt carry that
same identity. Before every operation the child checks the actual canonical
RUNNING row, reservation, payload and observed launcher/runtime. A live lease
for another request is insufficient. Duplicate before-GO identities, missing
launcher/runtime, another run's supervisor, altered command/attempt or inconsistent
cost/exit proofs reject. The 310 named copies must occur inside the registered
checkpoint-load span. Worker and journal duration must fit the known supervision
and durable outcome times.

Source-review and output paths are validated before the inner reservation or
worker dispatch. Both source and native evidence reviewers use the same explicit
slash/backslash/case/whitespace normalization; builder aliases reject. This logical
identity policy does not authenticate same-user files.

Contained worker success, pending publication and independent serving approval
remain distinct stages. Interrupted pending/final publication preserves the raw
worker, journal, completion, supervisor, terminal row and measured cost. Recovery
uses the read-only OperationsStore adapter over the existing canonical database;
it cannot register, reserve, launch, reset, reconcile or release native work.
Create-only hard-link publication accepts an identical existing receipt or
concurrent identical winner, preserves partial/conflicting bytes and rejects a
conflict. A repeated qualification command against an existing output routes to
recovery and never invokes a worker again. Unknown/active jobs cannot recover a
success publication.

The explicit no-worker recovery command is:

`python -B -m experiments.rineya_centralized_reasoning_system_v1.reader --recover-publication --contract-sha256 SHA --review SOURCE_REVIEW --output CRS/results/RETAINED_RUN --job-key FROZEN_KEY --epoch EPOCH`

Recovery publishes only evidence-pending, including while an inherited claim is
quarantined; it cannot publish serving PASS. Final publication and serving still
reject inherited quarantine and require independent post-run review over all 15
artifact references, including launch intent/attempt and terminal job/exit proof.

The final CPU fixtures test protocol schemas and failure/recovery behavior only.
Their synthetic IDs, resources, hashes and temporary scheduler states are labeled
CPU_SYNTHETIC_INTERFACE_NOT_NATIVE. They neither load a model nor leave a native
success/qualification receipt. The full recovery fixture retains the same terminal
row/cost after interrupted writes, asserts no Popen, accepts identical retry and
rejects a conflicting pending manifest. It uses temporary SQLite, not canonical
host state. Numerical/resource limits, upper dose and old FAILED bulk capture
remain unchanged.

This is the reader milestone's third and final review/repair round. Fresh root
scientific and data/operations reviewers must assess these exact final bytes.
Any remaining material defect quarantines this reader candidate/claim; there is
no fourth repair, namespace bypass or automatic model attempt. Both earlier
reader versions and their failed reviews remain preserved in history.
