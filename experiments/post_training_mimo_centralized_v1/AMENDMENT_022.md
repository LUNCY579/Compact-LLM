# Amendment 022: exact-prefix fixed-trace capture successor

1 October 2026. Prospectively register one new non-evaluation mechanical
qualification on verified central source
`19551ddebc87c93161c303865ce6d0f130de0093`. The native D2-WAIT-V2 receipt on that
source passed all six fixed-shape protected-hidden/logit comparisons bitwise.
The backend40 mechanics receipt also passed; neither run will be repeated.
Byte-identical copies bind those facts without modifying the original receipts.
No newer local exact-prefix registration was found across matching Git worktrees.

Question: does independently selecting saved targets from exactly the same
prefix shapes reproduce the raw-model likelihoods observed during greedy
generation? Use only the already exposed mechanical 128-token prompt and its
historical three-token trace `[16, 22, 151645]`. This is deliberately smaller
than the original two-fixture capture registration and cannot satisfy that
registration or its existing general capture/DEV validators.

Load the unchanged 596,049,920-parameter BF16 Titan directly on CUDA through
the existing 2 MiB streaming loader. Keep eager attention, eval, no gradients,
no adapter, no cache, raw greedy argmax with lowest-ID ties and one retained
logit row. Reconstruct the prompt with the pinned tokenizer/chat template and
verify its complete token array against the byte-frozen prompt before forwarding.
Generate at most three tokens. Verify raw choices against the historical trace;
a changed choice, early EOS, missing natural terminal EOS or invalid likelihood
stops this attempt. No target token is inserted into online choice computation.
Retain terminal EOS and independently decode raw and terminal-only-stripped text.

After completing and durably recording the online trace, separately build
teacher-forced inputs from the frozen prompt and saved emitted targets. For
target ordinal i, input exactly prompt + targets[:i], excluding that target
and every future token. Use independent forwards in order 2, 1, 0; do not reuse
online tensors or scalar likelihoods. Select each saved target from the full
vocabulary FP32 log-softmax and restore the original ordinal order for comparison.
Both paths use lengths 128, 129, 130 and logits_to_keep=1. Freeze finite raw-model
logprob maximum/mean tolerances at the existing 0.03125/0.01. Use those same bounds
for the online historical-reproduction comparison. Report raw-logit fingerprints
and observed bitwise equality; do not substitute deterministic behavior logprob 0
for model likelihood. Frozen-prefix and target identities must be independently
checked, including the final EOS target.

Freeze one constructor, one pretrained load, 310 initial serialized tensor copies,
six model forwards, 774 total forward input positions, 318 logical calls and
three generated tokens. Zero backward/optimizer steps, training positions,
checkpoint saves/reloads, additional head projections or teacher calls. There
is no retry, continuation, extra fixture or automatic scope expansion. Preserve
attempted/returned/failed/pending operation records immediately and reconstruct
their journal. Failure and terminal receipts must contain the same current costs.

Retain LAPTOP_CAPTURE_RESIDENT_V1: process-tree RSS <=2 GiB, system reserve
>=1.25 GiB, GPU free >=4 GiB at admission and >=0.5 GiB during work, disk >=4 GiB.
Before the worker, require two consecutive >=3.3125 GiB RAM observations, two
seconds apart, with a maximum 30-second admission window. After imports reuse
the registered 60-second, two-second-poll dynamic growth-budget wait; recheck
source and perform strict fresh preload admission. Retain the external 100 ms
watchdog and Windows kill-on-close Job containment with a 600-second hard worker
ceiling. Fresh research-worker/GPU/lease overlap checks and the shared central
GPU lease precede launch. Never stop another job or user application.

Pre-native independent review requires two additional ownership/evidence controls.
The supervisor decorates the existing, byte-unchanged Windows Job factory to bind
GO to its reviewed source snapshot, live PID/creation time, unique lease token,
assigned worker-root PID/creation time and specific held Job handle. Before any
framework import, the worker verifies ancestor and lease identities, duplicates
that exact Job handle and verifies its membership in that Job. The supervisor
then validates returned source/config receipts and rechecks source after completion.
After an external stop it reads at most 8 MiB of journal metadata, reconstructs
complete durable records and retains partial-tail/pending uncertainty. Matching
recovery failure/result/cost receipts preserve existing worker receipts and fill
missing terminal records. No unrecorded operation completion is inferred.

Commit the registration and reviewed clean source before native execution.
Bind the exact runtime, assets, failed bulk receipt and passed D2 prerequisite;
recheck source/HEAD, frozen base, assets and raw journal after the run. Focused
stdlib controls test independent prefix assembly, reverse-order target alignment,
future-token exclusion, wrong input identities, EOS/rendering, unchanged numerical
gates, operation budgets and complete failure costs. CPU success is orchestration
evidence only; a contained native pass is fixture-specific capture mechanics only.

The five unpublished commits remain on their existing worktrees and are not
cherry-picked or rewritten: 669ba121 (CLARE scope), 22423340 (paired audit),
25b9860f (original capture prompt identity), d2be4c5b (checkpoint byte budget),
07b65f3e (original capture failure accounting). The successor explicitly binds its
own prompt and writes current failure costs; those required controls do not amend
or claim integration of the older validator/producer fixes. Checkpoint byte
contracts are outside this read-only, no-checkpoint run.

Preserve the original bulk failure and its unchanged tolerances. A successor pass
does not resolve prefix/full bulk-shape equivalence, prove intelligence, qualify
bulk training likelihood for RL, establish language/long-context/capability, or
authorize DEV collection, CHECK/FINAL, E1 or S1. Current best remains unchanged.
Any broader capture or validator successor needs separate prospective registration
and exact native measurement; no capability score is produced here.
