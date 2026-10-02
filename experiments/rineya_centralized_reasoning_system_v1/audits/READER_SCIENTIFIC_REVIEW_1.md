# CRS-001 reader scientific review, round 1

Recorded 2026-10-02. Reviewer `/root/scientific_audit`; builder
`/root/runtime_audit`. Disposition: **REPAIR_REQUIRED; NO SOURCE LAUNCH ACCEPTANCE
OR NATIVE QUALIFICATION FROM THIS REVIEW.** This is round 1 of the new reader
milestone, within the three-round limit. The separately scoped exact-prefix
design remains scientifically viable; these findings concern its implementation
and evidence gate.

I based the review on the frozen source and contracts directly, rather than on
the builder's summary, then ran only stdlib CPU tests and in-memory adversarial fixtures.
No Torch, Transformers or tokenizers module was imported; no weights, model,
GPU worker, protected CHECK/FINAL label, new research bank or process termination
was used. The reviewer did not build these files. The reviewer and builders share
model lineage, workspace, privileges and much historical evidence; this is fresh
reasoning over artifacts, not independent ground truth or enforced OS secrecy.

## Frozen first-round inputs

| File | SHA256 |
| --- | --- |
| reader.py | 804110837c260d7dcb67860ca0bdf2a49fd01b1afb7428d2d0e46331335017fd |
| reader_execution.py | e83984c8ce3d25d44558643b9e3d3440b8706c94f331c6457275aebb18bbd193 |
| tests/test_reader.py | 480aa79762bdad1a1a9c936b2c693866be53ee989846bba998a76a71dd9cd85a |
| contracts/READER_QUALIFICATION_V1.json | f53a116d171730bf5ecc5cb6d23dfd6ec38fa879c8b5b1d01785d2b0124998b6 |
| READER_PROTOCOL_V1.md | 3ab918ebbef216f933aeb41a22221820b47a2a37eab72578ae3c85cb0e3ff52e |
| audits/READER_QUALIFICATION_DESIGN_REVIEW_1.md | 0b0bdb275e9b8b50a0a7cef7bf6866e6cbaa7a2ce8abefefe699e902e79e6444 |

Also inspected the retained original capture result and `qualify_capture.py` /
`greedy_capture.py`, current native helper source, the deferred worker/supervisor,
and all reader tests. Source hashes above were checked on the initial frozen
bytes, before the builder began the reported repairs. This dirty primary checkout
is not a committed launch identity.

## What the source already implements

The owned loop has strict public prompt fields, input 1..512, cap 1..64, total
576, all-one unpadded masks, raw greedy token validation and first emitted EOS
precedence. The native callback uses BF16 eager stock Qwen3, no cache, no adapter,
fixed masks/positions, passive normalized-hidden witnesses and frozen parameters.
Stock generate is observed without replacing its selection code; effective
processors must be empty and stops must be length/EOS. Raw FP32 generation copies
are checked against BF16 forward bytes. Same-prefix replay and raw hidden/logit
parity remain distinct from bulk diagnostics.

The schedule arithmetic is correct: 278 maximum forwards, 146316 input positions,
590 construction/load/copy/forward calls and at most 134 owned-plus-stock natural
emissions. For a long natural rollout of N tokens, counts become 86+3N forwards
and 41900+N+3*sum(512+i for i in range(N)) positions. Forced prefixes 512..576
are prospectively fixed mechanical inputs, not natural emissions. Maximum-shape
suffix interventions protect positions 572..mutation_start-1 at starts 573,574,575.
Historical bulk failure, .03125/.01 limits and closed training/RL operations are
preserved. These are source observations, not executed native controls or fit
evidence. The recorded low-RAM/unregistered-GPU admission block remains intact.

## READER-SCI-01: completion validation does not establish the required controls

**Evidence:** `reader_execution.py:442` through `:495`, then completion/publication
at `:625` and `:700`. The validator reconstructs aggregate calls, but does not
associate forward call stages/positions with measured records or result control
manifests. It accepts six status-only blocks; does not require direct/no-op,
forced-prefix or bulk records; checks only equality of arbitrary causal strings;
does not validate complete binding, record schema/hash syntax/argmax/logprob,
registered mutation positions, mask metadata/position hashes, or exact final
result-to-journal correspondence. Stock and owned steps need only share strings.

**Impact:** the supervisor can return SUCCEEDED and publish serving PASS while
the retained evidence does not establish the claimed maximum shape, causal
controls, raw parity, extraction, historical reference or native EOS. A worker
bug, incomplete serialization or coherent result corruption is not rejected by
this external gate. Hash association alone cannot repair missing meaning.

**Executed CPU reproducer:** the following in-memory fixture was accepted. It
uses no model and is deliberately not native evidence. It has 89 aggregate
forwards/43437 positions for N=1, but every measurement is a one-token record;
one unrelated ledger call claims the remaining positions. There is no bulk,
direct/no-op or forced-prefix evidence, the binding contains only commit, raw
rollout hashes are `not-a-hash`, causal hashes are `x`/`y`, and long output [17]
is incorrectly labeled EOS.

```python
import copy
from experiments.rineya_centralized_reasoning_system_v1 import reader as r
from experiments.rineya_centralized_reasoning_system_v1 import reader_execution as e
from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting
cfg = r.read_json(r.CONTRACT)
events = []
def emit(kind, **detail): events.append({'kind': kind, **detail})
a = CallAccounting(emit, maximum_optimizer_calls=0)
for op, n in [('model_construct',1), ('checkpoint_load',1), ('tensor_copy',310)]:
    for _ in range(n): a.call(op, lambda: None, stage='synthetic')
for i in range(89):
    a.call('forward', lambda: None, stage='unbound_control',
           input_positions=43349 if i == 0 else 1)
for _ in range(89):
    emit('reader_forward_measurement', record={
        'input_ids':[16], 'input_ids_sha256':r.digest([16]), 'input_length':1,
        'witnesses':{'mask':{'raw_sha256':e._mask_sha(1)},
                     'positions':{'ids':[[0]]}}})
def rollout(prompt, generated):
    steps = [{'input_ids':prompt+generated[:i],
              'input_ids_sha256':r.digest(prompt+generated[:i]),
              'raw_logits_sha256':'not-a-hash', 'raw_hidden_sha256':'not-a-hash',
              'selected_model_logprob':-.1} for i in range(len(generated))]
    owned = {'generated_ids':generated, 'stop_reason':'eos', 'steps':steps}
    stock = copy.deepcopy(owned)
    stock['policy'] = {'processors':[], 'stops':['MaxLengthCriteria','EosTokenCriteria']}
    return {'owned':owned, 'stock':stock, 'replay':copy.deepcopy(steps)}
emit('resource', stage='complete')
result = {
    'status':'COMPLETE_READER_QUALIFICATION_WORKER_ONLY', 'scope':r.SCOPE,
    'binding':{'source_commit':'a'*40}, 'base_unchanged':True, 'model_loads':1,
    'bulk_capture_qualified':False, 'training_qualified':False,
    'rollouts':{'historical':rollout(cfg['historical_prompt_ids'],cfg['historical_generated_ids']),
                'long':rollout(cfg['long_prompt_ids'],[17])},
    'accounting':a.snapshot(), 'resources':[{'stage':'complete'}],
    'blocks':[{'status':'COMPLETE'}]*6,
    'checks':{'maximum_shape_causality':[{
        'left_logits_sha256':'x', 'right_logits_sha256':'x',
        'left_hidden_sha256':'y', 'right_hidden_sha256':'y'}]*3}}
print(e.validate_qualification_result(result,cfg,'a'*40,events))
```

Observed return includes `durable_journal_verified=True`,
`maximum_mechanical_shape=576`, `natural_eos_observed=True` and 89/43437.

**Verification criterion:** define exact bounded per-block manifests and validate
the full binding against the registered contract/commit. Bind each ordered
forward call to its exact stage, input IDs/hash/length/position charge, keep policy,
shapes/dtypes, finite nonpositive selected logprob, token/argmax evidence, complete
mask/rotary witnesses and full hex raw hashes. Canonically associate every saved
manifest/rollout/control with exactly one durable measurement and correct ledger
entry; reject missing, duplicate, altered and coherently altered registered
geometries. Require direct/no-op pairs, all 65 fixed prefixes including the
nonemitting576 probe, both bulk diagnostics/registered historical anchor, and
three fixed mutation/protected-position comparisons. Recompute actual per-block
counts, natural token/EOS/cap rules, and causal parity. A valid complete CPU
fixture must pass; each omitted/control-corrupt variant above must fail.

## READER-SCI-02: historical prefix numerical reference is not frozen/enforced

**Evidence:** `qualify_session`'s `natural` at `:273` checks historical generated
IDs [16,22,151645], and `bulk_diagnostics` at `:294` checks historical bulk raw
hash. Neither the worker nor contract compares the original online keep1
logprob/raw-argmax scalar evidence with the newly observed prefix path.
Original immutable capture at
`experiments/post_training_mimo_centralized_v1/results/headless_capture_qual_v1_20260930_1140/CAPTURE_RESULT.json`
has SHA256 `e669a6945520b7c91bb5d00382ceab2ac4ab5cdc8f09d9273ebd4915ae8017a2`.
Its online logprobs are [-0.5026993751525879,-0.41050243377685547,
-0.0005460678366944194], selected raw maxima [20.625,20.25,25.25], and first-max
token IDs [16,22,151645]. The same keep1 route is explicit in
`qualify_capture.py:174` and surrounding code. The old online-to-bulk failure
remains a separate diagnostic.

**Impact/counterexample:** stock/owned/replay can agree on a coherently changed
prefix distribution and unchanged selected token IDs. That agreement does not
reproduce the preexisting prefix numerical reference required by design section2.
The SCI-01 fixture's three -.1 prefix values agree internally and are accepted,
but comparison to the retained original vector gives maximum .4026993751525879,
mean .2708852470309163 and `passed=False` under unchanged .03125/.01 bounds.
This does not assert that actual stock execution drifts; it identifies an absent
gate that would detect such drift.

**Verification criterion:** pin the original receipt and prefix token/scalar
reference before launch, enforce the registered historical argmax/raw-selected
reference and original likelihood limits, and retain actual comparison/outcome.
Do not refresh expected values after outcomes or infer online consistency from
bulk hash/token equality. Add stdlib coherent-drift fixtures; original-reference
fixture passes and drift beyond either unchanged bound fails its scope.

## READER-SCI-03: source launch approval is reused as native evidence approval

**Evidence:** `review_matches` at `reader_execution.py:73` consumes prelaunch
source approval. `supervise` at `:700` immediately publishes served PASS and puts
that same `args.review` under `independent_review`. `reader.verify_qualification`
at `:139` accepts its `APPROVED_FOR_READER_QUALIFICATION` status/binding/reviewer
as sufficient independent review, with no post-run worker/journal/completion/
supervisor evidence binding or review kind.

**Impact/reproducer:** a scientifically approved source contract is being treated
as an independently reviewed actual native qualification. With file/hash reads
mocked to simulate correctly bound retained files, a review explicitly declaring
`review_kind='PRELAUNCH_SOURCE_ONLY'` and `actual_native_evidence_reviewed=False`
was accepted by `verify_qualification`; it returned serving PASS. The mock does
not test cryptography or filesystem authority: it isolates the missing semantic
distinction. The actual source also publishes precisely that prereview without a
fresh native-evidence review step.

**Verification criterion:** separate prelaunch source approval from independent
post-run evidence disposition. Fully completed worker/supervisor produces an
evidence-pending result. A subsequent independent evidence review must bind the
exact worker, EVENTS, completion, containment/runtime/model/tokenizer and
supervisor references for this run and scope before a serving PASS can be
published/verified. Reject a source-only approval or review from another run,
even with otherwise correct source/model binding. Preserve worker/whole-path
failure and admission refusal; never promote either using a source review.

## Validation and remaining scope

Registered interpreter, `python -B -m unittest ...tests.test_reader
...tests.test_learning_delta -v`: all 24 tests passed in .016 seconds. The 12
reader tests establish policy/geometry/rejection mechanics but contain no
complete external completion fixture; passing them did not close SCI-01..03.
In-memory reproducers ran in bounded stdlib processes and confirmed no framework
imports. The builder acknowledged the findings and planned one coherent repair
freeze; no repaired bytes are approved by this first-round report.

No numerical/resource checks should be weakened to close these findings. No
native qualification, 16DEV extraction, calibration, B/C gain, autonomous
procedure discovery, weight acquisition/retention, bulk training or RL result is
established. Preserve old FAILED capture and whole-path failures and the current
host refusal. A later source acceptance may permit only a gated qualification
attempt; actual qualification requires its own fully retained evidence.
