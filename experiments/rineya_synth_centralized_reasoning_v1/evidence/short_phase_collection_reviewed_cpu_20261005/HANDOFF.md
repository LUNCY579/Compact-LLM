# Phase-observed M0 collector — staged CPU candidate

Four production modules are staged: `short_phase_contract.py`,
`short_phase_observation.py`, `short_phase_worker.py`, and
`short_phase_supervisor.py`. Tests and a fresh-root simulator are beside them.
Native entrypoints reject before paths, artifacts or APIs. Root files and all
original helper bytes remain untouched. This candidate grants no native launch,
scoring, training, retention or model-fit authority.

The worker imports the original ForwardDose, LocalGeneratedDecoder,
DurableRawCollection and collect_loaded as identical objects. No row, decoder,
model loader, parity, base-check, collection, EOS/cap or cleanup math is copied
or changed. The dispatcher differs from the original only in declared bootstrap,
purpose, contract and worker path. ShortRamMonitor and ownership/admission/
containment mechanisms remain exact.

The reviewed Recorder, GuardTorchProxy and observed_guard are imported directly.
The original Journal is invoked once per existing operation. Separate API and
actual-primitive spans preserve a primitive return before its following guard.
Unjournaled main phases use spans only: no resource guard or CUDA query is added.
Every original guard query returns its exact result/exception; false init skips
memory. The original terminal helper still writes worker_result before optional
CUDA diagnostics. A separate final accounting receipt follows its return, so a
swallowed terminal diagnostic_error cannot turn into phase-complete admission.

The append-only phase ledger fsyncs each marker. Creation, marker writes, close
and the projected single final receipt write have conserved IO counters. Native
exit zero requires the raw collection and final phase accounting both complete.
Actual query errors retain priority over recording errors. Partial writes,
fsync/close/final-write errors and pending phases reject completion.

Early pure admission checks exact main order, API/primitive/Journal order and
costs, original publication joins, and each API's before-guard/primitive/
after-guard nesting. Empty guard query paths are allowed only at pre_import.
The guard-erasure negative removes a whole guard and nested queries, renumbers
and re-seals every count/byte field, and still fails source-order admission.

Full64 early-EOS fixtures produce 194 actual calls/580 positions. The full64
32-step fixture reaches 2050 calls/35844 positions, 61798 markers and about
14.13MB; all64 outputs honestly retain generation_cap/truncated metadata. These
are sparse CPU fake-model outputs, never native evidence or graded answers.
The full32 storage fsync timing is injected; other controls exercise actual fsync
and IO failures. No native IO-speed, wall-time or intrinsic-fit conclusion follows.

The prospective native observation budget uses only the unchanged loader source
shapes and the pinned 5612-byte public size receipt SHA b6cac2bc…ba1. No model or
runtime binary bytes were read. Let N=2048, API=N+322=2370, K=682 summed BF16
tensor chunk iterations, and A=588 asset read iterations including EOF. The
original guard count is

`G = 9 + 2A + 6K + 2API + 2N + 2 = 14115`.

The guard terms are loader fixed boundaries, asset reads, checkpoint copies and
both device digests, Journal before/after guards, component before/after forward
guards, and pre_import/completed. Init+memory is conservatively budgeted before
cuda_init too; only pre_import necessarily has no Torch. Thus guard phases are
3G−2. Add two phases per API, N+68 original publications,18 main spans and4
existing terminal queries:49221 phases /98442 markers.

All3463 registered phase names (main/API/primitive/publication/guard/terminal,
including all310 tensor suffixes and64×32 publication names) are compared with
all7 successful result schemas, start/end records, ordinal≤200000, full UTC ISO
and signed finite binary64 clock representation. The longest name is74 ASCII
characters; a deterministic (length,name) tie-break gives identical profiles
under PYTHONHASHSEED=1 and999. The previous unsorted tie is preserved and its
cross-process mismatch is reproduced by a negative control. The conservative
marker size is381 bytes, yielding37506402 bytes, below200000 markers/64MiB.
Prospective overflow is rejected in the profile before dispatch; runtime writer
bounds also remain. Failure text is separately bounded and failed scopes never
receive successful admission.

The original64/512/32/2050 dose, complete row order, family/scorer-source bindings,
all384 atomic rows/96 components, held-out RS and retention objective remain.
Runtime64 plus AST65, decoder17, data5 and model7 remain exact. D001 is pinned as
current observation provenance only; it neither identifies nor cures old aborts.

Remaining prerequisites: independent review, reviewed root integration, a clean
new manifest and externally frozen exact source/profile/config/commit inventory,
and a new-only independent terminal auditor. The pure phase admission functions
are not that full auditor: whole64 raw/decoder/model/source/resource/cleanup and
complex loader replay still need the independent audit. Caller hash anchors bind
only externally selected evidence; self-signed receipts do not create authority.
Old successful batches keep their own receipts; old failed/partial batches stay
excluded. Real finish-policy/scoring and the full-panel import gate remain separate.

Original11 source files are byte-preserved. The initial Windows test-teardown
failure and the19-pass nondeterministic-budget predecessor are preserved with
source/test/log hashes. CPU_VALIDATION.json records the final named-control and
fresh-root results, full32 counts, source hashes and closed flags. No ignored
stage dependency is required by copied root tests.
