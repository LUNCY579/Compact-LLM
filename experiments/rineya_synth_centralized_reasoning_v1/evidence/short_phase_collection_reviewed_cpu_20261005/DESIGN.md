# Proposed phase-observed short M0 collector — design only

This is a separate, unimplemented source profile for the same fixed M0 raw batch.
It is motivated by D001's current observations, not by an established explanation
or cure for either historical batch04 abort. Native dispatch, model work, grading,
integration and launch remain unauthorized by this proposal.

## Small source delta

Add `short_phase_contract.py`, `short_phase_worker.py`,
`short_phase_supervisor.py`, and a small `short_phase_observation.py` helper under
this stage directory. No original collector, component, loader, precision,
ownership, resource, RAM callback, monitor or diagnostic helper is edited.

The new worker imports the byte-pinned original `ForwardDose`,
`LocalGeneratedDecoder` and `collect_loaded` from `short_collection_worker.py`.
That also retains the original `DurableRawCollection` implementation. The
collection, generated-only decoding, row validation, EOS/cap metadata, full
FP32 zero-B parity, all310 base check and cleanup loops therefore remain exact.
No second model loader, decoder, solver or grading rule is introduced.

Import `Recorder`, `GuardTorchProxy` and `observed_guard` directly from the
reviewed `context_phase_worker.py`. Do not import or invoke its `context_phases`
or `lazy_native_api` pipeline: those would add diagnostic queries and guards
which are not part of the existing collector. The real Torch object remains the
model API. Only the original `make_guard` getter returns the two-query proxy.

The dispatcher is a declared adaptation of the current short dispatcher with
only bootstrap, purpose, contract and pinned worker path changed. It continues
to use the exact current `ShortRamMonitor` and all original owned Job/admission,
deadline, telemetry, containment and post-join diagnostic machinery.

## Observation without extra queries or guards

Record begin/returned/failed boundaries around the exact existing import,
version, physical growth reserve, direct preload query, precision configuration,
each seed, process peak-counter reset, loader, decoder, collection, final
preflight and terminal preservation operations. These spans add no resource
guard calls to formerly unguarded operations and execute their callable once.
The existing explicit `cuda_context_init` Journal boundary remains once.

An observational Journal subclass surrounds the original `Journal.call` and
`Journal.evidence` methods once. It does not call Journal recursively, change
their primitive names, or duplicate their before/after guards. This observes
loader/tensor-copy/forward and raw token/final-row publication boundaries without
changing model or row behavior. Large model/report return objects are not copied
into phase records; bounded scalar/opaque metadata is enough to mark completion.

The unchanged guard still issues exactly its existing `is_initialized` call and,
when true, its existing `mem_get_info` call. The reviewed proxy records these
calls independently and returns the same result or exception object. There is
no new CUDA memory/allocator query, sampling omission, resource attribution,
cache clear, device reset, unowned control or retry.

Use the unchanged Recorder API with a separate nonrecursive preservation writer
that appends named canonical records to one bounded, exclusive-created
`PHASE_OBSERVATIONS.jsonl`, flushing and fsyncing each start/end record. This
avoids tens of thousands of new files while retaining every actual guard query.
A separate writer ledger accounts attempted/returned/failed/pending writes and
bytes; it never recursively records its own recording. Marker-write failure
stops subsequent work through recorder health checks. An actual guard/query
exception retains priority over a simultaneous recording error.

The original terminal helper writes `worker_result.json` before its optional
CUDA diagnostics. A distinct final phase-accounting receipt must therefore be
published after that helper returns, with all terminal query phases included.
Proxy only those already-existing terminal queries, once. Project and reconcile
the single final receipt write against its actual IO ledger before exit zero;
receipt failure forces nonzero and cannot erase the primary collection error.
Neither an earlier raw-complete status nor a partial phase ledger admits success.

## Separate manifest and auditor authority

Use a new contract/version, purpose and `SHORT_M0_PHASE_OBSERVED_PHYSICAL_RAM_V3`
profile. Compose the unchanged RAM2 profile with exact observation helper and
D001 audit file/body pins; D001 grants observation provenance only. A new
manifest reconstructs the same original public plan and fixed selector and
rebinds every field. Stage worker, contract and dispatcher reject before paths,
artifacts, probes, framework imports or model bytes.

Retain whole64, context512, generation32, capture2048 plus parity2 =2050,
the original position formula, fresh dense M0/seed101/zero-B profile,
resource/numerical floors, CPU100ms/.2s freshness, preload4GiB, during-work.5GiB,
RSS2GiB, RAM1.25GiB plus unchanged growth reserve, disk4GiB and whole600s.
All384 atomic rows and96 components, order, family/scorer-source bindings,
held-out RS and retention obligations stay intact. Failed/partial old batches
cannot fill missing rows; old complete batches keep their own source receipts.

A new-only staged auditor adapts the preserved RAM2 auditor's exact whole64,
config/source/model/decoder/raw/finish/numeric/resource/cleanup checks. It requires
externally selected exact inventory file SHA/profile-body SHA, root/package/
config paths and launch commit. Keep original runtime64 values plus AST =65,
decoder17, data5 and model7. Freeze the entire actual source path/hash map and
source count at the future launch HEAD under the existing owned ignored
registration directory. Bind D001/profile evidence separately. No dynamic
length-only admission or self-signed audit authority is accepted.

The new audit additionally replays every phase name/ordinal/start/end, exact
source-defined nesting and operation order, original guard/query count and
false-init bypass, API/raw-publication joins, IO conservation and final receipt.
It derives repetitions from the actual complete capture count, never D001's145
fixed phases. Any missing, duplicate, pending, reordered, contradictory,
nonfinite, altered, unregistered or unsuccessfully published phase rejects
success before decoder/model-heavy independent auditing. Real scoring remains
closed under its separate finish-policy and full-panel provenance gates.

## Bounded CPU controls before independent review

Run existing relevant short-collection controls against the new staged caller
and exact imported original collection routine: whole64/order/prompt-only
validation, actual2050/position caps, EOS/cap/partial metadata, generated-only
decoder binding, parity/base failure, actual returned cost on sink failure,
sticky halt, ownership/admission/cleanup and unchanged monitor floors.

Add fake-phase controls for exact import/seed/query/loader order; false-init
bypass; original guard result/exception identity; no added or recursive queries;
publication failure before another forward; native stage refusal before APIs;
partial phase/call/IO costs when actual work or persistence fails; terminal
query chronology and final receipt failure; source/profile/runtime omission,
mutation and re-signing; phase roster corruption; and portable root tests in a
temporary layout with no ignored stage. CPU fixtures provide no model-fit,
language, retention, grading or native readiness result.

No implementation or native manifest has been created. Independent source and
CPU review, a new auditor inventory and explicit future owned admission are
required before any model run.
