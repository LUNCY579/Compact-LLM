# Staged short M0 RAM/monitor successor

This is CPU preparation only. The original collector and shared helper bytes
are preserved in `original_source_preserved/` and `ORIGINAL_SOURCE_PINS.json`.
No root source, existing config, failed receipt, original M0 batch, model, GPU,
or grading policy was changed. Stage dispatch/preflight/worker entrypoints fail
before native operations. Root integration and a fresh source-bound run require
explicit review of this recipe and its CPU evidence.

The worker change is only the repeated guard's `available` callback: existing
`inference_qual_worker.make_guard` receives the proven
`worker_ram_probe.available_bytes`. Every invocation performs a fresh
`native_process.system_memory_status` physical-available query. The exact helper,
callback, matched512 CPU evidence, primary review and added AST runtime source
are bound in `successor_profile()`. No numeric cache, retained RAM value, fallback,
resource-floor change or guard-frequency change is introduced. The separate
post-import growth-reserve check remains unchanged.

The collector-specific `ShortRamMonitor` inherits the existing ownership,
containment, startup/GPU sampler and shutdown behavior. It selects locked CPU/GPU
publications and then captures its check clock. Existing `_fresh_cpu` and
`_fresh_gpu`, 200ms CPU freshness, 100ms CPU cadence, RAM/RSS/GPU/deadline floors
and identity checks remain intact. Only the check's clock ordering changes.
Small in-memory CPU phase markers expose sampling, publication, heartbeat IO,
check and cadence-wait state with separately named monotonic/performance clocks.

On abort, the original containment runs before one bounded diagnostic JSON write.
The diagnostic records the actual selected clock/sample timestamps, CPU predicate
flags/branch, caller thread and CPU phase. Diagnostic write failure is recorded
without replacing the original reason or its kill/Job-close fallback. The
supervisor records diagnostic status after sampler join, before inventory freeze.

The deterministic barrier control executes the **exact original check AST**:
its earlier `now=10` precedes lock acquisition; a CPU publication with start10.001
and completion10.002 is then selected and rejected. The successor selects that
publication first and captures now10.003, so it passes. Genuine aging, future or
nonfinite timestamps, overlong completion, and original resource/identity floors
still reject. This proves an ordering race is possible; it does not establish
that the race caused batch04.

Historical batch04 evidence reports `stale_external_CPU_watchdog`; exact failing
check time/selected publication/in-flight phase are unlogged. Every returned CPU
envelope is at most16ms and recorded preceding publication ages at most140ms.
The later supervisor-finish interval is not abort-check age. A RAM callback or
clock-order change cannot be claimed to cure that unlogged interval. The original
64-row packet, 44 complete rows, 352 preserved prefixes and missing step353 remain
partial evidence; no unchanged retry, partial resume, final-base proof or score
is inferred.

Validation includes15 new stdlib CPU controls and15 existing short-collection
controls explicitly loaded against staged sources. Two decoder cases use tiny
injected decoder/model-pin fixtures, avoiding published model/tokenizer reads and
installed decoder binary hashing. The old exact-supervisor-copy assertion is
replaced by the declared-delta check: the dispatch source matches the original
qualified algorithm after removal of the sole post-join diagnostic field, and
shared admission/ownership/cleanup functions retain object identity.

```powershell
python -B -m unittest discover -s experiments/rineya_synth_centralized_reasoning_v1/artifacts/short_ram_successor_work -p test_short_ram_successor.py -v
python -B experiments/rineya_synth_centralized_reasoning_v1/artifacts/short_ram_successor_work/run_existing_short_controls.py
```

Before any fresh batch04 attempt, review/preserve these exact sources and receipts,
register the successor source/runtime/profile closure and a new whole64-row scope,
refresh admission, and update the independent auditor for the declared new source
inventory/profile and abort diagnostic. Legacy source83/runtime64 assumptions
cannot silently qualify a changed scope; retain exact full inventories and
unchanged numerical/resource gates. Old batches01–03 remain valid only as their
old independently audited raw receipts. No new model, callback, monitor or
sustained native readiness is established by these CPU controls. Scoring and the
cap32 correctness policy remain closed; M1/M2, RS holdout and all retention
requirements are unchanged.
