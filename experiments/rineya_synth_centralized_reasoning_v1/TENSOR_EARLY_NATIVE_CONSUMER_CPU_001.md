# Provisional early failure and native-purpose consumer: CPU composition

Phase6 now writes an actual provisional return when startup fails before the
core returns a report. It uses the original journal's action/durable writer.
The report retains errors, selected source/counter snapshots and any validated
pre-core membership observation. Completed-core membership and runtime fields
remain absent. Cleanup failure still attempts publication; writer failure leaves
honest raw evidence and failed IO. A returned core report retains its original
container path. Root reproduces all five writer controls in 0.134 seconds.

The existing native preservation consumer composes this envelope with the later
journal/final summary. It also retains its entered-core membership, runtime,
source-copy, supervisor, admission, resource, lease and death predicates. Root
reproduces five joined controls in 1.414 seconds, including an envelope produced
by the actual early writer and an actual write failure. The fixtures supply
invented CPU supervisor/resource/Job/death metadata; executing those predicates
does not establish native event facts. The nonfixture shared-entry gate remains
closed.

[Independent five joined controls](evidence/tensor_early_native_consumer_reviewed_cpu_20261007/independent_review/INDEPENDENT_EARLY_NATIVE_JOIN001_SUCCESSOR.json)
pass in 1.207 seconds with 33 selected pins unchanged. They execute the actual
early writer/failure path and native-purpose byte consumer with explicit CPU
metadata. Initial fixture-path failures and bounded test-only relocations are
preserved; candidate bytes are unchanged.

The seven-field envelope explicitly requires final phase accounting. Its
eleven-field report is provisional and omits final cleanup, accounting and success
claims. Independent replay derives costs and joins registered source maps. A
missing final summary remains partial; an absent envelope after writer failure
is semantically refused while its raw bytes remain preserved. No error or
missing observation is invented to complete a denominator.

[Frozen handoff](evidence/tensor_early_native_consumer_reviewed_cpu_20261007/final_native_consumer001/NATIVE_ENTRY_AUDITOR_HANDOFF.md)
and versioned histories preserve source, fixtures, controls and failures.
Worker is `36c5f9d1462721adc76e46dd62dd79e42654790b442cd5cc824fbcda0ee7fd20`;
auditor `e06f987f252a6b1f6c5fa3557b4cc545905b305c3dc17c10eb727606a6993b58`.
[Preservation manifest](evidence/tensor_early_native_consumer_reviewed_cpu_20261007/ROOT_PRESERVATION_001.json)
binds 68 files and 1,164,557 bytes.

This is source composition, not native readiness or permission. Current source
promotion/dispatch, complete runtime review, actual provider/interface/library
facts and the exact tensor registration remain pending. No native APIs, ctypes,
DLL load, Torch, CUDA, tensors, models, training, scoring or fresh death probes
ran. The original resource/numerical/marker limits and protected banks remain
closed. The three-arm transfer and prior-correct comparison remains unmeasured.
