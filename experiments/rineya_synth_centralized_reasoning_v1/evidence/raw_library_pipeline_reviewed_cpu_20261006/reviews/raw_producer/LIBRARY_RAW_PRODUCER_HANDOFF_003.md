# Frozen actual raw observation producer

Only staged `prefix_library_policy.py` and new `test_library_raw_observations.py` were changed. Exact bd57 helper/managed tests are retained in `PRE_RAW_OBSERVATION_PRODUCER_BD57_001`. No L/W/C/P/S/A, independent evidence validator, tracked source, native/framework/OS/GPU API, runtime/model binary or launch permission was changed by this implementation.

Final helper SHA256 `2f8d5c30d85c066413accab77bfe80c2eb20cc184772cd44ec097acb8775b1c1`; producer tests `a3d1597c765ba4f04a39c367247e9f6c550c2ad8d2bef3ff31acb31df434da60`. Final82 controls (unchanged legacy66 +16 new) pass in4.485s, zero skips, unchanged own readset and quiet native-import/real-runtime byte tripwires. Actual receipt/output: `LIBRARY_RAW_PRODUCER_CPU_1791249425222818400.json/txt`; receipt body `ba1e548920eac6770cdd44a7a98d5add3ed36abcc2e13244158ab05c57547192`. Earlier source/parser interoperability diagnostics and provisional receipts remain with the evidence agent/Root; successful intermediate80/82 receipts remain in this reviewer directory.

## API and actual source observations

Every guard records from its initial checkpoint. `_CPU_make_guard(...,clock_ns=...)` permits an explicit inert clock only behind its existing CPU capability. Native recording uses `time.monotonic_ns`; checkpoints/metadata are bounded to600s,4096 observations,16MiB exported bytes and bounded error populations, with existing accounting-budget checks. Recording cannot suppress actual cookie/observer cleanup when the clock, accounting or checkpoint limit fails.

`guard.export_raw_packet(identity=None)` is a pure metadata export: it never reads a provider. It returns a sealed `schema_version=2`, `kind=OBSERVED_LIBRARY_GUARD_RAW_PACKET_V2` packet with externally supplied identity, policy digest, actual selected library map, reached observations, ordered failure records, terminal state, producer claims and actual fixture-origin/false native authority flags. Identity is not an external plan and is not native qualification. A missing/unavailable or malformed PID/FILETIME birth cannot claim completion.

The identity recipe used by Root/independent replay has the original8 fields plus `owned_birth_status=VALIDATED|UNAVAILABLE` and `owner_receipt_sha256`; unavailable birth remains null. Actual W/context/topology binding and publication are owned separately. A clean unbound guard is incomplete, not a fictional guard failure.

Each observation has `ordinal,time_ns,event,boundary_id,boundary,scope,snapshot,event_trace,directory_identities,cleanup,observation_errors`. Boundary is null for lifecycle points and otherwise actual `{id,parent_id,name}` from reached entry/parent stack. Events are initial/enter/exit/boundary_error/precleanup/cleanup. Data slots reset at every point. Raw values are copied immediately from required guard reads before their validation; absent/unrecordable reads stay null with explicit errors. Direct framework loads before a child entry remain in that child's actual entry snapshot/event batch. External ordered plan stays in the qualified context/auditor and is never reconstructed from producer counters.

`observation_errors` contains only the five actual raw components. Source/checkpoint/clock errors live in `failure_records`, and missing component markers have exact same-point records. Records preserve bounded ordinary type/message/message_exact, ordinal, stage, boundary ID and observation ordinal. Propagated parent errors get their own reached-point links. No exception locals/frames, tensor values, repr or subclass stringification hooks are inspected. Truncated/unavailable messages mark evidence incomplete.

Constructor wrappers allocate the guard first, retain reached partial values and actual cleanup, and attach `__library_raw_packet__` through the built-in exception dictionary descriptor. This changes neither the raised exception's type/message/identity nor observations. Failure before the native stage/guard is reached remains outside this journal rather than fabricated.

Provider scope/snapshot/event calls are unchanged. The required final raw directory identities add exactly one explicit physical directory pass after observer cleanup; the bd57 predecessor is preserved. Final snapshots and the accepted finalized event drain are actual post-close reads. The observer interface must retain that lossless sealed log after close. Native observer implementation/provenance remains absent.

## Remaining authority and integration

This is source/CPU mechanics. Native current runtime/byte rosters, fresh bootstrap/source/owned identity, lossless compiled observer, complete API/cost/resource admission, original/copy preservation and independent terminal envelope/source/plan/birth/death joins remain separate gates. All native qualification/authority flags stay false. No M0 population, M1/M2 training, grading, scientific fit, cause or transfer conclusion is added.
