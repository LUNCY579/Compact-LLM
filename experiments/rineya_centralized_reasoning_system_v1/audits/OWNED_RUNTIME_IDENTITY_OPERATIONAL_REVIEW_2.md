# Owned runtime identity diagnostic: operations review 2

Date: 2026-10-02. Reviewer `/root/data_audit`; repair builder `/root/scientific_audit`; original implementation builder `/root`. **REPAIR_REQUIRED_NOT_ADMITTED.** Scope is the separate bounded identity instrument, not another native D2 repair. No actual import probe, framework, model, tokenizer, GPU API, canonical host database, gold or protected bank was used. Only this report was written by the reviewer. Shared workspace/model ancestry is disclosed; agreement is not independent physical truth.

## Frozen subject

| Subject | SHA256 |
| --- | --- |
| owned_runtime_identity_probe.py | `946b4fbc6cbf013577b1a0e827e902c85a9cdef8382e9a5cb83c687f831d28e9` |
| tests/test_owned_runtime_identity_probe.py | `da0c05c7319ec3f90368ed54057791b1a42ef072344f1327f24bed606d3df278` |
| contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json | `651cdc696eaa08e1c1b1564542d82238e7e9b57c5373516a8d508742114258f7` |
| retained native.py | `8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3` |
| retained native contract | `d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a` |

The entire 863-line implementation, complete test file, contract and initial operations findings were read directly before builder interpretation. No source was edited. Root retains the exact first-round archive and designated this milestone's final third repair only after these fresh reviews; this report grants no execution permission.

## Replayed evidence and first-round closure

All 17 supplied CPU tests passed in 3.115 seconds, no skips/errors. Six included actual newly created, suspended, exact-Job stdlib process fixtures: five terminal API refusal controls and one independent deadline control. They imported no framework and executed no import experiment. These small owned process trees were terminated/verified empty and cleaned by the test harness; unrelated processes were untouched. Other controls used temporary SQLite and synthetic artifacts, not canonical host state.

OIDENT-001 is substantially closed: the callback captures builtin os._exit before third-party imports, writes/fsyncs a refusal then exits86 in finally even if publication fails. Main-thread, new-thread, C-call, serialization and swallowed-exception fixtures cannot execute their first/second/continued side effects. Python serialization wrappers are now explicitly covered. Internal C/native-thread/generic C-constructor limits remain false/unobserved; these fixtures do not prove zero hidden import effects.

OIDENT-002 is substantially closed: HashBudget checks individual64MiB and aggregate256MiB distinct file identities before reads, uses at most2MiB bounded chunks, checks open/fresh file identities and counts failed admitted reads conservatively. A separate Timer kills the exact Job independently of the observer and raw resource thread. Actual no-framework blocked-sample deadline termination/empty-tree proof passes; late check rejects. Observer and strict guard gaps are timestamped instead of presented as guaranteed20ms/100ms coverage.

OIDENT-003 is substantially closed: the bounded newline-complete, duplicate-key rejecting child journal must contain the entire ordered32-event idle/source/import/boundary/post/final schedule. It binds10 snapshots/2 source checks/2 metadata observations, clean idle/post durations, actual GO/intent/containment artifact hashes and authority. Initial minimal markers, missing containment, wrong worker, reordered/duplicate/partial journal, wrong source, altered checks and bool requested-count controls reject.

OIDENT-004 is substantially closed: the child uses canonical host_state_path and verifies this exact fixed KEY, RUNNING state, reserved60, owner/fence/epoch/request hash/payload/recorded launcher plus observed runtime, clean source, review references and dispatch authority before imports; every phase rechecks the admitted row. A valid other-milestone lease, unregistered/PENDING job, wrong epoch/worker/request is insufficient. Temporary query handles retain close-before-framework failure paths.

OIDENT-005 is substantially closed: known path/creation time/registry association/member/RSS and component timestamps survive subsequent hash failure. A fresh psutil.Process object detects PID identity drift. Every bootstrap/target/runtime strict guard links to durable raw observation IDs with gap/result/refusal evidence. Sampling/notifications remain explicitly non-exhaustive and cannot establish the old cause merely from coincident import timing.

OIDENT-006 is substantially closed for cleanup and early publication: cleanup isolates each action and retains its errors, attempts drain/watch/Job/notification/launcher/pipe cleanup even after earlier failures, and unsafe cleanup becomes UNKNOWN with full held cost. Lost Popen response/restart and early PROBE_RESULT publication failure retain the full60 seconds and no duplicate launch. The final post-reconciliation publication branch still has the separate consequential gap below.

## OID2-OPS-001 — P1: late finalization failure falsely claims held UNKNOWN reservation

Evidence: supervise writes PROBE_RESULT, reconciles the terminal outcome, releases its lease (792–809), then writes FINALIZATION_RECEIPT (811–817). On this last write failing, the handler sets local outcome/FINALIZATION_ERROR to OUTCOME_UNKNOWN and emits reservation_retained=reserved (818–830). The variable reserved means historically admitted; it does not reflect the already terminal scheduler. mark_unknown is attempted only for active states, so a SUCCEEDED row is not changed and its released lease is not held again.

Independent executed reproducer: use the actual supervise/finalization and real OperationsStore in a temporary CRS outputs database; substitute only launch/host/worker/phase boundaries with explicitly synthetic CPU success, then inject OSError **only** for FINALIZATION_RECEIPT.json. No executable or framework runs. Observed:

```json
{"return_code":2,"saved_probe_result_status":"SUCCEEDED","finalization_error_status":"OUTCOME_UNKNOWN","error_claims_reservation_retained":true,"actual_scheduler_state":"SUCCEEDED","actual_charged_seconds":2.600000000000001,"actual_lease_status":"RECONCILED","finalization_receipt_exists":false,"framework_calls":0,"actual_launches":0}
```

The actual known cost is conservatively charged; this is not a zero-cost erasure. The defect is that the error record/contract claims unknown work with a retained full reservation and held lease when the scheduler is terminal and released. A subsequent distinct qualified job could see no active unknown reservation, while the reported finalization says one is retained. The supplied publication-failure test injects the earlier PROBE_RESULT write and does not reach this branch.

Verification criterion: choose a coherent explicit finalization policy. Either safely retain admitted UNKNOWN/full reservation until required publication is durable, or distinguish a proven-exit/proven-cost terminal worker from a publication-only failure, report the actual terminal state/released reservation accurately and offer bounded idempotent no-worker recovery. Never set reservation_retained from the old admission flag. Bind recovery to existing actual artifact/scheduler bytes, accept only identical already completed output, reject conflicting bytes, retain all errors/costs, and never redispatch/reset this fixed KEY. Inject final receipt failure, post-reconcile/pre-release failure, release failure and lost publication response; the recorded status/state/cost/lease must agree. Do not edit old native/operations authority or relax any launch/resource/qualification gate.

## OID2-OPS-002 — P2: reviewer aliases bypass builder separation and exact role policy

Evidence: checked_reviews (275–288) rejects only the exact stripped `/root` and `/root/scientific_audit` strings. It does not enforce the contract’s independent_reviewers_required role map or normalize recognizable canonical aliases.

Independent executed temporary-JSON reproducer, paired with `/root/data_audit` operations review: `/root/scientific_audit` science review rejects; `scientific_audit`, `\\root\\scientific_audit`, and `/root/scientific_audit/` all pass the production gate with identical status/binding. The independent scientific reviewer reproduced this separately. These are builder identities, not an independent science approval.

Verification criterion: require exact canonical expected reviewer identity for each frozen role (`/root/runtime_audit` science and `/root/data_audit` operations), or another explicitly frozen role policy that rejects all builder aliases. Test root/original-builder and repair-builder bare/slash/backslash/trailing-separator/whitespace variants, wrong role and duplicate reviewer. Retain the exact independent artifacts and source/contract binding. Name strings and hashes remain logical provenance checks, not same-user OS authentication.

## Limits and final bounded gate

This round supports the repaired cooperative termination, hash bounds, exact scheduled-job authority and complete phase instrument under their tested scopes. It is not approved while the two findings remain. The actual probe still requires the final independently reviewed source, both role artifacts, clean exact commit, unchanged native/registry pins and fresh RAM/GPU/process/disk admission plus the canonical one-worker lease/reservation. No source review or successful identity observation qualifies native models, reader, capture, training, weights, causal mechanism or original culprit. C-side effects, native threads and missed/unresolvable notification identities remain unavailable coverage, not zeros.

The Root-designated third round is this distinct probe milestone's final repair. Close these targeted controls against new frozen bytes; unresolved material issues retain quarantine instead of opening a fourth repair or renaming the old native path. No broader features or additional model experiment are requested by this report.

Machine-readable equivalent:

```json
{"review_id":"CRS-OWNED-IDENTITY-OPS-REVIEW-002","reviewer":"/root/data_audit","builder":"/root/scientific_audit","status":"REPAIR_REQUIRED_NOT_ADMITTED","probe_sha256":"946b4fbc6cbf013577b1a0e827e902c85a9cdef8382e9a5cb83c687f831d28e9","tests_sha256":"da0c05c7319ec3f90368ed54057791b1a42ef072344f1327f24bed606d3df278","contract_sha256":"651cdc696eaa08e1c1b1564542d82238e7e9b57c5373516a8d508742114258f7","supplied_CPU_tests_passed":17,"actual_owned_stdlib_fixtures":6,"actual_import_probe_executed":false,"framework_calls":0,"model_calls":0,"GPU_API_calls":0,"canonical_host_DB_accessed":false,"findings":["OID2-OPS-001","OID2-OPS-002"],"native_qualified":false,"reader_qualified":false,"probe_dispatch_approved":false}
```
