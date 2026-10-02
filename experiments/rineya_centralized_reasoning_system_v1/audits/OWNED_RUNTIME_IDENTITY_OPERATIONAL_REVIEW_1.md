# Owned runtime identity diagnostic: independent operations review 1

Reviewer: `/root/data_audit`. Builder: `/root`. Client date: 2026-10-02.
Disposition: **REPAIR_REQUIRED_NOT_ADMITTED**. This is a new, separate resolving instrument, not a fourth repair of the quarantined native candidate. No actual probe, framework import, model, tokenizer, tensor, GPU API, or canonical host-state database was executed/accessed by this review. No executable registry or native source was changed. The reviewer owns only this report.

## Frozen inputs and scope

The complete implementation, tests, current contract, preceding scientific design, preserved native helpers and imported WindowsJob implementation were read directly. The following were independently hashed before controls:

| Input | SHA256 |
| --- | --- |
| `owned_runtime_identity_probe.py` | `ebee87e14e10b602bdd56e9ff3a920a107b361def71baa1c0043f1953c83fb35` |
| `tests/test_owned_runtime_identity_probe.py` | `2dff8979d6881c0d1829581e6ee2adf09ac140d691fbc757e5fc01d028a481e7` |
| `contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json` | `663ed38a22215808cc14953364bed91fc4e5e6f476e8621a59e117fce8aaa9aa` |
| Preserved `native.py` | `8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3` |
| Preserved native contract | `d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a` |
| `operations.py` | `ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30` |

This is an uncommitted stage review. The contract source_commit is null; a later clean, exact-commit attestation is still required. The inherited native source pins include the central WindowsJob (`d31370de4f6b09052fac2f3505fc38a40041e76bd2137435a1f3678f4346a54d`) and its contract module. These are reused mechanisms, not newly discovered containment. Native whole-path, reader, training, and numerical qualification remain closed.

## Controls and sound mechanics

The registered interpreter ran all 7 supplied stdlib tests successfully in 0.042 seconds. Independent controls used only mocked executable/Win32/telemetry boundaries and temporary SQLite under CRS ignored outputs. Real OperationsStore admission, reconciliation and budget accounting were exercised in those temporary stores. Synthetic source/review/telemetry acceptance was deliberately substituted to reach downstream failure boundaries; these are adversarial CPU fixtures, not evidence of actual Windows containment or framework behavior.

The IOCP declarations use HANDLE, ULONG_PTR-sized key storage, DWORD message storage and an output pointer-sized value consistent with the documented APIs. Association uses the exact WindowsJob handle and occurs before resuming the suspended launcher. A foreign notification key and a failed port call with error6 independently rejected; FALSE/NULL/error258 produced the expected empty nonblocking poll. The implementation correctly labels notifications as non-guaranteed and not sufficient process identities. [Microsoft association semantics](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_associate_completion_port), [GetQueuedCompletionStatus](https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-getqueuedcompletionstatus).

The suspended-create → assign exact Job → install notifications → raw inventory → strict guard → resume ordering preserves the previously reviewed launcher containment principle. The child closes a valid unconsumed query handle on GO/ancestry/lease rejection; the delegated native exact-Job witness closes a consumed handle in its finally block before imports. The strict flags and executable map were not weakened. No actual Win32 notification run was authorized for this review, so declaration inspection and mocks do not establish runtime IOCP delivery.

An injected Popen response loss after launch_attempted persisted OUTCOME_UNKNOWN with no actual_seconds and charged the full60-second reservation. A later attempt against that key made zero Popen calls, retained OUTCOME_UNKNOWN, and retained60 charged seconds. Verified terminal overruns were conservatively charged rather than erased. Blank/root/duplicate-role review checks reject; both separate review roles and exact source bindings are required by the public supervisor. Unknown-state retention and no duplicate launch are sound in the tested paths.

## OIDENT-001 — P1: a catchable profile exception removes the API boundary

Evidence: import_boundary, lines61–84, appends a violation then raises NativeBlocked from the profile callback. Child boundary checks occur after import returns; the supervisor reacts on a later observation. Python disables a profile callback when it raises. [Python profile behavior](https://docs.python.org/3.11/library/sys.html#sys.setprofile).

Independent reproducer: create an ordinary Python namespace with `__name__='torch.cuda'` and `is_available()` appending to a list. Install import_boundary; catch the first NativeBlocked; inspect sys.getprofile; call the same function again. Observed: first body blocked; `sys.getprofile() is None`; second call appends `executed`; only one violation exists. No Torch module was imported. A library that catches the first exception can therefore execute later forbidden work before milestone/parent checks. The new-thread fixture proves the first rejection in a new Python thread, not continued enforcement after that rejection.

Related coverage counterexample: forbidden_call('torch.serialization', 'load', 'call') and the equivalent save call both return False, despite the declared checkpoint-load/save coverage. C-call load/save names alone do not reject their Python wrappers. Generic C constructors and internal/native-thread side effects are explicitly admitted gaps; they cannot be converted into proof of zero side effects.

Verification criterion: the first detected forbidden operation must cause a durable, terminal child abort which cannot be swallowed and continued, while the parent still proves exact owned-tree cleanup and accounts the failure. Do not replace prohibited calls with fake successful results or mutate library code. Test caught Python and caught C-call refusal, new-thread refusal, failed refusal-journal publication, Python serialization wrappers, and no second forbidden side effect. Record any unobservable C/internal boundary honestly.

## OIDENT-002 — P1: the registered byte, deadline and observation bounds are not enforced

Evidence: inventory calls native.sha(executable) directly at line175 for every sample. No64MiB per-executable or256MiB distinct-byte bound, stable identity cache or hashing deadline is implemented. The sole wall/resource loop performs synchronous inventory, fsync, strict rehashing and external telemetry. Its60-second check is at line374, before these calls; terminal success at407–415 does not recheck the deadline. There is no independent watchdog while this work blocks.

Independent reproducer: a full mocked supervisor with real temporary OperationsStore advances an injected monotonic clock by65 seconds during the observer inventory after the loop's deadline check, then returns an exited worker/empty Job and otherwise acceptable terminal markers. Observed status and persisted job state: SUCCEEDED; wall_seconds65.01000000000002; actual and budget charges65.01000000000002. The accounting is conservative, but the bounded-execution/completion claim is false. A separate inventory control proves unqualified `sha('C:/unknown.exe')` is invoked with no size/budget check.

Impact: an unexpected executable or slow disk/hash/GPU identity lookup can hold the worker beyond the declared wall/RSS/log guard cadence. Sleeping .02 seconds after blocking work does not implement a20ms observer; synchronous telemetry can also postpone the100ms strict guard. The report's non-exhaustive warning does not enforce the hard execution ceiling.

Verification criterion: enforce bounded executable identity hashing before reads, track aggregate distinct bytes, retain unavailable reasons, and reject bound overflow. A separate fail-closed wall/resource watchdog must stop the exact Job independently of slow observer/hashing/telemetry work; all actual delays/gaps must be timestamped. Late completion must reject, even if the worker finally exits0. Use deterministic blocked/slow inventory, hash, journal and telemetry controls; preserve full/actual costs and strict resource floors. Do not relax native source or registry rules to make the observer faster.

## OIDENT-003 — P1: incomplete or corrupted child evidence can be finalized as success

Evidence: lines409–413 load ordinary json.loads records and require only the final marker, ten source_end indices and any qwen_symbol_import_end with cuda_initializedFalse. They do not validate CHILD_CONTAINMENT, complete import begin/end order, baseline/framework exclusion, two verify_sources/two runtime_metadata observations, registered sources on each step, post-import observation, profile state, source/lease identity on final evidence, or a complete canonical journal.

Independent reproducer: injected child journal contains only ten `{phase:'source_end',index:i}` records, one qwen_symbol_import_end with false CUDA, and one final marker. There are no source_begin, idle, torch, package-import or containment artifacts. With mocked exit0/empty exact Job, the real supervisor returns0 and persists SUCCEEDED. The independent fixture reports CHILD_CONTAINMENT absent. This directly challenges the production terminal validator, not a standalone reimplementation of its predicate.

Impact: a partial/corrupted output can establish 'COMPLETED_SPECIFIED_NO_MODEL_PATHS' without evidence for the specified paths. Caller-declared zero-operation fields and false CUDA alone are not operational attestation.

Verification criterion: parse a bounded newline-complete, duplicate-key rejecting journal; validate the entire registered ordered phase schedule and exact10/2/2 evidence, model/no-CUDA boundary evidence, query-handle closure and source/contract/owner/fence/worker bindings. Bind the final receipt to actual immutable bytes of all required artifacts. Missing, duplicated, reordered, forged-source, wrong-worker, malformed-count and incomplete-final-record controls must reject before SUCCEEDED/release. Keep observed requests, sampled side effects and unobservable internals separate.

## OIDENT-004 — P1: child entry is not coupled to this single reserved diagnostic

Evidence: child lines227–243 and its event helper require only a live owner/fence lease, GO ancestry and a strict exact-Job query handle. They never load KEY's admitted job, require RUNNING/reserved60, or compare its request/source/contract/epoch/recorded worker with GO. Child entry does not require this diagnostic's review bindings or a clean source snapshot. Main's child path validates file location and contract bytes, then delegates to child.

Independent reproducer: a mocked child has an OTHER-MILESTONE-LEASE, no scheduled-job API at all, fake exact-Job witness/ancestry and fake import symbols. It performs all three import attempts and returns0 after29 lease-only checks; no actual framework import occurred. A different valid shared lease is therefore sufficient to reach the import phase, without proving this diagnostic's one60-second reservation or reviewed dispatch.

Impact: an accidental direct child launch or GO/lease mix-up can consume resources under another milestone's lease and bypass the public supervisor's review/admission/reservation checks. Exact Job containment does not establish the correct scheduled job. This is a cooperative protocol integrity finding, not a claim that same-user Python can be an adversarial security sandbox.

Verification criterion: before phases, bind the child to the current immutable KEY request, RUNNING state,60-second reservation, exact owner/fence/epoch and observed launcher/runtime PID+creation identity, clean committed source, contract and approved dispatch/review evidence. Reject a valid lease belonging to another request, unreserved/PENDING/UNKNOWN/terminal job, wrong worker/source/GO and direct child entry without the bound launch. All rejected valid temporary query handles must close before import.

## OIDENT-005 — P2: partial identity data are discarded and drift rechecks use cached attributes

Evidence: inventory obtains creation time/path/member before hashing, but records them only after hashing and RSS succeed (179–181). A PermissionError/exit afterward leaves only PID plus query_error. Rechecking create_time/exe on the same psutil.Process at176 does not refresh either value: the pinned psutil7.2.2 implementation caches both attributes. Guard records have no linked observation IDs/current import phase, and initial/target guard failures do not receive the same begin/end/refusal journal treatment as the later RSS guard.

Independent reproducer: mocked Process returns known creation123/pathC:/unknown.exe/exact memberTrue, then sha raises PermissionError. The retained row contains only PID99, timestamp, error and non-inference flag; its already-observed path/time/member are lost. The pinned psutil source documents/implements the cache at `__init__.py` exe704 and create_time775.

Impact: the resolving instrument can lose the exact unexpected path it was created to observe, and a reused PID can be assigned stale attributes if the hash spans an identity change. The native strict guard remains unchanged; this finding concerns diagnostic provenance and cause attribution.

Verification criterion: publish each successfully resolved field and component timestamp before later risky queries; associate identity with a retained process handle or independent fresh PID/creation observation; record drift/disappearance explicitly. Link every strict guard, including bootstrap/target guards, to durable raw observation/notification IDs and the best available import-step evidence. A failed hash or RSS must preserve known path/time/member, while unresolved/reused notification PIDs never acquire an inferred old identity.

## OIDENT-006 — P2: cleanup exceptions can skip durable unknown finalization

Evidence: finally420–430 calls job.close, notifications.close, process.poll/kill/wait and pipe.close outside independently protected finalization. An exception there skips result publication, mark_unknown and store.close. The underlying active reservation is conservatively retained, so this is not a zero-cost loophole; it leaves RUNNING/RESERVED until later explicit stale reconciliation and loses the immediate failure record.

Counterexample: inject a job.close, notification.close, poll or wait exception after an admitted worker. Python exits this finally block before line432's uncertain classification and lines441–465 publication/accounting. The existing unknown reservation safety prevents a later valid automatic launch, but the registered durable OUTCOME_UNKNOWN/failure-and-cost instrument is incomplete. This path was inspected directly; no real cleanup failure was induced.

Verification criterion: isolate each cleanup action, preserve all errors, independently prove exact owned-tree exit or mark OUTCOME_UNKNOWN, then attempt durable result/accounting publication and close the store. No failed cleanup may fabricate successful exit or no-dispatch proof. Tests must include cleanup/receipt/disk failures and show reservation/lease retention with no duplicate launch.

## Review limits and next gate

No actual framework import, GPU API, process-launch/notification smoke, model asset or canonical host database was used. Python profile controls are cooperative observation/termination mechanisms; they do not contain malicious C code. IOCP delivery/identity remains non-exhaustive. Current resource observations reported to the team were below the unchanged admission floor; the design and this review grant no launch permission.

A first reviewer fixture incorrectly used OperationsStore as a context manager, then a restart fixture omitted ready(). Both harness errors were corrected and excluded from the production findings. The first left one ignored temporary SQLite at `experiments/rineya_centralized_reasoning_system_v1/outputs/tmpigyd6q25/host.sqlite3`; the interpreter exited and no canonical database was involved. Automatic policy review rejected recursive removal of that exact verified fixture directory as 'blocked by policy'; it was retained rather than retrying deletion. All successful follow-up fixtures closed their stores and removed only their verified own temporary paths.

This frozen candidate is not approved. Close the six findings on a new frozen diagnostic implementation/contract and independent targeted controls, then obtain both review roles, a clean exact source commit and fresh unchanged admission/one-host-wide lease/reservation. Old native failures, numerical results, source, registry, cumulative costs and quarantine remain authoritative and unchanged. A resolver observation alone grants no whitelist expansion, native qualification, reader qualification or training authority.
