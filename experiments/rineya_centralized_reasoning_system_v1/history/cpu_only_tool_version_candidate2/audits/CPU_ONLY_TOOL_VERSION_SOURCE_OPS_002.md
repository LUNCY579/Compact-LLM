# UV metadata controller candidate2: OPS frozen source review

2026-10-05. Reviewer `/root/cpu_runtime_control_ops`; coordinator `/root`.
Shared model/workspace/prior OPS advisories and candidate1 review are disclosed.
I authored none of candidate2 or its fixture/contract. This create-only review
reads frozen source/tests/contract/raw evidence/handoff and hashes the declared
files. No replay, imports, tests, UV/OS worker, framework/model, current host/
process/GPU/DB, network or source edit occurs. Only this report is written.

**Verdict: SOURCE_NOT_ACCEPTED_CANDIDATE2; ONE_CONCRETE_FINAL_CANDIDATE3_CORRECTION
JUSTIFIABLE.** Candidate2 provides meaningful library improvements and an honest
NOT_READY boundary. A failed-termination/no-exit-proof cleanup path still drops
the exact process ownership witness. Keep the32 passing groups and candidate2
bytes unchanged. No actual key/launch or whole-path acceptance follows.

## Frozen identity, authorship and measured evidence

| Artifact | SHA256 |
| --- | --- |
| FINAL_FREEZE.json | `924c6716cb517104ed61defd21c53d937d1457fd65a074e110def0e657c1ea69` |
| cpu_only_tool_version.py | `d3fe7660ca206ae88f23be34604145b1b8031e5a99af65c08ffa1076bc968aa8` |
| test_cpu_only_tool_version.py | `5dc7f6db06da5721d9e38bd2d8ff5205676f8489b3108c5e0d4b54b16aac9955` |
| CPU_ONLY_TOOL_VERSION_V2.json | `3edc3fff2ba27e3d9d07c4c669bd7d037259ee52f8874d614131b2b600f9c5a0` |
| READINESS.json | `72335a1e782ccd79803865222041f44e0bbaaf500c32cb2dab62c0d3b644821a` |
| TEST_RESULT.json | `33bab2e2110c14caae0a6074016d909f8dce5748e6e7056b98da5b70eff07558` |
| RAW_TEST_OUTPUT.txt | `48ae123acd557cb2e12e7f90f69e744f847c22887edd8333b20a4b2622958e12` |
| RAW_TERMINAL_OUTPUT.txt | `308c04d4c1849ed50cb155c6aa797d12cb5c1a0245c9bd1a50508f403083cce6` |
| Correction008 | `71aadbd50075e0a637b4821c31a465fe3ea78fa66a6de8b9c0c043bf253f5e90` |
| Same-candidate builder handoff | `cb055faf2375336b7f592d86a07ccb5644e02a8baf37dda2feb67bfad16e6c19` |
| Candidate1 preservation | `c18b44aea78fe55cb5b9230c29036be4c9b32fb9f1405a1085780d399f984e09` |

All13 final manifest members match physical hashes/lengths. Candidate1's previously
verified20-file preservation, old temporal discrepancy and excluded raw HTML
custody remain retained. The handoff distinguishes prior builder and finisher,
preserves an explicit prior no-execution statement plus partial construction,
and admits that absent results alone are not proof of no invocation. It records
zero known prior fixture invocations and one remaining same-candidate dose;
contrary evidence requires reconciliation, never a counter reset.

The sole invocation returned Python exit0 with32/32 groups,0 failures/errors.
Raw suite time is0.546s; suite/import interval0.844s; external process-return
interval0.9313759s. Complete terminal capture was closed/flushed before the final
immutable freeze. Resumed capture interval1.0168909s and freeze-creation boundary
1.0624363s are distinct from Root's supplied complete-through-freeze1.0699511s/
tool1.334506s. Interrupted original construction wall is unavailable/disclosed;
these resumed intervals do not manufacture it. No fixture replay, source edit
after outcome, real UV/host DB/network or actual native reservation is reported.

## Candidate1 closure at library scope

OPS-TV1-001's demonstrated partial-constructor/first-identity defect is materially
corrected: Deadline owns the partial WindowsPort before later failures; OwnedJob
registers ownership as soon as its handle exists; acquired pipe/process/thread
handles are stored before post-call refusal. First identity failure no longer
requires a successful identity re-query before terminating the exact created
handle. Cleanup steps do not skip every later resource after one close error.
Meaningful typed failure controls cover these specific paths. This does not close
the failed-termination ownership path identified below.

OPS-TV1-003's broad missing-adapter coverage is materially improved: actual typed
CreateProcessW/PROCESS_INFORMATION handle ownership, inheritance list, identity,
exact primary ResumeThread, pipe byte/EOF, memory/peak and error paths are exercised
through controlled kernel/psapi calls. Pointer-width handles exceed32 bits; no
global Toolhelp thread scan is used. Both EOF witnesses are required; unavailable
post-exit peak RSS refuses rather than become an assumed production witness.
Resources survive in a bounded journal. The controlled killed-event timer case
demonstrates callback execution while a fake ReadFile blocks; it does not prove
all actual OS/SQLite calls terminate promptly.

OPS-TV1-002's early canonical-success publication is removed. The library does
not reconcile SUCCEEDED/FAILED; it retains the full10 reservation in OUTCOME_UNKNOWN
and separates observed worker interval from caller internal-finalization wall.
A slow durable-receipt control remains late/unknown without shortened canonical
charge. Actual OperationsStore temporary-DB controls now reject zero dispatched
time, wrong owner/fence/checksum/request/worker/exit/duration proof and exercise
immutable terminal transitions/unknown full-cost retention. Their synthetic
observer flag is not independent real provenance. The complete10s operational
caller/observer finding remains open, rather than being closed by permanentUNKNOWN.

## OPS-TV2-001 — P1: failed termination or unproved exit still loses exact ownership

The specific source path is WindowsPort.cleanup273–299 and observe_version
finally398–413. Failed Job/created-process termination is recorded, but cleanup
continues to close Job and process handles. It does not require a signaled exact
process and owned-Job-empty proof before releasing those ownership witnesses,
or transfer them through an implemented acknowledged observer handoff.

Concrete pre-Job case: CreateProcess returns a suspended process; first identity
or assignment fails before the process joins the dedicated Job. If its exact
TerminateProcess call fails, the source records the failure, quiesces/cancels the
timer and still closes the known process handle. Closing that handle does not
terminate this live process; dedicated Job termination/kill-on-close cannot cover
a process never assigned to that Job. The original stable recovery handle is
then gone. Full reservation and an error string preserve accounting uncertainty
but do not preserve usable process ownership or remove the suspended orphan.

The deadline/no-exit-proof branch also closes these handles after attempting
termination without proving exit. A future observer is mentioned, but the source
contains no completed ownership transfer. PID/creation/image metadata cannot be
substituted for the just-created process handle or justify an arbitrary later
PID kill. Failed CloseHandle itself retains its value correctly; the defect is
the **successful close of a process whose failed termination/unproved exit still
needs that handle**, not a claim that every cleanup close drops its value.

The fixture does not demonstrate the decisive live-failure case. Kernel.terminate
128–130 sets exited=True/job_pid=None before returningFalse when fail=='terminate'.
Kernel.kill_job similarly applies successful exit side effects before a possible
failure return. No retained test requests the faithful condition where termination
fails and the process remains live. Passing mocked cleanup cases therefore cannot
close this source gap; no runtime was executed to derive it.

Minimal final candidate3 acceptance criterion: retain exact hProcess and relevant
Job/thread ownership safely and explicitly pending until signaled exit/required
Job proof is established or a concrete qualified observer acknowledges possession.
Continue safe independent pipe/attribute cleanup without discarding the recovery
witness; record live/unknown membership, retained handles and full cost. An
unimplemented observer promise is insufficient. A finite typed control must make
failed pre-Job TerminateProcess genuinely leave the worker live and prove no
process-handle close/lost owner, no resume and no unrelated-PID termination. Cover
unknown wait/signal, deadline-expired exit absence, Job termination failure and
failed ownership/close/handoff branches with the same bounded fixture dose.

## Timer, state and endpoint limitations

On ordinary WindowsPort cleanup, quiesce cancels/joins the callback before process/
Job handles close, so the demonstrated normal paths avoid callback reuse races.
The finally block's earlier termination calls can run concurrently with timer
termination, but do not themselves close those handles. An unquiescent callback
causes cleanup to retain handles and report an error. That is conservative;
source must make their retained owner accessible to the concrete observer.

The outer finally calls only guard.cancel, not unconditional quiesce. Paths with
no returned/owned port or unquiescent cleanup do not establish that no timer
survives function return, and no timer-quiescence result is included in the
returned record. Cancel alone is not a join. The final correction/whole-caller
boundary must report timer ownership/quiescence and preserve live callback/handle
state without closing or reusing its targets. Do not claim every timer branch
was proved safe by the one successful blocked-ReadFile control.

State recovery sensibly re-reads RESERVED/RUNNING/OUTCOME_UNKNOWN after a reserve
call exception and attempts mark_unknown, retaining the canonical full10. If
that recovery read/write also fails, the ledger may remain RESERVED with no worker
and result reservation knowledge is incomplete. Its ledger_errors must prompt
external reconciliation; neither absent worker nor a None result field permits
release/retry. This is not an observed host-state mutation here.

The independent timer limits known owned worker activity, not blocked caller/
SQLite/publication completion. The source explicitly admits it cannot abort
those calls and leaves terminal serialization/external observation incomplete.
It publishes a narrower worker-exit receipt, not complete operational finalization.
The temporary-DB success control reuses that worker interval to demonstrate the
canonical interface only; actual complete caller cost/proof must not be reduced
to that synthetic transition. A qualified observer must bind actual whole-call
completion and durable evidence honestly, including any overrun, with immutable
receipt identity compatible with the preexisting observation. Holding10 indefinitely
is not an implemented successful endpoint or permission to redefine the scope.

## External gates and finite disposition

Admission still accepts booleans/nonempty source/runtime strings and caller-
supplied physical-path assertions; no current clean commit/role/UV/search-root/
canonical host DB/floor/lease proof was established by these fixtures. The actual
entry point continues to refuse launch. Before any actual key, freeze and
independently qualify the bounded caller plus complete observer, exact reviewed
source/contract/runtime receipts, physical paths/loader/native DLLs, metadata-
tool trust, fresh resources and canonical request/worker/owner/fence/outcome
bindings. No OS filesystem/network/device secrecy or UV publisher/source proof
is inferred from Job/PATH/environment checks or an eventual emitted version.

One separately recorded **final candidate3** may correct OPS-TV2-001's retained
ownership/teardown/timer-state branches and the faithful live-failure fixtures,
within the same maximum3. Preserve candidate2 exactly; one targeted stdlib/
typed-fake/temporary-DB invocation<=60s, one complete terminal-before-freeze and
fresh independent reviews. Do not add a generic helper, rerun32 tests as a new
unregistered dose, change UV/argv, expand documentary/system scans or reset caps.
Candidate3 cannot claim launch completeness while caller/observer/trust/admission
remain external. Rejection at3 requires quarantine, no fourth repair.

Unchanged limits remain one future10s/no-retry --version, combined8KiB plus one
overflow sentinel, ActiveProcessLimit1/zero descendants,2GiB RSS and separate
2GiB Job commit, all host/disk/freshness/numerical-thread gates and cumulative
7200s epoch/14400s rolling24h with47.421s historical consumption. Old CPU-loader/
GPU/native/reader/acquisition quarantines, keys/Probe002, read-only weights and
CHECK/FINAL remain intact. No actual tool/bootstrap/install/model/learning result
or full-path qualification follows from this library review.

```json
{"review_id":"CRS-CPU-ONLY-TOOL-VERSION-SOURCE-OPS-002","reviewer":"/root/cpu_runtime_control_ops","status":"SOURCE_NOT_ACCEPTED_CANDIDATE2","freeze_sha256":"924c6716cb517104ed61defd21c53d937d1457fd65a074e110def0e657c1ea69","source_sha256":"d3fe7660ca206ae88f23be34604145b1b8031e5a99af65c08ffa1076bc968aa8","tests_sha256":"5dc7f6db06da5721d9e38bd2d8ff5205676f8489b3108c5e0d4b54b16aac9955","contract_sha256":"3edc3fff2ba27e3d9d07c4c669bd7d037259ee52f8874d614131b2b600f9c5a0","frozen_members_verified":13,"fixture_invocations":1,"retained_tests_passed":32,"library_improvements":["partial allocation/first-identity known-handle ownership","typed adapter/EOF/peakRSS/resource journal","conservative full10/temp canonical DB and no early terminal success"],"remaining_material_source_finding":"OPS-TV2-001 failed-termination/unproved-exit process-handle ownership loss","whole_caller_observer_accounting_closed":false,"timer_every_return_quiescence_proven":false,"finite_final_candidate3_correction_justifiable":true,"source_or_launch_approved":false,"actual_UV_hostDB_framework_model_network_actions":0,"reviewer_replay_import_test_live_actions":0,"old_counters_keys_quarantines_preserved":true}
```
