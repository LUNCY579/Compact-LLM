# Pinned UV metadata controller candidate1: OPS source review

2026-10-05. Reviewer `/root/cpu_runtime_control_ops`; coordinator `/root`.
Shared model/workspace and prior scope/artifact/design advisories are disclosed.
I did not author the candidate or run its fixtures/documentary collection. This
create-only review reads frozen source/contract/tests/raw evidence and hashes
declared retained files. No imports, replay, tests, current host/process/GPU/DB,
UV, framework/model, network or implementation action is performed. Only this
report is written. Source itself remains NOT_READY and provides no actual CLI;
the defects below have not been exercised against a real worker here.

**Verdict: SOURCE_NOT_ACCEPTED_CANDIDATE1; FINITE_CORRECTIVE_CANDIDATE2_JUSTIFIABLE,
NO_LAUNCH.** Candidate1 honestly discloses missing interface/caller/trust proof.
The source has concrete cleanup/deadline defects beyond those disclosures.
Preserve its exact bytes, one7-group fixture and all documentary/temporal records;
do not retest or patch this frozen candidate in place.

## Exact frozen identity and observed dose

| Artifact | SHA256 |
| --- | --- |
| FINAL_RETENTION_FREEZE.json | `a277fce07028f7133c3b76f22ca71d95a600f6eae92d597522a18f765d5258ac` |
| cpu_only_tool_version.py | `6df8b6e65c7e43b96e050708769320c69b751e8e4bca58ae7df393f72dae02c2` |
| test_cpu_only_tool_version.py | `aa481da13320b3486c9ede9b46ae0e38532b888df8bcba63ca42804ecc9a6a24` |
| CPU_ONLY_TOOL_VERSION_V1.json | `d0f7252084be04eb11c025018344bad897f161882ea0980163c4a760ab9bb2c3` |
| READINESS.json | `0c6a2fcde4e4ea9bd4365e433eda05b6dd2cefeb52431250a30db3736e1ea17d` |
| RAW_FIXTURE_OUTPUT.txt | `8c4a1cf5991ed85015ad0c39c39e3a2114bfb719335bd7501458cb37793e8e6b` |
| FIXTURE_RESULT.json | `1f3e4629939db454a09d51f571ca469ddf0e420dfa84c21fac9415e05139f6ea` |
| FIXTURE_COMPLETION_STDOUT.json | `ed7343e8c4bd3533d287379aa679dcb44dd6ed2e040e299555ac83e0e218b9f8` |
| Initial fixture FREEZE.json | `5beb10f9dcfe69f90098a81d5355ed73bfef3825cb524b868d2f5283d3890ec2` |
| Prior FINAL_FREEZE.json | `2cb5d601352c794b0e96fef457440c316ef47269768a250cf6a5bf971ff90ccb` |

Physical source/test/contract/readiness hashes match. All19 final-retention
manifest files match declared hashes/lengths; identical HTML and .html.txt copies
are retained with originals. One synthetic invocation reports7/7 groups,0.007s
suite time,0.31199999999989814s through first freeze and0.3743525s external whole
invocation. It exercises OwnedJob structure/config/assignment calls and mocked
Popen arguments, then substitutes a synthetic Port for identity/resume/pipe/RSS.
Those passes are not actual Windows/UV or complete typed-interface qualification.

Three selected Microsoft responses total178240 bytes,1.1409999999996217s through
documentary freeze/1.5597334s external command, within the registered6-response/
2MiB/120s dose. No duplicate refetch, fixture rerun, actual UV key/worker or current
DB action is reported. No native outcome or47.421s budget reset is inferred.

## OPS-TV1-001 — P1: partial constructor and pre-Job failures lose cleanup

WindowsPort.__init__ constructs OwnedJob before binding later kernel identity and
pipe API signatures. If one of those later operations raises, port_factory never
returns; observe_version's `port` remainsNone and its finally block cannot reach
the allocated Job. OwnedJob's own constructor cleanup covers its configuration
failure, not a subsequent WindowsPort construction failure. No constructor-wide
ownership/finally guard closes this partial resource.

At source196, spawn_suspended returns a known Popen process before the initial
identity query completes. If identity raises, `identity` remainsNone. Finally
calls kill_prejob(p,None), which re-queries identity and requires equality with
None. A successful later query produces a dictionary and fails equality; another
query failure also refuses. The known just-created suspended process is therefore
not terminated on this path. The broad cleanup try then skips subsequent pipe
and Job closure. Its unknown/full reservation classification is conservative,
but reservation retention does not substitute for known-handle cleanup.

Correction criterion: record each owned resource as soon as it exists, close a
partial port constructor independently, and use the exact created suspended
process handle for pre-Job termination when identity collection fails. Do not kill
an unrelated PID by name or discard a handle because an observation failed.
Make process termination/wait, pipe closure and Job closure independent bounded
cleanup steps; retain each failed handle/exit proof and unknown cost. Meaningful
fake-kernel fixtures must fail each allocation/typing/identity/assignment/resume
phase and demonstrate the correct owned cleanup with no resume or unrelated kill.

## OPS-TV1-002 — P1: combined deadline/accounting finalization is not enforced

The mainline checks clock around operations but supplies no independent10s guard
over blocking process/kernel/store/pipe work. The global Toolhelp thread loop has
a65536-entry cap but no in-loop deadline; a slow call/loop/ledger transaction can
pass the deadline before the next check. Canonical OperationsStore itself has a
5s busy timeout; successive calls can consume or exceed this tiny total dose.
The declared0.1s watchdog is currently the mainline's polling cadence, not an
independent guard while that line is blocked.

Source242 samples elapsed before durable outcome receipt/reconciliation252.
The success decision247 and recorded elapsed may therefore remain predeadline
while fsync/ledger finalization completes after10s. The written receipt/ledger
can claim SUCCEEDED and charge an earlier time despite later terminal work.
There is no atomicity between an earlier clock test and later durable publication.
The metadata result also remains SELF_REPORTED_VERSION_ONLY if later elapsed
causes a FAILED ledger outcome, making the two status layers easy to misread.

Correction criterion: one timer covers all declared preparation, launch, output,
owned exit and terminal work. Add independent bounded supervision for the owned
worker and deadline-aware kernel/stream/poll phases; account for slow/blocked
ledger work honestly. Do not freeze the clock before receipt writes or invent an
atomic durable-success check. Require actual final completion measurements and
consistent failed/unknown/late dispositions. If publication cannot certify final
completion before10s, hold the reservation/retain partial evidence and let a
separately identified observer reconcile the actual late outcome; record actual
overrun rather than silently undercharge or label it success. Original10s stays.
Include typed slow-call/snapshot/pipe and durable-receipt/reconcile deadline
crossing controls in the one corrective fixture invocation.

## OPS-TV1-003 — P1: critical adapter paths have no meaningful interface proof

The synthetic Port returns predetermined identity/output/RSS and immediately
exits at resume. Tests do not call actual WindowsPort identity, resume, pipe/RSS
or constructor failure through a complete controlled kernel/psapi adapter. They
therefore cannot expose001/002 or establish signatures, thread enumeration/
suspend-count handling, pipe errors/EOF/bounded reads, live/exited RSS failure,
explicit process-handle closure and non-skipping cleanup. Failed fake assignment
also sets its synthetic Job PID before returningFalse, unlike a useful refusal
model, so it cannot prove actual membership/cleanup behavior.

Output has no explicit EOF witness: the controller sets exit_proved after one
extra drain based on process/Job state, while output only returns chunks. Source
comments claim EOF evidence without retaining it. Provide a declared complete
stream/producer/exit proof, explicit capture termination and bounded final drain,
rather than assume mocked empty output is actual pipe EOF. Resource decisions
and their measured times/values should survive the final journal; current result
retains Job flags/worker identity but no complete sampled RSS/host/disk/telemetry
decision evidence. Sampling is not a guarantee of every unsampled peak.

Correction criterion: exercise the real typed adapter functions with finite
fake kernel/process/stream/clock/store inputs, including exact success, each
signature/error/snapshot/suspend/EOF/RSS branch, stale resources and cleanup.
Assert meaningful ordered decisions and retained measurements/outcome bindings.
No actual Win32/UV/process observation is needed for that fixture dose.

## External gates, documentary interpretation and temporal evidence

These prerequisites cannot be discharged merely by repairing the adapter.
validate_admission accepts booleans and nonempty free-form source/contract/runtime/
admission strings. The gate declares physical paths verified but never hashes UV;
the injected store is not checked against the canonical host DB/source object.
Only operations.py's file hash and job request/epoch/READY fields are checked.
The final entry point refuses launch and the docstring intentionally delegates
this to a future qualified caller, so this is not an observed unauthorized run.
Before source acceptance for an actual path, freeze that caller or enforceable
boundary: immutable nonbuilder role receipts, exact clean commit/source/runtime/
contract/UV/path pins, canonical DB identity, actual reservation/RUNNING owner/
fence/epoch/request/worker binding and fresh floor telemetry before resume.
Synthetic True fields and a passed-in fake store are not those proofs.

Path.is_relative_to is a lexical check, not canonical reparse/handle containment.
The future caller must validate exact empty roots and ancestor/file identity
immediately before use. Current kernel32/psapi controller DLL paths also need
the selected runtime binding. `command_hash` hashes requested argv; it is not
an observed OS command-line reading and must be labeled accordingly.

The retained Microsoft pages support the limited documentary interpretation:
API sets separate contracts from host implementation DLLs, loader ordering
includes API-set handling, and selected WaitForSingleObjectEx metadata names
KernelBase and the synch contract. This explains why005 exact-key lookup failure
was not an incompatibility result. It does not prove local alias forwarding,
complete UV/API-set resolution or all internal-C behavior. No nearby schema
name substitution or generic system expansion follows.

Pinned local UV metadata trust and Windows loader trust still need explicit
independent disposition. Job/PATH/one-process/environment controls are not OS
network/config/filesystem/device secrecy. Rust thread count remains unenforced;
numerical two-thread limits must not be promoted to UV concurrency proof. A
version result would still be an emitted lookup key, not publisher/source/setup
qualification. These external trust/admission facts remain launch gates.

Initial fixture FREEZE records FIXTURE_COMPLETION_STDOUT as0 bytes/empty SHA,
while the physical file is now128 bytes/ed7343e8. The completion output is written
after that first in-process freeze by terminal capture, so the initial manifest
cannot be asserted to match today's complete stdout. Preserve this temporal
discrepancy and the initial bytes; the accurate final-retention manifest binds
the completed file. Raw outcomes/source were not edited or rerun to resolve it.
Record future in-process freeze and external terminal capture as distinct phases
with an explicit final retention binding; do not erase the old manifest.

## Minimal finite candidate2 disposition

One separately registered corrective candidate2 within the same default3 may
address001–003: resource ownership/non-skipping cleanup, independent deadline and
honest final accounting, complete typed fake-kernel adapter/stream/resource proof
and explicit terminal retention. Keep exact UV/--version/10s/combined8KiB/
ActiveProcessLimit1/no-descendants,2GiB RSS and separate2GiB commit, all original
floors/freshness/canonical budgets, existing artifacts/source1/005/006 and keys.
Permit one targeted stdlib synthetic invocation<=60s, one freeze and fresh SCI/OPS
review. No UV execution, fixture replay, new documentary search, alternative
tool/flag/version, model or cap extension is needed by that correction scope.

Required caller/runtime/trusted-policy bindings must be concrete before exact-
commit/dispatch acceptance; do not mark them closed because candidate2 tests pass.
A refusal/unknown observation would still consume the eventual one registered
attempt and retain cost. Candidate1 consumes source round1, not an actual tool
attempt; candidate2 is not an old CPU-loader fourth repair or a counter reset.

```json
{"review_id":"CRS-CPU-ONLY-TOOL-VERSION-SOURCE-OPS-001","reviewer":"/root/cpu_runtime_control_ops","status":"SOURCE_NOT_ACCEPTED_CANDIDATE1","final_retention_freeze_sha256":"a277fce07028f7133c3b76f22ca71d95a600f6eae92d597522a18f765d5258ac","source_sha256":"6df8b6e65c7e43b96e050708769320c69b751e8e4bca58ae7df393f72dae02c2","test_sha256":"aa481da13320b3486c9ede9b46ae0e38532b888df8bcba63ca42804ecc9a6a24","contract_sha256":"d0f7252084be04eb11c025018344bad897f161882ea0980163c4a760ab9bb2c3","material_source_findings":["OPS-TV1-001 partial constructor/pre-Job cleanup","OPS-TV1-002 combined deadline/final accounting","OPS-TV1-003 critical adapter/stream/resource fixture gaps"],"retained_fixture_groups":7,"fixture_invocations":1,"documentary_responses":3,"documentary_bytes":178240,"initial_stdout_freeze_temporal_discrepancy_preserved":true,"finite_candidate2_correction_justifiable":true,"caller_commit_trusted_runtime_current_admission_accepted":false,"source_or_launch_approved":false,"actual_UV_attempts":0,"reviewer_import_test_replay_UV_network_host_DB_GPU_actions":0,"default_three_candidates_and_old_quarantines_preserved":true}
```
