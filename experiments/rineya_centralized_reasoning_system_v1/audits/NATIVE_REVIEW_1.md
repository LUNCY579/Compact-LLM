# CRS-001 native code/operations review 1

Client date: 2026-10-01. Reviewer `/root/data_audit`, not the native builder.
This review first inspected `native.py`, its CPU tests and diagnostic contract,
then the exact imported WindowsJob, stream-loader, call-accounting and dependency
source. Shared workspace privileges/model ancestry remain disclosed. No model
worker, model load, torch/CUDA execution, GPU telemetry qualification or canonical
outside coordination state was performed by this reviewer. The coordinator later
authorized one bounded stdlib-only process-containment fixture, recorded below.

**Initial disposition: REPAIR_REQUIRED; bounded native dispatch not approved
on the first inspected bytes.** The reviewer is checking repaired candidates in
the same bounded review milestone; final frozen bindings/disposition appear
below when available. A passing CPU mock is an instrument check only.

## First inspected source bindings

| Subject | SHA256 |
| --- | --- |
| original `native.py` | `cdd66c2fbc1085b6b12813c1c110d53fe0c81869fc79803c844f445b1e1b45c7` |
| original diagnostic contract | `7242ed791f009394bdcb0a26ab1bcac0d8418351e474ce542505453502da13d7` |
| imported central `stream_checkpoint.py` | `434c62698c2e397048db0d88a7959e34d5b8afb93f90cdf86f2aba852922c67f` |
| imported central `qualify_native.py` | `a0fbbbb67dcb7136270dd01efed16cccffc62b09342863f05d03d845805bf941` |
| its `laptop_profile.py` import | `2e8c9236aba1d48e222373c7484c353eec717ae176ce9bc108968b74d1b7e196` |
| imported Opallo `supervisor.py` | `d31370de4f6b09052fac2f3505fc38a40041e76bd2137435a1f3678f4346a54d` |
| its `contract.py` import | `bec805005c3732ce6b1fa86ff71c6c5d551dfbadf64db4c54a8d9900cab930fb` |

These reused components were read as source, not treated as a transferred native
qualification. The central source retains its own numerical/retention gates.
Installed runtime/checkpoint hashes remain the registrar's exact asset audit;
this reviewer checked that native verification binds them before loading and
after the diagnostic rather than qualifying assets from their names alone.

## Findings

### NATIVE-R1-001 [P1]: attempted process creation originally produced false zero/no-dispatch proof

**Evidence:** Original `supervise` set `worker_dispatched=True` only after
`subprocess.Popen` returned. Its finalization used that boolean alone to write
`never_dispatched=True`, `launch_attempted=False`, zero elapsed native cost.
The coordinator independently flagged this before dispatch.

**Impact:** A lost/raised launch response can have unknown worker outcome; the
old path could release the worker slot and reservation without proving that no
worker was created. No actual duplicate model worker was produced here.

**Reproducer:** CPU mock `Popen` to raise during attempted launch, supply
synthetic preflight/telemetry and a temporary host DB beneath CRS, and call
`supervise`. The original source's branch fabricates no-attempt evidence. The
builder repaired this during review before the reviewer's first executable
mock completed; the historical source defect remains recorded.

**Verification criterion:** Persist launch intent, set attempted state before
Popen, preserve OUTCOME_UNKNOWN/full reservation when no trustworthy process
identity returns, create no zero-cost proof, hold the lease until reconciliation.
An observed process whose identity registration fails must also stay unknown.

**Repair check executed:** On repaired `native.py`
`d01bb7a6dc7599c481345d8aedb924314eec864420830093969af900f8d7097a`, a
reviewer-owned CPU mock made Popen raise once. Supervisor returned 2 with
`launch_attempted=true`, `worker_dispatched=false`, state OUTCOME_UNKNOWN,
charged seconds 600.0 and no terminal outcome proof. No actual subprocess ran.
This closes the reproduced lost-response path on those repaired bytes.

### NATIVE-R1-002 [P2]: canonical builder identity bypassed the self-review rejection

**Evidence:** `review_reasons` rejected only the literal `runtime_audit`.
The actual canonical agent identity `/root/runtime_audit` was accepted.

**Impact:** An ordinary canonical builder label could be mistakenly treated as
an independent review. A self-written approval never expands owner authority.
This workspace does not enforce human/reviewer identity authentication.

**Reproducer:** A syntactically matching approval dictionary with
`reviewer='/root/runtime_audit'` returned an empty rejection list in the CPU
reviewer mock.

**Verification criterion:** Normalize supported canonical/short agent labels
and reject the builder's identity in every accepted representation. Require
an identified actual independent review with exact commit/native/contract
binding. Test both short and canonical builder labels and stale source bindings.

### NATIVE-R1-003 [P1]: filename heuristics admitted unclassified GPU compute processes

**Evidence:** `gpu_snapshot` marked a PID competing only when its process name
contained python/torch/llama/ollama/vllm/comfy/train. Admission did not inspect
the reported utilization for a competing workload and treated any resolved
nonmatching process name as clear.

**Impact:** Another native worker named `worker.exe` could be admitted alongside
the CRS worker while free-memory floors still pass. Resolving a PID's name
does not establish that its workload is benign. The host-wide CRS lease does
not coordinate other applications.

**Reproducer executed:** Mock nvidia-smi with the registered GPU UUID,
8192 MiB free, 90% utilization and `12345,unclassified_worker.exe`.
`gpu_snapshot` returned no unresolved/competing PIDs; with host4/disk10 GiB,
`admission_reasons` returned `[]`. This was CPU simulation, not current host
telemetry or a launched GPU job.

**Verification criterion:** Block every unclassified compute PID other than
the admitted owned worker. Any benign desktop allowlist must be frozen,
justified and bound to observed process identity; report the policy's limits.
Unknown identity, disappearing PID and conflicting compute activity remain
closed. Test nonmatching executable names, permitted owned workers, frozen
benign identities and name-spoof/identity mismatch cases.

### NATIVE-R1-004 [P1]: the pinned venv launcher created a runtime child outside the assigned Job

**Evidence:** The inspected supervisor assigns WindowsJob to the Popen launcher
before publishing create-only GO. The worker waits for GO before importing
model frameworks, which is the correct ordering. However, the registered
Windows venv executable is a launcher that can create a CPython child before
parent assignment. Assigning an already-running parent does not retroactively
assign its existing children. The coordinator independently identified this.

**Impact/scope:** The actual model may execute in a preexisting uncontained
runtime child while the supervisor observes the launcher. Parent-only kill/
exit is not sufficient containment/tree-exit evidence. Direct internal
`--worker` invocation with a manually created marker is also not an
independently contained path. Shared same-user privileges already preclude
a secrecy guarantee; this is an operational entry-point requirement.

**Reproducer executed:** The coordinator explicitly authorized a bounded,
stdlib-only process fixture using the pinned venv interpreter. It launched
`import os,time; print(os.getpid(),flush=True); time.sleep(12)` with no native
framework import. Popen launcher PID 574972 had preexisting CPython runtime
child PID 572316, executable
`C:/Users/clclc/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none/python.exe`.
After constructing the reused WindowsJob on the launcher, successful
`IsProcessInJob(handle, expected_job_handle)` queries reported launcher=true,
CPython child=false. Both owned fixture processes were cleaned up. The child
happened to exit after Job close in this fixture, which does not establish
that it was contained or would reliably exit during model work.

**Verification criterion:** Before GO, explicitly assign and verify every
preexisting runtime descendant in the exact supervisor-owned Job, with a stable
tree/worker-ready boundary, or use a qualified suspended-spawn path that prevents
descendants forming before containment. Bind actual runtime PID/creation time
and tree identities, not only launcher PID. Verify membership and kill-on-close
before framework import; direct uncontained entry stays closed. Exit proofs
must establish the whole recorded owned tree exited, and parent death/hard
limit tests must not leave a child alive. Assignment/query failure leaves GO
absent and only owned unloaded processes may be terminated. Real model/GPU
containment remains unqualified until its separate bounded runtime evidence.

### NATIVE-R1-005 [P1]: slow identity work relabeled old resource telemetry as fresh

**Evidence:** The original GPU view is collected before potentially slow process
attribution/CIM/executable hashing. `gpu_snapshot` stamps `sampled_at` afterward,
and `collect_telemetry` overwrites that timestamp again. The coordinator
independently identified this freshness issue during the current repair round.

**Impact:** An old GPU-memory/process observation can pass a five-second age
check and be used for admission or watchdog decisions while resources have
changed. Longer attribution/hashing makes this more consequential, rather than
providing a new measurement.

**Reproducer executed:** CPU mock the GPU-memory query at fake time 1000 and
advance time to 1010 during the process query. Permit the synthetic owned PID
to isolate freshness from process classification. The snapshot claimed
`sampled_at=1010`; admission at 1010 returned `[]`, despite the ten-second-old
GPU view. No actual telemetry/GPU worker was involved.

**Verification criterion:** Preserve component observation timestamps and use
the oldest required view for conservative age validation, or refresh every
resource/process view after slow identity work. `collect_telemetry` must not
erase GPU/process age. Bound/measure collector latency and fail closed when
required views age out. Cached exact executable identities need change checks
and a declared scope; caching cannot manufacture a fresh process/resource view.
Add a slow-query/CIM/hash test demonstrating rejection of the old GPU view and
truthful per-component timestamps.

## Supporting source inspection and operating limits

The reused WindowsJob creates an unnamed Job, sets KILL_ON_JOB_CLOSE, assigns
the worker handle and closes the Job on assignment failure. The reviewed
supervisor path creates GO only after assignment. Cleanup targets its owned
Job/process; it does not enumerate and terminate unrelated applications.

The worker verifies clean committed source, exact independent review binding,
source/runtime/checkpoint hashes and the canonical shared DB path. The frozen
stream loader bounds its CPU buffer to 2 MiB, validates 310 BF16 tensor payloads,
uses blocking chunk copies and verifies loaded tensor hashes. It never imports
a native framework by itself or reads protected benchmark banks. Qwen3 stock
code and exact installed sources are pinned; no model-generated code runs.

Worker operations check the model lease/fence at durable event/guard boundaries;
the supervisor monitors the same fence. Attempted, failed, returned and pending
calls are distinguished by the imported accounting implementation. The planned
dose is four 131-token forwards (524 total input positions), no generation,
teacher, gradient or optimizer update. Baseline historical hashes and numerical
failure are retained. A fixed-shape result cannot grant reader/training/capture
qualification or establish acquisition/retention gains.

The external main watchdog targets 100ms polling of process-tree RSS and host
availability, a 600s wall ceiling and lease heartbeat. GPU/disk/process sampling
runs in a separate thread with command timeouts; missing attribution or stale
samples fail closed. The GPU/disk thread originally polls at 500ms and may block
up to command timeouts, so a claim of simultaneous 100ms GPU/disk measurements
would be incorrect. Software polling/fsync/OS scheduling has latency: actual
sample intervals and elapsed failure cost must remain in receipts, not be
reported as mathematically instantaneous caps. GPU-wide memory is measured;
WDDM per-process GPU memory is not qualified. No native process resume or hard
memory quota is established by this source review alone.

## Current approval boundary

Before any dispatch, freeze repaired native/tests/contract hashes and a clean
source commit, obtain this operations review and the separate scientific review,
refresh actual complete telemetry twice, and use only the registered supervised
diagnostic command. Approval is confined to that bounded diagnostic; no reader,
procedure efficacy or learning-atlas training gate opens from it. Failures keep
all work/cost and OUTCOME_UNKNOWN until trustworthy reconciliation.

## Final repaired-byte disposition

**APPROVED_FOR_BOUNDED_DIAGNOSTIC**, confined to the registered supervised
CRS-NATIVE-D2-V1 command, after the coordinator binds a clean committed source
tip and the separate scientific review and refreshes real admission evidence.
No further consequential code/operations finding remains open in the following
reviewed bytes. This is approval of the bounded diagnostic design/entry path,
not proof that any model run has succeeded.

| Frozen artifact | SHA256 |
| --- | --- |
| `native.py` | `183a177417b81ea938e2476bb49e8e739dd0316e42479bd11f6b5818e5061782` |
| `tests/test_native.py` | `9a5ae99fd421906eeff4fe4fe316eca6d1433e1fcd13768545cc83a0f6e264f2` |
| `contracts/NATIVE_DIAGNOSTIC_V1.json` | `a57a7fa091aeda2a63e8b1452fdeed53474f627246a3f4429b62af924a45babe` |
| imported `operations.py` | `ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30` |

Executed all 23 repaired native tests using the exact pinned venv interpreter
with reviewer-enforced temporary directories beneath CRS: **23 passed, 0
failures, 0 errors**, 0.852s on this reviewer invocation. Tests include the
actual stdlib-only suspended Windows Job fixture; none imports a model framework
or executes a GPU/model operation. A separate independent repaired containment
smoke also passed: launcher PID 584300 was in the exact Job before resume;
initial suspend count was 1; the four observed launcher/runtime descendants
584300, 590620, 588688 and 590680 were exact Job members; measured drain left
zero associated PIDs. These process IDs are historical fixture observations,
not a live production admission profile.

Additional reviewer-owned pure CPU controls on final bytes observed:

| Control | Final observed result |
| --- | --- |
| Slow process/identity collection at 1010 after GPU view at 1000 | `sampled_at=1000`, identity completion 1010, admission rejected as stale |
| Nonmatching `unclassified_worker.exe` PID12345 | Listed in unclassified GPU PIDs; admission rejects |
| Canonical `/root/runtime_audit` self-review | Rejected as builder approval |
| Lost Popen creation response | Test preserves OUTCOME_UNKNOWN/full600s, no outcome proof |
| Failure before any attempted Popen | Separate test permits identified no-dispatch zero cost |
| Signed-zero semantic numeric equality | Raw identity test rejects a false bitwise result |
| Changed positive GPU registry dependency | Registry test rejects executable/engine drift |
| Direct worker GO marker without supervised ancestry | Rejected before framework import |

NATIVE-R1-001–005 are closed for this design and demonstrated interfaces.
Suspended creation prevents the concrete preexisting runtime-child escape;
every owned descendant is checked against the exact Job before GO. The worker
also checks ancestry, actual current Job membership and kill-on-close before
framework import. Finalization drains and queries the owned Job before issuing
its exit proof; missing tree exit remains unknown and conservatively charged.
Pinned Git executables accommodate the worker's read-only commit guards;
vanished PIDs may be ignored only after a refreshed exact Job list confirms
absence. These observations do not qualify model/GPU resume or other native
execution shapes.

The positive desktop registry is an explicit narrow resource policy. It admits
exact installed graphical/system binaries by absolute path/hash. FlipHero's
exception additionally pins UnityPlayer.dll and records the unsigned/blank PE
metadata, matching Unity engine version, build receipt and actual observed GPU
C+G type. I approved that bounded observed-Unity graphical policy classification;
it does not prove absence of every inference workload or authorize controlling
or terminating that game. Unregistered/changed/unresolved identities still
block; memory/fencing/telemetry floors remain active.

Component GPU/process timestamps now precede identity hashing/CIM and remain
the conservative snapshot age through host/disk collection. Slow collectors
therefore deny rather than relabel an old view fresh. Main RAM/fencing watchdog
polling is nominal100ms; the separate GPU/disk collector and operating-system
scheduling remain sampled with explicit age checks. Report actual intervals,
peaks, failures and wall time, not an instantaneous hardware-quota guarantee.

The coordinator must now commit these unchanged source bytes, record that exact
clean HEAD in the launch review JSON together with the hashes above and this
reviewer's identity, and recheck both scientific approval and source/runtime/
checkpoint/host evidence. Changing native code, contract or imported code
requires a newly bound review. The review JSON is a permission/qualification
record, never an expansion of the owner's authorization. It approves no reader,
generation, training, benchmark bank or efficacy claim.

Machine-readable equivalent before commit binding:

```json
{
  "review_id": "CRS-NATIVE-OPS-REVIEW-001",
  "reviewer": "/root/data_audit",
  "status": "APPROVED_FOR_BOUNDED_DIAGNOSTIC",
  "native_source_sha256": "183a177417b81ea938e2476bb49e8e739dd0316e42479bd11f6b5818e5061782",
  "tests_sha256": "9a5ae99fd421906eeff4fe4fe316eca6d1433e1fcd13768545cc83a0f6e264f2",
  "contract_sha256": "a57a7fa091aeda2a63e8b1452fdeed53474f627246a3f4429b62af924a45babe",
  "operations_sha256": "ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30",
  "source_commit": null,
  "commit_binding_required_before_dispatch": true,
  "separate_scientific_review_required": true,
  "closed_findings": ["NATIVE-R1-001", "NATIVE-R1-002", "NATIVE-R1-003", "NATIVE-R1-004", "NATIVE-R1-005"],
  "native_tests_run": 23,
  "native_test_failures": 0,
  "native_test_errors": 0,
  "actual_windows_stdlib_containment_only": "PASS",
  "actual_model_workers_by_reviewer": 0,
  "framework_imports_by_reviewer": 0,
  "gpu_calls_by_reviewer": 0,
  "reader_qualified": false,
  "training_qualified": false,
  "model_quality_evidence": false,
  "current_host_admission_established_by_reviewer": false
}
```
