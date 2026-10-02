# CRS-001 independent operations review 1

Client date: 2026-10-01. Reviewer: `/root/data_audit`, rotating into the
code/operations adversary role. I did not build the operations module or its
tests. I previously built the data/scorer instruments, so this is not a fresh
review of those instruments. Agents share workspace privileges and inherited
model context; this review provides executable counterexamples, not independent
human truth or OS isolation.

Decision: **REPAIR_REQUIRED** for cooperative admission/recovery mechanics.
Native launch remains CLOSED. The proposed user-local shared-state integration
is suitable in scope for a single canonical coordination database only after
path alignment and the consequential reconciliation fixes receive fresh review.
No canonical host database was created in this review.

## Frozen subjects inspected first

| Subject | SHA256 |
| --- | --- |
| `operations.py` | `8a82ccfd5caea367b2d6c39287140eccdef01446f13f6324aff76d71dd39e855` |
| `tests/test_operations.py` | `118c8fed01da08415b75e92e334ebcc60f25a5e42728f835c3610799b14c56e6` |
| `contracts/OPERATIONS_V1.json` | `6f0245da1e742b209de0ba8344911773a9de813887b66d1c82516cadd21d583f` |
| `AUTHORIZATION_2026-10-01.md` | `0ff6262c729a1dea4dd910ef72d18e967c93eea79e13e9dbc156345b5042857a` |

Raw source/contracts/tests were inspected before builder interpretation.
The authorization hash above records the earlier `.codex/crs001` draft. The
operations implementation/contract selects `%LOCALAPPDATA%/Rineya/CRS-001/`.
The coordinator reported that it is aligning this draft; a changed amendment
must have its own hash and must not be represented by the hash above.

## Executed independent checks

Two reviewer-authored stdlib Python reproducers imported only the operations
module. They used temporary SQLite/receipt directories created beneath the CRS
package, verified their resolved paths stayed inside that package and closed
their connections before cleanup. No actual model, GPU operation, live PID,
worker termination, protected bank or outside host state was used. Worker PID
42 and process checks were explicitly synthetic witnesses.

Observed outcomes:

| Check | Raw observed result |
| --- | --- |
| Close a RUNNING job with arbitrary evidence and zero cost, then reserve another | Old state FAILED, old actual cost 0.0, new state RESERVED; old 600s reservation released despite its observed worker identity |
| Retry identical CPU completion after a lost response | Raises `CPU completion cannot erase native reservations or outcomes` |
| Reconcile expired lease with required evidence keys containing null | Accepted; new lease granted with fence 3 |
| Change job payload/state directly while retaining stored request hash, then ready dependent | SQLite integrity check `ok`; dependent READY |
| Simultaneous independent SQLite connections racing coordinator acquisition | Exactly one lease winner, the other rejected with `another live lease exists; use explicit heartbeat` |
| Verify canonical `{}` in receipt directory with a matching hash | Returns `{}` despite missing receipt schema/kind/key/payload |
| Inject `os.link` disk/publication failure | Zero completed result receipts, one retained `.partial-*`; same-request retry verifies payload `{"failure_cost":7}` |

The supplied 20 tests were read; the contract reports its earlier failed
constructor cleanup check and repaired passing run. This review did not erase
that history or reinterpret it as native qualification.

## OPS-R1-001 [P1]: unfenced reconciliation can clear a live worker and its cost

**Evidence:** `operations.py:297–323` accepts any nonempty dictionary as evidence
and has no owner/fence argument or lease check. It allows every ACTIVE state,
including RUNNING, to become terminal. `reserve` at lines 246–267 then sees no
active job and charges the substituted actual cost instead of the reservation.

**Impact:** A stale coordinator, a wrongly routed recovery callback or an
incorrect caller can close a live admitted job and admit a second while the
first continues. Its attempt cost can simultaneously become zero. This is a
cooperative protocol defect as well as a native risk; OS containment alone does
not fix the journal transition. The native launcher is absent/closed, so this
review did not actually duplicate a worker.

**Reproducer:** Register/ready `a`, acquire model lease `A`, reserve 600 seconds,
mark RUNNING with a PID/creation-time/command-hash witness, then call
`reconcile('a','FAILED',0,{'note':'still running; no exit evidence'})`.
Register/ready `b` and `reserve('b','A',fence,600)`. Both calls succeed. Job `a`
still retains its worker witness but now contributes 0 seconds to the ledger.

**Verification criterion:** Define an explicit fenced reconciliation authority
and bind recovery evidence to the particular job/request, model fence and
observed worker identity. A RUNNING/OUTCOME_UNKNOWN job must not release its
worker slot until an independently admitted current process/GPU reconciliation
establishes termination/no worker. Zero measured cost requires a specific
no-dispatch proof and must reject a previously observed worker unless an
explicitly justified accounting contract exists. Reject obsolete reconciliation
tokens and missing/null evidence. Test live-worker evidence rejection, stale
coordinator closure, lost launch responses and legitimate terminated/no-dispatch
recovery. Preserve the full reservation/unknown state while evidence is absent.

## OPS-R1-002 [P2]: reconciliation evidence has no validated content or binding

**Evidence:** `reconcile_lease` lines 216–231 checks only three key names and
`processes_clear is True`. Empty/null values qualify; timestamps, host identity,
process inventory, model-fence/job identity, worker exit and receipt hashes are
not bound. `reconcile` and `finish_cpu` accept arbitrary nonempty dictionaries.

**Impact:** Placeholder or stale evidence can be mistaken for current completion
or host clearance. The contract's phrase “complete current host/process
reconciliation evidence” is stronger than its implemented validation.
This does not imply that a SQLite module can independently discover real GPU
activity; an external qualified collector remains necessary.

**Reproducer:** After expiring a model lease with no active job, call
`reconcile_lease(owner,fence,{'host_process_check':None,
'gpu_activity_check':None,'worker_status':None},processes_clear=True)`, then
acquire the next lease. It succeeds.

**Verification criterion:** Freeze a typed, nonempty evidence envelope with
scope, current capture time, host/worker identity, explicit status and immutable
receipt binding; distinguish synthetic CPU witnesses from production witnesses.
Validate before any state mutation and recheck the receipt hashes. Old/empty/null
evidence must fail; a future native collector must qualify truth and freshness.
If the initial module stays generic, narrow the contract to caller assertions
and keep all admission requiring those assertions closed until a validated
collector/registrar exists.

## OPS-R1-003 [P2]: identical CPU completion is not idempotent on restart

**Evidence:** `finish_cpu` lines 325–339 permits only READY/PENDING/BLOCKED and
rejects every terminal state. Unlike `reconcile`, it does not accept an identical
stored terminal outcome.

**Impact:** A lost successful return followed by retry fails despite a durable
identical CPU result. This obstructs automatic DAG continuation and encourages
new job keys or special recovery paths. It does preserve the original outcome;
no evidence was overwritten in the reproducer.

**Reproducer:** Register/ready CPU job; call
`finish_cpu('cpu','SUCCEEDED',{'receipt':'unchanged'})` twice. The second raises.

**Verification criterion:** Same terminal key/outcome/evidence should return the
stored result without changing finished time, events or cost. Different outcome
or evidence must still reject. Add lost-response/reopen tests for SUCCEEDED,
FAILED and ABORTED CPU results, not just registration/native reconciliation.

## OPS-R1-004 [P2]: canonical bytes alone do not establish receipt validity

**Evidence:** `verify_receipt` lines 368–381 verifies directory/suffix/hash and
canonical JSON bytes, but not envelope schema, supported version, kind/key,
expected filename or expected job/payload binding.

**Impact:** A caller can mistake an unrelated or malformed canonical JSON file
for a completed immutable receipt. Path/hash validity is useful but does not
establish the receipt's scientific/operational meaning.

**Reproducer:** Write `{}` as `receipts/unbound.json`; pass that path and its
actual SHA256 to `verify_receipt`. It returns `{}` rather than rejecting.

**Verification criterion:** Validate exact envelope fields and types/version,
derive filename from the declared kind/key and bind the expected kind/key/job
where consumed. Reject foreign canonical JSON, renamed receipts, extra fields,
wrong type keys and mismatched payload/job identities. Receipt validation must
not itself confer launch permission or evidence truth.

## OPS-R1-005 [P2]: SQLite structural integrity misses logical job corruption

**Evidence:** Constructor runs `PRAGMA integrity_check`; `_job/get_job` trust
stored `request_hash`, payload, dependency graph, state and result. `ready`
requires only the dependency's state string `SUCCEEDED`.

**Impact:** Logical corruption from an interrupted migration, unintended SQL
write or incorrect callback can admit a dependent even with a broken immutable
request/result. This review models accidental/caller corruption; a malicious
same-user process already has shared privileges and is outside any claimed OS
secrecy boundary.

**Reproducer:** Register `corrupt` with `{'bound':True}`. Directly change its
payload to `{}`, state to SUCCEEDED and result to `{}` while leaving its request
hash intact. SQLite integrity check remains `ok`; a newly registered dependent
becomes READY. Identical original registration also trusts the unchanged stored
request hash without re-deriving it from those rows.

**Verification criterion:** Validate immutable row/request/dependency hashes and
legal state/result envelopes on reads and before READY/reservation; schema-
version the database. Bind terminal result evidence to valid immutable receipts.
Reject/quarantine corrupt records while preserving bytes and budget exposure.
Tests should mutate logical payload/state/result/dependency fields and show
fail-closed behavior. Don't claim `integrity_check` detects semantic corruption.

## Additional bounded limitations

Receipt publication correctly uses a create-only hard link after file fsync and
preserves partials in the injected failure. Directory/metadata durability under
power loss is not demonstrated; the present check covers publication failure
and process-level idempotency, not full disk/power crash durability. Clarify that
scope or add an appropriate platform-qualified persistence contract.

Independent SQLite connections excluded a simultaneous coordinator acquisition.
This is supporting evidence for cooperative locking. It does not qualify a
real multi-process GPU runner, creation-time/PID witness collector, watchdog,
fencing enforcement inside a worker, memory/disk quota, terminate-tree behavior,
native checkpoint resume or numerical equivalence. No launcher exists in this
module; no inference/training/cost efficacy can be derived from these tests.

Budgets preserve reservation cost for active unknowns, maintain epoch totals and
aggregate rolling totals across epochs in the inspected code. Actual cost is a
caller measurement, so accepting arbitrary reconciliation evidence is the
consequential cost-release hole. A new epoch identifier alone does not reset the
rolling budget in the normal code path.

## Shared-state integration disposition

The owner requested durable host-wide coordination and allowed narrowly reviewed
integration outside the experiment directory. One database at the resolved
`%LOCALAPPDATA%/Rineya/CRS-001/host_state.sqlite3` is proportionate; it must hold
coordination only and be shared by every CRS worktree. Keep a pinned canonical
path/profile binding and reject accidentally distinct production stores. The
constructor's arbitrary paths remain appropriate for explicitly temporary tests.

The reviewed authorization draft names a different `.codex` path. Align it with
the implemented canonical path and record the changed hash before creation.
This reviewer does not expand owner authority, approve unrelated state writes,
or permit native launch. Because OPS-R1-001–005 remain open in these frozen bytes,
cooperative admission/recovery approval is withheld and canonical production
state creation should wait for the registered repair/review round.

## Machine-readable review equivalent

```json
{
  "review_id": "CRS-OPS-REVIEW-001",
  "reviewer": "/root/data_audit",
  "date": "2026-10-01",
  "decision": "REPAIR_REQUIRED",
  "round": 1,
  "operations_sha256": "8a82ccfd5caea367b2d6c39287140eccdef01446f13f6324aff76d71dd39e855",
  "tests_sha256": "118c8fed01da08415b75e92e334ebcc60f25a5e42728f835c3610799b14c56e6",
  "contract_sha256": "6f0245da1e742b209de0ba8344911773a9de813887b66d1c82516cadd21d583f",
  "authorization_draft_sha256": "0ff6262c729a1dea4dd910ef72d18e967c93eea79e13e9dbc156345b5042857a",
  "findings": [
    {"id": "OPS-R1-001", "priority": 1, "status": "OPEN", "reproduced": true},
    {"id": "OPS-R1-002", "priority": 2, "status": "OPEN", "reproduced": true},
    {"id": "OPS-R1-003", "priority": 2, "status": "OPEN", "reproduced": true},
    {"id": "OPS-R1-004", "priority": 2, "status": "OPEN", "reproduced": true},
    {"id": "OPS-R1-005", "priority": 2, "status": "OPEN", "reproduced": true}
  ],
  "cooperative_cpu_admission_approved": false,
  "shared_state_integration_scope_suitable": true,
  "canonical_state_creation_approved_this_round": false,
  "native_launch_approved": false,
  "real_process_containment_qualified": false,
  "actual_model_workers": 0,
  "native_worker_seconds": 0,
  "outside_host_state_created": false,
  "limitations": ["shared workspace privileges", "shared model ancestry", "synthetic PID/GPU witnesses", "no power-loss qualification"]
}
```
