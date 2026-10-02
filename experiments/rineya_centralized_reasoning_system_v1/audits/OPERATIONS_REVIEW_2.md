# CRS-001 repaired operations review 2

Client date: 2026-10-01. Reviewer: `/root/data_audit`, code/operations adversary;
not the operations builder. This continues review 1 against repaired bytes.
Shared workspace/model ancestry remains disclosed. No actual model worker,
GPU execution, process termination or canonical outside state was created.

**Final decision: APPROVED_COOPERATIVE_CPU_MECHANICS_AND_NARROW_SHARED_STATE.**
This approves the cooperative durable coordination contract and the aligned
outside-package coordination database integration. It does not approve any
native launch, real process containment, numerical result, reader, training,
resource fit, resume equivalence or scientific promotion.

## Final bindings

| Artifact | SHA256 |
| --- | --- |
| `operations.py` | `ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30` |
| `tests/test_operations.py` | `8b1b0166d82c874a6076742a1eea13025e886b513e42dfac791fdf503af1da48` |
| `contracts/OPERATIONS_V1.json` | `9ab085d65791996b3fa2fe2396b95be6fbbccf55303c16e3b70b37144d9df66a` |
| aligned `AUTHORIZATION_2026-10-01.md` | `5a7c02737cad5f08150805278323c42be2c0be3af2ae89620907faf9be319a84` |

The initial repaired `operations.py` hash
`1f97883027cd048eb9313550d95a354290920618a3e731e2aaabdf4c7afb0159`
closed four original reproductions but still allowed accounting-row corruption.
That rejection/repair history is retained below; it was not approved.

## Executed verification and findings disposition

Reviewer first inspected repaired source/contracts and ran original adversarial
controls in temporary SQLite state inside CRS. The first repaired version
correctly rejected arbitrary/unfenced live reconciliation, null host evidence,
unbound `{}` receipts and immutable request/payload corruption; identical CPU
completion returned the unchanged result and event count.

It still accepted this counterexample: legitimate FAILED outcome with a bound
exit proof and 23 seconds, followed by direct accidental/caller SQL change
`actual_seconds=0`, produced a budget total of 0 while result/event/proof retained
23. This was reported to the builder before acceptance. Final repaired bytes
bind admission cost/time/owner/fence, observed worker, creation time and terminal
actual cost/finish time back to their durable events. The same independent
counterexample now raises `actual cost/finish time differs from durable outcome`.

Executed the 27 final operations tests with reviewer-enforced temporary
directories inside the CRS package: **27 passed, 0 failures, 0 errors** (2.342s
on this reviewer invocation). No tests used canonical host state. The builder's
shorter timing is a separate invocation, not substituted for this one.

Also executed a fresh independent reviewer script on the final bytes:

| Original finding / control | Final observation | Disposition |
| --- | --- | --- |
| OPS-R1-001 arbitrary RUNNING→FAILED/0 | Rejected without bound observation receipt and authority | Closed within cooperative caller/evidence contract |
| OPS-R1-002 null host/process evidence | Rejected | Closed for typed non-null interface; real collector truth remains unqualified |
| OPS-R1-003 identical CPU terminal retry | Identical result and unchanged event count | Closed |
| OPS-R1-004 canonical `{}` receipt | Rejected | Closed for schema/hash/kind/key/filename validation |
| OPS-R1-005 changed immutable payload | Rejected | Closed for independently changed logical rows |
| OPS-R1-005 changed actual cost after valid FAILED23 | Rejected; valid preceding cost was 23.0 | Closed after additional accounting repair |

Final source additionally requires exact model owner/fence and a create-only
`native_outcome_observation` receipt bound to job key/request/lease/observed
worker. A previously observed worker cannot use zero/no-dispatch evidence.
Stale job resolution requires an expired marked model lease plus a current
coordinator lease and its matching proof witness. Zero/no-dispatch is confined
to RESERVED with no observed worker and explicit no launch attempt. Native
supervisors must tell the truth about launch attempts; the separate native
review examines that obligation.

SQLite `BEGIN IMMEDIATE`/WAL/FULL and the independent connection race from
review 1 support cooperative exclusion; hard-link create-only receipt
publication retained partial files and permitted immutable same-request retry
under the injected failure. Power-loss/directory metadata durability and real
cross-process model recovery remain outside the demonstrated scope.

## Narrow integration acceptance

The aligned amendment records exactly
`%LOCALAPPDATA%/Rineya/CRS-001/host_state.sqlite3`. The implementation resolves
that canonical path without creating it on import. A single shared local
coordination database across CRS worktrees is proportionate to the owner's
durable lease/budget request and narrow integration allowance. This review
approves creating/using that coordination state when the coordinator records
these artifact hashes and keeps weights/data/protected labels out of it.
It does not expand owner authority or control unrelated applications.

Arbitrary temporary paths remain permitted for CPU fixtures. Production native
callers must enforce the canonical host path and separately qualify actual
process/GPU evidence, watchdog containment, source/checkpoint identities and
maximum shapes. A same-user process can coherently rewrite database plus
receipts/events; this module does not assert an OS security boundary. Human/
Codex assertions and synthetic witnesses are not physical worker observations.

## Machine-readable review equivalent

```json
{
  "review_id": "CRS-OPS-REVIEW-002",
  "reviewer": "/root/data_audit",
  "date": "2026-10-01",
  "decision": "APPROVED_COOPERATIVE_CPU_MECHANICS_AND_NARROW_SHARED_STATE",
  "operations_sha256": "ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30",
  "tests_sha256": "8b1b0166d82c874a6076742a1eea13025e886b513e42dfac791fdf503af1da48",
  "contract_sha256": "9ab085d65791996b3fa2fe2396b95be6fbbccf55303c16e3b70b37144d9df66a",
  "authorization_sha256": "5a7c02737cad5f08150805278323c42be2c0be3af2ae89620907faf9be319a84",
  "closed_findings": ["OPS-R1-001", "OPS-R1-002", "OPS-R1-003", "OPS-R1-004", "OPS-R1-005"],
  "tests_run": 27,
  "tests_failed": 0,
  "tests_errors": 0,
  "cooperative_cpu_admission_approved": true,
  "canonical_shared_coordination_integration_approved": true,
  "canonical_path": "%LOCALAPPDATA%/Rineya/CRS-001/host_state.sqlite3",
  "native_launch_approved": false,
  "real_process_containment_qualified": false,
  "power_loss_durability_qualified": false,
  "actual_model_workers": 0,
  "native_worker_seconds": 0,
  "outside_host_state_created_by_reviewer": false
}
```
