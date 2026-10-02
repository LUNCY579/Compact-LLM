# Prospective supervisor-death CPU controls

Registration commit: `5c588925c3da56e44423f4562add9d603de0824d`.
Verified source base: `863476aaf1e394a0796f4c59dcf96f0cc3793d4a`.
Corrective branch: `codex/mimo-supervisor-death-review-fix-20261002`.
Worktree: `C:\Users\clclc\Documents\Codex\2026-10-01\task\ChatBubble-mimo-supervisor-death-review-fix`.
Verified implementation parent: `bfd0e13723bcd63ab151fc96fe9dd797b1aff057`.
The original successor checkout, its dirty watchdog-order edit, fixture receipts
and proposal are preserved untouched. This checkout isolates the review fixes.
Config SHA-256:
`1d4903116e1c23a6f41abe6c6a43b72adca06a885f85745be94a2786a3e5ae18`.
The exact clean final HEAD and runtime source hashes must be bound by the later
parent-reviewed authorization; registration does not authorize execution.

G05a remains an inference from source and the documented
[last-Job-handle rule](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects#managing-job-objects).
There is no observed supervisor-death escape. The earlier CPU/native failures and
their causes are not reclassified. All 48 prior artifacts and consumed experiments
are preserved, and the typed HANDLE.value fix and consumed-V1 refusal are inherited.
No old execution source is modified by this successor.

The new harness is `scripts/qualify_supervisor_death_cpu.py`, with regression
fixtures in `tests/test_supervisor_death_cpu.py`. The observer creates a separate
outer safety Job around each direct pinned supervisor and retainer pair. It never
opens or duplicates the inner Job. The retainer deliberately retains one
noninheritable inner Job duplicate and then sleeps without further handle actions.

The two cases test supervisor death before and after remote duplicate closure.
Case A requires supervisor exit37 through its original held handle, retainer
survival through its held handle for at least one second, then successful remote
duplicate closure and retainer exit. Case B requires successful remote closure
first, both held handles nonsignaled for at least one second, then supervisor
termination and held-handle exits. Remote closure uses the documented
[DuplicateHandle NULL-target/CLOSE_SOURCE operation](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle).
It creates no observer inner-Job reference, is attempted once, and a failed API
cannot be rescued by later child exit. The child never touches the stale wrapper.

A successful retained-case control would demonstrate delayed inner cleanup until
the duplicate closes in that constructed state. It would not qualify automatic
cleanup on supervisor death while a duplicate remains open. No production
handle-lifetime repair or native containment pass is claimed by these fixtures.
The source-controlled retainer state is deliberately constructed; a later native
launcher must separately establish its handle lifetime and containment behavior.
Held process objects signal on termination; PID disappearance alone is insufficient.
[Microsoft process-termination semantics](https://learn.microsoft.com/en-us/windows/win32/procthread/terminating-a-process)
also explain why terminating a supervisor alone does not normally end its children.

Strict proposal: 30 seconds including admission, 45 seconds including cleanup,
five seconds per metadata/exit phase, 100 ms polling, 0.5 GiB full-tree RSS,
at most four simultaneous/nine lifetime experimental processes. Planned shape:
two sequential direct-interpreter pairs, four processes total, two active at once.
Keep 1.25 GiB RAM reserve, two preworker samples >=3.3125 GiB separated by two
seconds, >=4 GiB disk and <=1 MiB files. No expanded budget is proposed.
Autonomous role deadlines are >=50 seconds after arming, outside the outer bound.

The same independent deadline thread tracks the original suspended Popen root as
soon as Popen returns, before diagnostic/identity/Job-setup logging. Before outer
enrollment it may terminate only that original held root handle. After enrollment
it may terminate only the outer safety Job. Successful actual outer Job close
detaches the watchdog; accounting/logging before that close cannot detach it.
Termination precedes the separate diagnostic record. These changes address the
three P1 findings from independent review of implementation `bfd0e137` without
adding experimental cases, processes or budgets. Any
watchdog action, outer close while a process remains active, manual fallback,
logging/API failure, incomplete accounting or missing outcome makes qualification
fail. Valid logged creation/native-accounting observations remain lower bounds
when a final receipt is failed, invalid or missing; aggregate counts remain null.
Startup safety API records also preserve a confirmed nonzero creation lower bound
if the original creation/final receipt is missing. Aggregate outcomes remain null.
The outer Job lifetime count is queried before its final close. All experimental
exit witnesses precede safety cleanup and require held handles, not image queries
on exiting processes.

Before physical launch, parent review must cover the clean final source, fixture
receipt and this budget. Fresh matching-worker, lease, RAM/disk checks are required;
the October1 read-only inspection found no matching containment worker or either
GPU lease and about10.17 GiB available RAM, but cannot stand in for future checks.
An exact-source authorization file must be explicitly supplied by the parent at
`artifacts/mimo_central/authorizations/supervisor_death_cpu_v1_20261001.json`.
No authorization file exists in this delivery. The no-argument entry refuses
before Kernel/process/output creation without it. Any later attempt consumes
`artifacts/mimo_central/supervisor_death_cpu_v1_20261001`; a tracked
`receipts/SUPERVISOR_DEATH_CPU_V1_OUTCOME.json` also refuses reused registrations
when ignored outputs are absent in another checkout.

Proposed command, only after approval and fresh guards: the pinned venv Python
with `-B -m experiments.post_training_mimo_centralized_v1.scripts.qualify_supervisor_death_cpu`.
The authorization binds experiment ID, config SHA, exact final HEAD, runtime source
hashes, the complete frozen limits, independent fixture review and explicit parent
permission. Native/capture/capability/E1 authorization fields must all remain false.
Expected raw evidence: ADMISSION/API logs, CASE_STARTED, supervisor/retainer
READY/GO/ARMED and launch proofs, per-case CLEANUP/CASE_RESULT, suite
QUALIFICATION_RESULT, or a blocking OUTER_EXCEPTION. Copy and byte-manifest all
raw artifacts after that future attempt; never overwrite receipts or retry V1.

No physical controls, native capture, GPU lease/model operation, paid compute,
evaluation mutation, checkpoint overwrite, push, merge or tripleskill change is
included. The earlier three-case CPU successor is still unqualified, and the new
two-case control cannot replace those controls. Native/capability/E1 gates and the
failed bulk-shape likelihood tolerances remain unchanged.
