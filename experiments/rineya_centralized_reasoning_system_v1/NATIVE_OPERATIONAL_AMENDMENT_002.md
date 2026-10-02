# Native operational amendment 002: explicit owned Job identity

Recorded 2026-10-02, America/Chicago. Builder: `/root/runtime_audit`.
Base: clean `9fd345d9bbcd0cdc6b300dcb07f6b68dbe8914ec` in the isolated
`crs-native-desktop-v2` managed worktree. Independent review and a new clean
committed source/review binding are required before root may dispatch this
successor. This amendment records an operational repair within the owner's
existing authorization; it expands no execution authority.

The original contract is preserved byte-for-byte, create-only, at
`contracts/history/NATIVE_DIAGNOSTIC_V1_a57a7fa0.json`, SHA256
`a57a7fa091aeda2a63e8b1452fdeed53474f627246a3f4429b62af924a45babe`.
The primary checkout, model assets and original failed receipts are untouched.
The relocated-game admission amendment was cancelled; no GPU registry change
is included here.

The original contained attempt in
`results/d2_9fd345d9_epoch001_statechange2` exited before framework import.
Its complete tree-membership and empty-tree exit evidence remain intact.
**1.1090000000003783 native qualification worker seconds were charged**; there
were zero model loads/forwards/updates. The cost is not reset by this revision.
No unchanged retry is licensed by the failure.

An actual bounded stdlib-only child probe preserved in
`outputs/job_identity_probe_v1/NULL_VS_EXACT_OWN_JOB.json` observed
`QueryInformationJobObject(NULL)` flags `0x3000`, while a query-only duplicate
of the exact supervisor-owned Job returned `0x2000`. Membership tests succeeded
for both. These are observed different Job limits; we do not assert a complete
hierarchy or label NULL's Job as necessarily an outer Job. Microsoft documents
that NULL uses the process's associated/immediate nested Job.
[QueryInformationJobObject](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject),
[nested jobs](https://learn.microsoft.com/en-us/windows/win32/procthread/nested-jobs).

The supervisor now duplicates only `JOB_OBJECT_QUERY` access, noninheritable,
into the uniquely observed pinned CPython worker. GO binds the target PID and
creation time alongside the existing owner/fence/supervisor ancestry. The worker
checks membership and the unchanged kill-on-close/no-breakaway criteria against
this explicit owned Job handle. NULL flags remain recorded diagnostic data.
The temporary handle closes on successful verification and rejection **before
any framework import**, leaving the supervisor as the owned Job's live handle
holder during model work. No parent or immediate Job flags are relaxed or changed.
[DuplicateHandle](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle).

Actual CPU controls in `outputs/job_identity_probe_v2/` cover successful explicit
identity, mismatched PID and mismatched creation time. Each verifies the query
handle is unusable afterward. The successful child subsequently creates a
grandchild; all four launcher/CPython processes are verified members of the exact
owned Job. Closing the parent's last owned handle terminates the complete observed
tree without an explicit kill. Rejected children also close their query handles
and terminate under last-handle close. These controls import no framework and
make no GPU/model calls. The full native CPU suite has **27 passing tests**.

The change preserves the exact Titan/config/tokenizer/environment pins, BF16
dtype, eager/cache-off math, fixed 131-token trace, four-forward/524-position dose,
bitwise causal requirements and original failed bulk capture tolerances. It also
preserves 600 seconds, 2 GiB process-tree RSS, 1.25 GiB system reserve, 4/0.5 GiB
GPU floors, 4 GiB disk, 100 ms watchdog, one worker, fencing and cumulative
2-hour epoch/4-hour rolling native budgets. The runtime's model forward body and
scoring decisions are unchanged. Reader576, capture, training and efficacy
remain unqualified. A CPU containment pass cannot establish their behavior.
