# CRS native operational review 2: exact owned Job identity

Recorded 2026-10-02, America/Chicago. Reviewer `/root/data_audit`; builder
`/root/runtime_audit`. Reviewed only the isolated managed worktree
`C:/Users/clclc/.codex/worktrees/crs-native-desktop-v2/ChatBubble-rineya-centralized-reasoning-system-v1`,
based on `9fd345d9bbcd0cdc6b300dcb07f6b68dbe8914ec`. No primary checkout,
model asset, original scorer/result, GPU registry or user application was edited
or terminated by this reviewer. Reviewers share workspace/model ancestry;
executable kernel observations, rather than agreement, support the finding.

**Decision: APPROVED_FOR_BOUNDED_DIAGNOSTIC**, confined to the unchanged
CRS-NATIVE-D2-V1 supervised four-forward diagnostic after a clean successor
commit/review binding and fresh host admission. This approval covers operational
revision2; it is not a passing model result, reader/capture/training qualification
or authority expansion. The previous failed attempt and its cost remain intact.

## Frozen subjects

| Artifact | SHA256 |
| --- | --- |
| repaired `native.py` | `b94b4dc2ce9e6c041efecaff84a885e6b4fa6e6986d85832d6aa019128d7a852` |
| `tests/test_native.py` | `800eb02ffa7ea4889496ef9d478199ab205fe08f9709d6e84ec8caba33087d5b` |
| successor diagnostic contract | `291b7803e745ade2ccfcc5a6e9a8eafc31a9c174541ccd5436f730bde57a8707` |
| `NATIVE_OPERATIONAL_AMENDMENT_002.md` | `112f71e35090d75e4e8906bbe514306ba66ffd33e973a245ad13f8006f2dca94` |
| unchanged `operations.py` | `ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30` |
| create-only `contracts/history/NATIVE_DIAGNOSTIC_V1_a57a7fa0.json` | `a57a7fa091aeda2a63e8b1452fdeed53474f627246a3f4429b62af924a45babe` |

Raw original stderr/GO/TREE/watchdog/supervisor/exit-proof artifacts and the
actual NULL-versus-owned Job probe were inspected before the repaired source.
The repaired source, tests, amendment and full contract were then inspected.
No original failure was rerun or overwritten.

## Retained operational failure and causal finding

Original result directory:
`results/d2_9fd345d9_epoch001_statechange2`.
The worker exited1 at the pre-framework `current_windows_job_witness` check:
`NativeBlocked: current worker Job lacks strict kill-on-close/no-breakaway`.
The exact owned Job held launcher604024 and runtime606264 before GO and in four
watchdog samples. The supervisor subsequently verified an empty owned process
tree and recorded FAILED. Measured attempted qualification cost was
**1.1090000000003783 native worker seconds**, with no model/forward/update
evidence. This is an operational qualification failure, not a negative model
quality or reasoning result. Its native-outcome observation receipt SHA256 is
`208e838b8313c8ccb74dd2058c67dc80177e2ba3fd4843d6f7eec20ea673aa65`.

The earlier CPU smoke established supervisor-side exact tree membership but
did not exercise the real worker's NULL Job limit query. The actual launch
exposed that missing qualification boundary. Probe
`outputs/job_identity_probe_v1/NULL_VS_EXACT_OWN_JOB.json` (SHA256
`a2627233b14144547954f818d08aa50f7a2fd58b05eee5da9d08847599af9cb7`)
observed membership in both queried Jobs, but NULL flags0x3000 versus exact
supervisor-owned flags0x2000. It imported no framework and verified empty-tree
termination. NULL is not an identity for the intended supervisor-owned object;
Microsoft documents associated/immediate nested Job behavior. Do not infer a
complete hierarchy or necessarily call the NULL Job an outer Job.
[QueryInformationJobObject](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject).

**Impact:** Rejecting the wrong Job's flags prevents a valid contained diagnostic
from starting. Relaxing flags to admit0x3000 would instead weaken the intended
no-breakaway check. Neither is an acceptable resolution.

**Verification criterion:** Transfer and query the exact owned Job object,
require strict kill-on-close/no-breakaway and exact membership, bind the intended
PID/creation identity and owner/fence/ancestry, and close the temporary child
query handle before frameworks on success and every rejection. A retained
child handle would delay last-handle kill-on-close; Microsoft documents that
termination occurs when the final Job handle closes.
[Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

## Repaired source verification

The supervisor creates the launcher suspended, assigns its owned WindowsJob,
verifies exact membership and absence of children, resumes the single primary
thread and observes the uniquely pinned CPython descendant. It duplicates only
JOB_OBJECT_QUERY access(mask4), noninheritable, into that observed target after
PID/creation/executable/owned-membership checks. The target PID and creation time
are recorded in TREE/GO with the existing contract/owner/fence/supervisor/tree
identities. The duplicate's numeric value is valid in the receiving process;
the source API implements this access/inheritance choice directly.
[DuplicateHandle](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle).

The worker checks GO owner/fence/contract, recorded supervised ancestry and
command, then verifies target PID/creation identity, noninheritance, membership
and strict flags against the explicit transferred handle. NULL limits are
retained only as diagnostic observations. The temporary handle is closed in
the witness's finally block; earlier GO/ancestry rejection closes an unconsumed
valid transfer. No framework import or model operation occurs before that
boundary. Existing event/operation guards still require the live model fence,
and missing dispatch/tree-exit evidence still preserves OUTCOME_UNKNOWN and
the full reservation. Worker exit proof remains bound to the admitted request,
lease and owned tree; the prior1.109s is not reset.

The strict owned criteria remain flags0x2000 present, flags0x800/0x1000 absent
and actual membership true. No parent/immediate Job flags were changed or
ignored as authority. This is cooperative coordination and verified owned-tree
containment, not an OS secrecy boundary against a same-user attacker who can
rewrite source/receipts.

## Executed independent controls

Ran the exact pinned interpreter's repaired native CPU suite in the isolated
worktree, forcing every temporary directory below its CRS package:
**27 passed, zero failures/errors/skips**, 2.622s on the reviewer invocation.
No model, framework or GPU operation was performed. Tests include the actual
child witness, success/PID/creation-time rejection, unconsumed GO-fence rejection,
last-parent-handle tree termination and original lost-launch/full-reservation
and resource/numerical denial controls. Builder timing is a separate invocation.

Read the builder's frozen actual probe receipts:

| Probe | SHA256 | Observed result |
| --- | --- | --- |
| valid exact Job | `e19f52c58911cf902b157d58a30c66b5c58e91b8abdf85854949a4423844686a` | Exact flags0x2000, NULL0x3000 retained, handle unusable/error6 before frameworks; four-process tree terminated on parent last-handle close |
| target PID mismatch | `32b5364cc6a76325b1a83309276507daf49b713cb4f0e7ef047e4a3451091bf6` | Rejected before frameworks, handle unusable/error6; observed two-process tree terminated |
| creation-time mismatch | `f1b66d5c19f494807d5a880c7956f2819e6fa8b7bb967ab39cc19c927cdfd016` | Rejected before frameworks, handle unusable/error6; observed two-process tree terminated |

Also executed an independent adversarial actual-child control: transferred a
different reviewer-owned empty Job with otherwise valid strict flags0x2000 and
noninheritable QUERY access to the correctly identified owned runtime child.
The real repaired witness rejected `worker not contained in exact
supervisor-owned Job`; child status was REJECTED_BEFORE_FRAMEWORK, temporary
handle unusable/error6 afterward, model/framework/GPU counts0. Only the bounded
owned fixture processes/Job handles were created and cleaned up. Strict flags
alone therefore cannot substitute for exact membership.

No consequential unresolved operational finding remains for this revision's
registered diagnostic entry path. These tests do not establish a successful
Titan load/forward, device numerics, sustained training or model learning.

## Unchanged scientific/resource boundaries

Direct old/new contract comparison verified unchanged model assets, prompt and
generated token IDs, four-forward/524-input-position workload, generation0,
optimizer0, original capture tolerances and GPU registry. Resource limits remain
600s wall,2GiB process-tree RSS,1.25GiB host reserve,4/0.5GiB GPU floors,4GiB disk
and nominal100ms watchdog; component telemetry ages and one-worker/fencing/
rolling budgets remain enforced. The actual model-forward/scoring body was
not changed by this operational repair.

The relocated-game registry amendment is explicitly cancelled; current exact
process/GPU attribution still needs refreshing before admission. Approval
does not permit another unavailable/unsafe state retry, broaden any length or
dtype, change numerical tolerances, qualify Reader576/capture/training, or open
central CHECK/FINAL. The coordinator must commit these unchanged reviewed bytes,
bind the new clean source tip in the successor approval record and preserve
all predecessor results/costs. Existing source-bound scientific approval must
be rebound to this same unchanged scientific workload before dispatch.

## Machine-readable review equivalent before successor commit binding

```json
{
  "review_id": "CRS-NATIVE-OPERATIONAL-REVIEW-002",
  "reviewer": "/root/data_audit",
  "builder": "/root/runtime_audit",
  "client_date": "2026-10-02",
  "status": "APPROVED_FOR_BOUNDED_DIAGNOSTIC",
  "operational_revision": 2,
  "native_source_sha256": "b94b4dc2ce9e6c041efecaff84a885e6b4fa6e6986d85832d6aa019128d7a852",
  "tests_sha256": "800eb02ffa7ea4889496ef9d478199ab205fe08f9709d6e84ec8caba33087d5b",
  "contract_sha256": "291b7803e745ade2ccfcc5a6e9a8eafc31a9c174541ccd5436f730bde57a8707",
  "amendment_sha256": "112f71e35090d75e4e8906bbe514306ba66ffd33e973a245ad13f8006f2dca94",
  "operations_sha256": "ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30",
  "historical_parent_contract_sha256": "a57a7fa091aeda2a63e8b1452fdeed53474f627246a3f4429b62af924a45babe",
  "source_commit": null,
  "commit_binding_required_before_dispatch": true,
  "tests_passed": 27,
  "tests_failed": 0,
  "tests_errors": 0,
  "tests_skipped": 0,
  "actual_child_exact_job_query": "PASS",
  "wrong_job_rejection": "PASS",
  "pid_creation_fence_rejection": "PASS",
  "temporary_handle_closed_before_framework": "PASS",
  "parent_last_handle_tree_termination": "PASS",
  "prior_native_failure_worker_seconds_preserved": 1.1090000000003783,
  "model_calls_by_reviewer": 0,
  "gpu_calls_by_reviewer": 0,
  "reader_qualified": false,
  "training_qualified": false,
  "authority_expanded": false,
  "fresh_host_admission_required": true
}
```
