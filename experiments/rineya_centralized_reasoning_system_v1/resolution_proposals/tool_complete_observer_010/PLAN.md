# One complete metadata caller/observer boundary: design010

**Disposition: BLOCKED_COMPLETE_ENDPOINT_OR_ACCOUNTING_BOUNDARY.** This is one
design for the existing metadata question, not source acceptance or a launch.
The current exact library/accounting interface cannot establish the required
complete, fully charged, deadline-safe success. This conclusion concerns the
pinned API/contract below; it is not an impossibility claim about every platform
or a request to extend ten seconds.

Accepted library3 `82f211eb` and V3 `e1a23e03` remain immutable. Library history
is exhausted at3/3. Design010 grants no cumulative integration round4 exception,
fixture, actual key, reservation or query. Independent design reviews must assess
this blocked finding; the builder does not approve it.

## Selected concrete boundary

One external observer/broker O controls one caller C executing the immutable
library, and the one pinned UV process U. O has a supervision thread separate
from its potentially blocked receipt/accounting thread. C invokes the library
through an injected typed broker port/kernel interface; it must never fall back
to default local CreateProcessW. This interface is a design requirement only.
No broker, caller, RPC bridge or new executable is implemented here.

O creates C suspended and retains its exact process/thread and containment Job
handles before acknowledging C to any runner. O also performs every U creation
requested by C. O keeps the original U process/thread handles and U Job; C may
receive separately duplicated handles only after O records acknowledged custody.
O must retain U even if C dies before receiving that acknowledgement, before
first identity collection or inner Job assignment. Numeric handle/PID records
are evidence descriptions, not ownership. No PID reopening, unrelated kill,
replacement C/U, second query or retry is allowed.

The required birth-containment primitive must atomically place each created
process in its broker-held containment Job at creation, including the interval
before CreateProcessW returns. A post-return assignment promise is insufficient
if creation blocks across the deadline. The selected files do not qualify that
primitive, duplicate-handle/RPC semantics, nested Job compatibility or physical
Windows/runtime dependencies. These requirements remain a launch stop.

U's dedicated inner Job preserves ActiveProcessLimit1, no breakaway, zero U
descendants and separate2GiB commit. C/O control roles cannot hide memory or cost:
all controlled caller/worker/observer activity must fit an independently bound
combined2GiB RSS/commit resource scope, with the original host/disk/freshness and
numerical-thread policy. No baseline observer exclusion is invented. Exact
resource enforcement for that graph is unqualified; the design does not attest
that extra processes are admitted by an existing host profile.

If O loses C, it retains U/Job ownership and independently requests termination
on those exact held objects. Failed termination, nonsignaled/unknown wait,
uncertain membership or unquiescent callbacks retain the last valid handles and
the full unknown reservation. Other safe cleanup remains independent. If O
itself dies, no durable takeover is specified or qualified: this is a refusal,
not an inference that kill-on-close eliminated every pre-Job process. A new
observer cannot recover authority from PID metadata. This missing contingency
also precludes operational readiness.

## Exact ten-second event and publication boundary

O records monotonic t0 before C preparation, validation or canonical reservation;
the independent absolute deadline is D=t0+10. Only one predeclared request/key,
owner/fence and epoch may proceed. O's supervision thread remains able to request
termination of C/U/Jobs while the caller or accounting thread blocks. Requests
are not proof that an OS call or process ended exactly at D. Unknown or actual
late completion is preserved, never made success by an earlier clock sample.

The fixed leaf caller artifact is the complete serialized bytes of the library
result, raw output, EOF/exit/peak/resource evidence, source/runtime/request pins
and cleanup state. Its schema excludes the in-memory recovery object; any live
owner is retained by O, never serialized as if transferred. A caller artifact
with live/unknown ownership, incomplete timer quiescence or missing exit/EOF/
peak evidence is ineligible for success. C flushes its bytes, completes durable
publication and closes its publication/output streams; O measures tC only after
those events and exact caller exit have actually been witnessed.

O then validates those fixed leaves, publishes one complete-observation receipt
and performs canonical outcome accounting. The proposed validation limit is
min(0.5s,D-tC); the proposed receipt/accounting limit is min(0.5s,remaining time).
Both are finite refusal limits within the original ten seconds, not a new dose
or evidence that blocking SQLite/fsync calls can be forcibly bounded. Negative
remaining time immediately disqualifies success. Required observer work is not
moved outside D. Success endpoint tE is measured only after durable caller and
observer publications, complete serialization, owned exit/resource/EOF/cleanup
and canonical accounting completion; tE<=D is necessary.

There is no infinite chain of certificates certifying their own writes. O's
assessment selects the fixed caller leaves and records their completed events.
Its own required durable publication and accounting are included before tE;
there is no claim that a timestamp before that publication proves its endpoint.
The current API cannot safely realize that last requirement, as explained below.

For failure/unknown only, a prospective postdeadline observation/reconciliation
phase has a separate0.5s validation and0.5s accounting cutoff. It cannot authorize
continued C/U execution, backdate success or erase costs. If either call blocks,
its completion and cost remain unknown; these cutoffs are not asserted OS bounds
and no operational completion is declared. Design010 executes none of them.

## Distinct clocks and the pinned accounting conflict

O must retain monotonic t0, actual U dispatch start tU, verified held-handle U
exit tX, leaf caller completion tC, validation/publication/accounting events and
tE or the actual failure stop. Wall-clock lease/receipt times are separately
bound to O's monotonic observations; lease time is not a substitute stopwatch.
Report U interval tX-tU, caller interval tC-t0, observer interval after tC and
fully charged governed interval from t0 through actual required finalization.
Overlapping intervals are not summed twice. Qualification and failure work is
charged once under the same key; actual known overruns are never clipped to10.
Unknown work retains full10 and global-worker exclusion; that is unresolved
exposure, not completed qualification or permission for another dose.

**A — the receipt freezes a narrower cost.** Library3 finalization writes
`native_outcome_observation` for the same key, with elapsed_native_seconds equal
to its native_started/native_end interval. That starts after preparation and
reservation and ends before later caller/observer qualification. operations.py
write_receipt460–486 fixes one kind/key filename and refuses changed existing
bytes. verify_receipt also binds that exact filename. reconcile371–437 requires
that receipt kind/key and requires actual_native_seconds equal to its recorded
elapsed_native_seconds. Reusing it therefore charges the narrower U/library
interval, not the full governed qualification/failure interval. Replacing its
duration conflicts with immutability; substituting a differently named receipt
does not satisfy the pinned verifier. No bridge silently rewrites the accepted
receipt or diverts canonical accounting in this design.

**B — terminal success cannot wait for its verified durable endpoint.**
reconcile reads finished_at before UPDATE/event and the enclosing transaction
commit. It has no absolute-deadline/complete-observer commit primitive and makes
terminal outcomes immutable. A call begun before D can durably commit SUCCEEDED
after D, while the observer learns that fact only after the commit. Killing a
blocked caller/accounting thread does not demonstrate rollback; querying the
result later cannot turn immutable late SUCCEEDED into FAILED. Invoking it after
tE also moves required accounting outside the measured endpoint. A second
certificate or permanent UNKNOWN cannot resolve this ordering conflict.

Thus the exact pinned protocol provides no accepted transition that both fully
charges this proposed complete interval and forbids premature immutable success.
An additive observer job/reservation is not a remedy: reserve319–341 permits one
global admitted worker and unresolved work must be reconciled first. A new key,
second10s reservation or cost-free observer exclusion would change the existing
scope. No accounting edit or source exception is granted here.

## Required identities and finite falsifiers

Before any future GO, a separately reviewed exact caller/broker source and clean
commit would have to bind nonbuilder SCI/OPS receipts, accepted library/V3,
pinned runtime, physical UV hash/68317696bytes, controller DLL/search paths,
selected loader/trusted metadata disposition, empty physical roots, canonical
host DB identity, current lease/fence/request/epoch, fresh resources and exact
worker identities. Current symbolic assertions are not those receipts. The
fixed target remains C:/Users/clclc/.local/bin/uv.exe with only --version,
SHA5a7ec858, one attempt,8192 combined bytes plus one overflow sentinel, zero
descendants and no retry. Emitted text is only a self-reported release lookup
claim, not publisher/source/setup proof or OS secrecy.

The following are proposed review falsifiers, not executed tests:

| Case | Required observation; contrary result rejects the design |
| --- | --- |
| C blocks in fsync/SQLite/serialization across D | Independent stop request, exact owned handles retained, no success and full actual/unknown cost |
| C dies during U creation before acknowledgement or Job assignment | O still owns U from birth; no PID reconstruction or replacement query |
| U termination returns false or successful request remains nonsignaled | Live exact process/necessary Job retained; no last-handle close |
| Timer/callback is unquiescent or O dies | No completion claim or fabricated takeover; unknown ownership remains a stop |
| EOF/peak/resource/identity proof is missing | No release or metadata success despite a plausible version line |
| Caller leaves finish before D but observer publication commits after D | No canonical SUCCEEDED; this is not proved by current reconcile |
| Canonical COMMIT is blocked/crashes at D | No assumption of rollback or early immutable success; preserve exact unknown/late state |
| Complete cost differs from immutable narrow exit receipt | No shortened charge or receipt replacement; current equality refuses |
| An extra observer reservation/key is requested while unknown remains | Refuse additive admission and duplicate query |
| Apparent success is followed by late qualification failure | No immutable earlier success accepted; current API cannot repair it |

## Stop and restart condition

Freeze BLOCKED_COMPLETE_ENDPOINT_OR_ACCOUNTING_BOUNDARY for this exact design.
Retain accepted library evidence and all prior histories/custody/quarantines,
keys/Probe002, read-only weights/CHECKFINAL and7200/14400 budgets with47.421
historical seconds. No actual query is consumed or authorized.

Only independent disposition of these concrete blockers can determine whether a
different explicitly approved exact API/accounting boundary is warranted. That
would require a separately recorded cumulative same-milestone source exception,
not a reset of library3/3 or automatic round4/fifth/retry. This design grants
none. Preserved UNKNOWN is not the intended completed measurement.
