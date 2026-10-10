# Same-proposal clarification011

**NOT_READY_COST_ADMISSION_COMPLETE_ENDPOINT_AND_CREATED_WORKER_RECOVERY.**
This separate document corrects010's overstrong cost/endpoint premises. Original
PLAN/PROPOSAL/READINESS010 remain unchanged, including their rejected inferences.
SCIaf0272c6, OPS7d3aa2d2 and dispositiond699d3c5 distinguish a truthful native
worker charge from caller/observer wall, and permit a prospectively finite
observer interval after the unchanged complete caller10s. This clarification
does not declare a platform/API-wide impossibility or operational feasibility.
Authorizationa31f2575 permits this design reasoning only; no source4 is granted.

## Prospective activity and cost contract

The canonical key's proposed native subject is the one pinned UV metadata worker
and its launch/qualification/failure interval already measured by immutable
library3. Let N0/N1 be that library's actual native_started/native_end values.
N0 is before its launch preparation; it is not simply ResumeThread or first UV
instruction. N1 is its verified exit/output endpoint or proved cleanup exit.
All resource, identity, lease, stream, creation and failure work that occurs
inside N0..N1 stays inside the native charge. This proposal neither subtracts
CPU bookkeeping inside that interval nor changes its already-written duration.
Actual nonzero N1-N0, including known overruns, must exactly equal the canonical
native receipt. If the interval/exit is unproved, full10 remains reserved and
global-worker exclusion persists. Unknown is not zero or completed accounting.

The additional cost classification is a **prospective proposal requiring
independent disposition and exact request/admission binding before source or
GO**, not an authoritative exemption invented by this author. The authority for
making this map is clarification011 and the cited reviews; they require all
qualification/failure costs visible and charged to their applicable budgets.
They grant no actual CPU phase, extra native dose or host accounting mutation.

| Actor/activity | Prospective accounting and evidence |
| --- | --- |
| Library/UV governed work N0..N1 | Exact existing native field; reserve10; cumulative7200/14400 with historical47.421 unchanged; no clipping or relabeling failures |
| C preparation/reservation before N0; post-exit cleanup, durable caller artifact and serialization after N1 | Separately measured caller CPU/process and wall costs, including failures; all within caller's existing10s complete endpoint; no zero assumption |
| CPU orchestration inside N0..N1 | Report CPU/process metrics as another dimension, but do not subtract it from native charge or add overlapping wall twice |
| O validation, required observation publication and canonical reconciliation | One separately proposed2s CPU/observer wall allowance; actual process CPU and wall measured; all required qualification/failure work included, no baseline exemption |
| Source/design/fixture engineering | Distinct administrative/CPU provenance under its authorized file/byte/time scope; not an actual worker attempt; unavailable measurements remain unknown, never invented zero |
| Continued native activity or qualification outside a declared interval | Preserve actual activity/cost and stop; classification cannot make it free or fit by clipping. No new reservation/key or native retry |

Caller CPU's applicable prospective phase ceiling is10s wall and at most10s
process CPU; O's is2s wall and at most2s process CPU. These CPU admission fields
must be approved and bound before execution. They are additional visible metrics
of fixed phases, not additions to the one UV execution allowance. Sum only
nonoverlapping activity intervals when reporting total elapsed cost; CPU-seconds
and wall-seconds remain distinct. Unknown/over-budget CPU qualification costs
make the whole observation unaccepted. No separately governed native activity
can be moved into this CPU category solely because a controller performs it.

## One caller interval, one observer interval

An independently qualified owner records T0 before caller preparation, source/
path validation or reservation. Caller deadline D=T0+10 remains unchanged. TC
is observed only after verified UV exit, both EOF witnesses, required peak/RSS/
resource and identity evidence, cleanup/timer/ownership decisions, complete
caller serialization, durable caller artifact publication, closed caller streams
and exact caller exit. A pre-write timestamp or plausible version line is not
TC. A live/unknown owner, missing proof or TC>D is caller failure/unknown.

O begins its **single** retrospective interval at O0 immediately after the
observed caller completion/stop; queue delay after that event counts in it.
Observer deadline OD=O0+2s. The same interval covers validation, required durable
observation/cost artifacts, canonical reconciliation and completed observer
serialization/publication. It applies to success and failure; it is not followed
by another uncharged observation epilogue. It does not extend C/U beyond D.
Calling a blocking fsync/SQLite API is not proof that OD is enforceable.
Independent bounded stop/observation remains an unimplemented qualification gate.

The observer's actual completion OE must be witnessed after the selected fixed
artifacts are closed/published and the producer has actually finished. The
source/identity of that external completion trace must be qualified. O's own
prepublication sample and the ledger finished_at field cannot supply OE.
Select fixed caller/observer artifacts once; do not require a recursive chain
of certificates proving their own publication. No such completion producer is
implemented or attested here, so the design remains NOT_READY.

Late/blocking observer work preserves actual observed duration/cost or unknown
cost and a LATE/FAILED/UNKNOWN whole-observation disposition. It cannot backdate
TC/OE, suppress a later failure, reopen the UV attempt or silently stop counting
at OD. A stop request does not prove rollback of a SQLite COMMIT or process exit.

## Existing durable meanings and safe ordering

The immutable `native_outcome_observation` kind/key receipt remains native
evidence. Existing write_receipt fixes its filename and refuses changed bytes;
verify_receipt checks the canonical filename/kind/key; reconcile requires its
elapsed_native_seconds to equal actual_native_seconds. A larger caller/observer
wall is not automatically an undercharged native receipt under the prospective
map above. If qualifying evidence shows additional governed native time belongs
in that same field, a mismatch is a specific unresolved accounting stop; no
overwrite, diverted receipt or post-hoc cost definition is allowed.

Only after C's completed artifact and native exit/identity/cost evidence are
validated may O invoke canonical native reconciliation. A canonical
SUCCEEDED means **that admitted native worker outcome**, not successful observer
publication or acceptance of the complete scientific observation. Its
finished_at is sampled before transaction completion and cannot prove TC/OE.
Canonical late COMMIT does not retrospectively extend a completed caller, but
can make the observer interval fail. Never invoke SUCCEEDED while caller
completion or native cost/exit proof is merely pending.

Existing write_receipt can retain separately named, immutable caller and observer
candidate/cost evidence without replacing the native receipt or adding a job.
Its arbitrary payload does not itself enforce the whole-observation policy.
An observer candidate must explicitly state PENDING_EXTERNAL_COMPLETION; it must
not publish an immutable whole-ACCEPTED claim before OE is established. Whole
acceptance is a separately qualified assessment of fixed completed leaves,
native outcome, cost admission and externally witnessed TC/OE. The producer,
durable disposition and exact finite completion path for that assessment are
unqualified here; there is no fabricated new OperationsStore terminal state.

If canonical worker SUCCEEDED has already committed but O subsequently fails,
overruns or cannot publish/observe its completed artifacts, retain that native
row and record the distinct whole status LATE/FAILED/UNKNOWN in the admitted
external disposition. Do not overwrite the immutable worker outcome, call the
whole observation accepted or hide the observer failure. If the external
disposition itself cannot be durably completed, whole acceptance remains unknown
and the failed publication/cost is visible. That is a readiness stop, not a
claim of completed observation. No automatic retry or successor loop follows.

## Created-but-unregistered worker: precise remaining case

No GO/resume or absent mark_running is not never-dispatched proof. A created,
suspended UV process has creation/qualification/failure cost and exact ownership
even if it never executes an instruction. Preserve its measured nonzero native
interval or the full unknown reservation, plus caller/observer failure costs.

The fixed operations interface admits mark_running only from RESERVED. For a
row with worker=None, reconcile requires RESERVED plus never_dispatched=True,
launch_attempted=False and actual0/None. Therefore the current immutable-library
failure path that creates U before mark_running and then marks the same row
OUTCOME_UNKNOWN cannot be truthfully terminal-reconciled by pretending no launch
occurred. No existing mark_running-from-UNKNOWN transition is offered. O must
retain ownership and UNKNOWN rather than erase that cost.

This is a specific integration choice still missing, not universal API
impossibility. A prospectively qualified broker might register a genuinely
created identity in RESERVED before an UNKNOWN transition, with exact birth/
identity/owner/fence provenance and no fabricated future worker. That sequence
is not implemented or qualified here, may itself fail before identity exists,
and does not authorize patching library3 or direct DB repair. All such failure
cases must have truthful accounting/recovery before actual eligibility.

## Unqualified ownership/resources and finite review stop

010's broker-held birth ownership, duplicate-handle acknowledgement, pre-Job
creation/caller loss, observer loss and retained last-handle semantics remain
requirements, not a demonstrated implementation. No PID reopen, numeric-handle
takeover, new C/U or retry. The controlled caller/UV/observer resource graph must
respect original2GiB RSS and separate2GiB commit, host/disk/freshness/thread
policy; no memory or CPU exemption is hidden outside that graph. Exact clean
commit/roles, runtime/physical UV/system DLL/search paths/loader/trust, roots,
canonical host DB, current admission and lease/request/outcome evidence remain
unestablished. UV Rust thread enforcement and OS secrecy are not inferred.

Finite proposed falsifiers: delayed caller fsync/serialization across D;
caller loss during creation before mark_running; nonsignaled/unknown held-handle
exit; late observer validation/publication/COMMIT across OD; committed native
SUCCEEDED followed by observer failure; unknown CPU cost; additional governed
native time not represented by the immutable receipt; attempted zero cost for
a created suspended worker; unproved broker identity-before-UNKNOWN or observer
takeover. These are not executed tests.

Freeze this one clarification as NOT_READY. The broad010 premises are corrected,
but cost admission, completion/whole-status production and created-worker failure
accounting/ownership remain concrete open questions. Original source3/3, all
histories/custody/quarantines/keys/Probe002/read-only weights/CHECKFINAL and fixed
budgets remain preserved. Only independent review of this exact clarification
may determine a next disposition; no successive clarification, alternative
route, source4/fifth or actual query is automatically authorized.
