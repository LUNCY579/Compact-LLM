# Amendment 025: retained-handle supervisor-death CPU control

1 October 2026. Separately register
`RINEYA-SUPERVISOR-DEATH-RETAINED-JOB-CPU-V1` on
`codex/mimo-supervisor-death-cpu-20261001`, based on verified
`863476aaf1e394a0796f4c59dcf96f0cc3793d4a`. Commit this registration before
implementation. This authorizes CPU code, fixtures and review only. A physical
attempt requires later parent review and an explicit exact-source authorization
record. No such record is supplied by this amendment.

G05a is a source/API inference, not an observed escape. In the consumed V1 CPU
worker, `handshake` returns a duplicated inner Job handle; `worker` keeps it open
through dummy setup. Microsoft's Job Objects documentation states that
kill-on-close takes effect when the last Job handle closes. DuplicateHandle
creates another reference to the same object. Supervisor termination closes its
handles, but does not normally terminate its children. A retained child handle
therefore describes a distinct untested case. Normal exit, explicit Job
termination and wrong-Job refusal do not qualify supervisor-death cleanup.

Freeze two sequential controls with no retry:

1. `SUPERVISOR_DIES_WITH_RETAINED_DUPLICATE`: an outer observer enrolls a direct
   pinned interpreter supervisor in a separate kill-on-close safety Job before
   one verified resume. The supervisor enrolls one direct pinned interpreter
   retainer in an inner kill-on-close Job before resume. The retainer verifies
   specific membership with a noninheritable duplicate and stalls in a known
   sleep-only state. Hold both creation/image/lineage-bound process handles.
   Terminate only the verified stalled supervisor through its original Popen
   handle. Require that handle to signal with the requested exit code. Observe
   the held retainer handle nonsignaled for at least one second. Then close the
   registered duplicate remotely with documented DuplicateHandle using the held
   retainer process handle, NULL target/output and DUPLICATE_CLOSE_SOURCE. Require
   API success and retainer exit through the same held process handle, before
   any outer safety intervention.
2. `SUPERVISOR_DIES_AFTER_DUPLICATE_CLOSED`: establish the same tree; close the
   retainer duplicate first with that documented remote-close operation. Require
   both processes still active for at least one second while the supervisor owns
   the inner Job. Then terminate the stalled supervisor and require both held
   handles to signal before any outer safety intervention.

The observer never opens, queries or duplicates the inner Job. Remote close uses
no target Job handle, so it cannot retain another reference that confounds the
last-close condition. The retainer reports the scalar duplicate value before
stalling; the observer binds it to the authenticated nonce, frozen sources and
held PID/creation/image/lineage identity. It requests PROCESS_DUP_HANDLE only for
that verified process. No privilege, ACL, security-setting or access-denial bypass.
DuplicateHandle with CLOSE_SOURCE may close the source even when it fails: a
failed API therefore blocks qualification regardless of subsequent process exit.
The child never uses or closes the stale wrapper after remote closure.

The independent outer safety Job remains open throughout experimental exit
witnesses. An independent deadline watchdog may terminate that outer Job solely
for safety. Any such intervention, premature outer Job close, manual fallback,
failed/missing API evidence or absent outcome makes qualification fail. Safety
cleanup never supplies an experimental exit witness. Job accounting is queried
on the outer Job only, including before its final close. A missing/failed case
receipt makes aggregate creation counts unknown, with separate observed lower
bounds rather than zero. Process disappearance is not an exit witness.

Keep unchanged limits: 30-second suite including admission, 45-second external
ceiling including cleanup, five seconds per metadata/exit phase, 100 ms polling,
four simultaneous/nine cumulative experimental OS processes, 0.5 GiB tree RSS,
1.25 GiB available RAM reserve, two preworker samples of at least 3.3125 GiB
separated by two seconds, four GiB disk and one MiB per file. The planned direct
interpreter shape creates four experimental processes total, two at a time.
Autonomous supervisor/retainer deadlines are at least 50 seconds after arming,
outside the outer ceiling; they cannot supply an experimental exit in time.
No models, Torch, tokenizer, GPU query, lease acquisition, training or scoring.

Fresh matching-worker, lease, RAM and disk checks precede any later physical
attempt. The external launch authorization must bind experiment/config/source,
the exact budget, parent review and explicit permission. Missing authorization
refuses before Kernel or process creation and does not consume an attempt. Once
authorized admission starts, a tracked consumed receipt or existing output/start
namespace refuses reuse. Failure consumes this registration; a successor needs a
new ID, namespace and prospective registration.

Preserve all 13 native V1, nine first diagnostic and 26 suspended CPU V1 raw
artifacts, tracked copies and original bytes, all consumed registrations, the
typed HANDLE.value fix and consumed-V1 refusal. Do not modify those execution
sources or the separate tripleskill lane. New code is a separate control harness,
not a claimed repair or physical qualification of native launch containment.
Fixture success, physical CPU control and native mechanics remain separate.
Capture, capability and E1 grants remain false; frozen bulk tolerances stay fixed.

Public API basis:
[Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects#managing-job-objects),
[DuplicateHandle](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle),
[Terminating a Process](https://learn.microsoft.com/en-us/windows/win32/procthread/terminating-a-process).
Validate the exact implementation with focused CPU fixtures and independent
read-only review before returning a prospective physical proposal. Requested main
settings remain GPT-6.1 Sol / Max / Standard; service tier is not independently
visible through the review interface.
