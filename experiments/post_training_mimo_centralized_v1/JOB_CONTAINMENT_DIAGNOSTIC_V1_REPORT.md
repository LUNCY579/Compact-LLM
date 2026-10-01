# CPU Job diagnosis V1: launch ordering gap reproduced

One CPU-only invocation used independently reviewed clean source
`3b747dc9b50e3b9a480f57ff49f590084845b623` and prospective Amendment 023.
It stopped in the external-controller bootstrap before either experimental case.
No Torch, model, GPU query/allocation, scoring or training occurred.

The actual interpreter PID 62032 was created **15.6217 ms before** the recorded
start of assigning launcher PID 73968 to the supervisor's Job. The membership
query returned **1 / last-error 0 / member false**. Later assignment of the
verified interpreter returned **0 / error 5**. These witnesses physically
reproduce a late-assignment gap in this instrumented launch sequence. The
original native V1 lacks the corresponding witnesses, so its exact historical
predicate cause remains unproved. The OS reason for the late assignment denial
is also unproved; do not label error 5 as confirmed nested-Job causation.

The two registered experimental worker roots were never launched. The outer
CPU controller used a launcher/interpreter pair, **0.750 seconds** and
**24,469,504 bytes** peak observed tree RSS. Cleanup remains recorded as BLOCKED:
an image identity query returned 0/error31 while the process was exiting. A later
read-only check found both recorded PIDs absent; that observation does not rewrite
the cleanup receipt into a pass.

[Outcome and byte identities](receipts/JOB_CONTAINMENT_DIAGNOSTIC_V1_OUTCOME.json)
bind [all nine preserved raw files](results/job_containment_diagnostic_v1_20261001/outer/OUTER_RESULT.json),
including READY, exact API events, preflight, cleanup and stdout/stderr. All
thirteen original and tracked native V1 artifacts remain byte-identical.
The final CPU fixture receipt records 73 passing tests; fixture success is not a
physical containment qualification.

Next CPU-only registration: create the pinned launcher suspended, assign and
verify its specific kill-on-close Job before resuming the uniquely owned initial
thread. Verify actual interpreter/descendant membership, refusal of a wrong Job,
termination and normal-exit cleanup through held creation-bound handles. Wait for
Job-close exit before treating an exiting image query as a live identity failure.
Do not resume on missing ownership, unexpected threads or any assignment error.
Preserve this consumed diagnostic and the native V1 failure. No second native
capture is authorized.

The ordering mechanism is consistent with Microsoft's [Job Objects documentation](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects):
child association follows an already associated parent. The proposed correction
uses documented [CREATE_SUSPENDED](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags)
and [ResumeThread](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-resumethread).
Physical successor evidence is still required.
