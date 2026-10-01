# Suspended Job CPU V1: preserved blocked qualification

The single registered attempt from clean `98c6e08e0db8f91bf6fde72979d495dbce6de29b` is consumed and **BLOCKED**.
99 CPU fixtures and independent prelaunch review passed before this physical run.
It stopped in NORMAL_EXIT_CLEANUP before WORKER_ARMED: `publish_go` tried
`int(job_handle)` on a ctypes `wintypes.HANDLE` returned by DuplicateHandle.
That requires its integer `.value`. DUMMY_GO was never published; the controller
waited five seconds and blocked. The other two registered cases never started.

Physical evidence confirms specific membership before GO for the controller and
venv actual interpreter, successful parent-query dummy membership, two single
resumes with previous count one, and held-handle exit of all four owned processes.
No manual cleanup fallback or lost API log occurred. This proves observed launch
containment and failure cleanup; the full three-case controls remain unqualified.
The interpreter was created 116.5275 ms
after this launcher's successful assignment, whereas the preserved older bootstrap
measured creation 15.6217 ms before assignment. The historical native V1 predicate
cause and the OS reason for its later error5 still remain unproved.

Outer time 8.188 s; inner time 5.407 s.
Job lifetime/peak active counts 4/4. Observed tree RSS
64.035 MiB, minimum RAM
9.722 GiB and disk
81.544 GiB. All model/GPU counters were zero.

The RAM upgrade check found ASUS TUF Gaming A16 FA607PI, Ryzen9 7940HX,
two16GiB modules configured5200MT/s,31.22GiB exposed by Windows, RTX4070Laptop8188MiB.
Fresh CPU preflight recorded no matching workers, both GPU leases absent,9.686GiB
RAM and81.545GiB disk free. Neither lease was acquired.

[Outcome/byte manifest](receipts/SUSPENDED_JOB_CPU_V1_OUTCOME.json) binds 26
unchanged originals and tracked copies, alongside all13 prior native V1 and9
prior diagnostic artifacts. Raw [suite result](results/suspended_job_cpu_qual_v1_20261001/suite/QUALIFICATION_RESULT.json),
[outer result](results/suspended_job_cpu_qual_v1_20261001/outer/OUTER_RESULT.json) and
[worker traceback](results/suspended_job_cpu_qual_v1_20261001/suite/NORMAL_EXIT_CLEANUP/stderr.log) remain failures.
The independently passed source/review and blocked earlier reviews are preserved.

Next: fix typed-handle serialization with focused fixtures and independent review
on a separate source. Propose a fresh CPU qualification with identical30/45s,
four simultaneous/nine total processes,0.5GiB RSS,1.25GiB host reserve,3.3125GiB
two-sample preworker reserve,4GiB disk and1MiB files. Do not rerun this registration.

No second native capture is authorized. After CPU controls and parent review,
a separately registered native proposal would retain six forwards,774 input
positions,three generated tokens,one construct/load,310 tensor copies,318 logical
calls and600s maximum, using existing streaming/short-context reserves. The
original bulk error max0.0414324/mean0.0262994 still fails frozen0.03125/0.01.
Capture/capability/E1 grants remain false; backend40 is mechanics-only and E1 still
requires independent references/grader authorization,long-context/resource fit
and native evidence validators. No scoring,checkpoint,training or other-lane work
was changed. Five unpublished patch objects remain preserved and unintegrated by
this task. Requested Sol Max/Standard settings were preserved; reviewer service
tier is not verifiable through its interface.
