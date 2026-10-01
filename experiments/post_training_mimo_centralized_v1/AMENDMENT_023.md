# Amendment 023: CPU-only Windows Job containment diagnosis

1 October 2026. Parent continuation authorizes diagnostic and CPU fix work only.
No second native capture is authorized. Base is the preserved V1 evidence commit
`299b25fb219423b87880ca2ceca1a272ba7e780c`; executable source remains reviewed V1
`fa36c38d467f7c48711dbee3823c1bac1cb9c93d`. New isolated branch:
`codex/mimo-job-containment-20261001`. Keep V1 registration, all thirteen raw and
preserved attempt artifacts, its consumed attempt and failed bulk evidence intact.

Question: did the specific-Job guard encounter an API failure or a successful
nonmembership query, and what process lineage/assignment ordering can explain a
reproduction? Do not assume the venv launcher race. Record integer API returns,
last-error immediately after each call, output membership separately, process
PID/parent PID, native creation FILETIME, image path, supervisor and worker-root
identities, unique attempt token and monotonic timestamps.

Register exactly two CPU-only worker launches, in order. Case IMMEDIATE_ASSIGN
uses the existing launch ordering: Popen with CREATE_NEW_PROCESS_GROUP, then
kill-on-close Job creation/configuration/assignment, then GO. Case
READY_BEFORE_ASSIGN uses the same launcher but waits for the actual interpreter's
durable READY before assigning the launcher. The latter is a controlled timing
condition, not evidence that the original V1 had that same ordering. No retries.
Use a traced equivalent of the frozen original WindowsJob constructor, preserving
its structures, flags and assignment target. Record its API calls and parent and
worker membership, including the worker's duplicated specific supervisor handle.
An invalid Job-handle query is a CPU-only error-reporting negative control.

Each worker only records stdlib process/API metadata, waits for GO and exits.
No Torch, Transformers, tokenizer, model assets, GPU query/allocation, training,
checkpoint, scoring or native capture. Neither GPU lease is acquired or changed.
Fresh read-only matching-worker/lease checks must be clear before the suite.
Use the pinned existing venv executable and verify its byte identity.

Hard suite budget: thirty seconds, at most two worker roots and at most four
expected OS processes (venv launcher/interpreter pairs), at most one MiB per
metadata/log file. Per readiness/GO phase at most five seconds; 100 ms supervisor
poll, process-tree RSS <=0.5 GiB and host reserve >=1.25 GiB, disk >=4 GiB.
Prelaunch reuse two >=3.3125 GiB RAM observations two seconds apart. An external
45-second invocation ceiling is a backup; no other process may be stopped.
The backup is a separate CPU-only controller Job. Before either probe it joins
only the verified owned diagnostic supervisor, identified by held handle, native
creation time, image, lineage, source and token, and publishes bootstrap GO only
after successful specific membership. Its inherited nested Job supplies the
external 45-second kill-on-close bound even when a probe is outside its own inner
Job. Controller processes are separate from the two experimental worker roots;
at most four OS processes may be active in the controller Job at once. This CPU
bootstrap is not a native launch qualification or a change to V1.
On cleanup close/terminate only the owned Job and, if an interpreter is proved
outside it, use only a held handle whose PID, creation time, image and descendant
lineage match this CPU attempt. Confirm all held owned processes have exited.
Failing admission, unverifiable identity, API failure or cleanup is a blocking
receipt, not permission to relax containment or add launch attempts.

Commit protocol and clean diagnostic source before physical execution. Preserve
raw API events and READY/GO/result/cleanup records. Interpret a successful query
with false membership separately from API failure. If the controlled case
reproduces an assignment-order defect, distinguish that from the original V1's
missing witnesses. A later CPU-only fix and physical containment qualification
must receive separate prospective registration, focused tests and independent
review. A native capture successor is a proposal requiring parent authorization.

Parent reports this continuation selected gpt-6.1-sol, thinking=max and
service_tier=default. Preserve those settings and global settings. Reviewer
model/reasoning will be explicitly requested; its service tier cannot be verified
through the collaboration interface and must not be claimed as proven.

API interpretation follows Microsoft's [IsProcessInJob documentation](https://learn.microsoft.com/en-us/windows/win32/api/jobapi/nf-jobapi-isprocessinjob)
and [Job Objects documentation](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).
These describe separate API status/membership outputs and child Job inheritance
after association; the local diagnostic must supply the actual ordering evidence.
