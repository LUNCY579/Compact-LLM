# Amendment 024: suspended Windows Job CPU qualification

1 October 2026. CPU-only continuation on `codex/mimo-suspended-job-launch-20261001`,
based on `a8f5ed6dda8485cabb7670101bb87880634648b6`. This separately registers a
correction qualification. Both earlier attempts remain consumed. Keep all thirteen
native V1 and nine diagnostic raw artifacts and their byte-identical copies.
No second native capture is registered or authorized.

The measured CPU bootstrap showed the actual interpreter existed 15.6217 ms before
launcher assignment. Its specific-Job query succeeded with false membership; later
verified assignment returned zero/error 5. The denial's OS cause and the historical
V1 guard cause remain unproved. The recorded cleanup failure remains a failure.

Question: does CREATE_SUSPENDED followed by assignment and successful specific-Job
membership before one ResumeThread prevent that gap, and does the contained
interpreter/descendant tree exit through owned handles under cleanup controls?

Only the CPU launcher helper changes. Use public Popen creationflags 0x204, a new
kill-on-close Job with the existing 0x2000 limit, and verify the suspended root's
native PID/creation/image/lineage identity. Require exactly one snapshot thread
owned by that Popen root, open it with 0x0802 rights, verify its owner and native
creation time, then recheck root identity and specific membership before resuming.
Resume exactly once; previous suspend count must equal one. Any failure blocks GO.
No late assignment fallback, privilege adjustment, ACL change or undocumented API.
Retain aggregate API returns, immediate last-error, output membership, source hashes,
unique token and both native creation and monotonic timestamps.

Freeze three sequential main worker roots in config order, without retries:

1. NORMAL_EXIT_CLEANUP: contained pinned venv worker creates one stdlib dummy
   descendant, using the separately pinned actual interpreter. Hold and verify all
   descendant identities while the main worker is live. Release the main worker
   to exit zero; prove the dummy is still active through its held handle before
   closing the final owned inner Job handle. All retained process handles must
   become signaled; closing the Job must cause cleanup, with no manual fallback.
2. TERMINATE_ACTIVE_TREE: main worker and its single dummy are active and contained.
   Terminate only their verified inner Job and confirm every retained process
   handle becomes signaled, with no manual process fallback.
3. WRONG_JOB_GO: give the correctly contained worker a separately created empty
   Job. Require successful IsProcessInJob with false membership, refusal before
   any CPU work marker and expected exit two. Verify owned cleanup.

The external controller uses the separately pinned actual interpreter directly,
so its one process plus a venv pair and one dummy stay within the unchanged
four-process simultaneous outer Job ceiling. It too is launched suspended and
verified before resume/BOOT_GO. Total created processes <=9: one controller,
three launcher/interpreter pairs and two single-process dummies. Root invocations
above that controller are the existing execution tool/pinned venv, not experimental
workers. No hidden retry or unregistered diagnostic launch.
Enforce the cumulative ceiling with public Job lifetime TotalProcesses accounting,
including a final query before the Job closes; sampled PID sets are separate
observations. Bind each actual interpreter's READY/held image path and bytes to
the frozen actual interpreter before GO. Termination-control exit waits precede
inner Job close, so that close cannot substitute for the tested termination.
The dummy's autonomous exit is at least fifty seconds after arming, outside the
entire unchanged 45-second outer window. The outer containment still bounds its
actual run. Record that deadline. Require the active-termination control's held
exit codes to equal the requested TerminateJobObject code two.

Unchanged resource limits: 30-second suite, 45-second external ceiling including
cleanup, five seconds per metadata phase, 100 ms guard polling, <=0.5 GiB observed
full-tree RSS, >=1.25 GiB available host RAM and >=4 GiB disk. Before workers use
two >=3.3125 GiB host RAM samples at least two seconds apart. Each metadata/log file
<=1 MiB. Failure consumes this registration. The RAM upgrade does not expand any
limit. Fresh read-only matching worker and lease checks precede the attempt.
No GPU query, lease acquisition, model, Torch, tokenizer, training or scoring in
the qualification itself. Hardware check-in queries are separate read-only actions.

Cleanup waits on each previously identity-verified held process handle after Job
close. If still active after a bounded wait, verify live identity before any
owned termination. An image-query failure may be classified as an exit race only
if a subsequent bounded wait on that same held handle proves it has exited.
Otherwise block and preserve the failure. Attempt all owned cleanup even after
logging/API errors. No kill of enumerated or unrelated PIDs. PASS requires held
handle exit confirmation, intact evidence and both deadlines; absence alone fails.

Commit registration before implementation; commit clean reviewed code and focused
fixture results before the single physical attempt. Obtain independent read-only
review before and after qualification. Preserve raw evidence, source and config
hashes, process accounting, observed RSS/minimum RAM/disk, cleanup and grant flags.
CPU fixture success, physical CPU containment and native mechanical/capability
evidence are distinct. All capture/capability/E1 grants remain false.

The requested main execution settings remain GPT-6.1 Sol / Max / Standard.
Reviewer model/reasoning are requested explicitly; collaboration exposes no
service-tier verification. Neither global settings nor original V1 launch code
are changed.

Public API basis: [CREATE_SUSPENDED](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags),
[thread rights](https://learn.microsoft.com/en-us/windows/win32/procthread/thread-security-and-access-rights),
[ResumeThread](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-resumethread).
Their semantics justify the ordering; this registered local attempt must prove the
actual containment and cleanup.
