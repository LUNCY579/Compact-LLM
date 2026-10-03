# Direct telemetry and owned-Job accounting successor — 2026-10-03

Attempt 4 terminated with `stale_external_GPU_observation`; its owned worker is
dead and all raw outputs plus committed source/config snapshots are retained in
[evidence/nq001_attempt4_20261003](evidence/nq001_attempt4_20261003).
The loader returned after copying all 310 base tensors. Zero-adapter forward and
greedy capture API boundaries returned, followed by one native backward and two
native forwards. No optimizer step returned, no worker-result receipt was written,
and no scoped qualification, acquisition, transfer, or retention pass is claimed.
Intermediate numerical values were in memory when the supervisor killed the Job.

The observed minimum free GPU memory was 2,456,813,568 bytes; the peak owned-tree
RSS was 1,356,242,944 bytes. Neither resource floor caused termination. The last
CLI GPU observations took 1.281 and 0.891 seconds. Age is conservatively measured
from query start; the prior observation crossed the fixed two-second limit while
the next CLI call was pending. Preserve this failure; do not retry its config.

Before another attempt, register three implementation changes:

1. Query the actual absolute Windows System32 `nvml.dll` directly for GPU0 UUID
   and free/total memory. Pin its bytes with the runtime files. Use only NVML
   initialization, identification, memory and cleanup APIs. Keep the 500-ms
   observation cadence, two-second query timeout and two-second freshness limit.
   A timed-out DLL call is explicitly pending; it is never silently retried.
2. The worker queries the owned Windows Job's process IDs with a query-only
   handle, closes it immediately, and sums their RSS. This replaces repeated
   whole-system descendant enumeration during each small streaming chunk.
   Retain process-tree accounting, current-worker membership and fail-closed
   access/query errors. The independent 100-ms supervisor watchdog remains.
3. Write bounded, write-once stage measurements after loader/parity/capture and
   mechanics checks, including update, restoration and reference comparison
   milestones. Preserve capture tokens and numerical measurements before
   testing the gate so a hard termination cannot erase earlier completed work.
   Verify complete writes and encode invalid nonfinite measurements explicitly
   in strict JSON. Persist the terminal worker receipt before optional final
   CUDA diagnostics. These partial files do not establish qualification.

Read-only direct NVML controls and real CPU Job controls are retained in
[nvml_current_source_control_20261003.json](evidence/nvml_current_source_control_20261003.json)
and [native_job_accounting_control_20261003.json](evidence/native_job_accounting_control_20261003.json).
Twenty current-source NVML queries took at most 47 ms; five Job RSS samples took
at most 2.362 ms. Those observations establish control functionality only.

External sampler and backend threads must be verified terminal during shutdown.
If any remain live, record `SUPERVISOR_SHUTDOWN_UNVERIFIED` and do not claim a
frozen partial-output hash inventory. Close the owned worker Job regardless.

Real CPU-only Job accounting and read-only NVML controls must pass before source
freeze. No model is used by these controls. The registered model, 40-row dose,
two native updates, one reference update, numerical tolerances, two-GiB RSS,
system/GPU floors, forward caps and whole-scope 600-second deadline stay fixed.
The fifth source-bound attempt is mechanical qualification only; production
M0/M1/M2 training and protected CHECK/FINAL access remain closed.
