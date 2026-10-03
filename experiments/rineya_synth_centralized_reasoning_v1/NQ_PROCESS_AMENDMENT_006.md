# Direct physical-memory and process observation successor — 2026-10-03

Attempt 6's worker completed all registered mechanics and final source checks.
The [independent audit](evidence/nq001_attempt6_independent_audit_20261003.json)
verified 620 returned API boundaries, 75 forwards / 16,476 positions, 40 native
backwards, two native updates, 20 reference gradients and one reference update.
All 60 CE checks passed, all 112 gradient comparisons were exact, restoration and
optimizer continuation were bitwise, and the 310 base tensors remained unchanged.
The worker wrote `PASS_SCOPED_NATIVE_MECHANICS`; the controller aborted with
`stale_external_CPU_watchdog` and exit 137. Full supervised qualification failed.
Do not reinterpret the worker report as a successful supervised launch.

The prior CPU observation start to the next publication was 219 ms, beyond the
fixed 200-ms bound. The same three PIDs occurred on both sides. Earlier temporary
children are not established as the cause. All measured resource floors held.
Raw outputs, the local private checkpoint and source/config snapshots are kept in
[attempt 6 evidence](evidence/nq001_attempt6_20261003).

CPU-only phase profiling identified psutil's physical-memory query as the major
sampler cost: its GetPerformanceInfo path took approximately 25–35 ms. Direct
GlobalMemoryStatusEx available physical memory matched the before/after psutil
readings in all 50 bracket comparisons, with identical physical totals and no
tolerance. It took approximately 10–18 microseconds. Complete raw controls and
primary API provenance are retained in
[native process/available-memory controls](evidence/native_process_available_control_20261003).
This supports a different implementation of the same physical-memory metric;
the available-memory floor remains 1.25 GiB and preworker admission 3.3125 GiB.

The successor also opens one query/synchronize process handle per current Job
PID, verifies Job membership, reads exact raw creation FILETIME and working-set
RSS, and closes the handle immediately. Terminality requires a signaled process
handle; access denial is not death and receives no broad-system fallback.
New identities use exact raw birth comparisons; historical psutil birth records
retain their legacy checks. The actual absolute System32 kernel32.dll bytes join
the frozen runtime binding. Per-phase and per-API latency records distinguish
measurement work from scheduler delay; coarse monotonic clock zeros do not mean
zero API cost.

Disk checks, 100-ms cadence, 200-ms CPU freshness, 500-ms GPU cadence, two-second
GPU timeout/freshness, all RSS/GPU/system floors, 600-second scope, model, dose
and numerical tolerances remain unchanged. A proposed one-GiB resident CPU
control was denied before child launch by the unchanged RAM admission floor;
it is not a passed control. A new frozen config may attempt native mechanics
only after current admission succeeds. Study training, held-out evaluation,
protected banks and publication remain closed.
