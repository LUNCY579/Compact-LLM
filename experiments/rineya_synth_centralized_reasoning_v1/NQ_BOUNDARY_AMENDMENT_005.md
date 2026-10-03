# Milestone boundary and external sampler successor — 2026-10-03

Attempt 5 is terminal; the owned Job is dead and sampler/backend shutdown was
verified. Raw outputs, 35 stage files, the terminal worker report, resource
observations and exact committed source/config snapshots are retained in
[evidence/nq001_attempt5_20261003](evidence/nq001_attempt5_20261003).

The worker returned 20 native forward/loss/backward sequences and no optimizer
steps. A producer emitted `mechanics_initial_A_gradient_check`, while the worker
writer accepts lowercase stage names. The callback rejected this name before
the first update. The worker report records `NATIVE_QUALIFICATION_FAILED` with
`ValueError: evidence name`. The supervisor additionally stopped the owned Job
when the latest CPU observation exceeded its fixed 200-ms freshness window.
Neither a worker-result file nor earlier passing numeric checks imply a pass.

Completed observations: zero-adapter output parity is exact; both greedy
repetitions and prefix/bulk selected likelihoods agree exactly; the maximum
completion CE/reference error over 20 rows is 4.76837158203125e-7, below 2e-6.
Peak owned-tree RSS was 1,361,661,952 bytes; minimum available RAM was
3,051,470,848 bytes; minimum externally observed GPU free memory was 877,387,776
bytes. Direct GPU observations took at most 16 ms. Resource floors were retained.
These results qualify only the completed individual checks. Acquisition,
optimizer/restoration mechanics, transfer and retention remain unmeasured.

Before another source-bound attempt, use the lowercase producer name and enforce
the same name contract before invoking a preservation callback. A CPU toy must
exercise the actual worker journal across successful mechanics, including all
eight milestone files, 60 loss files, hashes and attempted/returned operation
counts. The new integration control passes without changing model operations,
optimizer dose, numerical tolerances or any resource limit.

Review external CPU observation timing separately. Retain the 100-ms cadence,
200-ms freshness, actual owned Job membership/birth identities, complete tree RSS,
system reserve and disk checks. Register any implementation successor and its
CPU-only controls before freezing a new config; do not retry the failed config.
The full M0/M1/M2 study and protected CHECK/FINAL access remain closed.

The timing review found consecutive 109-ms and 94-ms snapshots: the prior
observation start to the next publication was 203 ms. The successor samples
each current owned-Job PID with one fresh process object, reading its birth time
and RSS together and checking the leased root's birth identity. Previously the
sampler constructed and queried each process twice. RAM and disk observations
remain in the same sample; no checks or freshness thresholds are removed.

[CPU sampler controls](evidence/cpu_sampler_query_control_20261003/SUMMARY.json)
retain before/after source bindings and ownership records. The after control's
50 complete frames are available; missing original before frames are explicitly
identified. Mean idle sample time changed from 25.3 to 26.9 ms, so there is no
observed speedup. Both controls stayed within 200 ms; a mock also proves the
duplicate construction is removed. The new scoped native attempt must still
satisfy the live watchdog rather than inheriting idle qualification.
