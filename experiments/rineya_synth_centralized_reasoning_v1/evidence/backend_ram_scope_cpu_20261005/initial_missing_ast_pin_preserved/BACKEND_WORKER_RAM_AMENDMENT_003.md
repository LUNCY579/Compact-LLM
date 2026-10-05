# Backend worker RAM callback amendment 003 — staging only

Recomputation phase A stopped at its unchanged external CPU freshness gate; no
C9/W9/W10, sustained-nine or fresh-process continuation qualification exists.
The original A1/V2 configs and failed receipts remain immutable. This candidate
changes only the backend worker's `make_guard(..., available=...)` callback.
Original inference/native helpers and the external supervisor are unchanged.

The existing qualified `native_process.system_memory_status()` makes a fresh
`GlobalMemoryStatusEx` call. The candidate extracts `available_bytes`, requires
the exact two-field schema and nonboolean UInt64 values with
`0 <= available <= total`, `total > 0`, and propagates API/errors. It caches no
numeric observation and has no fallback. Every original guard invocation,
owned-RSS/RAM floor, deadline, GPU check and heartbeat rule remains in place.

The primary-source review binds psutil7.2.2 `GetPerformanceInfo` available pages
times page size and `GlobalMemoryStatusEx.ullAvailPhys` to the same documented
physical-memory meaning and byte units. Separate volatile observations need not
agree bitwise. The preserved CPU comparison completes 512 default and 512 direct
callback guards with source/runtime equality, partial-before-metadata evidence
and owned cleanup. It measures CPU components only; native model/tensor/GPU
guard costs, historical bottleneck causality, disk-stall prevention and full-run
fit remain unproved. The staged transplant retains the proven callback body AST;
only safe stage/root package bootstrap differs.

`SYNTH-BACKENDQ-003-WORKER-RAM` has a separate combined profile, recipe and exact
callback/helper/CPU-proof/primary-review policy binding. It retains the existing
82 runtime pins and checkpoint V2 mechanism: eval28, zero dropout, BF16 resident
base/FP32 computation, 112 rank8 q/v factors and 1,146,880 trainables. Rows, masks,
optimizer/global LR, 131,072-token study ceiling and numerical checks do not
change. Phase A remains 202 forwards/48,112 positions/200 backwards/10 steps/16
full-base checks and 56/5,600/5,600 decoder regions. B remains 42 forwards/10,650
positions/20 backwards/one step/20 independent reference gradients/nine base
checks and 56/1,120/1,120 regions. Qualification checkpoints never warmstart the
study.

All unchanged gates remain: four-GiB admission/disk, half-GiB work GPU floor,
600-second whole scope, 200-ms CPU freshness and 100-ms cadence. Staging denies
config writing and native dispatch before ownership/framework work. Reviewed
integration, clean committed source/config bytes, a new write-once explicitly
authorized qualification manifest, fresh resource admission and independent
terminal audit are still required. This draft grants no native run, study,
scoring, DEV collection, protected-bank, paid-service or retry permission.
