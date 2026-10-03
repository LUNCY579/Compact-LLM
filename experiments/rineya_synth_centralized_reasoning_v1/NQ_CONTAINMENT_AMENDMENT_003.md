# Qualification heartbeat sharing successor — 2026-10-03

Attempt 3 is terminal with owned Job termination and exit 137. Its raw logs,
events, resource samples and source/config snapshot are retained under
[evidence/nq001_attempt3_20261003](evidence/nq001_attempt3_20261003).
Interpreter membership was verified and CUDA initialization returned. No model
construction or checkpoint-loading event was recorded.

Windows refused the supervisor's atomic heartbeat replacement while the worker
was reading the old file with ordinary Python sharing flags. This is a file
sharing failure; it establishes no numerical or pretrained-model result.

An initial proposed read/write/delete-shared Windows reader still encountered a
transient replacement denial in a real CPU-only concurrent read/replace control.
It was not used for a new native attempt. Do not add retries or widen freshness.

Before another attempt, register an append-only `supervisor_heartbeat.jsonl`.
The single CPU monitor writes a complete JSON line plus newline and flushes it;
no replacement or temporary file is needed. The worker reads at most the final
4 KiB of a journal bounded by 1 MiB and selects the newest complete line, ignoring
an incomplete trailing append. Reject stale/future timestamps and wrong
supervisor identity under the same 2-second limit. Immutable terminal receipts
and every source/dose/resource/numerical check remain unchanged. Real concurrent
Windows append/read controls and partial-line controls must pass before freeze.
Do not retry the unchanged failed config or alter the 100-ms watchdog.
