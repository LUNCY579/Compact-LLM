# Qualification interpreter ownership successor — 2026-10-03

Attempt 2 at commit `942ef867` is terminal with exit 1. Its complete logs,
resource samples and source/config snapshot are retained in
[evidence/nq001_attempt2_20261003](evidence/nq001_attempt2_20261003).
The fixed monitor resumed its contained worker. The Python ownership gate then
failed before the event journal or framework import.

The saved Job tree contains root PID 619816 and interpreter descendants 749208
and 611928. The virtualenv launcher is therefore not reliably the actual Python
interpreter executing the script. Requiring the interpreter PID to equal the
launcher's PID, or its parent to equal the supervisor, rejects a contained child.

Register a separately bound successor before another attempt. Give the owned
kill-on-close Job a fresh exclusive name, persist that name and supervisor
PID/create-time alongside the launcher identity and config hash, and verify the
actual interpreter's membership through `IsProcessInJob`. Close its query-only
Job handle immediately before importing frameworks so it cannot extend Job
lifetime. Record the verified actual interpreter identity in a new receipt.
Reject an uncontained child, stale/reused supervisor PID or changed config.

The single-worker lease and all tree-based RAM monitoring/termination remain.
No numerical tolerance, resource floor, deadline, dose or production permission
changes. Preserve both failures and do not retry either unchanged config.
