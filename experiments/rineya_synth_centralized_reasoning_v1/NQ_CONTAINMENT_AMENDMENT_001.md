# Qualification containment successor — 2026-10-03

Attempt 1 at commit `19a858fc1f9db84e0ebbfc9dc996cf2b166b69b5` is terminal.
The Job was closed, the owned process exited 137 and its PID/create-time is
confirmed dead. Its complete logs, resource observations, source/config snapshot
and preservation receipt are retained in
[evidence/nq001_attempt1_20261003](evidence/nq001_attempt1_20261003).
No worker event journal or model execution result was produced.

The measured failure is a startup watchdog race. The dispatch CPU sample began
at monotonic 47336.468 and completed at 47336.500. The first owned-tree sample
began at 47336.640 and completed at 47336.671, but publication followed heartbeat
IO. The monitor could reject its initial dispatch snapshot as older than 200 ms
before seeing the newly acquired sample. Observed available RAM was
4,873,138,176 bytes and worker RSS 2,088,960 bytes; the failure does not establish
model memory demand or a numerical failure.

Register a separately source-bound successor before another attempt:

1. Construct the worker suspended and assign it to the owned kill-on-close Job.
2. Persist identity/ownership, start external monitoring, and publish a current
   owned-tree CPU sample before heartbeat IO and before worker resume.
3. Verify monitor readiness, current GPU sample, all unchanged floors and the
   original absolute 600-second deadline while the worker is still suspended.
4. Resume only after these checks. A slow or failed initialization remains
   suspended and is terminated through the owned Job.

Keep 100 ms CPU sampling, 200 ms sample freshness, 500 ms GPU polling/2-second
timeout/freshness, all RAM/GPU/disk floors, total scope, numerical tolerances and
forward/update dose fixed. Do not widen a guard or repeat the unchanged failed
configuration. The successor needs fresh source/config pins, CPU controls,
clean committed code and fresh live admission. This preserves the old failure;
it grants no three-arm training or capability claim.
