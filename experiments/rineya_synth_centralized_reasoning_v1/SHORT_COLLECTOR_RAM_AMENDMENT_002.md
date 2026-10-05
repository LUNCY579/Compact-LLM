# Collector RAM guard and coherent watchdog checks

The short M0 collector now injects the already-proven fresh physical-available
RAM callback into its existing worker guard. Its external monitor selects locked
CPU/GPU publications before reading the check clock. The original freshness
predicates, 100 ms cadence, 200 ms CPU window, resource floors, deadline and
ownership/containment primitives remain unchanged. The original shared supervisor,
worker guard, process-memory helper, RAM callback and reusable old auditor retain
their exact bytes.

The monitor records small in-memory phase/check markers. On abort, a single owner
finishes the original termination/Job-close containment before attempting one
bounded diagnostic publication. Duplicate abort callers cannot publish early.
Diagnostic failure preserves the original abort reason and cleanup fallback.
The supervisor records diagnostic state after sampler join and before inventory
freeze. A successful terminal cannot carry an abort-diagnostic contradiction.

The [reviewed source/history](evidence/short_ram_successor_reviewed_cpu_20261005)
preserves the original collector, the rejected 30-control candidate, the exact
concurrent-abort reproduction, runtime omission and portability predecessors.
All [44 root controls](evidence/short_ram_successor_root_cpu_validation_20261005.json)
pass against the actual integrated files: 18 successor, 16 legacy collector and
10 new-auditor controls. Fresh-checkout tests use tracked originals and temporary
staged fixtures. The executing runtime inventory is the unchanged 64-source core
plus `ast.py`, all 65 entries present in preflight and binding maps.

A deterministic CPU barrier proves that the original clock-before-lock ordering
can reject a later publication as future-dated. The failed native batch04 lacks
its actual failing check clock and in-flight phase; neither this race nor the RAM
callback is established as its cause. All earlier failures and partial prefixes
remain retained and unadmitted. No unchanged retry, partial resume or mechanical
fit claim follows from these CPU controls.

Future collection needs a new committed, source-bound whole-64-row config and
fresh owned resource admission. The new-only auditor requires externally trusted
inventory/profile hashes, exact source commit, root/package/config paths and full
source/runtime65/decoder17/data5/model7 maps. Its write-once inventory registration
belongs under `artifacts/short_m0_audit_registrations/`, created at the exact launch
HEAD after code/config commits; that HEAD stays fixed through terminal cleanup.
This avoids a commit self-reference without weakening byte anchors. Preserve the
registration with the completed raw evidence afterward. The old auditor and old
batches remain valid only within their previously audited source scopes.

No scoring, prior-correct enrollment, study training, language/long-context
permission, protected-bank access or paid-service permission is added.
