# Offline quality-packet runbook

This runbook covers CPU preparation only. It imports no other experiment's native permit.
Use the assigned worktree and its existing read-only hermes interpreter. Do not install
dependencies into shared environments, hash/load large weights for inventory, or interrupt
another owner. Preserve the stricter shared resource rules.

The supervisor uses the shared `hardware-worker.lock` with Windows FileShare.None, never
deleting or reclaiming another handle. If occupied, the worker writes BLOCKED and exits;
there is no background retry queue. Each command takes a fresh output directory.

```powershell
$qualityPython = 'C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe'
$qualitySupervisor = 'C:/Users/clclc/BusinessProject/ChatBubble-triskill-geo001-sectionh/experiments/titan_quality_repair_v1/offline_supervise.py'
$qualityArtifacts = 'C:/Users/clclc/BusinessProject/ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1'
& $qualityPython -B $qualitySupervisor --stage containment --output-dir (Join-Path $qualityArtifacts 'review-containment-NEW')
& $qualityPython -B $qualitySupervisor --stage tests --output-dir (Join-Path $qualityArtifacts 'review-fixtures-NEW')
& $qualityPython -B $qualitySupervisor --stage prepare --output-dir (Join-Path $qualityArtifacts 'review-packet-NEW')
```

Replace NEW with a unique review receipt name; never overwrite prior runs. The last command
generates all 16 gzip-JSONL prospective training sequences, development membership, new final archive,
oracle/leakage/token receipts and an immutable registry. It does not train or predict.
Final gold is used only by the prospective compiler's deterministic validation; no final
model output exists. Source/core metadata and sidecar hashes are checked; the full weight
digest remains a native-qualified source-custody requirement.

Training artifacts are gzip-JSONL with both compressed-artifact and uncompressed-payload
digests. `packet.read_training_dataset` verifies both hashes, exact 2,000-row dose and
400/0 replay count; it rejects development/final rows and duplicate row IDs. Compression
changes storage only. It does not truncate targets, omit examples or raise output caps.

CPU envelope: one CPU affinity, frameworks blocked, worker-tree RSS/private-commit <=512 MiB,
wall <=900 s, new output <=50 MiB, host admission >=2 GiB, reserve >=1.5 GiB. Cache/TEMP paths
are external. Job assignment precedes fixture imports; descendant termination is tested
with payload-free children. This does not qualify native model/optimizer containment.

Admission and completion source snapshots must match. Consumers require a PASS supervisor
receipt as well as the child's CPU validation; a passing child cannot override resource
limits or source drift. `packet.verify_packet` rejects stale or failed supervisor evidence.

Budget accounting declares 24 cumulative hours: one qualification, eight training and
fifteen evaluation/export. Sixteen training attempts consume their full 1,800-second
declared allocations before entry, even if they fail or die; observed overruns remain
charged. Canonical ledger paths cannot rotate to recover budget. Metadata budgets alone
are not live watchdog enforcement or a native grant.

Read `CPU_VALIDATION.json`, `FROZEN_RECIPE_REGISTRY.json`, `RESOURCE_PLAN.json`, and the
supervisor receipt. `offline_validation_passed=true` is distinct from native qualification.
Missing full HF serialization parity, checkpoint custody, measured throughput or native
containment produces a blocker, not permission to quantize, shorten targets, drop cases,
change recipes or reopen final data. Reconcile source hashes after any implementation edit.
