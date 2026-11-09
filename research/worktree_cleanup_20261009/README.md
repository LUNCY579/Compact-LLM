# Rineya worktree cleanup record — 2026-10-09

This record preserves the user-authorized cleanup, its source/evidence custody,
and one byte-identical parser dependency handoff. Research branches and open
questions remain; an idle checkout retirement is not a scientific rejection.

The timestamped cleanup endpoint recorded **14 checkout removals**, **3 stale
registrations pruned**, **30,052,209,270 logical file bytes removed**,
**19 verified remote source tags**, and **8 independently downloaded and
hash-verified private scientific releases**. The endpoint had 29 populated
worktrees. These are historical events and file lengths, not a current net
disk-allocation measurement.

The later publication snapshot at `2026-10-10T03:30:04.390487+00:00` counted **35 populated worktrees**, including **2 restored paths**. Other owners continued independently. Their active branches, indexes, new checkouts, and unpublished work were not reset or published as part of this record.

## Records and recovery

- [Human-readable audit](audit-report.html) and [structured result](audit-report.json).
- [Detailed metadata index](evidence/index.json), with minimized compressed inventories.
- [First-pass removal receipt](cleanup-receipt.json) and [second-pass receipt](phase2-cleanup-receipt.json).
- [Later owner restorations/current snapshot](publication-current-state.json).
- [Remote source tags](remote-preservation.json) and [additional source tags](phase2-verified-tags.json).
- [Private scientific archive verification](phase2-remote-release-verification.json) and [hybrid archive verification](supa-hybrid-remote-verification.json).
- [Parser handoff](phase2-parser-migration-receipt.json) and [independent source review](publication-parser-review.json).

Add a worktree at the original path from its retained branch or archive tag to
restore a retired checkout. For snapshots containing local material, download
the linked private release and extract its archive at that checkout root;
consult `_WORKTREE_STATE.json` where present for tracked differences and
deletions. The earlier hybrid archive uses `ARCHIVE_MANIFEST.json` instead.
Do not rewrite historical receipts. Future native runs require fresh admission.

| Historical checkout removal | Source commit | Recovery source tag |
| --- | --- | --- |
| ChatBubble-rineya-bf16-attention-gproj-v1 | `e4e30cbff2b8` | `archive/2026-10-09/worktrees/bf16-attention-gproj-v1` |
| ChatBubble-rineya-reasoning-corpus-v1 | `1f91ddcf124b` | `archive/2026-10-09/worktrees/reasoning-corpus-v1` |
| ChatBubble-rineya-compute-depth-elasticity-v1 | `7a7f7b60e38c` | `archive/2026-10-09/worktrees/compute-depth-elasticity-v1` |
| ChatBubble-efe-func001 | `71cfb145be65` | `archive/2026-10-09/worktrees/efe-func001-validation` |
| ChatBubble-titan-tangent-task-arithmetic-v1 | `bd512fadeedf` | `archive/2026-10-09/worktrees/titan-tangent-task-arithmetic-v1` |
| reverent-tharp-a9382e | `016da519cc68` | `archive/2026-10-09/worktrees/claude-reverent-tharp-duplicate` |
| SupaExperiment-hybrid-memory-v1 | `b1813444f76f` | `archive/2026-10-09/worktrees/supa-hybrid-stage1` |
| ChatBubble-rineya-ppt-centralized-reasoning-v1 | `ff7bdac1fc8e` | `archive/2026-10-09/worktrees/rineya-ppt-centralized-reasoning-v1` |
| ChatBubble-rineya-quality-sft-v1 | `b96683586956` | `archive/2026-10-09/worktrees/rineya-quality-sft-v1` |
| ChatBubble-rineya-reasoning-posttraining-v1 | `11da2c239277` | `archive/2026-10-09/worktrees/rineya-reasoning-posttraining-v1` |
| ChatBubble-ced-looped-flow-paired-host-v1 | `627253edde9a` | `archive/2026-10-09/worktrees/ced-looped-flow-paired-host-v1` |
| ChatBubble-rvs001 | `ac9b79ac07ad` | `archive/2026-10-09/worktrees/rvs001-frozen-pilot` |
| ChatBubble-comp-rs-amendment-1 | `48efd7d147ef` | `archive/2026-10-09/worktrees/comp-rs-001-amendment-1` |
| ChatBubble-small-vocab-depth | `52430ac6c5b7` | `archive/2026-10-09/worktrees/rineya-small-vocab-depth-v1` |

## Parser source history

`research/rineya_quality_patterns_v1/recompute_behavior.py` was originally
untracked in the retained pruning checkout. An original-source baseline commit
followed by the path relocation commit makes the actual change reviewable as
one line. The old helper SHA256 is
`5e07d30e47e5af8387ec17bdc87d57db12eb0e17201a145476ddb7c52f633228`;
the new helper SHA256 is
`807b1f87a322ccd8c2521f5cc94721e24a91ac75c3772678b6ef7aa545dc3c28`.
The authoritative exact new hash is in the handoff receipt. The target parser
remains `c52b6a48cb811e3c272793a66765e82152fe170e8307eea153c3b1b6c5c58322`.
The original helper and historical result receipts are preserved in the small
vocabulary checkout's private release. Historical reports keep their original
source hash/path. No scoring, selection, threshold, outcome, or model run changed.

## Scope and limits

Active owner/custody-related branches with local-only committed history remain
listed in the timestamped report. The 19 tags do not imply that every subsequent
active-owner change has been backed up. Old source/state observations are not
new native qualification or capability evidence.

Compression was serial at BelowNormal priority with 1 MiB streaming buffers,
without model imports or GPU work. Complete runtime environments and current
runtime donors were retained. Automatic approval review rejected the optional
17-file standalone bytecode-cache cleanup with `blocked by policy`; those cache
candidates were left untouched.

Original scientific receipts/manifests are byte-preserved. Derived reports
remove local app/thread/process context and replace owner identities with
anonymous state counts. Raw chat/goal snapshots, raw process lists, broad search
logs, duplicate downloaded assets, corrupt-ref quarantine bytes, and destructive
mutation/replay scripts are excluded. Historical audit/archive sources are
provided only for inspection under `historical-tools/`; their original absolute
paths and old assumptions make them unsuitable for automatic replay.

The independent consistency review's 34-worktree observation and correction
requests predate the later 35-worktree publication snapshot and repaired export.
Those earlier observations are retained as timestamped review evidence; the
current-state note and this publication manifest describe the resolved package.

`PUBLICATION_MANIFEST.json` identifies each published file and any transformation.
Run `python verify_publication.py` to verify file lengths/hashes without network,
model, or native-worker execution.
