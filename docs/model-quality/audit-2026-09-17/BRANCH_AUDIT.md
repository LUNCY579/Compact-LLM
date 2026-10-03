# Branch reconciliation and execution isolation audit

Audited 2026-09-17 UTC. Read-only Git and filesystem inspection; no merge, checkout,
worktree creation, model load, or training was performed. The accompanying
`branch_snapshot.json` records full ref identities, ancestry counts, worktree state,
and measured disk/tree sizes. Audit-document creation is the only mutation here.

## Current branch already contains the model-quality work

Current branch: `codex/dataset-retention-audit-v1` at
`fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`.

| Compared ref | Current-only commits | Other-only commits | Finding |
|---|---:|---:|---|
| `origin/codex/dataset-retention-audit-v1` | 0 | 0 | Aligned |
| `codex/model-quality-audit-v2@59df72c` | 85 | 0 | Exact ancestor; nothing missing |
| `origin/research/rineya-reasoning-posttraining-v1@d417e10` | 54 | 0 | Exact ancestor; nothing missing |
| `origin/main@a6ee824` | 238 | 0 | Older ancestor |

Fresh `git ls-remote` confirms the live dataset-retention and reasoning heads. The
model-quality remote branch has been deleted; the local branch and its reachable
history remain. Its worktree and the reasoning worktree are clean.

This is stronger than ancestry alone: the scoped Git-tree diff from `59df72c` to
current HEAD is empty for `experiments/model_quality_gpu_v1`, `docs/model-quality`,
and `docs/history/model-quality-audit-2026-09-15`. The qualified MQ implementation,
supervisor, evidence archive, and historical document snapshots are unchanged in
the committed tree. Commit `59df72c` already contributed supervised workers,
evidence preservation, and removal of automatic checkpoint deletion. These fixes
must not be reimplemented or cherry-picked a second time.

The earlier reconciliation is documented in
`research/reasoning_posttraining_v1/rrp004/REVIEW_AND_PLAN.md`; the reasoning
`EXPERIMENT_REGISTRY.json` and `SOURCE_AUDIT.md` bind its parent to `59df72c`.

## Preserve existing worktree changes

At the start of this audit the current worktree contained three changes:

- Deleted `experiments/architecture_benchmark_001/evidence/retention-design-v1/PROCESS_sv001_supa.json`.
- Modified `experiments/rab001_exploration/process_probe.py`.
- Untracked `ERRATUM_FLOOR_RULE.md`.

The latter two describe instrument-specific metadata and floor-rule corrections.
The missing committed model receipt is a separate preservation concern. None was
modified, staged, restored, or discarded by this audit. Additional changes by
concurrent task work may appear in the timestamped snapshot.

Other dirty worktrees contain untracked experiments/evidence in `Chat-Bubble`,
`Chat-Bubble-pcsm`, `ChatBubble-cross-scale`, and
`ChatBubble-cross-scale-mini-experiment`; `RineyaBeta-supa-fwa-v1` also has five
tracked modifications. They are independent work and must be preserved.

## Complementary remote packages do not justify a blanket merge

| Remote tip | Unintegrated content | Decision for the dataset/retention experiment |
|---|---|---|
| Opallo collector `171a301` | Six commits, 19 additive files: behavior screen and bounded native collector | Useful future Opallo measurement package; its README explicitly says native qualification is pending and it introduces no training or quality claim. Import the complete isolated package only for that separately qualified measurement. |
| Paired CED host `627253e` | One additive commit, nine files | Fixture-tested architecture collector, not an ordinary Supa training improvement. Keep isolated. |
| Cross-scale framework `3059e57` | 35 branch-only commits; tip adds 30 package files | Separate architecture intervention with unqualified native integration. Historical custom Supa-A1.1 is not the exact ungated Supa 1.7 source. Avoid a broad merge that changes the causal comparison. |
| Historical RAB `f53cf4a` | One update to the old root `CONTINUATION.md` | Retain its measured memory-warning lesson if relevant; do not overwrite the current continuation with obsolete experiment instructions. |

The Opallo and cross-scale READMEs explicitly distinguish CPU/fixture mechanics
from native execution and model improvement. None supplies a demonstrated trained
replacement for the current source. No MQ or reasoning merge is necessary.

## A clean independent sparse local clone is feasible

`experiments/reasoning_posttraining_v1/controller.py:109-123` accepts only the exact
branch names `research/rineya-reasoning-posttraining-v1` and
`codex/dataset-retention-audit-v1`, requires their registered base ancestors, and
rejects every tracked or untracked worktree change. A normally named new worktree
would fail that branch check; forcing the already checked-out branch into another
shared worktree creates avoidable shared-ref risk.

Use an **independent sparse local clone** of the committed current head, with its own
`codex/dataset-retention-audit-v1` branch and `core.autocrlf=false`. A local clone can
reuse Git objects through filesystem hardlinks while retaining its own refs. Do
not use a shared object-alternates clone that depends on future source pruning.
Commit only reviewed task changes in the isolated clone and bind new execution to
that exact checkout. This preserves all current uncommitted work and does not
require lowering the controller's clean-checkout or ancestry guards. Clone
creation has not been performed by this audit.

Measured committed tree: **6,351 files, 2,421,523,989 bytes (2.255 GiB)**. Git reports
about **0.59 GiB** of loose and packed objects, excluding reported temporary garbage;
there is no need to remove that garbage. Available C: space at inspection was
about **31.11 GiB**. The training guard at `controller.py:258-265` requires
`2 * 314618880 * 12 + 2 GiB`, about **9.03 GiB**, plus its unchanged **14 GiB** free
floor. A full source checkout and even a conservative copied-object allowance
therefore fit one fresh attempt, with approximately 5.2 GiB remaining above those
reservations. A full checkout is inappropriate for a two-arm comparison: after
retaining the first arm's approximately 7.03 GiB of checkpoints, the second fresh
attempt would fail its 23.03 GiB gate.

Cone-mode sparse checkout of `experiments` and `research`, plus Git's included root
files, needs only **64,454,022 bytes (61.47 MiB)**: experiments is 49,385,485 bytes;
research is 14,740,039 bytes; root files are 328,498 bytes. The inspected MQ/RRP
controller import closure stays within `experiments` and the existing environment.
Prepared banks and model source are explicit external arguments. The controller's
"full repository" check only validates repository-root identity; it does not ban
sparse checkout. Full commit history and ordinary clean status are retained.

An appropriate construction sequence is a local, single-branch, no-checkout clone
of `codex/dataset-retention-audit-v1`, `core.autocrlf=false`, then cone-mode
`sparse-checkout set experiments research` and checkout of that same local branch.
With hardlinked objects and this small materialized tree, retaining both arms'
checkpoints is plausibly feasible within measured space without deleting evidence.
Additional reviewed task files can be included as needed. Recheck free space and
actual disk charges immediately before each launch; this is not capacity for an
unbounded experiment sweep.

The timestamped snapshot leaves only about **0.90 GiB** above the second-attempt
gate before new data, logs, metadata, and unrelated disk growth. Keep this an
explicit stop condition. Legacy `process_probe.py` and `long_context_probe.py`
write evidence inside the checkout, which would make a later controller launch
refuse it; preserve that evidence using a reviewed external-output path or a new
honest commit/binding. Do not hide it from Git status. The process probe's data
root is the repository's sibling `rineya-dataset-retention-records`, so placing
the isolated clone directly under `BusinessProject` preserves that path.

Supply the existing regular source directory directly as an absolute `--source`:
`C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/Supa`.
It contains seven assets, including the 1,192,135,096-byte weight file. The source
checker in `experiments/model_quality_gpu_v1/native.py:25` accepts external regular
paths and verifies registered hashes; no model copy or symlink is needed. Keep
prepared data and create-only output journals outside the clean checkout. Fresh
source/data/runtime qualification, a new bounded grant, and current resource
guards still apply; feasibility is not launch qualification or evidence of gain.
