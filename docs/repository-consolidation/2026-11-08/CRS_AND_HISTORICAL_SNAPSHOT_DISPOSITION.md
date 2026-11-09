# CRS and historical snapshot disposition — 2026-11-08

## Scope and exact original source tips

| Branch | Exact source commit | Decision |
| --- | --- | --- |
| `codex/archive-20261004/historical-main-snapshots-24` | `6b539d9abf3a71e0c60c7b8342a3c79871205f0a` | Preserve full commit history as a second-or-later parent of `main`; retirement of this **snapshot/exception ref name** only after SHA, ancestor, open-PR and protection checks |
| `codex/archive-20261004/historical-main-snapshots-27` | `4f273e4efc443d422ed5c151bb00938d7fa65ccb` | Preserve full commit history as a second-or-later parent of `main`; retirement of this **snapshot/exception ref name** only after SHA, ancestor, open-PR and protection checks |
| `codex/crs-owner-exception-001` | `e47796f1e9b198f9c3213f471f0151ce517a183b` | Preserve full commit history as a second-or-later parent of `main`; retirement of this **snapshot/exception ref name** only after SHA, ancestor, open-PR and protection checks |
| `codex/worktree-save-20261002/detached-crs-native-desktop-v2` | `f9e5e11997bc7e23bdb0378574fe447dc3c3ee8b` | Preserve full commit history as a second-or-later parent of `main`; retirement of this **snapshot/exception ref name** only after SHA, ancestor, open-PR and protection checks |

The two `historical-main-snapshots-24/27` tips contain, respectively, 3,960 tracked blobs (~2.34 GB) and 2,271 tracked blobs (~1.18 GB), whose large model/data archives were already in the public repository's historical refs. Their full source commits are retained as ancestors without importing or overwriting any of those files in the default `main` checkout. This makes an existing immutable commit SHA the recovery handle rather than relying on a mutable branch name.

The older `codex/crs-owner-exception-001` CRS package contains 423 blobs (~6.1 MB), and `codex/worktree-save-20261002/detached-crs-native-desktop-v2` contains 68 blobs (~1.88 MB). Some of their historical reports described CPU qualification and prospective native handoffs. **These reports are not current CRS authorization.** Their exact commits, all older code, original receipts and evidence stay reachable through merge-parent ancestry; none has replaced the durable CRS runtime or current `STATUS.md`.

## Research track separation and authority

Three dedicated branches remain distinct and are **not designated obsolete**:

- `research/rineya-centralized-reasoning-system-v1`, canonical durable CRS branch. The current status reports `NOT_READY`, cumulative source6 rejected before its fixture and scope014 exhausted; no native execution, extra retry, fresh grant or full-scoreboard conclusion is implied.
- `research/rineya-ppt-centralized-reasoning-v1`, complementary PPT sampler branch. CPU sampler receipts are not native model-quality results. Its charter requires explicit evidence-based disposition for retirement or replacement.
- `research/rineya-synth-centralized-reasoning-v1`, large synthetic CRS research branch (15,104 blobs, ~1.06 GB). Its content and any protected data roles remain isolated from default-branch experiments.

This is a **history-preserving archive of superseded temporary branch names**, not a merger of the scientific CRS/PPT/Synth execution authorities or a transfer of launch permissions. The unresolved questions are native numerical and source/containment qualification, independently powered search/system outcomes, source custody, and model-retention/capability comparisons. No successor has established those results merely through archival. The restart condition for any still-open line is a newly scoped, source-reviewed and authorized experimental proposal, not this Git commit.

## Reversibility

An exact source tip can be inspected at `https://github.com/LUNCY579/Compact-LLM/tree/<40-character-SHA>` even after branch deletion. Recreate a branch only after explicit future direction and by supplying the pinned SHA to a trusted GitHub owner tool. This merge itself creates no branch and does not force-push, rewrite or delete source evidence. The source commits are second-through-fifth parents of the exact main merge commit that references this file; the subsequent pruning workflow checks current SHA and ancestry again. Existing historical tags and private releases are unchanged.

## No scientific qualification implied

No native model runs, data ingestion, training, sealed-bank reads, CI tests, credential modifications, workflow permission changes or model promotions were executed to construct this history-preserving merge. The default `main` experiment, workflow and data trees are not replaced. No code was reinterpreted as current just because an old source SHA is now an ancestor.
