# Post-promotion branch and Titan Stage-A merge audit — 2026-11-08

## Current snapshot before this merge

- Default branch: `main` at `25651b41f51f33304b7ede16fdf5961800165a83`, after PR #51.
- GitHub branches: **49** in the observed live listing (not the 64-branch historical audit).
- Verified fully contained: `codex/exp-peer-stabilizer-matrix`, `research/ced-dual28-local-p1-preparation-v1`.
- Of the remaining non-main refs, **46** have divergent histories and `research/tgsa-programme-ledger` has no common ancestor. Divergence does not mean a safe or unique code feature.
- PRs #33 and #34 were **closed without merging**; their damaged inherited data and exact-tokenizer problems remain unresolved. Closing a PR is not a scientific qualification.
- Earlier source-preservation and bulk research merges #29–#51 should not be repeated indiscriminately.

## This selective Titan Stage-A history merge

Source: `codex/triskill-stage-a` at `ef4b4d208dfa1a603a5f5bde88a98a06c5ca81bf`.

The 28 preexisting tracked entries in `experiments/titan_third_skill_v1/` were verified to have **exactly matching Git object IDs** in source and `main` (zero changed shared blobs or directories). This update adds the source's previously absent Stage A result, receipt and diagnostic files, Python scripts, captured Stage A predictions, post-stop C/T/H predictions, and small launch-probe records.

To avoid duplicating >100 MB of historical tokenizer/sidecar assets in the default tree, the directory `experiments/titan_third_skill_v1/arms/` and `experiments/titan_third_skill_v1/poststop_null/arms/` are **not imported**. They remain reachable through the source commit retained as a merge parent. No original main file is overwritten or deleted. The copied evidence is historical; no training, GPU evaluation, or tests were rerun.

**Scientific result:** the C donor passed the acquisition A1 gate, but TT+C failed the preregistered T-retention threshold at each tested beta, so Stage B remained sealed. See [RESULTS_STAGE_A.md](../../experiments/titan_third_skill_v1/RESULTS_STAGE_A.md).

## Pruning and remaining work

This interface has Git reference inspection and write operations, but no branch-deletion operation. **No branches were deleted in this operation.** Check current tip SHA, durable commit reachability, protected refs, open PRs and branch-name consumers before owner-side deletion. Two refs identified as fully contained in the pre-merge snapshot are `codex/exp-peer-stabilizer-matrix` and `research/ced-dual28-local-p1-preparation-v1`; this merge also preserves the `codex/triskill-stage-a` history as a second parent. A branch name may still be used by external tools even if its commits are contained.

Keep CRS/native qualification, incompatible shared-path experiments, historical worktree evidence and blocked-quality data apart until specifically validated. There are no new branches and no modifications to test thresholds, model training code that already existed on `main`, dataset bytes, workflow definitions, or repository settings.
