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

**Scientific result:** the C donor passed the acquisition A1 gate, but TT+C failed the preregistered T-retention threshold at each tested beta, so Stage B remained sealed. See [RESULTS_STAGE_A.md](../../../experiments/titan_third_skill_v1/RESULTS_STAGE_A.md).

## Pruning and remaining work

This interface has Git reference inspection and write operations, but no branch-deletion operation. **No branches were deleted in this operation.** Check current tip SHA, durable commit reachability, protected refs, open PRs and branch-name consumers before owner-side deletion. Two refs identified as fully contained in the pre-merge snapshot are `codex/exp-peer-stabilizer-matrix` and `research/ced-dual28-local-p1-preparation-v1`; this merge also preserves the `codex/triskill-stage-a` history as a second parent. A branch name may still be used by external tools even if its commits are contained.

Keep CRS/native qualification, incompatible shared-path experiments, historical worktree evidence and blocked-quality data apart until specifically validated. There are no new branches and no modifications to test thresholds, model training code that already existed on `main`, dataset bytes, workflow definitions, or repository settings.

## Second selective merge: geometric / quality-repair archive

Source: `research/triskill-geo001-parentq` at `ae026104a0ba42ec44f9c2aa4db8920d161212f1`. Five previously missing experiment subtrees were imported at their **exact Git tree IDs**: `research_evidence_20261005`, `titan_quality_repair_diagnostics_v1`, `titan_quality_repair_v1`, `triskill_geo001_parentq_v1`, `triskill_geometric_pilot_v1`. These record the geometric-pilot CPU-only mechanics, unfrozen PARENTQ revision, offline Titan-Transition quality-repair source, diagnostics, and reviewed ~4.5 MB evidence archive. Their source reports explicitly deny native quality qualification or autonomous launch approval. Scoped `.gitattributes` was extended without removing the existing main rule, to preserve byte-bound historical receipts. The existing `experiments/titan_third_skill_v1` differs between histories and is **deliberately kept as-is on main** (the reviewed Stage A merge takes precedence). No experiment code already on main is replaced. Source history is retained as a second merge parent. Branch deletion remains unsupported by this connected GitHub interface.

## Post-merge verification

Two direct two-parent commits now exist on `main`: `28e8b397967d5bc6a80d5e6604ab6d5c5bacd304` and `0ba5af9da42af969bbb8ae09cda209ac60f9a40c`. GitHub compare confirms `codex/triskill-stage-a` and `research/triskill-geo001-parentq` are now entirely behind `main`, as are the previously contained `codex/exp-peer-stabilizer-matrix` and `research/ced-dual28-local-p1-preparation-v1`. **At least four** live branches are strict ancestors of the new main; their branch names were not deleted. Read-only tree checks established only intended root changes. No runtime tests or native model evaluation were executed in this consolidation.

## Third selective merge: DMRF plus tangent-space study

Source: `research/titan-tangent-task-arithmetic-v1` at `bd512fadeedf50de15bd65367d49a465387767d0`. The experiment directories `delta_multi_reasoning_v1` and `titan_tangent_arithmetic_v1` were **absent from main** and have been added at their exact source Git tree IDs (`a37de2dd0ff919745648611b5a776a101e202ace`, `263a1e4db632eab3d1a91f7d5fe96515bb2af884`). No preexisting experiment files, evaluation policies or workflow definitions are overwritten. The source branch is a second parent of the resulting main commit. The DMRF package has bounded native/open and sealed-test records, including confirmed abstention damage; the tangent-space package has CPU mechanics evidence only and no pretrained execution. Neither receives an automatic release or fresh experimental grant through this preservation merge. Other DMRF branch variants remain divergent and are **not** silently replaced or declared superseded by this import.

## Fourth selective merge: MiMo exact-prefix/Windows containment

Source `codex/mimo-supervisor-death-review-fix-20261002` at `8424bfc46398113757157f763fa6e8a9821d11fc`. 139 previously absent file blobs imported under the existing MiMo package (root documentation plus `scripts/`, `tests/`, `configs/`, `receipts/`, and `results/`). All incoming blob SHAs are the exact recorded source objects. 4 differing common-source files were deliberately left at their existing newer `main` content, including capture code, receipt accounting and experiment registries. No source-specific `artifacts/` directory or hidden/external model checkpoint was imported. Source history is preserved as merge parent. **Nothing here grants a native run**; no tests or GPU inference were executed in this integration. All physical execution and evidence-status failures remain recorded. Branch name deletion is unavailable through the connected tools.

## Fifth merge: source-level artifact-stream and fixed-input safety

Exact source heads: `codex/mimo-artifact-stream-budget-20260930@9d882d38d307c5a6a2cecb8075e9ae314ae879b3` and `codex/mimo-capture-fixed-input-audit-20260930@bbe2343a5817e9512a408270f9215df0116625d0`. Four file blobs in the MiMo package are taken from the reviewed branch tips: `scripts/native_backend_contract.py`, `tests/test_native_backend_contract.py`, `scripts/capture_evidence.py`, `tests/test_capture_evidence.py`. Independent file comparisons found only the intended bounded-read and pin-test additions; no newer `main`-only lines were displaced in these four files. The newer `scripts/qualify_capture.py` was **not** replaced (preserving the call-accounting repair). Original source histories are extra merge parents; historical evidence and all unrelated existing files remain in `main`. No native/GPU execution or test run is claimed for this GitHub-level change. Follow-up CI on the updated main is still required for runtime qualification.
