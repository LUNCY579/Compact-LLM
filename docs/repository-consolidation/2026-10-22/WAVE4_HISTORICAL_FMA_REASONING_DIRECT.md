# Direct consolidation — two superseded research lineages (2026-10-22)

This record preserves two live source branch tips as additional Git merge parents on `research/post-training-mimo-centralized-v1`, **without creating PR/staging branches** and without replacing newer central working-tree files.

| Source branch | Pinned source head | Current-tree treatment |
| --- | --- | --- |
| `research/factorized-memory-attention-v1` | `1cce3aaa7d5975c97a21d0534e9652bf993a61f2` | Historical ancestry retained; divergent package revisions and two source-only test files remain available from the source commit, not promoted into central's live test suite |
| `research/rineya-reasoning-posttraining-v1` | `11da2c23927764b47148fd7c9300b1a60ac8ae57` | Historical ancestry retained; three ancillary source scripts and divergent controller revisions remain available from the source commit, not silently substituted into central's newer controller |

**History-only merges, not active implementation promotions.** The exact original file blobs and trees remain accessible through the parent commits after branch retirement. Use `git show <source-sha>:<original-path>` to inspect a historical file.

All pre-existing central working-tree paths remain unchanged; this document is the only new file. No training, model evaluation, native validation or scientific qualification is asserted.

Before manually deleting the two source branch names, check commit reachability, external worktree or runner references, and checkpoint/launcher branch-name consumers. No source branches are deleted by this operation.

Do not change `main`, historical `post-training-Mimo`, open blocked PRs #33/#34, or separate CRS.
