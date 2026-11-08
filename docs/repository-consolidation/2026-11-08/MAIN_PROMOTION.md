# Main integration promotion — 2026-11-08

This integration merges the current `research/post-training-mimo-centralized-v1` history into default `main` while retaining its original main lineage as an explicit second parent. **No force push, branch deletion, or history rewrite.**

## Exact parents and snapshot
- Historical main before integration: `6db5c085ad721098d62f1f5a71c91dcdf26f8c32`.
- Protected recovery branch created before promotion: `archive/main-preconsolidation-20261108` at that exact SHA.
- Research integration source after bulk PR #50: `e1629a54198283be5a3ce5dd505006df3ba144f2`.
- Git compare of historical main against the source showed **one** main-only README modification. Its bytes are preserved in [the archived main README](../main-readme-preconsolidation-20261108.md).
- The current README is an updated copy of the integrated research README, explaining the default-branch change.

## Status and limitations

This is **not** a claim that all 66+ live branch heads have been fully reconciled. In particular, [PR #33](https://github.com/LUNCY579/Compact-LLM/pull/33) remains blocked by damaged inherited data; [PR #34](https://github.com/LUNCY579/Compact-LLM/pull/34) remains blocked by its exact tokenizer qualification fixture. DMRF and MiMo supervisor divergences, distinct CRS scope records, and historical worktree snapshots still need selective review. Source branches remain unchanged. The imported research packages from [PR #50](https://github.com/LUNCY579/Compact-LLM/pull/50) were subtree unions, not newly executed model or unit tests.

The promotion is reversible with the exact recovery branch and Git history. It does **not** weaken checks or imply commercial/public release readiness.
