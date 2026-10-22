# Wave 3 — CED paired-looped-flow host adapter (historical integration)

This scoped integration preserves `research/ced-looped-flow-paired-host-v1` at exact commit `627253edde9a0e67334034d252e5d9dd9daead9a` in `research/post-training-mimo-centralized-v1`.

- Additive source location: `experiments/paired_lf_host_v1/` (nine original blob files, copied at exact Git SHA and mode).
- Preservation: the source commit is the second parent of the integration commit, so its original lineage remains reachable without replacing files from the central tree.
- Evidence: the source's `evidence/RESULT.json` reports 90 core tests and 15 host adapter fixture tests, zero reported failures/skips; these are historical source claims, not independently rerun in this integration.
- **Native status: BLOCKED / UNEXECUTED**. Official model assets and pinned sources were not available to that experiment; no native weights loaded or backbone optimizer steps were performed, and no efficacy claim is established.
- This imports an isolated research adapter, not a supported default host or official qualification.
- Source branch retirement is owner-managed only after confirming this PR merged, verifying commit ancestry, and checking external worktree, runner, and branch-name dependencies. Retain immutable evidence and pinned parent SHAs.
- `main`, `post-training-Mimo`, and the separate CRS research publication stay untouched.
