# Wave 3 — CED D28-001 local P1 preparation (historical integration)

This document registers a reviewed, source-preserving integration from `research/ced-dual28-local-p1-preparation-v1` into `research/post-training-mimo-centralized-v1`.

- Exact source head: `6a56cb64ec03fb729b006192fd58a7d55975e707`
- Original target execution commit (as stated by the source): `cd6ef9bd49724da6879bfa4d945af3bfaa6d9758`
- Imported paths: `tools/dual28_local_p1/README.md`, `local_p1.py`, `test_local_p1.py`, `verification.json`.
- Preservation: all four files imported at **original Git blob SHA and mode**, with no replacement of existing target files. Source head is an additional parent of the integration commit, preserving ancestry.
- Qualification: **PREPARATION ONLY**. The source's verification record reports 11 helper unit tests passed but says the official P1 was not executed; no official weights loaded, no performance or deployment claim.
- This integration is documentation/code preservation, **not** approval to run expensive training or to treat this experimental launcher as a supported default.
- Do not retire the original branch name until this PR has merged, its commit/ancestry is verified, and branch-name consumers (runbooks, launchers, external hosts and local worktrees) are handed over to the central branch or a pinned source SHA. Retain any pinned evidence tags.
- Preserve `main`, `post-training-Mimo`, and the separate durable CRS branch unchanged.

Review scope: four additive source files plus this provenance document; no deletions or edits of preexisting files.
