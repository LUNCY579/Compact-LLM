# RNGF-001 evidence index

## Publication snapshot: September 19, 2026 (America/Chicago)

Read [PUBLICATION_RECONCILIATION_20260919.json](PUBLICATION_RECONCILIATION_20260919.json)
for the reconciled publication and repository-source record. Connected GitHub reads
observed `research/rola-ngu-factorial-v1` at
`ebfb6aeec1e6eb55bf520afb604ab46d652d9297`, two commits ahead and zero behind
`fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`:

```text
fbec90a0d0653aeb745d5223137ae3cfdd9c48dd  original code base
  -> 3a83618a96a82f87255e982e888d0ecb78dd94b0  mechanics publication
  -> ebfb6aeec1e6eb55bf520afb604ab46d652d9297  integration publication
```

This is a dated observation, not an assertion that the branch can never move.
Two observed implementation commits are not a count of successful API calls.

## Evidence roles and precedence

| Record | Meaning | What it does not establish |
|---|---|---|
| `PUBLICATION_RECONCILIATION_20260919.json` | Observed ref, commit ancestry, package tree, and recorded source-digest binding | Native execution, training authority, or the publication of its own containing commit |
| `integration_20260918/VALIDATION.json` and companion files | Recorded 152-test CPU-fixture pass, native blocked | A fresh rerun, exact-source qualification, or model improvement |
| `REMOTE_STATUS.json` | Historical September 18 failed publication attempt with zero successful writes in that attempt | Current branch head or lifetime publication count |
| Original root-level mechanics/test/preflight receipts | Historical evidence for the original 97-test package | The later integration or native qualification |

The newer publication snapshot supersedes only the **current-publication reading**
of `REMOTE_STATUS.json`. The failed attempt remains historical evidence and is
not rewritten. No test result, experiment outcome, registration, or source pin is
relabelled. The old 97 tests and later 152 tests are not added together.

## Source binding without history rewriting

The observed implementation package tree is
`7617dcc8ab7d41036e1f4fe5106faa3524d121ac` (33 blobs). `manifest.json` and the
integration validation both record source digest
`e88e246b37c1b0febb4266919089eea7bb48c07da7e0aaa25b47fa9d8582afa2`.
The reconciliation checks Git object identity and recorded-digest agreement; it
does not rehash all source-file SHA256 values or rerun the 152 tests.

Only this index and the publication JSON may be added. Preserve all 33 existing
package blobs, including the package README, manifest, registration, source lock,
and every prior evidence file. The package README is part of the source manifest,
so editing it would unnecessarily change the historical source binding.

The evidence-only successor has a new commit and tree; do not label the old full
package tree as its own tree. Verify its publication independently using the
remote successor ref, commit parent, and an exact two-added-file comparison.
Keep that final delivery receipt outside its own commit to avoid circular
self-binding. The original experiment branch is not moved by this change.

## Authority remains closed

Publication verification does not verify the seven assets declared by
`source_lock.json` and does not authorize an exact-source run. Exact-source native
qualification, pretrained loading/training, sealed evaluation, and quality,
efficiency, or release claims remain blocked. Production asynchronous throughput
is not established by the CPU simulator.

This reconciliation executes no experiment runner, model loader, optimizer step,
sealed read, or workflow dispatch. The prior fixture's 128 toy optimizer steps
remain in its historical record; they are neither erased nor called pretrained
training. An unavailable remote read means **unknown**, not **unpublished**.
