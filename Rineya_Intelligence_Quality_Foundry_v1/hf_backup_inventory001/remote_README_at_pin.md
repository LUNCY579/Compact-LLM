---
pretty_name: CRS-001 private research preservation archive
task_categories: []
tags:
  - research-artifact-archive
  - not-training-eligible
---
# CRS-001 private research preservation archive

This private dataset repository preserves the complementary Rineya CRS-001
research track. It is an artifact archive, not a training dataset or model
release. Snapshots retain supplied source, research documents, open TRAIN/DEV
fixtures, raw predictions, quality reviews, failed attempts and quarantined
history with their original paths, byte counts and SHA-256 hashes.

All included records have `training_eligible: false` in the archive inventory.
Qualified CPU fixtures, native observations, failures and quarantined records
retain their original qualification boundaries. Backup does not reopen a gate,
approve a hypothesis, establish model gains, or authorize training.

`snapshots/<content_sha256>/archive.tar` contains separately prefixed `primary`
and, only when explicitly frozen and supplied, `exception` sources. The adjacent
manifest records source Git HEAD as provenance, inventory, coverage, exclusions,
limits and archive integrity. External paths mentioned inside JSON are never
followed. Model weights, credentials, protected central CHECK/FINAL banks, live
databases, caches, opaque compressed source bundles and backup-local state are
excluded. Checked individual upstream source, license and paper evidence may
remain as research provenance; their original licensing still applies.

Snapshots are content addressed. Existing snapshot bytes must verify before a
same-content no-op; changed content receives a new directory and Hub commit.
No remote deletions, public visibility changes or scheduled uploader are used.
