# Private dataset backup metadata inventory

The target is the private Hugging Face **dataset** repository
`Luncy1978/rineya-crs-001-research`. The observed main commit is
`919fd0d96ca9f272591b9e36d84a3a564bee46f2`, last modified
`2026-10-05 21:54:50+00:00`. All remote reads were pinned to that commit.

The root has `.gitattributes`, `README.md`, and `snapshots/`. Thirteen
content-addressed snapshot directories each contain `archive.tar` and
`manifest.json`. Each manifest's bytes were verified against the pinned Git
blob SHA-1. Each manifest's recorded archive size and SHA-256 agree with the
Hub LFS metadata. This verifies metadata consistency; archive payloads were
never downloaded, opened, or extracted, so archive-byte restoration was not
tested.

All thirteen manifests declare source roots from these two folders:

- `C:/Users/clclc/BusinessProject/ChatBubble-rineya-centralized-reasoning-system-v1`
- `C:/Users/clclc/.codex/worktrees/crs-native-desktop-v2/ChatBubble-rineya-centralized-reasoning-system-v1`

No observed manifest declares `C:/Users/clclc/BusinessProject/Rineya-Dataset`
as a source root. The README describes CRS research preservation and records
an exclusion policy for weights, credentials, protected banks, live databases,
caches, opaque compressed bundles, and backup-local state. Source Git HEAD
is explicitly provenance only. Manifest inventory eligibility labels retain
their source meanings; no row or model claim is established by this review.

The authorized local metadata source was
`C:/Users/clclc/BusinessProject/Rineya-Dataset/backup_20261010`. Its observed
immediate names were `INVENTORY_METADATA001.json`,
`INVENTORY_METADATA002.json`, `INVENTORY_METADATA003.json`, and
`prepare_snapshot001.py`; no `prepared` directory was present at observation.
All three inventories name `Rineya-Dataset` as source, the specified private
dataset repo as destination, and `rineya-dataset-snapshots` as namespace. All
three record `METADATA_ONLY_NOT_PREPARED`.

Inventory003 metadata SHA-256 is
`a034f6a694868503afc402f0d2bd925497a49c8e99c0a638395651da0e816640`.
Its declared selection contains 11,386 files and 2,348,707,701 bytes:

| Recorded backup role | Files | Recorded bytes |
| --- | ---: | ---: |
| train_candidate_or_train_audit | 46 | 1,149,916,451 |
| declared_training_export_pending_source_status | 56 | 162,033,754 |
| source_document_or_metadata | 459 | 205,819,606 |
| evidence_or_fixture_not_train | 56 | 152,863,124 |
| excluded_quality_evidence | 10,769 | 678,074,766 |

The 102 files with TRAIN-related role labels total 1,311,950,205 recorded
bytes. Those labels do not establish canonical TRAIN selection or row
admission. The inventories contain no per-file SHA-256 and no owner fields.
Their relative paths were never followed. Their counts therefore cannot prove
current source bytes, an existing exact-content backup, source stability, or
the identity and state of a concurrent source owner.

The 4,076-byte preparation script was read statically and was never executed.
Its SHA-256 is
`8fd25664cfd45af16c64dbbb438e7fc7627e88853e6c7d772cd89577c575d389`.
It plans `backup_20261010/prepared/archive.tar.gz` and a manifest from
Inventory003. Its loop would read every selected inventory file, including
the role `excluded_quality_evidence`; it does not establish a TRAIN-only
selection. That script is not evidence that preparation or upload occurred.

A fresh `rineya-dataset-snapshots/<verified-content-sha256>/` prefix does not
collide with the observed root and matches the namespace recorded in the
local plan. A concrete content identity remains to be established from a
reviewed, authorized, stable selection; no content SHA should be invented or
inferred from the date folder. Recheck the repo tip before an eventual upload.

Detailed evidence is in `remote_metadata.json`,
`manifest_identity_projection.json`, `remote_README_at_pin.md`,
`local_metadata_projection.json`, and
`inventory003_train_role_projection.json`. Source payload files, protected
payloads, archives, and weights were not opened. No source paths were changed,
no preparation script was run, and no upload occurred.
