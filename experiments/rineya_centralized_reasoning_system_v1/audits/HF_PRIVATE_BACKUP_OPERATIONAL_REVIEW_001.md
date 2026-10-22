# Private research backup: independent OPS review001

Date: 2026-10-04. Reviewer `/root/exception_ops`; builder
`/root/hub_backup_builder`. **SOURCE_APPROVAL_WITHHELD_FIRST_UPLOAD_BLOCKED.**
Three consequential findings remain in these exact frozen bytes. The intended
private research archive is authorized, but this helper must be repaired and
reviewed before its first upload. This report approves no probe/native source,
model, training, protected-bank access or quarantine transition.

I authored none of this helper, tests or contract. Shared model/workspace ancestry
is disclosed. I changed only this report. I inspected source, contract, retained
builder stdout/inventory and the actual owner grant, then ran three new narrowly
scoped synthetic controls in a temporary directory with every external subprocess
poisoned. No real HF/network call, framework/model/tokenizer/GPU operation or
canonical host database access occurred. The22-test suite was not repeated.

## Exact reviewed evidence

| File | SHA256 |
| --- | --- |
| hub_backup.py | `966b13c07cb18527508e4380fa7ce970c131df8a331621586f12eb778c184f22` |
| tests/test_hub_backup.py | `a09df01169b7c86a6b8d863a216478819d46b8c698138b2a48228b0636e02dc1` |
| contracts/HF_PRIVATE_BACKUP_V1.json | `e3714214c07cad2633ef2b94b64ef4cc1ab6f8e528a874e4dc1a0094be47a418` |
| TEMP/crs-hf-backup-builder-20261004-tests.txt | `a9ac69cff6a813fa71d1bb8ef3af025916da21ea3323d480c902678810fd91ea` |
| TEMP/crs-hf-backup-builder-20261004-inventory.json | `5be36889771c5192663b0702c287399468389b295ddc03f2ca1afb4df2f2c5c3` |

All pins independently match. Builder stdout retains22/22 tests, zero skips or
errors, in1.273s, with all HF operations faked. Initial inventory retains998
primary files/23,875,405 bytes, largest5,198,153 bytes, eight exclusions and digest
`1a8d3825c506d9c2b45c470a1c1a1bb254ae3e8cdd40f21a7d75c2a78db693c2`.
This is an initial inventory, not current packaging/exception freeze approval.
Root reports the fixed repository is already private at initial commit
`8f6672c700a574fe744b1d1de626f6463bcb8349`; I did not independently query it.
The actual owner receipt fixes dataset `Luncy1978/rineya-crs-001-research`, private
visibility, meaningful-update preservation and no remote deletion/public changes.

## HF-OPS-001 — P1: prepared manifest metadata bypasses credential refusal

At389–450, `verify_prepared` validates manifest destination, inventory and tar
integrity, but the manifest bytes themselves never pass `validate_bytes` or an
equivalent credential scan. Unchecked fields are allowed, and content identity
includes only archive paths, byte counts and artifact hashes. A local manifest
can therefore change while the prepared content digest remains valid.

Independent synthetic control prepared one harmless JSON artifact, then added
an unchecked manifest metadata field containing a synthetic credential-shaped
string. The helper's own SECRET_BYTES detector matched the modified manifest,
yet `verify_prepared` accepted it and returned that metadata. Both pre-upload
verification calls would accept those bytes, and the sole folder upload includes
manifest.json. The archive rows and tar remained unchanged. No real credential
or network upload was used.

Required closure: revalidate every uploaded file, including the complete manifest
bytes, against credential/binary rules; validate/bind manifest provenance and
metadata rather than permitting unchecked changes. A new control must reject
credential-shaped metadata without echoing it, before a write command. Preserve
the intentional exclusion of HEAD/receipts from meaningful content identity.

## HF-OPS-002 — P1: remote absence check is not atomic with the upload

At573–579, the helper checks current private identity and absence of the snapshot
prefix, then invokes `client.upload`. HFCLI.upload at491–492 sends no expected
remote commit or equivalent compare-and-swap authority. A snapshot appearing
after the final info response can be replaced at the same addressed path.

Independent fake-network control introduced concurrent archive bytes inside the
fake upload boundary after both info checks, then exercised the current folder
write behavior. It overwrote that archive and returned VERIFIED after comparing
the newly uploaded bytes. The existing test covers appearance at the final info
check, which is earlier than this race. This finding concerns the source's absent
atomic write condition; no live concurrent Hub operation was attempted.

Required closure: bind the write to the exact privately verified remote commit,
so a changed branch rejects without overwriting an existing addressed snapshot.
Treat a conflict/uncertain response conservatively and reconcile explicitly;
retain the concurrent bytes and require a separate decision rather than a blind
retry. Verify same-content history remains a no-op and no deletion is introduced.

## HF-OPS-003 — P2: malformed remote inventory is interpreted as absence

At495–499, `sibling_paths` requires a list but silently drops non-dictionaries,
missing names and non-string filenames. The independent control
`siblings=[{}, {rfilename: null}]` returned an empty set. Both reconciliation and
the final absence check can therefore treat an unusable inventory as proof that
the target is absent and proceed to a write.

Required closure: reject malformed/ambiguous inventory entries rather than
discarding them. Before any upload or RECONCILED_ABSENT receipt, require a complete
valid remote path inventory. Add a narrow malformed-inventory no-write control.

## Inspected strengths and bounded disposition

Source reads refuse path escapes, symlinks/reparse points, hardlinks, nonregular
files, unexpected file types and common credential patterns. Database, tensor,
weights, protected-bank, cache, opaque-archive and backup-state classifications
occur before file content reads. JSON paths are never followed. Reads associate
device/inode/size/time and expected bytes/hash; preparation repeats inventory.
File/count/total/depth/walk bounds are explicit. The optional exception inventory
requires an explicit matching digest and remains separately prefixed. Tar members
are regular, uniquely inventoried, bounded and checked by actual bytes without
disk extraction. Local folder membership rejects extra upload files. Every row
remains training_eligible=false and retains original qualification boundaries.

The sole NUL-text exception matches the original AI-Newton extraction exactly:
SHA9e70f23c...,28,914 bytes,two NULs. Other NUL-bearing text rejects; it is no
general binary-text exception. Destination checks require fixed dataset identity,
private=True and a valid commit. Upload PENDING is persisted before the command;
ordinary failures retain UNKNOWN, and upload refuses PENDING/UNKNOWN until an
explicit reconciliation. Remote manifest/archive verification is commit-pinned.
There is no token argument, raw CLI output rendering, remote deletion, scheduler
or receipt-driven self-trigger. These strengths do not close the findings above.

The three new controls took0.12290619999930641s, all reproduced their targeted
defects, zero unexpected failures; one fake folder upload and zero real remote
operations occurred. The complete consequential finding set is HF-OPS-001/002/003.
Withhold this helper's source acceptance and first-upload approval. Retain the
frozen builder evidence, repair within the separately authorized backup scope,
then obtain fresh independent review and Root's concrete first-upload review.
Any later packaging must use a fresh inventory and explicit frozen exception
digest; this report does not freeze the mutable exception worktree. Backup is
research preservation, never training eligibility, scientific success or model gain.

```json
{"review_id":"CRS-HF-PRIVATE-BACKUP-OPS-001","reviewer":"/root/exception_ops","builder":"/root/hub_backup_builder","status":"SOURCE_APPROVAL_WITHHELD_FIRST_UPLOAD_BLOCKED","helper_sha256":"966b13c07cb18527508e4380fa7ce970c131df8a331621586f12eb778c184f22","tests_sha256":"a09df01169b7c86a6b8d863a216478819d46b8c698138b2a48228b0636e02dc1","contract_sha256":"e3714214c07cad2633ef2b94b64ef4cc1ab6f8e528a874e4dc1a0094be47a418","findings":["HF-OPS-001","HF-OPS-002","HF-OPS-003"],"retained_builder_tests":22,"full_suite_reruns":0,"independent_targeted_controls":3,"targeted_seconds":0.12290619999930641,"unexpected_control_failures":0,"fake_uploads":1,"real_network_operations":0,"external_subprocesses":0,"canonical_host_DB_accessed":false,"source_edits":0,"upload_approved":false,"probe_or_native_source_approval":false,"training_eligible":false}
```
