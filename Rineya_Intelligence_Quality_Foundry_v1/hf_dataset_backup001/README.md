# Prospective Rineya-Dataset TRAIN-declaration byte backup

Status: **SOURCE AND SYNTHETIC TESTS WRITTEN; NOT EXECUTED; NOT CAPTURED; NOT UPLOADED.**

The requested destination is the private Hugging Face dataset `Luncy1978/rineya-crs-001-research`. Existing CRS snapshots belong to a different project and do not establish Rineya-Dataset coverage. This tool contains no upload or network code.

`prospective_selection.json` retains the 102 declared TRAIN-related inventory rows (1,311,950,205 recorded bytes), plus 38 small metadata files (331,985 recorded bytes). These are prospective declarations, not verified existence, hashes, ownership, canonicality, admission, or training authorization. Original roles and `training_eligible=false` remain unchanged. Owner, canonicality, native/model boundaries, and rights-use stay UNKNOWN; gains stay UNMEASURED.

`backup.py` accepts only the pinned inventory projection and exact source root for production. The default `describe` command does not inspect source payloads. Production `capture` requires an independently prepared receipt pinning the current source, test source, selection file, inventory, and exact source root. Its required status is `APPROVED_FOR_CAPTURE`, with `independent_review=true` and `synthetic_tests_passed=true`. No such approval or passed-test claim is supplied here.

Synthetic tests are saved as `test_backup.py` and must pass in disjoint temporary fixtures after independent source review. Review evidence, source, prospective selection, manifests, and receipts may be committed to Git. Archive payloads (`*.tar.gz`) are ignored by the local `.gitignore` and belong only in the private HF dataset.

After successful capture and full readback, upload may use only the fresh prefix `rineya-dataset-snapshots/<content_sha256>/` reported by the manifest. The uploader must independently confirm the destination remains private, authenticated account/authorization remain appropriate, and the entire prefix is absent or byte-identical. It must never replace the dataset root README, existing CRS snapshots, or an existing differing object. A capture failure prohibits upload.

See `DESIGN.md` for the gates and `SOURCE.json` for metadata provenance. The preserved `backup_pre_review001.py` is historical source evidence of the first reviewed draft; it is not an approved executable target.
