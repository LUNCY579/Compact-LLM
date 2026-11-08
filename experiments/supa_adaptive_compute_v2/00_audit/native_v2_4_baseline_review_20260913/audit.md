# Independent v2.4 A0 baseline identity review

Both v2.3 and v2.4 contain 128 unique architecture cases across the same 30 populated deterministic shards. Shards 15 and 23 are empty in the frozen bank. All case IDs, metadata and ordering match the bank and manifests.

Exact raw identity: 128/128 token sequences, 128/128 reference NLL values, and 128/128 saved logit archive SHA256 values. Maximum absolute NLL difference: 0.0. No tolerances were applied. Any mismatching IDs are listed in receipt.json. No accuracy, efficacy or candidate selection was calculated.

Both sets of manifests bind their respective frozen registration, configuration, sources, dataset and passed GPU qualification. Canonical checkpoint and state hashes, zero trainable backbone parameters/gradients, before/after preservation, and deterministic FP32 eager runtime receipts reconcile within and across versions. All 147 current v2.4 source files were rehashed. The preserved v2.3 source map was reconciled against its own manifests.

Validation: 5383 checks; 1071 distinct files hashed. The standard-library audit reuses the unchanged previous review helper and records both audit-script hashes. Run `python audit_baseline.py` for a read-only repeat; `--write` is accepted only when the two exclusive output files do not exist.

Limits: this is a saved-evidence audit, not a new model execution or checkpoint reread. Saved logit tensors are not loaded. Qualification metrics are not recomputed. Scores and answer/response fields are discarded during JSON projection and only traversed as opaque bytes for integrity reconciliation. Confirmation and Track B data remain unopened. This review does not open the five-arm architecture or capability gate.
