# Independent v2.3 A0 baseline review

Both native GPU versions have all 128 architecture cases across the same 30 populated deterministic shards; shard indices 15 and 23 are empty in the frozen bank. All expected IDs are unique, metadata and shard order match the frozen bank, and every results, case and saved logit artifact was hashed and reconciled.

Raw comparison: 128/128 token ID sequences, 128/128 saved NLL values, and 128/128 logit archives are identical. Maximum absolute NLL difference: 0.0. Difference IDs, if present, are listed in receipt.json. No accuracy or candidate-selection analysis was performed.

Each version binds its own configuration, registration, sources and successful GPU qualification. Canonical class, checkpoint/state identities, zero trainable parameters/gradients, before/after preservation receipts and FP32 eager deterministic runtime match throughout and across versions. The current v2.3 source bytes were rehashed. Original failed receipts remain excluded and preserved.

Validation: 5373 checks; 1067 distinct files hashed. Run `python audit_baseline.py` for a read-only repeat, or `python audit_baseline.py --write` only when the exclusive outputs do not exist. The script uses only the standard library.

Limits: this audit checks saved evidence, not fresh model execution or live checkpoint bytes. Case files have no independent original manifest SHA, so full opaque JSON equality binds them to the SHA-verified results. Logit archive bytes are compared without tensor deserialization. Qualification metric values and historical Git source blobs were not recomputed. Bank and result field projection discards answers/responses/scores; these are only traversed as opaque serialized bytes for integrity checks. Confirmation and Track B data remain unopened. This baseline audit does not open the five-arm architecture or capability gate.
