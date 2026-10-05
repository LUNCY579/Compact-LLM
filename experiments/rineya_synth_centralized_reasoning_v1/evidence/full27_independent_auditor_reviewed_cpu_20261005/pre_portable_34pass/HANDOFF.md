# Full27 independent chunk auditor — stage only

The candidate is an independent receipt auditor, not a native launcher or study
importer. There is no new full27 native result. CPU controls exercise generated
JSON/protocol/resource/Job/lease records and synthetic ZIP storage bytes; they
cannot issue native PASS, qualify a worker, restore a model or provide study state.

Files: auditor.py and test_full27_audit.py are owned by guard_stage_review.
symbolic_checkpoint.py and test_symbolic_checkpoint.py are owned by engine_review.
The combined CPU_VALIDATION_001.json records 34 controls and all35 source hashes
before/after, including all31 current native sources and unchanged original26
qualified helpers. Rejected test bytes/results and the helper predecessor are
preserved in the subdirectories. Existing native audits and root code are intact.

The API is:

    audit(attempt_path, preserved_path, config_path, inventory_registration_path,
          *, terminal_signal, expected, output_path=None)

No reads of native output/assets and no receipt writes occur until the caller
supplies terminal_signal="ROOT_TERMINAL_AND_PRESERVED" and every explicit expected
anchor. The expected dict has exactly inventory_file_sha256,
profile_body_sha256, config_file_sha256, result_file_sha256,
supervisor_file_sha256, preservation_body_sha256, source_commit, arm and chunk.
These values must be independently selected by the root; computing a self-hash
does not authenticate authority. There is no default permit or native dispatch.

A write-once inventory registration belongs to exactly
PACKAGE/artifacts/full27_audit_registrations, with regular files and no symlinks.
Root freezes it after committing the reviewed source and concrete config, before
native dispatch; HEAD remains unchanged through that scope. It is ignored to
avoid a Git commit/self-reference cycle, then preserved with the terminal raw
evidence. Its schema is:

    {
      "schema_version": 1,
      "kind": "SYNTH_FULL27_INDEPENDENT_AUDIT_INVENTORY_V1",
      "root_path": "<absolute workspace>",
      "package_path": "<absolute SYNTH package>",
      "config_path": "<absolute owned configs JSON>",
      "config_file_sha256": "<externally selected file hash>",
      "source_commit": "<exact launch HEAD>",
      "profile_body_sha256": "<digest of profile/base_compute_profile/checkpoint_amendment/worker_ram_amendment>",
      "counts": {"source":31,"runtime":83,"data":8,"model":7},
      "inventories": {
        "source":"<entire config source_pins map>",
        "runtime":"<entire config runtime_pins map>",
        "data":"<entire config data_pins map>",
        "model":"<entire config model_pins map>"
      },
      "auditor_source_pins": {
        "<absolute auditor.py>":"<reviewed exact file hash>",
        "<absolute symbolic_checkpoint.py>":"<reviewed exact file hash>"
      },
      "previous_authority":"<exact config.previous_authority or null>",
      "sha256":"<canonical body digest excluding sha256>"
    }

No count-only acceptance is used. The current complete root qualification
contract is rebuilt, and the frozen recipe,540-row packet, global LR, source and
prerequisite pins must match. Source/runtime/model binaries are refreshed only
during an explicitly released terminal audit, not during these CPU tests.

The preserved terminal leaf must use raw/ and source_snapshots/ with
config_snapshot.json, audit_inventory_snapshot.json and preservation_receipt.json.
The receipt provides exact original raw_directory/preserved_directory/source_commit,
raw_files and source_snapshots byte maps, config_snapshot/audit_inventory_snapshot
byte descriptors and canonical sha256. An audit checks every original/preserved
file plus all current independently registered source/runtime/data/model bytes
before and after validation. The real artifact leaf and persistent lease remain
immutable during the audit.

A successful receipt has the exact declared contract.AUDIT_KIND,
PASS_FULL27_QUALIFICATION_CHUNK_MECHANICS_ONLY status, source/arm/chunk and trusted
worker/SUP file hashes. It binds the checkpoint descriptor, all native saved bytes,
detached witness, current physical dose, prior/current cumulative cost projection,
reference controls, decoder work, owned cleanup and complete byte closure. This
receipt can be explicitly pinned by the root in the next same-arm qualification
config; it does not authorize automatic resume or study initialization.

Failure/partial/foreign-arm/source-drift/rollback evidence is rejected. Recorded
native API counts are distinguished from wrapper protocol counts and internal
ATen operations. All180 training CE calls must return from the frozen checked-loss
implementation; training per-row CE values are not separately persisted by this
backend. The20 independent reference row CE checks are durably recorded and their
EOS denominators/errors are checked. Reference112 shape/order/finite/passed claims
must come from the pinned preclip comparison, at1e-6/1e-5. The auditor does not
rerun model math or independently reconstruct unsaved reference tensors.

The symbolic helper interprets whitelisted protocol2 opcodes into inert references,
not pickle.loads/Unpickler/Torch objects. It checks all450 unique whole storages:
112 FP32 native q/v factors,224 Adam moments,112 exact scalar steps and2 CPU-stored
RNG tensors. Parameter order, shapes, byte hashes, finite values/nonnegative second
moments, empty gradients, Python RNG private-probe encoding, occurrence order,
global LR and metadata match the saved witness. Torch CPU/CUDA generator encoding
is not probed; those bytes are compared exactly. Frozen310 base bytes are proven
by the pinned native content-check calls/source loader map, not inside the adapter
archive. No model weights, gradients, restore or CUDA run occurs during auditing.

The34-control CPU result establishes auditor mechanics only. Actual separately
owned scopes still need fresh resource admission, unchanged600s/RAM/RSS/GPU/disk/
freshness/numerical gates and independent complete terminal audit. Broad full27
fit, continuous27 bitwise parity, transfer, retention, scoring, DEV collection,
study training and qualification-checkpoint warmstart remain closed.

