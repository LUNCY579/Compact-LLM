# Encoder registration preparation — CPU only, pending review

The integrated long_native_contract requires an externally selected regular file
under PACKAGE/artifacts/long_native_audit_registrations, exact file SHA, body kind
SYNTH_LONG_ENCODER_RUNTIME_REGISTRATION_V1, canonical body SHA and runtime_pins
whose path set equals encoder_runtime_paths(). This preparation retains that
schema. It creates no file in that final registration directory and no config.

encoder_inventory.py enumerates installed distribution RECORD metadata and the
physical package trees. All .py and .pyd files form one exact executable superset;
missing RECORD entries, undeclared physical additions, changed paths, versions,
hashes, stale fields and symlink/reparse indirection reject. No count-only shortcut
or fallback is used. The observed roster is45 files: tokenizers0.22.2 has17,
Jinja2 3.1.6 has25, MarkupSafe3.0.3 has3. Two pyd binaries are byte-hashed only;
none is loaded. Native model weights, Torch and GPU APIs are never opened.

The staged candidate binds four explicitly allowed prompt assets (tokenizer.json,
tokenizer_config.json, vocab.json, merges.txt), exact UTF8 chat-template hash,
the PILOT_DECISIONS/TOKENIZER_RECEIPT binding sources, current prompt/native
plumbing helper source bytes and the reviewed TRAIN packet208f3533. Packet source
origins are re-read and compared to the original CPU preparation pins. No encoder
construction, template rendering, token generation, decoding or grading occurs.

ENCODER_CANDIDATE_*.json uses a CPU-only outer kind and status. Its enclosed
native_registration_preview is review material, never issued native authority.
verify_candidate requires an externally selected body hash then reconstructs
every current source/runtime/asset/template/input value. Self-signing a changed
candidate cannot substitute for that reconstruction or for independent review.
The root chooses and writes the eventual native registration only after review;
the native worker still checks all actual byte pins inside fresh owned admission.

Tests use isolated synthetic package trees and fake .pyd bytes, never encoder
execution or native audit PASS. Candidate files, CPU receipts and any rejected
predecessors remain write-once. Native launch, study, scoring, enrollment,
resource qualification and historical model-fit/cause claims stay closed.
